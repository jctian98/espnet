# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Parallelization utilities for HuggingFace Qwen3 models.

This module provides grouped MoE replacement, activation checkpointing,
torch.compile, and FSDP2 wrapping for HuggingFace Qwen3 (dense and MoE)
models used in the SpeechLM framework. It follows TorchTitan's
parallelization patterns adapted for the HuggingFace model structure.

HuggingFace Qwen3 model structure:
    model.model.embed_tokens  - Token embeddings
    model.model.layers        - List of transformer layers
    model.model.norm          - Final RMSNorm
    model.lm_head             - Output projection

For MoE models (e.g., Qwen3-30B-A3B), some layers have:
    layer.mlp = Qwen3MoeSparseMoeBlock
        .gate: nn.Linear(hidden_size, num_experts)   # Router
        .experts: nn.ModuleList of Qwen3MoeMLP        # Individual experts

Additional multimodal components (added by ParallelHFModel):
    model.multimodal_io_dict  - Dict of multimodal IO handlers
    model.adaptor             - Dict of linear adaptors for continuous modalities
    model.stream_emb          - Stream embeddings
"""

import logging
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torchtitan.distributed import ParallelDims

from espnet2.speechlm.model.speechlm.parallel_utils.grouped_moe import (
    GroupedMoeBlock,
)

logger = logging.getLogger(__name__)


def _is_moe_layer(layer: nn.Module) -> bool:
    """Check if a transformer layer uses MoE (has gate + experts in mlp)."""
    return hasattr(layer.mlp, "gate") and hasattr(layer.mlp, "experts")


def parallelize_qwen3_hf(
    model: nn.Module,
    parallel_dims: ParallelDims,
    titan_config: Dict[str, Any],
) -> nn.Module:
    """Apply parallelization to HuggingFace Qwen3 model.

    Order: Grouped MoE -> AC -> torch.compile -> FSDP
    (following TorchTitan's convention)

    Args:
        model: HuggingFace Qwen3 model (possibly wrapped with multimodal components)
        parallel_dims: TorchTitan ParallelDims object with device meshes
        titan_config: Configuration dict containing:
            - activation_checkpoint: AC ratio 0.0-1.0 (default: 0.0).
              1.0 = all layers, 0.5 = every other layer.
            - compile: Whether to enable torch.compile (default: false)
            - compile_mode: Compile mode (default: "default")
            - mixed_precision_param: Parameter dtype (default: "bfloat16")
            - mixed_precision_reduce: Reduce dtype (default: "float32")
            - reshard_after_forward: Whether to reshard params after forward
              (default: true). true saves memory, false is faster.

    Returns:
        Parallelized model
    """
    # 1. Grouped MoE (must come first — replaces MoE blocks with fused grouped_mm)
    model = apply_grouped_moe_qwen3(model)

    # 2. Activation Checkpointing
    ac_ratio = titan_config.get("activation_checkpoint", 0.0)
    if ac_ratio > 0.0:
        model = apply_activation_checkpoint_qwen3(model, ratio=ac_ratio)

    # 3. Torch Compile
    if titan_config.get("compile", False):
        model = apply_torch_compile_qwen3(model, titan_config)

    # 4. FSDP
    if parallel_dims.fsdp_enabled:
        model = apply_fsdp_qwen3(model, parallel_dims, titan_config)

    return model


def memory_efficient_load_balancing_loss(
    gate_logits, num_experts=None, top_k=2, attention_mask=None,
):
    """Memory-efficient load balancing loss — numerically identical to HF version.

    Eliminates the massive one_hot tensor (N_total, K, E) by using bincount.
    Processes per-layer to avoid concatenating all router logits.

    Memory: O(E) instead of O(N_total * K * E).
    """
    if gate_logits is None or not isinstance(gate_logits, tuple) or len(gate_logits) == 0:
        return 0

    device = gate_logits[0].device
    total_tokens = 0
    expert_counts = torch.zeros(num_experts, device=device)
    router_prob_sum = torch.zeros(num_experts, device=device)

    for layer_gate in gate_logits:
        total_tokens += layer_gate.shape[0]

        routing_weights = F.softmax(layer_gate, dim=-1, dtype=torch.float)
        _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

        expert_counts = expert_counts + torch.bincount(
            selected_experts.reshape(-1), minlength=num_experts
        ).float()
        router_prob_sum = router_prob_sum + routing_weights.sum(dim=0)

    tokens_per_expert = expert_counts / total_tokens
    router_prob_per_expert = router_prob_sum / total_tokens

    return torch.dot(tokens_per_expert, router_prob_per_expert) * num_experts


def apply_grouped_moe_qwen3(model: nn.Module) -> nn.Module:
    """Replace MoE blocks with GroupedMoeBlock for fused grouped_mm.

    Iterates through transformer layers, detects MoE layers, and replaces
    each Qwen3MoeSparseMoeBlock with a GroupedMoeBlock that uses fused
    grouped_mm computation. All experts remain on every rank.

    Must be applied BEFORE activation checkpointing, compile, and FSDP.

    Args:
        model: HuggingFace Qwen3 MoE model

    Returns:
        Model with MoE blocks replaced by GroupedMoeBlock
    """
    has_moe = False
    for layer in model.model.layers:
        if _is_moe_layer(layer):
            layer.mlp = GroupedMoeBlock(layer.mlp)
            has_moe = True

    if has_moe:
        # Attach memory-efficient load balancing loss for MoE auxiliary loss.
        # Replaces HF's load_balancing_loss_func which causes ~18GB memory spike
        # from one_hot expansion. Our version uses bincount for O(E) memory.
        model.load_balancing_loss_func = memory_efficient_load_balancing_loss

    return model


def apply_fsdp_qwen3(
    model: nn.Module,
    parallel_dims: ParallelDims,
    titan_config: Dict[str, Any],
) -> nn.Module:
    """Apply FSDP2 to HuggingFace Qwen3 model structure.

    Moves modules from CPU to GPU one FSDP unit at a time, then shards
    immediately. This avoids materializing the full model on every GPU
    (which would waste ~60GB for a 30B model). Peak GPU memory during
    init is ~1 transformer layer instead of the entire model.

    Strategy:
    - Move each module to GPU, then immediately shard via fully_shard.
    - Call empty_cache periodically to release freed non-local shards.
    - All layers (dense and MoE) are wrapped uniformly on dp_mesh.

    Args:
        model: HuggingFace Qwen3 model (on CPU) to wrap with FSDP
        parallel_dims: TorchTitan ParallelDims with device meshes
        titan_config: Configuration dict

    Returns:
        FSDP-wrapped model (on GPU, sharded)
    """
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    # (1) Build FSDP config
    param_dtype = getattr(torch, titan_config.get("mixed_precision_param", "bfloat16"))
    reduce_dtype = getattr(torch, titan_config.get("mixed_precision_reduce", "float32"))
    reshard_after_forward = titan_config.get("reshard_after_forward", True)

    if parallel_dims.dp_replicate_enabled:
        dp_mesh = parallel_dims.get_mesh(["dp_replicate", "fsdp"])
    else:
        dp_mesh = parallel_dims.get_mesh("fsdp")

    fsdp_config = {
        "mesh": dp_mesh,
        "mp_policy": MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype),
        "reshard_after_forward": reshard_after_forward,
    }

    def _move_and_shard(module: nn.Module):
        """Move module to GPU and immediately shard via FSDP."""
        module.to(device)
        fully_shard(module, **fsdp_config)

    # (2.1) input embeddings
    _move_and_shard(model.model.embed_tokens)

    # (2.2) layers — move one at a time to avoid full-model GPU peak
    for idx, layer in enumerate(model.model.layers):
        _move_and_shard(layer)

    # (2.3) norm, lm_head, stream_emb
    _move_and_shard(model.model.norm)
    _move_and_shard(model.lm_head)
    _move_and_shard(model.stream_emb)

    # (2.4) root — moves remaining modules (multimodal_io_dict, adaptor, etc.)
    # NOTE(Jinchuan): The FSDP2 DTensor operation doesn't support convolution ops.
    # We put all remained peripheral modules to the root FSDP2 unit, where the conv
    # ops can always stay locally and will not trigger the DTensor check.
    # We still don't know why the modules wrapped by root FSDP2 unit will not
    # trigger the DTensor check, but it works in practice.
    model.to(device)
    fully_shard(model, **fsdp_config)

    logger.info(
        f"Incremental FSDP init complete — peak GPU memory: "
        f"{torch.cuda.max_memory_allocated(device) / 1e9:.1f} GB"
    )

    # (2.5) Multi-layer FSDP prefetch (must be after all modules are sharded)
    _setup_fsdp_prefetch(model, titan_config)

    return model


def apply_activation_checkpoint_qwen3(
    model: nn.Module, ratio: float = 1.0
) -> nn.Module:
    """Apply activation checkpointing to transformer layers.

    Wraps transformer layers with checkpoint_wrapper for memory savings.
    Must be applied before torch.compile and FSDP.

    Args:
        model: HuggingFace Qwen3 model
        ratio: Fraction of layers to checkpoint (0.0-1.0).
            1.0 = all layers, 0.5 = every other layer, etc.

    Returns:
        Model with activation checkpointing applied
    """
    num_layers = len(model.model.layers)
    num_to_checkpoint = max(1, round(num_layers * ratio))

    # Evenly space checkpointed layers across the stack
    count = 0
    for idx in range(num_layers):
        if count < num_to_checkpoint and (idx + 1) * num_to_checkpoint > count * num_layers:
            model.model.layers[idx] = checkpoint_wrapper(model.model.layers[idx])
            count += 1

    logger.info(
        f"Applied activation checkpointing to {count}/{num_layers} layers "
        f"(ratio={ratio})"
    )
    return model


def apply_torch_compile_qwen3(
    model: nn.Module,
    titan_config: Dict[str, Any],
) -> nn.Module:
    """Apply torch.compile to transformer layers.

    Compiles each transformer layer individually. Must be applied after
    activation checkpointing and before FSDP.

    Args:
        model: HuggingFace Qwen3 model
        titan_config: Configuration dict

    Returns:
        Model with compiled transformer layers
    """
    compile_mode = titan_config.get("compile_mode", "default")

    torch._dynamo.config.capture_scalar_outputs = True
    # Disable LRU cache to prevent recompilation from MoE dynamic shapes
    torch._C._dynamo.eval_frame._set_lru_cache(False)

    for idx, layer in enumerate(model.model.layers):
        model.model.layers[idx] = torch.compile(layer, mode=compile_mode)

    logger.info(
        f"Applied torch.compile (mode={compile_mode}) to "
        f"{len(model.model.layers)} layers"
    )

    return model


def _setup_fsdp_prefetch(model: nn.Module, titan_config: Dict[str, Any]) -> None:
    """Set up multi-layer FSDP prefetch for forward and backward passes.

    After each layer's all-gather copy-out, FSDP will immediately issue
    all-gathers for the next ``prefetch_depth`` layers, overlapping
    communication with the current layer's compute.

    Memory cost: ~1.2GB per prefetched MoE layer (unsharded bf16 params).

    Args:
        model: FSDP-wrapped HuggingFace Qwen3 model
        titan_config: Must contain ``prefetch_depth`` (int, default 0).
            0 = no explicit prefetch (FSDP default behavior).
            1 = same as default 1-layer-ahead prefetch but issued earlier.
            2+ = aggressive multi-layer prefetch.
    """
    depth = titan_config.get("prefetch_depth", 0)
    if depth <= 0:
        return

    layers = list(model.model.layers)
    num_layers = len(layers)

    # --- Forward prefetch ---
    # embed_tokens → first `depth` transformer layers
    fwd_targets = layers[:min(depth, num_layers)]
    model.model.embed_tokens.set_modules_to_forward_prefetch(fwd_targets)

    for i, layer in enumerate(layers):
        next_layers = layers[i + 1 : i + 1 + depth]
        if next_layers:
            layer.set_modules_to_forward_prefetch(next_layers)
        elif i == num_layers - 1:
            # Last transformer layer → prefetch norm + lm_head + stream_emb
            layer.set_modules_to_forward_prefetch(
                [model.model.norm, model.lm_head, model.stream_emb]
            )

    # --- Backward prefetch ---
    # In backward, layers execute in reverse order. Each layer prefetches
    # `depth` earlier layers (which are the "next" in backward execution).
    # stream_emb/lm_head → last `depth` transformer layers
    bwd_targets = layers[max(0, num_layers - depth):]
    model.lm_head.set_modules_to_backward_prefetch(bwd_targets)
    model.stream_emb.set_modules_to_backward_prefetch(bwd_targets)

    for i in range(num_layers - 1, -1, -1):
        prev_layers = layers[max(0, i - depth) : i]
        if prev_layers:
            layers[i].set_modules_to_backward_prefetch(prev_layers)
        elif i == 0:
            # First transformer layer → prefetch embed_tokens
            layers[0].set_modules_to_backward_prefetch(
                [model.model.embed_tokens]
            )

    logger.info(
        f"Set up {depth}-layer FSDP prefetch on {num_layers} layers "
        f"(forward + backward)"
    )


