# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Parallelization utilities for HuggingFace Qwen3 models.

This module provides Expert Parallelism, activation checkpointing,
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
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torchtitan.distributed import ParallelDims

from espnet2.speechlm.model.speechlm.parallel_utils.expert_parallel import (
    ExpertParallelMoeBlock,
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

    Order: EP -> AC -> torch.compile -> FSDP (EP-aware)
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
    # 1. Expert Parallelism (must come first — replaces MoE blocks)
    if parallel_dims.ep_enabled:
        model = apply_expert_parallel_qwen3(model, parallel_dims)

        # Attach load balancing loss function for MoE auxiliary loss.
        # This is a standalone function in HF transformers that must be
        # explicitly bound to the model for ParallelLLM._loss() to use it.
        from transformers.models.qwen3_moe.modeling_qwen3_moe import (
            load_balancing_loss_func,
        )
        model.load_balancing_loss_func = load_balancing_loss_func

    # 2. Activation Checkpointing
    ac_ratio = titan_config.get("activation_checkpoint", 0.0)
    if ac_ratio:
        if ac_ratio is True:
            ac_ratio = 1.0
        model = apply_activation_checkpoint_qwen3(model, ratio=ac_ratio)

    # 3. Torch Compile
    if titan_config.get("compile", False):
        model = apply_torch_compile_qwen3(model, titan_config)

    # 4. FSDP (EP-aware wrapping for MoE layers)
    if parallel_dims.fsdp_enabled:
        model = apply_fsdp_qwen3(model, parallel_dims, titan_config)

    return model


def apply_expert_parallel_qwen3(
    model: nn.Module,
    parallel_dims: ParallelDims,
) -> nn.Module:
    """Replace MoE blocks with Expert Parallel versions.

    Iterates through transformer layers, detects MoE layers, and replaces
    each Qwen3MoeSparseMoeBlock with an ExpertParallelMoeBlock that
    distributes experts across EP ranks via all-to-all communication.

    Must be applied BEFORE activation checkpointing, compile, and FSDP.

    Args:
        model: HuggingFace Qwen3 MoE model
        parallel_dims: TorchTitan ParallelDims with EP mesh

    Returns:
        Model with MoE blocks replaced by EP-aware blocks
    """
    ep_mesh = parallel_dims.get_mesh("ep")
    for layer in model.model.layers:
        if _is_moe_layer(layer):
            layer.mlp = ExpertParallelMoeBlock(layer.mlp, ep_mesh)
    return model


def apply_fsdp_qwen3(
    model: nn.Module,
    parallel_dims: ParallelDims,
    titan_config: Dict[str, Any],
) -> nn.Module:
    """Apply FSDP2 to HuggingFace Qwen3 model structure.

    Strategy:
    - Only wrap modules that have trainable parameters.
    - Wrap individual components first, then the root model.
    - For MoE layers with EP: wrap GroupedExperts on edp_mesh (with
      gradient_divide_factor) BEFORE wrapping the layer on dp_mesh.
    - After wrapping, permanently unshard peripheral components (norm, lm_head,
      stream_emb, multimodal_io_dict, adaptor) to avoid repeated all-gather
      overhead for small or frequently-accessed modules. Only embed_tokens and
      transformer layers stay in the normal FSDP shard/unshard cycle.

    Args:
        model: HuggingFace Qwen3 model to wrap with FSDP
        parallel_dims: TorchTitan ParallelDims with device meshes
        titan_config: Configuration dict

    Returns:
        FSDP-wrapped model
    """
    # (1) Build FSDP config for dense (non-expert) params
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

    # Build expert FSDP config (only when EP is enabled)
    ep_enabled = parallel_dims.ep_enabled
    if ep_enabled:
        edp_mesh_names = (
            ["dp_replicate", "efsdp"]
            if parallel_dims.dp_replicate_enabled
            else ["efsdp"]
        )
        edp_mesh = parallel_dims.get_optional_mesh(edp_mesh_names)
        fsdp_ep_config = {
            "mesh": edp_mesh,
            "mp_policy": MixedPrecisionPolicy(
                param_dtype=param_dtype, reduce_dtype=reduce_dtype
            ),
            "reshard_after_forward": reshard_after_forward,
        }
        gradient_divide_factor = parallel_dims.fsdp_gradient_divide_factor

    def _shard(module: nn.Module):
        fully_shard(module, **fsdp_config)

    # (2.1) input embeddings
    _shard(model.model.embed_tokens)

    # (2.2) layers (with EP-aware expert wrapping for MoE layers)
    for layer in model.model.layers:
        if ep_enabled and isinstance(layer.mlp, ExpertParallelMoeBlock):
            # Wrap GroupedExperts (stacked expert weights) on edp_mesh
            # BEFORE wrapping the layer on dp_mesh. FSDP2 hooks fire on
            # GroupedExperts.__call__() which is invoked during forward.
            fully_shard(layer.mlp.experts, **fsdp_ep_config)
            layer.mlp.experts.set_gradient_divide_factor(
                gradient_divide_factor
            )
        _shard(layer)

    # (2.3) norm, lm_head, stream_emb
    _shard(model.model.norm)
    _shard(model.lm_head)
    _shard(model.stream_emb)

    # (2.5) root
    # NOTE(Jinchuan): The FSDP2 DTensor operation doesn't support convolution ops.
    # We put all remained peripheral modules to the root FSDP2 unit, where the conv
    # ops can always stay locally and will not trigger the DTensor check.
    # We still don't know why the modules wrapped by root FSDP2 unit will not 
    # trigger the DTensor check, but it works in practice.
    fully_shard(model, **fsdp_config)

    # (3) Set up explicit FSDP prefetching when EP is enabled
    _setup_fsdp_prefetching(model, ep_enabled)

    return model


def _setup_fsdp_prefetching(model: nn.Module, ep_enabled: bool):
    """Set up explicit FSDP forward/backward prefetching for transformer layers.

    When EP is enabled, D2H syncs in EP can interfere with FSDP's implicit
    prefetching. Explicit prefetching ensures the next layer's params are
    all-gathered while the current layer computes. Follows torchtitan's pattern.
    """
    if not ep_enabled:
        return

    layers = list(model.model.layers)
    if not layers:
        return

    # Forward: embed_tokens prefetches layer[0]; layer[i] prefetches layer[i+1]
    if hasattr(model.model.embed_tokens, "set_modules_to_forward_prefetch"):
        model.model.embed_tokens.set_modules_to_forward_prefetch([layers[0]])

    for i in range(len(layers) - 1):
        layer = layers[i]
        next_layer = layers[i + 1]
        if not hasattr(layer, "set_modules_to_forward_prefetch"):
            continue
        if isinstance(next_layer.mlp, ExpertParallelMoeBlock):
            layer.set_modules_to_forward_prefetch(
                [next_layer, next_layer.mlp.experts]
            )
        else:
            layer.set_modules_to_forward_prefetch([next_layer])

    # Backward: layer[i] prefetches layer[i-1]; layer[0] prefetches embed_tokens
    reversed_layers = list(reversed(layers))
    for i in range(len(reversed_layers) - 1):
        layer = reversed_layers[i]
        prev_layer = reversed_layers[i + 1]
        if not hasattr(layer, "set_modules_to_backward_prefetch"):
            continue
        if isinstance(prev_layer.mlp, ExpertParallelMoeBlock):
            layer.set_modules_to_backward_prefetch(
                [prev_layer, prev_layer.mlp.experts]
            )
        else:
            layer.set_modules_to_backward_prefetch([prev_layer])

    if hasattr(reversed_layers[-1], "set_modules_to_backward_prefetch"):
        reversed_layers[-1].set_modules_to_backward_prefetch(
            [model.model.embed_tokens]
        )

    logger.info(
        f"Set up explicit FSDP prefetching for {len(layers)} layers (EP enabled)"
    )


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
