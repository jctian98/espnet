# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Grouped MoE block for HuggingFace Qwen3 MoE models.

This module replaces `Qwen3MoeSparseMoeBlock` with a `GroupedMoeBlock` that
uses `torch._grouped_mm` for fused multi-expert computation (instead of a
Python loop over individual experts).

All experts reside on every rank (EP=1). Only FSDP is used for parallelism.

Data flow per MoE block:
    1. Route tokens (gate computation)
    2. Sort token-expert pairs by expert index
    3. Count tokens per expert (histc)
    4. Process all experts via fused grouped_mm
    5. Permutation scatter + weighted sum (no index_add_)
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeSparseMoeBlock,
)

logger = logging.getLogger(__name__)


def _run_grouped_mm(w1, w2, w3, x, num_tokens_per_expert):
    """Core SwiGLU grouped_mm computation.

    SwiGLU: out = down_proj(silu(gate_proj(x)) * up_proj(x))
    Uses torch._grouped_mm which requires bfloat16 inputs.
    """
    offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
    x_bf16 = x if x.dtype == torch.bfloat16 else x.bfloat16()
    w1_t = (
        w1.transpose(-2, -1)
        if w1.dtype == torch.bfloat16
        else w1.bfloat16().transpose(-2, -1)
    )
    w3_t = (
        w3.transpose(-2, -1)
        if w3.dtype == torch.bfloat16
        else w3.bfloat16().transpose(-2, -1)
    )
    w2_t = (
        w2.transpose(-2, -1)
        if w2.dtype == torch.bfloat16
        else w2.bfloat16().transpose(-2, -1)
    )
    h = F.silu(torch._grouped_mm(x_bf16, w1_t, offs=offsets))
    h = h * torch._grouped_mm(x_bf16, w3_t, offs=offsets)
    return torch._grouped_mm(h, w2_t, offs=offsets).type_as(x)


class GroupedExperts(nn.Module):
    """Stacked expert weights with fused grouped_mm computation.

    Stacks individual expert Linear weights into 3D Parameter tensors
    and uses torch._grouped_mm for fused multi-expert matrix multiplication.

    Weight layout (Qwen3 MoE SwiGLU, no biases):
        w1 (gate_proj): (num_experts, intermediate_size, hidden_size)
        w2 (down_proj): (num_experts, hidden_size, intermediate_size)
        w3 (up_proj):   (num_experts, intermediate_size, hidden_size)

    Forward: out = down_proj(silu(gate_proj(x)) * up_proj(x))

    Args:
        experts: List of Qwen3MoeMLP modules to stack.
    """

    def __init__(self, experts: list):
        super().__init__()
        self.num_experts = len(experts)

        # Verify no biases (Qwen3 MoE experts don't have biases)
        assert experts[0].gate_proj.bias is None, (
            "GroupedExperts does not support biases"
        )

        # Stack weights from individual experts into 3D tensors
        self.w1 = nn.Parameter(
            torch.stack([e.gate_proj.weight for e in experts])
        )
        self.w2 = nn.Parameter(
            torch.stack([e.down_proj.weight for e in experts])
        )
        self.w3 = nn.Parameter(
            torch.stack([e.up_proj.weight for e in experts])
        )

        logger.info(
            f"GroupedExperts: stacked {self.num_experts} experts, "
            f"w1={list(self.w1.shape)}, w2={list(self.w2.shape)}, "
            f"w3={list(self.w3.shape)}"
        )

    def forward(
        self,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        """Process tokens through stacked experts using grouped_mm.

        Args:
            x: Input tokens sorted by expert, shape (total_tokens, hidden_dim).
            num_tokens_per_expert: Token count per expert,
                shape (num_experts,).

        Returns:
            Output tensor, shape (total_tokens, hidden_dim).
        """
        # Extract local tensors from DTensor (torch._grouped_mm requires
        # regular tensors, not DTensors — needed for FSDP2 compatibility)
        if isinstance(self.w1, DTensor):
            w1, w2, w3 = self.w1.to_local(), self.w2.to_local(), self.w3.to_local()
        else:
            w1, w2, w3 = self.w1, self.w2, self.w3

        return _run_grouped_mm(w1, w2, w3, x, num_tokens_per_expert)


class GroupedMoeBlock(Qwen3MoeSparseMoeBlock):
    """MoE block using fused grouped_mm for all experts (no EP).

    Inherits from `Qwen3MoeSparseMoeBlock` so that HF's `OutputRecorder`
    (which uses `isinstance` check) can find this module and capture
    router_logits for the load balancing auxiliary loss.

    All experts reside on every rank. Uses GroupedExperts with
    torch._grouped_mm for fused multi-expert computation instead of a
    Python loop over individual experts.

    Args:
        original_block: The original Qwen3MoeSparseMoeBlock to replace.
    """

    def __init__(self, original_block: Qwen3MoeSparseMoeBlock):
        # Skip parent __init__ (which would create all experts from config).
        # Directly call nn.Module.__init__ instead.
        nn.Module.__init__(self)

        self.num_experts = original_block.num_experts
        self.top_k = original_block.top_k
        self.norm_topk_prob = original_block.norm_topk_prob

        # Router (replicated on all ranks — same gate weights)
        self.gate = original_block.gate

        # Stack ALL experts into GroupedExperts for fused grouped_mm
        self.experts = GroupedExperts(list(original_block.experts))

        logger.info(
            f"GroupedMoeBlock: {self.num_experts} experts, "
            f"top_k={self.top_k}, using grouped_mm"
        )

    @staticmethod
    def _record(ev_list, tag):
        """Record a CUDA event into ev_list for profiling."""
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        ev_list.append((tag, ev))

    def forward(self, hidden_states: torch.Tensor) -> tuple:
        """Forward pass with fused grouped_mm computation.

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_dim).

        Returns:
            Tuple of (output, router_logits):
                - output: shape (batch, seq_len, hidden_dim)
                - router_logits: shape (batch * seq_len, num_experts)
        """
        ev = getattr(self, "_profile_events", None)
        # Only profile the first forward pass; skip AC recompute
        do_profile = ev is not None and len(ev) == 0

        bsz, seq_len, hidden_dim = hidden_states.shape
        num_tokens = bsz * seq_len
        hidden_states_flat = hidden_states.view(-1, hidden_dim)

        if do_profile:
            self._record(ev, "start")

        # --- Step 1: Route ---
        router_logits = self.gate(hidden_states_flat)  # (T, E)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )  # (T, K)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states_flat.dtype)

        if do_profile:
            self._record(ev, "route")

        # --- Step 2: Sort by expert index ---
        flat_expert_indices = selected_experts.view(-1)  # (T*K,)
        sort_order = flat_expert_indices.argsort(stable=True)  # (T*K,)

        # Gather tokens in expert-sorted order
        token_indices = (
            torch.arange(num_tokens, device=hidden_states.device)
            .unsqueeze(1)
            .expand(-1, self.top_k)
            .reshape(-1)
        )  # (T*K,)
        sorted_tokens = hidden_states_flat[token_indices[sort_order]]  # (T*K, D)

        # Count tokens per expert
        num_tokens_per_expert = torch.histc(
            flat_expert_indices.float(),
            bins=self.num_experts,
            min=0,
            max=self.num_experts,
        ).long()

        if do_profile:
            self._record(ev, "sort")

        # --- Step 3: Process via grouped_mm (with padding for alignment) ---
        processed_tokens = self.experts(sorted_tokens, num_tokens_per_expert)

        if do_profile:
            self._record(ev, "experts")

        # --- Step 4: Unsort via gather + weighted sum ---
        # Compute inverse permutation via cheap int-only scatter (2MB),
        # then gather the full (T*K, D) tensor (sequential writes, faster
        # than the scatter approach which has random writes on 1.8GB).
        TK = num_tokens * self.top_k
        inv_sort_order = torch.empty(
            TK, dtype=sort_order.dtype, device=sort_order.device
        )
        inv_sort_order[sort_order] = torch.arange(
            TK, dtype=sort_order.dtype, device=sort_order.device
        )
        unsorted = processed_tokens[inv_sort_order]  # gather: sequential writes
        # routing_weights is already (T, K) in original token order
        final_output = (
            unsorted.view(num_tokens, self.top_k, hidden_dim)
            * routing_weights.unsqueeze(-1)
        ).sum(dim=1)

        if do_profile:
            self._record(ev, "unsort")

        return final_output.view(bsz, seq_len, hidden_dim), router_logits
