"""Fused linear cross-entropy loss using Liger Kernel.

Replaces the separate matmul + cross_entropy with a single fused kernel
that never materializes the full [B*T, V] logits tensor, saving ~19GB
for typical configurations (32k tokens × 150k vocab).
"""

from typing import Dict, Optional, Tuple

import torch

from liger_kernel.ops.fused_linear_cross_entropy import (
    LigerFusedLinearCrossEntropyFunction,
)


def fused_cross_entropy_loss(
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    lm_head_weight: torch.Tensor,
    multimodal_vocab_range: Optional[Tuple[int, int]],
    num_stream: int,
    training: bool,
    ce_weight: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """Compute cross-entropy loss using Liger's fused linear + CE kernel.

    Uses reduction="sum" with pre-masked targets to work around Liger's
    reduction="none" backward bug. Two Liger calls: one for stream 0
    (full vocab) and one for streams 1+ (multimodal vocab subset).

    Args:
        hidden_states: [B, T, N, H] — unshifted
        input_ids: [B, T, N] — unshifted
        loss_mask: [B, T, N] — unshifted, float (0/1)
        lm_head_weight: [V, H] — may be DTensor
        multimodal_vocab_range: (mm_start, mm_end) or None
        num_stream: int
        training: bool
        ce_weight: [V] optional per-class weight

    Returns:
        (ce_loss scalar, count scalar, stats dict)
    """
    # Convert DTensor to regular tensor if needed (e.g. FSDP2)
    if hasattr(lm_head_weight, "full_tensor"):
        lm_head_weight = lm_head_weight.full_tensor()

    # Shift for next-token prediction
    hidden_states = hidden_states[:, :-1]
    input_ids = input_ids[:, 1:]
    loss_mask = loss_mask[:, 1:]

    B, T, N, H = hidden_states.shape
    stats = {}

    # ---- Stream 0: full vocabulary ----
    s0_hidden = hidden_states[:, :, 0].reshape(-1, H)
    s0_targets = input_ids[:, :, 0].reshape(-1).clone()
    s0_mask = loss_mask[:, :, 0].reshape(-1)

    # Pre-mask: set ignored positions to ignore_index=0 (pad token)
    s0_targets[s0_mask == 0] = 0

    s0_loss, _, s0_acc = LigerFusedLinearCrossEntropyFunction.apply(
        s0_hidden,          # _input: [B*T, H]
        lm_head_weight,     # weight: [V, H]
        s0_targets,         # target: [B*T]
        None,               # bias
        ce_weight,          # ce_weight
        0,                  # ignore_index
        0.0,                # lse_square_scale
        0.0,                # label_smoothing
        "sum",              # reduction
        None,               # softcap
        False,              # return_z_loss
        None,               # accum_dtype
        False,              # use_token_scaling
        True,               # return_token_accuracy
    )

    # ---- Streams 1+: multimodal vocab subset (single call) ----
    mm_loss = torch.tensor(0.0, device=hidden_states.device)
    mm_acc = None

    if num_stream > 1 and multimodal_vocab_range is not None:
        mm_start, mm_end = multimodal_vocab_range

        mm_hidden = hidden_states[:, :, 1:].reshape(-1, H)
        mm_targets = input_ids[:, :, 1:].reshape(-1)
        mm_mask = loss_mask[:, :, 1:].reshape(-1)

        # Remap targets to local indices within [mm_start, mm_end)
        valid = (mm_targets >= mm_start) & (mm_targets < mm_end)
        mm_targets_adj = torch.where(
            valid,
            mm_targets - mm_start,
            torch.tensor(-100, device=mm_targets.device),
        )

        # Pre-mask: set ignored positions to -100 (can't use 0, it's a valid local index)
        mm_targets_adj[mm_mask == 0] = -100

        mm_ce_weight = ce_weight[mm_start:mm_end] if ce_weight is not None else None

        mm_loss, _, mm_acc = LigerFusedLinearCrossEntropyFunction.apply(
            mm_hidden,                          # _input: [B*T*(N-1), H]
            lm_head_weight[mm_start:mm_end],    # weight: [mm_V, H]
            mm_targets_adj,                     # target: [B*T*(N-1)]
            None,                               # bias
            mm_ce_weight,                       # ce_weight
            -100,                               # ignore_index
            0.0,                                # lse_square_scale
            0.0,                                # label_smoothing
            "sum",                              # reduction
            None,                               # softcap
            False,                              # return_z_loss
            None,                               # accum_dtype
            False,                              # use_token_scaling
            True,                               # return_token_accuracy
        )

    # ---- Combine ----
    count = (loss_mask[:, :, 0] != 0).float().sum()
    ce_loss = (s0_loss + mm_loss) / count

    # ---- Accuracy stats ----
    if s0_acc is not None:
        stats["acc_layer0"] = s0_acc
    if mm_acc is not None:
        stats["acc_audio"] = mm_acc

    return ce_loss, count, stats
