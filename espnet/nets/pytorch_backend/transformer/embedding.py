#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Positional Encoding Module."""

import logging
import math

import torch
from packaging.version import parse as V


def _pre_hook(
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
):
    """Perform pre-hook in load_state_dict for backward compatibility.

    Note:
        We saved self.pe until v.0.5.2 but we have omitted it later.
        Therefore, we remove the item "pe" from `state_dict` for backward compatibility.

    """
    k = prefix + "pe"
    if k in state_dict:
        state_dict.pop(k)


class PositionalEncoding(torch.nn.Module):
    """Positional encoding.

    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.
        reverse (bool): Whether to reverse the input position. Only for
        the class LegacyRelPositionalEncoding. We remove it in the current
        class RelPositionalEncoding.
    """

    def __init__(self, d_model, dropout_rate, max_len=5000, reverse=False):
        """Construct an PositionalEncoding object."""
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.reverse = reverse
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))
        self._register_load_state_dict_pre_hook(_pre_hook)

    def extend_pe(self, x):
        """Reset the positional encodings."""
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1):
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        pe = torch.zeros(x.size(1), self.d_model)
        if self.reverse:
            position = torch.arange(
                x.size(1) - 1, -1, -1.0, dtype=torch.float32
            ).unsqueeze(1)
        else:
            position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor):
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).

        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
        """
        self.extend_pe(x)
        x = x * self.xscale + self.pe[:, : x.size(1)]
        return self.dropout(x)


class ScaledPositionalEncoding(PositionalEncoding):
    """Scaled positional encoding module.

    See Sec. 3.2  https://arxiv.org/abs/1809.08895

    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.

    """

    def __init__(self, d_model, dropout_rate, max_len=5000):
        """Initialize class."""
        super().__init__(d_model=d_model, dropout_rate=dropout_rate, max_len=max_len)
        self.alpha = torch.nn.Parameter(torch.tensor(1.0))

    def reset_parameters(self):
        """Reset parameters."""
        self.alpha.data = torch.tensor(1.0)

    def forward(self, x):
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).

        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).

        """
        self.extend_pe(x)
        x = x + self.alpha * self.pe[:, : x.size(1)]
        return self.dropout(x)


class LearnableFourierPosEnc(torch.nn.Module):
    """Learnable Fourier Features for Positional Encoding.

    See https://arxiv.org/pdf/2106.02795.pdf

    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.
        gamma (float): init parameter for the positional kernel variance
            see https://arxiv.org/pdf/2106.02795.pdf.
        apply_scaling (bool): Whether to scale the input before adding the pos encoding.
        hidden_dim (int): if not None, we modulate the pos encodings with
            an MLP whose hidden layer has hidden_dim neurons.
    """

    def __init__(
        self,
        d_model,
        dropout_rate=0.0,
        max_len=5000,
        gamma=1.0,
        apply_scaling=False,
        hidden_dim=None,
    ):
        """Initialize class."""
        super(LearnableFourierPosEnc, self).__init__()

        self.d_model = d_model

        if apply_scaling:
            self.xscale = math.sqrt(self.d_model)
        else:
            self.xscale = 1.0

        self.dropout = torch.nn.Dropout(dropout_rate)
        self.max_len = max_len

        self.gamma = gamma
        if self.gamma is None:
            self.gamma = self.d_model // 2

        assert (
            d_model % 2 == 0
        ), "d_model should be divisible by two in order to use this layer."
        self.w_r = torch.nn.Parameter(torch.empty(1, d_model // 2))
        self._reset()  # init the weights

        self.hidden_dim = hidden_dim
        if self.hidden_dim is not None:
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(d_model, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, d_model),
            )

    def _reset(self):
        self.w_r.data = torch.normal(
            0, (1 / math.sqrt(self.gamma)), (1, self.d_model // 2)
        )

    def extend_pe(self, x):
        """Reset the positional encodings."""
        position_v = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1).to(x)

        cosine = torch.cos(torch.matmul(position_v, self.w_r))
        sine = torch.sin(torch.matmul(position_v, self.w_r))
        pos_enc = torch.cat((cosine, sine), -1)
        pos_enc /= math.sqrt(self.d_model)

        if self.hidden_dim is None:
            return pos_enc.unsqueeze(0)
        else:
            return self.mlp(pos_enc.unsqueeze(0))

    def forward(self, x: torch.Tensor):
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).

        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
        """
        pe = self.extend_pe(x)
        x = x * self.xscale + pe
        return self.dropout(x)


class LegacyRelPositionalEncoding(PositionalEncoding):
    """Relative positional encoding module (old version).

    Details can be found in https://github.com/espnet/espnet/pull/2816.

    See : Appendix B in https://arxiv.org/abs/1901.02860

    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.

    """

    def __init__(self, d_model, dropout_rate, max_len=5000):
        """Initialize class."""
        super().__init__(
            d_model=d_model,
            dropout_rate=dropout_rate,
            max_len=max_len,
            reverse=True,
        )

    def forward(self, x):
        """Compute positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).

        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
            torch.Tensor: Positional embedding tensor (1, time, `*`).

        """
        self.extend_pe(x)
        x = x * self.xscale
        pos_emb = self.pe[:, : x.size(1)]
        return self.dropout(x), self.dropout(pos_emb)


class RelPositionalEncoding(torch.nn.Module):
    """Relative positional encoding module (new implementation).

    Details can be found in https://github.com/espnet/espnet/pull/2816.

    See : Appendix B in https://arxiv.org/abs/1901.02860

    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.

    """

    def __init__(self, d_model, dropout_rate, max_len=5000):
        """Construct an PositionalEncoding object."""
        super(RelPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x):
        """Reset the positional encodings."""
        if self.pe is not None:
            # self.pe contains both positive and negative parts
            # the length of self.pe is 2 * input_len - 1
            if self.pe.size(1) >= x.size(1) * 2 - 1:
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        # Suppose `i` means to the position of query vecotr and `j` means the
        # position of key vector. We use position relative positions when keys
        # are to the left (i>j) and negative relative positions otherwise (i<j).
        pe_positive = torch.zeros(x.size(1), self.d_model)
        pe_negative = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        # Reserve the order of positive indices and concat both positive and
        # negative indices. This is used to support the shifting trick
        # as in https://arxiv.org/abs/1901.02860
        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor):
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).

        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).

        """
        self.extend_pe(x)
        x = x * self.xscale
        pos_emb = self.pe[
            :,
            self.pe.size(1) // 2 - x.size(1) + 1 : self.pe.size(1) // 2 + x.size(1),
        ]
        return self.dropout(x), self.dropout(pos_emb)


class StreamPositionalEncoding(torch.nn.Module):
    """Streaming Positional encoding.

    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.

    """

    def __init__(self, d_model, dropout_rate, max_len=5000):
        """Construct an PositionalEncoding object."""
        super(StreamPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.pe = None
        self.tmp = torch.tensor(0.0).expand(1, max_len)
        self.extend_pe(self.tmp.size(1), self.tmp.device, self.tmp.dtype)
        self._register_load_state_dict_pre_hook(_pre_hook)

    def extend_pe(self, length, device, dtype):
        """Reset the positional encodings."""
        if self.pe is not None:
            if self.pe.size(1) >= length:
                if self.pe.dtype != dtype or self.pe.device != device:
                    self.pe = self.pe.to(dtype=dtype, device=device)
                return
        pe = torch.zeros(length, self.d_model)
        position = torch.arange(0, length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe.to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, start_idx: int = 0):
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).

        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).

        """
        self.extend_pe(x.size(1) + start_idx, x.device, x.dtype)
        x = x * self.xscale + self.pe[:, start_idx : start_idx + x.size(1)]
        return self.dropout(x)


class ConvolutionalPositionalEmbedding(torch.nn.Module):
    """Convolutional positional embedding.
       Used in wav2vec2/HuBERT SSL models.
       https://arxiv.org/abs/1904.11660

    Args:
        embed_dim (int): Feature dimension of the input Tensor.
        dropout (float): unused
        max_len (int): unused
        kernel_size (int): The number of frames to be use.
        groups (int): The number of groups in feature dimensions.
    """

    def __init__(
        self,
        embed_dim: int,
        dropout: float,
        max_len: int = 5000,
        num_layers: int = 1,
        kernel_size: int = 128,
        groups: int = 16,
        weight_norm: str = "legacy",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        convs = []
        for layer in range(num_layers):
            conv = torch.nn.Conv1d(
                in_channels=embed_dim,
                out_channels=embed_dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=groups,
            )
            # torch.nn.utils.weight_norm leads to weird behavior with copy.deepcopy()
            # usually isnt needed, but its important for models that use EMA
            if weight_norm == "new":
                if V(torch.__version__) >= V("2.2.0"):
                    conv = torch.nn.utils.parametrizations.weight_norm(
                        conv, name="weight", dim=2
                    )
                else:
                    weight_norm = "legacy"
                    logging.warning(
                        f"torch.nn.utils.parametrizations.weight_norm is only "
                        + "supported for pytorch versions >= 2.2.0. "
                        + "Defaulting to torch.nn.utils.weight_norm."
                    )
            if weight_norm == "legacy":
                conv = torch.nn.utils.weight_norm(conv, name="weight", dim=2)
            convs.append(conv)
        self.convs = torch.nn.ModuleList(convs)
        self.num_remove: int = 1 if kernel_size % 2 == 0 else 0

    def __prepare_scriptable__(self):
        for hook in self.conv._forward_pre_hooks.values():
            # The hook we want to remove is an instance of WeightNorm class, so
            # normally we would do `if isinstance(...)` but this class is not accessible
            # because of shadowing, so we check the module name directly.
            # https://github.com/pytorch/pytorch/blob/be0ca00c5ce260eb5bcec3237357f7a30cc08983/torch/nn/utils/__init__.py#L3
            if (
                hook.__module__ == "torch.nn.utils.weight_norm"
                and hook.__class__.__name__ == "WeightNorm"
            ):
                _LG.warning("Removing weight_norm from %s", self.__class__.__name__)
                torch.nn.utils.remove_weight_norm(self.conv)
        return self

    def forward(self, x):
        """
        Args:
            x (Tensor): shape ``[batch, frame, feature]``.

        Returns:
            Tensor: The resulting feature. Shape ``[batch, frame, feature]``.
        """
        x = x.transpose(-2, -1)
        for conv in self.convs:
            x = conv(x)
            if self.num_remove > 0:
                x = x[..., : -self.num_remove]
            x = torch.nn.functional.gelu(x)
        x = x.transpose(-2, -1)
        return x


class RoPEPositionalEncoding(torch.nn.Module):
    """Rotary Position Embedding (RoPE) module.
    
    As described in "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    https://arxiv.org/abs/2104.09864
    
    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.
    """
    
    def __init__(self, d_model, dropout_rate, max_len=5000):
        """Construct a RoPEPositionalEncoding object."""
        super(RoPEPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.max_len = max_len
        
        # RoPE requires dimension to be even
        assert d_model % 2 == 0, "d_model must be even for RoPE"
        
        # Create frequency matrix for RoPE
        # Each pair of dimensions shares the same frequency
        # freq_seq ranges from 0 to d_model/2-1
        freq_seq = torch.arange(0, d_model, 2, dtype=torch.float32)
        
        # Calculate frequencies: theta_i = 10000^(-2i/d)
        self.inv_freq = 1.0 / (10000 ** (freq_seq / d_model))
        
        # We'll compute the actual position embeddings during the forward pass
        self.cached_rotations = None
        self.cached_seq_len = 0
    
    def _compute_rope_embeddings(self, seq_len, device, dtype):
        """Compute rotary position embeddings."""
        # If we've already computed for this sequence length and device, reuse
        if self.cached_rotations is not None and seq_len <= self.cached_seq_len and self.cached_rotations.device == device:
            return self.cached_rotations[:seq_len]
        
        # Create position indices
        positions = torch.arange(0, seq_len, dtype=torch.float32, device=device).unsqueeze(1)
        
        # Compute position * frequency for each dimension
        # This gives us the rotation angles
        angles = positions * self.inv_freq.to(device)
        
        # For each position, create a rotation matrix for each pair of dimensions
        cos = torch.cos(angles).repeat_interleave(2, dim=-1)  # [seq_len, d_model]
        sin = torch.sin(angles).repeat_interleave(2, dim=-1)  # [seq_len, d_model]
        
        # Prepare sin and cos for the rotation
        # For even indices: [cos, sin]
        # For odd indices: [-sin, cos]
        sin_cos = torch.stack([cos, sin], dim=-1).reshape(seq_len, self.d_model, 2)
        
        # Cache results for reuse
        self.cached_rotations = sin_cos
        self.cached_seq_len = seq_len
        
        return sin_cos
    
    def _apply_rotary_pos_emb(self, x, sin_cos):
        """Apply rotary position embeddings to input tensor x."""
        # Reshape x to [batch, seq_len, ..., d_model]
        original_shape = x.shape
        x = x.view(*x.shape[:-1], -1, 2)
        
        # Extract cos and sin components
        cos, sin = sin_cos[..., 0], sin_cos[..., 1]
        
        # For each position and pair of dimensions, apply the rotation:
        # [x_i, x_{i+1}] -> [x_i*cos - x_{i+1}*sin, x_i*sin + x_{i+1}*cos]
        x1 = x[..., 0::2]  # even indices
        x2 = x[..., 1::2]  # odd indices
        
        # Apply rotation
        rotated_x = torch.stack([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1).flatten(-2)
        
        # Restore original shape
        rotated_x = rotated_x.reshape(original_shape)
        
        return rotated_x
    
    def extend_pe(self, x):
        """Reset the positional encodings cache if needed."""
        seq_len = x.size(1)
        if self.cached_rotations is None or seq_len > self.cached_seq_len or self.cached_rotations.device != x.device:
            self._compute_rope_embeddings(max(seq_len, self.max_len), x.device, x.dtype)
    
    def forward(self, x: torch.Tensor):
        """Add positional encoding.
        
        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).
            
        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
            torch.Tensor: Position embeddings for attention computation.
        """
        self.extend_pe(x)
        seq_len = x.size(1)
        
        # Scale the embeddings
        x = x * self.xscale
        
        # Get the rotations for the current sequence length
        sin_cos = self._compute_rope_embeddings(seq_len, x.device, x.dtype)
        
        # In RoPE, we return the input itself as the first tensor
        # and the rotation matrices as the second tensor
        # The actual rotation is applied within the attention mechanism
        return self.dropout(x), sin_cos