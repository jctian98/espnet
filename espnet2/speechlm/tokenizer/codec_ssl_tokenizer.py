#!/usr/bin/env python3

# Copyright 2025 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import torch
import os
import numpy as np
import librosa
import logging

from espnet2.speechlm.tokenizer.abs_tokenizer import AbsTokenizer
from espnet2.speechlm.tokenizer.codec_tokenizer import CodecTokenizer
from espnet2.speechlm.tokenizer.ssl_tokenizer import SSLTokenizer
# FIXME(Jinchuan): Current SSL doesn't support "from_pretrained" method from HF.
# The initialization is a workaround.
from espnet2.tasks.ssl import SSLTask


class CodecSSLTokenizer(AbsTokenizer):
    """CodecSSL Tokenizer implementation, which combines the output of a codec model
       and an SSL+Kmeans model frame-by-frame
    """

    def __init__(
        self,
        codec_config: dict,
        ssl_config: dict,
        device: str,
        tolerance: int = 2,
    ):
        super(CodecSSLTokenizer, self).__init__()
        self.codec = CodecTokenizer(**codec_config, device=device)
        self.ssl = SSLTokenizer(**ssl_config, device=device)
        self.device = device

        # codec config
        self.n_codebook = self.codec.n_codebook
        self.size_codebook = self.codec.size_codebook
        self.sample_rate = self.codec.sample_rate
        self.subsample = self.codec.subsample

        # ssl config, hard code for XEUS for now.
        assert self.sample_rate == self.ssl.sample_rate
        assert self.subsample == self.ssl.subsample
        self.codec_bias = self.ssl.size_codebook
        self.tolerance = tolerance

    def forward(self, wavs):
        """ Convert input audio waveforms into K-Means centroid 
        
        Input:
            wavs (torch.Tensor): float tensor in shape [B, 1, n_sample]
        Output:
            codes (torch.Tensor): int tensor in shape [B, T, n_q]
        """

        codec_codes, resyn_audio = self.codec(wavs)
        ssl_codes = self.ssl(wavs)

        codec_codes = codec_codes.reshape(codec_codes.size(0), -1, self.n_codebook)
        codec_codes = codec_codes.cpu().numpy()
        ssl_codes = np.expand_dims(ssl_codes, axis=-1)

        codec_len, ssl_len = codec_codes.shape[1], ssl_codes.shape[1]
        if abs(codec_len - ssl_len) > self.tolerance:
            logging.warning(f"Codec-SSL length mismtch: {codec_len} vs. {ssl_len}")
        
        length = min(codec_len, ssl_len)
        codes = np.concatenate([
            ssl_codes[:, :length],
            codec_codes[:, :length] + self.codec_bias,
        ], axis=2)

        return codes
    
    def online_tokenize(self, value):
        """ Conduct online tokenization for one audio """
        if isinstance(value, str):
            if not os.path.exists(value):
                raise ValueError(f"Cannot find {value}, online tokenization fails")
            
            array, sr = librosa.load(value, mono=True, sr=16000)
            # TODO: check sr compatibility
        
        if isinstance(array, np.ndarray):
            assert array.ndim == 1, "Only 1D audio is supported"
            array = torch.from_numpy(array)
        
        elif isinstance(array, torch.Tensor):
            assert array.ndim() == 1, "Only 1D audio is supported"
        
        array = array.view(1, 1, -1).to(self.device)

        return self.forward(array)

    def detokenize(self, tokens):
        if tokens.ndim == 1:
            tokens = tokens.unsqueeze(0)
        
        B = tokens.size(0)
        assert tokens.size(1) % (self.n_codebook + 1) == 0
        tokens = tokens.view(B, -1, (self.n_codebook + 1))
        tokens = tokens[:, :, 1:]
        tokens = tokens.reshape(B, -1) - self.codec_bias

        return self.codec.detokenize(tokens)
