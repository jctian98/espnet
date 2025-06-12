#!/usr/bin/env python3

# Copyright 2025 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import os
import joblib
import torch
import numpy as np
from pathlib import Path

from espnet2.speechlm.tokenizer.abs_tokenizer import AbsTokenizer
from espnet2.tasks.ssl import SSLTask


class SSLTokenizer(AbsTokenizer):
    """ A tokenizer based on self-supervised learning (SSL) models plus K-Means """

    def __init__(
        self,
        hf_model_tag: str,
        device: str = 'cpu',
    ):
        super(SSLTokenizer, self).__init__()
        self.device = device

        # FIXME(Jinchuan): Current SSL doesn't support "from_pretrained" method from HF.
        # The initialization is a workaround.
        if os.path.exists(hf_model_tag):
            ssl_path = Path(hf_model_tag)
        
        elif hf_model_tag.startswith("espnet/"):
            try:
                from huggingface_hub import snapshot_download
            except ImportError:
                raise ImportError("Make sure you installed huggingface-hub")
            
            ssl_cache = snapshot_download(hf_model_tag)
            # Currently, only XEUS model is supported.
            # https://huggingface.co/espnet/xeus
            if hf_model_tag == "espnet/xeus":
                ssl_path = Path(ssl_cache) / "model.pth"
            else:
                raise ValueError("Unrecognized ESPnet SSL model")
        else:
            raise ValueError(f"Cannot initialize SSL model with {hf_model_tag}")
        
        self.ssl, self.ssl_args = SSLTask.build_model_from_file(
            None,
            ssl_path,
            device
        )

        self.km = ApplyKmeans(
            ssl_path.parent / 'kmeans' / 'km.mdl',
            use_gpu=device != 'cpu',
        )

        # hard code, cannot find it in the config ssl_args
        self.sample_rate = 16000
        self.subsample = 320
        self.size_codebook = self.km.C.size(1)

    def forward(self, wavs):
        """ Convert input audio waveforms into K-Means centroid 
        
        Input:
            wavs (torch.Tensor): float tensor in shape [B, 1, n_sample]
        Output:
            codes (torch.Tensor): int tensor in shape [B, T]
        """
        # Currently, not to support multi-channel
        assert wavs.size(1) == 1
        wavs = wavs.squeeze(1)

        lengths = torch.ones(wavs.size(0)) * wavs.size(1)
        lengths = lengths.to(self.device).int()
        feats = self.ssl.encode(wavs, lengths)['encoder_output'][-1]

        B, T, _ = feats.size()
        codes = self.km(feats.view(B * T, -1))

        return codes.reshape(B, T)
    
    def detokenization(self, tokens):
        """ Don't have an SSL based vocoder """
        raise NotImplementedError

# NOTE(Jinchuan): copied from the original ASR2 recipe
# <espnet>/egs2/TEMPLATE/asr1/pyscripts/feats/dump_km_label.py
class ApplyKmeans(object):
    def __init__(self, km_path, use_gpu):
        self.km_model = joblib.load(km_path)
        self.C_np = self.km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np**2).sum(0, keepdims=True)

        self.C = torch.from_numpy(self.C_np)
        self.Cnorm = torch.from_numpy(self.Cnorm_np)
        if use_gpu and torch.cuda.is_available():
            self.C = self.C.cuda()
            self.Cnorm = self.Cnorm.cuda()

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            x = x.to(self.C.device)
            dist = (
                x.pow(2).sum(1, keepdim=True) - 2 * torch.matmul(x, self.C) + self.Cnorm
            )
            return dist.argmin(dim=1).cpu().numpy()
        else:
            dist = (
                (x**2).sum(1, keepdims=True)
                - 2 * np.matmul(x, self.C_np)
                + self.Cnorm_np
            )
            return np.argmin(dist, axis=1)

        

        
