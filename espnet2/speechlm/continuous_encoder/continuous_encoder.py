#!/usr/bin/env python3

# Copyright 2025 Jinchuan Tian
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import torch
from abc import ABC, abstractmethod

class AbsContinuousEncoder(torch.nn.Module):
    """ Abstract class for continuous encoders """

    def __init__(self):
        super(AbsContinuousEncoder, self).__init__()
        self.model = None
        self.connector = None
        self.connector_choice = None
        self.connector_idim = None
    
    def register_projection(self, odim):
        if self.connector_choice == "linear":
            self.connector = torch.nn.Linear(self.connector_idim, odim)
        else:
            raise NotImplementedError(
                f"Connector choice {self.connector_choice} is not supported yet"
            )

    def forward(
        self,
        emb: torch.Tensor,
        conti_feats: list,
        modality: str,
    ):
        """ 
        Forward on the continuous encoder and 
          replace the hidden results into embedding.
        """

        if conti_feats is None:
            return emb

        # (1) aggregate all continuous features for the current continuous encoder
        feats, batch_idxs, time_idxs, durations = [], [], [], []
        for batch_idx, batch in enumerate(conti_feats):
            for feat, _modality, time_idx, duration in batch:
                if modality == _modality:
                    feats.append(feat)
                    batch_idxs.append(batch_idx)
                    time_idxs.append(time_idx)
                    durations.append(duration)
        
        if len(feats) == 0:
            return emb
        
        # (2) get the continuous features
        feats = self.forward_encoder(feats)

        # (3) apply connector
        feats = self.connector(feats)

        # (4) replace the hidden results into embedding
        for feat, batch_idx, time_idx, duration in zip(
            feats, batch_idxs, time_idxs, durations
        ):
            assert feat.dim() == 2
            emb[batch_idx, time_idx: time_idx + duration] = feat[:duration]

        return emb

    @abstractmethod
    def forward_encoder(self, feat_list: list):
        """ Forward the continuous encoder with the given feature list."""
        raise NotImplementedError


class HuggingfaceVisionEncoder(AbsContinuousEncoder):
    """A warpper for HuggingFace Vision Encoder """

    def __init__(
        self, 
        hf_tag,
        connector_choice: str = "linear",
    ):
        super(HuggingfaceVisionEncoder, self).__init__()

        if hf_tag.startswith("google/siglip"):
            try:
                from transformers import SiglipVisionModel
            except:
                raise ImportError("Please install transformers")
            
            self.model = SiglipVisionModel.from_pretrained(hf_tag)
            self.connector_idim = self.model.config.hidden_size
        else:
            raise NotImplementedError(f"HF Tag {hf_tag} is not supported yet")
        
        self.connector_choice = connector_choice
        
    def forward_encoder(self, feat_list: list):
        feats = torch.stack(feat_list, dim=0)
        feats = self.model(feats).last_hidden_state
        return feats

class HFQwen2AudioEncoder(AbsContinuousEncoder):
    """ A warpper for Qwen2Audio Encoder """
    def __init__(
        self,
        hf_tag,
        checkpoint,
        connector_choice,
    ):
        super(HFQwen2AudioEncoder, self).__init__()
        
        assert hf_tag == "Qwen/Qwen2-Audio-7B", "hf_tag should only be Qwen/Qwen2-Audio-7B"

        try:
            from transformers import Qwen2AudioEncoder, AutoFeatureExtractor
        except ImportError:
            raise ImportError(f"Cannot import Qwen2AudioEncoder object")
        
        self.model = Qwen2AudioEncoder.from_pretrained(checkpoint)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(hf_tag)
        self.n_samples = self.feature_extractor.n_samples

        self.connector_choice = connector_choice
        self.connector_idim = 1280 # empirical, hard code
    
    def forward_encoder(self, feat_list: list):
        # NOTE(Jinchuan): Assume all feats are of the same shape, aka, 30s
        feats = torch.stack(feat_list, dim=0)
        feats = self.model(input_features=feats)['last_hidden_state']
        
        return feats

