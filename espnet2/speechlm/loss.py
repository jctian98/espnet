import torch
import logging
from espnet2.speechlm.net_utils import length_mask
import torch.distributed

class SpeechLMCrossEntropyLoss(torch.nn.Module):
    def __init__(
        self,
        pad,
        vocab_size,
        token_bias,
        modality_weights,
        z_loss_weight: float = 0.0,
        lm_head: torch.nn.Linear = None,
        aux_lm_head: torch.nn.Linear = None,
    ):
        """
        Compute the CrossEntropy for SpeechLM. The main motivation of this module is to save computing.
        From the second layer and then on, the target codes can only be the codec codes or paddings, 
        which helps us to shrink the vocabulary during the loss computing. 
        """
        super().__init__()

        self.pad = pad
        self.use_aux_ce_loss = aux_lm_head is not None # temp modification for convinience.
        self.z_loss_weight = z_loss_weight

        if z_loss_weight > 0:
            raise ValueError('z_loss is not implemented yet')

        # (1) parse the weights of tokens
        token_bias = token_bias.copy()
        if self.use_aux_ce_loss:
            self.aux_start, self.aux_end = token_bias["codec"]
        else:
            self.aux_start, self.aux_end = 0, 0

        # prepare the weight for first-layer
        weight = torch.ones(vocab_size).float()
        for modality_name, modality_weight in modality_weights.items():

            if modality_name not in token_bias:
                logging.warning(f"The specified modality {modality_name} is not in token_bias. Skip it")
                continue
            
            start, end = token_bias[modality_name]
            del token_bias[modality_name]
            weight[start: end] = modality_weight

        for modality in token_bias.keys():
            logging.warning(f"weight for modality {modality} not specified. Set to 1.0")
        self.weight = weight
        
        # (2) create CE loss module
        assert lm_head is not None
        self.lm_head = lm_head

        if self.use_aux_ce_loss:
            assert aux_lm_head is not None
            self.aux_lm_head = aux_lm_head

        ce_loss_imp = torch.nn.CrossEntropyLoss
        self.ce_loss = ce_loss_imp(
            ignore_index=self.pad,
            reduction='none'
        )
        if self.use_aux_ce_loss:
            self.aux_ce_loss = ce_loss_imp(
                ignore_index=self.pad - self.aux_start,
                reduction='none'
            )
    
    def forward(
        self, 
        logits: torch.Tensor,
        targets: torch.Tensor, 
        loss_mask: torch.Tensor,
    ):
        # NOTE(Jinchuan): keep the weight on the correct device in the first forward.
        # We don't want to keep the weights registered as model parameters as they 
        # should always be specified by external configurations.
        device, dtype = logits[0].device, logits[0].dtype
        self.weight = self.weight.to(device).to(dtype)

        # sanity check
        logits, aux_logits = logits
        assert logits.dim() == 4 and logits.size(2) == 1
        B, T, _, _ = logits.size()
        
        if aux_logits is not None:
            assert logits.dim() == 4
            assert logits.size()[:2] == aux_logits.size()[:2]
            assert logits.size(2) + aux_logits.size(2) == targets.size(2)
        
        # element-wise loss
        _targets = targets[:, :, :1].flatten()
        ce_loss = self.apply_ce_loss(
            logits.flatten(end_dim=2),
            _targets,
            self.lm_head,
            self.ce_loss,
            chunk_size=16384 * 100,
        )
        ce_loss = ce_loss * self.weight[_targets]
        ce_loss = ce_loss.view(B, T, 1)

        if aux_logits is not None:
            _targets = targets[:, :, 1:].flatten()
            assert torch.all(torch.logical_or(
                _targets == self.pad,
                torch.logical_and(
                    _targets >= self.aux_start, 
                    _targets < self.aux_end,
                )
            ))
            
            aux_ce_loss = self.apply_ce_loss(
                aux_logits.flatten(end_dim=2),
                _targets - self.aux_start,
                self.aux_lm_head,
                self.aux_ce_loss,
                chunk_size=32768 * 100 ,
            )

            aux_ce_loss = aux_ce_loss * self.weight[_targets]
            aux_ce_loss = aux_ce_loss.view(B, T, -1)

            ce_loss = torch.cat([ce_loss, aux_ce_loss], dim=2)
        
        ce_loss = ce_loss * loss_mask
        weight = loss_mask[..., 0].eq(1).float().sum()

        ce_loss = ce_loss.sum() / weight
        stats = {"ce_loss": ce_loss.clone().detach(), "weight": weight.clone().detach()}

        # logging, if not training
        if not self.training:
            logits = self.lm_head(logits)
            acc = logits.argmax(-1) == targets[:, :, :1]
            if aux_logits is not None:
                aux_logits = self.aux_lm_head(aux_logits)
                aux_acc = aux_logits.argmax(-1) == targets[:, :, 1:] - self.aux_start
                acc = torch.cat([acc, aux_acc], dim=2)
            
            acc = torch.where(loss_mask.bool(), acc, False)

            acc_all = acc.float().sum() / loss_mask.float().sum()
            stats["acc_all"] = acc_all.clone().detach()
            
            for idx in range(targets.size(2)):
                layer_weight = loss_mask[:, :, idx].float().sum()
                if layer_weight == 0:
                    stats[f"acc_layer{idx}"] = 0.0
                else:
                    layer_acc = acc[:, :, idx:idx+1].float().sum() 
                    layer_acc = layer_acc / loss_mask[:, :, idx:idx + 1].float().sum()
                    stats[f"acc_layer{idx}"] = layer_acc.clone().detach()
        
        return ce_loss, stats, weight
    
    def apply_ce_loss(self, 
        input, 
        target,
        linear_module,
        loss_module,
        chunk_size=10000
    ):
        # Apply CE loss chunk-by-chunk to avoid memory spike

        assert input.dim() == 2
        assert target.dim() == 1

        start = 0
        ce_loss = []
        while start < input.size(0):
            end = min(input.size(0), start + chunk_size)
            logits = linear_module(input[start: end])
            piece_ce_loss = loss_module(
                logits,
                target[start: end],
            )
            if self.z_loss_weight > 0:
                z_loss = logits.logsumexp(-1).pow(2)
                z_loss = z_loss * (target[start: end] != loss_module.ignore_index).float()
                piece_ce_loss = piece_ce_loss + self.z_loss_weight * z_loss
            ce_loss.append(piece_ce_loss)
            start += chunk_size

        return torch.cat(ce_loss)

class SpeechLMCrossEntropyLossV2(torch.nn.Module):
    def __init__(
        self,
        pad,
        token_bias,
        modality_weights,
        image_interval_split,
        lm_head: torch.nn.Linear = None,
    ):
        super().__init__()

        self.pad = pad
        self.lm_head = lm_head
        self.modality_weights = modality_weights

        # (1) loss weight
        vocab_size = lm_head.weight.size(0)
        self.weight = torch.ones(vocab_size).float()
        for name, m_weight in modality_weights.items():
            if name not in token_bias:
                logging.warning(f"Modality {name} not in token_bias. Skip it")
                continue
            
            start, end = token_bias[name]
            self.weight[start: end] = m_weight
        
        # (2) aux loss interval
        self.aux_loss_interval = []
        # Codec codebook is small, put them in one interval for efficiency
        if 'codec' in token_bias:
            self.aux_loss_interval.append(token_bias['codec'])
        # Image codebook is large, compute them separately
        if 'image' in token_bias:
            start, end = token_bias['image']
            assert (end - start) % image_interval_split == 0
            inc = (end - start) // image_interval_split
            i = 0
            while start + i * inc < end:
                self.aux_loss_interval.append((
                    start + i * inc,
                    start + (i+1) * inc,
                ))
                i += 1
        
    def forward(self, hidden, targets, loss_mask):

        tmp = [
            hidden.cpu().numpy(),
            targets.cpu().numpy(),
            loss_mask.cpu().numpy()
        ]
        import pickle
        pickle.dump(tmp, 'tmp.pkl')
        assert 1 == 2

        # NOTE(Jinchuan): keep the weight on the correct device in the first forward.
        # We don't want to keep the weights registered as model parameters as they 
        # should always be specified by external configurations.
        device, dtype = hidden[0].device, hidden[0].dtype
        self.weight = self.weight.to(device).to(dtype)

        # (1) check shape
        assert hidden.dim() == 4 # [B, T, nq, D]
        assert targets.dim() == 3 # [B, T, nq]
        assert loss_mask.dim() == 3 # [B, T, nq]
        assert hidden.size()[:3] == targets.size()
        assert hidden.size()[:3] == loss_mask.size()

        # (2) apply loss mask to targets
        targets = torch.where(loss_mask, targets, self.pad)
        
        elem_loss = torch.zeros_like(targets).to(dtype)
        # default prediction is never equal to a target
        pred = torch.ones_like(targets) * -1 if not self.training else None 

        # (3) first stream
        this_loss, this_pred, this_mask = self.forward_interval(
            hidden[:, :, 0], targets[:, :, 0]
        )
        elem_loss[:, :, 0][this_mask] = this_loss
        if this_pred is not None:
            pred[:, :, 0][this_mask] = this_pred
        
        # (4) all remained stream.
        # TODO: Check this is safe for single-stream LM.
        for interval in self.aux_loss_interval:
            this_loss, this_pred, this_mask = self.forward(
                hidden[:, :, 1:],
                targets[:, :, 1:],
                interval=interval,
            )
            elem_loss[:, :, 1:][this_mask] = this_loss
            if this_pred is not None:
                pred[:, :, 1:][this_mask] = this_pred

        # (5) summarize
        frame_count = loss_mask[:, :, 0].float().sum()
        loss = loss.sum() / frame_count
        stats = {
            "ce_loss": loss.clone().detach(),
            "weight": frame_count.clone().detach(),
        }

        if pred is not None:
            tok_count = loss_mask.float().sum()
            acc = pred.eq(targets).float().sum() / tok_count
            stats['acc_all'] = acc.clone().detach()

            for n in range(targets.size(2)):
                token_count = loss_mask[:, :, n].float().sum()
                if tok_count > 0:
                    acc = pred[:, :, n].eq(targets[:, :, n]).float().sum() / tok_count
                else:
                    acc = 0
                stats[f'acc_layer{n}'] = acc.clone().detach()

        
        return loss, stats, frame_count
    
    def forward_interval(self, hidden, targets, interval=None):
        shape = target.size()
        hidden = hidden.flatten(end_dim=-2)
        targets = targets.flatten()
        
        # mask and mask select
        if interval is None:
            mask = targets != self.pad
            linear_weight = self.lm_head.weight
            weight = self.weight
            start = 0
        else:
            start, end = interval
            mask = torch.logical_and(
                targets >= start,
                targets < end,
            )
            linear_weight = self.lm_head.weight[start: end]
            weight = self.weight[start: end]
        hidden, targets = hidden[mask], targets[mask]
        
        # loss computing
        logits = torch.matmul(hidden, linear_weight)
        loss = torch.nn.functional.cross_entropy(
            logits,
            targets - start,
            weight=weight,
            reduction='none'
        )

        if not self.training:
            pred = logits.argmax(dim=-1)
        else:
            pred = None
        
        assert 1 == 2
        return loss, pred, mask.view(shape)
