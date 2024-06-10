#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging

import k2
import torch


class StarCTC(torch.nn.Module):
    def __init__(
        self, 
        vocab_size: int,
        star_id: int,
        penalty: float = 0.0,
        standard_ctc: bool = False,
        flexible_start_end = True,
    ):
        super().__init__()

        self.v = vocab_size
        self.star_id = star_id
        self.p = penalty
        self.standard_ctc = standard_ctc
        self.flexible_start_end = flexible_start_end

    def forward(self, nnet_output, ys_pad, hlens, ylens):
        # Reorder and filter out invalid examples:
        # A. K2 requires that hlens are in descending order;
        # B. remove all examples whose hlens is smaller than necessary.
        indices = torch.argsort(hlens, descending=True)
        ys, min_hlens = self.find_minimum_hlens(ys_pad[indices], ylens[indices])
        valid_sample_indices = (min_hlens <= hlens[indices]).nonzero(as_tuple=True)[0]

        if len(valid_sample_indices) < 1:
            logging.warning(
                "All examples are invalid for Bayes Risk CTC. Skip this batch"
            )
            return torch.Tensor([0.0]).to(nnet_output.device)

        indices = indices[valid_sample_indices]
        nnet_output, hlens, ylens = nnet_output[indices], hlens[indices], ylens[indices]
        ys = [ys[i.item()] for i in valid_sample_indices]

        assert nnet_output.size(2) == self.v, f"Invalid input size: {nnet_output.size()}"
        nnet_output = self.organize_nnet_output(nnet_output)

        # Core implementation
        loss_utt = self.forward_core(nnet_output, ys, hlens, ylens)

        # Recover the original order. Invalid examples are excluded.
        indices2 = torch.argsort(indices)
        loss_utt = loss_utt[indices2]

        return loss_utt

    def forward_core(self, nnet_output, ys, hlens, ylens):
        # (1) Find the shape
        (B, T, _), U = nnet_output.size(), max(ylens)

        # (2) Build DenseFsaVec and CTC graphs
        supervision = torch.stack(
            [torch.arange(B), torch.zeros(B), hlens.cpu()], dim=1
        ).int()

        dense_fsa_vec = k2.DenseFsaVec(nnet_output, supervision)
        ctc_graphs = self.graphs(ys).to(nnet_output.device)

        # (3) Intersection and get the loss
        lats = k2.intersect_dense(ctc_graphs, dense_fsa_vec, 1e20)
        loss_fsas = lats.get_tot_scores(True, True)

        return - loss_fsas
    
    def find_minimum_hlens(self, ys_pad, ylens):
        device = ys_pad.device
        ys_pad, ylens = ys_pad.cpu().tolist(), ylens.cpu().tolist()
        ys, min_hlens = [], []

        for y_pad, ylen in zip(ys_pad, ylens):
            y, min_hlen = [], 0
            prev = None

            for i in range(ylen):
                y.append(y_pad[i])
                min_hlen += 1

                if y_pad[i] == prev:
                    min_hlen += 1

                prev = y_pad[i]

            ys.append(y)
            min_hlens.append(min_hlen)

        min_hlens = torch.Tensor(min_hlens).long().to(device)

        return ys, min_hlens
    
    def graphs(self, labels):
        fsas = [self.graph(label_seq) for label_seq in labels]
        fsas = k2.create_fsa_vec(fsas)
        fsas = k2.arc_sort(fsas)
        return fsas
    
    def graph(self, labels):
        # (1) organize the star token.
        if self.standard_ctc:
            assert self.star_id not in labels, "Star is not allowed in vanilla CTC"
        else:
            if self.flexible_start_end:
                if labels[0] != self.star_id:
                    labels = [self.star_id] + labels
                if labels[-1] != self.star_id:
                    labels = labels + [self.star_id]
        labels = labels + [-1] # k2 always ends with an arc of -1

        new_labels = []
        with_star = False
        for label in labels:
            if label == self.star_id:
                with_star = True
            else:
                new_labels.append((label, with_star))
                with_star = False
        labels = new_labels

        idx = 0
        string = ''
        p = self.p if self.training else -10000 # valid loss should be standard CTC

        # (1) starting block
        label, with_star = labels[0]
        string += f'{idx} {idx} 0 0\n'
        if with_star:
            string += f'{idx} {idx} {self.v} {p}\n'
        string += f'{idx} {idx + 1} {label} 0\n'
        string += f'{idx + 1} {idx + 1} {label} 0\n'
        idx += 1
        prev_label = label

        # (2) all the remained blocks
        for count, (label, with_star) in enumerate(labels[1:]):

            # (2.1) Go directly without blank
            if label != prev_label:
                string += f'{idx} {idx + 2} {label} 0\n'
            
            # (2.2) Go with blank inserted
            string += f'{idx} {idx + 1} 0 0\n'
            if with_star:
                string += f'{idx} {idx + 1} {prev_label + self.v} {p}\n'
            string += f'{idx + 1} {idx + 1} 0 0\n'
            if with_star:
                string += f'{idx + 1} {idx + 1} {self.v} {p}\n'
            string += f'{idx + 1} {idx + 2} {label} 0\n'

            # (2.3) self-loop. Skip the endding node.
            if count < len(labels) - 2:
                string += f'{idx + 2} {idx + 2} {label} 0\n'
            prev_label = label
            idx += 2
        
        # (3) final state node
        string += f'{idx}'
        graph = k2.Fsa.from_str(string, acceptor=True)
        return graph
    
    def organize_nnet_output(self, nnet_output):
        def log_sub_exp(a, b):
            max_ab = torch.max(a, b)
            return  max_ab + torch.log(torch.exp(a - max_ab) - torch.exp(b - max_ab))
        
        non_blank = torch.logsumexp(nnet_output[..., 1:], dim=-1, keepdim=True)
        non_blank_each = log_sub_exp(non_blank, nnet_output[..., 1:])

        return torch.cat([nnet_output, non_blank, non_blank_each], dim=-1)

if __name__ == "__main__":
    hidden_output = torch.rand(1, 7, 5).log_softmax(dim=-1)
    ys_pad = torch.Tensor([[3, 4, 2, 3, 1]]).long()
    ylen = torch.Tensor([5]).long()
    hlen = torch.Tensor([7]).long()

    ctc = StarCTC(
        vocab_size=5,
        star_id=2,
        penalty=-0.4, 
        standard_ctc=False,
        flexible_start_end=False,
    )
    ctc_k2 = ctc(hidden_output, ys_pad, hlen, ylen)
    print('k2 loss: ', ctc_k2)

    builtin_loss = torch.nn.functional.ctc_loss(
        log_probs=hidden_output.transpose(0, 1),
        targets=ys_pad,
        input_lengths=hlen,
        target_lengths=ylen,
        reduction="none"
    )
    print(builtin_loss, 'builtin loss')