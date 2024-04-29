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
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.star_id = star_id
        self.penalty = penalty

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
        # ctc_graphs = k2.ctc_graph(ys).to(nnet_output.device)
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
        print('build graph for: ', labels)
        labels = labels + [-1]

        idx = 0
        string = ''

        # (1) starting block
        string += f'{idx} {idx} 0 0\n'
        string += f'{idx} {idx + 1} {labels[0]} 0\n'
        string += f'{idx + 1} {idx + 1} {labels[0]} 0\n'
        idx += 1
        prev_label = labels[0]

        # (2) all the remained blocks
        for count, label in enumerate(labels[1:]):

            # (2.1) Go directly without blank
            if label != prev_label:
                string += f'{idx} {idx + 2} {label} 0\n'
            
            # (2.2) Go with blank inserted
            string += f'{idx} {idx + 1} 0 0\n'
            string += f'{idx + 1} {idx + 1} 0 0\n'
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