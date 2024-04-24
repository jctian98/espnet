import torch
import k2
import logging

class StarCTC(torch.nn.Module):
    def __init__(
        self, 
        vocab_size: int,
        star_id: int,
        star_penalty: float,
        ctc_type: str = 'stc',
    ):
        super(StarCTC, self).__init__()

        assert star_penalty <= 0, f'Star penalty should be negative: {star_penalty}'
        assert star_id < vocab_size, f'Star ID should be in vocabulary: {star_id}'
        assert ctc_type in ['ctc', 'stc'], f"unsupported CTC type {ctc_type}"

        self.v = vocab_size
        self.star_id = star_id
        self.p = star_penalty
        self.ctc_type = ctc_type

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
        nnet_output, hlens = nnet_output[indices], hlens[indices]
        ys = [ys[i.item()] for i in valid_sample_indices]
        nnet_output = self.organize_nnet_output(nnet_output)

        # Core implementation
        loss_utt = self.forward_core(nnet_output, ys, hlens)

        # Recover the original order. Invalid examples are excluded.
        indices2 = torch.argsort(indices)
        loss_utt = loss_utt[indices2]

        return loss_utt
    
    def forward_core(self, nnet_output, ys, hlens):
        B = nnet_output.size(0)

        # (1) Build DenseFsaVec and CTC graphs
        ctc_graphs = self.graphs(ys).to(nnet_output.device)
        supervision = torch.stack(
            [torch.arange(B), torch.zeros(B), hlens.cpu()], dim=1
        ).int()
        dense_fsa_vec = k2.DenseFsaVec(nnet_output, supervision)
        
        # (2) Compose lattice and get the loss
        lats = k2.intersect_dense(ctc_graphs, dense_fsa_vec, 1e20)
        scores = lats._get_tot_scores(True, True)

        return - scores
    
    def graphs(self, labels):
        fsas = [self.graph(label_seq) for label_seq in labels]
        fsas = k2.create_fsa_vec(fsas)
        return fsas
    
    def graph(self, labels):
        # (1) reorganize the labels
        if self.ctc_type == "ctc":
            assert self.star_id not in labels, "Standard CTC doesn't need star tokens"
            labels = labels + [-1]
        else:
            # (1.1) always allow star string in the two ends
            assert labels[0] != self.star_id and labels[-1] != self.star_id
            labels = [self.star_id] + labels + [self.star_id, -1]

        # (1.2) remove star and combine with normal labels
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

        # (2) starting block
        string += f'{idx} {idx} 0 0\n'
        if self.ctc_type == 'stc':
            string += f'{idx} {idx} {self.v} {self.p}\n'
        string += f'{idx} {idx + 1} {labels[0][0]} 0\n'
        string += f'{idx + 1} {idx + 1} {labels[0][0]} 0\n'

        idx += 1
        prev_label = labels[0][0]

        # (3) all the remained blocks
        for count, (label, with_star) in enumerate(labels[1:]):

            # (3.1) Go directly without blank
            if label != prev_label:
                string += f'{idx} {idx + 2} {label} 0\n'
            
            # (3.2) Go with blank inserted
            string += f'{idx} {idx + 1} 0 0\n'
            if with_star and self.ctc_type == 'stc':
                string += f'{idx} {idx + 1} {prev_label + self.v} {self.p}\n'
            string += f'{idx + 1} {idx + 1} 0 0\n'
            if with_star and self.ctc_type == 'stc':
                string += f'{idx + 1} {idx + 1} {self.v} {self.p}\n'
            string += f'{idx + 1} {idx + 2} {label} 0\n'

            # (3.3) self-loop. Skip the endding node.
            if count < len(labels) - 2:
                string += f'{idx + 2} {idx + 2} {label} 0\n'
            prev_label = label
            idx += 2
        
        # (4) final state node
        string += f'{idx}'
        graph = k2.Fsa.from_str(string, acceptor=True)
        return graph
    
    def find_minimum_hlens(self, ys_pad, ylens):
        device = ys_pad.device
        ys_pad, ylens = ys_pad.cpu().tolist(), ylens.cpu().tolist()
        ys, min_hlens = [], []

        for y_pad, ylen in zip(ys_pad, ylens):
            y, min_hlen = [], 0
            prev = None

            for i in range(ylen):
                y.append(y_pad[i])

                if y_pad[i] == self.star_id:
                    continue

                min_hlen += 1

                if y_pad[i] == prev:
                    min_hlen += 1

                prev = y_pad[i]
            
            # start and end cannot be star_id.
            if y[0] == self.star_id:
                y = y[1:]
            if y[-1] == self.star_id:
                y = y[:-1]

            ys.append(y)
            min_hlens.append(min_hlen)

        min_hlens = torch.Tensor(min_hlens).long().to(device)

        return ys, min_hlens
    
    def organize_nnet_output(self, nnet_output):
        def log_sub_exp(a, b):
            max_ab = torch.max(a, b)
            return  max_ab + torch.log(torch.exp(a - max_ab) - torch.exp(b - max_ab))
        
        non_blank = torch.logsumexp(nnet_output[..., 1:], dim=-1, keepdim=True)
        non_blank_each = log_sub_exp(non_blank, nnet_output[..., 1:])

        return torch.cat([nnet_output, non_blank, non_blank_each], dim=-1)

if __name__ == "__main__":
    nnet_output = torch.rand(1, 5, 5).log_softmax(dim=-1)
    ys_pad = torch.Tensor([[4, 2, 4, 1]]).long()
    ylen = torch.Tensor([4]).long()
    hlen = torch.Tensor([5]).long()

    ctc = StarCTC(vocab_size=5, star_id=4, star_penalty=-0.4)
    _ = ctc(nnet_output, ys_pad, hlen, ylen)

    
