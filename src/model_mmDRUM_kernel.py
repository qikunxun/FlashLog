import sys
import os
import torch
import numpy as np
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from flashlog_kernel.flashlog_kernel import (
    sampled_sparse_propagation,
    build_csr_structure,
    prepare_csr_masks,
    dense_csr_max,
)
from utils import log_loss


class Model(nn.Module):
    def __init__(self, n, T, L, E, N, emb_size, tau_1=1, tau_2=1, use_gpu=False,
                 dropout=0.1, c1=1000, use_soft=False, use_topk=False, c2=400):
        super(Model, self).__init__()
        self.T = T
        self.L = L
        self.E = E
        self.n = n
        self.r_size = (self.n - 1) // 2
        self.N = N
        self.use_soft = use_soft
        self.use_topk = use_topk
        if c2 < 0 or c2 == 1:
            raise ValueError("c2 must be 0 (unlimited) or at least 2")
        self.c2 = c2
        self.emb_size = emb_size
        self.tau_1 = tau_1
        self.tau_2 = tau_2
        self.c1 = c1
        self.emb = nn.Parameter(torch.Tensor(self.n, self.emb_size))
        nn.init.kaiming_uniform_(self.emb, a=np.sqrt(5))
        self.lstm = nn.ModuleList()
        for _ in range(self.L):
            self.lstm.append(nn.LSTM(self.emb_size, self.emb_size, 1, bidirectional=True))
        self.linear = nn.Linear(2 * self.emb_size, self.n)

        if self.use_soft:
            self.weight = nn.Parameter(torch.Tensor(self.N, 1))
            nn.init.zeros_(self.weight)
        else:
            self.weight = None

        self.use_gpu = use_gpu
        self.dropout = nn.Dropout(dropout)
        self.one = torch.autograd.Variable(torch.Tensor([1])).detach()
        if use_gpu:
            self.one = self.one.cuda()

        self._csr_struct = None
        self._csr_struct_key = None
        self._mask_values = None

    def _ensure_csr_struct(self, e2triple, triple2e, r2triple):
        row_indices = e2triple[0]
        col_indices = triple2e[1]
        r_indices = r2triple[0]
        mask_values = e2triple[2]

        struct_key = (row_indices.data_ptr(), col_indices.data_ptr())
        if self._csr_struct is None or self._csr_struct_key != struct_key:
            self._csr_struct = build_csr_structure(
                row_indices, col_indices, r_indices, self.E
            )
            self._csr_struct_key = struct_key

        self._mask_values = mask_values
        return self._csr_struct, mask_values

    def forward(self, input_x, input_r, e2triple, triple2e, r2triple,
                is_training=False, input_y=None, csr_struct=None, mask_values=None):
        x_ori_i = torch.nn.functional.one_hot(input_x.long(), self.E).unsqueeze(1)
        x_ori_i = x_ori_i.repeat(1, self.L, 1)

        input_emb_ori = torch.index_select(self.emb, index=input_r, dim=0)
        input_emb = torch.stack([input_emb_ori] * (self.T + 1), dim=1)
        input_emb[:, -1, :] = self.emb[-1]
        input_emb = input_emb.transpose(1, 0)

        w_all = []
        for l in range(self.L):
            rnn_outputs, _ = self.lstm[l](input_emb)
            rnn_outputs = rnn_outputs.transpose(1, 0)
            outputs = self.linear(rnn_outputs[:, :-1, :])
            w_all.append(outputs)
        w_all = torch.stack(w_all, dim=2)

        if csr_struct is not None and mask_values is not None:
            _csr_struct = csr_struct
            _mask_values = mask_values.to(input_x.device)
        else:
            _csr_struct, _mask_values = self._ensure_csr_struct(e2triple, triple2e, r2triple)
        _prepared_masks = (None if is_training and self.use_topk else
                           prepare_csr_masks(_csr_struct, _mask_values, self.weight))

        states = [x_ori_i]
        for t in range(self.T):
            w_probs = w_all[:, t, :, :]
            w = torch.softmax(w_probs / self.tau_1, dim=-1)
            x = states[-1]

            if is_training and self.use_topk:
                s = sampled_sparse_propagation(x.float().to_sparse().coalesce(), w,
                    _csr_struct, _mask_values, self.r_size, self.weight, False,
                    self.c1, self.c2 // 2, "max").to_dense()
            else:
                s_i, s_h, s_t = dense_csr_max(
                    x.float(), w, _csr_struct, _mask_values,
                    self.r_size, weight=self.weight,
                    use_topk=self.use_topk, topk_nodes=self.c1,
                    topk_edges=self.c2 // 2,
                    prepared_masks=_prepared_masks
                )
                s = torch.max(torch.stack([s_i, s_h, s_t], dim=0), dim=0).values
            s_sum = s.sum(dim=-1, keepdims=True)
            s = s / s_sum.clamp(1e-7)
            if is_training:
                s = self.dropout(s)
            states.append(s)

        return states[-1].max(dim=1)[0]

    def log_loss(self, p_score, label):
        return log_loss(p_score, label, self.E, self.tau_2, self.one)
