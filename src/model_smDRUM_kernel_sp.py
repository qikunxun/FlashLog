"""smDRUM: MAX propagation; normalized channels combined by SUM."""
import sys
import os
import torch
import numpy as np
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from flashlog_kernel.flashlog_kernel import (
    FlashLogFunctionSparse3DTopKMax,
    build_csr_structure,
    normalize_sparse3d,
)
from utils import log_loss_common_sp


def sum_sp_3d(s):
    s = s.coalesce()
    idx = s.indices()
    val = s.values()
    B, _, E = s.shape
    idx2 = torch.stack([idx[0], idx[2]], dim=0)
    return torch.sparse_coo_tensor(idx2, val, [B, E]).coalesce()


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
        self._csr_struct = None
        self._csr_struct_key = None

    def _ensure_csr_struct(self, e2triple, triple2e, r2triple):
        row_indices = e2triple[0]
        col_indices = triple2e[1]
        r_indices = r2triple[0]
        struct_key = (row_indices.data_ptr(), col_indices.data_ptr())
        if self._csr_struct is None or self._csr_struct_key != struct_key:
            self._csr_struct = build_csr_structure(
                row_indices, col_indices, r_indices, self.E
            )
            self._csr_struct_key = struct_key
        return self._csr_struct

    def forward(self, input_x, input_r, e2triple, triple2e, r2triple,
                is_training=False, input_y=None, csr_struct=None, mask_values=None):
        B = input_x.shape[0]
        L = self.L
        E = self.E
        device = input_x.device

        input_emb_ori = torch.index_select(self.emb, index=input_r, dim=0)
        input_emb = torch.stack([input_emb_ori] * (self.T + 1), dim=1)
        input_emb[:, -1, :] = self.emb[-1]
        input_emb = input_emb.transpose(1, 0)
        w_all = []
        for l in range(L):
            rnn_outputs, _ = self.lstm[l](input_emb)
            rnn_outputs = rnn_outputs.transpose(1, 0)
            outputs = self.linear(rnn_outputs[:, :-1, :])
            w_all.append(outputs)
        w_all = torch.stack(w_all, dim=2)

        if csr_struct is not None and mask_values is not None:
            _csr_struct = csr_struct
            _mask_values = mask_values.to(device)
        else:
            _csr_struct = self._ensure_csr_struct(e2triple, triple2e, r2triple)
            _mask_values = e2triple[2].to(device)

        sp_batch = torch.arange(B, device=device, dtype=torch.long).repeat_interleave(L)
        sp_level = torch.arange(L, device=device, dtype=torch.long).repeat(B)
        sp_entity = input_x.long().repeat_interleave(L)
        sp_value = torch.ones(B * L, device=device, dtype=torch.float)
        state = torch.sparse_coo_tensor(
            torch.stack([sp_batch, sp_level, sp_entity], dim=0), sp_value, [B, L, E]
        ).coalesce()
        states = [state]

        (ori_row_ptr, ori_col_ind, ori_r_ind, order_ori,
         inv_row_ptr, inv_col_ind, inv_r_ind, order_inv,
         ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
         inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count) = _csr_struct

        for t in range(self.T):
            w_probs = w_all[:, t, :, :]
            w_t = torch.softmax(w_probs / self.tau_1, dim=-1)
            topk_edges = self.c2 // 2 if (self.use_topk and self.c2 > 0) else (1 << 60)
            curr_state = states[-1] if states[-1].is_coalesced() else states[-1].coalesce()
            s = FlashLogFunctionSparse3DTopKMax.apply(
                curr_state.indices()[0], curr_state.indices()[1], curr_state.indices()[2], curr_state.values(), w_t,
                ori_row_ptr, ori_col_ind, ori_r_ind, order_ori,
                inv_row_ptr, inv_col_ind, inv_r_ind, order_inv,
                ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
                inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count,
                _mask_values, self.weight, B, L, E, self.r_size, False, topk_edges,
                False, self.c1 if self.use_topk else None, torch.is_grad_enabled()
            )
            s = normalize_sparse3d(s)
            if is_training:
                s = s if s.is_coalesced() else s.coalesce()
                s = torch.sparse_coo_tensor(s.indices(), self.dropout(s.values()), s.shape, is_coalesced=True)
            states.append(s)

        return sum_sp_3d(states[-1])

    def log_loss(self, p_score, label, thr=1e-7):
        return log_loss_common_sp(p_score, label, self.E, self.tau_2)
