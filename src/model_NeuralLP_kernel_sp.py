"""
model_NeuralLP_kernel_sp.py

Sparse-state NeuralLP model accelerated with flashlog_kernel sparse3d kernel.
The model keeps the original NeuralLP semantics: a single state channel is
propagated across steps, so sparse3d uses L=1 internally.
"""

import sys
import os
import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from flashlog_kernel.flashlog_kernel import (
    FlashLogFunctionSparse3DAllEdgesSum,
    FlashLogFunctionSparse3DTopKSum,
    prepare_csr_masks,
    vectorized_operation_sparse3d_csr_sum,
    build_csr_structure,
    normalize_sparse3d,
)
from utils import log_loss_common_sp


def sum_sp_3d(s):
    s = s.coalesce()
    idx = s.indices()
    val = s.values()
    b_dim, _, e_dim = s.shape
    idx2 = torch.stack([idx[0], idx[2]], dim=0)
    return torch.sparse_coo_tensor(idx2, val, [b_dim, e_dim]).coalesce()


class Model(nn.Module):
    def __init__(self, n, T, L, E, N, emb_size, tau_1=1, tau_2=1, use_gpu=False,
                 dropout=0.1, c1=1000, use_soft=False, use_topk=False, c2=400):
        super(Model, self).__init__()
        self.T = T
        self.L = L
        self.E = E
        self.n = n - 1
        self.r_size = (n - 1) // 2
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
        self.emb = nn.Embedding(self.n + 1, self.emb_size)
        self.lstm = nn.LSTM(self.emb_size, self.emb_size, 1)
        self.linear = nn.Linear(self.emb_size, self.n)
        if self.use_soft:
            self.weight = nn.Parameter(torch.Tensor(self.N, 1))
            nn.init.zeros_(self.weight)
        else:
            self.weight = None

        self.use_gpu = use_gpu
        self.dropout = nn.Dropout(dropout)

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

    def _match_att(self, state, att):
        state = state if state.is_coalesced() else state.coalesce()
        idx = state.indices()
        val = state.values()
        att_exp = torch.index_select(att, dim=0, index=idx[0])
        val = val * att_exp
        return torch.sparse_coo_tensor(idx, val, state.shape, is_coalesced=True)

    def forward(self, input_x, input_r, e2triple, triple2e, r2triple,
                is_training=False, input_y=None, csr_struct=None, mask_values=None):
        batch_size = input_x.shape[0]
        device = input_x.device

        input_emb_ori = self.emb(input_r)
        input_emb = torch.stack([input_emb_ori] * (self.T + 1), dim=1)
        input_emb[:, -1, :] = self.emb(torch.ones_like(input_r) * self.n)

        rnn_outputs, _ = self.lstm(input_emb.transpose(1, 0))
        rnn_outputs = rnn_outputs.transpose(1, 0)
        w_all = self.linear(rnn_outputs[:, :-1, :])

        if csr_struct is not None and mask_values is not None:
            _csr_struct = csr_struct
            _mask_values = mask_values.to(device)
        else:
            _csr_struct, _mask_values = self._ensure_csr_struct(e2triple, triple2e, r2triple)
        use_cuda = self.use_gpu and torch.cuda.is_available()
        use_all_edges = use_cuda and not self.use_topk and not (
            self.weight is not None and bool(getattr(self.weight, "requires_grad", False))
        )
        prepared_masks = prepare_csr_masks(_csr_struct, _mask_values, self.weight) if use_all_edges else None

        sp_batch = torch.arange(batch_size, device=device, dtype=torch.long)
        sp_level = torch.zeros(batch_size, device=device, dtype=torch.long)
        sp_entity = input_x.long()
        sp_value = torch.ones(batch_size, device=device, dtype=torch.float)
        state0 = torch.sparse_coo_tensor(
            torch.stack([sp_batch, sp_level, sp_entity], dim=0),
            sp_value,
            [batch_size, 1, self.E]
        ).coalesce()
        states = [state0]

        for t in range(self.T + 1):
            a = rnn_outputs[:, t]
            b = rnn_outputs[:, :t + 1]
            x = torch.einsum("bd, btd -> bt", a, b)
            attention = torch.softmax(x, dim=-1)

            weighted_states = []
            for j, state in enumerate(states):
                weighted_states.append(self._match_att(state, attention[:, j]))
            input_state = weighted_states[0]
            for state in weighted_states[1:]:
                input_state = (input_state + state).coalesce()

            if t < self.T:
                w_probs = w_all[:, t, :].unsqueeze(dim=1)
                w_t = torch.softmax(w_probs / self.tau_1, dim=-1)

                if use_cuda:
                    # GPU path: use CUDA sparse kernel directly
                    input_c = input_state.coalesce()
                    sp_b = input_c.indices()[0]
                    sp_l = input_c.indices()[1]
                    sp_e = input_c.indices()[2]
                    sp_v = input_c.values()

                    (ori_row_ptr, ori_col_ind, ori_r_ind, order_ori,
                     inv_row_ptr, inv_col_ind, inv_r_ind, order_inv,
                     ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
                    inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count) = _csr_struct

                    topk_edges = self.c2 // 2 if (self.use_topk and self.c2 > 0) else (1 << 60)
                    if use_all_edges:
                        ori_mask, inv_mask = prepared_masks
                        s = FlashLogFunctionSparse3DAllEdgesSum.apply(
                            sp_b, sp_l, sp_e, sp_v, w_t,
                            ori_col_ind, ori_mask,
                            ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
                            inv_col_ind, inv_mask,
                            inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count,
                            batch_size, 1, self.E, self.r_size, True, torch.is_grad_enabled()
                        )
                    else:
                        s = FlashLogFunctionSparse3DTopKSum.apply(
                            sp_b, sp_l, sp_e, sp_v, w_t,
                            ori_row_ptr, ori_col_ind, ori_r_ind, order_ori,
                            inv_row_ptr, inv_col_ind, inv_r_ind, order_inv,
                            ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
                            inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count,
                            _mask_values, self.weight, batch_size, 1, self.E, self.r_size, True, topk_edges,
                            False, self.c1 if self.use_topk else None, torch.is_grad_enabled()
                        )
                else:
                    # CPU path: use unified dispatch
                    s = vectorized_operation_sparse3d_csr_sum(
                        input_state, w_t, _csr_struct, _mask_values,
                        self.r_size, weight=self.weight,
                        use_topk=self.use_topk, topk_edges=self.c2 // 2, wot_i=True,
                        topk_nodes=self.c1
                    )
                s = normalize_sparse3d(s)
                if is_training:
                    s = s if s.is_coalesced() else s.coalesce()
                    val = self.dropout(s.values())
                    s = torch.sparse_coo_tensor(s.indices(), val, s.shape, is_coalesced=True)
            else:
                s = input_state

            states.append(s)

        return sum_sp_3d(states[-1])

    def log_loss(self, p_score, label):
        return log_loss_common_sp(p_score, label, self.E, self.tau_2)
