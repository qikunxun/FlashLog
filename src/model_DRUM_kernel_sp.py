import sys
import os
import torch
import numpy as np
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from flashlog_kernel.flashlog_kernel import (
    FlashLogFunctionSparse3DAllEdgesSum,
    FlashLogFunctionSparse3DTopKSum,
    configure_flashlog_cpu_threads,
    prepare_csr_masks,
    vectorized_operation_sparse3d_csr_sum,
    vectorized_operation_sparse3d_csr_sum_norm,
    build_csr_structure,
    normalize_sparse3d,
)
from utils import log_loss_common_sp


def sum_sp_3d(s):
    s = s.coalesce()
    idx = s.indices()   # [3, nnz]
    val = s.values()    # [nnz]
    B, L, E = s.shape
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
        for l in range(self.L):
            self.lstm.append(nn.LSTM(self.emb_size, self.emb_size, 1, bidirectional=True))
        self.linear = nn.Linear(2 * self.emb_size, self.n)

        if self.use_soft:
            self.weight = nn.Parameter(torch.Tensor(self.N, 1))
            nn.init.zeros_(self.weight)
        else:
            self.weight = None

        self.use_gpu = use_gpu
        self.dropout = nn.Dropout(dropout)
        self.cpu_fused_norm = os.environ.get("FLASHLOG_CPU_FUSED_NORM", "1") != "0"
        if not self.use_gpu:
            self.flashlog_cpu_threads = configure_flashlog_cpu_threads()
        else:
            self.flashlog_cpu_threads = None

        # CSR cache
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
        B = input_x.shape[0]
        E = self.E
        L = self.L
        device = input_x.device

        # Generate rule weights w_all: [B, T, L, n]
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
        w_all = torch.stack(w_all, dim=2)  # [B, T, L, n]

        # Use pre-built CSR if provided, else fall back to internal cache
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

        sp_batch = torch.arange(B, device=device, dtype=torch.long).repeat_interleave(L)
        sp_level = torch.arange(L, device=device, dtype=torch.long).repeat(B)
        sp_entity = input_x.long().repeat_interleave(L)
        sp_value = torch.ones(B * L, device=device, dtype=torch.float)
        idx0 = torch.stack([sp_batch, sp_level, sp_entity], dim=0)
        state = torch.sparse_coo_tensor(idx0, sp_value, [B, L, E]).coalesce()
        states = [state]

        for t in range(self.T):
            w_probs = w_all[:, t, :, :]                        # [B, L, n]
            w_t = torch.softmax(w_probs / self.tau_1, dim=-1)  # [B, L, n]
            normalized_in_kernel = False

            if use_cuda:
                # GPU path: use CUDA sparse kernel directly
                (ori_row_ptr, ori_col_ind, ori_r_ind, order_ori,
                 inv_row_ptr, inv_col_ind, inv_r_ind, order_inv,
                 ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
                 inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count) = _csr_struct

                topk_edges = self.c2 // 2 if (self.use_topk and self.c2 > 0) else (1 << 60)
                if use_all_edges:
                    ori_mask, inv_mask = prepared_masks
                    s = FlashLogFunctionSparse3DAllEdgesSum.apply(
                        sp_batch, sp_level, sp_entity, sp_value, w_t,
                        ori_col_ind, ori_mask,
                        ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
                        inv_col_ind, inv_mask,
                        inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count,
                        B, L, E, self.r_size, False
                    )
                else:
                    s = FlashLogFunctionSparse3DTopKSum.apply(
                        sp_batch, sp_level, sp_entity, sp_value, w_t,
                        ori_row_ptr, ori_col_ind, ori_r_ind, order_ori,
                        inv_row_ptr, inv_col_ind, inv_r_ind, order_inv,
                        ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
                        inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count,
                        _mask_values, self.weight, B, L, E, self.r_size, False, topk_edges,
                        False, self.c1 if self.use_topk else None, torch.is_grad_enabled()
                    )
            else:
                # CPU path: use unified dispatch
                state_sp = torch.sparse_coo_tensor(
                    torch.stack([sp_batch, sp_level, sp_entity], dim=0),
                    sp_value, [B, L, E]).coalesce()
                if self.cpu_fused_norm:
                    s = vectorized_operation_sparse3d_csr_sum_norm(
                        state_sp, w_t, _csr_struct, _mask_values,
                        self.r_size, weight=self.weight,
                        use_topk=self.use_topk, topk_edges=self.c2 // 2, topk_nodes=self.c1
                    )
                    normalized_in_kernel = True
                else:
                    s = vectorized_operation_sparse3d_csr_sum(
                        state_sp, w_t, _csr_struct, _mask_values,
                        self.r_size, weight=self.weight,
                        use_topk=self.use_topk, topk_edges=self.c2 // 2, topk_nodes=self.c1
                    )
            if not normalized_in_kernel:
                s = normalize_sparse3d(s)

            if is_training:
                s = s if s.is_coalesced() else s.coalesce()
                val = self.dropout(s.values())
                s = torch.sparse_coo_tensor(s.indices(), val, s.shape, is_coalesced=True)

            states.append(s)

            # Unpack for next step
            s_c = s if s.is_coalesced() else s.coalesce()
            sp_batch = s_c.indices()[0]
            sp_level = s_c.indices()[1]
            sp_entity = s_c.indices()[2]
            sp_value = s_c.values()

        # Final: sum over L -> sparse [B, E]
        final = sum_sp_3d(states[-1])
        return final

    def log_loss(self, p_score, label):
        return log_loss_common_sp(p_score, label, self.E, self.tau_2)
