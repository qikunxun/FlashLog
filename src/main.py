import argparse
import time
import random
import json
import inspect
import torch
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from tqdm import tqdm
from dataloader import KGDataset
from utils import scatter_sum

class Option(object):
    def __init__(self, d):
        self.__dict__ = d
    def save(self):
        if not os.path.exists(self.exps_dir):
            os.mkdir(self.exps_dir)
        self.exp_dir = os.path.join(self.exps_dir, self.exp_name)
        if not os.path.exists(self.exp_dir):
            os.mkdir(self.exp_dir)
        with open(os.path.join(self.exp_dir, "option.txt"), "w") as f:
            json.dump(self.__dict__, f, indent=1)
        return True


def normalize_time_checkpoints(values, min_time=-1, max_time=-1):
    if not values:
        return []
    if min_time != -1:
        raise ValueError('--time_checkpoints cannot be combined with --min_time')

    checkpoints = sorted(set(float(value) for value in values))
    if any(not np.isfinite(value) or value <= 0 for value in checkpoints):
        raise ValueError('--time_checkpoints values must be finite and greater than zero')
    if max_time != -1 and checkpoints[-1] > max_time:
        raise ValueError('the largest --time_checkpoints value cannot exceed --max_time')
    return checkpoints


def due_time_checkpoints(checkpoints, next_index, elapsed):
    end_index = next_index
    while end_index < len(checkpoints) and elapsed >= checkpoints[end_index]:
        end_index += 1
    return checkpoints[next_index:end_index], end_index


def time_checkpoint_filename(checkpoint_time):
    return 'model-{}.pt'.format(float(checkpoint_time))

def set_seed(option):
    random.seed(option.seed)
    np.random.seed(option.seed)
    torch.manual_seed(option.seed)
    os.environ['PYTHONHASHSEED'] = str(option.seed)
    if option.use_gpu: torch.cuda.manual_seed_all(option.seed)


def create_graph(kg, device):
    i_h = torch.from_numpy(kg[0].values).int().to(device)
    i_t = torch.from_numpy(kg[2].values).int().to(device)
    i_r = torch.from_numpy(kg[1].values).int().to(device)
    i_v = torch.ones_like(i_h).bool()

    e2triple = (i_h, None, i_v)
    triple2e = (None, i_t, i_v)
    r2triple = (i_r, None, i_v)

    return e2triple, triple2e, r2triple


def move_graph_to_device(e2triple, triple2e, r2triple, device):
    def move_tuple(t):
        return tuple(x.to(device) if x is not None else None for x in t)

    return move_tuple(e2triple), move_tuple(triple2e), move_tuple(r2triple)


def instantiate_model(Model, option, relation_size, entity_size, N):
    kwargs = {
        'n': relation_size,
        'T': option.L,
        'L': option.N,
        'E': entity_size,
        'N': N,
        'emb_size': option.emb_size,
        'tau_1': option.tau_1,
        'tau_2': option.tau_2,
        'use_gpu': option.use_gpu,
        'dropout': option.dropout,
        'c1': option.c1,
        'use_soft': option.use_soft,
        'use_topk': option.use_topk,
        'c2': option.c2,
    }
    sig = inspect.signature(Model.__init__)
    return Model(**{k: v for k, v in kwargs.items() if k in sig.parameters})


def call_model_forward(model, input_x, input_r, e2triple, triple2e, r2triple,
                       is_training=False, csr_struct=None, mask_values=None, mask_values_gpu=None):
    kwargs = {
        'is_training': is_training,
        'csr_struct': csr_struct,
        'mask_values': mask_values_gpu if mask_values_gpu is not None else mask_values,
    }
    sig = inspect.signature(model.forward)
    valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return model(input_x, input_r, e2triple, triple2e, r2triple, **valid_kwargs)


def create_graph_csr(kg, entity_size, device):
    """Build CSR structure for kernel models.
    Returns (csr_struct, mask_values_cpu, mask_values_gpu).
    """
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from flashlog_kernel.flashlog_kernel import build_csr_structure
    i_h = torch.from_numpy(kg[0].values)
    i_t = torch.from_numpy(kg[2].values)
    i_r = torch.from_numpy(kg[1].values)
    mask_values_cpu = torch.ones(i_h.shape[0], dtype=torch.bool)  # Persistent edge-visibility mask

    # build_csr_structure does all heavy work on CPU, moves results to device at end
    csr_struct = build_csr_structure(i_h, i_t, i_r, entity_size)
    csr_struct = tuple(t.to(device) for t in csr_struct)

    mask_values_gpu = mask_values_cpu.to(device)
    return csr_struct, mask_values_cpu, mask_values_gpu


def create_eval_filter_csr(dataset, entity_size, device):
    """Build compact CPU known-answer indices; retain the legacy return shape."""
    import pandas as pd
    # Filtering needs only (source, relation) -> target lookups, not a complete
    # propagation CSR or edge masks. Keep held-out edges separate to avoid a
    # second copy of the entire training DataFrame.
    lookups = []
    for frame in (dataset.kg, pd.concat([dataset.valid_data, dataset.test_data],
                                      ignore_index=True)):
        h, r, t = (torch.from_numpy(frame[i].values) for i in range(3))
        lookups.append((
            _build_filter_edge_lookup(h, r, t, dataset.relation_size),
            _build_filter_edge_lookup(t, r, h, dataset.relation_size),
        ))
    return {"filter_lookups": lookups}, None, None


def _build_filter_edge_lookup(source, relation, target, relation_size):
    from flashlog_kernel.flashlog_kernel import _group_starts_and_counts
    keys = source.to(dtype=torch.long, copy=True)
    keys.mul_(int(relation_size)).add_(relation)
    order = keys.argsort()
    sorted_keys = keys[order]
    del keys
    columns = target[order].to(torch.int32)
    del order
    starts, counts = _group_starts_and_counts(sorted_keys)
    group_keys = torch.empty(starts.numel(), dtype=torch.long)
    for lo in range(0, starts.numel(), 1 << 20):
        hi = min(lo + (1 << 20), starts.numel())
        group_keys[lo:hi] = sorted_keys[starts[lo:hi].long()]
    return group_keys, starts, counts, columns

def mask_data(values, indices, score):
    values[indices] = score


def mask_data_csr(mask_values_cpu, mask_values_gpu, indices, score):
    """Update mask on CPU and sync changed entries to GPU."""
    s = bool(score) if mask_values_cpu.dtype == torch.bool else float(score)
    mask_values_cpu[indices] = s
    mask_values_gpu[indices] = s


_FILTER_CSR_LOOKUP_CACHE = {}


def _filter_csr_cache_key(csr_struct, relation_size):
    fields = (1, 5, 8, 9, 10, 11, 12, 13, 14, 15)
    return (
        int(relation_size),
        tuple(
            (str(csr_struct[i].device), int(csr_struct[i].data_ptr()),
             tuple(csr_struct[i].shape), str(csr_struct[i].dtype))
            for i in fields
        )
    )


def _build_filter_side_lookup(row_group_ptr, group_rel, group_edge_start,
                              group_edge_count, col_ind, relation_size):
    row_group_ptr = row_group_ptr.detach().cpu()
    group_rel = group_rel.detach().cpu()
    group_edge_start = group_edge_start.detach().cpu()
    group_edge_count = group_edge_count.detach().cpu()
    col_ind = col_ind.detach().cpu()

    if group_rel.numel() == 0:
        empty = torch.empty(0, dtype=torch.long)
        return empty, empty, empty, col_ind

    group_key = torch.empty(group_rel.numel(), dtype=torch.long)
    for lo in range(0, group_rel.numel(), 1 << 20):
        hi = min(lo + (1 << 20), group_rel.numel())
        ids = torch.arange(lo, hi, dtype=row_group_ptr.dtype)
        sources = torch.searchsorted(row_group_ptr, ids, right=True).long() - 1
        group_key[lo:hi] = sources.mul_(int(relation_size)).add_(group_rel[lo:hi])
    return group_key.contiguous(), group_edge_start.contiguous(), group_edge_count.contiguous(), col_ind.contiguous()


def _get_filter_csr_lookup(csr_struct, relation_size):
    key = _filter_csr_cache_key(csr_struct, relation_size)
    cached = _FILTER_CSR_LOOKUP_CACHE.get(key)
    if cached is not None:
        return cached

    (_, ori_col_ind, _, _,
     _, inv_col_ind, _, _,
     ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
     inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count) = csr_struct

    cached = (
        _build_filter_side_lookup(
            ori_row_group_ptr, ori_group_rel, ori_group_edge_start,
            ori_group_edge_count, ori_col_ind, relation_size
        ),
        _build_filter_side_lookup(
            inv_row_group_ptr, inv_group_rel, inv_group_edge_start,
            inv_group_edge_count, inv_col_ind, relation_size
        ),
    )
    _FILTER_CSR_LOOKUP_CACHE[key] = cached
    return cached


def _expand_filter_side(batch_pos, query_key, lookup):
    group_key, edge_start, edge_count, col_ind = lookup
    if query_key.numel() == 0 or group_key.numel() == 0:
        empty = torch.empty(0, dtype=torch.long)
        return empty, empty

    pos = torch.searchsorted(group_key, query_key)
    safe_pos = pos.clamp(max=max(group_key.numel() - 1, 0))
    valid = (pos < group_key.numel()) & (group_key[safe_pos] == query_key)
    if not bool(valid.any().item()):
        empty = torch.empty(0, dtype=torch.long)
        return empty, empty

    matched_pos = safe_pos[valid]
    matched_batch = batch_pos[valid]
    starts = edge_start[matched_pos].long()
    counts = edge_count[matched_pos].long()
    nonzero = counts > 0
    if not bool(nonzero.any().item()):
        empty = torch.empty(0, dtype=torch.long)
        return empty, empty

    starts = starts[nonzero]
    counts = counts[nonzero]
    matched_batch = matched_batch[nonzero]
    total = int(counts.sum().item())
    if total == 0:
        empty = torch.empty(0, dtype=torch.long)
        return empty, empty

    out_batch = torch.repeat_interleave(matched_batch, counts)
    offsets = torch.empty(counts.numel() + 1, dtype=torch.long)
    offsets[0] = 0
    offsets[1:] = counts.cumsum(0)
    local = torch.arange(total, dtype=torch.long) - torch.repeat_interleave(offsets[:-1], counts)
    edge_pos = torch.repeat_interleave(starts, counts) + local
    return out_batch, col_ind[edge_pos]


def build_filter_indices_csr(input_x, input_r, relation_size, csr_struct):
    if isinstance(csr_struct, dict) and "filter_lookups" in csr_struct:
        results = [_filter_indices_from_lookup(input_x, input_r, relation_size, lookup)
                   for lookup in csr_struct["filter_lookups"]]
        return tuple(torch.cat([r[i] for r in results]) for i in range(2))
    return _filter_indices_from_lookup(
        input_x, input_r, relation_size,
        _get_filter_csr_lookup(csr_struct, relation_size))


def _filter_indices_from_lookup(input_x, input_r, relation_size, lookup):
    device = input_x.device
    input_x_cpu = input_x.detach().cpu().long()
    input_r_cpu = input_r.detach().cpu().long()
    batch_pos = torch.arange(input_x_cpu.numel(), dtype=torch.long)
    ori_lookup, inv_lookup = lookup

    ori_mask = input_r_cpu < int(relation_size)
    ori_batch, ori_entities = _expand_filter_side(
        batch_pos[ori_mask],
        input_x_cpu[ori_mask] * int(relation_size) + input_r_cpu[ori_mask],
        ori_lookup
    )
    inv_mask = ~ori_mask
    inv_batch, inv_entities = _expand_filter_side(
        batch_pos[inv_mask],
        input_x_cpu[inv_mask] * int(relation_size) + (input_r_cpu[inv_mask] - int(relation_size)),
        inv_lookup
    )

    if ori_batch.numel() == 0 and inv_batch.numel() == 0:
        empty = torch.empty(0, device=device, dtype=torch.long)
        return empty, empty

    batch_ids = torch.cat([ori_batch, inv_batch], dim=0).to(device=device, non_blocking=True)
    entity_ids = torch.cat([ori_entities, inv_entities], dim=0).to(device=device, dtype=torch.long, non_blocking=True)
    return batch_ids, entity_ids


def build_filter_indices_graph(batches, graph, graph_inv, dataset, device):
    batch_ids = []
    entity_ids = []
    for j, items in enumerate(batches):
        h, r = items['h'], items['r']
        if r < dataset.relation_size:
            if (h, r) not in graph.indices.keys():
                continue
            entities = torch.from_numpy(graph.get_group((h, r))[2].values).long().to(device)
        else:
            r_ = r - dataset.relation_size
            if (h, r_) not in graph_inv.indices.keys():
                continue
            entities = torch.from_numpy(graph_inv.get_group((h, r_))[0].values).long().to(device)
        batch_ids.append(torch.full((entities.shape[0],), j, device=device, dtype=torch.long))
        entity_ids.append(entities)

    if not batch_ids:
        empty = torch.empty(0, device=device, dtype=torch.long)
        return empty, empty
    return torch.cat(batch_ids, dim=0), torch.cat(entity_ids, dim=0)


def _sparse_truth_scores(state, targets):
    state = state.coalesce()
    idx = state.indices()
    values = state.values()
    batch_size, entity_size = state.shape
    truth_scores = torch.zeros(batch_size, device=values.device, dtype=values.dtype)
    if values.numel() == 0:
        return truth_scores

    keys = idx[0].long() * entity_size + idx[1].long()
    order = torch.argsort(keys)
    keys_sorted = keys[order]
    values_sorted = values[order]
    target_keys = torch.arange(batch_size, device=values.device, dtype=torch.long) * entity_size + targets.long()
    pos = torch.searchsorted(keys_sorted, target_keys)
    valid_pos = pos < keys_sorted.numel()
    safe_pos = pos.clamp(max=max(keys_sorted.numel() - 1, 0))
    valid = valid_pos & (keys_sorted[safe_pos] == target_keys)
    truth_scores[valid] = values_sorted[safe_pos[valid]]
    return truth_scores


def calculate_raw_hits_sparse(state, batches, hits):
    state = state.coalesce()
    idx = state.indices()
    values = state.values()
    batch_size, entity_size = state.shape
    targets = torch.LongTensor([items['t'] for items in batches]).to(values.device)
    truth_scores = _sparse_truth_scores(state, targets)

    explicit_counts = torch.zeros(batch_size, device=values.device, dtype=values.dtype)
    ge_counts = torch.zeros(batch_size, device=values.device, dtype=values.dtype)
    if values.numel() > 0:
        batch_idx = idx[0].long()
        explicit_counts.scatter_add_(0, batch_idx, torch.ones_like(values))
        ge_counts.scatter_add_(0, batch_idx, (values >= truth_scores[batch_idx]).to(values.dtype))

    missing_counts = entity_size - explicit_counts
    ranks = ge_counts + torch.where(truth_scores <= 0, missing_counts, torch.zeros_like(missing_counts))
    mrr = 0.0
    for rank in ranks.detach().cpu():
        # Includes the target and all tied entities, including implicit zeros.
        rank = int(rank.item())
        hits[rank - 1:] += 1
        mrr += 1.0 / rank
    return mrr


def calculate_hits_full_sparse(state, batches, filter_indices, hits, use_eql=False):
    if not state.is_coalesced():
        state = state.coalesce()
    idx = state.indices()
    values = state.values()
    batch_size, entity_size = state.shape
    device = values.device
    targets = torch.LongTensor([items['t'] for items in batches]).to(device)
    target_keys = torch.arange(batch_size, device=device, dtype=torch.long) * entity_size + targets.long()

    if filter_indices is None:
        filter_keys = target_keys
    else:
        filter_batch, filter_entity = filter_indices
        filter_keys = filter_batch.long() * entity_size + filter_entity.long()
        filter_keys = torch.cat([filter_keys, target_keys], dim=0)
    filter_keys = torch.unique(filter_keys, sorted=True)

    if values.is_cuda and os.environ.get("FLASHLOG_CUDA_SPARSE_EVAL", "1") != "0":
        try:
            from flashlog_kernel.flashlog_kernel import sparse2d_filtered_ranks_cuda
            ranks = sparse2d_filtered_ranks_cuda(
                idx[0], idx[1], values, targets, filter_keys,
                batch_size, entity_size, use_eql=use_eql
            )
            ranks_cpu = ranks.detach().cpu().numpy()
            mrr = float((1.0 / ranks_cpu).sum())
            for rank_value in ranks_cpu:
                hits += (float(rank_value) <= np.arange(1, len(hits) + 1))
            return mrr
        except (ImportError, RuntimeError, AttributeError) as exc:
            if os.environ.get("FLASHLOG_STRICT_CUDA_EVAL", "") == "1":
                raise
            if os.environ.get("FLASHLOG_VERBOSE_FALLBACK", "") == "1":
                print("Falling back to torch sparse eval:", exc)

    truth_scores = _sparse_truth_scores(state, targets)

    if values.numel() > 0:
        keys = idx[0].long() * entity_size + idx[1].long()
        is_filtered = torch.isin(keys, filter_keys)
        keep = ~is_filtered
        keep_batch = idx[0].long()[keep]
        keep_values = values[keep]
    else:
        keep_batch = torch.empty(0, device=device, dtype=torch.long)
        keep_values = values

    greater = torch.zeros(batch_size, device=device, dtype=values.dtype)
    equal = torch.zeros(batch_size, device=device, dtype=values.dtype)
    explicit_unfiltered = torch.zeros(batch_size, device=device, dtype=values.dtype)
    if keep_values.numel() > 0:
        keep_truth = truth_scores[keep_batch]
        greater.scatter_add_(0, keep_batch, (keep_values > keep_truth).to(values.dtype))
        equal.scatter_add_(0, keep_batch, (keep_values == keep_truth).to(values.dtype))
        explicit_unfiltered.scatter_add_(0, keep_batch, torch.ones_like(keep_values))

    filter_batch_unique = filter_keys // entity_size
    filter_counts = torch.zeros(batch_size, device=device, dtype=values.dtype)
    filter_counts.scatter_add_(0, filter_batch_unique, torch.ones(filter_batch_unique.shape[0], device=device, dtype=values.dtype))

    if use_eql:
        missing_unfiltered = entity_size - filter_counts - explicit_unfiltered
        equal = equal + torch.where(truth_scores == 0, missing_unfiltered, torch.zeros_like(missing_unfiltered))
        equal = equal + 1
        ranks = greater + (equal + 1) / 2
    else:
        ranks = greater + 1

    mrr = 0
    for rank in ranks.detach().cpu():
        rank_value = rank.item()
        hits += (rank_value <= np.arange(1, len(hits) + 1))
        mrr += 1 / rank_value
    return mrr


def calculate_hits_full_with_filter_indices(state, batches, filter_indices, hits, use_eql=False):
    batch_size, entity_size = state.shape
    device = state.device
    targets = torch.LongTensor([items['t'] for items in batches]).to(device)
    batch_idx = torch.arange(batch_size, device=device, dtype=torch.long)
    truth_scores = state[batch_idx, targets].clone()

    greater = (state > truth_scores.unsqueeze(1)).sum(dim=1).float()
    if use_eql:
        equal = (state == truth_scores.unsqueeze(1)).sum(dim=1).float()

    if filter_indices is not None:
        filter_batch, filter_entity = filter_indices
        filter_batch = torch.cat([filter_batch.long(), batch_idx], dim=0)
        filter_entity = torch.cat([filter_entity.long(), targets.long()], dim=0)
        filter_keys = torch.unique(filter_batch * entity_size + filter_entity)
        filter_batch = filter_keys // entity_size
        filter_entity = filter_keys % entity_size

        filter_scores = state[filter_batch, filter_entity]
        filtered_greater = torch.zeros(batch_size, device=device, dtype=torch.float)
        filtered_greater.scatter_add_(
            0, filter_batch, (filter_scores > truth_scores[filter_batch]).float()
        )
        greater = greater - filtered_greater

        if use_eql:
            filtered_equal = torch.zeros(batch_size, device=device, dtype=torch.float)
            filtered_equal.scatter_add_(
                0, filter_batch, (filter_scores == truth_scores[filter_batch]).float()
            )
            equal = equal - filtered_equal

    if use_eql:
        equal = equal + 1
        ranks = greater + (equal + 1) / 2
    else:
        ranks = greater + 1

    mrr = 0
    for rank in ranks.detach().cpu():
        rank_value = rank.item()
        hits += (rank_value <= np.arange(1, len(hits) + 1))
        mrr += 1 / rank_value
    return mrr


def calculate_hits(state, top_k, batches, graph, graph_inv, dataset, hits,
                   filter_mask=None, filter_indices=None):
    if filter_indices is not None:
        scores, indices = torch.topk(state, k=top_k)
        targets = torch.LongTensor([items['t'] for items in batches]).to(state.device)
        is_target = indices == targets.unsqueeze(1)
        filter_batch, filter_entity = filter_indices
        if filter_batch.numel() > 0:
            entity_size = state.shape[1]
            filter_keys = torch.unique(filter_batch.long() * entity_size + filter_entity.long())
            top_batch = torch.arange(indices.shape[0], device=state.device, dtype=torch.long).unsqueeze(1)
            top_keys = top_batch * entity_size + indices.long()
            candidate_mask = torch.isin(top_keys.reshape(-1), filter_keys).reshape_as(indices)
            scores = scores.masked_fill(candidate_mask & ~is_target, -1e20)
        top_count = min(10, scores.shape[1])
        reranked = torch.topk(scores, k=top_count, dim=1).indices
        target_hits = torch.gather(is_target, dim=1, index=reranked).detach().cpu()
        for target_hit in target_hits:
            hit_pos = torch.nonzero(target_hit, as_tuple=False)
            if hit_pos.numel() == 0:
                continue
            hits[int(hit_pos[0].item()):] += 1
        return

    if filter_mask is not None:
        scores, indices = torch.topk(state, k=top_k)
        targets = torch.LongTensor([items['t'] for items in batches]).to(state.device)
        candidate_mask = torch.gather(filter_mask, dim=1, index=indices)
        is_target = indices == targets.unsqueeze(1)
        scores = scores.masked_fill(candidate_mask & ~is_target, -1e20)
        top_count = min(10, scores.shape[1])
        reranked = torch.topk(scores, k=top_count, dim=1).indices
        target_hits = torch.gather(is_target, dim=1, index=reranked).detach().cpu()
        for target_hit in target_hits:
            hit_pos = torch.nonzero(target_hit, as_tuple=False)
            if hit_pos.numel() == 0:
                continue
            hits[int(hit_pos[0].item()):] += 1
        return

    scores, indices = torch.topk(state, k=top_k)
    scores = scores.detach().cpu()
    indices = indices.detach().cpu()
    for j, items in enumerate(batches):
        h, r, t = items['h'], items['r'], items['t']
        indices_list = list(indices[j])
        if t in indices_list:
            truth_index = indices_list.index(t)
            if r < dataset.relation_size:
                scores_tail = scores[j, :].clone()
                if (h, r) in graph.indices.keys():
                    group = graph.get_group((h, r))
                    truths = set(group[2].values.tolist())
                    for tail in indices_list:
                        if tail.item() not in truths: continue
                        if tail.item() == t: continue
                        scores_tail[indices_list.index(tail)] = -1e20
                    sorted = torch.topk(scores_tail, k=10)
                    indices_ = list(sorted[1])
                    if truth_index in indices_:
                        rank = indices_.index(truth_index)
                        hits[rank:] += 1
            else:
                scores_head = scores[j, :].clone()
                if (h, r - dataset.relation_size) in graph_inv.indices.keys():
                    group = graph_inv.get_group((h, r - dataset.relation_size))
                    truths = set(group[0].values.tolist())
                    for head in indices_list:
                        if head.item() not in truths: continue
                        if head.item() == t: continue
                        scores_head[indices_list.index(head)] = -1e20
                sorted = torch.topk(scores_head, k=10)
                indices_ = list(sorted[1])
                if truth_index in indices_:
                    rank = indices_.index(truth_index)
                    hits[rank:] += 1

def calculate_hits_full_with_filter(state, batches, filter_mask, hits, use_eql=False):
    scores_all = state.clone()
    targets = torch.LongTensor([items['t'] for items in batches]).to(scores_all.device)
    batch_idx = torch.arange(len(batches), device=scores_all.device)
    truth_scores = scores_all[batch_idx, targets].clone()
    scores_all = scores_all.masked_fill(filter_mask, -1e20)

    greater = (scores_all > truth_scores.unsqueeze(1)).int().sum(dim=1).float()
    if use_eql:
        equal = (scores_all == truth_scores.unsqueeze(1)).int().sum(dim=1).float() + 1
        ranks = greater + (equal + 1) / 2
    else:
        ranks = greater + 1

    mrr = 0
    for rank in ranks.detach().cpu():
        rank_value = rank.item()
        hits += (rank_value <= np.arange(1, len(hits) + 1))
        mrr += 1 / rank_value
    return mrr


def calculate_hits_full(state, batches, e2triple, triple2e, r2triple, graph, graph_inv,
                        dataset, hits, use_eql=False, filter_mask=None, filter_indices=None):
    if filter_indices is not None:
        return calculate_hits_full_with_filter_indices(
            state, batches, filter_indices, hits, use_eql=use_eql
        )
    if filter_mask is not None:
        return calculate_hits_full_with_filter(state, batches, filter_mask, hits, use_eql=use_eql)

    scores_all = state
    mrr = 0
    input_x = []
    input_y = []
    r_mask_x = []
    r_mask_y = []
    truths = []
    truths_eval = []
    for j, items in enumerate(batches):
        h, r, t = items['h'], items['r'], items['t']
        scores = scores_all[j]
        truth_score = scores[t].clone()
        truths.append(truth_score)
        if r < dataset.relation_size:
            input_x.append([j, h])
            mask = (r2triple[0] == r).float()
            r_mask_x.append(mask)
            if (h, r) in graph.indices.keys():
                group = graph.get_group((h, r))
                eval = torch.from_numpy(group[2].values).to(scores_all.device).long()
            else:
                eval = torch.LongTensor([]).to(scores_all.device)
        else:
            input_y.append([j, h])
            r_ = r - dataset.relation_size
            mask = (r2triple[0] == r_).float()
            r_mask_y.append(mask)
            if (h, r - dataset.relation_size) in graph_inv.indices.keys():
                group = graph_inv.get_group((h, r - dataset.relation_size))
                eval = torch.from_numpy(group[0].values).to(scores_all.device).long()
            else:
                eval = torch.LongTensor([]).to(scores_all.device)

        eval = torch.sparse.FloatTensor(torch.stack([eval, torch.zeros_like(eval)], dim=0),
                                          torch.ones_like(eval), torch.Size([scores_all.shape[1], 1]))
        truths_eval.append(eval)

    truths_eval = torch.stack(truths_eval, dim=0).to_dense().squeeze(dim=-1)
    if len(r_mask_x) != 0:
        r_mask_x = torch.stack(r_mask_x, dim=0)
        input_x = torch.LongTensor(input_x).to(scores_all.device)
        input_x_oh = torch.nn.functional.one_hot(input_x[:, 1], e2triple[0].shape[0])
        x_facts = torch.index_select(input_x_oh, index=e2triple[0], dim=1) * r_mask_x
        x_mask = scatter_sum(x_facts, triple2e[1].long(), dim=1, dim_size=scores_all.shape[-1])
        scores_all[input_x[:, 0]] = scores_all[input_x[:, 0]] - x_mask * 1e20 - truths_eval[input_x[:, 0]] * 1e20
    if len(r_mask_y) != 0:
        r_mask_y = torch.stack(r_mask_y, dim=0)
        input_y = torch.LongTensor(input_y).to(scores_all.device)
        input_y_oh = torch.nn.functional.one_hot(input_y[:, 1], e2triple[0].shape[0])
        y_facts = torch.index_select(input_y_oh, index=triple2e[1], dim=1) * r_mask_y
        y_mask = scatter_sum(y_facts, e2triple[0].long(), dim=1, dim_size=scores_all.shape[-1])
        scores_all[input_y[:, 0]] = scores_all[input_y[:, 0]] - y_mask * 1e20 - truths_eval[input_y[:, 0]] * 1e20

    for j, items in enumerate(batches):
        truth_score = truths[j]
        scores = scores_all[j]
        m = (scores > truth_score).int().sum()
        if use_eql:
            n = (scores == truth_score).int().sum() + 1
            rank = m + (n + 1) / 2
        else:
            rank = m + 1
        hits += (rank.item() <= np.arange(1, len(hits) + 1))
        mrr += 1 / rank.item()
    return mrr

def topk(matrix, k):
    index_part = np.argpartition(matrix, -k)[-k:]
    part_sort_k = np.argsort(matrix[index_part], axis=-1)
    top_k_index = np.flip(index_part[part_sort_k], axis=-1)
    top_k_scores = matrix[top_k_index]
    return top_k_scores, top_k_index

def valid_process(valid_data, dataset, model, e2triple, triple2e, r2triple, graph_entity,
                  option, raw=False, name='Valid', top_k=100, is_sampled=False,
                  csr_struct=None, mask_values=None, mask_values_gpu=None,
                  filter_csr_struct=None, filter_mask_values=None, filter_mask_values_gpu=None):
    model.eval()
    hits = np.zeros(10)
    count = 0
    mrr = 0
    batch_size = option.batch_size
    used_sparse_full_ranking = False
    with torch.inference_mode():
        eval_e2triple, eval_triple2e, eval_r2triple = e2triple, triple2e, r2triple
        if option.use_gpu and csr_struct is not None and e2triple is not None and e2triple[0].device.type != 'cuda':
            eval_e2triple, eval_triple2e, eval_r2triple = move_graph_to_device(
                e2triple, triple2e, r2triple, 'cuda'
            )
        graph = graph_inv = None
        if filter_csr_struct is None:
            graph = dataset.graph_entity[0]
            graph_inv = dataset.graph_entity[1]
        data = []
        for index in valid_data.index:
            h, r, t = valid_data[0][index], valid_data[1][index], valid_data[2][index]
            data.append({'h': h, 'r': r, 't': t})
            data.append({'h': t, 'r': r + dataset.relation_size, 't': h})
        if len(data) == 0: return
        if len(data) % batch_size == 0:
            batch_num = int(len(data) / batch_size)
        else:
            batch_num = int(len(data) / batch_size) + 1
        for i in tqdm(range(batch_num)):
            input_x = []
            input_r = []
            batches = data[i * batch_size: (i + 1) * batch_size]
            for items in batches:
                h = items['h']
                r = items['r']
                input_x.append(h)
                input_r.append(r)
            input_x = torch.LongTensor(input_x)
            input_r = torch.LongTensor(input_r)
            if option.use_gpu:
                input_x = input_x.cuda()
                input_r = input_r.cuda()
            state = call_model_forward(
                model, input_x, input_r, eval_e2triple, eval_triple2e, eval_r2triple,
                is_training=False, csr_struct=csr_struct,
                mask_values=mask_values, mask_values_gpu=mask_values_gpu
            )
            if getattr(state, 'is_sparse', False):
                if name == 'Valid' and option.raw:
                    mrr += calculate_raw_hits_sparse(state, batches, hits)
                    count += len(batches)
                    continue

                if filter_csr_struct is not None:
                    filter_indices = build_filter_indices_csr(
                        input_x, input_r, dataset.relation_size, filter_csr_struct
                    )
                elif graph is not None and graph_inv is not None:
                    filter_indices = build_filter_indices_graph(
                        batches, graph, graph_inv, dataset, state.device
                    )
                else:
                    filter_indices = None

                mrr_batch = calculate_hits_full_sparse(
                    state, batches, filter_indices, hits, use_eql=True
                )
                mrr += mrr_batch
                count += len(batches)
                used_sparse_full_ranking = True
                continue

            filter_mask = None
            filter_indices = None
            if filter_csr_struct is not None and not (name == 'Valid' and option.raw):
                filter_indices = build_filter_indices_csr(
                    input_x, input_r, dataset.relation_size, filter_csr_struct
                )
            if name == 'Valid' and option.raw:
                scores = state.detach().cpu()
                for j, items in enumerate(batches):
                    h, r, t = items['h'], items['r'], items['t']
                    truth_score_ori = scores[j][t]
                    # Keep the existing pessimistic tie policy; ranks are one-based.
                    rank = int((scores[j, :] >= truth_score_ori).sum().item())
                    hits[rank - 1:] += 1
                    mrr += 1.0 / rank
                    count += 1
            else:
                if option.eval_top:
                    calculate_hits(state, top_k, batches, graph, graph_inv, dataset, hits,
                                   filter_mask=filter_mask, filter_indices=filter_indices)
                else:
                    mrr_batch = calculate_hits_full(state, batches, eval_e2triple, eval_triple2e, eval_r2triple,
                                                    graph, graph_inv, dataset, hits, use_eql=True,
                                                    filter_mask=filter_mask, filter_indices=filter_indices)
                    mrr += mrr_batch
                count += len(batches)

    if count > 0:
        hits /= count
        mrr /= count
    print(hits)
    if option.eval_top and not used_sparse_full_ranking and not (name == 'Valid' and option.raw):
        hk_prev = 0
        for i in range(10):
            hk = hits[i]
            hk_diff = hk - hk_prev
            mrr += hk_diff * (1.0 / (i + 1))
            hk_prev = hk

    print('{} Count:{}\tMrr:{}\tHit@1:{}\tHit@3:{}\tHit@10:{}'.format(name, count, mrr, hits[0], hits[2], hits[9]))
    return mrr, hits[0], hits[2], hits[9], count

def train(dataset, valid_data, graph_entity, e2triple, triple2e, r2triple, option,
          csr_struct=None, mask_values=None, mask_values_gpu=None,
          filter_csr_struct=None, filter_mask_values=None, filter_mask_values_gpu=None):
    entity_size = dataset.entity_dict.shape[0]
    kg_size = dataset.kg.shape[0]
    N = e2triple[-1].shape[0] if e2triple is not None else dataset.kg.shape[0]
    device = 'cuda' if option.use_gpu else 'cpu'
    print('Current Time: {}, Max running time: {}'.format(time.strftime('%Y-%m-%d %H:%M:%S'), option.max_time))
    print('Entity Num:', entity_size)
    print('Relation Num:', len(dataset.relation2id))
    print('Train KG Size:', kg_size)
    model = instantiate_model(Model, option, len(dataset.relation2id), entity_size, N)

    if option.use_gpu:
        model = model.cuda()
        torch.cuda.empty_cache()
    if option.use_soft: model.weight.cpu()

    optimizer = torch.optim.Adam(model.parameters(), lr=option.learning_rate)

    end_flag = False
    max_score = -1
    max_record = {'mrr': 0, 'hit_1': 0, 'hit_3': 0, 'hit_10': 0, 'epoch': 0}
    saved_flag = False
    ori_time = time.time()
    min_time = option.min_time
    time_checkpoints = option.time_checkpoints
    next_checkpoint_index = 0
    checkpoint_manifest = None
    checkpoint_manifest_path = os.path.join(option.exp_dir, 'time_checkpoints.json')
    if time_checkpoints:
        checkpoint_manifest = {
            'timer_scope': ('wall time from before the first training epoch; batch-index sampling, '
                            'per-batch data preparation, validation and checkpoint serialization are included; '
                            'data loading, CSR construction and model/optimizer setup are excluded'),
            'requested_times_seconds': time_checkpoints,
            'checkpoints': [],
        }
        with open(checkpoint_manifest_path, 'w') as f:
            json.dump(checkpoint_manifest, f, indent=1)
    for e in range(option.max_epoch):
        model.train()
        total_loss = 0
        if end_flag: break
        for k, batch in enumerate(dataset.batch_iter()):
            start_time = time.time()
            elapsed = start_time - ori_time
            if time_checkpoints:
                due_checkpoints, next_checkpoint_index = due_time_checkpoints(
                    time_checkpoints, next_checkpoint_index, elapsed
                )
                for checkpoint_time in due_checkpoints:
                    checkpoint_filename = time_checkpoint_filename(checkpoint_time)
                    checkpoint_start = time.time()
                    torch.save(model.state_dict(), os.path.join(option.exp_dir, checkpoint_filename))
                    checkpoint_record = {
                        'requested_time_seconds': checkpoint_time,
                        'actual_elapsed_seconds': elapsed,
                        'save_duration_seconds': time.time() - checkpoint_start,
                        'epoch': e,
                        'next_batch': k,
                        'filename': checkpoint_filename,
                    }
                    checkpoint_manifest['checkpoints'].append(checkpoint_record)
                    with open(checkpoint_manifest_path, 'w') as f:
                        json.dump(checkpoint_manifest, f, indent=1)
                    print('Saved time checkpoint: {}'.format(checkpoint_record))
            elif min_time != -1 and elapsed > min_time:
                torch.save(model.state_dict(), os.path.join(option.exp_dir, 'model-{}.pt'.format(min_time)))
                min_time += option.min_time

            if option.max_time != -1 and elapsed > option.max_time:
                end_flag = True
                break
            indices, triples, _ = batch
            if csr_struct is not None:
                mask_data_csr(mask_values, mask_values_gpu, indices, False)
            else:
                mask_data(e2triple[-1], indices, False)
            loss = 0
            input_x = []
            input_r = []
            input_y = []
            for x, r, y in triples:
                input_x.append(x)
                input_r.append(r)
                input_y.append(y)

            input_x = torch.LongTensor(input_x)
            input_r = torch.LongTensor(input_r)
            input_y = torch.LongTensor(input_y)
            if option.use_gpu:
                input_x = input_x.cuda()
                input_r = input_r.cuda()
                input_y = input_y.cuda()
            state = call_model_forward(
                model, input_x, input_r, e2triple, triple2e, r2triple,
                is_training=True, csr_struct=csr_struct,
                mask_values=mask_values, mask_values_gpu=mask_values_gpu
            )
            end_time_forward = time.time()
            loss = model.log_loss(state, input_y)

            end_time_loss = time.time()
            total_loss += loss.item()
            loss.backward()
            if (k + 1) % option.accum_step == 0:
                optimizer.step()
                optimizer.zero_grad()
            end_time = time.time()
            if (k + 1) % 10 == 0:
                print('Epoch: {}, Batch: {}, Loss: {}, Time cost: {}\t{}\t{}'.format(e, k, loss.item(),
                                                                                     end_time_forward - start_time,
                                                                                     end_time_loss - start_time,
                                                                                     end_time - start_time))
            if csr_struct is not None:
                mask_data_csr(mask_values, mask_values_gpu, indices, True)
            else:
                mask_data(e2triple[-1], indices, True)
        print('Epoch: {}, Total Loss: {}'.format(e, total_loss))
        if option.early_stop and e > -1:
            mrr, hit_1, hit_3, hit_10, _ = valid_process(valid_data, dataset, model, e2triple, triple2e,
                                                 r2triple, graph_entity, option, is_sampled=False, name='Valid',
                                                 csr_struct=csr_struct,
                                                 mask_values=mask_values, mask_values_gpu=mask_values_gpu,
                                                 filter_csr_struct=filter_csr_struct,
                                                 filter_mask_values=filter_mask_values,
                                                 filter_mask_values_gpu=filter_mask_values_gpu)
            if hit_10 > max_score:
                max_score = hit_10
                max_record['mrr'] = {'valid': mrr}
                max_record['hit_1'] = {'valid': hit_1}
                max_record['hit_3'] = {'valid': hit_3}
                max_record['hit_10'] = {'valid': hit_10}
                max_record['epoch'] = e

                torch.save(model.state_dict(), os.path.join(option.exp_dir, 'model.pt'))
                saved_flag = True
            print('=' * 100)
            print('Valid Max Score : Epoch:{}\tMrr:{}\tHit@1:{}\tHit@3:{}\tHit@10:{}'.format(max_record['epoch'],
                                                                            max_record['mrr']['valid'], max_record['hit_1']['valid'],
                                                                            max_record['hit_3']['valid'], max_record['hit_10']['valid']))
            if max_score == 1: end_flag = True
            if e - max_record['epoch'] >= option.max_epoch // 2: end_flag = True
    if not saved_flag: torch.save(model.state_dict(), os.path.join(option.exp_dir, 'model.pt'))

def test(dataset, graph_entity, e2triple, triple2e, r2triple, option,
         csr_struct=None, mask_values=None, mask_values_gpu=None,
         filter_csr_struct=None, filter_mask_values=None, filter_mask_values_gpu=None):
    N = e2triple[-1].shape[0] if e2triple is not None else dataset.kg.shape[0]
    model = instantiate_model(
        Model, option, len(dataset.relation2id), dataset.entity_dict.shape[0], N
    )
    if option.ckpt_step == -1:
        model_save_path = os.path.join(option.exp_dir, 'model.pt')
    else:
        model_save_path = os.path.join(option.exp_dir, time_checkpoint_filename(option.ckpt_step))
    model.load_state_dict(torch.load(model_save_path, map_location=torch.device('cpu')))
    if option.use_gpu:
        model = model.cuda()
        torch.cuda.empty_cache()
    if option.use_soft: model.weight.cpu()
    test_data = dataset.test_data
    if option.eval_valid: test_data = dataset.valid_data
    eval_relation_ids = getattr(option, 'eval_relation_ids', None)
    if eval_relation_ids is not None:
        test_data = test_data[test_data[1].isin(eval_relation_ids)]
    mrr, hit_1, hit_3, hit_10, _ = valid_process(test_data, dataset, model, e2triple, triple2e,
                                                     r2triple, graph_entity, option, name='Test',
                                                     csr_struct=csr_struct,
                                                     mask_values=mask_values, mask_values_gpu=mask_values_gpu,
                                                     filter_csr_struct=filter_csr_struct,
                                                     filter_mask_values=filter_mask_values,
                                                     filter_mask_values_gpu=filter_mask_values_gpu)
    return mrr, hit_1, hit_3, hit_10

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="FlashLog model training and evaluation")
    parser.add_argument('--data_dir', required=True, type=str)
    parser.add_argument('--exps_dir', required=True, type=str)
    parser.add_argument('--exp_name', required=True, type=str)
    parser.add_argument('--model_name', default='DRUM', choices=['DRUM', 'NeuralLP', 'smDRUM', 'mmDRUM', 'DRUM_kernel', 'NeuralLP_kernel', 'smDRUM_kernel', 'mmDRUM_kernel'])
    parser.add_argument('--use_gpu',  default=False, action="store_true")
    parser.add_argument('--gpu_id', default=0, type=int)
    # model architecture
    parser.add_argument('--N', default=3, type=int)
    parser.add_argument('--L', default=3, type=int)
    parser.add_argument('--tau_1', default=1, type=float)
    parser.add_argument('--tau_2', default=1, type=float)
    parser.add_argument('--use_topk', '-use_topk', default=False, action="store_true",
                        help="Budgeted propagation: weighted sampling in training, exact top-k in evaluation")
    parser.add_argument('--c1', default=1000, type=int)
    parser.add_argument('--c2', default=400, type=int,
                        help='Total fact budget; each direction uses c2 // 2 (0: unlimited)')
    parser.add_argument('--emb_size', default=128, type=int)
    parser.add_argument('--lambda', default=0.0, type=float)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--eval_top', default=False, action="store_true")
    parser.add_argument('--sparse', default=False, action="store_true")
    parser.add_argument('--use_soft', default=False, action="store_true")
    # optimization
    parser.add_argument('--max_epoch', default=10, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--accum_step', default=1, type=int)
    parser.add_argument('--iteration_per_batch', default=100000, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--early_stop', default=False, action="store_true")
    parser.add_argument('--do_train', default=False, action="store_true")
    parser.add_argument('--do_test', default=False, action="store_true")
    parser.add_argument('--wot_inv', default=False, action="store_true")
    parser.add_argument('--eval_valid', default=False, action="store_true")
    parser.add_argument('--raw', default=False, action="store_true")
    parser.add_argument('--max_time', default=-1, type=float)
    parser.add_argument('--min_time', default=-1, type=float)
    parser.add_argument('--time_checkpoints', nargs='+', default=None, type=float)
    parser.add_argument('--threshold', default=1e-6, type=float)
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--ckpt_step', default=-1, type=float)
    parser.add_argument('--test_checkpoints', nargs='+', default=None, type=float)
    parser.add_argument(
        '--eval_relation_ids',
        nargs='+',
        default=None,
        type=int,
        help=(
            'optional base relation IDs to evaluate; the complete valid/test '
            'sets are still loaded for filtered ranking'
        ),
    )

    d = vars(parser.parse_args())
    if not d['model_name'].endswith('_kernel'):
        d['model_name'] += '_kernel'
    d['time_checkpoints'] = normalize_time_checkpoints(
        d['time_checkpoints'], min_time=d['min_time'], max_time=d['max_time']
    )

    option = Option(d)
    option.use_csr_eval = option.model_name is not None and 'kernel' in option.model_name
    if option.use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(option.gpu_id)
    option.tag = time.strftime("%y-%m-%d %H:%M:%S")
    set_seed(option)
    dataset = KGDataset(option)

    device = 'cpu'
    if option.use_gpu: device = 'cuda'

    is_kernel = option.use_csr_eval
    csr_struct = None
    mask_values = None
    mask_values_gpu = None
    filter_csr_struct = None
    filter_mask_values = None
    filter_mask_values_gpu = None


    if is_kernel:
        e2triple = triple2e = r2triple = None
        entity_size = dataset.entity_dict.shape[0]
        csr_struct, mask_values, mask_values_gpu = create_graph_csr(dataset.kg, entity_size, device)
        print('CSR structure built.')
        if option.early_stop or option.do_test:
            filter_csr_struct, filter_mask_values, filter_mask_values_gpu = create_eval_filter_csr(
                dataset, entity_size, 'cpu'
            )
            print('Compact eval filter index built on CPU.')
    else:
        e2triple, triple2e, r2triple = create_graph(dataset.kg, device)
    graph_entity = None
    if option.early_stop or option.do_test: graph_entity = getattr(dataset, 'graph_entity', None)

    import importlib
    module_name = 'model_' + option.model_name + ('_sp' if option.sparse else '')
    Model = importlib.import_module(module_name).Model
    if option.do_train:
        bl = option.save()
        print("Option saved.")
        ori_time = time.time()
        train(dataset, dataset.valid_data, graph_entity, e2triple, triple2e, r2triple, option,
              csr_struct=csr_struct,
              mask_values=mask_values, mask_values_gpu=mask_values_gpu,
              filter_csr_struct=filter_csr_struct,
              filter_mask_values=filter_mask_values,
              filter_mask_values_gpu=filter_mask_values_gpu)
        print('Current Time: {}, Total cost: {}'.format(time.strftime('%Y-%m-%d %H:%M:%S'), time.time() - ori_time))
    if option.do_test:
        option.exp_dir = os.path.join(option.exps_dir, option.exp_name)
        test_checkpoints = option.test_checkpoints
        if test_checkpoints is None:
            test_checkpoints = [option.ckpt_step]
        test_records = []
        for checkpoint_time in test_checkpoints:
            option.ckpt_step = checkpoint_time
            ori_time = time.time()
            print('Testing checkpoint: {}'.format(checkpoint_time))
            mrr_all, hit_1_all, hit_3_all, hit_10_all = test(
                dataset, graph_entity, e2triple, triple2e, r2triple, option,
                csr_struct=csr_struct,
                mask_values=mask_values, mask_values_gpu=mask_values_gpu,
                filter_csr_struct=filter_csr_struct,
                filter_mask_values=filter_mask_values,
                filter_mask_values_gpu=filter_mask_values_gpu
            )
            test_record = {
                'checkpoint_time_seconds': float(checkpoint_time),
                'mrr': float(mrr_all),
                'hit_1': float(hit_1_all),
                'hit_3': float(hit_3_all),
                'hit_10': float(hit_10_all),
                'evaluation_time_seconds': time.time() - ori_time,
            }
            test_records.append(test_record)
            print('Test Score: Mrr:{}\tHit@1:{}\tHit@3:{}\tHit@10:{}'.format(
                mrr_all, hit_1_all, hit_3_all, hit_10_all
            ))
            print('Current Time: {}, Total cost: {}'.format(
                time.strftime('%Y-%m-%d %H:%M:%S'), test_record['evaluation_time_seconds']
            ))
        if option.test_checkpoints is not None:
            with open(os.path.join(option.exp_dir, 'test_time_checkpoints.json'), 'w') as f:
                json.dump(test_records, f, indent=1)
