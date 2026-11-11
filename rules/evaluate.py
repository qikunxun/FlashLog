#!/usr/bin/env python3
"""Evaluate extracted Freebase DRUM rules with FlashLog CUDA CSR kernels."""

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from flashlog_kernel.flashlog_kernel import (  # noqa: E402
    build_csr_structure,
    sparse2d_filtered_ranks_cuda,
    vectorized_operation_sparse3d_fixed_relation,
)


CSR_RULE_FIELDS = frozenset((1, 5, 8, 9, 10, 11, 12, 13, 14, 15))
RANK_STREAMING_RULE_THRESHOLD = int(
    os.environ.get("FLASHLOG_RANK_STREAMING_RULE_THRESHOLD", "1024")
)


try:
    from .indexed_bank import IndexedRuleBank, IndexedHeadRules
except ImportError:
    from indexed_bank import IndexedRuleBank, IndexedHeadRules


def dictionary_size(path):
    with path.open("rb") as file:
        file.seek(0, 2)
        end = file.tell()
        position = end - 1
        while position > 0:
            file.seek(position)
            if file.read(1) == b"\n" and position < end - 1:
                break
            position -= 1
        file.seek(position + 1 if position > 0 else 0)
        return int(file.readline().split(b"\t", 1)[0]) + 1


def load_triples(path):
    triples = []
    with path.open() as file:
        for line in file:
            triples.append(tuple(map(int, line.split())))
    return triples


def load_rules(path, weight_field="weight"):
    rules = []
    with path.open(newline="") as file:
        for rule_id, row in enumerate(csv.DictReader(file, delimiter="\t")):
            body = tuple(map(int, row["body_ids"].split())) if row["body_ids"] else ()
            if weight_field not in row:
                raise ValueError(
                    f"rule weight field {weight_field!r} is not present in {path}"
                )
            rules.append(
                {
                    "rule_id": rule_id,
                    "head": int(row["head_id"]),
                    "body": body,
                    "weight": float(row[weight_field]),
                    "rank_weights": (
                        tuple(map(float, row["rank_weights"].split()))
                        if row.get("rank_weights")
                        else None
                    ),
                    "group_id": (
                        int(row["group_id"])
                        if row.get("group_id") is not None
                        and row["group_id"] != ""
                        else None
                    ),
                }
            )
    return rules


def compact_routed_rules(rules):
    if isinstance(rules, IndexedHeadRules):
        return rules.compact()
    """Share grounding for scalar Horn instances with the same head and body."""
    if any(rule["group_id"] is None for rule in rules):
        raise ValueError("routed_normalized requires a group_id column")
    group_count = max(rule["group_id"] for rule in rules) + 1
    compact = {}
    for rule in rules:
        key = (rule["head"], rule["body"])
        if key not in compact:
            compact[key] = {
                **rule,
                "weight": 1.0,
                "rank_weights": [0.0] * group_count,
                "group_id": None,
            }
        rank_weights = compact[key]["rank_weights"]
        group = rule["group_id"]
        if rank_weights[group] != 0:
            raise ValueError("duplicate Horn rule/group instance")
        rank_weights[group] = rule["weight"]
    return [
        {**rule, "rank_weights": tuple(rule["rank_weights"])}
        for rule in compact.values()
    ]


def build_evaluation(triples, num_relations, max_triples=None):
    triples = triples if max_triples is None else triples[:max_triples]
    evaluation = []
    for head, relation, tail in triples:
        evaluation.append((head, relation, tail))
        evaluation.append((tail, relation + num_relations, head))
    return evaluation


def build_compact_rule_csr(train_path, num_entities, device, cache_path=None):
    started = time.time()
    if cache_path is not None and cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        if payload["num_entities"] != num_entities:
            raise ValueError("CSR cache entity count does not match the dataset")
        cpu_csr = payload["csr"]
        gpu_csr = tuple(
            tensor.to(device) if index in CSR_RULE_FIELDS else torch.empty(0, device=device)
            for index, tensor in enumerate(cpu_csr)
        )
        del payload, cpu_csr
        print(
            f"Loaded compact rule CSR cache {cache_path} in {time.time() - started:.1f}s",
            flush=True,
        )
        return gpu_csr

    train = pd.read_csv(
        train_path,
        sep="\t",
        header=None,
        engine="c",
        dtype={0: np.int32, 1: np.int32, 2: np.int32},
    )
    print(f"Loaded {len(train):,} train edges in {time.time() - started:.1f}s", flush=True)
    heads = torch.from_numpy(train[0].values).long()
    relations = torch.from_numpy(train[1].values).long()
    tails = torch.from_numpy(train[2].values).long()
    build_started = time.time()
    cpu_csr = build_csr_structure(heads, tails, relations, num_entities)
    print(f"Built CPU CSR in {time.time() - build_started:.1f}s", flush=True)
    empty_cpu = torch.empty(0)
    compact_cpu_csr = tuple(
        tensor if index in CSR_RULE_FIELDS else empty_cpu
        for index, tensor in enumerate(cpu_csr)
    )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"num_entities": num_entities, "csr": compact_cpu_csr}, cache_path
        )
        print(f"Wrote compact rule CSR cache to {cache_path}", flush=True)
    empty = torch.empty(0, device=device)
    gpu_csr = tuple(
        tensor.to(device) if index in CSR_RULE_FIELDS else empty
        for index, tensor in enumerate(compact_cpu_csr)
    )
    del cpu_csr, compact_cpu_csr, heads, relations, tails, train
    torch.cuda.empty_cache()
    print(
        f"Moved compact rule CSR to {device}; total setup {time.time() - started:.1f}s",
        flush=True,
    )
    return gpu_csr


def initial_rule_state(sources, num_rules, num_entities):
    batch_size = sources.numel()
    batch = torch.arange(batch_size, device=sources.device).repeat_interleave(num_rules)
    level = torch.arange(num_rules, device=sources.device).repeat(batch_size)
    entity = sources.repeat_interleave(num_rules)
    indices = torch.stack((batch, level, entity))
    values = torch.ones(indices.shape[1], device=sources.device)
    return torch.sparse_coo_tensor(
        indices, values, (batch_size, num_rules, num_entities), is_coalesced=True
    )


def advance_single_rule_state_memory_bounded(
    state,
    operators,
    csr,
    num_relations,
    input_chunk_size=64,
):
    """Advance one query/rule without materializing every expanded edge at once.

    The regular fixed-relation kernel expands the whole frontier before it
    coalesces duplicate endpoints.  A few high-degree Freebase paths can make
    that raw expansion larger than GPU memory even though the coalesced result
    is small.  This fallback partitions the input frontier and accumulates each
    coalesced partial result into one dense endpoint vector.  It preserves path
    counts exactly while bounding the temporary raw-edge buffer.
    """
    state = state.coalesce()
    if state.shape[0] != 1 or state.shape[1] != 1:
        raise ValueError("memory-bounded advance requires one query and one rule")
    if tuple(operators.shape) != (1, 1):
        raise ValueError("memory-bounded operators must have shape [1, 1]")
    if state._nnz() == 0:
        return state

    device = state.device
    num_entities = state.shape[2]
    accumulated = torch.zeros(num_entities, device=device, dtype=state.dtype)
    indices = state.indices()
    values = state.values()
    start = 0
    chunk_size = min(input_chunk_size, state._nnz())
    while start < state._nnz():
        end = min(start + chunk_size, state._nnz())
        partial = torch.sparse_coo_tensor(
            indices[:, start:end],
            values[start:end],
            state.shape,
            device=device,
            is_coalesced=True,
        )
        expanded = None
        try:
            expanded = vectorized_operation_sparse3d_fixed_relation(
                partial, operators, csr, num_relations
            ).coalesce()
            accumulated.index_add_(
                0, expanded.indices()[2], expanded.values()
            )
        except torch.cuda.OutOfMemoryError:
            del partial, expanded
            torch.cuda.empty_cache()
            if chunk_size == 1:
                raise
            chunk_size = max(1, chunk_size // 2)
            continue
        del partial, expanded
        start = end

    output_entities = torch.nonzero(accumulated, as_tuple=True)[0]
    output_indices = torch.stack(
        (
            torch.zeros_like(output_entities),
            torch.zeros_like(output_entities),
            output_entities,
        )
    )
    output_values = accumulated[output_entities]
    return torch.sparse_coo_tensor(
        output_indices,
        output_values,
        state.shape,
        device=device,
        is_coalesced=True,
    )


def reduce_rule_state(state, weights, boolean=False):
    state = state.coalesce()
    indices = state.indices()
    values = state.values()
    if boolean:
        values = (values > 0).to(values.dtype)
    values = values * weights[indices[1]]
    reduced = torch.sparse_coo_tensor(
        torch.stack((indices[0], indices[2])),
        values,
        (state.shape[0], state.shape[2]),
    )
    return reduced.coalesce()


def reduce_rule_state_by_rank(state, rank_weights, boolean=False):
    """Aggregate explicit Horn proofs into their DRUM rank components.

    With ``boolean=True``, each rule casts at most one vote for an endpoint,
    regardless of how many graph paths prove that same grounded conclusion.
    """
    state = state.coalesce()
    indices = state.indices()
    values = state.values()
    if boolean:
        values = (values > 0).to(values.dtype)
    num_ranks = rank_weights.shape[1]
    expanded_indices = indices.unsqueeze(2).expand(-1, -1, num_ranks)
    ranks = torch.arange(
        num_ranks, device=indices.device, dtype=indices.dtype
    ).view(1, 1, num_ranks).expand(1, indices.shape[1], -1)
    output_indices = torch.cat(
        (
            expanded_indices[0:1],
            ranks,
            expanded_indices[2:3],
        ),
        dim=0,
    ).reshape(3, -1)
    output_values = (
        values.unsqueeze(1) * rank_weights[indices[1]]
    ).reshape(-1)
    positive = output_values > 0
    return torch.sparse_coo_tensor(
        output_indices[:, positive],
        output_values[positive],
        (state.shape[0], num_ranks, state.shape[2]),
    ).coalesce()


def reduce_rule_state_by_group(state, weights, groups, num_groups):
    """Aggregate each scalar-weighted Horn rule into one routed group."""
    state = state.coalesce()
    indices = state.indices()
    if groups.shape != weights.shape:
        raise ValueError("groups and weights must have the same shape")
    if num_groups <= 0 or torch.any(groups < 0) or torch.any(groups >= num_groups):
        raise ValueError("group IDs must be within the routed group count")
    output_indices = torch.stack(
        (indices[0], groups[indices[1]], indices[2])
    )
    output_values = state.values() * weights[indices[1]]
    positive = output_values > 0
    return torch.sparse_coo_tensor(
        output_indices[:, positive],
        output_values[positive],
        (state.shape[0], num_groups, state.shape[2]),
    ).coalesce()


def normalize_and_sum_rank_scores(scores):
    """Normalize each rank distribution, then sum ranks like DRUM."""
    scores = scores.coalesce()
    indices = scores.indices()
    values = scores.values()
    keys = indices[0] * scores.shape[1] + indices[1]
    totals = torch.zeros(
        scores.shape[0] * scores.shape[1],
        device=values.device,
        dtype=values.dtype,
    )
    totals.scatter_add_(0, keys, values)
    normalized = values / totals[keys].clamp_min(1e-30)
    return torch.sparse_coo_tensor(
        torch.stack((indices[0], indices[2])),
        normalized,
        (scores.shape[0], scores.shape[2]),
    ).coalesce()


def normalize_sparse_rows(scores):
    """Normalize each sparse query row independently."""
    scores = scores.coalesce()
    indices = scores.indices()
    values = scores.values()
    totals = torch.zeros(
        scores.shape[0], device=values.device, dtype=values.dtype
    )
    totals.scatter_add_(0, indices[0], values)
    normalized = values / totals[indices[0]].clamp_min(1e-30)
    return torch.sparse_coo_tensor(
        indices, normalized, scores.shape, is_coalesced=True
    )


def add_sparse(left, right):
    if left is None:
        return right
    left = left.coalesce()
    right = right.coalesce()
    return torch.sparse_coo_tensor(
        torch.cat((left.indices(), right.indices()), dim=1),
        torch.cat((left.values(), right.values())),
        left.shape,
    ).coalesce()


def max_sparse(left, right):
    if left is None:
        return right
    left = left.coalesce()
    right = right.coalesce()
    indices = torch.cat((left.indices(), right.indices()), dim=1)
    values = torch.cat((left.values(), right.values()))
    keys = indices[0] * left.shape[1] + indices[1]
    unique_keys, inverse = torch.unique(keys, sorted=True, return_inverse=True)
    reduced_values = torch.full(
        (unique_keys.numel(),),
        -torch.inf,
        device=values.device,
        dtype=values.dtype,
    )
    reduced_values.scatter_reduce_(
        0, inverse, values, reduce="amax", include_self=True
    )
    reduced_indices = torch.stack(
        (unique_keys // left.shape[1], unique_keys % left.shape[1])
    )
    return torch.sparse_coo_tensor(
        reduced_indices,
        reduced_values,
        left.shape,
        is_coalesced=True,
    )


def max_reduce_rule_state(state, weights):
    state = state.coalesce()
    indices = state.indices()
    values = (state.values() > 0).to(state.values().dtype)
    values = values * weights[indices[1]]
    keys = indices[0] * state.shape[2] + indices[2]
    unique_keys, inverse = torch.unique(keys, sorted=True, return_inverse=True)
    reduced_values = torch.full(
        (unique_keys.numel(),),
        -torch.inf,
        device=values.device,
        dtype=values.dtype,
    )
    reduced_values.scatter_reduce_(
        0, inverse, values, reduce="amax", include_self=True
    )
    reduced_indices = torch.stack(
        (unique_keys // state.shape[2], unique_keys % state.shape[2])
    )
    return torch.sparse_coo_tensor(
        reduced_indices,
        reduced_values,
        (state.shape[0], state.shape[2]),
        is_coalesced=True,
    )


def _score_rank_batch_serial(
    sources,
    rules,
    csr,
    num_relations,
    num_entities,
    rule_chunk_size,
    modes,
):
    """Share each streamed rank grounding across path and Boolean modes."""
    if any(rule["rank_weights"] is None for rule in rules):
        raise ValueError("rank aggregation requires a rank_weights column")
    widths = {len(rule["rank_weights"]) for rule in rules}
    if len(widths) != 1:
        raise ValueError("rank_weights widths must match")
    rank_count = next(iter(widths))
    requested = tuple(
        "weighted_boolean" if boolean else "weighted_path_count"
        for boolean in modes.values()
    )
    requested = tuple(dict.fromkeys(requested))
    results = {name: None for name in modes}
    for rank in range(rank_count):
        ranked_rules = [
            {**rule, "weight": rule["rank_weights"][rank]}
            for rule in rules
            if rule["rank_weights"][rank] > 0
        ]
        rank_scores = score_rule_batch(
            sources,
            ranked_rules,
            csr,
            num_relations,
            num_entities,
            rule_chunk_size,
            requested,
        )
        for name, boolean in modes.items():
            aggregation = (
                "weighted_boolean" if boolean else "weighted_path_count"
            )
            results[name] = add_sparse(
                results[name],
                normalize_sparse_rows(rank_scores[aggregation]),
            )
    return results


_SHARED_CHUNK_LIMITS = {}


def _ground_shared_rules(sources, chunk, csr, relations, entities):
    """Compute unweighted path counts once for all channel coefficients."""
    identity = 2 * relations
    steps = max(len(r['body']) for r in chunk)
    paths = [r['body'] + (identity,) * (steps - len(r['body'])) for r in chunk]
    operators = torch.tensor(paths, device=sources.device, dtype=torch.long)
    state = initial_rule_state(sources, len(chunk), entities)
    for step in range(steps):
        if state._nnz() == 0:
            break
        ops = operators[:, step].unsqueeze(0).expand(sources.numel(), -1)
        try:
            state = vectorized_operation_sparse3d_fixed_relation(state, ops, csr, relations)
        except torch.cuda.OutOfMemoryError:
            if len(chunk) != 1 or sources.numel() != 1:
                raise
            torch.cuda.empty_cache()
            state = advance_single_rule_state_memory_bounded(state, ops, csr, relations)
    return state


@torch.inference_mode()
def _score_rank_batch_shared(sources, rules, csr, num_relations, num_entities, rule_chunk_size, modes):
    """Stream shared path counts, then normalize each channel after all rules."""
    from rules.shared_inference import score
    return {name:score(sys.modules[__name__],sources,rules,csr,num_relations,num_entities,
                       rule_chunk_size,reduction='sum',boolean_paths=boolean)
            for name,boolean in modes.items()}


_SERIAL_RANK_HEADS = set()


@torch.inference_mode()
def score_rank_batch_streamed_modes(sources, rules, csr, num_relations,
                                    num_entities, rule_chunk_size, modes):
    """Use shared grounding when memory permits; retain the serial fallback."""
    key = (id(csr), str(sources.device), int(sources.numel()), rules[0].get('head'))
    if key not in _SERIAL_RANK_HEADS:
        try:
            return _score_rank_batch_shared(
                sources, rules, csr, num_relations, num_entities, rule_chunk_size, modes
            )
        except torch.cuda.OutOfMemoryError:
            pass
        # Release the failed shared execution before retrying all rules serially.
        torch.cuda.empty_cache()
        _SERIAL_RANK_HEADS.add(key)
    return _score_rank_batch_serial(
        sources, rules, csr, num_relations, num_entities, rule_chunk_size, modes
    )


@torch.inference_mode()
def score_rule_batch(
    sources,
    rules,
    csr,
    num_relations,
    num_entities,
    rule_chunk_size,
    aggregations,
):
    requested_aggregations = tuple(aggregations)
    if not rules:
        empty_indices = torch.empty(
            (2, 0), device=sources.device, dtype=torch.long
        )
        empty_values = torch.empty(0, device=sources.device)
        shape = (sources.numel(), num_entities)
        empty = torch.sparse_coo_tensor(
            empty_indices, empty_values, shape, is_coalesced=True
        )
        return {name: empty for name in requested_aggregations}

    if requested_aggregations == ("routed_normalized",):
        compact_rules = compact_routed_rules(rules)
        compact_scores = score_rule_batch(
            sources,
            compact_rules,
            csr,
            num_relations,
            num_entities,
            rule_chunk_size,
            ("rank_normalized",),
        )
        return {"routed_normalized": compact_scores["rank_normalized"]}

    streamed_scores = {}
    if len(rules) > RANK_STREAMING_RULE_THRESHOLD:
        rank_modes = {
            name: boolean
            for name, boolean in (
                ("rank_normalized", False),
                ("rank_boolean_normalized", True),
            )
            if name in requested_aggregations
        }
        if rank_modes:
            streamed_scores.update(
                score_rank_batch_streamed_modes(
                    sources,
                    rules,
                    csr,
                    num_relations,
                    num_entities,
                    rule_chunk_size,
                    rank_modes,
                )
            )
        aggregations = tuple(
            name
            for name in requested_aggregations
            if name not in streamed_scores
        )
        if not aggregations:
            return {
                name: streamed_scores[name].coalesce()
                for name in requested_aggregations
            }

    identity = 2 * num_relations
    scores = {name: None for name in aggregations}
    routed_scores = None
    routed_group_count = None
    if "routed_normalized" in scores:
        if any(rule["group_id"] is None for rule in rules):
            raise ValueError(
                "routed_normalized requires a group_id column"
            )
        routed_group_count = max(rule["group_id"] for rule in rules) + 1
    rank_scores = {
        name: None
        for name in (
            "rank_normalized",
            "rank_boolean_normalized",
        )
        if name in scores
    }
    for start in range(0, len(rules), rule_chunk_size):
        chunk = rules[start : start + rule_chunk_size]
        steps = max(len(rule["body"]) for rule in chunk)
        paths = [rule["body"] + (identity,) * (steps - len(rule["body"])) for rule in chunk]
        operators = torch.tensor(paths, device=sources.device, dtype=torch.long)
        weights = torch.tensor(
            [rule["weight"] for rule in chunk], device=sources.device
        )
        state = initial_rule_state(sources, len(chunk), num_entities)
        for step in range(steps):
            step_operators = operators[:, step].unsqueeze(0).expand(sources.numel(), -1)
            input_state = state
            try:
                state = vectorized_operation_sparse3d_fixed_relation(
                    input_state, step_operators, csr, num_relations
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if sources.numel() != 1 or len(chunk) != 1:
                    raise
                rule = chunk[0]
                print(
                    "Memory-bounded exact fallback: "
                    f"head={rule['head']}, rule_id={rule['rule_id']}, "
                    f"body={' '.join(map(str, rule['body']))}, step={step + 1}",
                    flush=True,
                )
                state = advance_single_rule_state_memory_bounded(
                    input_state,
                    step_operators,
                    csr,
                    num_relations,
                )
        if "weighted_path_count" in scores:
            scores["weighted_path_count"] = add_sparse(
                scores["weighted_path_count"],
                reduce_rule_state(state, weights),
            )
        if "weighted_boolean" in scores:
            scores["weighted_boolean"] = add_sparse(
                scores["weighted_boolean"],
                reduce_rule_state(state, weights, boolean=True),
            )
        if "routed_normalized" in scores:
            groups = torch.tensor(
                [rule["group_id"] for rule in chunk],
                device=sources.device,
                dtype=torch.long,
            )
            routed_scores = add_sparse(
                routed_scores,
                reduce_rule_state_by_group(
                    state, weights, groups, routed_group_count
                ),
            )
        if rank_scores:
            if any(rule["rank_weights"] is None for rule in chunk):
                raise ValueError(
                    "rank aggregation requires a rank_weights column"
                )
            widths = {len(rule["rank_weights"]) for rule in chunk}
            if len(widths) != 1:
                raise ValueError("rank_weights widths must match")
            rank_weights = torch.tensor(
                [rule["rank_weights"] for rule in chunk],
                device=sources.device,
                dtype=state.dtype,
            )
            for name, boolean in (
                ("rank_normalized", False),
                ("rank_boolean_normalized", True),
            ):
                if name in rank_scores:
                    component_weights = rank_weights
                    rank_scores[name] = add_sparse(
                        rank_scores[name],
                        reduce_rule_state_by_rank(
                            state, component_weights, boolean=boolean
                        ),
                    )
        if "max_confidence" in scores:
            scores["max_confidence"] = max_sparse(
                scores["max_confidence"],
                max_reduce_rule_state(state, weights),
            )

    for name, ranked in rank_scores.items():
        scores[name] = normalize_and_sum_rank_scores(ranked)
    if "routed_normalized" in scores:
        scores["routed_normalized"] = normalize_and_sum_rank_scores(
            routed_scores
        )
    scores.update(streamed_scores)
    return {
        name: scores[name].coalesce() for name in requested_aggregations
    }


def build_split_known(valid, test, num_relations):
    known = defaultdict(set)
    for head, relation, tail in valid + test:
        known[(head, relation)].add(tail)
        known[(tail, relation + num_relations)].add(head)
    return known


@torch.inference_mode()
def filter_keys_for_batch_flashlog(
    csr,
    batch,
    sources,
    split_known,
    num_relations,
    num_entities,
    include_targets=True,
):
    """Build filtered-answer keys from train CSR plus supplemental truths."""
    key_parts = []
    extra_keys = []
    for batch_index, (source, relation, target) in enumerate(batch):
        # Some Freebase relations have exceptionally high degree.  Expanding
        # filter adjacency one source at a time bounds the temporary raw-edge
        # buffer without changing the final compact filter keys.
        state = initial_rule_state(sources[batch_index : batch_index + 1], 1, num_entities)
        operators = torch.tensor(
            [[relation]], device=sources.device, dtype=torch.long
        )
        state = vectorized_operation_sparse3d_fixed_relation(
            state, operators, csr, num_relations
        ).coalesce()
        key_parts.append(batch_index * num_entities + state.indices()[2])
        answers = set(split_known.get((source, relation), set()))
        if include_targets:
            answers.add(target)
        extra_keys.extend(
            batch_index * num_entities + entity for entity in answers
        )
    if extra_keys:
        key_parts.append(
            torch.tensor(extra_keys, device=sources.device, dtype=torch.long)
        )
    return torch.unique(torch.cat(key_parts), sorted=True)


def target_coverage(scores, targets):
    scores = scores.coalesce()
    indices = scores.indices()
    keys = indices[0] * scores.shape[1] + indices[1]
    target_keys = torch.arange(
        scores.shape[0], device=targets.device, dtype=torch.long
    ) * scores.shape[1] + targets
    if keys.numel() == 0:
        return 0
    keys = torch.sort(keys).values
    positions = torch.searchsorted(keys, target_keys)
    safe = positions.clamp(max=keys.numel() - 1)
    return int(((positions < keys.numel()) & (keys[safe] == target_keys)).sum().item())


def select_sparse_score_row(scores, row):
    """Return one sparse score row with its batch coordinate rebased to zero."""
    scores = scores.coalesce()
    indices = scores.indices()
    selected = indices[0] == row
    row_indices = torch.stack(
        (
            torch.zeros_like(indices[1, selected]),
            indices[1, selected],
        )
    )
    return torch.sparse_coo_tensor(
        row_indices,
        scores.values()[selected],
        (1, scores.shape[1]),
        is_coalesced=True,
    )


def update_metrics(accumulator, scores, targets, filter_keys):
    scores = scores.coalesce()
    ranks = sparse2d_filtered_ranks_cuda(
        scores.indices()[0],
        scores.indices()[1],
        scores.values(),
        targets,
        filter_keys,
        scores.shape[0],
        scores.shape[1],
        use_eql=True,
    )
    ranks = ranks.detach().cpu().numpy().astype(np.float64)
    accumulator["ranks"].extend(ranks.tolist())
    accumulator["covered_targets"] += target_coverage(scores, targets)


def finalize_metrics(accumulator):
    ranks = np.asarray(accumulator["ranks"], dtype=np.float64)
    if not ranks.size:
        return {
            "count": 0,
            "mrr": 0.0,
            "hit_1": 0.0,
            "hit_3": 0.0,
            "hit_10": 0.0,
            "covered_targets": int(accumulator["covered_targets"]),
            "target_coverage": 0.0,
        }
    return {
        "count": int(ranks.size),
        "mrr": float(np.mean(1 / ranks)),
        "hit_1": float(np.mean(ranks <= 1)),
        "hit_3": float(np.mean(ranks <= 3)),
        "hit_10": float(np.mean(ranks <= 10)),
        "covered_targets": accumulator["covered_targets"],
        "target_coverage": accumulator["covered_targets"] / int(ranks.size),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--rule-file",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--csr-cache",
        type=Path,
        required=True,
    )
    parser.add_argument("--eval-split", choices=("valid", "test"), default="test")
    parser.add_argument(
        "--query-head-file",
        type=Path,
        help=(
            "optional directed relation IDs to evaluate; this restricts only "
            "post-extraction evaluation queries"
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--query-batch-size", type=int, default=4)
    parser.add_argument("--rule-chunk-size", type=int, default=16)
    parser.add_argument("--indexed-rules", action="store_true",
                        help="Load routed TSV rules by relation using a reusable byte index")
    parser.add_argument("--weight-field", default="weight")
    parser.add_argument(
        "--aggregations",
        default="routed_normalized",
        help=(
            'comma-separated subset of routed_normalized, rank_normalized, rank_boolean_normalized, weighted_path_count, weighted_boolean, max_confidence'
        ),
    )
    parser.add_argument("--max-triples", type=int)
    parser.add_argument("--relation-index-start", type=int, default=0)
    parser.add_argument("--relation-index-end", type=int)
    parser.add_argument("--query-index-start", type=int, default=0)
    parser.add_argument("--query-index-end", type=int)
    parser.add_argument(
        "--group-query-sources",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "score each distinct (relation, source) only once and reuse the "
            "scores for all of its targets; filtered ranks remain exact"
        ),
    )
    parser.add_argument("--source-index-start", type=int, default=0)
    parser.add_argument("--source-index-end", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    started = time.time()
    device = torch.device(args.device)
    num_relations = dictionary_size(args.data_dir / "relations.dict")
    num_entities = dictionary_size(args.data_dir / "entities.dict")
    valid = load_triples(args.data_dir / "valid_id.txt")
    test = load_triples(args.data_dir / "test_id.txt")
    triples = valid if args.eval_split == "valid" else test
    evaluation_label = args.eval_split
    split_known = build_split_known(valid, test, num_relations)
    evaluation = build_evaluation(triples, num_relations, args.max_triples)
    if args.query_head_file is not None:
        allowed_query_heads = {
            int(line.strip())
            for line in args.query_head_file.read_text().splitlines()
            if line.strip()
        }
        evaluation = [
            query for query in evaluation if query[1] in allowed_query_heads
        ]
    aggregations = tuple(
        name.strip() for name in args.aggregations.split(",") if name.strip()
    )
    allowed_aggregations = frozenset(
        (
            "weighted_path_count",
            "weighted_boolean",
            "routed_normalized",
            "rank_normalized",
            "rank_boolean_normalized",
            "max_confidence",
        )
    )
    if not aggregations or not set(aggregations) <= allowed_aggregations:
        raise ValueError(
            "--aggregations must contain only: "
            + ", ".join(sorted(allowed_aggregations))
        )
    metric_names = aggregations
    indexed_bank = None
    if args.indexed_rules:
        if aggregations != ("routed_normalized",):
            raise ValueError("Indexed TSV loading requires routed_normalized only")
        indexed_bank = IndexedRuleBank(args.rule_file, args.output_dir / "rules.head-index.json", args.weight_field)
    rules = [] if indexed_bank is not None else load_rules(args.rule_file, args.weight_field)

    queries_by_relation = defaultdict(list)
    for query in evaluation:
        queries_by_relation[query[1]].append(query)
    rules_by_relation = indexed_bank if indexed_bank is not None else defaultdict(list)
    for rule in rules:
        if rule["head"] in queries_by_relation:
            rules_by_relation[rule["head"]].append(rule)

    csr = build_compact_rule_csr(
        args.data_dir / "train_id.txt", num_entities, device, args.csr_cache
    )
    accumulators = {
        name: {"ranks": [], "covered_targets": 0} for name in metric_names
    }
    relation_accumulators = {
        relation: {
            name: {"ranks": [], "covered_targets": 0}
            for name in metric_names
        }
        for relation in queries_by_relation
    }
    scored_queries = 0
    all_relation_items = sorted(queries_by_relation.items())
    relation_end = (
        len(all_relation_items)
        if args.relation_index_end is None
        else args.relation_index_end
    )
    relation_items = all_relation_items[args.relation_index_start : relation_end]
    if args.query_index_start < 0:
        raise ValueError("--query-index-start must be non-negative")
    if args.query_index_end is not None and args.query_index_end < args.query_index_start:
        raise ValueError("--query-index-end must not precede --query-index-start")
    relation_items = [
        (
            relation,
            queries[args.query_index_start : args.query_index_end],
        )
        for relation, queries in relation_items
    ]
    if args.source_index_start < 0:
        raise ValueError("--source-index-start must be non-negative")
    if (
        args.source_index_end is not None
        and args.source_index_end < args.source_index_start
    ):
        raise ValueError("--source-index-end must not precede --source-index-start")
    if (
        args.source_index_start != 0 or args.source_index_end is not None
    ) and not args.group_query_sources:
        raise ValueError("source-index slicing requires --group-query-sources")
    if args.group_query_sources:
        source_sliced_items = []
        for relation, queries in relation_items:
            queries_by_source = {}
            for query in queries:
                queries_by_source.setdefault(query[0], []).append(query)
            source_groups = list(queries_by_source.values())[
                args.source_index_start : args.source_index_end
            ]
            source_sliced_items.append(
                (relation, [query for group in source_groups for query in group])
            )
        relation_items = source_sliced_items
    selected_query_count = sum(len(queries) for _, queries in relation_items)
    inference_started = time.time()
    for relation_index, (relation, queries) in enumerate(relation_items, start=1):
        relation_rules = rules_by_relation.get(relation, [])
        relation_aggregations = aggregations
        aggregation_to_metric = {name: name for name in aggregations}

        def evaluate_batch(batch, rule_chunk_size):
            nonlocal scored_queries
            sources = torch.tensor(
                [query[0] for query in batch], device=device, dtype=torch.long
            )
            targets = torch.tensor(
                [query[2] for query in batch], device=device, dtype=torch.long
            )
            filter_keys = None
            score_by_aggregation = None
            out_of_memory = False
            try:
                # Build one-hop train filters before materializing the much
                # larger rule score tensors, so their peaks do not overlap.
                filter_keys = filter_keys_for_batch_flashlog(
                    csr,
                    batch,
                    sources,
                    split_known,
                    num_relations,
                    num_entities,
                )
                score_by_aggregation = score_rule_batch(
                    sources,
                    relation_rules,
                    csr,
                    num_relations,
                    num_entities,
                    rule_chunk_size,
                    relation_aggregations,
                )
            except torch.cuda.OutOfMemoryError:
                out_of_memory = True
            if out_of_memory:
                del sources, targets, filter_keys, score_by_aggregation
                torch.cuda.empty_cache()
                if len(batch) > 1:
                    print(
                        f"OOM fallback: relation={relation}, "
                        f"query_batch={len(batch)} -> single queries",
                        flush=True,
                    )
                    middle = len(batch) // 2
                    evaluate_batch(batch[:middle], rule_chunk_size)
                    evaluate_batch(batch[middle:], rule_chunk_size)
                    return
                if rule_chunk_size > 1:
                    print(
                        f"OOM fallback: relation={relation}, "
                        f"rule_chunk={rule_chunk_size} -> "
                        f"{max(1, rule_chunk_size // 2)}",
                        flush=True,
                    )
                    evaluate_batch(batch, max(1, rule_chunk_size // 2))
                    return
                raise torch.cuda.OutOfMemoryError(
                    f"relation {relation} failed at batch=1, rule_chunk=1"
                )
            for aggregation, scores in score_by_aggregation.items():
                name = aggregation_to_metric[aggregation]
                update_metrics(
                    accumulators[name],
                    scores,
                    targets,
                    filter_keys,
                )
                update_metrics(
                    relation_accumulators[relation][name],
                    scores,
                    targets,
                    filter_keys,
                )
            scored_queries += len(batch)
            del score_by_aggregation, filter_keys

        source_batch_limit = args.query_batch_size

        def evaluate_source_groups(source_groups, rule_chunk_size):
            """Ground unique sources; retain smaller batches after an OOM."""
            nonlocal scored_queries, source_batch_limit
            if len(source_groups) > source_batch_limit:
                limit = source_batch_limit
                for offset in range(0, len(source_groups), limit):
                    evaluate_source_groups(
                        source_groups[offset:offset + limit], rule_chunk_size
                    )
                return
            representatives = [source_group[0] for source_group in source_groups]
            sources = torch.tensor(
                [query[0] for query in representatives],
                device=device,
                dtype=torch.long,
            )
            base_filter_keys = None
            score_by_aggregation = None
            out_of_memory = False
            try:
                # Train and split-known filters depend only on relation/source.
                # Add the current target below to preserve per-query filtering.
                base_filter_keys = filter_keys_for_batch_flashlog(
                    csr,
                    representatives,
                    sources,
                    split_known,
                    num_relations,
                    num_entities,
                    include_targets=False,
                )
                score_by_aggregation = score_rule_batch(
                    sources,
                    relation_rules,
                    csr,
                    num_relations,
                    num_entities,
                    rule_chunk_size,
                    relation_aggregations,
                )
            except torch.cuda.OutOfMemoryError:
                out_of_memory = True
            if out_of_memory:
                del sources, base_filter_keys, score_by_aggregation
                torch.cuda.empty_cache()
                if len(source_groups) > 1:
                    print(
                        f"OOM fallback: relation={relation}, "
                        f"source_batch={len(source_groups)} -> split",
                        flush=True,
                    )
                    middle = len(source_groups) // 2
                    source_batch_limit = min(source_batch_limit, max(1, middle))
                    evaluate_source_groups(
                        source_groups[:middle], rule_chunk_size
                    )
                    evaluate_source_groups(
                        source_groups[middle:], rule_chunk_size
                    )
                    return
                if rule_chunk_size > 1:
                    print(
                        f"OOM fallback: relation={relation}, "
                        f"rule_chunk={rule_chunk_size} -> "
                        f"{max(1, rule_chunk_size // 2)}",
                        flush=True,
                    )
                    evaluate_source_groups(
                        source_groups, max(1, rule_chunk_size // 2)
                    )
                    return
                raise torch.cuda.OutOfMemoryError(
                    f"relation {relation} failed at source-group, rule_chunk=1"
                )

            for source_index, source_group in enumerate(source_groups):
                source_filter_keys = base_filter_keys[
                    torch.div(
                        base_filter_keys,
                        num_entities,
                        rounding_mode="floor",
                    )
                    == source_index
                ] % num_entities
                source_scores = {
                    aggregation: select_sparse_score_row(scores, source_index)
                    for aggregation, scores in score_by_aggregation.items()
                }
                for query in source_group:
                    targets = torch.tensor(
                        [query[2]], device=device, dtype=torch.long
                    )
                    filter_keys = torch.unique(
                        torch.cat((source_filter_keys, targets)), sorted=True
                    )
                    for aggregation, scores in source_scores.items():
                        name = aggregation_to_metric[aggregation]
                        update_metrics(
                            accumulators[name], scores, targets, filter_keys
                        )
                        update_metrics(
                            relation_accumulators[relation][name],
                            scores,
                            targets,
                            filter_keys,
                        )
            scored_queries += sum(len(group) for group in source_groups)
            del score_by_aggregation, base_filter_keys

        if args.group_query_sources:
            queries_by_source = {}
            for query in queries:
                queries_by_source.setdefault(query[0], []).append(query)
            source_groups = list(queries_by_source.values())
            for start in range(0, len(source_groups), args.query_batch_size):
                evaluate_source_groups(
                    source_groups[start : start + args.query_batch_size],
                    args.rule_chunk_size,
                )
        else:
            for start in range(0, len(queries), args.query_batch_size):
                evaluate_batch(
                    queries[start : start + args.query_batch_size],
                    args.rule_chunk_size,
                )
        if relation_index % 25 == 0 or relation_index == len(relation_items):
            print(
                f"Evaluated {relation_index}/{len(relation_items)} relations; "
                f"queries={scored_queries}/{selected_query_count}",
                flush=True,
            )

    metrics = {
        name: finalize_metrics(accumulator)
        for name, accumulator in accumulators.items()
    }
    per_relation_metrics = {
        str(relation): {
            name: finalize_metrics(accumulator)
            for name, accumulator in relation_accumulators[relation].items()
        }
        for relation, _ in relation_items
    }
    payload = {
        "dataset": args.data_dir.resolve().name,
        "evaluation_split": evaluation_label,
        "query_head_file": (
            str(args.query_head_file.resolve())
            if args.query_head_file is not None
            else None
        ),
        "validation_or_test_labels_read": True,
        "rule_file": str(args.rule_file.resolve()),
        "inference_engine": "FlashLog fixed-relation sparse CSR CUDA",
        "protocol": (
            "rules execute over train edges only; bidirectional filtered ranking "
            "over all entities with tie-aware average ranks"
        ),
        "evaluation_triples": len(evaluation) // 2,
        "evaluation_queries": selected_query_count,
        "query_relations": len(relation_items),
        "relation_index_start": args.relation_index_start,
        "relation_index_end": relation_end,
        "total_query_relations": len(all_relation_items),
        "query_index_start": args.query_index_start,
        "query_index_end": args.query_index_end,
        "group_query_sources": args.group_query_sources,
        "source_index_start": args.source_index_start,
        "source_index_end": args.source_index_end,
        "rules_used": sum(
            len(rules_by_relation.get(relation, [])) for relation, _ in relation_items
        ),
        "weight_field": args.weight_field,
        "aggregations": metric_names,
        "requested_aggregations": aggregations,
        "query_batch_size": args.query_batch_size,
        "rule_chunk_size": args.rule_chunk_size,
        "setup_seconds": inference_started - started,
        "inference_seconds": time.time() - inference_started,
        "rule_index_seconds": indexed_bank.index_seconds if indexed_bank is not None else 0.0,
        "rule_bank_materialization_seconds": indexed_bank.materialization_seconds if indexed_bank is not None else 0.0,
        "rule_bank_loaded_heads": indexed_bank.loaded_heads if indexed_bank is not None else None,
        "metrics": metrics,
        "per_relation_metrics": per_relation_metrics,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shard_suffix = ""
    if args.relation_index_start != 0 or relation_end != len(all_relation_items):
        shard_suffix = f"_relations_{args.relation_index_start}_{relation_end}"
    if args.query_index_start != 0 or args.query_index_end is not None:
        query_end = "end" if args.query_index_end is None else args.query_index_end
        shard_suffix += f"_queries_{args.query_index_start}_{query_end}"
    if args.source_index_start != 0 or args.source_index_end is not None:
        source_end = (
            "end" if args.source_index_end is None else args.source_index_end
        )
        shard_suffix += f"_sources_{args.source_index_start}_{source_end}"
    output_path = args.output_dir / (
        f"{evaluation_label}_metrics_flashlog{shard_suffix}.json"
    )
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
