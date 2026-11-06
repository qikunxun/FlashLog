"""Lazy loader for the compiled Train-only soft-guided grounding kernel."""

from pathlib import Path

import math
import hashlib

import torch
from torch.utils import cpp_extension


_MODULES = {}


def _module(temperature=1.0, steps=3):
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("sampling temperature must be finite and positive")
    steps = int(steps)
    if steps <= 0:
        raise ValueError("rule steps must be positive")
    key = (temperature, steps)
    if key not in _MODULES:
        source = Path(__file__).with_name("source") / "soft_grounding.cpp"
        suffix = hashlib.sha256(repr(key).encode()).hexdigest()[:12]
        flags = ["-O3", "-std=c++17", f"-DFLASHLOG_SAMPLING_TEMPERATURE={temperature!r}", f"-DFLASHLOG_RULE_STEPS={steps}"]
        if torch.backends.openmp.is_available():
            flags.append("-fopenmp")
        _MODULES[key] = cpp_extension.load(
            name=f"flashlog_grounding_{suffix}", sources=[str(source)],
            extra_cflags=flags, verbose=False,
        )
    return _MODULES[key]


def ground_head_batch_source_folds_sampled_capped_counts(
    attentions,
    heads,
    num_relations,
    source_folds,
    csr,
    walks_per_source,
    topk_edges,
    mask_query_head_from_source,
    seeds,
    transition_cache_size,
    num_threads,
    candidates_per_length,
    sampling_temperature=1.0,
):
    """Return ordered candidates, counts, and attributed C++ wall times.

    Shared head groups apportion their wall time by source count; those
    contributions are not independently measured concurrent head latencies.
    """
    budgets = tuple(map(int, candidates_per_length))
    attentions = torch.as_tensor(
        attentions, dtype=torch.float32, device="cpu"
    ).contiguous()
    heads = list(map(int, heads))
    seeds = list(map(int, seeds))
    folds = [[
        torch.as_tensor(sources, dtype=torch.int64, device="cpu").contiguous()
        for sources in head_folds
    ] for head_folds in source_folds]
    if attentions.ndim != 4 or attentions.shape[1] <= 0:
        raise ValueError("attentions must have shape [heads, steps, rank, operators+1]")
    if len(budgets) != attentions.shape[1] or any(b < 0 for b in budgets):
        raise ValueError("candidates_per_length requires one nonnegative budget per step")
    if not (len(heads) == len(seeds) == len(folds) == len(attentions)):
        raise ValueError("attentions, heads, source_folds and seeds must have equal lengths")
    if any(not head_folds or any(sources.ndim != 1 for sources in head_folds)
           for head_folds in folds):
        raise ValueError("each head must have nonempty folds of one-dimensional vectors")
    required = (1, 5, 8, 9, 10, 11, 12, 13, 14, 15)
    tensors = [csr[index].cpu().contiguous() for index in required]
    candidates, raw_counts, seconds = (
        _module(sampling_temperature, attentions.shape[1]).ground_head_batch_source_folds_sampled_capped_counts(
            attentions, heads, int(num_relations), folds, *tensors,
            int(walks_per_source), int(topk_edges), bool(mask_query_head_from_source),
            seeds, int(transition_cache_size), int(num_threads), list(budgets),
        )
    )
    return candidates, [tuple(map(int, counts.tolist())) for counts in raw_counts], seconds.tolist()


def sample_relation_sources_reservoir(
    row_group_ptr, group_relations, relations, max_sources, seed
):
    """Uniformly sample source rows per relation in one linear CSR pass."""
    row_group_ptr = torch.as_tensor(
        row_group_ptr, dtype=torch.int32, device="cpu"
    ).contiguous()
    group_relations = torch.as_tensor(
        group_relations, dtype=torch.int16, device="cpu"
    ).contiguous()
    relations = torch.as_tensor(
        relations, dtype=torch.int64, device="cpu"
    ).contiguous()
    if row_group_ptr.ndim != 1 or group_relations.ndim != 1:
        raise ValueError("CSR group metadata must be one-dimensional")
    if relations.ndim != 1:
        raise ValueError("relations must be a vector")
    if max_sources <= 0:
        raise ValueError("max_sources must be positive")
    return _module().sample_relation_sources_reservoir(
        row_group_ptr,
        group_relations,
        relations,
        int(max_sources),
        int(seed),
    )
