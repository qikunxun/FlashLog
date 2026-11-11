#!/usr/bin/env python3
"""Recover a compact Horn bank from one frozen soft checkpoint and Train.

Candidate relation chains are sampled from the frozen soft transition policy
over the Train graph.  Candidates from disjoint source folds are combined by union.
The same compiled walk records Train-target endpoint hits, so candidates with
no positive grounding can be discarded before rank-weight annotation.  No
validation/test labels, reference bank, or soft-model training are used.
Detailed output weights are sums of analytical attention rank weights;
Train precision is diagnostic only and does not set inference coefficients.
"""

import argparse
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "rules"))

from rules.evaluate import dictionary_size  # noqa: E402
from rules.common import attention_batch, instantiate_model, read_heads, sha256
from flashlog_kernel.soft_grounding_kernel import (
    ground_head_batch_source_folds_sampled_capped_counts,
)


FIELDS = (
    "head_id",
    "body_ids",
    "length",
    "weight",
    "walk_hits",
    "source_hits",
    "positive_walk_hits",
    "positive_source_hits",
    "minimum_fold_walk_hits",
    "minimum_fold_source_hits",
    "minimum_fold_positive_walk_hits",
    "minimum_fold_positive_source_hits",
    "origin",
    "rank_weights",
    "sampled_train_precision",
)
SCALAR_FIELDS = (
    "head_id",
    "body_ids",
    "length",
    "weight",
    "group_id",
    "origin",
)


def _three_step_weights(attention, rows, identity):
    """Vectorized equivalent of collapsed_attention_weights_by_rank."""
    attention = np.asarray(attention)
    weights = np.zeros((rows.shape[0], attention.shape[1]), dtype=np.float64)
    if rows.shape[0] == 0:
        return weights
    identity_weights = attention[:, :, identity].astype(np.float64, copy=False)

    def values(step, operators):
        return attention[step, :, operators].astype(np.float64, copy=False)

    indices = np.flatnonzero(rows[:, 0] == 1)
    if indices.size:
        operator = rows[indices, 1]
        weights[indices] = (
            values(0, operator) * identity_weights[1] * identity_weights[2]
            + identity_weights[0] * values(1, operator) * identity_weights[2]
            + identity_weights[0] * identity_weights[1] * values(2, operator)
        )

    indices = np.flatnonzero(rows[:, 0] == 2)
    if indices.size:
        first = rows[indices, 1]
        second = rows[indices, 2]
        weights[indices] = (
            values(0, first) * values(1, second) * identity_weights[2]
            + values(0, first) * identity_weights[1] * values(2, second)
            + identity_weights[0] * values(1, first) * values(2, second)
        )

    indices = np.flatnonzero(rows[:, 0] == 3)
    if indices.size:
        weights[indices] = (
            values(0, rows[indices, 1])
            * values(1, rows[indices, 2])
            * values(2, rows[indices, 3])
        )
    return weights


def collapsed_rule_weights(attention, rows, identity, maximum=False):
    """Collapse identity placements with a SUM or MAX dynamic program."""
    attention = np.asarray(attention, dtype=np.float64)
    steps, channels, _ = attention.shape
    result = np.zeros((len(rows), channels), dtype=np.float64)
    for length in range(1, steps + 1):
        indices = np.flatnonzero(rows[:, 0] == length)
        # Bound temporary memory for large rule banks and long bodies.
        for start in range(0, len(indices), 4096):
            batch = indices[start:start + 4096]
            operators = rows[batch, 1:1 + length]
            dp = np.zeros((len(batch), length + 1, channels), dtype=np.float64)
            dp[:, 0] = 1.0
            for step in range(steps):
                next_dp = dp * attention[step, :, identity]
                for position in range(1, min(length, step + 1) + 1):
                    contribution = dp[:, position - 1] * attention[step, :, operators[:, position - 1]]
                    if maximum:
                        np.maximum(next_dp[:, position], contribution, out=next_dp[:, position])
                    else:
                        next_dp[:, position] += contribution
                dp = next_dp
            result[batch] = dp[:, length]
    return result


def attention_rank_weights(attention, rows, identity):
    # Retain the original arithmetic order for the common three-step case.
    if attention.shape[0] == 3:
        return _three_step_weights(attention, rows, identity)
    return collapsed_rule_weights(attention, rows, identity)


FALLBACK_CHANNEL_AGGREGATION = "sum"


def fallback_rule_scores(rank_weights):
    """Score only the empty-positive-set fallback across rule channels."""
    if FALLBACK_CHANNEL_AGGREGATION == "max":
        return rank_weights.max(axis=1)
    return rank_weights.sum(axis=1)


def select_and_annotate_array(
    candidates,
    attention,
    identity,
    minimum_positive_walks,
):
    selected_indices = np.flatnonzero(
        candidates[:, -6] >= minimum_positive_walks
    )
    used_fallback = False
    all_rank_weights = None
    if selected_indices.size == 0 and candidates.shape[0]:
        used_fallback = True
        all_rank_weights = attention_rank_weights(
            attention, candidates, identity
        )
        totals = fallback_rule_scores(all_rank_weights)
        selected_indices = np.asarray(
            [
                max(
                    range(candidates.shape[0]),
                    key=lambda index: (
                        float(totals[index]),
                        tuple(
                            candidates[index, 1 : 1 + candidates[index, 0]]
                        ),
                    ),
                )
            ]
        )

    selected = candidates[selected_indices]
    if all_rank_weights is None:
        rank_weights = attention_rank_weights(attention, selected, identity)
    else:
        rank_weights = all_rank_weights[selected_indices]
    trials = selected[:, -8].astype(np.float64)
    positives = selected[:, -6].astype(np.float64)
    precision = np.divide(
        positives,
        trials,
        out=np.zeros_like(positives),
        where=trials != 0,
    )
    body_columns = [np.where(selected[:, 0] >= position, selected[:, position], -1)
                    for position in range(1, attention.shape[0] + 1)]
    order = np.lexsort(tuple(reversed(body_columns)) +
                       (-rank_weights.sum(axis=1), -selected[:, -5]))
    return (
        selected[order],
        precision[order],
        rank_weights[order],
        used_fallback,
    )


def compute_attention_array(model, batch_heads, device):
    """Compute one unchanged attention batch and report its worker time."""
    started = time.perf_counter()
    attentions = attention_batch(
        model,
        torch.tensor(batch_heads, device=device, dtype=torch.long),
    )
    if torch.is_tensor(attentions):
        attentions = attentions.detach().cpu().numpy()
    return attentions, time.perf_counter() - started


def output_lines(head, selected, precision, rank_weights):
    lines = []
    for row, row_precision, row_rank_weights in zip(
        selected, precision, rank_weights
    ):
        length = int(row[0])
        body = " ".join(map(str, row[1 : 1 + length]))
        rank_text = " ".join(f"{weight:.17g}" for weight in row_rank_weights)
        lines.append(
            "\t".join(
                map(
                    str,
                    (
                        head,
                        body,
                        length,
                        f"{sum(row_rank_weights):.17g}",
                        row[-8],
                        row[-7],
                        row[-6],
                        row[-5],
                        row[-4],
                        row[-3],
                        row[-2],
                        row[-1],
                        "flashlog-graph-guided-rule",
                        rank_text,
                        f"{row_precision:.17g}",
                    ),
                )
            )
            + "\r\n"
        )
    return lines


def output_rule_lines(
    head, selected, precision, rank_weights, operator_text=None,
):
    """Format detailed and rank-split rows in one traversal."""
    detailed_lines = []
    scalar_lines = []
    detailed_append = detailed_lines.append
    scalar_append = scalar_lines.append
    head_text = str(head)
    for row, row_precision, row_rank_weights in zip(
        selected, precision, rank_weights
    ):
        length = int(row[0])
        if operator_text is None:
            body = " ".join(map(str, row[1 : 1 + length]))
        elif length == 1:
            body = operator_text[row[1]]
        elif length == 2:
            body = f"{operator_text[row[1]]} {operator_text[row[2]]}"
        elif length == 3:
            body = f"{operator_text[row[1]]} {operator_text[row[2]]} {operator_text[row[3]]}"
        else:
            body = " ".join(operator_text[op] for op in row[1:1 + length])
        weight_texts = tuple(
            f"{weight:.17g}" for weight in row_rank_weights
        )
        weight_sum_text = f"{sum(row_rank_weights):.17g}"
        detailed_append(
            f"{head_text}\t{body}\t{length}\t{weight_sum_text}\t"
            f"{row[-8]}\t{row[-7]}\t{row[-6]}\t{row[-5]}\t"
            f"{row[-4]}\t{row[-3]}\t{row[-2]}\t{row[-1]}\t"
            "flashlog-graph-guided-rule\t"
            f"{' '.join(weight_texts)}\t{row_precision:.17g}\r\n"
        )
        scalar_prefix = f"{head_text}\t{body}\t{length}\t"
        for group_id, (weight, weight_text) in enumerate(
            zip(row_rank_weights, weight_texts)
        ):
            if weight <= 0:
                continue
            scalar_append(
                f"{scalar_prefix}{weight_text}\t{group_id}\t"
                "flashlog-rank-weighted-rule\r\n"
            )
    return detailed_lines, scalar_lines


def write_extraction_batch(
    batch, identity, minimum_positive_walks, operator_text,
    destination, rules_destination,
):
    """Finish one batch in order; only the output worker writes the files."""
    totals = Counter()
    length_counts = Counter()
    fallback_heads = []
    per_head = {}
    for (
        head, attention, candidates, available_sources,
        source_counts, raw_unique, head_kernel_seconds,
    ) in batch:
        selection_started = time.perf_counter()
        selected, precision, rank_weights, used_fallback = (
            select_and_annotate_array(
                candidates, attention, identity,
                minimum_positive_walks,
            )
        )
        totals["selection_seconds"] += time.perf_counter() - selection_started
        if used_fallback:
            fallback_heads.append(head)
        formatting_started = time.perf_counter()
        if rules_destination is not None:
            detailed_lines, scalar_lines = output_rule_lines(
                head, selected, precision, rank_weights, operator_text,

            )
        else:
            detailed_lines = output_lines(
                head, selected, precision, rank_weights
            )
        totals["formatting_seconds"] += time.perf_counter() - formatting_started
        write_started = time.perf_counter()
        destination.writelines(detailed_lines)
        if rules_destination is not None:
            rules_destination.writelines(scalar_lines)
            totals["scalar_count"] += len(scalar_lines)
        totals["write_seconds"] += time.perf_counter() - write_started
        length_counts.update(map(int, selected[:, 0]))
        totals["candidate_count"] += len(candidates)
        totals["output_count"] += len(selected)
        per_head[str(head)] = {
            "available_sources": available_sources,
            "sources_per_fold": source_counts,
            "raw_unique_per_fold": raw_unique,
            "stable_horn_candidates": len(candidates),
            "train_positive_rules": int(
                np.count_nonzero(candidates[:, -6] >= minimum_positive_walks)
            ),
            "output_rules": len(selected),
            "fallback": used_fallback,
            "kernel_seconds": head_kernel_seconds,
        }
    return totals, length_counts, fallback_heads, per_head


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--csr-cache",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--source-cache",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--query-head-file",
        type=Path,
        required=True,
    )
    parser.add_argument("--source-folds", type=int, default=1)
    parser.add_argument("--sources-per-fold", type=int, default=512)
    parser.add_argument("--walks-per-source", type=int, default=4096)
    parser.add_argument("--c2", type=int, default=400,
                        help="Total neighbor budget; each direction uses c2 // 2")
    parser.add_argument("--sampling-temperature", type=float, default=1.0,
                        help="Proposal sampling temperature (default: 1)")
    parser.add_argument("--tau-1", type=float, default=1.0)
    parser.add_argument("--tau-2", type=float, default=1.0)
    parser.add_argument(
        "--candidates-per-length",
        type=int,
        nargs="+",
        default=None,
        help="One cap per body length; default: 64, 4096, then 16384 for all longer bodies",
    )
    parser.add_argument("--minimum-positive-walks", type=int, default=1)
    parser.add_argument("--mask-query-head-from-source", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--transition-cache-size", type=int, default=500000)
    parser.add_argument("--attention-batch-size", type=int, default=16)
    parser.add_argument("--sampling-batch-size", type=int, default=None,
                        help="CPU relation batch; a multiple of attention batch size")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--L", type=int, default=3, help="Checkpoint reasoning steps (maximum rule length)")
    parser.add_argument("--N", type=int, default=3, help="Checkpoint DRUM rule channels")
    parser.add_argument("--emb-size", type=int, default=1024)
    parser.add_argument(
        "--rules-output",
        type=Path,
        help="optionally write rank-split scalar rules during recovery",
    )
    return parser.parse_args()


@torch.inference_mode()
def main():
    args = parse_args()
    if args.L <= 0 or args.N <= 0:
        raise ValueError("L and N must be positive")
    if args.candidates_per_length is None:
        args.candidates_per_length = [64, 4096][:args.L] + [16384] * max(0, args.L - 2)
    if len(args.candidates_per_length) != args.L:
        raise ValueError("provide exactly one candidate budget per reasoning step")
    if args.c2 < 2:
        raise ValueError("extraction c2 must be at least 2")
    sampling_batch_size = args.sampling_batch_size
    if sampling_batch_size is None:
        sampling_batch_size = args.attention_batch_size
    if (args.attention_batch_size <= 0 or sampling_batch_size <= 0 or
            sampling_batch_size % args.attention_batch_size != 0):
        raise ValueError("sampling batch size must be a positive multiple of attention batch size")

    def compute_sampling_attention(model, batch_heads, device):
        started = time.perf_counter()
        parts = [compute_attention_array(
            model, batch_heads[start:start + args.attention_batch_size], device
        )[0] for start in range(0, len(batch_heads), args.attention_batch_size)]
        attention = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)
        return attention, time.perf_counter() - started

    if any(not np.isfinite(v) or v <= 0 for v in (args.sampling_temperature, args.tau_1, args.tau_2)):
        raise ValueError("temperatures must be finite and positive")

    positive_names = (
        "source_folds",
        "sources_per_fold",
        "walks_per_source",
        "c2",
        "minimum_positive_walks",
        "threads",
        "attention_batch_size",
    )
    if any(getattr(args, name) <= 0 for name in positive_names):
        raise ValueError("sampling, batching, and thread counts must be positive")
    if any(value <= 0 for value in args.candidates_per_length):
        raise ValueError("candidate budgets must be positive")
    if args.transition_cache_size < 0:
        raise ValueError("transition cache size must be nonnegative")

    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rule_path = args.output_dir / "rule_statistics.tsv"
    num_relations = dictionary_size(args.data_dir / "relations.dict")
    num_entities = dictionary_size(args.data_dir / "entities.dict")
    identity = 2 * num_relations
    operator_text = tuple(map(str, range(identity)))
    heads = read_heads(args.query_head_file, num_relations)
    load_started = time.time()
    source_payload = torch.load(
        args.source_cache, map_location="cpu", weights_only=True
    )
    if set(heads) - set(source_payload["sources"]):
        raise ValueError("source cache does not cover every requested head")
    csr_payload = torch.load(
        args.csr_cache, map_location="cpu", weights_only=True
    )
    if int(csr_payload["num_entities"]) != num_entities:
        raise ValueError("CSR entity count does not match the dataset")
    csr = csr_payload["csr"]
    device = torch.device(args.device)
    model = instantiate_model(
        args.checkpoint,
        num_relations,
        num_entities,
        args.emb_size,
        device, tau_1=args.tau_1, tau_2=args.tau_2, L=args.L, N=args.N,
    )
    load_seconds = time.time() - load_started

    attention_seconds = 0.0
    attention_wait_seconds = 0.0
    source_fold_seconds = 0.0
    kernel_seconds = 0.0
    kernel_cpp_seconds = 0.0
    output_wait_seconds = 0.0
    output_totals = Counter()
    fallback_heads = []
    length_counts = Counter()
    per_head = {}

    def collect_output(future):
        nonlocal output_wait_seconds
        wait_started = time.perf_counter()
        totals, lengths, fallbacks, batch_heads = future.result()
        output_wait_seconds += time.perf_counter() - wait_started
        output_totals.update(totals)
        length_counts.update(lengths)
        fallback_heads.extend(fallbacks)
        for head, stats in batch_heads.items():
            per_head[head] = stats
            completed = len(per_head)
            if (
                completed <= 10
                or completed % 100 == 0
                or completed == len(heads)
            ):
                print(
                    f"quality {completed}/{len(heads)} head={head} "
                    f"stable={stats['stable_horn_candidates']:,} "
                    f"rules={stats['output_rules']:,} "
                    f"kernel={stats['kernel_seconds']:.3f}s",
                    flush=True,
                )

    with ExitStack() as stack:
        destination = stack.enter_context(rule_path.open("w", newline=""))
        destination.write("\t".join(FIELDS) + "\r\n")
        rules_destination = None
        if args.rules_output is not None:
            args.rules_output.parent.mkdir(parents=True, exist_ok=True)
            rules_destination = stack.enter_context(
                args.rules_output.open("w", newline="")
            )
            rules_destination.write("\t".join(SCALAR_FIELDS) + "\r\n")
        batches = [
            heads[start : start + sampling_batch_size]
            for start in range(0, len(heads), sampling_batch_size)
        ]
        with (
            ThreadPoolExecutor(max_workers=1) as attention_executor,
            ThreadPoolExecutor(max_workers=1) as output_executor,
        ):
            output_future = None
            attention_future = attention_executor.submit(
                compute_sampling_attention, model, batches[0], device
            )
            for batch_index, batch_heads in enumerate(batches):
                attention_wait_started = time.perf_counter()
                attentions, head_attention_seconds = attention_future.result()
                attention_wait_seconds += (
                    time.perf_counter() - attention_wait_started
                )
                attention_seconds += head_attention_seconds
                if batch_index + 1 < len(batches):
                    attention_future = attention_executor.submit(
                        compute_sampling_attention,
                        model,
                        batches[batch_index + 1],
                        device,
                    )
                source_fold_started = time.perf_counter()
                batch_source_folds = []
                for head in batch_heads:
                    available = source_payload["sources"][head]
                    source_folds = [
                        list(
                            available[fold::args.source_folds][
                                : args.sources_per_fold
                            ]
                        )
                        for fold in range(args.source_folds)
                    ]
                    batch_source_folds.append(source_folds)
                source_fold_seconds += time.perf_counter() - source_fold_started
                kernel_started = time.perf_counter()
                candidate_tensors, batch_raw_unique, head_kernel_times = (
                    ground_head_batch_source_folds_sampled_capped_counts(
                        attentions, batch_heads, num_relations,
                        batch_source_folds, csr, args.walks_per_source,
                        args.c2 // 2, args.mask_query_head_from_source,
                        [args.seed ^ (head * 0x9E3779B1) for head in batch_heads],
                        args.transition_cache_size, args.threads,
                        args.candidates_per_length,
                        sampling_temperature=args.sampling_temperature,
                    )
                )
                kernel_seconds += time.perf_counter() - kernel_started
                kernel_cpp_seconds += sum(head_kernel_times)
                recovered_batch = []
                for (
                    head, attention, candidate_tensor, source_folds,
                    raw_unique, head_kernel_seconds,
                ) in zip(
                    batch_heads, attentions, candidate_tensors, batch_source_folds,
                    batch_raw_unique, head_kernel_times,
                ):
                    recovered_batch.append((
                        head, attention, candidate_tensor.numpy(),
                        (len(source_payload["sources"][head])),
                        list(map(len, source_folds)), raw_unique,
                        head_kernel_seconds,
                    ))
                # At most one batch is being written while this batch samples.
                # Await it before submitting another to bound retained arrays.
                if output_future is not None:
                    collect_output(output_future)
                output_future = output_executor.submit(
                    write_extraction_batch, recovered_batch, identity,
                    args.minimum_positive_walks, operator_text,
                    destination, rules_destination,
                )
            if output_future is not None:
                collect_output(output_future)

    payload = {
        "combination": "union",
        "sampling_temperature": args.sampling_temperature,
        "model_tau_1": args.tau_1,
        "model_tau_2": args.tau_2,
        "algorithm": (
            "single-checkpoint compiled joint-fold source-grounded Horn "
            "recovery with Train-positive filtering and analytical attention weights"
        ),
        "rule_representation": f"explicit_horn_relation_chain_length_1_to_{args.L}",
        "L": args.L,
        "N": args.N,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "data_dir": str(args.data_dir.resolve()),
        "csr_cache": str(args.csr_cache.resolve()),
        "source_cache": str(args.source_cache.resolve()),
        "query_head_file": str(args.query_head_file.resolve()),
        "heads": len(heads),
        "num_base_relations": num_relations,
        "source_folds": args.source_folds,
        "sources_per_fold": args.sources_per_fold,
        "walks_per_source": args.walks_per_source,
        "c2_total": args.c2,
        "c2_per_direction": args.c2 // 2,
        "candidate_budgets_per_length_per_fold": list(
            args.candidates_per_length
        ),
        "graph_guided_proposal": True,
        "proposal_distribution": (
            "attention times available edge multiplicity"
        ),
        "candidate_generation_threshold": None,
        "minimum_positive_walks": args.minimum_positive_walks,
        "mask_query_head_from_source": args.mask_query_head_from_source,
        "stable_horn_candidates": output_totals["candidate_count"],
        "output_rules": output_totals["output_count"],
        "fallback_heads": fallback_heads,
        "body_length_counts": dict(sorted(length_counts.items())),
        "load_seconds": load_seconds,
        "attention_seconds": attention_seconds,
        "attention_wait_seconds": attention_wait_seconds,
        "source_fold_seconds": source_fold_seconds,
        "kernel_seconds": kernel_seconds,
        "kernel_cpp_seconds": kernel_cpp_seconds,
        "kernel_call_overhead_seconds": kernel_seconds - kernel_cpp_seconds,
        "kernel_call_count": len(batches),
        "per_head_kernel_timing_scope": (
            "C++ work wall including teardown; shared head groups "
            "apportioned by source count, not independent per-head latency"
        ),
        "selection_seconds": output_totals["selection_seconds"],
        "formatting_seconds": output_totals["formatting_seconds"],
        "write_seconds": output_totals["write_seconds"],
        "postprocess_wait_seconds": output_wait_seconds,
        "postprocess_prefetch_batches": 1,
        "python_executable": sys.executable,
        "torch_version": str(torch.__version__),
        "numpy_version": str(np.__version__),
        "seed": args.seed,
        "threads": args.threads,
        "attention_batch_size": args.attention_batch_size,
        "transition_cache_size": args.transition_cache_size,
        "elapsed_seconds": time.time() - started,
        "graph_split": "train",
        "train_targets_used_for_filtering": True,
        "train_targets_used_for_weighting": False,
        "output_schema_version": 2,
        "weight_field_definition": "sum of analytical attention rank weights",
        "rank_weight_definition": "analytical attention, including all identity placements",
        "output_order": "positive_source_hits descending, attention weight sum descending, body ascending",
        "reference_rule_bank_used": False,
        "soft_model_retrained": False,
        "validation_entities_used": False,
        "validation_labels_used": False,
        "test_entities_used": False,
        "test_labels_used": False,
        "output_file": str(rule_path.resolve()),
        "rules_output_file": (
            str(args.rules_output.resolve())
            if args.rules_output is not None
            else None
        ),
        "scalar_rule_instances": (
            output_totals["scalar_count"] if args.rules_output is not None else None
        ),
        "per_head": per_head,
    }
    metadata_path = args.output_dir / "extraction.json"
    metadata_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in payload.items() if key != "per_head"},
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
