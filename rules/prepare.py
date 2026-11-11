#!/usr/bin/env python3
"""Build the reusable Train CSR and directed-source index for Horn recovery."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "rules"))

from rules.evaluate import (  # noqa: E402
    dictionary_size,
)
from flashlog_kernel.flashlog_kernel import _build_group_metadata  # noqa: E402
from rules.common import (  # noqa: E402
    sample_sources_by_relation,
)


def read_heads(path, num_relations):
    heads = sorted(
        int(line) for line in path.read_text().splitlines() if line.strip()
    )
    if len(heads) != len(set(heads)):
        raise ValueError("query-head file contains duplicate relations")
    if any(head < 0 or head >= 2 * num_relations for head in heads):
        raise ValueError("query-head file contains an invalid directed relation")
    return heads


def directed_heads_from_train_relations(relations, num_relations):
    values = np.asarray(relations)
    # Bound the scratch mask. Large/sparse vocabularies and noninteger inputs
    # retain the original generic path; Train's small integer vocabulary can
    # avoid unique()'s full input copy and general-purpose hash table.
    if (
        values.dtype.kind in "iu"
        and isinstance(num_relations, (int, np.integer))
        and 0 <= num_relations <= 65536
    ):
        if values.size and (values.min() < 0 or values.max() >= num_relations):
            raise ValueError("Train contains an invalid base relation")
        present = np.zeros(int(num_relations), dtype=np.bool_)
        present[values] = True
        base_heads = np.flatnonzero(present).astype(values.dtype, copy=False)
    else:
        base_heads = np.unique(values)
    if np.any(base_heads < 0) or np.any(base_heads >= num_relations):
        raise ValueError("Train contains an invalid base relation")
    return np.concatenate((base_heads, base_heads + num_relations)).tolist()


def build_rule_csr_structure(row_indices, col_indices, r_indices, num_nodes):
    """Build only rule-serving CSR fields, preserving the training sort order.

    Keep the same int64 keys, argsort and group builder as build_csr_structure.
    In particular, do not re-sort equal-key neighbors: their positions are part
    of the seeded walk semantics. Edge-CSR row pointers and exported sort
    permutations are unused by rule extraction and are never materialized.
    """
    row_cpu = row_indices.long().cpu()
    col_cpu = col_indices.long().cpu()
    r_cpu = r_indices.long().cpu()
    rel_base = int(r_cpu.max().item()) + 1 if r_cpu.numel() > 0 else 1

    def build_side(rows, columns):
        order = (rows * rel_base + r_cpu).argsort()
        sorted_rows = rows[order]
        sorted_columns = columns[order].to(torch.int32)
        sorted_relations = r_cpu[order].to(torch.int16)
        metadata = _build_group_metadata(
            sorted_rows, sorted_relations.to(torch.long), num_nodes
        )
        # The order and sorted work arrays die before constructing the other
        # direction, instead of remaining live until the full graph is built.
        return (sorted_columns, *metadata)

    forward = build_side(row_cpu, col_cpu)
    inverse = build_side(col_cpu, row_cpu)
    empty = torch.empty(0)
    return (
        empty, forward[0], empty, empty,
        empty, inverse[0], empty, empty,
        *forward[1:], *inverse[1:],
    )


def load_train_columns(path):
    """Load three integer columns, using Arrow's parallel parser when present."""
    try:
        import pyarrow as pa
        import pyarrow.csv as arrow_csv
    except ImportError:
        train = pd.read_csv(
            path,
            sep="\t",
            header=None,
            engine="c",
            dtype={0: np.int32, 1: np.int32, 2: np.int32},
        )
        return tuple(
            train[column].to_numpy(copy=False) for column in range(3)
        ), "pandas-c"

    table = arrow_csv.read_csv(
        path,
        read_options=arrow_csv.ReadOptions(
            column_names=("source", "relation", "target"),
            block_size=8 << 20,
            use_threads=True,
        ),
        parse_options=arrow_csv.ParseOptions(delimiter="\t"),
        convert_options=arrow_csv.ConvertOptions(
            column_types={
                "source": pa.int32(),
                "relation": pa.int32(),
                "target": pa.int32(),
            }
        ),
    )
    columns = tuple(
        table.column(name).combine_chunks().to_numpy(zero_copy_only=True)
        for name in ("source", "relation", "target")
    )
    return columns, "pyarrow-csv-parallel"


def build_source_payload(csr, heads, num_relations, max_sources, seed):
    rng = np.random.default_rng(seed)
    forward = [head for head in heads if head < num_relations]
    inverse = [head - num_relations for head in heads if head >= num_relations]
    indexed = {}
    available = {}
    if forward:
        indexed["forward"], available["forward"] = sample_sources_by_relation(
            csr[8], csr[9], forward, max_sources, rng
        )
    if inverse:
        indexed["inverse"], available["inverse"] = sample_sources_by_relation(
            csr[12], csr[13], inverse, max_sources, rng
        )
    sources = {}
    available_by_head = {}
    for head in heads:
        if head < num_relations:
            direction = "forward"
            relation = head
        else:
            direction = "inverse"
            relation = head - num_relations
        sources[head] = indexed[direction][relation][:max_sources]
        available_by_head[head] = available[direction][relation]
    return sources, available_by_head


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--data-dir", type=Path, required=True)
    head_source = parser.add_mutually_exclusive_group(required=True)
    head_source.add_argument("--query-head-file", type=Path)
    head_source.add_argument("--derive-heads-from-train", action="store_true")
    parser.add_argument(
        "--head-output-file",
        type=Path,
        help=(
            "where to write Train-derived directed heads; defaults to "
            "OUTPUT_DIR/train_relation_heads.txt"
        ),
    )
    parser.add_argument("--max-sources", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_sources <= 0:
        raise ValueError("--max-sources must be positive")
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    num_entities = dictionary_size(args.data_dir / "entities.dict")
    num_relations = dictionary_size(args.data_dir / "relations.dict")

    load_started = time.time()
    train_columns, train_parser = load_train_columns(
        args.data_dir / "train_id.txt"
    )
    train_source, train_relation, train_target = train_columns
    train_edges = len(train_source)
    train_load_seconds = time.time() - load_started
    head_started = time.time()
    if args.derive_heads_from_train:
        heads = directed_heads_from_train_relations(
            train_relation, num_relations
        )
        head_output_file = (
            args.head_output_file
            if args.head_output_file is not None
            else args.output_dir / "train_relation_heads.txt"
        )
        head_output_file.parent.mkdir(parents=True, exist_ok=True)
        head_output_file.write_text("".join(f"{head}\n" for head in heads))
    else:
        if args.head_output_file is not None:
            raise ValueError(
                "--head-output-file is only valid with --derive-heads-from-train"
            )
        heads = read_heads(args.query_head_file, num_relations)
        head_output_file = args.query_head_file
    head_discovery_seconds = time.time() - head_started

    build_started = time.time()
    csr = build_rule_csr_structure(
        torch.from_numpy(train_source),
        torch.from_numpy(train_target),
        torch.from_numpy(train_relation),
        num_entities,
    )
    csr_build_seconds = time.time() - build_started
    csr_path = args.output_dir / "train_compact_csr.pt"
    save_started = time.time()
    torch.save({"num_entities": num_entities, "csr": csr}, csr_path)
    csr_save_seconds = time.time() - save_started
    del train_columns
    del train_source
    del train_relation
    del train_target

    source_started = time.time()
    sources, available = build_source_payload(
        csr, heads, num_relations, args.max_sources, args.seed
    )
    source_path = args.output_dir / f"train_sources_{args.max_sources}.pt"
    torch.save(
        {
            "heads": heads,
            "max_sources": args.max_sources,
            "sources": sources,
            "seed": args.seed,
            "source_csr_cache": str(csr_path.resolve()),
            "included_source_cache": None,
        },
        source_path,
    )
    source_seconds = time.time() - source_started
    elapsed = time.time() - started
    metadata = {
        "algorithm": (
            "one-pass reusable Train CSR and linear-reservoir source-index build"
        ),
        "source_sampling": "per-relation uniform reservoir without replacement",
        "data_dir": str(args.data_dir.resolve()),
        "query_head_file": (
            None
            if args.query_head_file is None
            else str(args.query_head_file.resolve())
        ),
        "head_source": (
            "train_relation_column"
            if args.derive_heads_from_train
            else "query_head_file"
        ),
        "head_output_file": str(head_output_file.resolve()),
        "train_edges": train_edges,
        "train_parser": train_parser,
        "num_entities": num_entities,
        "num_base_relations": num_relations,
        "directed_heads": len(heads),
        "max_sources": args.max_sources,
        "seed": args.seed,
        "train_load_seconds": train_load_seconds,
        "head_discovery_seconds": head_discovery_seconds,
        "csr_build_seconds": csr_build_seconds,
        "csr_save_seconds": csr_save_seconds,
        "source_index_seconds": source_seconds,
        "elapsed_seconds": elapsed,
        "csr_cache": str(csr_path.resolve()),
        "source_cache": str(source_path.resolve()),
        "available_train_sources": {
            str(head): available[head] for head in heads
        },
        "graph_split": "train",
        "validation_entities_used": False,
        "validation_labels_used": False,
        "test_entities_used": False,
        "test_labels_used": False,
    }
    (args.output_dir / "index.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                key: value
                for key, value in metadata.items()
                if key != "available_train_sources"
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
