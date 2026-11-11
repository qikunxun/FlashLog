#!/usr/bin/env python3
r"""Run FlashLog extraction or rule inference for the four supported models.

Examples (remaining arguments are forwarded to the native entrypoint):
  python rules/run.py extract --method DRUM OUTPUT_DIR --data-dir DATA \
      --checkpoint MODEL --emb-size 128 --csr-cache INDEX/train_compact_csr.pt \
      --source-cache INDEX/train_sources_8192.pt \
      --query-head-file INDEX/train_relation_heads.txt --c2 400 \
      --rules-output OUTPUT_DIR/rules.tsv
  python rules/run.py evaluate --method DRUM --data-dir DATA \
      --rule-file OUTPUT_DIR/rules.tsv --output-dir EVAL_DIR \
      --csr-cache INDEX/train_compact_csr.pt

Set --L and --N to the checkpoint reasoning steps and DRUM channels.
Both default to three. Use the checkpoint's embedding size. Method-specific adapters
retain the FlashLog attention coefficients and rule aggregation semantics.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('phase', choices=['extract', 'evaluate'])
    parser.add_argument('--method', choices=['DRUM', 'smDRUM', 'mmDRUM', 'NeuralLP'], default='DRUM')
    args, remaining = parser.parse_known_args()
    from rules import adapters
    adapters.install(args.method)
    sys.argv = [sys.argv[0], *remaining]
    if args.phase == 'extract':
        adapters.recovery.main()
    else:
        import torch
        with torch.inference_mode():
            adapters.evaluation.main()


if __name__ == '__main__':
    main()
