#!/usr/bin/env python
"""
Phase 2 — build reproducible, leakage-safe NSL-KDD preprocessing artefacts.

Runs the full pipeline defined in ``src/preprocessing.py``:

    load raw NSL-KDD  ->  inspect  ->  stratified 70/15/15 partitions
    ->  fit encoder/scaler/variance-filter on TRAIN only
    ->  mutual-information feature selection on TRAIN only
    ->  SMOTE on TRAIN only
    ->  write processed arrays + fitted artefacts + statistics report

Everything is deterministic given ``--seed``. Re-running overwrites cleanly.

Examples
--------
    python scripts/prepare_nsl_kdd.py                      # defaults (binary target)
    python scripts/prepare_nsl_kdd.py --target multiclass --n-features 30
    python scripts/prepare_nsl_kdd.py --split-mode pooled --balance none
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_loaders import load_nsl_kdd, inspect_nsl_kdd            # noqa: E402
from src.preprocessing import PreprocessConfig, prepare_dataset       # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", default=str(REPO_ROOT / "data" / "raw" / "nsl_kdd"))
    p.add_argument("--out-dir", default=str(REPO_ROOT / "data" / "processed" / "nsl_kdd"))
    p.add_argument("--artifacts-dir", default=str(REPO_ROOT / "artifacts" / "nsl_kdd"))
    p.add_argument("--stats-dir", default=str(REPO_ROOT / "results" / "nsl_kdd"))
    p.add_argument("--target", choices=["binary", "multiclass"], default="binary")
    p.add_argument("--split-mode", choices=["predefined", "pooled"],
                   default="predefined")
    p.add_argument("--n-features", type=int, default=20)
    p.add_argument("--balance", choices=["smote", "none"], default="smote")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-write", action="store_true",
                   help="run the pipeline but do not write any files")
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    log = logging.getLogger("prepare_nsl_kdd")

    log.info("loading NSL-KDD from %s", args.raw_dir)
    bundle = load_nsl_kdd(args.raw_dir)

    report = inspect_nsl_kdd(bundle)
    log.info("inspection:\n%s", json.dumps(report, indent=2))

    cfg = PreprocessConfig(
        dataset="nsl_kdd", target=args.target, split_mode=args.split_mode,
        n_features=args.n_features, balance=args.balance, seed=args.seed,
    )
    log.info("config: %s", cfg)

    out_dir = None if args.no_write else args.out_dir
    art_dir = None if args.no_write else args.artifacts_dir
    stats_dir = None if args.no_write else args.stats_dir

    result = prepare_dataset(bundle, cfg, out_dir=out_dir,
                             artifacts_dir=art_dir, stats_dir=stats_dir)

    s = result["stats"]
    sp = s["splits"]
    print("\n" + "=" * 68)
    print("NSL-KDD PREPROCESSING SUMMARY")
    print("=" * 68)
    print(f"  target                : {cfg.target}")
    print(f"  split mode            : {cfg.split_mode}")
    print(f"  samples (pool/test)   : {s['samples']['raw_train_pool']:,} / "
          f"{s['samples']['raw_test']:,}  (total {s['samples']['grand_total']:,})")
    print(f"  features before       : {s['features']['before_preprocessing']}")
    print(f"  features after encode : "
          f"{s['features']['after_encoding_and_variance_filter']}")
    print(f"  features after select : {s['features']['after_selection']}")
    print(f"  train / val / test    : {sp['train']['n_samples']:,} "
          f"({sp['train']['percent_of_total']:.1f}%) / "
          f"{sp['validation']['n_samples']:,} "
          f"({sp['validation']['percent_of_total']:.1f}%) / "
          f"{sp['test']['n_samples']:,} ({sp['test']['percent_of_total']:.1f}%)")
    print(f"  X_train shape         : {tuple(result['X_train'].shape)}")
    print(f"  X_val shape           : {tuple(result['X_val'].shape)}")
    print(f"  X_test shape          : {tuple(result['X_test'].shape)}")
    if "X_train_balanced" in result:
        print(f"  X_train_balanced      : {tuple(result['X_train_balanced'].shape)}")
    cb = s["class_balance"]
    print(f"  train balance before  : {cb['train_before_balancing']}")
    print(f"  train balance after   : {cb['train_after_balancing']}")
    if not args.no_write:
        print(f"\n  processed arrays -> {args.out_dir}")
        print(f"  artefacts       -> {args.artifacts_dir}")
        print(f"  statistics      -> {args.stats_dir}/dataset_stats.(json|md)")
    print("=" * 68 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
