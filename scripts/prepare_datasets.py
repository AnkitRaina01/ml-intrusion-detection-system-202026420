#!/usr/bin/env python
"""
Phase 4 — unified, leakage-safe preprocessing for every supported dataset.

    python scripts/prepare_datasets.py --dataset nsl_kdd
    python scripts/prepare_datasets.py --dataset cicids2017
    python scripts/prepare_datasets.py --dataset unsw_nb15

Each dataset is loaded by its own adapter in ``src/data_loaders.py`` into the
SAME bundle shape, then run through the SAME ``src.preprocessing.prepare_dataset``
pipeline that Phase 2 built for NSL-KDD:

    load raw  ->  inspect schema / labels  ->  reproducible stratified split
    ->  fit impute+scale (+one-hot) on TRAIN only
    ->  mutual-information feature selection on TRAIN only
    ->  SMOTE on TRAIN only
    ->  write processed arrays + fitted preprocessor + statistics report

Nothing dataset-specific leaks into the pipeline: differences (feature spaces,
label taxonomies, whether an official split exists, sub-sampling) are handled in
the adapter and recorded in ``dataset_meta`` / DEVELOPMENT_NOTES.md section 11.

Outputs (per dataset ``<ds>``):
    data/processed/<ds>/*.npy          (git-ignored, regenerate with this script)
    artifacts/<ds>/preprocessor.joblib (git-ignored)
    results/<ds>/dataset_stats.{json,md}   (committed dissertation evidence)
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

from src.data_loaders import (load_nsl_kdd, load_cicids2017,        # noqa: E402
                              load_unsw_nb15, inspect_bundle)
from src.preprocessing import PreprocessConfig, prepare_dataset     # noqa: E402

# per-dataset: (loader, default split_mode, raw-dir name)
DATASETS = {
    "nsl_kdd":    (load_nsl_kdd,     "predefined", "nsl_kdd"),
    "cicids2017": (load_cicids2017,  "predefined", "cicids2017"),
    "unsw_nb15":  (load_unsw_nb15,   "predefined", "unsw_nb15"),
}


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    p.add_argument("--raw-dir", default=None,
                   help="override data/raw/<dataset>")
    p.add_argument("--out-root", default=str(REPO_ROOT))
    p.add_argument("--target", choices=["binary", "multiclass"], default="binary")
    p.add_argument("--split-mode", choices=["predefined", "pooled"], default=None)
    p.add_argument("--n-features", type=int, default=20)
    p.add_argument("--balance", choices=["smote", "none"], default="smote")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cicids-sample-size", type=int, default=200_000,
                   help="CIC-IDS2017 only: rows in the seeded stratified sub-sample")
    p.add_argument("--no-write", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    log = logging.getLogger("prepare_datasets")

    loader, default_split, raw_name = DATASETS[args.dataset]
    raw_dir = args.raw_dir or str(REPO_ROOT / "data" / "raw" / raw_name)
    split_mode = args.split_mode or default_split
    out_root = Path(args.out_root)

    log.info("loading %s from %s", args.dataset, raw_dir)
    if args.dataset == "cicids2017":
        bundle = loader(raw_dir, sample_size=args.cicids_sample_size, seed=args.seed)
    elif args.dataset == "unsw_nb15":
        bundle = loader(raw_dir, seed=args.seed)
    else:
        bundle = loader(raw_dir)

    report = inspect_bundle(bundle)
    log.info("schema inspection:\n%s", json.dumps(report, indent=2, default=str))

    cfg = PreprocessConfig(
        dataset=args.dataset, target=args.target, split_mode=split_mode,
        n_features=args.n_features, balance=args.balance, seed=args.seed,
    )
    log.info("config: %s", cfg)

    out_dir = None if args.no_write else out_root / "data" / "processed" / args.dataset
    art_dir = None if args.no_write else out_root / "artifacts" / args.dataset
    stats_dir = None if args.no_write else out_root / "results" / args.dataset

    result = prepare_dataset(bundle, cfg, out_dir=out_dir,
                             artifacts_dir=art_dir, stats_dir=stats_dir)

    s = result["stats"]
    sp = s["splits"]
    print("\n" + "=" * 70)
    print(f"{args.dataset.upper()} PREPROCESSING SUMMARY")
    print("=" * 70)
    meta = s.get("dataset_meta", {}) or {}
    if meta.get("subsample"):
        print(f"  sub-sampling          : {meta['subsample']}")
    print(f"  target / split        : {cfg.target} / {cfg.split_mode}")
    print(f"  samples (pool/test)   : {s['samples']['raw_train_pool']:,} / "
          f"{s['samples']['raw_test']:,}  (total {s['samples']['grand_total']:,})")
    print(f"  features before/enc/sel: {s['features']['before_preprocessing']} / "
          f"{s['features']['after_encoding_and_variance_filter']} / "
          f"{s['features']['after_selection']}")
    print(f"  train / val / test    : {sp['train']['n_samples']:,} "
          f"({sp['train']['percent_of_total']:.1f}%) / "
          f"{sp['validation']['n_samples']:,} "
          f"({sp['validation']['percent_of_total']:.1f}%) / "
          f"{sp['test']['n_samples']:,} ({sp['test']['percent_of_total']:.1f}%)")
    print(f"  X_train / X_val / X_test: {tuple(result['X_train'].shape)} / "
          f"{tuple(result['X_val'].shape)} / {tuple(result['X_test'].shape)}")
    if "X_train_balanced" in result:
        print(f"  X_train_balanced      : {tuple(result['X_train_balanced'].shape)}")
    cb = s["class_balance"]
    print(f"  train balance before  : {cb['train_before_balancing']}")
    print(f"  train balance after   : {cb['train_after_balancing']}")
    if not args.no_write:
        print(f"\n  processed arrays -> {out_dir}")
        print(f"  artefacts       -> {art_dir}")
        print(f"  statistics      -> {stats_dir}/dataset_stats.(json|md)")
    print("=" * 70 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
