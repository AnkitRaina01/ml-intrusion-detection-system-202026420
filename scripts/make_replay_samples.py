#!/usr/bin/env python
"""
Phase 7 — carve a small, committed **raw** CIC-IDS2017 flow sample for replay.

CIC-IDS2017's raw set is 8 CSVs / ~844 MB, too heavy to iterate live in a demo.
This writes ``data/replay_samples/cicids2017_raw_sample.csv`` (~3000 rows) with
the dataset's 77 native feature columns plus ``_true_binary`` / ``_true_category``
(ground truth, for scoring the replay — never fed to the model).

NSL-KDD and UNSW-NB15 are replayed straight from their raw test files
(``KDDTest+.txt`` / ``UNSW_NB15_testing-set.csv``), so no sample is needed there.

    python scripts/make_replay_samples.py            # ~3 min (reads the 8 raw CSVs)
    python scripts/make_replay_samples.py --n 5000 --seed 7
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main(argv=None) -> int:
    import pandas as pd
    from src.data_loaders import load_cicids2017

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--per-category", type=int, default=600)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--raw-dir", default=str(REPO_ROOT / "data" / "raw" / "cicids2017"))
    ap.add_argument("--out", default=str(REPO_ROOT / "data" / "replay_samples"
                                        / "cicids2017_raw_sample.csv"))
    args = ap.parse_args(argv)

    # a comfortable sub-sample so every coarse class has >= 2 members for the
    # loader's internal stratified split, then we carve our own selection.
    print("reading CIC-IDS2017 raw CSVs (memory-safe two-pass) ...")
    b = load_cicids2017(args.raw_dir, sample_size=150_000, seed=args.seed)
    cols = list(b["feature_columns"])
    te = b["test"].copy()
    df = te[cols].copy()
    df["_true_binary"] = te["binary_label"].map({0: "normal", 1: "attack"}).values
    df["_true_category"] = te["attack_category"].astype(str).values

    df = (df.sample(frac=1.0, random_state=args.seed)
            .groupby("_true_category", group_keys=False, sort=False)
            .head(args.per_category)
            .sample(frac=1.0, random_state=args.seed + 1)
            .head(args.n)
            .reset_index(drop=True))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {out}  shape={df.shape}")
    print("  binary  :", df["_true_binary"].value_counts().to_dict())
    print("  category:", df["_true_category"].value_counts().to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
