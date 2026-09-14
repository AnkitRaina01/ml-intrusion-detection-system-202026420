#!/usr/bin/env python
"""
Phase 3 — build fixed-length sequence windows of NSL-KDD records for the LSTM.

WHY THIS IS NOT "rows are automatically time series"
---------------------------------------------------
NSL-KDD rows are individual connection records. There is **no timestamp and no
session id**, and in the published file order the class label changes roughly
every other row (~62.7k switches over 125,973 training records, mean run length
~2). So NSL-KDD does **not** contain strong temporal sequences.

What we do instead, and its limitations, is documented in DEVELOPMENT_NOTES.md
section 10.3. In short: a "sequence" = ``W`` consecutive records in the file
order; the window's label is the **last** record's label (a causal
"classify this record given its recent context" framing). This is a recognised
NSL-KDD LSTM setup; treat the LSTM here as "an LSTM over a context window of
tabular records", not as genuine packet-level temporal modelling.

Leakage safety
--------------
* KDDTrain+ is split into train / val by a **contiguous** cut with a ``W``-row
  guard gap, so no window straddles the boundary and no record is shared.
* The encoder / scaler / feature selector (`FittedPreprocessor`) is fit on the
  **training records only** and then applied to val and test.
* KDDTest+ is the held-out test set, untouched.

Outputs (default): ``data/processed/nsl_kdd_seq/`` and ``artifacts/nsl_kdd_seq/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_loaders import load_nsl_kdd                                  # noqa: E402
from src.preprocessing import PreprocessConfig, fit_preprocessor          # noqa: E402


def _windows(X: np.ndarray, y: np.ndarray, anchors: np.ndarray, W: int):
    """Stack windows [a-W+1 .. a] for each anchor a; label = y[a]."""
    Xs = np.stack([X[a - W + 1: a + 1] for a in anchors]).astype(np.float32)
    ys = y[anchors].astype(np.int64)
    return Xs, ys


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", default=str(REPO_ROOT / "data" / "raw" / "nsl_kdd"))
    p.add_argument("--out-dir",
                   default=str(REPO_ROOT / "data" / "processed" / "nsl_kdd_seq"))
    p.add_argument("--artifacts-dir",
                   default=str(REPO_ROOT / "artifacts" / "nsl_kdd_seq"))
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--n-features", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-frac-of-total", type=float, default=0.15,
                   help="validation fraction of (KDDTrain+ + KDDTest+)")
    return p


def main(argv=None) -> int:
    a = build_arg_parser().parse_args(argv)
    W, S = a.window, a.stride
    bundle = load_nsl_kdd(a.raw_dir)
    feat = list(bundle["feature_columns"])
    cat = list(bundle["categorical_columns"])
    num = list(bundle["numeric_columns"])
    pool = bundle["train"].reset_index(drop=True)      # file order preserved
    test = bundle["test"].reset_index(drop=True)

    n_pool, n_test = len(pool), len(test)
    grand_total = n_pool + n_test
    val_n = round(a.val_frac_of_total * grand_total)
    cut = n_pool - val_n                               # contiguous train | val
    if not W < cut < n_pool:
        raise ValueError(f"bad cut={cut} for pool={n_pool}, W={W}")

    train_rows = pool.iloc[:cut]
    cfg = PreprocessConfig(dataset="nsl_kdd_seq", target="binary",
                           n_features=a.n_features, seed=a.seed)
    pre = fit_preprocessor(train_rows, feat, cat, num, "binary_label", cfg)

    Xpool = pre.transform(pool[feat])                  # (n_pool, F)
    Xtest = pre.transform(test[feat])                  # (n_test, F)
    ypool = pool["binary_label"].to_numpy()
    ytest = test["binary_label"].to_numpy()
    F = Xpool.shape[1]

    # anchors (last index of each window), stride S
    tr_anchors = np.arange(W - 1, cut, S)                       # windows fully in [0, cut)
    va_anchors = np.arange(cut - 1 + W, n_pool, S)              # guard gap of W rows
    te_anchors = np.arange(W - 1, n_test, S)

    Xtr, ytr = _windows(Xpool, ypool, tr_anchors, W)
    Xva, yva = _windows(Xpool, ypool, va_anchors, W)
    Xte, yte = _windows(Xtest, ytest, te_anchors, W)

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    art = Path(a.artifacts_dir); art.mkdir(parents=True, exist_ok=True)
    np.save(out / "Xseq_train.npy", Xtr); np.save(out / "yseq_train.npy", ytr)
    np.save(out / "Xseq_val.npy", Xva);   np.save(out / "yseq_val.npy", yva)
    np.save(out / "Xseq_test.npy", Xte);  np.save(out / "yseq_test.npy", yte)
    pre.save(art / "preprocessor.joblib")
    (art / "feature_names.json").write_text(json.dumps(pre.feature_names, indent=2))

    def dist(y):
        u, c = np.unique(y, return_counts=True)
        return {int(k): int(v) for k, v in zip(u, c)}

    meta = {
        "dataset": "nsl_kdd_seq", "seed": a.seed,
        "window": W, "stride": S, "n_features": F,
        "split": "contiguous KDDTrain+ cut with W-row guard gap; KDDTest+ = test",
        "record_counts": {"pool": n_pool, "train_rows": int(cut),
                          "val_rows": int(n_pool - cut), "test": n_test},
        "window_counts": {"train": int(len(Xtr)), "val": int(len(Xva)),
                          "test": int(len(Xte))},
        "window_label_distribution": {"train": dist(ytr), "val": dist(yva),
                                      "test": dist(yte)},
        "window_label": "last record in the window",
        "selected_features": pre.feature_names,
        "shapes": {"Xseq_train": list(Xtr.shape), "Xseq_val": list(Xva.shape),
                   "Xseq_test": list(Xte.shape)},
        "leakage_note": ("preprocessor fit on the first %d KDDTrain+ records only; "
                         "no window crosses the train/val cut" % cut),
        "limitations": [
            "NSL-KDD has no timestamps / session ids",
            "file-order label autocorrelation is near zero (mean run length ~2)",
            "windows impose an order that is only weakly meaningful; the LSTM here "
            "is effectively 'LSTM over a context window of tabular records'",
        ],
    }
    (out / "sequence_metadata.json").write_text(json.dumps(meta, indent=2))
    (art / "sequence_metadata.json").write_text(json.dumps(meta, indent=2))

    print("\n" + "=" * 64)
    print("NSL-KDD SEQUENCE BUILD")
    print("=" * 64)
    print(f"  window / stride       : {W} / {S}")
    print(f"  features per step     : {F}")
    print(f"  record cut (train|val): {cut} | {n_pool - cut}   (test {n_test})")
    print(f"  windows train/val/test: {len(Xtr):,} / {len(Xva):,} / {len(Xte):,}")
    print(f"  window-label dist trn : {dist(ytr)}")
    print(f"  window-label dist val : {dist(yva)}")
    print(f"  window-label dist test: {dist(yte)}")
    print(f"  Xseq_train shape      : {Xtr.shape}")
    print(f"  arrays -> {out}")
    print(f"  artefacts -> {art}")
    print("=" * 64 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
