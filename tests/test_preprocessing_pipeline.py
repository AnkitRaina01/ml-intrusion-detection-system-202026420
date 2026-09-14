"""
Fast invariant tests for the Phase-2 preprocessing pipeline.

These use a tiny synthetic NSL-KDD-shaped bundle (no disk, no download) so they
run in well under a second, and assert the properties that matter:
leakage-safety, split sizing, feature reduction, SMOTE balancing, and
train == inference transform parity.

Run:  python -m pytest tests/test_preprocessing_pipeline.py -q
      (pytest is not a hard project dependency; the file also runs under
       `python tests/test_preprocessing_pipeline.py`)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_loaders import (NSL_KDD_FEATURES, NSL_KDD_CATEGORICAL,        # noqa: E402
                              NSL_KDD_NUMERIC, NSL_KDD_CATEGORIES)
from src.preprocessing import (PreprocessConfig, prepare_dataset,          # noqa: E402
                               FittedPreprocessor)


def _synth_split(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.integers(0, 5, n).astype(float) for c in NSL_KDD_NUMERIC})
    df["src_bytes"] = rng.integers(0, 100_000, n).astype(float)
    df["dst_bytes"] = rng.integers(0, 50_000, n).astype(float)
    df["num_outbound_cmds"] = 0.0  # constant feature, must be dropped
    df["protocol_type"] = rng.choice(["tcp", "udp", "icmp"], n)
    df["service"] = rng.choice(["http", "private", "domain_u", "smtp"], n)
    df["flag"] = rng.choice(["SF", "S0", "REJ"], n)
    # label: ~55% normal, rest spread across 4 attack families
    lab = rng.choice(["normal", "neptune", "satan", "guess_passwd", "rootkit"],
                     n, p=[0.55, 0.25, 0.12, 0.06, 0.02])
    df["label"] = lab
    df["difficulty"] = rng.integers(0, 21, n)
    df = df[NSL_KDD_FEATURES + ["label", "difficulty"]]
    from src.data_loaders import NSL_KDD_ATTACK_CATEGORY
    df["attack_category"] = df["label"].map(NSL_KDD_ATTACK_CATEGORY)
    df["binary_label"] = (df["label"] != "normal").astype(int)
    return df


def _bundle():
    return {
        "train": _synth_split(6000, 1),
        "test": _synth_split(1400, 2),
        "feature_columns": list(NSL_KDD_FEATURES),
        "categorical_columns": list(NSL_KDD_CATEGORICAL),
        "numeric_columns": list(NSL_KDD_NUMERIC),
        "categories": list(NSL_KDD_CATEGORIES),
    }


def _run(**over):
    cfg = PreprocessConfig(dataset="synth", seed=42, n_features=12, **over)
    return prepare_dataset(_bundle(), cfg), cfg


def test_no_nan_or_inf():
    res, _ = _run()
    for k in ("X_train", "X_val", "X_test", "X_train_balanced"):
        a = res[k]
        assert np.isfinite(a).all(), f"{k} has non-finite values"


def test_feature_reduction_and_constant_drop():
    res, cfg = _run()
    feats = res["stats"]["features"]
    assert feats["before_preprocessing"] == 41
    assert feats["after_selection"] == cfg.n_features
    assert feats["after_selection"] < feats["after_encoding_and_variance_filter"]
    assert "num__num_outbound_cmds" in feats["constant_features_dropped"]


def test_split_sizes_are_disjoint_and_roughly_70_15_15():
    res, _ = _run()
    sp = res["stats"]["splits"]
    tr, va, te = (sp["train"]["n_samples"], sp["validation"]["n_samples"],
                  sp["test"]["n_samples"])
    assert tr + va == 6000                       # pool split, no overlap
    assert te == 1400                            # held-out test untouched
    total = tr + va + te
    assert 0.66 < tr / total < 0.74
    assert 0.12 < va / total < 0.18


def test_scaler_fit_on_train_only_no_leakage():
    """Standardised TRAIN numeric cols have ~0 mean / ~1 std; VAL/TEST do not."""
    res, _ = _run()
    names = res["feature_names"]
    num_idx = [i for i, n in enumerate(names) if n.startswith("num__")]
    assert num_idx
    tr_mean = np.abs(res["X_train"][:, num_idx].mean(axis=0)).max()
    assert tr_mean < 1e-4, f"train numeric mean not ~0 ({tr_mean})"
    # val/test were transformed with train stats -> means drift away from 0
    va_mean = np.abs(res["X_val"][:, num_idx].mean(axis=0)).max()
    assert va_mean > tr_mean


def test_smote_balances_training_set_only():
    res, _ = _run(balance="smote")
    yb = res["y_train_balanced"]
    _, counts = np.unique(yb, return_counts=True)
    assert counts.min() == counts.max()                      # perfectly balanced
    assert len(res["y_val"]) == res["stats"]["splits"]["validation"]["n_samples"]
    # unbalanced training labels still available and NOT balanced
    _, raw_counts = np.unique(res["y_train"], return_counts=True)
    assert raw_counts.min() != raw_counts.max()


def test_train_inference_parity(tmp_path=None):
    import tempfile
    d = Path(tempfile.mkdtemp())
    bundle = _bundle()
    cfg = PreprocessConfig(dataset="synth", seed=42, n_features=12)
    res = prepare_dataset(bundle, cfg, artifacts_dir=d)
    pre = FittedPreprocessor.load(d / "preprocessor.joblib")
    x_infer = pre.transform(bundle["test"])
    assert x_infer.shape == res["X_test"].shape
    assert np.array_equal(x_infer, res["X_test"]), "inference transform != training X_test"


def test_determinism():
    a, _ = _run()
    b, _ = _run()
    assert np.array_equal(a["X_train"], b["X_train"])
    assert np.array_equal(a["X_train_balanced"], b["X_train_balanced"])


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    raise SystemExit(1 if failed else 0)
