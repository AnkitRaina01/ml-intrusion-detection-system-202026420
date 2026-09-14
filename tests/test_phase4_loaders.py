"""
Fast tests for the Phase-4 dataset adapters (CIC-IDS2017, UNSW-NB15) and the
cross-dataset common flow schema. Synthetic CSVs written to a tmp dir — no
download, runs in well under a second.

    python -m pytest tests/test_phase4_loaders.py -q
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_loaders import (load_cicids2017, load_unsw_nb15,           # noqa: E402
                              make_common_schema_bundle, COMMON_FLOW_FEATURES,
                              align_cicids_to_common, align_unsw_to_common,
                              inspect_bundle)
from src.preprocessing import PreprocessConfig, prepare_dataset          # noqa: E402

_CICIDS_COLS = ["Destination Port", "Flow Duration", "Total Fwd Packets",
                "Total Backward Packets", "Total Length of Fwd Packets",
                "Total Length of Bwd Packets", "Fwd Packet Length Mean",
                "Bwd Packet Length Mean", "Flow Bytes/s", "Flow Packets/s",
                "Down/Up Ratio", "Fwd Header Length.1"]


def _write_cicids(dirpath: Path, n=800, seed=0):
    rng = np.random.default_rng(seed)
    labels = rng.choice(
        ["BENIGN", "DoS Hulk", "DDoS", "PortScan", "FTP-Patator",
         "Web Attack � Brute Force", "Bot", "Heartbleed"],
        n, p=[0.6, 0.12, 0.1, 0.08, 0.04, 0.03, 0.02, 0.01])
    files = []
    for i, part in enumerate(np.array_split(np.arange(n), 3)):
        df = pd.DataFrame({c: rng.normal(100, 30, len(part)) for c in _CICIDS_COLS})
        df["Flow Duration"] = rng.integers(1, 5_000_000, len(part))  # microseconds
        df["Flow Bytes/s"] = rng.choice([1.0, np.inf, 5.0], len(part))  # inf present
        df["Label"] = labels[part]
        p = dirpath / f"part{i}-WorkingHours.pcap_ISCX.csv"
        df.to_csv(p, index=False)
        files.append(p.name)
    return files


def _write_unsw(dirpath: Path, n=900, seed=1):
    def _mk(m, s):
        r = np.random.default_rng(s)
        cats = r.choice(["Normal", "Exploits", "Fuzzers", "DoS", "Generic",
                         "Reconnaissance", "Worms"], m,
                        p=[0.45, 0.2, 0.13, 0.09, 0.08, 0.04, 0.01])
        df = pd.DataFrame({
            "id": np.arange(m),
            "dur": r.random(m), "proto": r.choice(["tcp", "udp", "arp"], m),
            "service": r.choice(["-", "dns", "http"], m),
            "state": r.choice(["FIN", "CON", "INT"], m),
            "spkts": r.integers(1, 100, m), "dpkts": r.integers(0, 80, m),
            "sbytes": r.integers(40, 9000, m), "dbytes": r.integers(0, 9000, m),
            "rate": r.random(m) * 1000, "smean": r.integers(40, 900, m),
            "dmean": r.integers(0, 900, m), "ct_state_ttl": r.integers(0, 6, m),
            "attack_cat": cats,
            "label": (cats != "Normal").astype(int),
        })
        return df
    tr, te = _mk(n, seed), _mk(n // 2, seed + 5)
    tr.to_csv(dirpath / "UNSW_NB15_training-set.csv", index=False)
    te.to_csv(dirpath / "UNSW_NB15_testing-set.csv", index=False)


# --------------------------------------------------------------------------- #
# CIC-IDS2017
# --------------------------------------------------------------------------- #
def test_cicids_bundle_shape_and_labels():
    d = Path(tempfile.mkdtemp())
    files = _write_cicids(d)
    b = load_cicids2017(d, sample_size=500, keep_all_below=50, seed=42, files=files)
    for k in ("train", "test", "feature_columns", "categorical_columns",
              "numeric_columns", "categories", "attack_category_map", "dataset_meta"):
        assert k in b
    assert b["categorical_columns"] == []
    # identifier / duplicate column dropped
    assert "Fwd Header Length.1" not in b["feature_columns"]
    for split in ("train", "test"):
        df = b[split]
        assert {"label", "attack_category", "binary_label"} <= set(df.columns)
        assert set(df["binary_label"].unique()) <= {0, 1}
        # benign rows -> binary_label 0
        assert (df.loc[df["label"] == "benign", "binary_label"] == 0).all()
        # every attack_category is a known coarse class
        assert set(df["attack_category"]) <= set(b["categories"])
    # corrupted en-dash label was normalised and mapped
    assert "web_attack" in set(b["train"]["attack_category"]) | set(b["test"]["attack_category"])


def test_cicids_deterministic():
    d = Path(tempfile.mkdtemp())
    files = _write_cicids(d)
    b1 = load_cicids2017(d, sample_size=500, keep_all_below=50, seed=42, files=files)
    b2 = load_cicids2017(d, sample_size=500, keep_all_below=50, seed=42, files=files)
    pd.testing.assert_frame_equal(b1["train"], b2["train"])
    pd.testing.assert_frame_equal(b1["test"], b2["test"])


def test_cicids_feeds_prepare_dataset_leakage_safe():
    d = Path(tempfile.mkdtemp())
    files = _write_cicids(d, n=1500)
    b = load_cicids2017(d, sample_size=1200, keep_all_below=40, seed=42, files=files)
    cfg = PreprocessConfig(dataset="cicids2017", target="binary",
                           split_mode="predefined", n_features=6, seed=42)
    res = prepare_dataset(b, cfg)
    for k in ("X_train", "X_val", "X_test", "X_train_balanced"):
        assert np.isfinite(res[k]).all()          # inf/NaN cleaned, imputed on train
    # train numeric columns standardised (fit on train only)
    assert np.abs(res["X_train"].mean(axis=0)).max() < 1e-3
    assert np.abs(res["X_val"].mean(axis=0)).max() > np.abs(res["X_train"].mean(axis=0)).max()


# --------------------------------------------------------------------------- #
# UNSW-NB15
# --------------------------------------------------------------------------- #
def test_unsw_bundle_shape_and_dedup():
    d = Path(tempfile.mkdtemp())
    _write_unsw(d)
    b = load_unsw_nb15(d, seed=42)
    assert "id" not in b["feature_columns"]
    assert set(b["categorical_columns"]) == {"proto", "service", "state"}
    assert "proto" not in b["numeric_columns"]
    for split in ("train", "test"):
        df = b[split]
        assert {"label", "attack_category", "binary_label"} <= set(df.columns)
        # derived binary label matches the 'normal' category
        assert (df.loc[df["attack_category"] == "normal", "binary_label"] == 0).all()
        assert (df.loc[df["attack_category"] != "normal", "binary_label"] == 1).all()
    assert b["dataset_meta"]["train_full_row_duplicates_dropped"] >= 0


def test_unsw_feeds_prepare_dataset():
    d = Path(tempfile.mkdtemp())
    _write_unsw(d, n=1200)
    b = load_unsw_nb15(d, seed=42)
    cfg = PreprocessConfig(dataset="unsw_nb15", target="binary",
                           split_mode="predefined", n_features=8, seed=42)
    res = prepare_dataset(b, cfg)
    assert np.isfinite(res["X_train"]).all() and np.isfinite(res["X_test"]).all()
    assert res["X_train"].shape[1] == 8


# --------------------------------------------------------------------------- #
# common flow schema (cross-dataset)
# --------------------------------------------------------------------------- #
def test_common_schema_alignment():
    d1, d2 = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp())
    files = _write_cicids(d1)
    _write_unsw(d2)
    cic = load_cicids2017(d1, sample_size=500, keep_all_below=50, seed=42, files=files)
    unsw = load_unsw_nb15(d2, seed=42)

    a_cic = align_cicids_to_common(cic["train"])
    a_unsw = align_unsw_to_common(unsw["train"])
    for a in (a_cic, a_unsw):
        assert list(COMMON_FLOW_FEATURES) == [c for c in a.columns
                                              if c in COMMON_FLOW_FEATURES]
        assert np.isfinite(a[COMMON_FLOW_FEATURES].to_numpy()).all()  # no inf/NaN
        assert {"binary_label"} <= set(a.columns)

    cb = make_common_schema_bundle(cic, "cicids2017")
    assert cb["feature_columns"] == list(COMMON_FLOW_FEATURES)
    assert cb["categorical_columns"] == []
    rep = inspect_bundle(cb)
    assert rep["train"]["n_feature_columns"] == len(COMMON_FLOW_FEATURES)

    # a source-fitted preprocessor can transform the *other* dataset
    cfg = PreprocessConfig(dataset="cicids2017_common", target="binary",
                           split_mode="predefined",
                           n_features=len(COMMON_FLOW_FEATURES), balance="none",
                           seed=42)
    res = prepare_dataset(cb, cfg)
    ub = make_common_schema_bundle(unsw, "unsw_nb15")
    x_cross = res["preprocessor"].transform(ub["test"])
    assert x_cross.shape[1] == len(COMMON_FLOW_FEATURES)
    assert np.isfinite(x_cross).all()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    bad = 0
    for fn in fns:
        try:
            fn(); print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            bad += 1; print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns) - bad}/{len(fns)} passed")
    raise SystemExit(1 if bad else 0)
