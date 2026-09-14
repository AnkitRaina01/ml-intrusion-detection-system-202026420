"""
Fast unit tests for the Phase-3 model + evaluation helpers (no disk, no NSL-KDD
download, < a few seconds). Full training/evaluation lives in
scripts/train_models.py and scripts/evaluate_models.py and is run end-to-end
against the real dataset during the phase.

    python -m pytest tests/test_phase3_models.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import models as M          # noqa: E402
from src import evaluation as E      # noqa: E402


def _toy_tabular(n=1500, d=8, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    w = rng.normal(size=d)
    y = (X @ w + 0.4 * rng.normal(size=n) > 0).astype(int)
    return X, y


def _toy_sequences(n=600, T=6, F=5, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, T, F)).astype(np.float32)
    y = (X[:, -1, :].sum(axis=1) + 0.3 * rng.normal(size=n) > 0).astype(np.int64)
    return X, y


def test_expand_grid():
    assert M._expand_grid({}) == [{}]
    g = M._expand_grid({"a": [1, 2], "b": [3]})
    assert {"a": 1, "b": 3} in g and {"a": 2, "b": 3} in g and len(g) == 2


def test_select_sklearn_logreg_uses_validation_only():
    Xtr, ytr = _toy_tabular(1500, seed=0)
    Xva, yva = _toy_tabular(500, seed=99)
    res = M.select_sklearn_model(
        "logreg", {"solver": "lbfgs", "max_iter": 500, "class_weight": "balanced"},
        {"C": [0.1, 1.0]}, Xtr, ytr, Xva, yva, seed=42)
    assert res.best_params["C"] in (0.1, 1.0)
    assert 0.0 <= res.best_val_macro_f1 <= 1.0
    assert len(res.grid) == 2
    # estimator is fitted and usable
    assert hasattr(res.estimator, "predict")
    assert res.estimator.predict(Xva).shape == (500,)


def test_select_sklearn_respects_subsample():
    Xtr, ytr = _toy_tabular(2000, seed=3)
    Xva, yva = _toy_tabular(400, seed=7)
    res = M.select_sklearn_model(
        "svm_rbf", {"kernel": "rbf", "class_weight": "balanced"}, {"C": [1.0]},
        Xtr, ytr, Xva, yva, seed=42, train_subsample=500)
    assert res.notes["train_rows_used"] == 500
    assert res.notes["train_rows_available"] == 2000


def test_compute_metrics_shapes_and_keys():
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0])
    score = np.array([0.1, 0.6, 0.8, 0.7, 0.4, 0.2, 0.9, 0.3])
    m = E.compute_metrics(y_true, y_pred, score)
    for k in ("accuracy", "precision_attack", "recall_attack", "macro_f1",
              "weighted_f1", "roc_auc", "confusion_matrix", "per_class"):
        assert k in m
    assert np.array(m["confusion_matrix"]["matrix"]).shape == (2, 2)
    assert set(m["per_class"]) == {"normal", "attack"}
    assert 0.0 <= m["roc_auc"] <= 1.0


def test_compute_metrics_without_scores_has_no_auc():
    m = E.compute_metrics([0, 1, 1, 0], [0, 1, 0, 0], None)
    assert m["roc_auc"] is None


def test_measure_latency_keys():
    X, _ = _toy_tabular(300)
    lat = E.measure_latency(lambda a: a.sum(axis=1), X, n_repeats=3, warmup=1)
    for k in ("n_samples", "batch_seconds_median", "per_sample_ms_median",
              "throughput_samples_per_s"):
        assert k in lat
    assert lat["n_samples"] == 300 and lat["throughput_samples_per_s"] > 0


def test_lstm_forward_and_predict_proba():
    M.set_global_seed(42)
    X, y = _toy_sequences(400, T=6, F=5)
    model = M.build_lstm(input_features=5, lstm_units=8, dense_units=8, dropout=0.1)
    import torch
    with torch.no_grad():
        logits = model(torch.from_numpy(X[:16]).float())
    assert tuple(logits.shape) == (16,)
    prob = M.lstm_predict_proba(model, X[:50], batch_size=16)
    assert prob.shape == (50,) and prob.min() >= 0.0 and prob.max() <= 1.0


def test_select_lstm_end_to_end_tiny():
    M.set_global_seed(42)
    Xtr, ytr = _toy_sequences(600, seed=1)
    Xva, yva = _toy_sequences(200, seed=2)
    cfg = {"fixed": {"lstm_units": 8, "dropout": 0.1, "dense_units": 8,
                     "learning_rate": 1e-2, "batch_size": 128, "epochs": 3},
           "grid": {"lstm_units": [8]}, "early_stopping_patience": 3,
           "window": 6, "stride": 1}
    res = M.select_lstm(cfg, Xtr, ytr, Xva, yva, seed=42)
    assert 0.0 <= res.best_val_macro_f1 <= 1.0
    assert res.notes["train_windows"] == 600
    assert res.notes["input_features"] == 5


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
