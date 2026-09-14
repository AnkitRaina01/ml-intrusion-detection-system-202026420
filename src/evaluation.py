"""
Shared evaluation utilities (Phase 3).

Pure, model-agnostic helpers:
  * ``compute_metrics``  - accuracy / precision / recall / macro-F1 / per-class /
                           confusion matrix / ROC-AUC from labels (+ optional scores)
  * ``measure_latency``  - wall-clock inference latency of a predict callable
  * plotting helpers for confusion matrices, ROC curves and metric-comparison bars
  * ``write_metric_tables`` - machine-readable ``metrics.json`` + ``metrics.csv``

Nothing here trains a model or touches the test set on its own - callers do that.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,        # noqa: E402
                             precision_recall_fscore_support, precision_score,
                             recall_score, roc_auc_score, roc_curve)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def compute_metrics(y_true: Sequence[int],
                    y_pred: Sequence[int],
                    y_score: Optional[Sequence[float]] = None,
                    *,
                    pos_label: int = 1,
                    labels: Sequence[int] = (0, 1),
                    class_names: Sequence[str] = ("normal", "attack")) -> dict:
    """
    Binary classification metrics. ``y_score`` = probability / decision-function
    value for the positive class (used only for ROC-AUC); pass ``None`` to skip.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, pos_label=pos_label,
                                 zero_division=0))
    rec = float(recall_score(y_true, y_pred, pos_label=pos_label,
                             zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted",
                                 zero_division=0))
    binary_f1 = float(f1_score(y_true, y_pred, pos_label=pos_label,
                               zero_division=0))

    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=list(labels), zero_division=0)
    per_class = {
        class_names[i]: {"precision": float(p[i]), "recall": float(r[i]),
                         "f1": float(f[i]), "support": int(s[i])}
        for i in range(len(labels))
    }

    cm = confusion_matrix(y_true, y_pred, labels=list(labels))

    roc_auc = None
    if y_score is not None and len(np.unique(y_true)) == 2:
        roc_auc = float(roc_auc_score(y_true, np.asarray(y_score)))

    return {
        "n_samples": int(len(y_true)),
        "accuracy": acc,
        "precision_attack": prec,
        "recall_attack": rec,
        "f1_attack": binary_f1,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "roc_auc": roc_auc,
        "confusion_matrix": {
            "labels": list(class_names),
            "matrix": cm.tolist(),
            "layout": "rows = true class, cols = predicted class",
        },
        "per_class": per_class,
    }


def roc_points(y_true: Sequence[int], y_score: Sequence[float]
               ) -> Tuple[np.ndarray, np.ndarray, float]:
    fpr, tpr, _ = roc_curve(np.asarray(y_true).astype(int), np.asarray(y_score))
    return fpr, tpr, float(roc_auc_score(y_true, y_score))


# --------------------------------------------------------------------------- #
# latency
# --------------------------------------------------------------------------- #
def measure_latency(predict_fn: Callable[[np.ndarray], object],
                    X: np.ndarray,
                    *,
                    n_repeats: int = 5,
                    warmup: int = 2) -> dict:
    """
    Time ``predict_fn(X)`` over the whole array, ``n_repeats`` times, after
    ``warmup`` untimed calls. Reports batch and per-sample figures. Excludes any
    preprocessing / windowing done before the call.
    """
    for _ in range(max(0, warmup)):
        predict_fn(X)
    times: List[float] = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        predict_fn(X)
        times.append(time.perf_counter() - t0)
    times = np.asarray(times)
    n = int(len(X))
    med = float(np.median(times))
    return {
        "n_samples": n,
        "n_repeats": int(n_repeats),
        "batch_seconds_median": med,
        "batch_seconds_min": float(times.min()),
        "per_sample_ms_median": float(med / n * 1e3),
        "per_sample_ms_min": float(times.min() / n * 1e3),
        "throughput_samples_per_s": float(n / med),
    }


# --------------------------------------------------------------------------- #
# plots
# --------------------------------------------------------------------------- #
def plot_confusion_matrix(cm: Sequence[Sequence[int]],
                          class_names: Sequence[str],
                          title: str, path: str | Path) -> Path:
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)), labels=class_names)
    ax.set_yticks(range(len(class_names)), labels=class_names)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    thresh = cm.max() / 2.0 if cm.max() else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150); plt.close(fig)
    return path


def plot_roc_curves(curves: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
                    path: str | Path,
                    title: str = "ROC curves (NSL-KDD test set)") -> Path:
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    for name, (fpr, tpr, auc) in curves.items():
        ax.plot(fpr, tpr, lw=1.8, label=f"{name} (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="chance")
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.set_title(title); ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.grid(alpha=0.3)
    fig.tight_layout()
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150); plt.close(fig)
    return path


def plot_metric_bars(rows: List[dict], metrics: Sequence[str],
                     path: str | Path, title: str) -> Path:
    names = [r["model"] for r in rows]
    x = np.arange(len(names))
    width = 0.8 / len(metrics)
    fig, ax = plt.subplots(figsize=(max(6.5, 1.6 * len(names)), 4.8))
    for i, m in enumerate(metrics):
        vals = [float(r.get(m, np.nan)) for r in rows]
        ax.bar(x + (i - (len(metrics) - 1) / 2) * width, vals, width,
               label=m)
    ax.set_xticks(x, labels=names, rotation=15, ha="right")
    ax.set_ylim(0, 1.02); ax.set_ylabel("score"); ax.set_title(title)
    ax.legend(fontsize=9, ncol=len(metrics)); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150); plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# machine-readable output
# --------------------------------------------------------------------------- #
_CSV_FIELDS = ["model", "n_test_samples", "accuracy", "precision_attack",
               "recall_attack", "f1_attack", "macro_f1", "weighted_f1",
               "roc_auc", "per_sample_ms_median", "throughput_samples_per_s",
               "train_seconds", "selected_hyperparameters"]


def write_metric_tables(per_model: Dict[str, dict],
                        results_dir: str | Path,
                        extra: Optional[dict] = None) -> Dict[str, Path]:
    """
    ``per_model`` : {model_name: {"metrics": <compute_metrics dict>,
                                  "latency": <measure_latency dict>,
                                  "train_seconds": float,
                                  "hyperparameters": dict, ...}}
    Writes ``metrics.json`` (full) and ``metrics.csv`` (flat summary).
    """
    results_dir = Path(results_dir); results_dir.mkdir(parents=True, exist_ok=True)
    payload = {"models": per_model}
    if extra:
        payload.update(extra)
    json_path = results_dir / "metrics.json"
    json_path.write_text(json.dumps(payload, indent=2, default=str))

    import csv
    csv_path = results_dir / "metrics.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
        w.writeheader()
        for name, blob in per_model.items():
            met = blob.get("metrics", {})
            lat = blob.get("latency", {})
            w.writerow({
                "model": name,
                "n_test_samples": met.get("n_samples"),
                "accuracy": _r(met.get("accuracy")),
                "precision_attack": _r(met.get("precision_attack")),
                "recall_attack": _r(met.get("recall_attack")),
                "f1_attack": _r(met.get("f1_attack")),
                "macro_f1": _r(met.get("macro_f1")),
                "weighted_f1": _r(met.get("weighted_f1")),
                "roc_auc": _r(met.get("roc_auc")),
                "per_sample_ms_median": _r(lat.get("per_sample_ms_median"), 5),
                "throughput_samples_per_s": _r(lat.get("throughput_samples_per_s"), 1),
                "train_seconds": _r(blob.get("train_seconds"), 2),
                "selected_hyperparameters": json.dumps(blob.get("hyperparameters", {})),
            })
    return {"json": json_path, "csv": csv_path}


def _r(v, nd: int = 6):
    return None if v is None else round(float(v), nd)
