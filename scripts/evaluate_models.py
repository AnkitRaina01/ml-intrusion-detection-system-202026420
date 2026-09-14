#!/usr/bin/env python
"""
Phase 3 — unified final evaluation of the four trained models on the NSL-KDD
held-out test set (KDDTest+).  Run AFTER ``scripts/train_models.py``.

Produces, under ``results/nsl_kdd/``:
    metrics.json                machine-readable, full detail
    metrics.csv                 flat summary (one row per model)
    model_comparison.md         human-readable table
    plots/confusion_matrix_<model>.png   (x4)
    plots/roc_curves.png
    plots/model_comparison.png           (accuracy / precision / recall / macro-F1 / ROC-AUC)
    plots/precision_recall_f1.png

Every number is computed here from real predictions on the test set. Nothing is
taken from the repository README.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import models as M                                              # noqa: E402
from src import evaluation as E                                         # noqa: E402

SKLEARN_KEYS = ["logreg", "random_forest", "svm_rbf"]
PRETTY = {"logreg": "Logistic Regression", "random_forest": "Random Forest",
          "svm_rbf": "SVM (RBF)", "lstm": "LSTM"}


def _load_lstm(models_dir: Path):
    import torch
    arch = json.loads((models_dir / "lstm_arch.json").read_text())
    model = M.build_lstm(arch["input_features"], arch["lstm_units"],
                         arch["dense_units"], arch["dropout"])
    model.load_state_dict(torch.load(models_dir / "lstm.pt", weights_only=True))
    model.eval()
    return model, arch


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "phase3_nsl_kdd.json"))
    ap.add_argument("--latency-repeats", type=int, default=5)
    args = ap.parse_args(argv)

    cfg = json.loads(Path(args.config).read_text())
    M.set_global_seed(int(cfg["seed"]))
    processed = REPO_ROOT / cfg["processed_dir"]
    seq_dir = REPO_ROOT / cfg["sequence_dir"]
    models_dir = REPO_ROOT / cfg["models_dir"]
    results_dir = REPO_ROOT / cfg["results_dir"]
    plots_dir = results_dir / "plots"

    # ---- test data (untouched until now) ------------------------------- #
    X_test = np.load(processed / "X_test.npy")
    y_test = np.load(processed / "y_test_binary.npy")
    Xseq_test = np.load(seq_dir / "Xseq_test.npy")
    yseq_test = np.load(seq_dir / "yseq_test.npy")
    seq_meta = json.loads((seq_dir / "sequence_metadata.json").read_text())

    train_summary = json.loads((models_dir / "training_summary.json").read_text())
    per_model: dict = {}
    roc_curves: dict = {}

    # ---- sklearn models ---------------------------------------------- #
    for key in SKLEARN_KEYS:
        mp = models_dir / f"{key}.joblib"
        if not mp.exists():
            print(f"skip {key}: {mp} missing")
            continue
        est = joblib.load(mp)
        y_pred = est.predict(X_test)
        y_score = M.sklearn_scores(est, X_test)
        met = E.compute_metrics(y_test, y_pred, y_score)
        lat = E.measure_latency(lambda X: est.predict(X), X_test,
                                n_repeats=args.latency_repeats)
        sel = json.loads((models_dir / f"{key}_selection.json").read_text())
        per_model[PRETTY[key]] = {
            "metrics": met, "latency": lat,
            "train_seconds": sel.get("train_seconds"),
            "hyperparameters": {**sel.get("fixed_params", {}), **sel.get("best_params", {})},
            "val_macro_f1": sel.get("best_val_macro_f1"),
            "test_set": {"n_samples": int(len(y_test)),
                         "source": "KDDTest+ (all records)"},
            "score_source": ("predict_proba[:,1]" if hasattr(est, "predict_proba")
                             else "decision_function"),
        }
        fpr, tpr, auc = E.roc_points(y_test, y_score)
        roc_curves[PRETTY[key]] = (fpr, tpr, auc)
        print(f"{PRETTY[key]:20s} acc={met['accuracy']:.4f} "
              f"macroF1={met['macro_f1']:.4f} AUC={met['roc_auc']:.4f} "
              f"{lat['per_sample_ms_median']*1e3:.1f}us/sample")

    # ---- LSTM ------------------------------------------------------- #
    if (models_dir / "lstm.pt").exists():
        model, arch = _load_lstm(models_dir)
        prob = M.lstm_predict_proba(model, Xseq_test)
        y_pred = (prob >= 0.5).astype(int)
        met = E.compute_metrics(yseq_test, y_pred, prob)
        lat = E.measure_latency(lambda X: M.lstm_predict_proba(model, X),
                                Xseq_test, n_repeats=args.latency_repeats)
        lsel = json.loads((models_dir / "lstm_selection.json").read_text())
        per_model["LSTM"] = {
            "metrics": met, "latency": lat,
            "train_seconds": lsel.get("train_seconds"),
            "hyperparameters": {**lsel.get("fixed_params", {}),
                                **lsel.get("best_params", {}),
                                "window": arch["window"], "stride": arch["stride"]},
            "val_macro_f1": lsel.get("best_val_macro_f1"),
            "test_set": {
                "n_samples": int(len(yseq_test)),
                "source": (f"KDDTest+ windowed (W={arch['window']}); "
                           f"{len(yseq_test)} of {seq_meta['record_counts']['test']} "
                           f"records can anchor a full window"),
            },
            "score_source": "sigmoid(logit)",
            "sequence_representation": seq_meta["limitations"],
        }
        fpr, tpr, auc = E.roc_points(yseq_test, prob)
        roc_curves["LSTM"] = (fpr, tpr, auc)
        print(f"{'LSTM':20s} acc={met['accuracy']:.4f} "
              f"macroF1={met['macro_f1']:.4f} AUC={met['roc_auc']:.4f} "
              f"{lat['per_sample_ms_median']*1e3:.1f}us/sample")

    # ---- machine-readable output ----------------------------------- #
    extra = {
        "phase": 3, "dataset": cfg["dataset"], "task": cfg["task"],
        "seed": cfg["seed"],
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "methodology": (
            "fit on train, hyper-parameters chosen by macro-F1 on validation, "
            "test (KDDTest+) evaluated once here; no refit on train+val; "
            "class_weight='balanced' (LSTM: pos_weight in BCE loss)."),
        "imbalance_strategy": cfg["imbalance_strategy"],
        "environment": train_summary.get("environment", {}),
        "notes": {
            "svm_rbf": "trained on a seeded stratified subsample; ROC-AUC from decision_function",
            "lstm": ("PyTorch; evaluated on windowed KDDTest+ "
                     f"({len(yseq_test)}/{seq_meta['record_counts']['test']} records); "
                     "see sequence_metadata.json / DEVELOPMENT_NOTES 10.3"),
        },
    }
    paths = E.write_metric_tables(per_model, results_dir, extra=extra)
    print(f"\nwrote {paths['json']}  and  {paths['csv']}")

    # ---- plots --------------------------------------------------- #
    key_by_pretty = {v: k for k, v in PRETTY.items()}
    for name, blob in per_model.items():
        cm = blob["metrics"]["confusion_matrix"]
        E.plot_confusion_matrix(
            cm["matrix"], cm["labels"], f"Confusion matrix — {name}",
            plots_dir / f"confusion_matrix_{key_by_pretty[name]}.png")
    if roc_curves:
        E.plot_roc_curves(roc_curves, plots_dir / "roc_curves.png")
    rows = [{"model": n,
             "accuracy": b["metrics"]["accuracy"],
             "precision_attack": b["metrics"]["precision_attack"],
             "recall_attack": b["metrics"]["recall_attack"],
             "f1_attack": b["metrics"]["f1_attack"],
             "macro_f1": b["metrics"]["macro_f1"],
             "roc_auc": b["metrics"]["roc_auc"]}
            for n, b in per_model.items()]
    E.plot_metric_bars(rows, ["accuracy", "precision_attack", "recall_attack",
                              "macro_f1", "roc_auc"],
                       plots_dir / "model_comparison.png",
                       "Model comparison — NSL-KDD test set")
    E.plot_metric_bars(rows, ["precision_attack", "recall_attack", "f1_attack",
                              "macro_f1"],
                       plots_dir / "precision_recall_f1.png",
                       "Precision / Recall / F1 — NSL-KDD test set")
    print(f"wrote plots -> {plots_dir}")

    # ---- markdown summary --------------------------------------- #
    _write_markdown(per_model, results_dir / "model_comparison.md", extra)
    print(f"wrote {results_dir / 'model_comparison.md'}\n")
    return 0


def _write_markdown(per_model: dict, path: Path, extra: dict) -> None:
    L = [f"# Phase 3 — model comparison ({extra['dataset']}, {extra['task']} task)",
         "",
         f"_Generated {extra['generated_utc']} · seed {extra['seed']}_",
         "",
         "> All figures computed from real predictions on the held-out NSL-KDD "
         "test set (KDDTest+) by `scripts/evaluate_models.py`. "
         "No values are taken from the repository README.",
         "",
         extra["methodology"], "",
         "| model | test n | accuracy | precision (attack) | recall (attack) | "
         "F1 (attack) | macro-F1 | ROC-AUC | latency (ms/sample) | throughput (/s) | "
         "train (s) | val macro-F1 |",
         "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for name, b in per_model.items():
        m, lat = b["metrics"], b["latency"]
        L.append(
            f"| {name} | {m['n_samples']:,} | {m['accuracy']:.4f} | "
            f"{m['precision_attack']:.4f} | {m['recall_attack']:.4f} | "
            f"{m['f1_attack']:.4f} | {m['macro_f1']:.4f} | "
            f"{(m['roc_auc'] if m['roc_auc'] is not None else float('nan')):.4f} | "
            f"{lat['per_sample_ms_median']:.4f} | "
            f"{lat['throughput_samples_per_s']:,.0f} | "
            f"{(b.get('train_seconds') or 0):.1f} | "
            f"{(b.get('val_macro_f1') or float('nan')):.4f} |")
    L += ["",
          "## Per-class detail & confusion matrices",
          "See `metrics.json` (`per_class`, `confusion_matrix`) and "
          "`plots/confusion_matrix_*.png`.",
          "",
          "## Notes", ""]
    for k, v in extra.get("notes", {}).items():
        L.append(f"- **{k}**: {v}")
    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
