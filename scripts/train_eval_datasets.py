#!/usr/bin/env python
"""
Phase 4 — WITHIN-DATASET training + evaluation for CIC-IDS2017 and UNSW-NB15.

Trains the three sklearn models that Phase 3 ran on NSL-KDD
(logreg, random_forest, svm_rbf), using the *same* methodology and the *same*
metric code (``src/evaluation.py``), so the per-dataset tables are directly
comparable. The LSTM is deliberately excluded for the flow datasets — see
DEVELOPMENT_NOTES.md section 11.4.

    python scripts/train_eval_datasets.py --dataset cicids2017
    python scripts/train_eval_datasets.py --dataset unsw_nb15

Reads ``data/processed/<ds>/*.npy`` (produced by scripts/prepare_datasets.py) and
writes, under ``results/<ds>/``:
    metrics.json / metrics.csv          (machine-readable — committed evidence)
    model_comparison.md                 (human-readable table — committed)
    plots/confusion_matrix_<model>.png, roc_curves.png, model_comparison.png
and, under ``models/<ds>/``: ``<model>.joblib`` + ``<model>_selection.json``.

"Within-dataset" = trained and tested on the same dataset's own splits. Every
number here is computed from real predictions on that dataset's held-out test
split. Cross-dataset generalisation is a separate script
(scripts/cross_dataset_experiment.py).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import models as M                                             # noqa: E402
from src import evaluation as E                                        # noqa: E402

SKLEARN_KEYS = ["logreg", "random_forest", "svm_rbf"]
PRETTY = {"logreg": "Logistic Regression", "random_forest": "Random Forest",
          "svm_rbf": "SVM (RBF)"}
CFG_SECTION = {"logreg": "logistic_regression", "random_forest": "random_forest",
               "svm_rbf": "svm_rbf"}
DISPLAY = {"cicids2017": "CIC-IDS2017", "unsw_nb15": "UNSW-NB15",
           "nsl_kdd": "NSL-KDD"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True,
                    choices=["cicids2017", "unsw_nb15", "nsl_kdd"])
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "phase4_models.json"))
    ap.add_argument("--latency-repeats", type=int, default=3)
    ap.add_argument("--latency-max-rows", type=int, default=20_000,
                    help="cap the row count used for the latency timing loop "
                         "(RBF-SVM predict on a large test set is very slow); "
                         "per-sample figures are unaffected by the cap")
    ap.add_argument("--only", nargs="+", choices=SKLEARN_KEYS, default=SKLEARN_KEYS)
    args = ap.parse_args(argv)

    cfg = json.loads(Path(args.config).read_text())
    seed = int(cfg["seed"])
    M.set_global_seed(seed)

    ds = args.dataset
    processed = REPO_ROOT / "data" / "processed" / ds
    models_dir = REPO_ROOT / "models" / ds
    results_dir = REPO_ROOT / "results" / ds
    plots_dir = results_dir / "plots"
    models_dir.mkdir(parents=True, exist_ok=True)

    X_train = np.load(processed / "X_train.npy")
    y_train = np.load(processed / "y_train_binary.npy")
    X_val = np.load(processed / "X_val.npy")
    y_val = np.load(processed / "y_val_binary.npy")
    X_test = np.load(processed / "X_test.npy")
    y_test = np.load(processed / "y_test_binary.npy")
    feat_names = json.loads((processed / "feature_names.json").read_text())

    print(f"{DISPLAY[ds]}: train {X_train.shape}  val {X_val.shape}  test {X_test.shape}")

    per_model: dict = {}
    roc_curves: dict = {}
    summary_models: dict = {}

    for key in [k for k in SKLEARN_KEYS if k in args.only]:
        mc = cfg[CFG_SECTION[key]]
        print(f"\n=== training {key} on {ds} ===")
        res = M.select_sklearn_model(
            key, mc.get("fixed", {}), mc.get("grid", {}),
            X_train, y_train, X_val, y_val, seed,
            train_subsample=mc.get("train_subsample"))
        mp = models_dir / f"{key}.joblib"
        joblib.dump(res.estimator, mp)
        sel = {
            "model_file": mp.name, "dataset": ds,
            "best_params": res.best_params,
            "best_val_macro_f1": res.best_val_macro_f1,
            "train_seconds": res.train_seconds,
            "validation_search": res.grid,
            "fixed_params": mc.get("fixed", {}),
            "notes": res.notes,
        }
        (models_dir / f"{key}_selection.json").write_text(json.dumps(sel, indent=2))
        print(f"  best {res.best_params}  val macro-F1 = {res.best_val_macro_f1:.4f}"
              f"  ({res.train_seconds:.1f}s)")

        est = res.estimator
        y_pred = est.predict(X_test)
        y_score = M.sklearn_scores(est, X_test)
        met = E.compute_metrics(y_test, y_pred, y_score)
        X_lat = X_test[:args.latency_max_rows]
        lat = E.measure_latency(lambda X: est.predict(X), X_lat,
                                n_repeats=args.latency_repeats)
        lat["timing_rows"] = int(len(X_lat))
        per_model[PRETTY[key]] = {
            "metrics": met, "latency": lat,
            "train_seconds": res.train_seconds,
            "hyperparameters": {**mc.get("fixed", {}), **res.best_params},
            "val_macro_f1": res.best_val_macro_f1,
            "test_set": {"n_samples": int(len(y_test)),
                         "source": f"{ds} held-out test split"},
            "score_source": ("predict_proba[:,1]" if hasattr(est, "predict_proba")
                             else "decision_function"),
        }
        fpr, tpr, auc = E.roc_points(y_test, y_score)
        roc_curves[PRETTY[key]] = (fpr, tpr, auc)
        summary_models[key] = {k: sel[k] for k in
                               ("best_params", "best_val_macro_f1",
                                "train_seconds", "notes")}
        print(f"  test: acc={met['accuracy']:.4f} macroF1={met['macro_f1']:.4f} "
              f"AUC={met['roc_auc']:.4f}")

    extra = {
        "phase": 4, "evaluation_type": "within-dataset",
        "dataset": ds, "dataset_display": DISPLAY[ds], "task": "binary",
        "seed": seed,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_selected_features": len(feat_names),
        "selected_features": feat_names,
        "methodology": (
            "fit on train split, hyper-parameters chosen by macro-F1 on the "
            "validation split, test split evaluated once here; no refit on "
            "train+val; class_weight='balanced'. Identical to Phase 3 "
            "(NSL-KDD) so the datasets are directly comparable."),
        "imbalance_strategy": cfg["imbalance_strategy"],
        "environment": {"python": platform.python_version(),
                        "numpy": np.__version__},
        "notes": {
            "svm_rbf": "trained on a seeded stratified subsample; ROC-AUC from decision_function",
            "lstm": ("excluded for flow datasets - see DEVELOPMENT_NOTES.md 11.4"),
        },
    }
    try:
        import sklearn
        extra["environment"]["scikit_learn"] = sklearn.__version__
    except Exception:
        pass

    paths = E.write_metric_tables(per_model, results_dir, extra=extra)
    print(f"\nwrote {paths['json']}  and  {paths['csv']}")

    (models_dir / "training_summary.json").write_text(json.dumps({
        "generated_utc": extra["generated_utc"], "seed": seed, "dataset": ds,
        "task": "binary", "imbalance_strategy": cfg["imbalance_strategy"],
        "environment": extra["environment"], "models": summary_models,
    }, indent=2))

    # ---- plots ---- #
    key_by_pretty = {v: k for k, v in PRETTY.items()}
    for name, blob in per_model.items():
        cm = blob["metrics"]["confusion_matrix"]
        E.plot_confusion_matrix(
            cm["matrix"], cm["labels"], f"Confusion matrix — {name} ({DISPLAY[ds]})",
            plots_dir / f"confusion_matrix_{key_by_pretty[name]}.png")
    if roc_curves:
        E.plot_roc_curves(roc_curves, plots_dir / "roc_curves.png",
                          title=f"ROC curves — {DISPLAY[ds]} test split")
    rows = [{"model": n,
             "accuracy": b["metrics"]["accuracy"],
             "precision_attack": b["metrics"]["precision_attack"],
             "recall_attack": b["metrics"]["recall_attack"],
             "f1_attack": b["metrics"]["f1_attack"],
             "macro_f1": b["metrics"]["macro_f1"],
             "roc_auc": b["metrics"]["roc_auc"]}
            for n, b in per_model.items()]
    if rows:
        E.plot_metric_bars(rows, ["accuracy", "precision_attack", "recall_attack",
                                  "macro_f1", "roc_auc"],
                           plots_dir / "model_comparison.png",
                           f"Model comparison — {DISPLAY[ds]} test split")
    print(f"wrote plots -> {plots_dir}")

    _write_markdown(per_model, results_dir / "model_comparison.md", extra)
    print(f"wrote {results_dir / 'model_comparison.md'}\n")
    return 0


def _write_markdown(per_model: dict, path: Path, extra: dict) -> None:
    L = [f"# Phase 4 — within-dataset model comparison "
         f"({extra['dataset_display']}, {extra['task']} task)",
         "",
         f"_Generated {extra['generated_utc']} · seed {extra['seed']}_",
         "",
         f"> **Within-dataset evaluation.** All figures are computed from real "
         f"predictions on the {extra['dataset_display']} held-out **test split** "
         f"by `scripts/train_eval_datasets.py`. "
         f"Cross-dataset generalisation is reported separately in "
         f"`results/cross_dataset/generalization.md`.",
         "",
         extra["methodology"], "",
         f"Features: {extra['n_selected_features']} selected by mutual information "
         f"on the training split only (leakage-safe).", "",
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
    L += ["", "## Per-class detail & confusion matrices",
          "See `metrics.json` (`per_class`, `confusion_matrix`) and "
          "`plots/confusion_matrix_*.png`.", "",
          "## Notes", ""]
    for k, v in extra.get("notes", {}).items():
        L.append(f"- **{k}**: {v}")
    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
