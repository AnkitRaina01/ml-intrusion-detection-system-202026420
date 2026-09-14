#!/usr/bin/env python
"""
Phase 3 — train the four required supervised models on NSL-KDD (binary target).

    logreg  random_forest  svm_rbf  lstm

For every model: fit on TRAIN, choose hyper-parameters by macro-F1 on VALIDATION,
save the fitted model and a JSON record of the validation search. The TEST set is
NOT touched here - run ``scripts/evaluate_models.py`` afterwards.

    python scripts/train_models.py                       # uses configs/phase3_nsl_kdd.json
    python scripts/train_models.py --only lstm
    python scripts/train_models.py --config path/to/config.json
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

from src import models as M                                              # noqa: E402

SKLEARN_KEYS = ["logreg", "random_forest", "svm_rbf"]
ALL_KEYS = SKLEARN_KEYS + ["lstm"]


def _load_tabular(processed_dir: Path):
    d = processed_dir
    return {
        "X_train": np.load(d / "X_train.npy"),
        "y_train": np.load(d / "y_train_binary.npy"),
        "X_val": np.load(d / "X_val.npy"),
        "y_val": np.load(d / "y_val_binary.npy"),
    }


def _load_sequences(seq_dir: Path):
    d = seq_dir
    return {
        "X_train": np.load(d / "Xseq_train.npy"),
        "y_train": np.load(d / "yseq_train.npy"),
        "X_val": np.load(d / "Xseq_val.npy"),
        "y_val": np.load(d / "yseq_val.npy"),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "phase3_nsl_kdd.json"))
    ap.add_argument("--only", nargs="+", choices=ALL_KEYS, default=ALL_KEYS)
    args = ap.parse_args(argv)

    cfg = json.loads(Path(args.config).read_text())
    seed = int(cfg["seed"])
    M.set_global_seed(seed)

    processed_dir = REPO_ROOT / cfg["processed_dir"]
    seq_dir = REPO_ROOT / cfg["sequence_dir"]
    models_dir = REPO_ROOT / cfg["models_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": seed, "task": cfg["task"], "dataset": cfg["dataset"],
        "imbalance_strategy": cfg["imbalance_strategy"],
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "config_file": str(Path(args.config)),
        "models": {},
    }
    try:
        import sklearn
        summary["environment"]["scikit_learn"] = sklearn.__version__
    except Exception:
        pass

    tab = _load_tabular(processed_dir) if set(args.only) & set(SKLEARN_KEYS) else None

    for key in [k for k in SKLEARN_KEYS if k in args.only]:
        mc = cfg[{"logreg": "logistic_regression",
                  "random_forest": "random_forest",
                  "svm_rbf": "svm_rbf"}[key]]
        print(f"\n=== training {key} ===")
        res = M.select_sklearn_model(
            key, mc.get("fixed", {}), mc.get("grid", {}),
            tab["X_train"], tab["y_train"], tab["X_val"], tab["y_val"], seed,
            train_subsample=mc.get("train_subsample"),
        )
        path = models_dir / f"{key}.joblib"
        joblib.dump(res.estimator, path)
        rec = {
            "model_file": path.name,
            "best_params": res.best_params,
            "best_val_macro_f1": res.best_val_macro_f1,
            "train_seconds": res.train_seconds,
            "validation_search": res.grid,
            "fixed_params": mc.get("fixed", {}),
            "notes": res.notes,
        }
        (models_dir / f"{key}_selection.json").write_text(json.dumps(rec, indent=2))
        summary["models"][key] = {k: rec[k] for k in
                                  ("best_params", "best_val_macro_f1",
                                   "train_seconds", "notes")}
        print(f"  best {res.best_params}  val macro-F1 = {res.best_val_macro_f1:.4f}"
              f"  ({res.train_seconds:.1f}s)")

    if "lstm" in args.only:
        print("\n=== training lstm (PyTorch) ===")
        seq = _load_sequences(seq_dir)
        lcfg = cfg["lstm"]
        res = M.select_lstm(lcfg, seq["X_train"], seq["y_train"],
                            seq["X_val"], seq["y_val"], seed)
        import torch
        arch = {
            "framework": "pytorch", "torch_version": torch.__version__,
            "window": lcfg["window"], "stride": lcfg["stride"],
            "input_features": int(seq["X_train"].shape[2]),
            "lstm_units": res.best_params.get("lstm_units",
                                              lcfg["fixed"]["lstm_units"]),
            "dense_units": lcfg["fixed"]["dense_units"],
            "dropout": lcfg["fixed"]["dropout"],
        }
        torch.save(res.estimator.state_dict(), models_dir / "lstm.pt")
        (models_dir / "lstm_arch.json").write_text(json.dumps(arch, indent=2))
        rec = {
            "model_file": "lstm.pt", "arch": arch,
            "best_params": res.best_params,
            "best_val_macro_f1": res.best_val_macro_f1,
            "train_seconds": res.train_seconds,
            "validation_search": res.grid,
            "fixed_params": lcfg["fixed"],
            "notes": res.notes,
        }
        (models_dir / "lstm_selection.json").write_text(json.dumps(rec, indent=2))
        summary["models"]["lstm"] = {
            "best_params": res.best_params,
            "best_val_macro_f1": res.best_val_macro_f1,
            "train_seconds": res.train_seconds,
            "best_epoch": res.notes.get("best_epoch"),
            "arch": arch,
        }
        print(f"  best {res.best_params}  val macro-F1 = {res.best_val_macro_f1:.4f}"
              f"  best_epoch={res.notes.get('best_epoch')}  ({res.train_seconds:.1f}s)")

    # merge with any previous partial run so a sequence of --only calls still
    # produces one complete training_summary.json
    spath = models_dir / "training_summary.json"
    if spath.exists() and set(args.only) != set(ALL_KEYS):
        try:
            prev = json.loads(spath.read_text())
            merged = dict(prev.get("models", {}))
            merged.update(summary["models"])
            summary["models"] = merged
            summary["note"] = "assembled from multiple --only runs"
        except Exception:
            pass
    spath.write_text(json.dumps(summary, indent=2))
    print(f"\nsaved models + selection reports -> {models_dir}")
    print("next: python scripts/evaluate_models.py\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
