#!/usr/bin/env python
"""
Phase 4 — CROSS-DATASET generalisation experiment: CIC-IDS2017  <->  UNSW-NB15.

Why only these two: both are bidirectional-flow datasets, so a defensible common
feature space exists. NSL-KDD is a connection-record dataset with no comparable
flow-timing features and is therefore NOT forced into this alignment (documented,
see results/cross_dataset/common_schema.md).

Protocol
--------
1. Load CIC-IDS2017 and UNSW-NB15 (their own adapters).
2. Project BOTH onto the 12-feature common flow schema
   (src.data_loaders.COMMON_FLOW_FEATURES). Derived rates are recomputed from
   primitives on both sides; CIC-IDS2017 duration (µs) -> seconds. This is a
   LOSSY approximation.
3. For each dataset: leakage-safe prep (impute+scale fit on TRAIN split only;
   SMOTE on train only) via the SAME src.preprocessing.prepare_dataset.
4. Train logreg / random_forest / svm_rbf on each dataset's own common-space
   training split (class_weight='balanced').
5. Evaluate every model FOUR ways with the SAME metric code:
       within  CIC-IDS2017   (CIC model  -> CIC test)
       within  UNSW-NB15     (UNSW model -> UNSW test)
       cross   CIC -> UNSW   (CIC model, UNSW test transformed by CIC's scaler)
       cross   UNSW -> CIC   (UNSW model, CIC test transformed by UNSW's scaler)

Outputs (results/cross_dataset/):
    metrics.json          every cell, machine-readable  (committed evidence)
    generalization.md     within-vs-cross table + the generalisation gap
    common_schema.md      the exact feature mapping / unit conversions
    plots/generalization_macro_f1.png

    python scripts/cross_dataset_experiment.py
    python scripts/cross_dataset_experiment.py --cicids-sample-size 150000
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_loaders import (load_cicids2017, load_unsw_nb15,          # noqa: E402
                              make_common_schema_bundle,
                              COMMON_FLOW_FEATURES, COMMON_FLOW_MAPPING)
from src.preprocessing import PreprocessConfig, prepare_dataset         # noqa: E402
from src import models as M                                            # noqa: E402
from src import evaluation as E                                        # noqa: E402

SKLEARN_KEYS = ["logreg", "random_forest", "svm_rbf"]
PRETTY = {"logreg": "Logistic Regression", "random_forest": "Random Forest",
          "svm_rbf": "SVM (RBF)"}
MODEL_CFG = {
    "logreg": {"fixed": {"solver": "lbfgs", "max_iter": 2000,
                         "class_weight": "balanced"},
               "grid": {"C": [0.01, 0.1, 1.0, 10.0]}},
    "random_forest": {"fixed": {"class_weight": "balanced", "n_jobs": -1},
                      "grid": {"n_estimators": [200, 400], "max_depth": [None, 20, 40]}},
    "svm_rbf": {"fixed": {"kernel": "rbf", "class_weight": "balanced",
                          "gamma": "scale", "cache_size": 512},
                "grid": {"C": [1.0, 10.0]}, "train_subsample": 20000},
}


def _prep_common(bundle, name, seed):
    # balance='none': the cross-dataset models use class_weight='balanced'
    # (same imbalance strategy as Phase 3 / the within-dataset script).
    cfg = PreprocessConfig(dataset=f"{name}_common", target="binary",
                           split_mode="predefined", n_features=len(COMMON_FLOW_FEATURES),
                           balance="none", seed=seed)
    res = prepare_dataset(bundle, cfg)          # no writes
    return {
        "pre": res["preprocessor"],
        "X_train": res["X_train"], "y_train": res["y_train_binary"],
        "X_test": res["X_test"], "y_test": res["y_test_binary"],
        "raw_test": bundle["test"],             # aligned 12-col df + labels
        "n_train": int(len(res["X_train"])), "n_test": int(len(res["X_test"])),
    }


def _train_models(data, seed):
    out = {}
    for key in SKLEARN_KEYS:
        mc = MODEL_CFG[key]
        # tiny "validation" via a held-out slice of train for model selection
        n = len(data["X_train"])
        rng = np.random.RandomState(seed)
        idx = rng.permutation(n)
        cut = int(n * 0.85)
        tr, va = idx[:cut], idx[cut:]
        res = M.select_sklearn_model(
            key, mc["fixed"], mc["grid"],
            data["X_train"][tr], data["y_train"][tr],
            data["X_train"][va], data["y_train"][va], seed,
            train_subsample=mc.get("train_subsample"))
        out[key] = res.estimator
    return out


def _evaluate(est, X, y):
    y_pred = est.predict(X)
    y_score = M.sklearn_scores(est, X)
    return E.compute_metrics(y, y_pred, y_score)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cicids-sample-size", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    seed = args.seed
    M.set_global_seed(seed)

    out_dir = REPO_ROOT / "results" / "cross_dataset"
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    print("loading + aligning CIC-IDS2017 ...")
    cic_full = load_cicids2017(REPO_ROOT / "data" / "raw" / "cicids2017",
                               sample_size=args.cicids_sample_size, seed=seed)
    cic_bundle = make_common_schema_bundle(cic_full, "cicids2017")
    del cic_full
    gc.collect()

    print("loading + aligning UNSW-NB15 ...")
    unsw_full = load_unsw_nb15(REPO_ROOT / "data" / "raw" / "unsw_nb15", seed=seed)
    unsw_bundle = make_common_schema_bundle(unsw_full, "unsw_nb15")
    del unsw_full
    gc.collect()

    print("leakage-safe prep on the common schema ...")
    cic = _prep_common(cic_bundle, "cicids2017", seed)
    unsw = _prep_common(unsw_bundle, "unsw_nb15", seed)

    print("training models on each dataset (common schema) ...")
    cic_models = _train_models(cic, seed)
    unsw_models = _train_models(unsw, seed)

    # transform each dataset's raw aligned TEST frame with the OTHER's fitted scaler
    cic_test_via_unsw = unsw["pre"].transform(cic["raw_test"])
    unsw_test_via_cic = cic["pre"].transform(unsw["raw_test"])

    results = {
        "phase": 4, "experiment": "cross-dataset generalisation",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": seed, "task": "binary",
        "common_features": list(COMMON_FLOW_FEATURES),
        "datasets": {
            "cicids2017": {"n_train": cic["n_train"], "n_test": cic["n_test"],
                           "sample_size": args.cicids_sample_size},
            "unsw_nb15": {"n_train": unsw["n_train"], "n_test": unsw["n_test"]},
        },
        "caveats": [
            "12-feature LOSSY alignment of two different flow-exporter pipelines "
            "(CICFlowMeter vs Argus/Bro); see common_schema.md",
            "cross-dataset = target test data standardised with the SOURCE "
            "dataset's training mean/std (standard domain-shift protocol)",
            "no fine-tuning / adaptation - this measures raw transfer",
        ],
        "models": {},
    }

    for key in SKLEARN_KEYS:
        name = PRETTY[key]
        within_cic = _evaluate(cic_models[key], cic["X_test"], cic["y_test"])
        within_unsw = _evaluate(unsw_models[key], unsw["X_test"], unsw["y_test"])
        cross_c2u = _evaluate(cic_models[key], unsw_test_via_cic, unsw["y_test"])
        cross_u2c = _evaluate(unsw_models[key], cic_test_via_unsw, cic["y_test"])
        results["models"][name] = {
            "within_cicids2017": within_cic,
            "within_unsw_nb15": within_unsw,
            "cross_cicids2017_to_unsw_nb15": cross_c2u,
            "cross_unsw_nb15_to_cicids2017": cross_u2c,
            "generalisation_gap_macro_f1": {
                "cicids2017_source": round(within_cic["macro_f1"]
                                           - cross_c2u["macro_f1"], 4),
                "unsw_nb15_source": round(within_unsw["macro_f1"]
                                          - cross_u2c["macro_f1"], 4),
            },
        }
        print(f"  {name:20s} within C={within_cic['macro_f1']:.3f} "
              f"U={within_unsw['macro_f1']:.3f}  |  cross C->U={cross_c2u['macro_f1']:.3f} "
              f"U->C={cross_u2c['macro_f1']:.3f}")

    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2, default=str))
    _write_generalization_md(results, out_dir / "generalization.md")
    _write_common_schema_md(out_dir / "common_schema.md", cic, unsw)
    _plot(results, out_dir / "plots" / "generalization_macro_f1.png")
    print(f"\nwrote {out_dir}/metrics.json, generalization.md, common_schema.md")
    return 0


def _row(m: dict) -> str:
    return (f"{m['accuracy']:.4f} | {m['precision_attack']:.4f} | "
            f"{m['recall_attack']:.4f} | {m['f1_attack']:.4f} | "
            f"{m['macro_f1']:.4f} | "
            f"{(m['roc_auc'] if m['roc_auc'] is not None else float('nan')):.4f}")


def _write_generalization_md(r: dict, path: Path) -> None:
    L = ["# Phase 4 — cross-dataset generalisation (CIC-IDS2017 ↔ UNSW-NB15)",
         "",
         f"_Generated {r['generated_utc']} · seed {r['seed']} · binary task_",
         "",
         "> **Within-dataset** rows: trained and tested on the same dataset's own "
         "splits (12-feature common schema). **Cross-dataset** rows: the model is "
         "trained on the *source* dataset and evaluated on the *other* dataset's "
         "held-out test set, standardised with the source's training statistics. "
         "No adaptation / fine-tuning. Every number is a real prediction score "
         "from `scripts/cross_dataset_experiment.py`.",
         "",
         f"Common feature space: `{', '.join(r['common_features'])}` "
         f"({len(r['common_features'])} features). See `common_schema.md` for the "
         "exact per-dataset construction and its limitations.",
         "",
         f"Data: CIC-IDS2017 train {r['datasets']['cicids2017']['n_train']:,} / "
         f"test {r['datasets']['cicids2017']['n_test']:,} "
         f"(seeded {r['datasets']['cicids2017']['sample_size']:,}-row sub-sample); "
         f"UNSW-NB15 train {r['datasets']['unsw_nb15']['n_train']:,} / "
         f"test {r['datasets']['unsw_nb15']['n_test']:,}.",
         ""]
    for name, b in r["models"].items():
        L += [f"## {name}", "",
              "| evaluation | accuracy | precision (atk) | recall (atk) | "
              "F1 (atk) | macro-F1 | ROC-AUC |",
              "|---|--:|--:|--:|--:|--:|--:|",
              f"| within · CIC-IDS2017 | {_row(b['within_cicids2017'])} |",
              f"| within · UNSW-NB15 | {_row(b['within_unsw_nb15'])} |",
              f"| cross · CIC-IDS2017 → UNSW-NB15 | {_row(b['cross_cicids2017_to_unsw_nb15'])} |",
              f"| cross · UNSW-NB15 → CIC-IDS2017 | {_row(b['cross_unsw_nb15_to_cicids2017'])} |",
              "",
              f"Generalisation gap (within − cross, macro-F1): "
              f"CIC-IDS2017 source = "
              f"**{b['generalisation_gap_macro_f1']['cicids2017_source']:+.4f}**, "
              f"UNSW-NB15 source = "
              f"**{b['generalisation_gap_macro_f1']['unsw_nb15_source']:+.4f}**.",
              ""]
    L += ["## Caveats", ""] + [f"- {c}" for c in r["caveats"]]
    path.write_text("\n".join(L) + "\n")


def _write_common_schema_md(path: Path, cic: dict, unsw: dict) -> None:
    L = ["# Common flow schema — CIC-IDS2017 ↔ UNSW-NB15 alignment",
         "",
         "> This is the **documented transformation** required by the Phase-4 "
         "brief: the two datasets are NOT forced into an identical feature space "
         "silently. NSL-KDD is excluded from this alignment because it has no "
         "comparable bidirectional-flow timing features.",
         "",
         "Both datasets describe bidirectional network flows, but they are built "
         "by different tools (CIC-IDS2017: CICFlowMeter; UNSW-NB15: Argus + Bro/"
         "Zeek) with different units and slightly different definitions. The 12 "
         "features below are the subset that can be constructed the same way on "
         "both sides. Derived rates are **recomputed from the primitives** on "
         "both datasets so the definition is identical; CIC-IDS2017 `Flow "
         "Duration` (microseconds) is divided by 1e6 to match UNSW `dur` "
         "(seconds). Non-finite results (division by zero duration) are set to 0.",
         "",
         "| common feature | from CIC-IDS2017 | from UNSW-NB15 |",
         "|---|---|---|"]
    for feat, (c_src, u_src) in COMMON_FLOW_MAPPING.items():
        L.append(f"| `{feat}` | {c_src} | {u_src} |")
    L += ["",
          "## Why this is lossy (limitations)",
          "",
          "- **Different exporters.** CICFlowMeter and Argus segment flows, count "
          "packets, and handle timeouts differently, so the *same* named quantity "
          "is not measured identically.",
          "- **`fwd_pkt_len_mean` / `bwd_pkt_len_mean`** are taken dataset-native "
          "(CIC `Fwd/Bwd Packet Length Mean` vs UNSW `smean`/`dmean`); these are "
          "close in intent but not guaranteed identical in definition.",
          "- **No protocol / service / TCP-flag features** are shared "
          "(UNSW `proto` has 130+ values; CIC encodes flags as counts). The "
          "common schema is purely volume/rate/duration.",
          "- **Label semantics differ**: CIC-IDS2017 attack mix (DoS/DDoS/"
          "PortScan/brute-force/web/bot/infiltration) is not the UNSW-NB15 mix "
          "(Exploits/Fuzzers/Generic/Recon/…). Only the binary benign-vs-attack "
          "target is aligned.",
          "",
          f"Common-schema sample sizes actually used: CIC-IDS2017 train "
          f"{cic['n_train']:,} / test {cic['n_test']:,}; UNSW-NB15 train "
          f"{unsw['n_train']:,} / test {unsw['n_test']:,}."]
    path.write_text("\n".join(L) + "\n")


def _plot(r: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(r["models"])
    x = np.arange(len(names))
    w = 0.2
    series = {
        "within CIC": [r["models"][n]["within_cicids2017"]["macro_f1"] for n in names],
        "within UNSW": [r["models"][n]["within_unsw_nb15"]["macro_f1"] for n in names],
        "cross CIC→UNSW": [r["models"][n]["cross_cicids2017_to_unsw_nb15"]["macro_f1"] for n in names],
        "cross UNSW→CIC": [r["models"][n]["cross_unsw_nb15_to_cicids2017"]["macro_f1"] for n in names],
    }
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for i, (lbl, vals) in enumerate(series.items()):
        ax.bar(x + (i - 1.5) * w, vals, w, label=lbl)
    ax.set_xticks(x, names, rotation=10, ha="right")
    ax.set_ylabel("macro-F1"); ax.set_ylim(0, 1.02)
    ax.set_title("Within- vs cross-dataset macro-F1 (12-feature common flow schema)")
    ax.legend(fontsize=8, ncol=2); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
