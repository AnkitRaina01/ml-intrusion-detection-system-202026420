"""
Reusable, leakage-safe preprocessing pipeline (Phase 2).

Everything dataset-specific lives in ``src/data_loaders.py``. This module turns a
loaded bundle into model-ready, reproducible train / validation / test matrices
plus the fitted artefacts required to reproduce *exactly* the same transformation
at inference time.

Key guarantees
--------------
* The encoder, scaler, variance filter and feature selector are **fit on the
  training partition only**. Validation and test are transformed, never fitted.
* SMOTE (optional) is applied to the **training partition only**, after
  transformation and feature selection.
* One ``FittedPreprocessor`` object performs the whole raw-DataFrame ->
  model-matrix transform, so training and inference cannot drift apart.
* Given the same inputs, seed and config, re-running produces identical outputs.

Public API
----------
``PreprocessConfig``        - all knobs, serialisable.
``FittedPreprocessor``      - ``.transform(raw_df) -> np.ndarray``; ``.save`` / ``.load``.
``prepare_dataset(...)``    - orchestrator: returns arrays + stats, optionally writes
                              processed arrays, artefacts and a statistics report.
``class_distribution(...)`` - tidy count/percentage table for a label vector.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

ARTIFACT_VERSION = 2  # bump if the artefact layout changes


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class PreprocessConfig:
    dataset: str = "nsl_kdd"
    # target used for stratification, SMOTE and the primary y that models train on
    target: str = "binary"                 # "binary" | "multiclass"
    # split
    split_mode: str = "predefined"         # "predefined" | "pooled"
    train_frac: float = 0.70               # of the grand total
    val_frac: float = 0.15
    test_frac: float = 0.15                # only used when split_mode == "pooled"
    seed: int = 42
    # cleaning
    drop_full_row_duplicates: bool = True  # exact dup rows in the TRAIN pool only
    # feature selection
    n_features: int = 20                   # top-k by mutual information (fit on train)
    variance_threshold: float = 0.0        # drop features with <= this variance (train)
    # class imbalance
    balance: str = "smote"                 # "smote" | "none"
    smote_k_neighbors: int = 5

    def validate(self) -> None:
        if self.target not in ("binary", "multiclass"):
            raise ValueError(f"bad target: {self.target}")
        if self.split_mode not in ("predefined", "pooled"):
            raise ValueError(f"bad split_mode: {self.split_mode}")
        if self.balance not in ("smote", "none"):
            raise ValueError(f"bad balance: {self.balance}")
        if self.split_mode == "pooled":
            s = self.train_frac + self.val_frac + self.test_frac
            if abs(s - 1.0) > 1e-6:
                raise ValueError(f"pooled fractions must sum to 1.0 (got {s})")


# --------------------------------------------------------------------------- #
# Column transformer (encode + scale + drop constant columns)
# --------------------------------------------------------------------------- #
def build_column_pipeline(numeric: Sequence[str],
                          categorical: Sequence[str],
                          variance_threshold: float) -> Pipeline:
    """
    numeric  : median-impute -> standard-scale
    symbolic : most-frequent-impute -> one-hot (unknown categories ignored)
    then a VarianceThreshold drops constant columns (e.g. num_outbound_cmds).
    """
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    ct = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, list(numeric)),
            ("cat", categorical_pipe, list(categorical)),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    return Pipeline([
        ("columns", ct),
        ("variance", VarianceThreshold(threshold=variance_threshold)),
    ])


# --------------------------------------------------------------------------- #
# Fitted preprocessor  (the single source of truth for train == inference)
# --------------------------------------------------------------------------- #
class FittedPreprocessor:
    """
    Wraps the fitted column pipeline + the chosen feature subset. ``transform``
    takes a raw feature DataFrame (the dataset's native columns) and returns the
    final model-ready float32 matrix, columns in ``self.feature_names`` order.
    """

    def __init__(self,
                 column_pipeline: Pipeline,
                 transformed_names: List[str],
                 selected_names: List[str],
                 feature_columns: List[str],
                 categorical_columns: List[str],
                 numeric_columns: List[str],
                 mi_scores: Dict[str, float],
                 config: PreprocessConfig):
        self.column_pipeline = column_pipeline
        self.transformed_names = list(transformed_names)
        self.feature_names = list(selected_names)
        self.feature_columns = list(feature_columns)
        self.categorical_columns = list(categorical_columns)
        self.numeric_columns = list(numeric_columns)
        self.mi_scores = dict(mi_scores)
        self.config = config
        self._sel_idx = np.array(
            [self.transformed_names.index(n) for n in self.feature_names], dtype=int
        )
        self.artifact_version = ARTIFACT_VERSION

    # -- core -------------------------------------------------------------- #
    def transform(self, raw_df: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.feature_columns if c not in raw_df.columns]
        if missing:
            raise ValueError(f"input is missing required columns: {missing}")
        X = self.column_pipeline.transform(raw_df[self.feature_columns])
        X = np.asarray(X, dtype=np.float32)
        return X[:, self._sel_idx]

    # -- persistence ----------------------------------------------------- #
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: str | Path) -> "FittedPreprocessor":
        obj = joblib.load(path)
        if not isinstance(obj, FittedPreprocessor):
            raise TypeError(f"{path} does not contain a FittedPreprocessor")
        return obj


# --------------------------------------------------------------------------- #
# Stats helpers
# --------------------------------------------------------------------------- #
def class_distribution(y: Sequence, name: str = "y") -> pd.DataFrame:
    s = pd.Series(list(y), name=name)
    counts = s.value_counts().sort_index()
    pct = (counts / len(s) * 100).round(4)
    return pd.DataFrame({"class": counts.index, "count": counts.values,
                         "percent": pct.values})


def _dist_dict(y: Sequence) -> Dict[str, Dict[str, float]]:
    # Build straight from the counts Series. (Do NOT iterate a mixed-dtype
    # DataFrame with iterrows(): it upcasts integer class labels to float.)
    s = pd.Series(list(y))
    counts = s.value_counts().sort_index()
    total = int(len(s))
    return {str(k): {"count": int(v),
                     "percent": round(float(v) / total * 100, 4)}
            for k, v in counts.items()}


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
def make_train_val_split(pool: pd.DataFrame,
                         stratify_col: str,
                         cfg: PreprocessConfig,
                         grand_total: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    ``predefined`` mode: ``pool`` is KDDTrain+, the test set is KDDTest+ (kept
    whole).  We split the pool into train/val so that, relative to the grand
    total (pool + test), the partitions are ~ ``train_frac`` / ``val_frac``.
    """
    val_n = round(cfg.val_frac * grand_total)
    val_frac_of_pool = val_n / len(pool)
    if not 0.0 < val_frac_of_pool < 1.0:
        raise ValueError(f"computed val fraction of pool = {val_frac_of_pool}")
    train_df, val_df = train_test_split(
        pool, test_size=val_frac_of_pool, random_state=cfg.seed,
        stratify=pool[stratify_col],
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def make_pooled_split(all_df: pd.DataFrame,
                      stratify_col: str,
                      cfg: PreprocessConfig
                      ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, rest = train_test_split(
        all_df, test_size=(1.0 - cfg.train_frac), random_state=cfg.seed,
        stratify=all_df[stratify_col],
    )
    rel_test = cfg.test_frac / (cfg.val_frac + cfg.test_frac)
    val_df, test_df = train_test_split(
        rest, test_size=rel_test, random_state=cfg.seed,
        stratify=rest[stratify_col],
    )
    return (train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True))


# --------------------------------------------------------------------------- #
# Fit the encode/scale/variance + feature-selection stack on a training frame
# --------------------------------------------------------------------------- #
def fit_preprocessor(train_df: pd.DataFrame,
                     feature_columns: Sequence[str],
                     categorical_columns: Sequence[str],
                     numeric_columns: Sequence[str],
                     target_col: str,
                     cfg: PreprocessConfig) -> FittedPreprocessor:
    """
    Fit the column pipeline (impute -> scale / one-hot -> variance filter) and the
    mutual-information feature selector **on ``train_df`` only** and return a
    ready ``FittedPreprocessor``. Used by ``prepare_dataset`` and by the LSTM
    sequence builder so both share exactly the same transform logic.
    """
    feature_columns = list(feature_columns)
    col_pipe = build_column_pipeline(numeric_columns, categorical_columns,
                                     cfg.variance_threshold)
    Xtr_full = np.asarray(col_pipe.fit_transform(train_df[feature_columns]),
                          dtype=np.float32)
    transformed_names = list(col_pipe.get_feature_names_out())

    discrete_mask = np.array([n.startswith("cat__") for n in transformed_names])
    y = train_df[target_col].to_numpy()
    mi = mutual_info_classif(Xtr_full, y, discrete_features=discrete_mask,
                             random_state=cfg.seed)
    mi_scores = {n: float(s) for n, s in zip(transformed_names, mi)}
    k = min(cfg.n_features, len(transformed_names))
    top = {n for n, _ in sorted(mi_scores.items(), key=lambda kv: kv[1],
                                reverse=True)[:k]}
    selected_names = [n for n in transformed_names if n in top]  # stable order

    return FittedPreprocessor(
        column_pipeline=col_pipe, transformed_names=transformed_names,
        selected_names=selected_names, feature_columns=feature_columns,
        categorical_columns=list(categorical_columns),
        numeric_columns=list(numeric_columns),
        mi_scores=mi_scores, config=cfg,
    )


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def prepare_dataset(bundle: Dict[str, object],
                    cfg: PreprocessConfig,
                    out_dir: Optional[str | Path] = None,
                    artifacts_dir: Optional[str | Path] = None,
                    stats_dir: Optional[str | Path] = None) -> Dict[str, object]:
    """
    Run the full leakage-safe pipeline on a loaded dataset bundle.

    Returns a dict with ``X_train/y_train`` (+ ``*_balanced``), ``X_val/y_val``,
    ``X_test/y_test`` (numpy), the ``FittedPreprocessor``, and a ``stats`` dict.
    If the ``*_dir`` args are given, arrays / artefacts / stats are also written.
    """
    cfg.validate()
    rng_note = f"seed={cfg.seed}"
    feat_cols: List[str] = list(bundle["feature_columns"])      # type: ignore[index]
    cat_cols: List[str] = list(bundle["categorical_columns"])   # type: ignore[index]
    num_cols: List[str] = list(bundle["numeric_columns"])       # type: ignore[index]
    target_col = "binary_label" if cfg.target == "binary" else "attack_category"

    train_pool: pd.DataFrame = bundle["train"].copy()           # type: ignore[index]
    given_test: pd.DataFrame = bundle["test"].copy()            # type: ignore[index]

    # ---- 1. duplicates (train pool only; never touch val/test) ---------- #
    pool_full_dups = int(train_pool.duplicated().sum())
    dropped_dups = 0
    if cfg.drop_full_row_duplicates and pool_full_dups:
        train_pool = train_pool.drop_duplicates().reset_index(drop=True)
        dropped_dups = pool_full_dups

    # ---- 2. splits ---------------------------------------------------- #
    if cfg.split_mode == "predefined":
        grand_total = len(train_pool) + len(given_test)
        train_df, val_df = make_train_val_split(
            train_pool, "attack_category", cfg, grand_total)
        test_df = given_test.reset_index(drop=True)
    else:  # pooled
        all_df = pd.concat([train_pool, given_test], ignore_index=True)
        train_df, val_df, test_df = make_pooled_split(
            all_df, "attack_category", cfg)

    # ---- 3+4. fit encoder/scaler/variance-filter + MI selection on TRAIN #
    fitted = fit_preprocessor(train_df, feat_cols, cat_cols, num_cols,
                              target_col, cfg)
    col_pipe = fitted.column_pipeline
    transformed_names = fitted.transformed_names
    mi_scores = fitted.mi_scores
    selected_names = fitted.feature_names
    dropped_constant = _dropped_by_variance(col_pipe, num_cols, cat_cols)

    Xtr = fitted.transform(train_df[feat_cols])
    Xva = fitted.transform(val_df[feat_cols])
    Xte = fitted.transform(test_df[feat_cols])

    # ---- 5. label vectors (both views kept for every split) --------- #
    y = {
        "train": _labels(train_df), "val": _labels(val_df), "test": _labels(test_df),
    }
    ytr_primary = y["train"][cfg.target]

    # ---- 6. imbalance handling: SMOTE on TRAIN only ---------------- #
    balanced = None
    if cfg.balance == "smote":
        from imblearn.over_sampling import SMOTE
        min_class = int(pd.Series(ytr_primary).value_counts().min())
        kn = max(1, min(cfg.smote_k_neighbors, min_class - 1))
        sm = SMOTE(random_state=cfg.seed, k_neighbors=kn)
        Xtr_bal, ytr_bal = sm.fit_resample(Xtr, ytr_primary)
        balanced = {"X": np.asarray(Xtr_bal, dtype=np.float32), "y": ytr_bal,
                    "k_neighbors": kn}

    # ---- 7. statistics ------------------------------------------------ #
    stats = _build_stats(cfg, bundle, train_pool, train_df, val_df, test_df,
                         feat_cols, transformed_names, selected_names,
                         mi_scores, dropped_constant, dropped_dups,
                         pool_full_dups, balanced, rng_note)

    result: Dict[str, object] = {
        "config": cfg,
        "preprocessor": fitted,
        "feature_names": selected_names,
        "X_train": Xtr, "y_train": ytr_primary,
        "y_train_binary": y["train"]["binary"],
        "y_train_multiclass": y["train"]["multiclass"],
        "X_val": Xva, "y_val": y["val"][cfg.target],
        "y_val_binary": y["val"]["binary"], "y_val_multiclass": y["val"]["multiclass"],
        "X_test": Xte, "y_test": y["test"][cfg.target],
        "y_test_binary": y["test"]["binary"], "y_test_multiclass": y["test"]["multiclass"],
        "stats": stats,
    }
    if balanced is not None:
        result["X_train_balanced"] = balanced["X"]
        result["y_train_balanced"] = balanced["y"]

    # ---- 8. persistence -------------------------------------------- #
    if out_dir:
        _write_arrays(Path(out_dir), result, y, balanced)
    if artifacts_dir:
        _write_artifacts(Path(artifacts_dir), fitted, bundle, stats)
    if stats_dir:
        write_stats_report(stats, Path(stats_dir))

    return result


# --------------------------------------------------------------------------- #
# internals
# --------------------------------------------------------------------------- #
def _labels(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    return {
        "binary": df["binary_label"].to_numpy().astype(np.int64),
        "multiclass": df["attack_category"].to_numpy().astype(object),
    }


def _dropped_by_variance(col_pipe: Pipeline, num_cols, cat_cols) -> List[str]:
    ct: ColumnTransformer = col_pipe.named_steps["columns"]
    pre_names = list(ct.get_feature_names_out())
    vt: VarianceThreshold = col_pipe.named_steps["variance"]
    keep = vt.get_support()
    return [n for n, k in zip(pre_names, keep) if not k]


def _build_stats(cfg, bundle, train_pool, train_df, val_df, test_df,
                 feat_cols, transformed_names, selected_names, mi_scores,
                 dropped_constant, dropped_dups, pool_full_dups, balanced,
                 rng_note) -> dict:
    n_total = len(train_pool) + len(bundle["test"])            # type: ignore[arg-type]
    meta = dict(bundle.get("dataset_meta", {}))                # type: ignore[call-overload]
    n_categories = len(bundle.get("categories", []) or [])     # type: ignore[call-overload]
    inspect = None
    try:
        from src.data_loaders import inspect_bundle
        inspect = inspect_bundle(bundle)
    except Exception:  # pragma: no cover - inspection is best-effort
        pass

    def split_block(df):
        return {
            "n_samples": int(len(df)),
            "percent_of_total": round(len(df) / n_total * 100, 4),
            "binary": _dist_dict(df["binary_label"]),
            "multiclass": _dist_dict(df["attack_category"]),
        }

    top_mi = sorted(mi_scores.items(), key=lambda kv: kv[1], reverse=True)

    stats = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": cfg.dataset,
        "dataset_meta": meta,
        "config": asdict(cfg),
        "reproducibility": rng_note,
        "samples": {
            "raw_train_pool": int(len(bundle["train"])),   # type: ignore[arg-type]
            "raw_test": int(len(bundle["test"])),          # type: ignore[arg-type]
            "grand_total": int(n_total),
        },
        "cleaning": {
            "missing_values_total_train_pool":
                int(bundle["train"][feat_cols].isna().sum().sum()),   # type: ignore[index]
            "missing_values_total_test":
                int(bundle["test"][feat_cols].isna().sum().sum()),    # type: ignore[index]
            "full_row_duplicates_in_train_pool": int(pool_full_dups),
            "full_row_duplicates_dropped": int(dropped_dups),
            "full_row_duplicates_dropped_by_loader":
                int(meta.get("train_full_row_duplicates_dropped", 0)),
            "feature_only_duplicates_train_pool":
                int(bundle["train"][feat_cols].duplicated().sum()),   # type: ignore[index]
            "note_feature_only_duplicates":
                f"rows with an identical {len(feat_cols)}-feature vector but a "
                "different label; kept in the training pool (dropping them "
                "changes the class balance)",
        },
        "features": {
            "before_preprocessing": int(len(feat_cols)),
            "after_encoding_and_variance_filter": int(len(transformed_names)),
            "after_selection": int(len(selected_names)),
            "constant_features_dropped": dropped_constant,
            "selection_method":
                f"top-{cfg.n_features} by mutual_info_classif (fit on train only, "
                f"{rng_note})",
            "selected_features": list(selected_names),
            "mutual_information_top20": [
                {"feature": n, "mi": round(s, 6)} for n, s in top_mi[:20]
            ],
        },
        "splits": {
            "mode": cfg.split_mode,
            "strategy": (
                meta.get(
                    "split_strategy_predefined",
                    "KDDTrain+ -> stratified train/val; KDDTest+ kept as held-out "
                    "test (contains novel attack types by NSL-KDD design)")
                if cfg.split_mode == "predefined"
                else meta.get(
                    "split_strategy_pooled",
                    "pooled train + test frames, stratified 3-way split")
            ),
            "stratified_on": f"attack_category ({n_categories}-class)",
            "train": split_block(train_df),
            "validation": split_block(val_df),
            "test": split_block(test_df),
        },
        "class_balance": {
            "primary_target": cfg.target,
            "method": cfg.balance,
            "train_before_balancing": _dist_dict(train_df[
                "binary_label" if cfg.target == "binary" else "attack_category"]),
            "train_after_balancing": (
                _dist_dict(balanced["y"]) if balanced is not None
                else "n/a (balance=none)"),
            "smote_k_neighbors": (balanced["k_neighbors"]
                                  if balanced is not None else None),
            "note": meta.get(
                "balance_note",
                "NSL-KDD binary target is only mildly imbalanced "
                "(~53/47); class_weight is an equally valid alternative."),
        },
    }
    if inspect is not None:
        stats["raw_inspection"] = inspect
    return stats


def _write_arrays(out_dir: Path, result, y, balanced) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "X_train.npy", result["X_train"])
    np.save(out_dir / "X_val.npy", result["X_val"])
    np.save(out_dir / "X_test.npy", result["X_test"])
    for split in ("train", "val", "test"):
        np.save(out_dir / f"y_{split}_binary.npy", y[split]["binary"])
        np.save(out_dir / f"y_{split}_multiclass.npy", y[split]["multiclass"])
    if balanced is not None:
        np.save(out_dir / "X_train_balanced.npy", balanced["X"])
        np.save(out_dir / "y_train_balanced.npy", np.asarray(balanced["y"]))
    (out_dir / "feature_names.json").write_text(
        json.dumps(list(result["feature_names"]), indent=2))
    logger.info("processed arrays written to %s", out_dir)


def _write_artifacts(art_dir: Path, fitted: FittedPreprocessor,
                     bundle, stats) -> None:
    art_dir.mkdir(parents=True, exist_ok=True)
    fitted.save(art_dir / "preprocessor.joblib")
    (art_dir / "feature_names.json").write_text(
        json.dumps(fitted.feature_names, indent=2))
    (art_dir / "transformed_feature_names.json").write_text(
        json.dumps(fitted.transformed_names, indent=2))
    label_maps = {
        "binary": {"0": "normal", "1": "attack"},
        "multiclass": list(bundle["categories"]),          # type: ignore[index]
        "attack_category_map": _attack_map(bundle),
    }
    (art_dir / "label_maps.json").write_text(json.dumps(label_maps, indent=2))
    meta = {
        "artifact_version": ARTIFACT_VERSION,
        "dataset": fitted.config.dataset,
        "config": asdict(fitted.config),
        "feature_columns_expected": fitted.feature_columns,
        "n_selected_features": len(fitted.feature_names),
        "generated_utc": stats["generated_utc"],
        "sklearn_pipeline_steps": [s for s, _ in fitted.column_pipeline.steps],
    }
    (art_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    logger.info("preprocessing artefacts written to %s", art_dir)


def _attack_map(bundle: Optional[Dict[str, object]] = None) -> Dict[str, str]:
    if bundle is not None and bundle.get("attack_category_map"):
        return dict(bundle["attack_category_map"])          # type: ignore[arg-type]
    from src.data_loaders import NSL_KDD_ATTACK_CATEGORY
    return dict(NSL_KDD_ATTACK_CATEGORY)


def write_stats_report(stats: dict, stats_dir: str | Path) -> Path:
    stats_dir = Path(stats_dir)
    stats_dir.mkdir(parents=True, exist_ok=True)
    (stats_dir / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    md = _stats_to_markdown(stats)
    (stats_dir / "dataset_stats.md").write_text(md)
    logger.info("dataset statistics written to %s", stats_dir)
    return stats_dir / "dataset_stats.md"


def _fmt_dist(d) -> str:
    if not isinstance(d, dict):
        return str(d)
    rows = ["| class | count | percent |", "|---|---:|---:|"]
    for k, v in d.items():
        rows.append(f"| {k} | {v['count']:,} | {v['percent']:.2f}% |")
    return "\n".join(rows)


def _stats_to_markdown(s: dict) -> str:
    sp = s["splits"]
    f = s["features"]
    meta = s.get("dataset_meta", {}) or {}
    disp = meta.get("display_name", s["dataset"])
    is_nsl = s["dataset"].startswith("nsl_kdd")
    pool_name = "raw KDDTrain+ pool" if is_nsl else "raw train pool"
    test_name = "raw KDDTest+" if is_nsl else "raw held-out test"
    src_line = (
        "> Figures below are computed by `scripts/prepare_nsl_kdd.py` from the "
        "real NSL-KDD files. Nothing here is hand-entered."
        if is_nsl else
        f"> Figures below are computed by `scripts/prepare_datasets.py "
        f"--dataset {s['dataset']}` from the real dataset files. "
        f"Nothing here is hand-entered.")
    L = [
        f"# Dataset statistics — {disp}",
        "",
        f"_Generated {s['generated_utc']} — {s['reproducibility']}_",
        "",
        src_line,
        "",
    ]
    if meta.get("subsample"):
        L += [f"> **Sub-sampling:** {meta['subsample']}", ""]
    L += [
        "## 1. Samples",
        "",
        "| set | rows |",
        "|---|---:|",
        f"| {pool_name} | {s['samples']['raw_train_pool']:,} |",
        f"| {test_name} | {s['samples']['raw_test']:,} |",
        f"| grand total | {s['samples']['grand_total']:,} |",
        "",
        "## 2. Cleaning",
        "",
        f"- missing values (train pool / test): "
        f"{s['cleaning']['missing_values_total_train_pool']} / "
        f"{s['cleaning']['missing_values_total_test']}",
        f"- exact full-row duplicates in train pool: "
        f"{s['cleaning']['full_row_duplicates_in_train_pool']} "
        f"(dropped: {s['cleaning']['full_row_duplicates_dropped']})",
        f"- feature-only duplicates in train pool: "
        f"{s['cleaning']['feature_only_duplicates_train_pool']} "
        f"— {s['cleaning']['note_feature_only_duplicates']}",
        *([f"- exact full-row duplicates removed by the loader **before** this "
           f"pipeline: {s['cleaning']['full_row_duplicates_dropped_by_loader']:,}"]
          if s['cleaning'].get('full_row_duplicates_dropped_by_loader') else []),
        "",
        "## 3. Features",
        "",
        f"- before preprocessing: **{f['before_preprocessing']}**",
        f"- after one-hot encoding + variance filter: "
        f"**{f['after_encoding_and_variance_filter']}**",
        f"- after selection: **{f['after_selection']}**",
        f"- constant feature(s) dropped: {f['constant_features_dropped']}",
        f"- selection method: {f['selection_method']}",
        "",
        "### Selected features",
        "",
        "```",
        "\n".join(f['selected_features']),
        "```",
        "",
        "### Top-20 features by mutual information (train only)",
        "",
        "| feature | MI |",
        "|---|---:|",
        *[f"| {r['feature']} | {r['mi']:.6f} |"
          for r in f["mutual_information_top20"]],
        "",
        "## 4. Train / validation / test partitions",
        "",
        f"Split mode: **{sp['mode']}** — {sp['strategy']}  ",
        f"Stratified on: {sp['stratified_on']}",
        "",
        "| partition | rows | % of total |",
        "|---|---:|---:|",
        f"| train | {sp['train']['n_samples']:,} | "
        f"{sp['train']['percent_of_total']:.2f}% |",
        f"| validation | {sp['validation']['n_samples']:,} | "
        f"{sp['validation']['percent_of_total']:.2f}% |",
        f"| test | {sp['test']['n_samples']:,} | "
        f"{sp['test']['percent_of_total']:.2f}% |",
        "",
        "## 5. Class distribution — binary (0 = normal/benign, 1 = attack)",
        "",
        "### train", "", _fmt_dist(sp["train"]["binary"]), "",
        "### validation", "", _fmt_dist(sp["validation"]["binary"]), "",
        "### test", "", _fmt_dist(sp["test"]["binary"]), "",
        f"## 6. Class distribution — {sp['stratified_on']}",
        "",
        "### train", "", _fmt_dist(sp["train"]["multiclass"]), "",
        "### validation", "", _fmt_dist(sp["validation"]["multiclass"]), "",
        "### test", "", _fmt_dist(sp["test"]["multiclass"]), "",
        "## 7. Class balancing",
        "",
        f"- primary target: **{s['class_balance']['primary_target']}**",
        f"- method: **{s['class_balance']['method']}** "
        f"(SMOTE k_neighbors = {s['class_balance']['smote_k_neighbors']})",
        f"- note: {s['class_balance']['note']}",
        "",
        "### train — before balancing", "",
        _fmt_dist(s["class_balance"]["train_before_balancing"]), "",
        "### train — after balancing", "",
        _fmt_dist(s["class_balance"]["train_after_balancing"]), "",
    ]
    return "\n".join(L) + "\n"
