"""
Dataset loaders / adapters.

Phase 2 adds a first-class loader for **NSL-KDD**. Each adapter returns raw
`pandas.DataFrame`s with named columns plus standardised label columns, so the
downstream preprocessing pipeline (`src/preprocessing.py`) does not need to know
anything dataset-specific.

Design goals:
- No assumptions baked in silently: the NSL-KDD schema comes from the dataset's
  own ``Field Names.csv``; the attack->category taxonomy is the well-established
  NSL-KDD 5-class mapping and every observed label is asserted to be covered.
- Loaders are pure functions (no global state, no writes).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# NSL-KDD
# --------------------------------------------------------------------------- #

# The 41 feature names, in order, as published in the NSL-KDD "Field Names.csv".
NSL_KDD_FEATURES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]

# Raw file column layout = 41 features + textual attack label + difficulty score.
NSL_KDD_RAW_COLUMNS = NSL_KDD_FEATURES + ["label", "difficulty"]

# Columns that are symbolic / categorical (per "Field Names.csv").
NSL_KDD_CATEGORICAL = ["protocol_type", "service", "flag"]
NSL_KDD_NUMERIC = [c for c in NSL_KDD_FEATURES if c not in NSL_KDD_CATEGORICAL]

# Canonical NSL-KDD attack -> 5-class category map.
# Covers every attack label that appears in KDDTrain+ (22 types) AND KDDTest+
# (37 types, incl. 17 "novel" attacks not present in training). Source: the
# standard KDD Cup 99 / NSL-KDD taxonomy used throughout the IDS literature.
NSL_KDD_ATTACK_CATEGORY: Dict[str, str] = {
    "normal": "normal",
    # ---- Denial of Service ----
    "back": "dos", "land": "dos", "neptune": "dos", "pod": "dos",
    "smurf": "dos", "teardrop": "dos", "mailbomb": "dos", "apache2": "dos",
    "processtable": "dos", "udpstorm": "dos", "worm": "dos",
    # ---- Probe ----
    "satan": "probe", "ipsweep": "probe", "nmap": "probe", "portsweep": "probe",
    "mscan": "probe", "saint": "probe",
    # ---- Remote to Local ----
    "guess_passwd": "r2l", "ftp_write": "r2l", "imap": "r2l", "phf": "r2l",
    "multihop": "r2l", "warezmaster": "r2l", "warezclient": "r2l", "spy": "r2l",
    "xlock": "r2l", "xsnoop": "r2l", "snmpguess": "r2l", "snmpgetattack": "r2l",
    "httptunnel": "r2l", "sendmail": "r2l", "named": "r2l",
    # ---- User to Root ----
    "buffer_overflow": "u2r", "loadmodule": "u2r", "rootkit": "u2r",
    "perl": "u2r", "sqlattack": "u2r", "xterm": "u2r", "ps": "u2r",
}

NSL_KDD_CATEGORIES = ["normal", "dos", "probe", "r2l", "u2r"]


def _read_nsl_kdd_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=NSL_KDD_RAW_COLUMNS)
    if df.shape[1] != len(NSL_KDD_RAW_COLUMNS):
        raise ValueError(
            f"{path.name}: expected {len(NSL_KDD_RAW_COLUMNS)} columns, "
            f"got {df.shape[1]}"
        )
    df["label"] = df["label"].str.strip().str.lower()
    return df


def _add_standard_labels(df: pd.DataFrame) -> pd.DataFrame:
    unknown = sorted(set(df["label"]) - set(NSL_KDD_ATTACK_CATEGORY))
    if unknown:
        raise ValueError(
            "NSL-KDD label(s) not in the attack->category map: "
            f"{unknown}. Update NSL_KDD_ATTACK_CATEGORY."
        )
    out = df.copy()
    out["attack_category"] = out["label"].map(NSL_KDD_ATTACK_CATEGORY)
    out["binary_label"] = (out["label"] != "normal").astype(int)  # 0=normal 1=attack
    return out


def load_nsl_kdd(raw_dir: str | Path,
                 train_file: str = "KDDTrain+.txt",
                 test_file: str = "KDDTest+.txt") -> Dict[str, object]:
    """
    Load NSL-KDD as delivered.

    Returns a dict with:
      ``train`` / ``test``     : DataFrames with the 41 named feature columns,
                                 ``difficulty`` (raw), textual ``label``,
                                 ``attack_category`` (5-class), ``binary_label``
                                 (0=normal, 1=attack).
      ``feature_columns``      : list of the 41 feature names.
      ``categorical_columns``  : ``["protocol_type", "service", "flag"]``.
      ``numeric_columns``      : the other 38 feature names.

    KDDTrain+ is the training *pool* and KDDTest+ is the held-out test set that,
    by NSL-KDD design, contains attack types absent from training.
    """
    raw_dir = Path(raw_dir)
    train_path = raw_dir / train_file
    test_path = raw_dir / test_file
    for p in (train_path, test_path):
        if not p.exists():
            raise FileNotFoundError(
                f"NSL-KDD file not found: {p}\n"
                "Download KDDTrain+.txt and KDDTest+.txt into this directory "
                "(see DEVELOPMENT_NOTES.md section 8)."
            )

    train = _add_standard_labels(_read_nsl_kdd_file(train_path))
    test = _add_standard_labels(_read_nsl_kdd_file(test_path))

    return {
        "train": train,
        "test": test,
        "feature_columns": list(NSL_KDD_FEATURES),
        "categorical_columns": list(NSL_KDD_CATEGORICAL),
        "numeric_columns": list(NSL_KDD_NUMERIC),
        "categories": list(NSL_KDD_CATEGORIES),
    }


def inspect_nsl_kdd(bundle: Dict[str, object]) -> dict:
    """
    Build a plain-dict inspection report of a loaded NSL-KDD bundle: shapes,
    dtypes, missing values, duplicates, categorical cardinality, and label
    counts. Pure function - callers decide whether to print / persist it.
    """
    report: dict = {}
    for split in ("train", "test"):
        df: pd.DataFrame = bundle[split]  # type: ignore[assignment]
        feats = df[bundle["feature_columns"]]
        report[split] = {
            "n_rows": int(len(df)),
            "n_feature_columns": int(feats.shape[1]),
            "missing_values_total": int(feats.isna().sum().sum()),
            "full_row_duplicates": int(df.duplicated().sum()),
            "feature_only_duplicates": int(feats.duplicated().sum()),
            "dtype_counts": {str(k): int(v)
                             for k, v in feats.dtypes.value_counts().items()},
            "categorical_cardinality": {
                c: int(df[c].nunique()) for c in bundle["categorical_columns"]
            },
            "constant_numeric_features": [
                c for c in bundle["numeric_columns"] if df[c].nunique() <= 1
            ],
            "binary_label_counts": {
                str(k): int(v)
                for k, v in df["binary_label"].value_counts().sort_index().items()
            },
            "attack_category_counts": {
                str(k): int(v)
                for k, v in df["attack_category"].value_counts().items()
            },
            "raw_label_counts": {
                str(k): int(v) for k, v in df["label"].value_counts().items()
            },
        }
    # categories present in test but never in training
    tr_labels = set(bundle["train"]["label"])          # type: ignore[index]
    te_labels = set(bundle["test"]["label"])            # type: ignore[index]
    report["novel_test_attack_labels"] = sorted(te_labels - tr_labels)
    return report


# ``inspect_nsl_kdd`` only uses the generic bundle keys (feature_columns,
# categorical_columns, numeric_columns, label, attack_category, binary_label),
# so it works unchanged for every Phase-4 dataset. Exposed under a neutral name.
inspect_bundle = inspect_nsl_kdd


# =========================================================================== #
# Phase 4 — CIC-IDS2017
# =========================================================================== #
#
# Source used: the 8 "MachineLearningCVE" daily flow CSVs
# (``*-WorkingHours*.pcap_ISCX.csv``), i.e. CICFlowMeter features + ``Label``.
# There is **no official train/test split**, so this loader creates a
# deterministic stratified one (seeded). Because the full set is ~2.83 M rows
# and this project runs on a 3.8 GiB box, the loader takes a **seeded,
# label-stratified sub-sample** (rare attack classes kept in full). Every one of
# these choices is a documented, reproducible transformation — see
# DEVELOPMENT_NOTES.md section 11.
#
# 78 numeric flow features, 0 categorical (``Destination Port`` is treated as
# numeric, as is standard for this dataset).

CICIDS2017_FILES = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
]

# Non-feature / identifier columns that appear in some mirrors of the CSVs and
# must never be fed to a model.
CICIDS2017_DROP_COLUMNS = [
    "Flow ID", "Source IP", "Src IP", "Source Port", "Src Port",
    "Destination IP", "Dst IP", "Timestamp", "SimillarHTTP", "Fwd Header Length.1",
    "Unnamed: 0",
]

# Raw ``Label`` value  ->  coarse attack category (documented grouping).
CICIDS2017_ATTACK_CATEGORY: Dict[str, str] = {
    "benign": "benign",
    "dos hulk": "dos", "dos goldeneye": "dos", "dos slowloris": "dos",
    "dos slowhttptest": "dos", "heartbleed": "dos",
    "ddos": "ddos",
    "portscan": "probe",
    "ftp-patator": "brute_force", "ssh-patator": "brute_force",
    # after _normalise_cicids_label() every en-dash variant becomes " - "
    "web attack - brute force": "web_attack",
    "web attack - xss": "web_attack",
    "web attack - sql injection": "web_attack",
    "bot": "botnet",
    "infiltration": "infiltration",
}

CICIDS2017_CATEGORIES = ["benign", "dos", "ddos", "probe", "brute_force",
                         "web_attack", "botnet", "infiltration"]


def _normalise_cicids_label(raw: str) -> str:
    """Lower-case, collapse whitespace, and normalise the several corrupted
    representations of the CP-1252 en-dash that appear in the public mirrors of
    the CIC-IDS2017 CSVs (raw 0x96, U+2013/U+2014, and the U+FFFD replacement
    char) so ``Web Attack - XSS`` etc. map cleanly."""
    s = str(raw).replace("\xef\xbf\xbd", "-")   # UTF-8 bytes of U+FFFD read as latin-1
    for bad in ("\x96", "–", "—", "�", "\xa0"):
        s = s.replace(bad, "-")
    s = " ".join(s.split()).strip().lower()
    return s


def _cicids_label_series(path: Path) -> pd.Series:
    s = pd.read_csv(path, usecols=lambda c: str(c).strip().lower() == "label",
                    encoding="utf-8", encoding_errors="replace").iloc[:, 0]
    return s.map(_normalise_cicids_label)


def _plan_per_class_keep(global_counts: pd.Series, target: int,
                         keep_all_below: int) -> Dict[str, int]:
    """How many rows to keep per raw-label class so the total is ~``target``,
    keeping every row of classes smaller than ``keep_all_below``."""
    if target >= int(global_counts.sum()):
        return {k: int(v) for k, v in global_counts.items()}
    small = global_counts[global_counts < keep_all_below]
    large = global_counts[global_counts >= keep_all_below]
    budget = max(0, target - int(small.sum()))
    large_total = int(large.sum()) or 1
    keep = {k: int(v) for k, v in small.items()}
    for cls, n in large.items():
        keep[cls] = min(int(n), max(1, round(budget * int(n) / large_total)))
    return keep


def load_cicids2017(raw_dir: str | Path,
                    sample_size: int = 200_000,
                    keep_all_below: int = 12_000,
                    test_size: float = 0.15,
                    seed: int = 42,
                    files: Optional[Sequence[str]] = None) -> Dict[str, object]:
    """
    Load CIC-IDS2017 (MachineLearningCVE CSVs) as a standard bundle.

    Memory-safe (two passes over the 8 daily CSVs, one file resident at a time —
    the full set is ~2.83 M rows and will not fit in this project's 3.8 GiB box):

      pass 1  count the raw ``Label`` values across all files;
      plan    a per-class keep count so the sub-sample is ~``sample_size`` rows,
              with every row of classes < ``keep_all_below`` retained;
      pass 2  per file: strip headers, drop identifier columns, coerce the 78
              flow features to float32 (``inf`` -> NaN), then take that file's
              seeded share of each class's keep count.

    Then derive ``label`` / ``attack_category`` (8-class coarse map) /
    ``binary_label`` (0 = benign, 1 = attack) and make a seeded stratified
    train-pool / held-out-test split. Everything is deterministic given ``seed``.

    Returns the same dict shape as :func:`load_nsl_kdd` (plus ``attack_category_map``
    and ``dataset_meta``).
    """
    raw_dir = Path(raw_dir)
    files = list(files) if files else list(CICIDS2017_FILES)
    paths = [raw_dir / f for f in files]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "CIC-IDS2017 CSV(s) not found: "
            + ", ".join(p.name for p in missing)
            + f"\nExpected under {raw_dir}. Run scripts/download_datasets.py.")

    # ---- pass 1: global label counts -------------------------------------- #
    per_file_labels = {p: _cicids_label_series(p) for p in paths}
    global_counts = pd.concat(per_file_labels.values()).value_counts()
    full_n = int(global_counts.sum())

    unknown = sorted(set(global_counts.index) - set(CICIDS2017_ATTACK_CATEGORY))
    if unknown:
        raise ValueError(
            f"CIC-IDS2017 label(s) not in the category map: {unknown}. "
            "Update CICIDS2017_ATTACK_CATEGORY.")

    keep_plan = _plan_per_class_keep(global_counts, sample_size, keep_all_below)

    # ---- pass 2: read + clean + per-file stratified take ----------------- #
    import gc
    feature_columns: Optional[list] = None
    taken: Dict[str, int] = {k: 0 for k in keep_plan}
    parts: list = []
    for i, p in enumerate(paths):
        df = pd.read_csv(p, low_memory=False, encoding="utf-8", encoding_errors="replace")
        df.columns = [str(c).strip() for c in df.columns]
        df = df.drop(columns=[c for c in CICIDS2017_DROP_COLUMNS if c in df.columns])
        label_col = next((c for c in df.columns if c.strip().lower() == "label"), None)
        df = df.rename(columns={label_col: "label"})
        df["label"] = df["label"].map(_normalise_cicids_label)

        if feature_columns is None:
            feature_columns = [c for c in df.columns if c != "label"]
        for c in feature_columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)
        df[feature_columns] = df[feature_columns].replace([np.inf, -np.inf], np.nan)

        last_file = (i == len(paths) - 1)
        for cls, grp in df.groupby("label", sort=True):
            want_total = keep_plan[cls]
            if last_file:
                k = min(len(grp), max(0, want_total - taken[cls]))
            else:
                # this file's proportional share of the class keep budget
                k = round(want_total * len(grp) / int(global_counts[cls]))
                k = min(len(grp), k, max(0, want_total - taken[cls]))
            if k > 0:
                parts.append(grp.sample(n=int(k), random_state=seed).copy())
                taken[cls] += int(k)
        del df
        gc.collect()

    sampled = pd.concat(parts, ignore_index=True)
    sampled = sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    del parts

    sampled["attack_category"] = sampled["label"].map(CICIDS2017_ATTACK_CATEGORY)
    sampled["binary_label"] = (sampled["label"] != "benign").astype(int)

    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(
        sampled, test_size=test_size, random_state=seed,
        stratify=sampled["label"])
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    meta = {
        "display_name": "CIC-IDS2017",
        "source": "MachineLearningCVE daily flow CSVs (CICFlowMeter features)",
        "raw_rows_available": full_n,
        "rows_after_subsample": int(len(sampled)),
        "subsample": (f"seeded (seed={seed}) label-stratified sub-sample to "
                      f"~{sample_size:,} of {full_n:,} rows; classes with "
                      f"< {keep_all_below:,} rows kept in full "
                      f"(memory-driven, reproducible)"),
        "split_strategy_predefined": (
            f"no official split exists; loader makes a seeded "
            f"{int(round((1 - test_size) * 100))}/{int(round(test_size * 100))} "
            f"stratified split of the sub-sample into a train pool + held-out "
            f"test set, then prepare_dataset carves train/validation from the pool"),
        "balance_note": (
            "CIC-IDS2017 is benign-heavy; SMOTE is applied to the training "
            "partition only. class_weight='balanced' is the alternative used by "
            "the model scripts."),
    }

    return {
        "train": train_df,
        "test": test_df,
        "feature_columns": list(feature_columns),
        "categorical_columns": [],
        "numeric_columns": list(feature_columns),
        "categories": list(CICIDS2017_CATEGORIES),
        "attack_category_map": dict(CICIDS2017_ATTACK_CATEGORY),
        "dataset_meta": meta,
    }


# =========================================================================== #
# Phase 4 — UNSW-NB15
# =========================================================================== #
#
# Source used: the official partitioned CSVs ``UNSW_NB15_training-set.csv`` /
# ``UNSW_NB15_testing-set.csv`` (Moustafa & Slay). 42 features + ``attack_cat`` +
# ``label``; an ``id`` column is dropped. 3 categorical features
# (``proto``, ``service``, ``state``).

UNSW_NB15_DROP_COLUMNS = ["id"]           # ``id`` = row identifier, never a feature
UNSW_NB15_CATEGORICAL = ["proto", "service", "state"]

# ``attack_cat`` is already a clean 9-attack taxonomy; we only lower-case it and
# rename the benign class to ``normal`` for cross-dataset consistency.
UNSW_NB15_CATEGORIES = ["normal", "fuzzers", "analysis", "backdoor", "dos",
                        "exploits", "generic", "reconnaissance", "shellcode",
                        "worms"]


def _unsw_attack_category(raw: str) -> str:
    s = str(raw).strip().lower()
    if s in ("", "nan", "normal", "-"):
        return "normal"
    if s == "backdoors":
        return "backdoor"
    return s


def _read_unsw_csv(path: Path):
    """Return (df_with_standard_labels, feature_columns, numeric_columns)."""
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    for c in ("attack_cat", "label"):
        if c not in df.columns:
            raise ValueError(f"{path.name}: expected '{c}' column")
    df = df.drop(columns=[c for c in UNSW_NB15_DROP_COLUMNS if c in df.columns])

    feature_columns = [c for c in df.columns if c not in ("attack_cat", "label")]
    numeric_columns = [c for c in feature_columns if c not in UNSW_NB15_CATEGORICAL]

    for c in numeric_columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)
    df[numeric_columns] = df[numeric_columns].replace([np.inf, -np.inf], np.nan)
    for c in UNSW_NB15_CATEGORICAL:
        df[c] = df[c].astype(str).str.strip().str.lower()

    raw_cat = df["attack_cat"].fillna("Normal").astype(str).str.strip()
    df["label"] = raw_cat.str.lower().replace({"": "normal"})
    df["attack_category"] = raw_cat.map(_unsw_attack_category)
    df["binary_label"] = pd.to_numeric(df["label"], errors="coerce")  # placeholder
    df["binary_label"] = (df["attack_category"] != "normal").astype(int)
    df = df.drop(columns=["attack_cat"])
    return df, feature_columns, numeric_columns


def load_unsw_nb15(raw_dir: str | Path,
                   train_file: str = "UNSW_NB15_training-set.csv",
                   test_file: str = "UNSW_NB15_testing-set.csv",
                   drop_train_duplicates: bool = True,
                   seed: int = 42) -> Dict[str, object]:
    """
    Load UNSW-NB15 (official partitioned CSVs) as a standard bundle.

    * ``id`` is dropped (row identifier). The other 42 features are kept.
    * ``proto`` / ``service`` / ``state`` are the categorical features; the rest
      are coerced to float32 with ``inf`` -> NaN.
    * ``attack_cat`` -> ``attack_category`` (lower-cased; benign -> ``normal``);
      ``binary_label`` = 0 for ``normal`` else 1 (matches the dataset's own
      ``label`` column, which is verified equal).
    * exact full-row duplicates are dropped **from the training pool only**
      (~38% of the training CSV are repeated flows); validation/test untouched.

    The official train/test split is respected (``split_mode="predefined"``).
    """
    raw_dir = Path(raw_dir)
    train_path = raw_dir / train_file
    test_path = raw_dir / test_file
    for p in (train_path, test_path):
        if not p.exists():
            raise FileNotFoundError(
                f"UNSW-NB15 file not found: {p}\nRun scripts/download_datasets.py.")

    # verify our derived binary label matches the dataset's own 'label' column
    for p in (train_path, test_path):
        chk = pd.read_csv(p, usecols=lambda c: c.strip() in ("attack_cat", "label"))
        chk.columns = [c.strip() for c in chk.columns]
        derived = (chk["attack_cat"].fillna("Normal").astype(str).str.strip().str.lower()
                   != "normal").astype(int)
        if not derived.equals(chk["label"].astype(int)):
            raise ValueError(f"{p.name}: attack_cat vs label disagree on "
                             f"{int((derived != chk['label']).sum())} rows")

    train, feat_cols, num_cols = _read_unsw_csv(train_path)
    test, _, _ = _read_unsw_csv(test_path)

    raw_train_n = len(train)
    dropped = 0
    if drop_train_duplicates:
        train = train.drop_duplicates().reset_index(drop=True)
        dropped = raw_train_n - len(train)

    meta = {
        "display_name": "UNSW-NB15",
        "source": "official UNSW_NB15_training-set.csv / _testing-set.csv",
        "raw_train_rows": int(raw_train_n),
        "train_full_row_duplicates_dropped": int(dropped),
        "dropped_columns": list(UNSW_NB15_DROP_COLUMNS),
        "split_strategy_predefined": (
            "official partitioned split respected: training-set.csv = train pool "
            "(exact duplicate rows removed), testing-set.csv = held-out test; "
            "prepare_dataset carves a stratified validation slice from the pool"),
        "balance_note": (
            "UNSW-NB15 is attack-majority (~68% attack in the training CSV); "
            "SMOTE is applied to the training partition only. The 'worms' class "
            "is tiny (130 train rows) - SMOTE k_neighbors is reduced accordingly."),
    }

    return {
        "train": train,
        "test": test,
        "feature_columns": list(feat_cols),
        "categorical_columns": list(UNSW_NB15_CATEGORICAL),
        "numeric_columns": list(num_cols),
        "categories": list(UNSW_NB15_CATEGORIES),
        "attack_category_map": {c: c for c in UNSW_NB15_CATEGORIES},
        "dataset_meta": meta,
    }


# =========================================================================== #
# Phase 4 — common flow schema for CROSS-DATASET generalisation
# =========================================================================== #
#
# CIC-IDS2017 and UNSW-NB15 are both *bidirectional-flow* datasets, but they are
# produced by different tools (CICFlowMeter vs Argus/Bro), with different units
# and slightly different definitions. NSL-KDD is a *connection-record* dataset
# with no comparable flow-timing features and is therefore NOT part of this
# alignment (documented, not forced).
#
# The 12 features below are the ones that can be defined the same way on both
# sides. Derived rates are recomputed from the primitives on BOTH datasets so the
# definition is identical; CIC-IDS2017 ``Flow Duration`` (microseconds) is
# converted to seconds to match UNSW ``dur``. This is a LOSSY approximation - see
# results/cross_dataset/common_schema.md.

COMMON_FLOW_FEATURES = [
    "dur_s", "fwd_pkts", "bwd_pkts", "fwd_bytes", "bwd_bytes",
    "fwd_pkt_len_mean", "bwd_pkt_len_mean",
    "flow_bytes_per_s", "flow_pkts_per_s", "fwd_pkts_per_s", "bwd_pkts_per_s",
    "down_up_ratio",
]

# provenance of each common feature: (cicids_source, unsw_source)
COMMON_FLOW_MAPPING = {
    "dur_s":            ("Flow Duration / 1e6  (µs -> s)", "dur  (already seconds)"),
    "fwd_pkts":         ("Total Fwd Packets", "spkts"),
    "bwd_pkts":         ("Total Backward Packets", "dpkts"),
    "fwd_bytes":        ("Total Length of Fwd Packets", "sbytes"),
    "bwd_bytes":        ("Total Length of Bwd Packets", "dbytes"),
    "fwd_pkt_len_mean": ("Fwd Packet Length Mean", "smean"),
    "bwd_pkt_len_mean": ("Bwd Packet Length Mean", "dmean"),
    "flow_bytes_per_s": ("(fwd_bytes+bwd_bytes)/dur_s  [recomputed]",
                         "(sbytes+dbytes)/dur  [recomputed]"),
    "flow_pkts_per_s":  ("(fwd_pkts+bwd_pkts)/dur_s  [recomputed]",
                         "(spkts+dpkts)/dur  [recomputed]"),
    "fwd_pkts_per_s":   ("fwd_pkts/dur_s  [recomputed]", "spkts/dur  [recomputed]"),
    "bwd_pkts_per_s":   ("bwd_pkts/dur_s  [recomputed]", "dpkts/dur  [recomputed]"),
    "down_up_ratio":    ("bwd_pkts/max(fwd_pkts,1)  [recomputed]",
                         "dpkts/max(spkts,1)  [recomputed]"),
}


def _finite(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _add_common_derived(out: pd.DataFrame) -> pd.DataFrame:
    dur = out["dur_s"].to_numpy(dtype="float64")
    safe = np.where(dur > 0, dur, np.nan)
    tot_b = out["fwd_bytes"].to_numpy("float64") + out["bwd_bytes"].to_numpy("float64")
    tot_p = out["fwd_pkts"].to_numpy("float64") + out["bwd_pkts"].to_numpy("float64")
    out["flow_bytes_per_s"] = tot_b / safe
    out["flow_pkts_per_s"] = tot_p / safe
    out["fwd_pkts_per_s"] = out["fwd_pkts"].to_numpy("float64") / safe
    out["bwd_pkts_per_s"] = out["bwd_pkts"].to_numpy("float64") / safe
    fwd_p = np.maximum(out["fwd_pkts"].to_numpy("float64"), 1.0)
    out["down_up_ratio"] = out["bwd_pkts"].to_numpy("float64") / fwd_p
    for c in COMMON_FLOW_FEATURES:
        out[c] = pd.to_numeric(out[c], errors="coerce").replace(
            [np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    return out


def align_cicids_to_common(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["dur_s"] = _finite(df["Flow Duration"]) / 1e6
    out["fwd_pkts"] = _finite(df["Total Fwd Packets"])
    out["bwd_pkts"] = _finite(df["Total Backward Packets"])
    out["fwd_bytes"] = _finite(df["Total Length of Fwd Packets"])
    out["bwd_bytes"] = _finite(df["Total Length of Bwd Packets"])
    out["fwd_pkt_len_mean"] = _finite(df["Fwd Packet Length Mean"])
    out["bwd_pkt_len_mean"] = _finite(df["Bwd Packet Length Mean"])
    out = _add_common_derived(out)
    for c in ("label", "attack_category", "binary_label"):
        if c in df.columns:
            out[c] = df[c].values
    return out


def align_unsw_to_common(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["dur_s"] = _finite(df["dur"])
    out["fwd_pkts"] = _finite(df["spkts"])
    out["bwd_pkts"] = _finite(df["dpkts"])
    out["fwd_bytes"] = _finite(df["sbytes"])
    out["bwd_bytes"] = _finite(df["dbytes"])
    out["fwd_pkt_len_mean"] = _finite(df["smean"])
    out["bwd_pkt_len_mean"] = _finite(df["dmean"])
    out = _add_common_derived(out)
    for c in ("label", "attack_category", "binary_label"):
        if c in df.columns:
            out[c] = df[c].values
    return out


def make_common_schema_bundle(bundle: Dict[str, object], source: str
                              ) -> Dict[str, object]:
    """Project a CIC-IDS2017 or UNSW-NB15 bundle onto :data:`COMMON_FLOW_FEATURES`.

    Returns a bundle with the same shape but only the 12 aligned numeric
    features, so the existing leakage-safe ``prepare_dataset`` pipeline can be
    reused verbatim for the cross-dataset experiment.
    """
    align = {"cicids2017": align_cicids_to_common,
             "unsw_nb15": align_unsw_to_common}[source]
    train = align(bundle["train"])          # type: ignore[arg-type]
    test = align(bundle["test"])            # type: ignore[arg-type]
    meta = dict(bundle.get("dataset_meta", {}))   # type: ignore[call-overload]
    meta["common_schema"] = (
        f"projected onto the {len(COMMON_FLOW_FEATURES)}-feature common flow "
        "schema for cross-dataset evaluation (lossy; see common_schema.md)")
    return {
        "train": train,
        "test": test,
        "feature_columns": list(COMMON_FLOW_FEATURES),
        "categorical_columns": [],
        "numeric_columns": list(COMMON_FLOW_FEATURES),
        "categories": list(bundle["categories"]),          # type: ignore[index]
        "attack_category_map": dict(bundle.get("attack_category_map", {})),  # type: ignore[call-overload]
        "dataset_meta": meta,
    }
