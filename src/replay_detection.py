"""
Phase 7: Replay / Simulation detection pipeline.

Demonstrates the complete detection path on a **historical dataset replayed as a
stream** — it is NOT live network traffic:

    raw network-flow record  (native dataset columns, read sequentially)
        -> production preprocessing        (artifacts/<ds>/preprocessor.joblib)
        -> feature selection               (the same FittedPreprocessor)
        -> trained model                   (models/<ds>/<model>.joblib)
        -> prediction + probability/confidence
        -> DetectionEvent (timestamp, class, proba, normal/attack, latency)
        -> alert            (predicted attack AND confidence >= threshold)
        -> dashboard / CLI

Everything here is labelled ``Replay / Simulation``. Live packet capture is a
separate, *optional* concern (see ``src/realtime_detection.py`` — an inherited
Phase-1 stub that needs root and is not wired in); this module never requires
privileged access.

Reuses (does not re-implement):
  * ``src.api.REGISTRY`` / ``Registry``  — lazy model + preprocessor loading
  * ``src.api._coerce_frame``            — raw dict/list -> validated 1-row frame
  * ``src.api.ApiError``                 — typed 4xx-style validation errors
  * ``FittedPreprocessor.transform``     — the exact production preprocessing

Public API
----------
``RawFlowSource(dataset, source="raw", limit=None, shuffle_seed=None)``
``DetectionEngine(dataset, model, threshold=0.90)``  -> ``.process_batch(records)``
``ReplaySession(dataset, model, ...)``               -> ``.step(batch_size)`` / ``.stats()``
``DetectionEvent``                                   -> ``.to_dict()``
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_ROOT = REPO_ROOT / "artifacts"
MODELS_ROOT = REPO_ROOT / "models"
PROCESSED_ROOT = REPO_ROOT / "data" / "processed"
RAW_ROOT = REPO_ROOT / "data" / "raw"
REPLAY_SAMPLES_ROOT = REPO_ROOT / "data" / "replay_samples"

DATASETS = ("nsl_kdd", "cicids2017", "unsw_nb15")
MODELS = ("random_forest", "logreg", "svm_rbf")

# synthetic replay clock — a fixed past anchor, deliberately NOT wall-clock-now,
# so a replay timestamp can never be mistaken for a live capture time.
REPLAY_MODE_LABEL = "Replay / Simulation"
REPLAY_SYNTHETIC_INTERVAL_S = 2.0
REPLAY_ANCHOR = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

from src.api import Registry, REGISTRY, ApiError, _coerce_frame   # noqa: E402


# --------------------------------------------------------------------------- #
# detection event
# --------------------------------------------------------------------------- #
@dataclass
class DetectionEvent:
    seq: int
    event_time: str                     # ISO-8601, synthetic replay clock
    dataset: str
    model: str
    status: str                         # "ok" | "error"
    predicted_class: Optional[str] = None
    is_intrusion: Optional[bool] = None
    confidence: Optional[float] = None
    probabilities: Optional[Dict[str, float]] = None
    decision_function: Optional[float] = None
    threat_level: str = "unknown"
    alert: bool = False
    true_binary: Optional[str] = None
    true_category: Optional[str] = None
    correct: Optional[bool] = None
    latency_ms: Optional[float] = None
    n_feature_columns_in: Optional[int] = None
    input_warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def _threat_level(pred_class: Optional[str], confidence: Optional[float]) -> str:
    if pred_class == "normal":
        return "none"
    if pred_class is None:
        return "unknown"
    if confidence is None:
        return "unknown"
    if confidence >= 0.90:
        return "high"
    if confidence >= 0.70:
        return "medium"
    return "low"


def _synthetic_time(seq: int) -> str:
    return (REPLAY_ANCHOR + timedelta(seconds=seq * REPLAY_SYNTHETIC_INTERVAL_S)
            ).isoformat(timespec="milliseconds").replace("+00:00", "Z")


# --------------------------------------------------------------------------- #
# raw flow source — reads native records sequentially from a dataset
# --------------------------------------------------------------------------- #
class RawFlowSource:
    """Yields raw network-flow records (dicts keyed by the dataset's native
    feature columns) in file order. Carries the ground-truth label alongside
    (``_true_binary`` / ``_true_category``) purely for scoring the replay — it
    is never fed to the model."""

    def __init__(self, dataset: str, limit: Optional[int] = None,
                 shuffle_seed: Optional[int] = None):
        if dataset not in DATASETS:
            raise ValueError(f"unknown dataset {dataset!r}")
        self.dataset = dataset
        self._rows: List[dict] = []
        self._true_binary: List[Optional[str]] = []
        self._true_category: List[Optional[str]] = []
        self._load()
        if shuffle_seed is not None:
            idx = np.random.RandomState(shuffle_seed).permutation(len(self._rows))
            self._rows = [self._rows[i] for i in idx]
            self._true_binary = [self._true_binary[i] for i in idx]
            self._true_category = [self._true_category[i] for i in idx]
        if limit is not None:
            self._rows = self._rows[:limit]
            self._true_binary = self._true_binary[:limit]
            self._true_category = self._true_category[:limit]

    def __len__(self) -> int:
        return len(self._rows)

    def batches(self, batch_size: int):
        n = len(self._rows)
        for lo in range(0, n, batch_size):
            hi = min(lo + batch_size, n)
            yield (list(range(lo, hi)),
                   self._rows[lo:hi],
                   self._true_binary[lo:hi],
                   self._true_category[lo:hi])

    # -- per-dataset raw readers -------------------------------------------- #
    def _load(self) -> None:
        if self.dataset == "nsl_kdd":
            self._load_nsl_kdd()
        elif self.dataset == "unsw_nb15":
            self._load_unsw_nb15()
        elif self.dataset == "cicids2017":
            self._load_cicids2017()

    def _feature_columns(self) -> List[str]:
        meta = json.loads(
            (ARTIFACTS_ROOT / self.dataset / "metadata.json").read_text())
        return list(meta["feature_columns_expected"])

    def _load_nsl_kdd(self) -> None:
        from src.data_loaders import (NSL_KDD_RAW_COLUMNS, NSL_KDD_FEATURES,
                                      NSL_KDD_ATTACK_CATEGORY)
        p = RAW_ROOT / "nsl_kdd" / "KDDTest+.txt"
        if not p.exists():
            raise FileNotFoundError(f"{p} not found (see DEVELOPMENT_NOTES §8)")
        df = pd.read_csv(p, header=None, names=NSL_KDD_RAW_COLUMNS)
        df["label"] = df["label"].astype(str).str.strip().str.lower()
        feats = df[NSL_KDD_FEATURES]
        self._rows = feats.to_dict("records")
        self._true_binary = [("normal" if lab == "normal" else "attack")
                             for lab in df["label"]]
        self._true_category = [NSL_KDD_ATTACK_CATEGORY.get(lab, "unknown")
                               for lab in df["label"]]

    def _load_unsw_nb15(self) -> None:
        p = RAW_ROOT / "unsw_nb15" / "UNSW_NB15_testing-set.csv"
        if not p.exists():
            raise FileNotFoundError(f"{p} not found (see DEVELOPMENT_NOTES §8)")
        df = pd.read_csv(p)
        df.columns = [c.strip() for c in df.columns]
        feat_cols = self._feature_columns()
        self._rows = df[feat_cols].to_dict("records")
        cat = df["attack_cat"].fillna("Normal").astype(str).str.strip().str.lower()
        self._true_binary = [("normal" if c in ("", "normal") else "attack")
                             for c in cat]
        self._true_category = list(cat.replace({"": "normal"}))

    def _load_cicids2017(self) -> None:
        p = REPLAY_SAMPLES_ROOT / "cicids2017_raw_sample.csv"
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found — generate it once with "
                "scripts/make_replay_samples.py (needs data/raw/cicids2017/).")
        df = pd.read_csv(p)
        feat_cols = self._feature_columns()
        self._rows = df[feat_cols].to_dict("records")
        self._true_binary = list(df.get("_true_binary",
                                        pd.Series(["?"] * len(df))))
        self._true_category = list(df.get("_true_category",
                                          pd.Series(["?"] * len(df))))


# --------------------------------------------------------------------------- #
# detection engine — one (dataset, model); raw record(s) -> DetectionEvent(s)
# --------------------------------------------------------------------------- #
class DetectionEngine:
    def __init__(self, dataset: str, model: str, threshold: float = 0.90,
                 registry: Optional[Registry] = None):
        self.dataset = dataset
        self.model = model
        self.threshold = float(threshold)
        self.load_error: Optional[str] = None
        self._blob: Optional[dict] = None
        reg = registry or REGISTRY
        try:
            self._blob = reg.get(dataset, model)
        except ApiError as exc:
            self.load_error = f"{exc.status}: {exc.detail}"
        except Exception as exc:                      # pragma: no cover
            self.load_error = f"{type(exc).__name__}: {exc}"

    @property
    def ready(self) -> bool:
        return self._blob is not None

    @property
    def classes(self) -> List[str]:
        return self._blob["classes"] if self._blob else ["normal", "attack"]

    @property
    def n_feature_columns(self) -> int:
        return self._blob["n_feature_columns"] if self._blob else 0

    def set_threshold(self, t: float) -> None:
        self.threshold = float(t)

    # -- core ------------------------------------------------------------- #
    def process_batch(self, seqs: List[int], records: List[dict],
                      true_binary: Optional[List] = None,
                      true_category: Optional[List] = None) -> List[DetectionEvent]:
        if not self.ready:
            return [self._error_event(s, "model unavailable: " + str(self.load_error))
                    for s in seqs]
        pre = self._blob["preprocessor"]
        est = self._blob["estimator"]
        classes = self._blob["classes"]
        n = len(records)
        tb = true_binary if true_binary is not None else [None] * n
        tc = true_category if true_category is not None else [None] * n

        # latency covers the WHOLE per-record detection path: validation +
        # production preprocessing + feature selection + model predict.
        t_all0 = time.perf_counter()

        # validate + build a batch frame; on per-record validation failure the
        # record gets an error event and is dropped from the model call.
        frames: List[pd.DataFrame] = []
        idx_ok: List[int] = []
        events: List[Optional[DetectionEvent]] = [None] * n
        warns: List[List[str]] = [[] for _ in range(n)]
        for i, rec in enumerate(records):
            try:
                df1, w = _coerce_frame(rec, pre)
                frames.append(df1)
                warns[i] = w
                idx_ok.append(i)
            except ApiError as exc:
                events[i] = self._error_event(seqs[i], f"{exc.detail}", tb[i], tc[i])
            except Exception as exc:                  # pragma: no cover
                events[i] = self._error_event(
                    seqs[i], f"input error: {type(exc).__name__}: {exc}",
                    tb[i], tc[i])

        if frames:
            batch_df = pd.concat(frames, ignore_index=True)
            try:
                with np.errstate(over="ignore", invalid="ignore"):
                    X = pre.transform(batch_df)      # exact production preprocessing
                # out-of-range inputs can overflow to +/-inf here; caught below
                finite_row = np.isfinite(X).all(axis=1)
                y_idx = np.full(len(X), -1, dtype=int)
                proba = None
                decfn = None
                good = np.where(finite_row)[0]
                if len(good):
                    Xg = X[good]
                    y_idx[good] = est.predict(Xg).astype(int)
                    if self._blob["has_proba"]:
                        pg = est.predict_proba(Xg)
                        proba = np.full((len(X), pg.shape[1]), np.nan)
                        proba[good] = pg
                    elif self._blob["has_decfn"]:
                        dg = np.ravel(est.decision_function(Xg))
                        decfn = np.full(len(X), np.nan)
                        decfn[good] = dg
                elapsed_ms = (time.perf_counter() - t_all0) * 1e3
                per_ms = elapsed_ms / max(1, len(X))
                for k, i in enumerate(idx_ok):
                    if not finite_row[k]:
                        events[i] = self._error_event(
                            seqs[i], "non-finite feature values after "
                            "preprocessing (out-of-range input?)", tb[i], tc[i],
                            warns[i])
                        continue
                    events[i] = self._ok_event(
                        seqs[i], int(y_idx[k]),
                        None if proba is None else proba[k],
                        None if decfn is None else float(decfn[k]),
                        classes, per_ms, tb[i], tc[i], warns[i])
            except Exception as exc:
                elapsed_ms = (time.perf_counter() - t_all0) * 1e3
                for i in idx_ok:
                    events[i] = self._error_event(
                        seqs[i],
                        f"preprocessing/model failure: {type(exc).__name__}: {exc}",
                        tb[i], tc[i], warns[i], latency_ms=elapsed_ms / max(1, len(idx_ok)))

        return [e for e in events if e is not None]

    # -- event builders ------------------------------------------------- #
    def _ok_event(self, seq, y_idx, p_row, decfn, classes, latency_ms,
                  tb, tc, warns) -> DetectionEvent:
        pred = classes[y_idx] if 0 <= y_idx < len(classes) else str(y_idx)
        probs = None
        conf = None
        if p_row is not None and np.isfinite(p_row).all():
            probs = {classes[j]: round(float(p_row[j]), 6)
                     for j in range(len(classes))}
            conf = round(float(np.max(p_row)), 6)
        is_intr = pred != "normal"
        alert = bool(is_intr and (conf is None or conf >= self.threshold))
        correct = None if tb is None else (pred == tb)
        return DetectionEvent(
            seq=seq, event_time=_synthetic_time(seq),
            dataset=self.dataset, model=self.model, status="ok",
            predicted_class=pred, is_intrusion=is_intr,
            confidence=conf, probabilities=probs,
            decision_function=(None if decfn is None or not np.isfinite(decfn)
                               else round(decfn, 6)),
            threat_level=_threat_level(pred, conf), alert=alert,
            true_binary=tb, true_category=tc, correct=correct,
            latency_ms=round(latency_ms, 4),
            n_feature_columns_in=self.n_feature_columns,
            input_warnings=list(warns or []),
        )

    def _error_event(self, seq, reason, tb=None, tc=None, warns=None,
                     latency_ms=None) -> DetectionEvent:
        return DetectionEvent(
            seq=seq, event_time=_synthetic_time(seq),
            dataset=self.dataset, model=self.model, status="error",
            threat_level="unknown", alert=False,
            true_binary=tb, true_category=tc,
            latency_ms=(None if latency_ms is None else round(latency_ms, 4)),
            n_feature_columns_in=self.n_feature_columns,
            input_warnings=list(warns or []), error=str(reason),
        )


# --------------------------------------------------------------------------- #
# replay session — source + engine + controls + live statistics
# --------------------------------------------------------------------------- #
class ReplaySession:
    def __init__(self, dataset: str, model: str, threshold: float = 0.90,
                 limit: Optional[int] = None, shuffle_seed: Optional[int] = None,
                 keep_events: int = 5000):
        self.dataset = dataset
        self.model = model
        self.keep_events = int(keep_events)
        self.state = "idle"                 # idle|running|paused|stopped|finished
        self.load_error: Optional[str] = None
        self.engine = DetectionEngine(dataset, model, threshold)
        try:
            self.source: Optional[RawFlowSource] = RawFlowSource(
                dataset, limit=limit, shuffle_seed=shuffle_seed)
        except Exception as exc:
            self.source = None
            self.load_error = f"{type(exc).__name__}: {exc}"
        if self.engine.load_error and not self.load_error:
            self.load_error = self.engine.load_error
        self._batch_iter = None
        self.cursor = 0
        self.events: List[DetectionEvent] = []       # bounded ring buffer
        self._reset_counters()

    # -- counters ------------------------------------------------------- #
    def _reset_counters(self) -> None:
        self.n_processed = 0
        self.n_errors = 0
        self.n_pred_attack = 0
        self.n_pred_normal = 0
        self.n_alerts = 0
        self.n_high_alerts = 0
        self.n_label_correct = 0
        self.n_label_known = 0
        self._lat: List[float] = []
        self._proc_wall_s = 0.0
        self._true_attack = 0
        self._true_normal = 0

    @property
    def total(self) -> int:
        return len(self.source) if self.source else 0

    # -- controls ----------------------------------------------------- #
    def start(self) -> None:
        if self.load_error:
            self.state = "stopped"; return
        if self.state in ("idle", "paused", "stopped"):
            self.state = "running"

    def pause(self) -> None:
        if self.state == "running":
            self.state = "paused"

    def stop(self) -> None:
        self.state = "stopped"

    def reset(self) -> None:
        """Clear all events + counters and rewind to the first record."""
        self.cursor = 0
        self._batch_iter = None
        self.events = []
        self._reset_counters()
        self.state = "idle"

    def set_threshold(self, t: float) -> None:
        self.engine.set_threshold(t)
        # re-derive alert flags on retained events for a consistent view
        for e in self.events:
            if e.status == "ok" and e.is_intrusion:
                e.alert = bool(e.confidence is None or e.confidence >= t)
                e.threat_level = _threat_level(e.predicted_class, e.confidence)

    # -- advance ---------------------------------------------------- #
    def step(self, batch_size: int = 32) -> List[DetectionEvent]:
        if self.load_error or self.source is None:
            self.state = "stopped"
            return []
        if self.state not in ("running", "idle"):
            return []
        if self.state == "idle":
            self.state = "running"
        if self.cursor >= self.total:
            self.state = "finished"
            return []
        lo, hi = self.cursor, min(self.cursor + batch_size, self.total)
        seqs = list(range(lo, hi))
        recs = self.source._rows[lo:hi]
        tb = self.source._true_binary[lo:hi]
        tc = self.source._true_category[lo:hi]

        w0 = time.perf_counter()
        evs = self.engine.process_batch(seqs, recs, tb, tc)
        self._proc_wall_s += time.perf_counter() - w0

        for e in evs:
            self.n_processed += 1
            if e.status == "error":
                self.n_errors += 1
            else:
                if e.predicted_class == "normal":
                    self.n_pred_normal += 1
                else:
                    self.n_pred_attack += 1
                if e.alert:
                    self.n_alerts += 1
                    if e.threat_level == "high":
                        self.n_high_alerts += 1
                if e.latency_ms is not None:
                    self._lat.append(e.latency_ms)
            if e.true_binary in ("normal", "attack"):
                self.n_label_known += 1
                self._true_attack += (e.true_binary == "attack")
                self._true_normal += (e.true_binary == "normal")
                if e.correct:
                    self.n_label_correct += 1
        self.events.extend(evs)
        if len(self.events) > self.keep_events:
            self.events = self.events[-self.keep_events:]
        self.cursor = hi
        if self.cursor >= self.total:
            self.state = "finished"
        return evs

    def run_to_end(self, batch_size: int = 64, progress_every: int = 0,
                   printer=None) -> None:
        self.start()
        while self.state == "running":
            self.step(batch_size)
            if progress_every and printer and self.n_processed % progress_every < batch_size:
                printer(self.stats())

    # -- statistics (all measured) --------------------------------- #
    def stats(self) -> dict:
        lat = np.asarray(self._lat, dtype=float)
        has_lat = lat.size > 0
        return {
            "replay_mode": REPLAY_MODE_LABEL,
            "is_live_capture": False,
            "dataset": self.dataset,
            "model": self.model,
            "threshold": self.engine.threshold,
            "state": self.state,
            "records_total": self.total,
            "samples_processed": self.n_processed,
            "errors": self.n_errors,
            "detections_alerts": self.n_alerts,
            "high_severity_alerts": self.n_high_alerts,
            "attacks_detected": self.n_pred_attack,
            "normal_samples": self.n_pred_normal,
            "true_attack_in_stream": self._true_attack,
            "true_normal_in_stream": self._true_normal,
            "agreement_with_labels": (round(self.n_label_correct / self.n_label_known, 4)
                                      if self.n_label_known else None),
            "latency_ms": {
                "avg": round(float(lat.mean()), 4) if has_lat else None,
                "min": round(float(lat.min()), 4) if has_lat else None,
                "max": round(float(lat.max()), 4) if has_lat else None,
                "p50": round(float(np.percentile(lat, 50)), 4) if has_lat else None,
                "p95": round(float(np.percentile(lat, 95)), 4) if has_lat else None,
                "n": int(lat.size),
            },
            "throughput_records_per_s": (round(self.n_processed / self._proc_wall_s, 1)
                                         if self._proc_wall_s > 0 else None),
            "processing_wall_seconds": round(self._proc_wall_s, 4),
        }

    def recent_events(self, n: int = 25) -> List[dict]:
        return [e.to_dict() for e in self.events[-n:]]

    def recent_alerts(self, n: int = 20) -> List[dict]:
        al = [e.to_dict() for e in self.events if e.alert]
        return al[-n:]

    def probe_single_record_latency(self, n: int = 200) -> dict:
        """Measure true one-at-a-time (batch=1) transform+predict latency on the
        first ``n`` records, WITHOUT touching the session counters/cursor."""
        if self.load_error or self.source is None:
            return {"n": 0}
        n = min(n, self.total)
        lat: List[float] = []
        for i in range(n):
            w = time.perf_counter()
            self.engine.process_batch([i], [self.source._rows[i]])
            lat.append((time.perf_counter() - w) * 1e3)
        a = np.asarray(lat)
        return {
            "n": int(a.size),
            "avg_ms": round(float(a.mean()), 4),
            "min_ms": round(float(a.min()), 4),
            "max_ms": round(float(a.max()), 4),
            "p50_ms": round(float(np.percentile(a, 50)), 4),
            "p95_ms": round(float(np.percentile(a, 95)), 4),
        }


# --------------------------------------------------------------------------- #
# discovery helper (used by the CLI + dashboard)
# --------------------------------------------------------------------------- #
def available_replay() -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for ds in DATASETS:
        pre = ARTIFACTS_ROOT / ds / "preprocessor.joblib"
        if not pre.exists():
            continue
        models = [m for m in MODELS if (MODELS_ROOT / ds / f"{m}.joblib").exists()]
        if not models:
            continue
        raw = {
            "nsl_kdd": RAW_ROOT / "nsl_kdd" / "KDDTest+.txt",
            "unsw_nb15": RAW_ROOT / "unsw_nb15" / "UNSW_NB15_testing-set.csv",
            "cicids2017": REPLAY_SAMPLES_ROOT / "cicids2017_raw_sample.csv",
        }[ds]
        out[ds] = {"models": models, "raw_source": str(raw.relative_to(REPO_ROOT)),
                   "raw_source_present": raw.exists()}
    return out
