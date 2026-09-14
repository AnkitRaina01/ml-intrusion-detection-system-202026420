"""
Phase 6 intrusion detection monitoring dashboard (Streamlit).

Shows **only real artefacts** produced by Phases 2-5:
  * system status / component check
  * dataset + model information            (artifacts/<ds>/*, results/<ds>/dataset_stats.json)
  * offline model performance metrics       (results/<ds>/metrics.json)
  * confusion matrices                      (from metrics.json)
  * model comparison + within-vs-cross      (results/<ds>/metrics.json, results/cross_dataset/metrics.json)
  * inference latency                       (metrics.json + this replay session, measured)
  * a clearly-labelled REPLAY of held-out test records for the
    normal/malicious feed, attack-category view, recent-detections table and
    alert panel.

Replay ≠ live capture. Live packet capture needs root and is unavailable in this
environment (DEVELOPMENT_NOTES.md §5, limit L1). Every replay-derived view is
labelled "REPLAY / SIMULATION" and the replay clock is explicitly synthetic.
This module never presents historical data as live network traffic.

Reuses the inherited ``src/dashboard.py`` *pattern* (Streamlit, sidebar page
nav, alert CSS) — its hard-coded placeholder content could not be reused. The
inherited file is left untouched.

Run:  streamlit run src/monitoring_dashboard.py
  or: python scripts/run_dashboard.py
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
RESULTS_ROOT = REPO_ROOT / "results"
PROCESSED_ROOT = REPO_ROOT / "data" / "processed"
LOGS_DIR = REPO_ROOT / "logs"

DATASETS = ("nsl_kdd", "cicids2017", "unsw_nb15")
SKLEARN_MODELS = ("random_forest", "logreg", "svm_rbf")
PRETTY_MODEL = {"random_forest": "Random Forest", "logreg": "Logistic Regression",
                "svm_rbf": "SVM (RBF)", "lstm": "LSTM"}
DATASET_DISPLAY = {"nsl_kdd": "NSL-KDD", "cicids2017": "CIC-IDS2017",
                   "unsw_nb15": "UNSW-NB15"}

# one replayed record == this many synthetic seconds on the replay clock
REPLAY_SYNTHETIC_INTERVAL_S = 2.0
REPLAY_BANNER = (
    "⚠️  **REPLAY / SIMULATION MODE** — the feed, attack-category chart, "
    "recent-detections table and alerts below are a **replay of held-out "
    "test-set records**, scored offline by the trained model. This is **not "
    "live network traffic**; timestamps are a synthetic replay clock.")

# Phase-7 full-pipeline replay engine (raw record -> production preprocessing ->
# feature selection -> model -> event -> alert). Optional import: the dashboard
# falls back to the fast preprocessed-matrix path if it is unavailable.
try:
    from src.replay_detection import ReplaySession as _ReplaySession  # noqa: F401
    _FULL_REPLAY_OK = True
except Exception:                                         # pragma: no cover
    _ReplaySession = None
    _FULL_REPLAY_OK = False


# =========================================================================== #
# real-data loaders (pure functions, no Streamlit — unit-testable)
# =========================================================================== #
def discover() -> Dict[str, Any]:
    """Component / readiness check across every dataset."""
    out: Dict[str, Any] = {"datasets": {}, "generated": _now_iso()}
    for ds in DATASETS:
        art = ARTIFACTS_ROOT / ds
        mdl = MODELS_ROOT / ds
        res = RESULTS_ROOT / ds
        proc = PROCESSED_ROOT / ds
        models_present = [m for m in SKLEARN_MODELS
                          if (mdl / f"{m}.joblib").exists()]
        entry = {
            "preprocessor": (art / "preprocessor.joblib").exists(),
            "metadata": (art / "metadata.json").exists(),
            "models_trained": models_present,
            "metrics_json": (res / "metrics.json").exists(),
            "dataset_stats": (res / "dataset_stats.json").exists(),
            "processed_test_matrix": (proc / "X_test.npy").exists(),
            "confusion_plots": sorted(p.name for p in (res / "plots").glob(
                "confusion_matrix_*.png")) if (res / "plots").exists() else [],
        }
        entry["ready_for_replay"] = bool(
            entry["processed_test_matrix"] and models_present)
        out["datasets"][ds] = entry
    out["cross_dataset_metrics"] = (RESULTS_ROOT / "cross_dataset"
                                    / "metrics.json").exists()
    out["api_log"] = _api_log_status()
    out["environment"] = _environment()
    return out


def _environment() -> Dict[str, str]:
    import platform
    env = {"python": platform.python_version(), "numpy": np.__version__,
           "pandas": pd.__version__}
    try:
        import sklearn
        env["scikit_learn"] = sklearn.__version__
    except Exception:
        pass
    try:
        import streamlit
        env["streamlit"] = streamlit.__version__
    except Exception:
        pass
    return env


def _api_log_status() -> Dict[str, Any]:
    p = LOGS_DIR / "api.log"
    if not p.exists():
        return {"present": False}
    lines = p.read_text(errors="replace").splitlines()
    predict_lines = [ln for ln in lines if "POST /predict" in ln]
    return {"present": True, "path": str(p.relative_to(REPO_ROOT)),
            "size_bytes": p.stat().st_size, "total_lines": len(lines),
            "predict_requests": len(predict_lines),
            "tail": lines[-12:]}


def load_metrics(dataset: str) -> Optional[dict]:
    p = RESULTS_ROOT / dataset / "metrics.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def load_dataset_info(dataset: str) -> Optional[dict]:
    res: Dict[str, Any] = {"dataset": dataset,
                           "display": DATASET_DISPLAY.get(dataset, dataset)}
    stats_p = RESULTS_ROOT / dataset / "dataset_stats.json"
    meta_p = ARTIFACTS_ROOT / dataset / "metadata.json"
    train_p = MODELS_ROOT / dataset / "training_summary.json"
    if not stats_p.exists():
        return None
    s = json.loads(stats_p.read_text())
    res["samples"] = s.get("samples", {})
    res["splits"] = {k: {"n": s["splits"][k]["n_samples"],
                         "pct": s["splits"][k]["percent_of_total"],
                         "binary": s["splits"][k]["binary"],
                         "multiclass": s["splits"][k]["multiclass"]}
                     for k in ("train", "validation", "test")
                     if k in s.get("splits", {})}
    res["stratified_on"] = s.get("splits", {}).get("stratified_on")
    res["features"] = {
        "before": s["features"]["before_preprocessing"],
        "after_encoding": s["features"]["after_encoding_and_variance_filter"],
        "selected": s["features"]["after_selection"],
        "selected_names": s["features"]["selected_features"],
        "method": s["features"]["selection_method"],
    }
    res["class_balance"] = s.get("class_balance", {})
    res["dataset_meta"] = s.get("dataset_meta", {})
    if meta_p.exists():
        m = json.loads(meta_p.read_text())
        res["artifact_version"] = m.get("artifact_version")
        res["n_feature_columns_expected"] = len(m.get("feature_columns_expected", []))
        res["config"] = m.get("config", {})
    if train_p.exists():
        t = json.loads(train_p.read_text())
        res["training_environment"] = t.get("environment", {})
        res["model_selection"] = {
            PRETTY_MODEL.get(k, k): {
                "best_params": v.get("best_params"),
                "val_macro_f1": v.get("best_val_macro_f1"),
                "train_seconds": v.get("train_seconds"),
            } for k, v in t.get("models", {}).items()}
    return res


def load_cross_dataset() -> Optional[dict]:
    p = RESULTS_ROOT / "cross_dataset" / "metrics.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def model_comparison_frame(metrics: dict) -> pd.DataFrame:
    rows = []
    for name, blob in metrics.get("models", {}).items():
        m = blob.get("metrics", {})
        lat = blob.get("latency", {})
        rows.append({
            "model": name,
            "accuracy": m.get("accuracy"),
            "precision_attack": m.get("precision_attack"),
            "recall_attack": m.get("recall_attack"),
            "f1_attack": m.get("f1_attack"),
            "macro_f1": m.get("macro_f1"),
            "roc_auc": m.get("roc_auc"),
            "per_sample_ms": lat.get("per_sample_ms_median"),
            "throughput_per_s": lat.get("throughput_samples_per_s"),
            "train_seconds": blob.get("train_seconds"),
            "val_macro_f1": blob.get("val_macro_f1"),
            "test_n": m.get("n_samples"),
        })
    return pd.DataFrame(rows)


# =========================================================================== #
# replay engine  (offline test records -> a labelled detection feed)
# =========================================================================== #
@dataclass
class ReplayEngine:
    dataset: str
    model: str
    _X: np.ndarray = field(default=None, repr=False)
    _yb: np.ndarray = field(default=None, repr=False)
    _ym: np.ndarray = field(default=None, repr=False)
    _est: Any = field(default=None, repr=False)
    _classes: List[str] = field(default_factory=lambda: ["normal", "attack"])
    _has_proba: bool = False
    _has_decfn: bool = False
    cursor: int = 0
    per_sample_ms_measured: Optional[float] = None
    load_error: Optional[str] = None

    def __post_init__(self):
        try:
            import joblib
            proc = PROCESSED_ROOT / self.dataset
            self._X = np.load(proc / "X_test.npy")
            self._yb = np.load(proc / "y_test_binary.npy")
            ym_p = proc / "y_test_multiclass.npy"
            self._ym = (np.load(ym_p, allow_pickle=True) if ym_p.exists()
                        else np.array(["?"] * len(self._X), dtype=object))
            lbl_p = ARTIFACTS_ROOT / self.dataset / "label_maps.json"
            if lbl_p.exists():
                lm = json.loads(lbl_p.read_text())
                self._classes = [lm["binary"]["0"], lm["binary"]["1"]]
            self._est = joblib.load(MODELS_ROOT / self.dataset / f"{self.model}.joblib")
            self._has_proba = hasattr(self._est, "predict_proba")
            self._has_decfn = hasattr(self._est, "decision_function")
        except Exception as exc:                       # pragma: no cover
            self.load_error = f"{type(exc).__name__}: {exc}"

    @property
    def total(self) -> int:
        return 0 if self._X is None else int(len(self._X))

    @property
    def anchor(self) -> datetime:
        # fixed synthetic start so the replay clock is reproducible & obviously
        # not wall-clock-now
        return datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    def reset(self) -> None:
        self.cursor = 0

    def step(self, n: int) -> List[dict]:
        if self._X is None or self.cursor >= self.total:
            return []
        lo, hi = self.cursor, min(self.cursor + n, self.total)
        Xb = self._X[lo:hi]
        t0 = time.perf_counter()
        y_idx = self._est.predict(Xb).astype(int)
        proba = self._est.predict_proba(Xb) if self._has_proba else None
        dt_ms = (time.perf_counter() - t0) / max(1, hi - lo) * 1e3
        self.per_sample_ms_measured = (
            dt_ms if self.per_sample_ms_measured is None
            else 0.7 * self.per_sample_ms_measured + 0.3 * dt_ms)
        decfn = (np.ravel(self._est.decision_function(Xb))
                 if (proba is None and self._has_decfn) else None)

        out: List[dict] = []
        for i in range(hi - lo):
            row = lo + i
            pi = int(y_idx[i])
            pred = self._classes[pi] if 0 <= pi < len(self._classes) else str(pi)
            conf = float(np.max(proba[i])) if proba is not None else None
            out.append({
                "row_index": row,
                "replay_time": (self.anchor + timedelta(
                    seconds=row * REPLAY_SYNTHETIC_INTERVAL_S)),
                "dataset": self.dataset,
                "model": self.model,
                "predicted_class": pred,
                "is_intrusion": pred != "normal",
                "confidence": None if conf is None else round(conf, 4),
                "decision_function": (None if decfn is None
                                      else round(float(decfn[i]), 4)),
                "true_binary": self._classes[int(self._yb[row])],
                "true_category": str(self._ym[row]),
                "correct": pred == self._classes[int(self._yb[row])],
                "threat_level": _threat_level(pred, conf),
            })
        self.cursor = hi
        return out


def _threat_level(pred_class: str, confidence: Optional[float]) -> str:
    if pred_class == "normal":
        return "none"
    if confidence is None:
        return "unknown"
    if confidence >= 0.90:
        return "high"
    if confidence >= 0.70:
        return "medium"
    return "low"


def alerts_from(detections: List[dict], threshold: float) -> List[dict]:
    out = []
    for d in detections:
        if not d["is_intrusion"]:
            continue
        c = d["confidence"]
        if c is None or c >= threshold:
            out.append(d)
    return out


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# =========================================================================== #
# Streamlit UI
# =========================================================================== #
def _run_streamlit() -> None:
    import streamlit as st
    import plotly.express as px
    import plotly.graph_objects as go

    st.set_page_config(page_title="IDS Monitoring", page_icon="🛡️",
                       layout="wide", initial_sidebar_state="expanded")
    st.markdown("""
    <style>
      .replay-banner {background:#fff3cd;border:1px solid #ffe08a;color:#664d03;
        padding:.6rem .9rem;border-radius:.4rem;margin:.2rem 0 1rem 0;font-size:.95rem;}
      .alert-high {background:#f8d7da;color:#842029;padding:.4rem .7rem;border-radius:.3rem;}
      .ok {color:#0f5132;} .bad {color:#842029;}
    </style>""", unsafe_allow_html=True)

    ss = st.session_state
    ss.setdefault("detections", [])
    ss.setdefault("replaying", False)
    ss.setdefault("engine_key", None)
    ss.setdefault("engine", None)

    # -- sidebar ------------------------------------------------------------- #
    st.sidebar.title("🛡️ IDS Monitoring")
    disc = discover()
    ready = [d for d, e in disc["datasets"].items() if e["ready_for_replay"]]
    default_ds = "nsl_kdd" if "nsl_kdd" in ready else (ready[0] if ready else "nsl_kdd")

    page = st.sidebar.radio("View", [
        "1 · System status",
        "2 · Dataset & model info",
        "3 · Detection replay (feed / categories / recent / alerts)",
        "4 · Model performance & confusion matrix",
        "5 · Model comparison & latency",
    ])
    st.sidebar.markdown("---")
    st.sidebar.subheader("Replay / Simulation settings")
    st.sidebar.caption("Replay of a historical dataset — **not live traffic.**")
    dataset = st.sidebar.selectbox(
        "Dataset / sample", DATASETS, index=DATASETS.index(default_ds),
        format_func=lambda d: DATASET_DISPLAY.get(d, d))
    avail_models = disc["datasets"].get(dataset, {}).get("models_trained", [])
    model = st.sidebar.selectbox(
        "Model", avail_models or SKLEARN_MODELS,
        format_func=lambda m: PRETTY_MODEL.get(m, m))
    pipe_opts = (["Full pipeline (raw → preprocess → model)",
                  "Fast (pre-processed test matrix)"] if _FULL_REPLAY_OK
                 else ["Fast (pre-processed test matrix)"])
    pipeline = st.sidebar.radio("Replay pipeline", pipe_opts, index=0)
    full = pipeline.startswith("Full")
    speed = st.sidebar.slider("Records per step", 5, 200, 40, step=5)
    alert_threshold = st.sidebar.slider("Detection threshold (confidence)",
                                        0.50, 0.99, 0.90, step=0.01)

    key = (dataset, model, "full" if full else "fast")
    if full:
        if ss.get("full_key") != key or ss.get("full_session") is None:
            ss.full_session = _ReplaySession(dataset, model,
                                             threshold=alert_threshold)
            ss.full_key = key
            ss.replaying = False
        session = ss.full_session
        session.set_threshold(alert_threshold)
        c1, c2, c3 = st.sidebar.columns(3)
        if c1.button("▶ Start", disabled=bool(session.load_error)):
            session.start(); ss.replaying = True
        if c2.button("⏸ Pause"):
            session.pause(); ss.replaying = False
        if c3.button("⏹ Stop"):
            session.stop(); ss.replaying = False
        b1, b2 = st.sidebar.columns(2)
        if b1.button("⏭ Step once", disabled=bool(session.load_error)):
            session.step(speed)
        if b2.button("🗑 Reset / clear"):
            session.reset(); ss.replaying = False
        if session.load_error:
            st.sidebar.error(f"replay unavailable: {session.load_error}")
        else:
            st.sidebar.caption(
                f"cursor {session.cursor:,} / {session.total:,} raw records "
                f"· state: {session.state}")
        if ss.replaying and not session.load_error:
            session.step(speed)
            if session.state != "running":
                ss.replaying = False
        engine = ss.get("engine")               # kept for view 4's latency line
        df = pd.DataFrame()                      # full path renders from session
    else:
        if ss.engine_key != key or ss.engine is None:
            ss.engine = ReplayEngine(dataset, model)
            ss.engine_key = key
            ss.detections = []
            ss.replaying = False
        engine = ss.engine
        session = None
        c1, c2, c3 = st.sidebar.columns(3)
        if c1.button("▶ Start", disabled=engine.load_error is not None):
            ss.replaying = True
        if c2.button("⏸ Stop"):
            ss.replaying = False
        if c3.button("⟳ Reset"):
            engine.reset(); ss.detections = []; ss.replaying = False
        if st.sidebar.button("⏭ Step once", disabled=engine.load_error is not None):
            ss.detections.extend(engine.step(speed))
        if engine.load_error:
            st.sidebar.error(f"replay unavailable: {engine.load_error}")
        else:
            st.sidebar.caption(
                f"replay cursor {engine.cursor:,} / {engine.total:,} test records "
                f"({DATASET_DISPLAY.get(dataset, dataset)})")
        if ss.replaying and engine.load_error is None:
            new = engine.step(speed)
            ss.detections.extend(new)
            if not new or engine.cursor >= engine.total:
                ss.replaying = False
        df = pd.DataFrame(ss.detections) if ss.detections else pd.DataFrame()

    st.title("Intrusion Detection — Monitoring Dashboard")
    st.caption("All metrics are from offline evaluation on held-out test sets "
               "(Phases 3-4). The detection feed is a labelled REPLAY — not live "
               "capture.")

    # =================================================================== #
    if page.startswith("1"):
        _view_system_status(st, disc)
    elif page.startswith("2"):
        _view_dataset_model_info(st, px, dataset)
    elif page.startswith("3"):
        if full and session is not None:
            _view_replay_full(st, px, session, dataset, model, alert_threshold)
        else:
            _view_replay(st, px, go, df, engine, dataset, model, alert_threshold)
        if ss.replaying:
            time.sleep(0.6)
            st.rerun()
    elif page.startswith("4"):
        _view_performance(st, go, dataset, engine)
    elif page.startswith("5"):
        _view_comparison(st, px, go, dataset)


def _view_system_status(st, disc):
    st.header("1 · System status")
    st.write(f"Snapshot generated {disc['generated']}")
    rows = []
    for ds, e in disc["datasets"].items():
        tick = lambda b: "✓" if b else "—"  # noqa: E731
        rows.append({
            "dataset": DATASET_DISPLAY.get(ds, ds),
            "preprocessor.joblib": tick(e["preprocessor"]),
            "models trained": ", ".join(e["models_trained"]) or "—",
            "metrics.json": tick(e["metrics_json"]),
            "dataset_stats.json": tick(e["dataset_stats"]),
            "processed X_test": tick(e["processed_test_matrix"]),
            "replay ready": tick(e["ready_for_replay"]),
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Environment")
        st.table(pd.DataFrame(disc["environment"].items(),
                              columns=["package", "version"]))
        st.write(f"cross-dataset metrics present: "
                 f"{'✓' if disc['cross_dataset_metrics'] else '—'}")
    with c2:
        st.subheader("Prediction API request log")
        al = disc["api_log"]
        if not al.get("present"):
            st.info("`logs/api.log` not found — the Phase-5 API has not been "
                    "run yet. (Start it with `python scripts/serve_api.py`.)")
        else:
            st.write(f"`{al['path']}` · {al['size_bytes']:,} B · "
                     f"{al['total_lines']:,} lines · "
                     f"{al['predict_requests']:,} real `/predict` requests")
            st.code("\n".join(al["tail"]), language="text")
        st.caption("This is the only source of *real* request traffic; it is "
                   "empty until the API is used. The replay feed is separate "
                   "and clearly labelled.")


def _view_dataset_model_info(st, px, dataset):
    st.header("2 · Dataset & model information")
    info = load_dataset_info(dataset)
    if not info:
        st.warning(f"no dataset_stats.json for {dataset} — run "
                   f"`python scripts/prepare_datasets.py --dataset {dataset}`")
        return
    st.subheader(f"{info['display']}")
    dm = info.get("dataset_meta") or {}
    if dm.get("subsample"):
        st.markdown(f"<div class='replay-banner'>Sub-sampling: "
                    f"{dm['subsample']}</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("raw grand total", f"{info['samples'].get('grand_total', 0):,}")
    c2.metric("feature columns (native)",
              info.get("n_feature_columns_expected", info["features"]["before"]))
    c3.metric("features used by model", info["features"]["selected"])
    c1.metric("after one-hot / variance filter", info["features"]["after_encoding"])
    c2.metric("preprocessor artifact version", info.get("artifact_version", "—"))
    c3.metric("stratified on", str(info.get("stratified_on", "—")))

    st.subheader("Train / validation / test split")
    srows = [{"partition": k, "rows": v["n"], "% of total": round(v["pct"], 2)}
             for k, v in info["splits"].items()]
    st.dataframe(pd.DataFrame(srows), width="stretch", hide_index=True)

    st.subheader("Class distribution — test split")
    tb = info["splits"].get("test", {})
    if tb:
        bcol, mcol = st.columns(2)
        with bcol:
            st.caption("binary")
            bd = pd.DataFrame([{"class": k, "count": v["count"],
                                "percent": v["percent"]}
                               for k, v in tb["binary"].items()])
            st.plotly_chart(px.bar(bd, x="class", y="count", text="count",
                                   title="binary (test)"),
                            width="stretch")
        with mcol:
            st.caption(f"attack categories ({info.get('stratified_on', '')})")
            md = pd.DataFrame([{"category": k, "count": v["count"],
                               "percent": v["percent"]}
                              for k, v in tb["multiclass"].items()]
                             ).sort_values("count", ascending=False)
            st.plotly_chart(px.bar(md, x="category", y="count", text="count",
                                   title="attack categories (test)"),
                            width="stretch")
            st.dataframe(md, width="stretch", hide_index=True)

    st.subheader("Model selection (validation macro-F1, from training)")
    ms = info.get("model_selection", {})
    if ms:
        st.dataframe(pd.DataFrame([
            {"model": k, "val_macro_f1": round(v["val_macro_f1"], 4)
             if v.get("val_macro_f1") is not None else None,
             "train_seconds": round(v["train_seconds"], 1)
             if v.get("train_seconds") is not None else None,
             "best_params": json.dumps(v["best_params"])}
            for k, v in ms.items()]),
            width="stretch", hide_index=True)

    st.subheader(f"Selected features ({info['features']['selected']})")
    st.caption(info["features"]["method"])
    st.code("\n".join(info["features"]["selected_names"]), language="text")


def _view_replay_full(st, px, session, dataset, model, threshold):
    """Phase-7 full-pipeline replay view: raw record -> production preprocessing
    -> feature selection -> model -> detection event -> alert."""
    st.header("3 · Detection replay — full pipeline")
    st.markdown(f"<div class='replay-banner'>{REPLAY_BANNER}<br>"
                "Each record is read from the dataset's <b>raw</b> file and put "
                "through the <b>exact production preprocessing</b> "
                "(<code>artifacts/&lt;ds&gt;/preprocessor.joblib</code>) → feature "
                "selection → trained model → detection event → alert.</div>",
                unsafe_allow_html=True)
    if session.load_error:
        st.error(f"replay unavailable: {session.load_error}")
        return
    try:
        from src.replay_detection import available_replay
        raw_src = available_replay().get(dataset, {}).get("raw_source", "?")
    except Exception:
        raw_src = "?"
    st.write(f"Pipeline: **raw flow record** → preprocessing → feature selection "
             f"→ **{PRETTY_MODEL.get(model, model)}** → prediction → detection "
             f"event → alert.  Source: `{raw_src}` "
             f"({session.total:,} records) · replayed **{session.cursor:,}** "
             f"· state **{session.state}**.")

    s = session.stats()
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("samples processed", f"{s['samples_processed']:,}")
    c2.metric("predicted normal", f"{s['normal_samples']:,}")
    c3.metric("predicted attack", f"{s['attacks_detected']:,}")
    c4.metric("detections (alerts)", f"{s['detections_alerts']:,}")
    c5.metric("errors", f"{s['errors']:,}")
    c6.metric("agreement w/ labels",
              "—" if s["agreement_with_labels"] is None
              else f"{s['agreement_with_labels']*100:.1f}%")

    lm = s["latency_ms"]
    l1, l2, l3, l4 = st.columns(4)
    l1.metric("inference latency — avg",
              "—" if lm["avg"] is None else f"{lm['avg']:.2f} ms")
    l2.metric("min / max",
              "—" if lm["min"] is None else f"{lm['min']:.2f} / {lm['max']:.2f} ms")
    l3.metric("p50 / p95",
              "—" if lm["p50"] is None else f"{lm['p50']:.2f} / {lm['p95']:.2f} ms")
    l4.metric("throughput",
              "—" if s["throughput_records_per_s"] is None
              else f"{s['throughput_records_per_s']:,.0f} rec/s")
    st.caption("Latency = one micro-batch transform+predict ÷ batch size — "
               "**measured this replay session**, not a stored figure.")

    if not session.events:
        st.info("Press **▶ Start** or **⏭ Step once** in the sidebar to begin "
                "the replay.")
        return

    ev = pd.DataFrame([e.to_dict() for e in session.events])
    ok = ev[ev["status"] == "ok"].copy()

    # ---- (3) normal vs malicious over replay time ---- #
    if len(ok):
        st.subheader("Predicted normal vs malicious over replay time")
        ok["bucket"] = (ok["seq"] // 50) * 50
        agg = (ok.groupby(["bucket", "predicted_class"]).size()
               .reset_index(name="count"))
        st.plotly_chart(
            px.area(agg, x="bucket", y="count", color="predicted_class",
                    labels={"bucket": "replay record index"},
                    title="rolling counts (REPLAY — synthetic order, not live)"),
            width="stretch")

    # ---- (4) attack categories (ground-truth of replayed malicious records) ---- #
    st.subheader("Attack categories among replayed malicious records "
                 "(from ground-truth labels)")
    mal = ev[(ev["true_binary"] == "attack") & (ev["true_category"].notna())]
    if len(mal):
        cat = mal["true_category"].value_counts().reset_index()
        cat.columns = ["category", "count"]
        a, b = st.columns([2, 1])
        a.plotly_chart(px.bar(cat, x="category", y="count", text="count",
                              title="true attack categories (replayed)"),
                       width="stretch")
        b.dataframe(cat, width="stretch", hide_index=True)
        st.caption("Served models are **binary** (normal vs attack); category is "
                   "the record's true label, shown for context.")
    else:
        st.write("No malicious records replayed yet.")

    # ---- (5) recent detection events ---- #
    st.subheader("Recent detection events (most recent 25 — REPLAY)")
    cols = ["event_time", "seq", "status", "predicted_class", "confidence",
            "threat_level", "alert", "true_binary", "true_category", "correct",
            "latency_ms", "error"]
    recent = ev.tail(25).iloc[::-1][[c for c in cols if c in ev.columns]]
    st.dataframe(recent, width="stretch", hide_index=True)

    # ---- (10) alerts ---- #
    st.subheader("Alerts raised during this REPLAY session")
    al = pd.DataFrame(session.recent_alerts(200))
    x1, x2, x3 = st.columns(3)
    x1.metric("alerts", f"{s['detections_alerts']:,}")
    x2.metric("high severity", f"{s['high_severity_alerts']:,}")
    x3.metric("threshold", f"≥ {threshold:.2f} confidence")
    if len(al):
        show = al.tail(20).iloc[::-1][[c for c in
              ["event_time", "seq", "predicted_class", "confidence",
               "threat_level", "true_category"] if c in al.columns]]
        st.dataframe(show, width="stretch", hide_index=True)
        st.caption("An alert = a replayed record predicted *attack* with "
                   "confidence ≥ threshold (or an uncalibrated-SVM attack "
                   "prediction). **REPLAY-derived — not a live alert.**")
    else:
        st.write("No alerts at this threshold yet.")

    # ---- errors (if any) ---- #
    errs = ev[ev["status"] == "error"]
    if len(errs):
        st.subheader(f"Malformed / rejected records ({len(errs)} — handled, not crashed)")
        st.dataframe(errs.tail(10).iloc[::-1][["event_time", "seq", "error"]],
                     width="stretch", hide_index=True)


def _view_replay(st, px, go, df, engine, dataset, model, alert_threshold):
    st.header("3 · Detection replay")
    st.markdown(f"<div class='replay-banner'>{REPLAY_BANNER}</div>",
                unsafe_allow_html=True)
    st.write(f"Source: **{DATASET_DISPLAY.get(dataset, dataset)}** held-out test "
             f"matrix `data/processed/{dataset}/X_test.npy` "
             f"({engine.total:,} records) · model **{PRETTY_MODEL.get(model, model)}** "
             f"· replayed **{engine.cursor:,}** so far.")
    if df.empty:
        st.info("Press **▶ Start** or **⏭ Step once** in the sidebar to begin "
                "the replay.")
        return

    # ---- (3) normal vs malicious ---- #
    n_mal = int(df["is_intrusion"].sum())
    n_norm = int(len(df) - n_mal)
    acc = float(df["correct"].mean())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("records replayed", f"{len(df):,}")
    c2.metric("predicted normal", f"{n_norm:,}")
    c3.metric("predicted malicious", f"{n_mal:,}")
    c4.metric("replay agreement w/ labels", f"{acc*100:.1f}%")

    st.subheader("Predicted normal vs malicious over replay time")
    tmp = df.copy()
    tmp["bucket"] = (tmp["row_index"] // 50) * 50
    agg = (tmp.groupby(["bucket", "predicted_class"]).size()
           .reset_index(name="count"))
    st.plotly_chart(
        px.area(agg, x="bucket", y="count", color="predicted_class",
                labels={"bucket": "replay record index"},
                title="rolling counts (REPLAY — synthetic order)"),
        width="stretch")

    # ---- (4) attack categories ---- #
    st.subheader("Attack categories in the replayed records (from ground-truth labels)")
    mal = df[df["true_binary"] != "normal"]
    if len(mal):
        cat = (mal["true_category"].value_counts().reset_index())
        cat.columns = ["category", "count"]
        p1, p2 = st.columns([2, 1])
        p1.plotly_chart(px.bar(cat, x="category", y="count", text="count",
                               title="true attack categories among replayed "
                                     "malicious records"),
                        width="stretch")
        p2.dataframe(cat, width="stretch", hide_index=True)
        st.caption("The served models are **binary** (normal vs attack); the "
                   "category is the record's true label, shown for context.")
    else:
        st.write("No malicious records replayed yet.")

    # ---- (5) recent detections ---- #
    st.subheader("Recent detections (most recent 25 — REPLAY)")
    recent = df.tail(25).iloc[::-1][[
        "replay_time", "row_index", "predicted_class", "confidence",
        "threat_level", "true_binary", "true_category", "correct"]]
    st.dataframe(recent, width="stretch", hide_index=True)

    # ---- (10) alerts ---- #
    st.subheader("Alerts raised during this REPLAY session")
    alerts = alerts_from(df.to_dict("records"), alert_threshold)
    a1, a2, a3 = st.columns(3)
    a1.metric("alerts", f"{len(alerts):,}")
    hi = sum(1 for a in alerts if a["threat_level"] == "high")
    a2.metric("high severity", f"{hi:,}")
    a3.metric("threshold", f"≥ {alert_threshold:.2f} confidence")
    if alerts:
        adf = pd.DataFrame(alerts).tail(20).iloc[::-1][[
            "replay_time", "row_index", "predicted_class", "confidence",
            "threat_level", "true_category"]]
        st.dataframe(adf, width="stretch", hide_index=True)
        st.caption("An alert = a replayed record the model predicted as *attack* "
                   "with confidence ≥ threshold (or an uncalibrated SVM attack "
                   "prediction). REPLAY-derived — not a live alert.")
    else:
        st.write("No alerts at this threshold yet.")


def _view_performance(st, go, dataset, engine):
    st.header("4 · Model performance & confusion matrix")
    st.caption("Offline evaluation on the held-out test set — Phases 3-4. "
               "Not replay, not live.")
    metrics = load_metrics(dataset)
    if not metrics:
        st.warning(f"no results/{dataset}/metrics.json — run "
                   f"`python scripts/train_eval_datasets.py --dataset {dataset}`")
        return
    st.write(f"**{DATASET_DISPLAY.get(dataset, dataset)}** · task "
             f"{metrics.get('task')} · seed {metrics.get('seed')} · generated "
             f"{metrics.get('generated_utc')}")

    cmp = model_comparison_frame(metrics)
    show = cmp[["model", "test_n", "accuracy", "precision_attack",
               "recall_attack", "f1_attack", "macro_f1", "roc_auc",
               "per_sample_ms", "throughput_per_s", "train_seconds",
               "val_macro_f1"]].copy()
    st.subheader("Metrics — every trained model")
    st.dataframe(show.style.format({
        "accuracy": "{:.4f}", "precision_attack": "{:.4f}",
        "recall_attack": "{:.4f}", "f1_attack": "{:.4f}", "macro_f1": "{:.4f}",
        "roc_auc": "{:.4f}", "per_sample_ms": "{:.4f}",
        "throughput_per_s": "{:,.0f}", "train_seconds": "{:.1f}",
        "val_macro_f1": "{:.4f}", "test_n": "{:,}"}, na_rep="—"),
        width="stretch", hide_index=True)

    st.subheader("Confusion matrix")
    model_names = list(metrics["models"])
    pick = st.selectbox("model", model_names)
    cm = metrics["models"][pick]["metrics"]["confusion_matrix"]
    mat = np.array(cm["matrix"])
    labels = cm["labels"]
    fig = go.Figure(data=go.Heatmap(
        z=mat, x=[f"pred {l}" for l in labels], y=[f"true {l}" for l in labels],
        text=mat, texttemplate="%{text:,}", colorscale="Blues", showscale=True))
    fig.update_layout(title=f"{pick} — {DATASET_DISPLAY.get(dataset, dataset)} "
                            f"test set ({cm['layout']})",
                      height=420)
    st.plotly_chart(fig, width="stretch")
    pc = metrics["models"][pick]["metrics"]["per_class"]
    st.dataframe(pd.DataFrame(pc).T.reset_index().rename(
        columns={"index": "class"}), width="stretch", hide_index=True)

    st.subheader("9 · Inference latency")
    lat_rows = []
    for name, blob in metrics["models"].items():
        lat = blob.get("latency", {})
        lat_rows.append({
            "model": name,
            "per-sample ms (offline eval)": lat.get("per_sample_ms_median"),
            "throughput /s (offline eval)": lat.get("throughput_samples_per_s"),
            "eval sample n": lat.get("n_samples"),
        })
    ldf = pd.DataFrame(lat_rows)
    st.dataframe(ldf.style.format({
        "per-sample ms (offline eval)": "{:.5f}",
        "throughput /s (offline eval)": "{:,.0f}",
        "eval sample n": "{:,}"}, na_rep="—"),
        width="stretch", hide_index=True)
    if engine is not None and getattr(engine, "per_sample_ms_measured", None) is not None:
        st.metric(f"this replay session — measured per-sample latency "
                  f"({PRETTY_MODEL.get(engine.model, engine.model)}, "
                  f"{DATASET_DISPLAY.get(engine.dataset, engine.dataset)})",
                  f"{engine.per_sample_ms_measured:.4f} ms")
        st.caption("Measured live while replaying test records through the "
                   "loaded model (batch predict, wall clock / n).")
    else:
        st.caption("Switch the sidebar to a *Fast* replay and step it to see a "
                   "live per-sample latency here; the *Full pipeline* replay "
                   "reports its own latency on view 3.")


def _view_comparison(st, px, go, dataset):
    st.header("5 · Model comparison")
    metrics = load_metrics(dataset)
    if metrics:
        cmp = model_comparison_frame(metrics)
        melt = cmp.melt(id_vars="model",
                        value_vars=["accuracy", "precision_attack",
                                    "recall_attack", "macro_f1", "roc_auc"],
                        var_name="metric", value_name="value")
        st.plotly_chart(
            px.bar(melt, x="model", y="value", color="metric", barmode="group",
                   title=f"{DATASET_DISPLAY.get(dataset, dataset)} — held-out "
                         f"test set (offline)"),
            width="stretch")
        st.dataframe(cmp[["model", "macro_f1", "roc_auc", "per_sample_ms",
                          "train_seconds"]].style.format({
            "macro_f1": "{:.4f}", "roc_auc": "{:.4f}", "per_sample_ms": "{:.4f}",
            "train_seconds": "{:.1f}"}, na_rep="—"),
            width="stretch", hide_index=True)

    st.subheader("Within-dataset vs cross-dataset generalisation (Phase 4)")
    cd = load_cross_dataset()
    if not cd:
        st.info("results/cross_dataset/metrics.json not found — run "
                "`python scripts/cross_dataset_experiment.py`")
        return
    rows = []
    for name, b in cd["models"].items():
        rows.append({
            "model": name,
            "within CIC-IDS2017": b["within_cicids2017"]["macro_f1"],
            "within UNSW-NB15": b["within_unsw_nb15"]["macro_f1"],
            "cross CIC→UNSW": b["cross_cicids2017_to_unsw_nb15"]["macro_f1"],
            "cross UNSW→CIC": b["cross_unsw_nb15_to_cicids2017"]["macro_f1"],
        })
    cdf = pd.DataFrame(rows)
    st.dataframe(cdf.style.format({c: "{:.4f}" for c in cdf.columns
                                   if c != "model"}),
                 width="stretch", hide_index=True)
    melt = cdf.melt(id_vars="model", var_name="evaluation", value_name="macro_f1")
    st.plotly_chart(
        px.bar(melt, x="model", y="macro_f1", color="evaluation",
               barmode="group",
               title="macro-F1 collapses across datasets (12-feature common "
                     "schema) — see results/cross_dataset/generalization.md"),
        width="stretch")
    st.caption("Common feature space: " + ", ".join(cd.get("common_features", [])))


# `streamlit run src/monitoring_dashboard.py` executes this file as __main__
if __name__ == "__main__":
    _run_streamlit()
