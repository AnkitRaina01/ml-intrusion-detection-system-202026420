"""
Phase 5 intrusion detection prediction API (Starlette + uvicorn).

A small, self-contained HTTP service that turns a single network-flow feature
vector into an intrusion-detection prediction, using **exactly the fitted
preprocessing artefacts saved during training** (``artifacts/<dataset>/
preprocessor.joblib``) and the trained sklearn models (``models/<dataset>/
<model>.joblib``).

Endpoints
---------
    GET  /            service info + the shapes each dataset expects
    GET  /health      liveness/readiness + which (dataset, model) pairs are loaded
    GET  /models      per-dataset: available models, expected feature columns
    POST /predict      {dataset?, model?, features}  ->  prediction + confidence

Design choices (kept deliberately minimal — see DEVELOPMENT_NOTES.md §12):
* No auth, no database, no async workers. Single-process uvicorn.
* Models are **lazy-loaded** and cached (the UNSW-NB15 RandomForest is ~140 MB
  on disk); the default (nsl_kdd / random_forest) is warmed at startup.
* The LSTM is intentionally not served — it needs a window of consecutive
  records, not a single flow (see §10.3 / §11.4).

Run
---
    python -m src.api                       # dev server on 127.0.0.1:8000
    uvicorn src.api:app --port 8000         # equivalent
    python scripts/serve_api.py --help      # thin launcher with flags
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import platform
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from src.preprocessing import FittedPreprocessor  # noqa: F401  (needed for joblib load)

# --------------------------------------------------------------------------- #
# paths & constants
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_ROOT = REPO_ROOT / "artifacts"
MODELS_ROOT = REPO_ROOT / "models"
LOGS_DIR = REPO_ROOT / "logs"

DATASETS = ("nsl_kdd", "cicids2017", "unsw_nb15")
MODELS = ("random_forest", "logreg", "svm_rbf")   # sklearn models only
DEFAULT_DATASET = "nsl_kdd"
DEFAULT_MODEL = "random_forest"

API_VERSION = "phase5"
START_TIME = time.time()

# --------------------------------------------------------------------------- #
# logging  (file + console; every request logged with a request id)
# --------------------------------------------------------------------------- #
LOGS_DIR.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("nids_api")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _fmt = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s")
    _fh = logging.handlers.RotatingFileHandler(
        LOGS_DIR / "api.log", maxBytes=2_000_000, backupCount=3)
    _fh.setFormatter(_fmt)
    _ch = logging.StreamHandler()
    _ch.setFormatter(_fmt)
    logger.addHandler(_fh)
    logger.addHandler(_ch)
    logger.propagate = False


# --------------------------------------------------------------------------- #
# errors
# --------------------------------------------------------------------------- #
class ApiError(Exception):
    def __init__(self, status: int, detail: str, **extra: Any):
        super().__init__(detail)
        self.status = status
        self.detail = detail
        self.extra = extra


# --------------------------------------------------------------------------- #
# model / preprocessor registry (lazy + cached)
# --------------------------------------------------------------------------- #
class Registry:
    def __init__(self, artifacts_root: Path, models_root: Path):
        self.artifacts_root = artifacts_root
        self.models_root = models_root
        self._cache: Dict[Tuple[str, str], dict] = {}
        self._lock = threading.Lock()

    # -- discovery ---------------------------------------------------------- #
    def available(self) -> Dict[str, dict]:
        out: Dict[str, dict] = {}
        for ds in DATASETS:
            pre_path = self.artifacts_root / ds / "preprocessor.joblib"
            if not pre_path.exists():
                continue
            models = [m for m in MODELS
                      if (self.models_root / ds / f"{m}.joblib").exists()]
            if not models:
                continue
            entry: Dict[str, Any] = {"models": models}
            meta_path = self.artifacts_root / ds / "metadata.json"
            lbl_path = self.artifacts_root / ds / "label_maps.json"
            feat_path = self.artifacts_root / ds / "feature_names.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                entry["feature_columns_expected"] = meta.get("feature_columns_expected", [])
                entry["n_feature_columns_expected"] = len(
                    meta.get("feature_columns_expected", []))
                entry["preprocessor_artifact_version"] = meta.get("artifact_version")
                entry["target"] = meta.get("config", {}).get("target", "binary")
            if feat_path.exists():
                sel = json.loads(feat_path.read_text())
                entry["n_features_used_by_model"] = len(sel)
                entry["selected_features"] = sel
            if lbl_path.exists():
                lbl = json.loads(lbl_path.read_text())
                entry["classes"] = [lbl["binary"][str(i)] for i in (0, 1)] \
                    if "binary" in lbl else ["normal", "attack"]
            out[ds] = entry
        return out

    # -- load one (dataset, model) --------------------------------------- #
    def get(self, dataset: str, model: str) -> dict:
        key = (dataset, model)
        if key in self._cache:
            return self._cache[key]
        with self._lock:
            if key in self._cache:            # double-checked
                return self._cache[key]
            pre_path = self.artifacts_root / dataset / "preprocessor.joblib"
            mdl_path = self.models_root / dataset / f"{model}.joblib"
            if not pre_path.exists():
                raise ApiError(404, f"unknown dataset '{dataset}'",
                               available=list(self.available()))
            if not mdl_path.exists():
                raise ApiError(404, f"model '{model}' not trained for dataset "
                                    f"'{dataset}'",
                               available_models=self.available()
                               .get(dataset, {}).get("models", []))
            t0 = time.perf_counter()
            pre: FittedPreprocessor = joblib.load(pre_path)
            est = joblib.load(mdl_path)
            lbl = json.loads(
                (self.artifacts_root / dataset / "label_maps.json").read_text())
            classes = [lbl["binary"]["0"], lbl["binary"]["1"]]
            has_proba = hasattr(est, "predict_proba")
            has_decfn = hasattr(est, "decision_function")
            blob = {
                "dataset": dataset, "model": model,
                "preprocessor": pre, "estimator": est,
                "classes": classes, "label_maps": lbl,
                "has_proba": has_proba, "has_decfn": has_decfn,
                "model_file": str(mdl_path.relative_to(REPO_ROOT)),
                "preprocessor_file": str(pre_path.relative_to(REPO_ROOT)),
                "n_features_used": len(pre.feature_names),
                "n_feature_columns": len(pre.feature_columns),
                "artifact_version": getattr(pre, "artifact_version", None),
                "target": getattr(pre.config, "target", "binary"),
            }
            self._cache[key] = blob
            logger.info("loaded dataset=%s model=%s in %.0f ms (proba=%s)",
                        dataset, model, (time.perf_counter() - t0) * 1e3,
                        has_proba)
            return blob

    def loaded_keys(self) -> List[Dict[str, str]]:
        return [{"dataset": d, "model": m} for (d, m) in self._cache]


REGISTRY = Registry(ARTIFACTS_ROOT, MODELS_ROOT)


# --------------------------------------------------------------------------- #
# request parsing / validation
# --------------------------------------------------------------------------- #
def _coerce_frame(features: Any, pre: FittedPreprocessor
                  ) -> Tuple[pd.DataFrame, List[str]]:
    """Build a 1-row DataFrame in the preprocessor's native column order from a
    dict {name: value} or an ordered list. Returns (frame, warnings)."""
    cols = list(pre.feature_columns)
    warnings: List[str] = []

    if isinstance(features, list):
        if len(features) != len(cols):
            raise ApiError(
                422, f"'features' list has {len(features)} values but this "
                     f"dataset expects {len(cols)} feature columns",
                expected_order=cols)
        row = dict(zip(cols, features))
    elif isinstance(features, dict):
        missing = [c for c in cols if c not in features]
        if missing:
            raise ApiError(422, f"missing {len(missing)} required feature "
                                f"column(s)", missing=missing,
                           expected=cols)
        extra = [k for k in features if k not in cols]
        if extra:
            warnings.append(f"ignored {len(extra)} unrecognised field(s): "
                            f"{extra[:10]}")
        row = {c: features[c] for c in cols}
    else:
        raise ApiError(422, "'features' must be a JSON object {name: value} or "
                            "an ordered array")

    df = pd.DataFrame([row], columns=cols)

    # numeric columns -> numbers (strings/None -> NaN, imputed by the pipeline)
    for c in pre.numeric_columns:
        before = df.at[0, c]
        val = pd.to_numeric(pd.Series([before]), errors="coerce").iloc[0]
        if pd.isna(val) and before is not None and str(before).strip() != "":
            warnings.append(f"non-numeric value for numeric feature '{c}' "
                            f"({before!r}) -> imputed")
        df[c] = np.float64(val) if not pd.isna(val) else np.nan
    # categorical columns -> string (unknown categories are ignored by the OHE)
    for c in pre.categorical_columns:
        df[c] = str(df.at[0, c])
    return df, warnings


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


def run_prediction(dataset: str, model: str, features: Any) -> dict:
    blob = REGISTRY.get(dataset, model)
    pre: FittedPreprocessor = blob["preprocessor"]
    est = blob["estimator"]
    classes: List[str] = blob["classes"]

    df, warnings = _coerce_frame(features, pre)

    t0 = time.perf_counter()
    X = pre.transform(df)                       # SAME fitted artefact as training
    if not np.isfinite(X).all():
        bad = [pre.feature_names[i] for i in np.where(~np.isfinite(X).all(axis=0))[0]]
        raise ApiError(
            422, "input produced non-finite values after preprocessing "
                 "(an out-of-range numeric feature?)",
            non_finite_features=bad)
    y_idx = int(est.predict(X)[0])

    probabilities: Optional[Dict[str, float]] = None
    confidence: Optional[float] = None
    decision_function: Optional[float] = None
    score_source = "predict"

    if blob["has_proba"]:
        p = est.predict_proba(X)[0]
        probabilities = {classes[i]: round(float(p[i]), 6)
                         for i in range(len(classes))}
        confidence = round(float(np.max(p)), 6)
        score_source = "predict_proba"
    elif blob["has_decfn"]:
        d = float(np.ravel(est.decision_function(X))[0])
        decision_function = round(d, 6)
        score_source = "decision_function"
    infer_ms = (time.perf_counter() - t0) * 1e3

    pred_class = classes[y_idx] if 0 <= y_idx < len(classes) else str(y_idx)
    return {
        "predicted_class": pred_class,
        "predicted_index": y_idx,
        "confidence": confidence,
        "probabilities": probabilities,
        "model": {
            "dataset": dataset,
            "name": model,
            "file": blob["model_file"],
            "task": blob["target"],
            "classes": classes,
        },
        "detection": {
            "is_intrusion": pred_class != "normal",
            "threat_level": _threat_level(pred_class, confidence),
            "score_source": score_source,
            "decision_function": decision_function,
        },
        "request": {
            "features_received": (len(features) if hasattr(features, "__len__")
                                  else None),
            "feature_columns_expected": blob["n_feature_columns"],
            "features_used_by_model": blob["n_features_used"],
            "preprocessor_file": blob["preprocessor_file"],
            "preprocessor_artifact_version": blob["artifact_version"],
            "input_warnings": warnings,
            "inference_ms": round(infer_ms, 3),
        },
    }


# --------------------------------------------------------------------------- #
# handlers
# --------------------------------------------------------------------------- #
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z")


async def homepage(request: Request) -> JSONResponse:
    return JSONResponse({
        "service": "nids-ml prediction api",
        "version": API_VERSION,
        "time": _now_iso(),
        "endpoints": {
            "GET /health": "liveness / readiness",
            "GET /models": "available datasets + models + expected feature columns",
            "POST /predict": "{dataset?, model?, features: {name: value} | [ordered]}",
        },
        "default": {"dataset": DEFAULT_DATASET, "model": DEFAULT_MODEL},
        "datasets": REGISTRY.available(),
        "note": "features must be the dataset's native flow columns; the same "
                "fitted preprocessor from training is applied.",
    })


async def health(request: Request) -> JSONResponse:
    status = "ok"
    detail = None
    try:
        REGISTRY.get(DEFAULT_DATASET, DEFAULT_MODEL)
    except Exception as exc:  # pragma: no cover - defensive
        status = "degraded"
        detail = str(exc)
    body = {
        "status": status,
        "service": "nids-ml prediction api",
        "version": API_VERSION,
        "time": _now_iso(),
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "python": platform.python_version(),
        "default": {"dataset": DEFAULT_DATASET, "model": DEFAULT_MODEL},
        "loaded": REGISTRY.loaded_keys(),
        "datasets_available": sorted(REGISTRY.available()),
    }
    if detail:
        body["detail"] = detail
    return JSONResponse(body, status_code=200 if status == "ok" else 503)


async def list_models(request: Request) -> JSONResponse:
    return JSONResponse({
        "time": _now_iso(),
        "default": {"dataset": DEFAULT_DATASET, "model": DEFAULT_MODEL},
        "datasets": REGISTRY.available(),
    })


async def predict(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
    except Exception:
        raise ApiError(400, "request body must be valid JSON")
    if not isinstance(payload, dict):
        raise ApiError(400, "request body must be a JSON object")

    dataset = str(payload.get("dataset", DEFAULT_DATASET))
    model = str(payload.get("model", DEFAULT_MODEL))
    if "features" not in payload:
        raise ApiError(422, "request must include 'features'")
    if dataset not in DATASETS:
        raise ApiError(404, f"unknown dataset '{dataset}'",
                       known=list(DATASETS))
    if model not in MODELS:
        raise ApiError(404, f"unknown model '{model}'", known=list(MODELS))

    result = run_prediction(dataset, model, payload["features"])
    result["timestamp"] = _now_iso()
    result["request"]["request_id"] = request.state.request_id
    return JSONResponse(result)


# --------------------------------------------------------------------------- #
# middleware: request id + access log + uniform error envelope
# --------------------------------------------------------------------------- #
async def _log_and_guard(request: Request, call_next):
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
    request.state.request_id = rid
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
        dt = (time.perf_counter() - t0) * 1e3
        response.headers["x-request-id"] = rid
        logger.info("rid=%s %s %s -> %s %.1fms client=%s",
                    rid, request.method, request.url.path,
                    response.status_code, dt,
                    request.client.host if request.client else "-")
        return response
    except ApiError as exc:
        dt = (time.perf_counter() - t0) * 1e3
        logger.warning("rid=%s %s %s -> %s %s (%.1fms)",
                       rid, request.method, request.url.path,
                       exc.status, exc.detail, dt)
        return JSONResponse(
            {"error": exc.detail, "status": exc.status,
             "request_id": rid, "timestamp": _now_iso(), **exc.extra},
            status_code=exc.status, headers={"x-request-id": rid})
    except Exception as exc:  # pragma: no cover - last-resort
        dt = (time.perf_counter() - t0) * 1e3
        logger.exception("rid=%s %s %s -> 500 %s (%.1fms)",
                         rid, request.method, request.url.path, exc, dt)
        return JSONResponse(
            {"error": "internal error", "status": 500, "request_id": rid,
             "timestamp": _now_iso()},
            status_code=500, headers={"x-request-id": rid})


def create_app(warm: bool = True) -> Starlette:
    from contextlib import asynccontextmanager

    from starlette.middleware import Middleware
    from starlette.middleware.base import BaseHTTPMiddleware

    routes = [
        Route("/", homepage, methods=["GET"]),
        Route("/health", health, methods=["GET"]),
        Route("/models", list_models, methods=["GET"]),
        Route("/predict", predict, methods=["POST"]),
    ]
    middleware = [Middleware(BaseHTTPMiddleware, dispatch=_log_and_guard)]

    @asynccontextmanager
    async def lifespan(_app: Starlette):
        avail = REGISTRY.available()
        logger.info("api starting: datasets available = %s", sorted(avail))
        if warm and DEFAULT_DATASET in avail:
            try:
                REGISTRY.get(DEFAULT_DATASET, DEFAULT_MODEL)
                logger.info("warmed default %s/%s", DEFAULT_DATASET, DEFAULT_MODEL)
            except Exception as exc:  # pragma: no cover
                logger.warning("could not warm default model: %s", exc)
        yield
        logger.info("api shutting down")

    return Starlette(routes=routes, middleware=middleware, lifespan=lifespan)


app = create_app()


# --------------------------------------------------------------------------- #
# dev entry point
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    import argparse
    import uvicorn

    p = argparse.ArgumentParser(description="run the intrusion detection prediction API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--log-level", default="info")
    args = p.parse_args(argv)
    uvicorn.run("src.api:app", host=args.host, port=args.port,
                log_level=args.log_level)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
