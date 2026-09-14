"""
Fast in-process tests for the Phase-5 prediction API (no server, no HTTP).

The full HTTP end-to-end test (subprocess uvicorn + real requests + docs) is
`python scripts/test_api.py`; these are the quick invariants for `pytest`.

    python -m pytest tests/test_phase5_api.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import api                       # noqa: E402
from src.data_loaders import NSL_KDD_FEATURES, NSL_KDD_CATEGORICAL  # noqa: E402

_ARTIFACTS_READY = (REPO_ROOT / "artifacts" / "nsl_kdd" / "preprocessor.joblib").exists()
pytestmark = pytest.mark.skipif(
    not _ARTIFACTS_READY,
    reason="run scripts/prepare_datasets.py + train_eval first")


def _nsl_features() -> dict:
    f = {c: 0.0 for c in NSL_KDD_FEATURES}
    f["protocol_type"] = "tcp"
    f["service"] = "http"
    f["flag"] = "SF"
    f["src_bytes"] = 215.0
    f["dst_bytes"] = 45076.0
    f["count"] = 1.0
    f["logged_in"] = 1.0
    assert set(NSL_KDD_CATEGORICAL) <= set(f)
    return f


def test_registry_discovers_datasets():
    avail = api.REGISTRY.available()
    assert "nsl_kdd" in avail
    assert "random_forest" in avail["nsl_kdd"]["models"]
    assert avail["nsl_kdd"]["n_feature_columns_expected"] == 41
    assert avail["nsl_kdd"]["classes"] == ["normal", "attack"]


def test_predict_schema_random_forest():
    r = api.run_prediction("nsl_kdd", "random_forest", _nsl_features())
    assert r["predicted_class"] in ("normal", "attack")
    assert 0.0 <= r["confidence"] <= 1.0
    assert set(r["probabilities"]) == {"normal", "attack"}
    assert r["model"]["dataset"] == "nsl_kdd" and r["model"]["name"] == "random_forest"
    assert r["detection"]["score_source"] == "predict_proba"
    assert r["detection"]["is_intrusion"] == (r["predicted_class"] != "normal")
    assert r["request"]["features_used_by_model"] == 20
    assert r["request"]["preprocessor_artifact_version"] == 2


def test_predict_accepts_ordered_list():
    ordered = [_nsl_features()[c] for c in NSL_KDD_FEATURES]
    r = api.run_prediction("nsl_kdd", "random_forest", ordered)
    assert r["predicted_class"] in ("normal", "attack")


def test_svm_has_no_probability_but_decision_function():
    r = api.run_prediction("nsl_kdd", "svm_rbf", _nsl_features())
    assert r["probabilities"] is None
    assert r["confidence"] is None
    assert r["detection"]["score_source"] == "decision_function"
    assert isinstance(r["detection"]["decision_function"], float)


def test_missing_feature_raises_422():
    with pytest.raises(api.ApiError) as ei:
        api.run_prediction("nsl_kdd", "random_forest", {"duration": 0})
    assert ei.value.status == 422
    assert len(ei.value.extra["missing"]) == 40


def test_wrong_length_list_raises_422():
    with pytest.raises(api.ApiError) as ei:
        api.run_prediction("nsl_kdd", "random_forest", [0.0, 1.0, 2.0])
    assert ei.value.status == 422


def test_unknown_dataset_and_model_raise_404():
    with pytest.raises(api.ApiError) as ei:
        api.run_prediction("nope", "random_forest", _nsl_features())
    assert ei.value.status == 404
    with pytest.raises(api.ApiError) as ei:
        api.run_prediction("nsl_kdd", "xgboost", _nsl_features())
    assert ei.value.status == 404


def test_non_numeric_value_is_coerced_and_warned():
    f = _nsl_features()
    f["src_bytes"] = "not-a-number"
    r = api.run_prediction("nsl_kdd", "random_forest", f)
    assert any("src_bytes" in w for w in r["request"]["input_warnings"])
    assert r["predicted_class"] in ("normal", "attack")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
