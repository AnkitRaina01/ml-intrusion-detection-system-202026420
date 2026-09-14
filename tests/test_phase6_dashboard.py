"""
Fast pytest checks for the Phase-6 monitoring dashboard.

The full verification (renders every view via streamlit AppTest + boots the
server) is `python scripts/test_dashboard.py`; this is the quick subset.

    python -m pytest tests/test_phase6_dashboard.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import monitoring_dashboard as D  # noqa: E402

_READY = (REPO_ROOT / "data" / "processed" / "nsl_kdd" / "X_test.npy").exists() \
    and (REPO_ROOT / "models" / "nsl_kdd" / "random_forest.joblib").exists()
pytestmark = pytest.mark.skipif(
    not _READY, reason="run scripts/prepare_datasets.py + train_eval_datasets.py first")


def test_discover_shape():
    d = D.discover()
    assert set(d["datasets"]) == set(D.DATASETS)
    assert "python" in d["environment"]
    assert isinstance(d["api_log"], dict)


def test_load_dataset_info_and_metrics():
    info = D.load_dataset_info("nsl_kdd")
    assert info["features"]["selected"] == 20
    assert set(info["splits"]) == {"train", "validation", "test"}
    assert "multiclass" in info["splits"]["test"]
    m = D.load_metrics("nsl_kdd")
    cmp = D.model_comparison_frame(m)
    assert {"Random Forest", "Logistic Regression"} <= set(cmp["model"])
    assert cmp["macro_f1"].between(0, 1).all()


def test_replay_engine_labels_and_determinism():
    e1 = D.ReplayEngine("nsl_kdd", "random_forest")
    assert e1.load_error is None and e1.total > 1000
    b1 = e1.step(50)
    assert len(b1) == 50
    for d in b1:
        assert d["predicted_class"] in ("normal", "attack")
        assert d["true_binary"] in ("normal", "attack")
        assert d["is_intrusion"] == (d["predicted_class"] != "normal")
        assert d["threat_level"] in ("none", "low", "medium", "high", "unknown")
    assert e1.per_sample_ms_measured and e1.per_sample_ms_measured > 0
    e2 = D.ReplayEngine("nsl_kdd", "random_forest")
    assert [d["predicted_class"] for d in e2.step(50)] == \
           [d["predicted_class"] for d in b1]


def test_replay_clock_is_synthetic_not_now():
    e = D.ReplayEngine("nsl_kdd", "random_forest")
    b = e.step(3)
    assert b[0]["replay_time"].year == 2020          # fixed anchor, obviously not live
    assert b[1]["replay_time"] > b[0]["replay_time"]


def test_alerts_are_intrusion_subset_above_threshold():
    e = D.ReplayEngine("nsl_kdd", "random_forest")
    dets = e.step(400)
    al = D.alerts_from(dets, 0.95)
    assert all(a["is_intrusion"] for a in al)
    assert all(a["confidence"] is None or a["confidence"] >= 0.95 for a in al)
    assert len(al) <= sum(1 for d in dets if d["is_intrusion"])


def test_cross_dataset_loads():
    cd = D.load_cross_dataset()
    assert cd is not None and len(cd["common_features"]) == 12
    assert {"Random Forest"} <= set(cd["models"])


def test_apptest_renders_every_view():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(REPO_ROOT / "src" / "monitoring_dashboard.py"),
                           default_timeout=120)
    at.run()
    assert not at.exception
    for opt in at.sidebar.radio[0].options:
        at.sidebar.radio[0].set_value(opt).run()
        assert not at.exception, f"view {opt!r} raised {at.exception}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
