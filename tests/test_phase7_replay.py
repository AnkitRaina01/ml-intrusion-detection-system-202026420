"""
Phase 7 — fast tests for the Replay / Simulation detection pipeline
(`src/replay_detection.py`). Real NSL-KDD records, no server, < a few seconds.

Required cases: valid / normal / attack / malformed / missing features /
non-finite values / model unavailable / preprocessing failure — plus a mini
end-to-end replay.

    python -m pytest tests/test_phase7_replay.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.replay_detection import (RawFlowSource, DetectionEngine,          # noqa: E402
                                  ReplaySession, DetectionEvent)

_READY = (REPO_ROOT / "artifacts" / "nsl_kdd" / "preprocessor.joblib").exists() \
    and (REPO_ROOT / "models" / "nsl_kdd" / "random_forest.joblib").exists() \
    and (REPO_ROOT / "data" / "raw" / "nsl_kdd" / "KDDTest+.txt").exists()
pytestmark = pytest.mark.skipif(
    not _READY, reason="run scripts/prepare_datasets.py + train_eval + have raw NSL-KDD")


@pytest.fixture(scope="module")
def src():
    return RawFlowSource("nsl_kdd", limit=400)


@pytest.fixture(scope="module")
def engine():
    return DetectionEngine("nsl_kdd", "random_forest", threshold=0.90)


def _first_where(src, want):  # want in {"normal","attack"}
    for r, tb in zip(src._rows, src._true_binary):
        if tb == want:
            return dict(r)
    raise AssertionError(f"no {want} record in sample")


_EVENT_FIELDS = {"seq", "event_time", "dataset", "model", "status",
                 "predicted_class", "is_intrusion", "confidence", "probabilities",
                 "threat_level", "alert", "latency_ms", "true_binary",
                 "true_category", "correct", "error"}


# --------------------------------------------------------------------------- #
def test_raw_flow_source_reads_native_records_sequentially(src):
    assert len(src) == 400
    rec0 = src._rows[0]
    assert len(rec0) == 41 and "protocol_type" in rec0 and "src_bytes" in rec0
    # sequential = file order: seqs increasing, source keeps ground truth aside
    assert set(src._true_binary) <= {"normal", "attack"}
    seqs, recs, tb, tc = next(src.batches(10))
    assert seqs == list(range(10)) and len(recs) == 10


def test_valid_sample(src, engine):
    ev = engine.process_batch([0], [dict(src._rows[0])],
                              [src._true_binary[0]], [src._true_category[0]])[0]
    assert isinstance(ev, DetectionEvent)
    assert ev.status == "ok"
    assert ev.predicted_class in ("normal", "attack")
    assert _EVENT_FIELDS <= set(ev.to_dict())
    assert ev.event_time.startswith("2020-01-01T")          # synthetic clock
    assert ev.latency_ms is not None and ev.latency_ms >= 0
    assert ev.n_feature_columns_in == 41


def test_normal_sample(src, engine):
    ev = engine.process_batch([1], [_first_where(src, "normal")], ["normal"], ["normal"])[0]
    assert ev.status == "ok"
    # a genuinely-benign NSL-KDD record: model should call it normal here
    assert ev.predicted_class == "normal"
    assert ev.is_intrusion is False and ev.alert is False
    assert ev.threat_level == "none"


def test_attack_sample(src, engine):
    ev = engine.process_batch([2], [_first_where(src, "attack")], ["attack"], ["dos"])[0]
    assert ev.status == "ok"
    assert ev.predicted_class == "attack" and ev.is_intrusion is True
    # attack + high confidence -> alert at threshold 0.90
    assert ev.alert is True
    assert ev.threat_level in ("high", "medium", "low")


def test_malformed_sample(engine):
    for bad in ("not a dict", 12345, [1, 2, 3]):
        ev = engine.process_batch([0], [bad])[0]
        assert ev.status == "error"
        assert ev.predicted_class is None and ev.alert is False
        assert ev.error and len(ev.error) > 0


def test_missing_features(src, engine):
    partial = {k: src._rows[0][k] for k in list(src._rows[0])[:12]}
    ev = engine.process_batch([0], [partial])[0]
    assert ev.status == "error"
    assert "missing" in ev.error.lower()


def test_non_finite_values(src, engine):
    rec = dict(src._rows[0]); rec["src_bytes"] = 1e308      # overflows on scaling
    ev = engine.process_batch([0], [rec])[0]
    assert ev.status == "error"
    assert "non-finite" in ev.error.lower()


def test_model_unavailable():
    eng = DetectionEngine("nsl_kdd", "xgboost")             # not a trained model
    assert eng.ready is False
    assert eng.load_error and "xgboost" in eng.load_error
    ev = eng.process_batch([0], [{"duration": 0}])[0]
    assert ev.status == "error" and "unavailable" in ev.error.lower()


def test_preprocessing_failure(src, engine):
    good = dict(src._rows[0])

    class _BoomPre:                                          # transform blows up
        feature_columns = engine._blob["preprocessor"].feature_columns
        numeric_columns = engine._blob["preprocessor"].numeric_columns
        categorical_columns = engine._blob["preprocessor"].categorical_columns
        feature_names = engine._blob["preprocessor"].feature_names

        def transform(self, df):
            raise RuntimeError("simulated preprocessing failure")

    eng2 = DetectionEngine("nsl_kdd", "random_forest")
    eng2._blob = dict(eng2._blob)
    eng2._blob["preprocessor"] = _BoomPre()
    evs = eng2.process_batch([0, 1], [good, good], ["attack", "normal"], ["dos", "normal"])
    assert all(e.status == "error" for e in evs)
    assert all("failure" in e.error.lower() for e in evs)
    assert evs[0].latency_ms is not None                    # timing still recorded


def test_batch_of_mixed_valid_and_bad(src, engine):
    recs = [dict(src._rows[0]), "bad", dict(src._rows[1]),
            {k: src._rows[2][k] for k in list(src._rows[2])[:5]}]
    evs = engine.process_batch([0, 1, 2, 3], recs,
                               ["attack", "attack", "attack", "attack"],
                               ["dos", "dos", "dos", "dos"])
    assert len(evs) == 4
    st = [e.status for e in evs]
    assert st == ["ok", "error", "ok", "error"]
    # seq order preserved
    assert [e.seq for e in evs] == [0, 1, 2, 3]


# --------------------------------------------------------------------------- #
def test_replay_session_end_to_end():
    sess = ReplaySession("nsl_kdd", "random_forest", threshold=0.90, limit=120)
    assert sess.load_error is None and sess.total == 120
    assert sess.state == "idle"
    sess.start()
    assert sess.state == "running"
    sess.step(40)
    assert sess.n_processed == 40 and sess.state == "running"
    sess.pause();  assert sess.state == "paused"
    assert sess.step(40) == []                              # paused -> no advance
    sess.start(); sess.run_to_end(batch_size=40)
    assert sess.state == "finished" and sess.n_processed == 120

    s = sess.stats()
    assert s["replay_mode"] == "Replay / Simulation"
    assert s["is_live_capture"] is False
    assert s["samples_processed"] == 120
    assert s["attacks_detected"] + s["normal_samples"] + s["errors"] == 120
    assert s["latency_ms"]["avg"] > 0 and s["latency_ms"]["min"] <= s["latency_ms"]["max"]
    assert s["throughput_records_per_s"] > 0
    assert 0.0 <= s["agreement_with_labels"] <= 1.0
    # every retained event has a synthetic timestamp + prediction/proba fields
    for e in sess.recent_events(120):
        assert e["event_time"].startswith("2020-01-01T")
        assert e["status"] in ("ok", "error")

    # controls: reset clears everything
    sess.reset()
    assert sess.n_processed == 0 and sess.events == [] and sess.cursor == 0
    assert sess.state == "idle"


def test_threshold_control_reclassifies_alerts():
    sess = ReplaySession("nsl_kdd", "random_forest", threshold=0.99, limit=200)
    sess.run_to_end(batch_size=64)
    hi = sess.stats()["detections_alerts"]
    sess.set_threshold(0.50)
    lo = sess.stats()["detections_alerts"]
    assert lo >= hi                                          # lower threshold -> >= alerts


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
