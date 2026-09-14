#!/usr/bin/env python
"""
Phase 6 — verify the monitoring dashboard.

Part A — every view's real-data path (pure functions + ReplayEngine).
Part B — every view renders without error, via streamlit's own AppTest harness
         (drives the sidebar radio through all 5 views + steps the replay).
Part C — the app actually serves: `streamlit run --server.headless` +
         `GET /_stcore/health` -> "ok" and `GET /` -> 200.

    python scripts/test_dashboard.py

Exit 0 iff every check passes.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

APP = REPO_ROOT / "src" / "monitoring_dashboard.py"
ROWS: list = []


def check(name: str, ok: bool, info: str = "") -> None:
    ROWS.append((name, bool(ok)))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  — {info}" if info else ""))


# --------------------------------------------------------------------------- #
def part_a() -> None:
    print("\n[A] real-data paths")
    from src import monitoring_dashboard as D

    disc = D.discover()
    check("discover() lists all datasets",
          set(disc["datasets"]) == set(D.DATASETS))
    check("environment captured", "python" in disc["environment"])

    ready = [d for d, e in disc["datasets"].items() if e["ready_for_replay"]]
    check("at least one dataset replay-ready", bool(ready), f"{ready}")

    for ds in ready:
        info = D.load_dataset_info(ds)
        check(f"[{ds}] load_dataset_info",
              info is not None and "splits" in info and info["features"]["selected"] > 0)
        met = D.load_metrics(ds)
        cmp = D.model_comparison_frame(met) if met else None
        check(f"[{ds}] load_metrics + comparison frame",
              met is not None and cmp is not None and len(cmp) >= 1
              and cmp["macro_f1"].notna().any())

        eng = D.ReplayEngine(ds, "random_forest")
        check(f"[{ds}] ReplayEngine loads", eng.load_error is None
              and eng.total > 0, f"{eng.total:,} records")
        b1 = eng.step(25)
        need = {"row_index", "replay_time", "predicted_class", "confidence",
                "true_binary", "true_category", "threat_level", "correct",
                "is_intrusion"}
        check(f"[{ds}] replay step yields labelled detections",
              len(b1) == 25 and need <= set(b1[0]))
        check(f"[{ds}] replay latency measured",
              eng.per_sample_ms_measured is not None
              and eng.per_sample_ms_measured > 0)
        # determinism
        eng2 = D.ReplayEngine(ds, "random_forest")
        b2 = eng2.step(25)
        check(f"[{ds}] replay is deterministic",
              [d["predicted_class"] for d in b1] == [d["predicted_class"] for d in b2])
        al = D.alerts_from(eng.step(300), 0.9)
        check(f"[{ds}] alerts_from returns intrusion subset",
              all(a["is_intrusion"] for a in al))

    check("load_cross_dataset()", D.load_cross_dataset() is not None)


# --------------------------------------------------------------------------- #
def part_b() -> None:
    print("\n[B] every view renders (streamlit AppTest)")
    try:
        from streamlit.testing.v1 import AppTest
    except Exception as e:  # pragma: no cover
        check("streamlit AppTest importable", False, str(e))
        return

    at = AppTest.from_file(str(APP), default_timeout=90)
    at.run()
    check("initial run has no exception", not at.exception,
          str(at.exception) if at.exception else "")

    radio = at.sidebar.radio[0]
    for opt in radio.options:
        at.sidebar.radio[0].set_value(opt).run()
        check(f"view renders: {opt.split(' · ')[0]}", not at.exception,
              str(at.exception) if at.exception else opt)

    # drive the replay on the replay view
    at.sidebar.radio[0].set_value(
        [o for o in radio.options if o.startswith("3")][0]).run()
    step_btns = [b for b in at.sidebar.button if "Step once" in b.label]
    if step_btns:
        step_btns[0].click().run()
        step_btns[0].click().run()
        check("replay 'Step once' advances without error", not at.exception,
              str(at.exception) if at.exception else "")
        # a dataframe / metric should now exist on the page
        check("replay view shows detections after stepping",
              len(at.dataframe) > 0 or len(at.metric) > 0)
    else:
        check("replay 'Step once' button present", False)

    # switch dataset + model
    if len(at.sidebar.selectbox) >= 2:
        ds_sel = at.sidebar.selectbox[0]
        if len(ds_sel.options) > 1:
            at.sidebar.selectbox[0].set_value(ds_sel.options[1]).run()
            check("switching dataset re-renders cleanly", not at.exception,
                  str(at.exception) if at.exception else "")


# --------------------------------------------------------------------------- #
def _free_port() -> int:
    s = socket.socket(); s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]; s.close(); return p


def part_c() -> None:
    print("\n[C] app serves (streamlit run --server.headless)")
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(APP),
         "--server.headless", "true", "--server.address", "127.0.0.1",
         "--server.port", str(port), "--browser.gatherUsageStats", "false",
         "--server.fileWatcherType", "none"],
        cwd=str(REPO_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True)
    base = f"http://127.0.0.1:{port}"
    ok_health = ok_root = False
    try:
        for _ in range(120):
            if proc.poll() is not None:
                break
            try:
                h = requests.get(base + "/_stcore/health", timeout=1)
                if h.status_code == 200 and h.text.strip().lower() == "ok":
                    ok_health = True
                    ok_root = requests.get(base, timeout=3).status_code == 200
                    break
            except requests.RequestException:
                pass
            time.sleep(0.5)
        check("GET /_stcore/health -> ok", ok_health)
        check("GET / -> 200", ok_root)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def main() -> int:
    part_a()
    part_b()
    part_c()
    passed = sum(1 for _, ok in ROWS if ok)
    print(f"\n{passed}/{len(ROWS)} checks passed")
    return 0 if passed == len(ROWS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
