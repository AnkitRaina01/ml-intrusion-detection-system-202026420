#!/usr/bin/env python
"""
Phase 7 — headless Replay / Simulation detection run (end-to-end demonstration).

    raw flow record -> production preprocessing -> feature selection -> model
      -> prediction + probability -> detection event -> alert -> saved results

This is a **replay of a historical dataset**, not live network traffic.

    python scripts/run_replay.py --dataset nsl_kdd --model random_forest
    python scripts/run_replay.py --dataset unsw_nb15 --model random_forest \
        --limit 5000 --batch-size 32 --threshold 0.9
    python scripts/run_replay.py --dataset cicids2017 --model random_forest

Writes ``results/replay/<dataset>_<model>.json`` (measured stats + event samples).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.replay_detection import ReplaySession, available_replay   # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True,
                    choices=["nsl_kdd", "cicids2017", "unsw_nb15"])
    ap.add_argument("--model", default="random_forest",
                    choices=["random_forest", "logreg", "svm_rbf"])
    ap.add_argument("--threshold", type=float, default=0.90)
    ap.add_argument("--limit", type=int, default=None,
                    help="max records to replay (default: whole raw source)")
    ap.add_argument("--batch-size", type=int, default=32,
                    help="records processed per step (micro-batch)")
    ap.add_argument("--shuffle-seed", type=int, default=None)
    ap.add_argument("--latency-probe", type=int, default=200,
                    help="records for the true single-record latency measurement")
    ap.add_argument("--out", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    avail = available_replay()
    if args.dataset not in avail:
        print(f"dataset {args.dataset!r} not available for replay. "
              f"have: {list(avail)}", file=sys.stderr)
        return 2
    if not avail[args.dataset]["raw_source_present"]:
        print(f"raw source missing: {avail[args.dataset]['raw_source']}\n"
              f"  (for cicids2017 run: python scripts/make_replay_samples.py)",
              file=sys.stderr)
        return 2

    print("=" * 68)
    print("REPLAY / SIMULATION  —  NOT live network traffic")
    print("=" * 68)
    print(f"  dataset={args.dataset}  model={args.model}  threshold={args.threshold}")
    print(f"  raw source: {avail[args.dataset]['raw_source']}")

    sess = ReplaySession(args.dataset, args.model, threshold=args.threshold,
                         limit=args.limit, shuffle_seed=args.shuffle_seed)
    if sess.load_error:
        print(f"  LOAD ERROR: {sess.load_error}", file=sys.stderr)
        return 1
    print(f"  records to replay: {sess.total:,}")

    t0 = time.perf_counter()
    sess.start()
    last = 0
    while sess.state == "running":
        sess.step(args.batch_size)
        if not args.quiet and sess.n_processed - last >= 2000:
            last = sess.n_processed
            s = sess.stats()
            print(f"  … {s['samples_processed']:>7,}/{sess.total:,}  "
                  f"attack={s['attacks_detected']:>6,}  alerts={s['detections_alerts']:>6,}  "
                  f"errors={s['errors']}  "
                  f"lat_avg={s['latency_ms']['avg']}ms  thr={s['throughput_records_per_s']}/s")
    wall = time.perf_counter() - t0

    probe = sess.probe_single_record_latency(args.latency_probe)
    stats = sess.stats()

    # ---- console summary ---- #
    print("\n" + "-" * 68)
    print("RESULT (measured — no fabricated values)")
    print("-" * 68)
    print(f"  replay mode              : {stats['replay_mode']}   (is_live_capture={stats['is_live_capture']})")
    print(f"  samples processed        : {stats['samples_processed']:,}")
    print(f"  errors                   : {stats['errors']}")
    print(f"  predicted normal         : {stats['normal_samples']:,}")
    print(f"  predicted attack         : {stats['attacks_detected']:,}")
    print(f"  detections (alerts >= thr): {stats['detections_alerts']:,}  "
          f"(high severity: {stats['high_severity_alerts']:,})")
    print(f"  true normal / attack     : {stats['true_normal_in_stream']:,} / "
          f"{stats['true_attack_in_stream']:,}")
    print(f"  agreement with labels    : {stats['agreement_with_labels']}")
    lm = stats["latency_ms"]
    print(f"  per-record latency (batch={args.batch_size}) : "
          f"avg {lm['avg']} ms  min {lm['min']}  max {lm['max']}  p50 {lm['p50']}  p95 {lm['p95']}")
    print(f"  single-record latency (batch=1, n={probe['n']}) : "
          f"avg {probe.get('avg_ms')} ms  min {probe.get('min_ms')}  max {probe.get('max_ms')}  "
          f"p50 {probe.get('p50_ms')}  p95 {probe.get('p95_ms')}")
    print(f"  processing throughput    : {stats['throughput_records_per_s']} records/s "
          f"(processing wall {stats['processing_wall_seconds']} s; total wall {wall:.2f} s)")

    # ---- persist ---- #
    out = Path(args.out) if args.out else (
        REPO_ROOT / "results" / "replay" / f"{args.dataset}_{args.model}.json")
    out = out if out.is_absolute() else (Path.cwd() / out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ev = sess.events
    err_examples = [e.to_dict() for e in ev if e.status == "error"][:5]
    payload = {
        "phase": 7,
        "mode": "replay_simulation",
        "is_live_network_traffic": False,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "dataset": args.dataset, "model": args.model,
            "threshold": args.threshold, "batch_size": args.batch_size,
            "limit": args.limit, "shuffle_seed": args.shuffle_seed,
            "raw_source": avail[args.dataset]["raw_source"],
        },
        "pipeline": ["raw flow record", "production preprocessing "
                     "(artifacts/<ds>/preprocessor.joblib)", "feature selection "
                     "(same FittedPreprocessor)", "model (models/<ds>/<model>.joblib)",
                     "prediction + probability", "detection event", "alert",
                     "dashboard/CLI"],
        "stats": stats,
        "single_record_latency_probe": probe,
        "total_wall_seconds": round(wall, 3),
        "predicted_class_counts_full_run": {
            "normal": stats["normal_samples"], "attack": stats["attacks_detected"],
            "errors": stats["errors"]},
        "predicted_class_counts_last_events": dict(Counter(
            e.predicted_class for e in ev if e.status == "ok")),
        "true_category_counts_last_events": dict(Counter(
            e.true_category for e in ev if e.true_category)),
        "_note_last_events": (f"the *_last_events maps cover only the retained "
                              f"event buffer (last {sess.keep_events}); "
                              f"'stats' and 'predicted_class_counts_full_run' "
                              f"are the authoritative full-run totals"),
        "example_events_first": [e.to_dict() for e in ev[:15]],
        "example_events_last": [e.to_dict() for e in ev[-15:]],
        "example_alerts": sess.recent_alerts(10),
        "example_errors": err_examples,
    }
    out.write_text(json.dumps(payload, indent=2, default=str))
    try:
        shown = out.resolve().relative_to(REPO_ROOT)
    except ValueError:
        shown = out
    print(f"\n  saved: {shown}")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
