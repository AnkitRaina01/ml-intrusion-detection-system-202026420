#!/usr/bin/env python
"""
Phase 4 — reproducible download of CIC-IDS2017 and UNSW-NB15.

Nothing is fetched automatically by the pipeline; run this once.

    python scripts/download_datasets.py                 # both
    python scripts/download_datasets.py --dataset unsw_nb15
    python scripts/download_datasets.py --dataset cicids2017

Sources (public mirrors, no login required):
  * CIC-IDS2017 : Hugging Face dataset  c01dsnap/CIC-IDS2017
                  (the 8 "MachineLearningCVE" daily flow CSVs; ~1.1 GB total)
  * UNSW-NB15   : GitHub  Nir-J/ML-Projects  (official partitioned CSVs)

After download the row counts are asserted against the canonical values so a
truncated / wrong file fails loudly.

NSL-KDD is downloaded by the Phase-2 instructions (DEVELOPMENT_NOTES.md §9.7).
"""

from __future__ import annotations

import argparse
import sys
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW = REPO_ROOT / "data" / "raw"

CICIDS_BASE = "https://huggingface.co/datasets/c01dsnap/CIC-IDS2017/resolve/main"
CICIDS_FILES = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
]
CICIDS_TOTAL_ROWS = 2_830_743          # canonical MachineLearningCVE row count

UNSW_BASE = ("https://raw.githubusercontent.com/Nir-J/ML-Projects/master/"
             "UNSW-Network_Packet_Classification")
UNSW_FILES = {
    "UNSW_NB15_training-set.csv": 175_341,
    "UNSW_NB15_testing-set.csv": 82_332,
}


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  exists  {dest.name}  ({dest.stat().st_size:,} B)")
        return
    print(f"  GET     {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as fh:  # noqa: S310
        while chunk := r.read(1 << 20):
            fh.write(chunk)
    tmp.rename(dest)
    print(f"  saved   {dest.name}  ({dest.stat().st_size:,} B)")


def _count_rows(path: Path) -> int:
    with open(path, "rb") as fh:
        return sum(buf.count(b"\n") for buf in iter(lambda: fh.read(1 << 20), b""))


def get_cicids2017() -> int:
    out = RAW / "cicids2017"
    print("CIC-IDS2017 ->", out)
    for f in CICIDS_FILES:
        _download(f"{CICIDS_BASE}/{urllib.parse.quote(f)}", out / f)
    total = sum(_count_rows(out / f) - 1 for f in CICIDS_FILES)   # minus header
    print(f"  total data rows: {total:,} (expected {CICIDS_TOTAL_ROWS:,})")
    if total != CICIDS_TOTAL_ROWS:
        print("  !! row count mismatch — a file may be truncated", file=sys.stderr)
        return 1
    return 0


def get_unsw_nb15() -> int:
    out = RAW / "unsw_nb15"
    print("UNSW-NB15 ->", out)
    rc = 0
    for f, expect in UNSW_FILES.items():
        _download(f"{UNSW_BASE}/{f}", out / f)
        n = _count_rows(out / f) - 1
        print(f"  {f}: {n:,} rows (expected {expect:,})")
        if n != expect:
            print(f"  !! {f} row count mismatch", file=sys.stderr)
            rc = 1
    return rc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["cicids2017", "unsw_nb15", "all"],
                    default="all")
    args = ap.parse_args(argv)
    rc = 0
    if args.dataset in ("all", "unsw_nb15"):
        rc |= get_unsw_nb15()
    if args.dataset in ("all", "cicids2017"):
        rc |= get_cicids2017()
    print("\ndone." if rc == 0 else "\ndone with warnings.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
