#!/usr/bin/env python
"""
Launcher for the Phase-6 monitoring dashboard.

    python scripts/run_dashboard.py                       # opens a browser
    python scripts/run_dashboard.py --port 8502 --headless
    # equivalent to:  streamlit run src/monitoring_dashboard.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
APP = REPO_ROOT / "src" / "monitoring_dashboard.py"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8501)
    p.add_argument("--headless", action="store_true",
                   help="run without opening a browser (server only)")
    args = p.parse_args(argv)

    cmd = [sys.executable, "-m", "streamlit", "run", str(APP),
           "--server.address", args.host, "--server.port", str(args.port),
           "--browser.gatherUsageStats", "false"]
    if args.headless:
        cmd += ["--server.headless", "true"]
    print("+ " + " ".join(cmd))
    return subprocess.call(cmd, cwd=str(REPO_ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
