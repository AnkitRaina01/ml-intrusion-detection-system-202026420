#!/usr/bin/env python
"""
Thin launcher for the prediction API (Phase 5).

    python scripts/serve_api.py                     # 127.0.0.1:8000
    python scripts/serve_api.py --host 0.0.0.0 --port 9000
    python scripts/serve_api.py --reload            # dev auto-reload

Equivalent to:  uvicorn src.api:app --host <h> --port <p>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main(argv=None) -> int:
    import uvicorn

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--log-level", default="info")
    p.add_argument("--reload", action="store_true")
    args = p.parse_args(argv)

    uvicorn.run("src.api:app", host=args.host, port=args.port,
                log_level=args.log_level, reload=args.reload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
