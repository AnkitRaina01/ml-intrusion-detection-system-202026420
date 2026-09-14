#!/usr/bin/env python
"""
Phase 5 — local end-to-end test of the prediction API.

Starts ``uvicorn src.api:app`` on a free port in a subprocess, waits for
``GET /health``, then exercises every endpoint with real feature vectors pulled
from the raw datasets (one benign + one attack row per dataset), plus the error
paths (missing feature, bad JSON, unknown dataset/model, wrong method).

    python scripts/test_api.py                 # run the checks, print a table
    python scripts/test_api.py --write-docs    # ...and (re)generate docs/API.md
                                               #    + examples/predict_*.json

Exit code 0 iff every check passes.
"""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# --------------------------------------------------------------------------- #
# real sample rows straight from the raw files (no full loader run)
# --------------------------------------------------------------------------- #
def _expected_cols(dataset: str) -> list:
    meta = json.loads(
        (REPO_ROOT / "artifacts" / dataset / "metadata.json").read_text())
    return meta["feature_columns_expected"]


def sample_rows(dataset: str):
    """Return (benign_features, attack_features) as {col: value} dicts."""
    import pandas as pd

    cols = _expected_cols(dataset)
    if dataset == "nsl_kdd":
        from src.data_loaders import NSL_KDD_RAW_COLUMNS
        df = pd.read_csv(REPO_ROOT / "data/raw/nsl_kdd/KDDTest+.txt",
                         header=None, names=NSL_KDD_RAW_COLUMNS, nrows=4000)
        df["label"] = df["label"].str.strip().str.lower()
        benign = df[df["label"] == "normal"].iloc[0]
        attack = df[df["label"] != "normal"].iloc[0]
    elif dataset == "unsw_nb15":
        df = pd.read_csv(REPO_ROOT / "data/raw/unsw_nb15/UNSW_NB15_testing-set.csv",
                         nrows=6000)
        df.columns = [c.strip() for c in df.columns]
        benign = df[df["label"] == 0].iloc[0]
        attack = df[df["label"] == 1].iloc[0]
    elif dataset == "cicids2017":
        df = pd.read_csv(
            REPO_ROOT / "data/raw/cicids2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
            nrows=200_000, encoding="utf-8", encoding_errors="replace",
            low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        lab = df.columns[-1]
        benign = df[df[lab].astype(str).str.strip().str.upper() == "BENIGN"].iloc[0]
        attack = df[df[lab].astype(str).str.strip().str.upper() != "BENIGN"].iloc[0]
    else:
        raise ValueError(dataset)

    def pick(row):
        out = {}
        for c in cols:
            v = row[c]
            if hasattr(v, "item"):
                v = v.item()
            out[c] = v
        return out

    return pick(benign), pick(attack)


# --------------------------------------------------------------------------- #
# server lifecycle
# --------------------------------------------------------------------------- #
def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def start_server(port: int) -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.api:app",
         "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
        cwd=str(REPO_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True)
    base = f"http://127.0.0.1:{port}"
    for _ in range(120):
        if proc.poll() is not None:
            raise RuntimeError("server exited early:\n" + (proc.stdout.read() or ""))
        try:
            if requests.get(base + "/health", timeout=1).status_code in (200, 503):
                return proc
        except requests.RequestException:
            pass
        time.sleep(0.5)
    proc.terminate()
    raise RuntimeError("server did not become ready in 60s")


# --------------------------------------------------------------------------- #
# checks
# --------------------------------------------------------------------------- #
class Checker:
    def __init__(self, base: str):
        self.base = base
        self.rows = []

    def check(self, name: str, ok: bool, info: str = ""):
        self.rows.append((name, bool(ok), info))
        print(f"  {'PASS' if ok else 'FAIL'}  {name}"
              + (f"  — {info}" if info else ""))

    def summary(self) -> bool:
        passed = sum(1 for _, ok, _ in self.rows if ok)
        print(f"\n{passed}/{len(self.rows)} checks passed")
        return passed == len(self.rows)


def run_checks(base: str, captures: dict) -> bool:
    c = Checker(base)

    # health
    r = requests.get(base + "/health", timeout=10)
    j = r.json()
    c.check("GET /health -> 200", r.status_code == 200, j.get("status"))
    c.check("health default is nsl_kdd/random_forest",
            j.get("default") == {"dataset": "nsl_kdd", "model": "random_forest"})
    captures["health"] = j

    # root + models
    r = requests.get(base + "/", timeout=10)
    c.check("GET / -> 200 with datasets", r.status_code == 200
            and "datasets" in r.json())
    r = requests.get(base + "/models", timeout=10)
    jm = r.json()
    c.check("GET /models lists nsl_kdd + expected feature columns",
            r.status_code == 200
            and jm["datasets"]["nsl_kdd"]["n_feature_columns_expected"] == 41)
    captures["models"] = jm

    # predictions per dataset
    for ds in ("nsl_kdd", "cicids2017", "unsw_nb15"):
        try:
            benign, attack = sample_rows(ds)
        except FileNotFoundError as e:
            c.check(f"[{ds}] raw sample rows available", False, str(e))
            continue

        rb = requests.post(base + "/predict",
                           json={"dataset": ds, "features": benign}, timeout=30)
        jb = rb.json()
        c.check(f"[{ds}] POST /predict benign -> 200",
                rb.status_code == 200,
                f"{jb.get('predicted_class')} conf={jb.get('confidence')}")
        c.check(f"[{ds}] benign response has required fields",
                all(k in jb for k in ("predicted_class", "model", "timestamp",
                                      "detection"))
                and "request_id" in jb["request"]
                and jb["model"]["dataset"] == ds)

        ra = requests.post(base + "/predict",
                           json={"dataset": ds, "features": attack}, timeout=30)
        ja = ra.json()
        c.check(f"[{ds}] POST /predict attack -> 200",
                ra.status_code == 200,
                f"{ja.get('predicted_class')} conf={ja.get('confidence')}")
        c.check(f"[{ds}] attack row predicted as intrusion",
                ja.get("detection", {}).get("is_intrusion") is True
                and ja.get("predicted_class") == "attack")

        captures.setdefault("predict", {})[ds] = {
            "request_benign": {"dataset": ds, "features": benign},
            "response_benign": jb,
            "request_attack": {"dataset": ds, "features": attack},
            "response_attack": ja,
        }

    # nsl_kdd: list-form input + per-model behaviour
    benign, _ = sample_rows("nsl_kdd")
    ordered = [benign[k] for k in _expected_cols("nsl_kdd")]
    rl = requests.post(base + "/predict",
                       json={"dataset": "nsl_kdd", "features": ordered}, timeout=20)
    c.check("nsl_kdd ordered-array 'features' accepted", rl.status_code == 200)

    rlog = requests.post(base + "/predict",
                         json={"dataset": "nsl_kdd", "model": "logreg",
                               "features": benign}, timeout=20).json()
    c.check("logreg returns probabilities + confidence",
            rlog.get("probabilities") is not None
            and rlog.get("confidence") is not None
            and rlog["detection"]["score_source"] == "predict_proba")

    rsvm = requests.post(base + "/predict",
                         json={"dataset": "nsl_kdd", "model": "svm_rbf",
                               "features": benign}, timeout=20).json()
    c.check("svm_rbf: no probability, decision_function present",
            rsvm.get("probabilities") is None
            and rsvm["detection"]["score_source"] == "decision_function"
            and rsvm["detection"]["decision_function"] is not None)
    captures["predict_logreg"] = rlog
    captures["predict_svm"] = rsvm

    # error paths
    r = requests.post(base + "/predict",
                      json={"dataset": "nsl_kdd", "features": {"duration": 0}},
                      timeout=20)
    c.check("missing features -> 422 with 'missing' list",
            r.status_code == 422 and isinstance(r.json().get("missing"), list))
    captures["error_missing"] = r.json()

    r = requests.post(base + "/predict",
                      data="{not json", headers={"content-type": "application/json"},
                      timeout=20)
    c.check("malformed JSON body -> 400", r.status_code == 400)

    r = requests.post(base + "/predict",
                      json={"dataset": "does_not_exist", "features": {}}, timeout=20)
    c.check("unknown dataset -> 404", r.status_code == 404)

    r = requests.post(base + "/predict",
                      json={"dataset": "nsl_kdd", "model": "xgboost",
                            "features": {}}, timeout=20)
    c.check("unknown model -> 404", r.status_code == 404)

    r = requests.get(base + "/predict", timeout=10)
    c.check("GET /predict (wrong method) -> 405", r.status_code == 405)

    return c.summary()


# --------------------------------------------------------------------------- #
# docs
# --------------------------------------------------------------------------- #
def write_docs(captures: dict) -> None:
    docs = REPO_ROOT / "docs"
    ex = REPO_ROOT / "examples"
    docs.mkdir(exist_ok=True)
    ex.mkdir(exist_ok=True)

    def block(obj) -> str:
        return "```json\n" + json.dumps(obj, indent=2) + "\n```"

    p = captures["predict"]["nsl_kdd"]
    for ds, blob in captures["predict"].items():
        (ex / f"predict_{ds}_benign.json").write_text(
            json.dumps(blob["request_benign"], indent=2) + "\n")
        (ex / f"predict_{ds}_attack.json").write_text(
            json.dumps(blob["request_attack"], indent=2) + "\n")

    L = [
        "# Intrusion Detection Prediction API",
        "",
        "_Phase 5. Starlette + uvicorn, no auth, no database. Generated by "
        "`python scripts/test_api.py --write-docs` from real requests against a "
        "live server — every response below is an actual server reply._",
        "",
        "## Run",
        "",
        "```bash",
        ". .venv/bin/activate",
        "python scripts/serve_api.py                 # 127.0.0.1:8000",
        "# or: uvicorn src.api:app --port 8000",
        "```",
        "",
        "## Endpoints",
        "",
        "| method | path | purpose |",
        "|---|---|---|",
        "| GET | `/` | service info + per-dataset expected feature columns |",
        "| GET | `/health` | liveness/readiness; which `(dataset, model)` pairs are loaded |",
        "| GET | `/models` | available datasets, models, and the exact feature columns each expects |",
        "| POST | `/predict` | classify one network-flow feature vector |",
        "",
        "## `POST /predict`",
        "",
        "### Request body",
        "",
        "```jsonc",
        "{",
        '  "dataset": "nsl_kdd",         // optional, default "nsl_kdd" '
        "(also: cicids2017, unsw_nb15)",
        '  "model":   "random_forest",   // optional, default "random_forest" '
        "(also: logreg, svm_rbf)",
        '  "features": { ... }           // REQUIRED: the dataset\'s native flow '
        "columns,",
        "                                //   as an object {name: value} OR an "
        "ordered array",
        "}",
        "```",
        "",
        "The **same fitted preprocessor saved during training** "
        "(`artifacts/<dataset>/preprocessor.joblib`) is applied to the input, so "
        "the request must carry that dataset's native feature columns. Get the "
        "exact list from `GET /models`. Unknown extra keys are ignored; "
        "non-numeric values in numeric columns are imputed (and reported in "
        "`request.input_warnings`).",
        "",
        "### Response",
        "",
        "| field | meaning |",
        "|---|---|",
        "| `predicted_class` | `\"normal\"` or `\"attack\"` |",
        "| `confidence` | max class probability (`null` for `svm_rbf`, which is uncalibrated) |",
        "| `probabilities` | per-class probability map (`null` for `svm_rbf`) |",
        "| `model` | dataset, model name, file, task, class list |",
        "| `timestamp` | ISO-8601 UTC of the response |",
        "| `detection` | `is_intrusion`, `threat_level` (none/low/medium/high/unknown), `score_source`, `decision_function` |",
        "| `request` | `request_id`, feature counts, preprocessor file + artifact version, `input_warnings`, `inference_ms` |",
        "",
        "### Example — NSL-KDD, attack flow (curl)",
        "",
        "```bash",
        "curl -s -X POST http://127.0.0.1:8000/predict \\",
        "  -H 'content-type: application/json' \\",
        "  -d @examples/predict_nsl_kdd_attack.json | jq",
        "```",
        "",
        "Request:",
        "",
        block(p["request_attack"]),
        "",
        "Response:",
        "",
        block(p["response_attack"]),
        "",
        "### Example — NSL-KDD, benign flow",
        "",
        "Request:",
        "",
        block(p["request_benign"]),
        "",
        "Response:",
        "",
        block(p["response_benign"]),
        "",
        "### Example — `model: \"svm_rbf\"` (no calibrated probability)",
        "",
        block(captures["predict_svm"]),
        "",
        "### Example — CIC-IDS2017 flow (attack)",
        "",
        "`GET /models` -> `datasets.cicids2017.feature_columns_expected` lists all "
        f"{len(captures['predict']['cicids2017']['request_attack']['features'])} "
        "required columns.",
        "",
        "Request:",
        "",
        block(captures["predict"]["cicids2017"]["request_attack"]),
        "",
        "Response:",
        "",
        block(captures["predict"]["cicids2017"]["response_attack"]),
        "",
        "### Example — UNSW-NB15 flow (attack)",
        "",
        "Request:",
        "",
        block(captures["predict"]["unsw_nb15"]["request_attack"]),
        "",
        "Response:",
        "",
        block(captures["predict"]["unsw_nb15"]["response_attack"]),
        "",
        "> The models are the Phase-4 binary classifiers; on hard rows they can be "
        "wrong. The API returns the model's actual call plus its confidence — it "
        "does not second-guess it. (Within-dataset test macro-F1: NSL-KDD RF "
        "0.77, CIC-IDS2017 RF 0.998, UNSW-NB15 RF 0.85.)",
        "",
        "## Errors",
        "",
        "Uniform envelope: `{\"error\", \"status\", \"request_id\", \"timestamp\", ...}`.",
        "",
        "| status | when |",
        "|---|---|",
        "| 400 | body is not valid JSON / not an object |",
        "| 404 | unknown `dataset` or `model` |",
        "| 405 | wrong HTTP method |",
        "| 422 | `features` missing, wrong length, or a required feature column absent |",
        "| 500 | unexpected server error |",
        "| 503 | `/health` only — default model failed to load |",
        "",
        "Missing-feature example (`422`):",
        "",
        block(captures["error_missing"]),
        "",
        "## `GET /health`",
        "",
        block(captures["health"]),
        "",
        "## `GET /models` (truncated)",
        "",
        block(_truncate_models(captures["models"])),
        "",
        "## Logging",
        "",
        "Every request is logged (file `logs/api.log`, rotating 2 MB × 3, plus "
        "stdout) as: `rid=<id> <method> <path> -> <status> <ms> client=<ip>`, "
        "with model-load and error lines. `request_id` is echoed in the "
        "`x-request-id` response header and in the JSON body.",
        "",
        "## Not included (by design — Phase 5 scope)",
        "",
        "- No authentication, no API keys, no database (per the brief).",
        "- The **LSTM is not served**: it needs a window of consecutive records, "
        "not a single flow (see DEVELOPMENT_NOTES.md §10.3 / §11.4).",
        "- Multiclass output: the trained models are binary "
        "(normal vs attack); `attack_category` is not predicted.",
        "",
    ]
    (docs / "API.md").write_text("\n".join(L) + "\n")
    print(f"\nwrote {docs/'API.md'} and {len(list(ex.glob('predict_*.json')))} "
          f"example payloads under {ex}/")


def _truncate_models(jm: dict) -> dict:
    out = {"time": jm.get("time"), "default": jm.get("default"), "datasets": {}}
    for ds, e in jm["datasets"].items():
        out["datasets"][ds] = {
            "models": e.get("models"),
            "n_feature_columns_expected": e.get("n_feature_columns_expected"),
            "feature_columns_expected": (e.get("feature_columns_expected", [])[:6]
                                         + ["..."]),
            "n_features_used_by_model": e.get("n_features_used_by_model"),
            "classes": e.get("classes"),
            "preprocessor_artifact_version": e.get("preprocessor_artifact_version"),
        }
    return out


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write-docs", action="store_true")
    ap.add_argument("--port", type=int, default=0)
    args = ap.parse_args(argv)

    port = args.port or _free_port()
    print(f"starting API on 127.0.0.1:{port} ...")
    proc = start_server(port)
    base = f"http://127.0.0.1:{port}"
    captures: dict = {}
    try:
        ok = run_checks(base, captures)
        if args.write_docs:
            write_docs(captures)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
