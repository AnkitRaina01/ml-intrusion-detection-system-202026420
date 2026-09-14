"""
Inference-time preprocessing.

Training (``scripts/prepare_nsl_kdd.py``) and inference **share the exact same
transformation** by loading the one fitted object that training saved:
``artifacts/<dataset>/preprocessor.joblib`` (a ``FittedPreprocessor``).

Typical use
-----------
    from src.inference import load_preprocessor, transform_raw
    pre = load_preprocessor("artifacts/nsl_kdd")
    X = transform_raw(raw_df, preprocessor=pre)   # -> (n, n_selected_features) float32
    proba = model.predict_proba(X)

Run this module directly for a train/inference parity check:
    python -m src.inference --artifacts-dir artifacts/nsl_kdd \
        --raw-dir data/raw/nsl_kdd --processed-dir data/processed/nsl_kdd
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.preprocessing import FittedPreprocessor                      # noqa: E402


def load_preprocessor(artifacts_dir: str | Path) -> FittedPreprocessor:
    path = Path(artifacts_dir) / "preprocessor.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run scripts/prepare_nsl_kdd.py first.")
    return FittedPreprocessor.load(path)


def load_label_maps(artifacts_dir: str | Path) -> dict:
    return json.loads((Path(artifacts_dir) / "label_maps.json").read_text())


def transform_raw(raw_df: pd.DataFrame,
                  artifacts_dir: Optional[str | Path] = None,
                  preprocessor: Optional[FittedPreprocessor] = None) -> np.ndarray:
    """
    Apply the fitted training transformation to a raw feature DataFrame.

    ``raw_df`` must contain the dataset's native feature columns (extra columns
    such as ``label`` / ``difficulty`` are ignored). Returns a float32 matrix
    whose columns are exactly ``preprocessor.feature_names`` in order.
    """
    if preprocessor is None:
        if artifacts_dir is None:
            raise ValueError("pass either artifacts_dir or preprocessor")
        preprocessor = load_preprocessor(artifacts_dir)
    return preprocessor.transform(raw_df)


# --------------------------------------------------------------------------- #
# parity check
# --------------------------------------------------------------------------- #
def _parity_check(artifacts_dir: str, raw_dir: str, processed_dir: str,
                  n: int = 500) -> int:
    from src.data_loaders import load_nsl_kdd

    pre = load_preprocessor(artifacts_dir)
    bundle = load_nsl_kdd(raw_dir)
    raw_test: pd.DataFrame = bundle["test"]           # type: ignore[assignment]

    X_saved = np.load(Path(processed_dir) / "X_test.npy")
    n = min(n, len(raw_test), len(X_saved))
    X_infer = transform_raw(raw_test.iloc[:n], preprocessor=pre)

    ok_shape = X_infer.shape == (n, len(pre.feature_names)) == \
        (n, X_saved.shape[1])
    ok_values = np.allclose(X_infer, X_saved[:n], rtol=0, atol=0, equal_nan=True)
    max_abs = float(np.abs(X_infer - X_saved[:n]).max())

    print("train/inference parity check")
    print(f"  rows compared         : {n}")
    print(f"  selected feature count : {len(pre.feature_names)}")
    print(f"  inference matrix shape : {X_infer.shape}")
    print(f"  saved matrix shape     : {X_saved.shape}")
    print(f"  shapes agree           : {ok_shape}")
    print(f"  max |Δ| vs saved X_test : {max_abs:.3e}")
    print(f"  bit-exact match         : {ok_values}")
    if ok_shape and ok_values:
        print("  RESULT: PASS")
        return 0
    print("  RESULT: FAIL")
    return 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="train/inference preprocessing parity check")
    p.add_argument("--artifacts-dir", default=str(REPO_ROOT / "artifacts" / "nsl_kdd"))
    p.add_argument("--raw-dir", default=str(REPO_ROOT / "data" / "raw" / "nsl_kdd"))
    p.add_argument("--processed-dir",
                   default=str(REPO_ROOT / "data" / "processed" / "nsl_kdd"))
    p.add_argument("-n", type=int, default=500)
    args = p.parse_args(argv)
    return _parity_check(args.artifacts_dir, args.raw_dir, args.processed_dir, args.n)


if __name__ == "__main__":
    raise SystemExit(main())
