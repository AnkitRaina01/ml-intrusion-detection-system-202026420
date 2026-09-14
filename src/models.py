"""
Phase 3 model builders + validation-based selection.

Exactly the four required supervised models, nothing else:

    logreg          sklearn LogisticRegression
    random_forest   sklearn RandomForestClassifier
    svm_rbf         sklearn SVC(kernel="rbf")           (seeded stratified subsample)
    lstm            PyTorch LSTM over fixed-length windows of NSL-KDD records

Methodology (identical for every model)
--------------------------------------
* fit on TRAIN only;
* choose hyper-parameters by **macro-F1 on the VALIDATION set**;
* the chosen model is the one already fitted on TRAIN with the best grid point
  (no refit on train+val -> keeps the comparison clean and the LSTM, which needs
  val for early stopping, on equal footing);
* the TEST set is never seen here - final evaluation is a separate script.

Imbalance strategy: ``class_weight="balanced"`` for the sklearn models, an
equivalent ``pos_weight`` in the LSTM loss. (NSL-KDD binary is only mildly
imbalanced; SMOTE arrays from Phase 2 remain available as an alternative.)
"""

from __future__ import annotations

import itertools
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVC


# --------------------------------------------------------------------------- #
# reproducibility
# --------------------------------------------------------------------------- #
def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# generic grid expansion
# --------------------------------------------------------------------------- #
def _expand_grid(grid: Dict[str, Sequence[Any]]) -> List[Dict[str, Any]]:
    if not grid:
        return [{}]
    keys = list(grid)
    return [dict(zip(keys, combo)) for combo in itertools.product(*grid.values())]


@dataclass
class SelectionResult:
    model_key: str
    best_params: Dict[str, Any]
    best_val_macro_f1: float
    grid: List[Dict[str, Any]] = field(default_factory=list)
    estimator: Any = None
    train_seconds: float = 0.0
    notes: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# sklearn models (logreg, random_forest, svm_rbf)
# --------------------------------------------------------------------------- #
_SKLEARN_CLS = {
    "logreg": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "svm_rbf": SVC,
}


def _subsample(X, y, n, seed):
    if n is None or n >= len(X):
        return X, y, len(X)
    sss = StratifiedShuffleSplit(n_splits=1, train_size=int(n), random_state=seed)
    idx, _ = next(sss.split(X, y))
    return X[idx], y[idx], int(n)


def select_sklearn_model(model_key: str,
                         fixed: Dict[str, Any],
                         grid: Dict[str, Sequence[Any]],
                         X_train: np.ndarray, y_train: np.ndarray,
                         X_val: np.ndarray, y_val: np.ndarray,
                         seed: int,
                         train_subsample: Optional[int] = None) -> SelectionResult:
    cls = _SKLEARN_CLS[model_key]
    Xt, yt, used_n = _subsample(X_train, y_train, train_subsample, seed)

    records: List[Dict[str, Any]] = []
    best: Optional[Tuple[float, Dict[str, Any], Any]] = None
    t0 = time.perf_counter()
    for params in _expand_grid(grid):
        kwargs = dict(fixed, **params)
        if "random_state" not in kwargs and model_key in ("random_forest", "svm_rbf",
                                                          "logreg"):
            kwargs["random_state"] = seed
        est = cls(**kwargs)
        est.fit(Xt, yt)
        val_pred = est.predict(X_val)
        macro = float(f1_score(y_val, val_pred, average="macro", zero_division=0))
        records.append({"params": params, "val_macro_f1": macro})
        if best is None or macro > best[0]:
            best = (macro, params, est)
    elapsed = time.perf_counter() - t0

    assert best is not None
    return SelectionResult(
        model_key=model_key, best_params=best[1], best_val_macro_f1=best[0],
        grid=records, estimator=best[2], train_seconds=elapsed,
        notes={"train_rows_used": used_n,
               "train_rows_available": int(len(X_train)),
               "class_weight": fixed.get("class_weight")},
    )


def sklearn_scores(estimator, X: np.ndarray) -> np.ndarray:
    """Positive-class score for ROC-AUC: predict_proba[:,1] or decision_function."""
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    return estimator.decision_function(X)


# --------------------------------------------------------------------------- #
# LSTM (PyTorch)
# --------------------------------------------------------------------------- #
def build_lstm(input_features: int, lstm_units: int, dense_units: int,
               dropout: float):
    import torch
    from torch import nn

    class LSTMClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_size=input_features, hidden_size=lstm_units,
                                batch_first=True)
            self.drop = nn.Dropout(dropout)
            self.fc1 = nn.Linear(lstm_units, dense_units)
            self.act = nn.ReLU()
            self.fc2 = nn.Linear(dense_units, 1)

        def forward(self, x):                       # x: (B, T, F)
            out, _ = self.lstm(x)
            last = out[:, -1, :]                    # last time step
            h = self.act(self.fc1(self.drop(last)))
            return self.fc2(h).squeeze(-1)          # logits (B,)

    torch.manual_seed(0)  # deterministic init given the global seed set earlier
    return LSTMClassifier()


def _lstm_train_one(params, Xtr, ytr, Xva, yva, cfg, seed):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    fixed = cfg["fixed"]
    units = params.get("lstm_units", fixed["lstm_units"])
    model = build_lstm(Xtr.shape[2], units, fixed["dense_units"], fixed["dropout"])

    pos = float((ytr == 1).sum()); neg = float((ytr == 0).sum())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=fixed["learning_rate"])

    g = torch.Generator().manual_seed(seed)
    dl = DataLoader(
        TensorDataset(torch.from_numpy(Xtr).float(), torch.from_numpy(ytr).float()),
        batch_size=fixed["batch_size"], shuffle=True, generator=g)
    Xva_t = torch.from_numpy(Xva).float()

    best_f1, best_state, best_epoch, patience = -1.0, None, -1, 0
    history = []
    for epoch in range(fixed["epochs"]):
        model.train()
        for xb, yb in dl:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            va_prob = torch.sigmoid(model(Xva_t)).numpy()
        va_pred = (va_prob >= 0.5).astype(int)
        f1m = float(f1_score(yva, va_pred, average="macro", zero_division=0))
        history.append({"epoch": epoch, "val_macro_f1": f1m})
        if f1m > best_f1:
            best_f1, best_epoch = f1m, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= cfg["early_stopping_patience"]:
                break

    model.load_state_dict(best_state)
    return model, best_f1, best_epoch, history


def select_lstm(cfg: Dict[str, Any],
                Xtr: np.ndarray, ytr: np.ndarray,
                Xva: np.ndarray, yva: np.ndarray,
                seed: int) -> SelectionResult:
    records: List[Dict[str, Any]] = []
    best = None
    t0 = time.perf_counter()
    for params in _expand_grid(cfg.get("grid", {})):
        model, f1m, epoch, hist = _lstm_train_one(params, Xtr, ytr, Xva, yva,
                                                  cfg, seed)
        records.append({"params": params, "val_macro_f1": f1m,
                        "best_epoch": epoch})
        if best is None or f1m > best[0]:
            best = (f1m, params, model, epoch, hist)
    elapsed = time.perf_counter() - t0

    assert best is not None
    return SelectionResult(
        model_key="lstm", best_params=best[1], best_val_macro_f1=best[0],
        grid=records, estimator=best[2], train_seconds=elapsed,
        notes={"best_epoch": best[3], "history": best[4],
               "window": cfg["window"], "stride": cfg["stride"],
               "input_features": int(Xtr.shape[2]),
               "train_windows": int(len(Xtr)), "val_windows": int(len(Xva)),
               "pos_weight_in_loss": True},
    )


def lstm_predict_proba(model, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i:i + batch_size]).float()
            out.append(torch.sigmoid(model(xb)).numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)
