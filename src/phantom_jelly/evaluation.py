"""Cross-validation schemes, metrics and the leakage demonstration."""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict

from . import config as cfg


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------
@dataclass
class Metrics:
    r2: float
    rmse: float
    mae: float
    mape: float
    n: int

    def as_dict(self) -> dict:
        return asdict(self)


def compute_metrics(y_true, y_pred) -> Metrics:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[ok], y_pred[ok]
    nonzero = np.abs(y_true) > 1e-12
    mape = (
        float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100)
        if nonzero.any()
        else float("nan")
    )
    return Metrics(
        r2=float(r2_score(y_true, y_pred)),
        rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        mae=float(mean_absolute_error(y_true, y_pred)),
        mape=mape,
        n=int(y_true.size),
    )


# --------------------------------------------------------------------------------------
# Cross-validation
# --------------------------------------------------------------------------------------
def make_cv(scheme: str, groups: pd.Series, n_splits: int = 5):
    """Build a CV splitter.

    ``"random"`` exists only to quantify how badly row-wise splitting inflates scores on
    this dataset; never use it to select a model.
    """
    if scheme == "random":
        return KFold(n_splits=n_splits, shuffle=True, random_state=cfg.RANDOM_STATE)
    if scheme == "group":
        return GroupKFold(n_splits=min(n_splits, groups.nunique()))
    raise ValueError(f"unknown cv scheme {scheme!r}")


def cross_validate_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    *,
    scheme: str = "group",
    n_splits: int = 5,
) -> tuple[Metrics, np.ndarray]:
    """Out-of-fold predictions and the metrics computed on them."""
    cv = make_cv(scheme, groups, n_splits)
    split_groups = groups if scheme == "group" else None
    preds = cross_val_predict(clone(model), X, y, cv=cv, groups=split_groups, n_jobs=1)
    return compute_metrics(y, preds), preds


def leakage_report(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    sessions: pd.Series,
    *,
    n_splits: int = 5,
) -> pd.DataFrame:
    """Score one model under three progressively stricter splits.

    The drop from row 1 to row 2 is the performance that came purely from having ROIs of
    the same acquisition on both sides of the split. The drop from row 2 to row 3 is the
    additional cost of generalising to an acquisition session the model has never seen —
    which is exactly what the supplied test file asks for.
    """
    specs = [
        ("random", groups, "random KFold (leaky)"),
        ("group", groups, "GroupKFold by scan"),
        ("group", sessions, "GroupKFold by session"),
    ]
    rows = []
    for scheme, grp, label in specs:
        metrics, _ = cross_validate_model(model, X, y, grp, scheme=scheme, n_splits=n_splits)
        rows.append({"split": label, **metrics.as_dict()})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Full evaluation of one model
# --------------------------------------------------------------------------------------
def evaluate(
    name: str,
    model,
    dataset,
    *,
    n_splits: int = 5,
) -> dict:
    """Session-grouped CV on train plus a score on the untouched held-out test session."""
    scan_cv, _ = cross_validate_model(
        model, dataset.X_train, dataset.y_train, dataset.groups_train, n_splits=n_splits
    )
    session_cv, _ = cross_validate_model(
        model, dataset.X_train, dataset.y_train, dataset.sessions_train, n_splits=n_splits
    )
    fitted = clone(model).fit(dataset.X_train, dataset.y_train)
    test_metrics = compute_metrics(dataset.y_test, fitted.predict(dataset.X_test))
    return {
        "model": name,
        "cv_scan_r2": scan_cv.r2,
        "cv_session_r2": session_cv.r2,
        "cv_session_rmse": session_cv.rmse,
        "test_r2": test_metrics.r2,
        "test_rmse": test_metrics.rmse,
        "test_mae": test_metrics.mae,
        "test_mape": test_metrics.mape,
    }


def results_table(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    return df.sort_values("test_r2", ascending=False).reset_index(drop=True)
