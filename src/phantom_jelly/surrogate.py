"""The conditional feature exception: predicting CT radiomics from NM-only inputs.

CT radiomics may only enter the pipeline through a surrogate that is itself validated to
cross-validated R² > 0.85 and that provably cannot see either target. Two guards enforce
that here.

1. *Circularity guard.* A CT feature that is an algebraic restatement of
   ``Volume_[ml]_CT`` (``original_shape_VoxelVolume_CT`` is the target in mm³) must be
   excluded. Predicting it and feeding it back in is just the volume model wearing a
   disguise, and it would make the pipeline look self-consistent while adding nothing.
2. *Provenance guard.* Surrogates are trained on NM-only features under session-grouped
   CV, and the surrogate values used downstream are always out-of-fold predictions on
   train and fresh predictions on test, so no CT measurement ever reaches the final model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, cross_val_predict

from . import config as cfg
from .models import _tree

try:
    from lightgbm import LGBMRegressor

    def _default_surrogate():
        return _tree(
            LGBMRegressor(
                n_estimators=400,
                num_leaves=15,
                learning_rate=0.05,
                colsample_bytree=0.6,
                subsample=0.8,
                subsample_freq=1,
                min_child_samples=20,
                reg_lambda=2.0,
                random_state=cfg.RANDOM_STATE,
                n_jobs=-1,
                verbose=-1,
            )
        )

except ImportError:  # pragma: no cover - optional dependency
    from sklearn.ensemble import HistGradientBoostingRegressor

    def _default_surrogate():
        return _tree(
            HistGradientBoostingRegressor(
                max_depth=4, learning_rate=0.05, max_iter=300, random_state=cfg.RANDOM_STATE
            )
        )


CIRCULARITY_THRESHOLD = 0.95
R2_ACCEPTANCE = 0.85


@dataclass
class SurrogateResult:
    scores: pd.DataFrame
    accepted: list[str]
    excluded_circular: pd.DataFrame
    oof_train: pd.DataFrame
    pred_test: pd.DataFrame


def circular_features(
    ct_frame: pd.DataFrame, targets: pd.DataFrame, threshold: float = CIRCULARITY_THRESHOLD
) -> pd.DataFrame:
    """CT columns whose |correlation| with either target exceeds ``threshold``."""
    rows = []
    for col in ct_frame.columns:
        corrs = {t: abs(ct_frame[col].corr(targets[t])) for t in targets.columns}
        worst = max(corrs.values())
        if np.isfinite(worst) and worst > threshold:
            rows.append({"feature": col, **{f"|r| {t}": round(corrs[t], 4) for t in corrs}})
    return pd.DataFrame(rows)


def fit_surrogates(
    X_nm_train: pd.DataFrame,
    ct_train: pd.DataFrame,
    X_nm_test: pd.DataFrame,
    groups: pd.Series,
    *,
    estimator=None,
    n_splits: int = 5,
    acceptance: float = R2_ACCEPTANCE,
) -> SurrogateResult:
    """Train one NM-only surrogate per CT feature and keep those that pass validation.

    Returns out-of-fold surrogate values for the training rows, so that a downstream model
    consuming them is never shown a surrogate that was fitted on the row it is predicting.
    """
    from sklearn.metrics import r2_score

    estimator = estimator if estimator is not None else _default_surrogate()
    cv = GroupKFold(n_splits=min(n_splits, groups.nunique()))

    scores, oof, preds = [], {}, {}
    for col in ct_train.columns:
        y = ct_train[col]
        if not np.isfinite(y).all() or y.nunique() <= 1:
            continue
        oof_pred = cross_val_predict(clone(estimator), X_nm_train, y, cv=cv, groups=groups)
        r2 = float(r2_score(y, oof_pred))
        scores.append({"ct_feature": col, "cv_r2": r2, "accepted": r2 > acceptance})
        if r2 > acceptance:
            oof[f"surrogate__{col}"] = oof_pred
            preds[f"surrogate__{col}"] = clone(estimator).fit(X_nm_train, y).predict(X_nm_test)

    score_df = pd.DataFrame(scores).sort_values("cv_r2", ascending=False).reset_index(drop=True)
    return SurrogateResult(
        scores=score_df,
        accepted=list(oof),
        excluded_circular=pd.DataFrame(),
        oof_train=pd.DataFrame(oof, index=X_nm_train.index),
        pred_test=pd.DataFrame(preds, index=X_nm_test.index),
    )
