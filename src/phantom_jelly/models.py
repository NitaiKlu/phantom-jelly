"""Model zoo: physics baselines, linear models, gradient boosting and a small MLP.

Every estimator is a :class:`~sklearn.pipeline.Pipeline` that starts with imputation, so
scaling and imputation statistics are always fitted inside the CV fold rather than on the
whole dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, HuberRegressor, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config as cfg

try:
    from lightgbm import LGBMRegressor

    HAS_LGBM = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_LGBM = False

try:
    from xgboost import XGBRegressor

    HAS_XGB = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_XGB = False


# --------------------------------------------------------------------------------------
# Physics baselines
# --------------------------------------------------------------------------------------
class ProportionalCalibration(BaseEstimator, RegressorMixin):
    """One-parameter calibration ``y = k * x`` on a single named column.

    SPECT reconstruction is linear in activity concentration, so the count rate in a ROI
    should be proportional to activity density with a scanner-specific constant. Fitting
    through the origin keeps that physical constraint and leaves a single parameter to
    estimate, which is the right complexity for a dataset with ~67 independent scans.
    """

    def __init__(self, column: str = cfg.NM_COUNTS_COLUMN):
        self.column = column

    def _x(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return X[self.column].to_numpy(dtype=float)
        return np.asarray(X, dtype=float).ravel()

    def fit(self, X, y):
        x = self._x(X)
        y = np.asarray(y, dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        # Least squares through the origin.
        self.k_ = float(np.dot(x[ok], y[ok]) / np.dot(x[ok], x[ok]))
        return self

    def predict(self, X):
        return self.k_ * self._x(X)


class AffineCalibration(BaseEstimator, RegressorMixin):
    """Two-parameter calibration ``y = a * x + b`` on a single named column."""

    def __init__(self, column: str = cfg.NM_VOLUME_COLUMN):
        self.column = column

    def _x(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return X[self.column].to_numpy(dtype=float)
        return np.asarray(X, dtype=float).ravel()

    def fit(self, X, y):
        x = self._x(X)
        y = np.asarray(y, dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        self.a_, self.b_ = np.polyfit(x[ok], y[ok], 1)
        return self

    def predict(self, X):
        return self.a_ * self._x(X) + self.b_


class PhysicsResidualRegressor(BaseEstimator, RegressorMixin):
    """Fit a physics calibration first, then learn its residual with a base estimator.

    Unlike :class:`PhysicsRatioRegressor` the correction is additive, which suits a target
    whose relation to the anchor has a real offset rather than a pure gain. The residual
    is shrunk by ``shrinkage`` because with ~6 independent sessions the residual model is
    the part most prone to fitting session-specific quirks.
    """

    def __init__(self, base_estimator=None, anchor_model=None, shrinkage: float = 1.0):
        self.base_estimator = base_estimator
        self.anchor_model = anchor_model
        self.shrinkage = shrinkage

    def fit(self, X, y):
        from sklearn.base import clone as _clone

        y = np.asarray(y, dtype=float)
        self.anchor_ = _clone(
            self.anchor_model if self.anchor_model is not None else AffineCalibration()
        )
        self.anchor_.fit(X, y)
        residual = y - self.anchor_.predict(X)
        self.estimator_ = _clone(
            self.base_estimator if self.base_estimator is not None else Ridge(alpha=100.0)
        )
        self.estimator_.fit(X, residual)
        return self

    def predict(self, X):
        return self.anchor_.predict(X) + self.shrinkage * self.estimator_.predict(X)


class PhysicsRatioRegressor(BaseEstimator, RegressorMixin):
    """Predict ``y / anchor`` with a base estimator, then multiply the anchor back in.

    Both targets are close to proportional to an NM measurement: activity density to the
    mean count rate, CT volume to the NM volume. Learning the *ratio* instead of the raw
    target has two consequences that matter here.

    First, it makes tree ensembles usable. A tree can only predict values it saw in
    training, and the held-out session is shifted well above the training range for both
    targets, so a tree trained on the raw target is capped below the values it needs to
    produce. The ratio, in contrast, is a dimensionless correction factor that occupies
    the same range in every session, and the multiplication by the anchor supplies the
    extrapolation.

    Second, it encodes the physics as a hard constraint rather than something the model
    has to rediscover from ~67 independent acquisitions, which leaves the limited signal
    available for the correction rather than the leading term.
    """

    def __init__(self, base_estimator=None, anchor: str = cfg.NM_COUNTS_COLUMN, clip: float = 5.0):
        self.base_estimator = base_estimator
        self.anchor = anchor
        self.clip = clip

    def _anchor(self, X) -> np.ndarray:
        a = np.asarray(X[self.anchor], dtype=float)
        return np.where(np.abs(a) < 1e-12, np.nan, a)

    def fit(self, X, y):
        from sklearn.base import clone as _clone

        anchor = self._anchor(X)
        ratio = np.asarray(y, dtype=float) / anchor
        ok = np.isfinite(ratio)
        self.ratio_median_ = float(np.nanmedian(ratio[ok]))
        # Guard against a handful of pathological ratios dominating the fit.
        lo, hi = self.ratio_median_ / self.clip, self.ratio_median_ * self.clip
        ratio = np.clip(ratio, min(lo, hi), max(lo, hi))
        self.estimator_ = _clone(
            self.base_estimator if self.base_estimator is not None else Ridge(alpha=10.0)
        )
        self.estimator_.fit(X[ok], ratio[ok])
        return self

    def predict(self, X):
        anchor = self._anchor(X)
        ratio = np.asarray(self.estimator_.predict(X), dtype=float)
        ratio = np.where(np.isfinite(ratio), ratio, self.ratio_median_)
        return ratio * np.nan_to_num(anchor, nan=0.0)


# --------------------------------------------------------------------------------------
# Pipeline helpers
# --------------------------------------------------------------------------------------
def _scaled(estimator) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", estimator),
        ]
    )


def _tree(estimator) -> Pipeline:
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("model", estimator)])


# --------------------------------------------------------------------------------------
# Zoo
# --------------------------------------------------------------------------------------
IS_ACTIVITY = "Activity_Density_uCiml_NM_40"


def anchor_for(target: str) -> str:
    """The NM measurement that a target is physically proportional to."""
    return cfg.NM_COUNTS_COLUMN if target == IS_ACTIVITY else cfg.NM_VOLUME_COLUMN


def model_zoo(target: str, *, include_slow: bool = True) -> dict[str, object]:
    """Return the candidate models for a target, keyed by a human-readable name."""
    rs = cfg.RANDOM_STATE
    anchor = anchor_for(target)

    physics_baseline = (
        ProportionalCalibration(cfg.NM_COUNTS_COLUMN)
        if target == IS_ACTIVITY
        else AffineCalibration(cfg.NM_VOLUME_COLUMN)
    )
    baseline_name = (
        "Physics: k x counts" if target == IS_ACTIVITY else "Physics: a x V_NM + b"
    )

    zoo: dict[str, object] = {
        baseline_name: physics_baseline,
        "Ridge": _scaled(Ridge(alpha=10.0, random_state=rs)),
        "ElasticNet": _scaled(ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=20000, random_state=rs)),
        "Huber": _scaled(HuberRegressor(alpha=1e-3, max_iter=2000)),
        "RandomForest": _tree(
            RandomForestRegressor(
                n_estimators=500,
                min_samples_leaf=4,
                max_features=0.3,
                random_state=rs,
                n_jobs=-1,
            )
        ),
        "HistGBM": _tree(
            HistGradientBoostingRegressor(
                max_depth=4,
                learning_rate=0.05,
                max_iter=400,
                min_samples_leaf=10,
                l2_regularization=1.0,
                random_state=rs,
            )
        ),
        "MLP": _scaled(
            MLPRegressor(
                hidden_layer_sizes=(64, 32),
                alpha=1e-2,
                learning_rate_init=3e-3,
                max_iter=3000,
                early_stopping=True,
                n_iter_no_change=30,
                random_state=rs,
            )
        ),
    }

    if include_slow and HAS_XGB:
        zoo["XGBoost"] = _tree(
            XGBRegressor(
                n_estimators=600,
                max_depth=4,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.5,
                reg_lambda=2.0,
                min_child_weight=5,
                random_state=rs,
                n_jobs=-1,
                verbosity=0,
            )
        )
    if include_slow and HAS_LGBM:
        zoo["LightGBM"] = _tree(
            LGBMRegressor(
                n_estimators=600,
                num_leaves=15,
                learning_rate=0.03,
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=0.5,
                reg_lambda=2.0,
                min_child_samples=15,
                random_state=rs,
                n_jobs=-1,
                verbose=-1,
            )
        )

    # Physics-anchored variants: the same learners applied to the dimensionless ratio.
    ratio_bases = {
        "Ratio+Ridge": _scaled(Ridge(alpha=10.0, random_state=rs)),
        "Ratio+RandomForest": _tree(
            RandomForestRegressor(
                n_estimators=500,
                min_samples_leaf=8,
                max_features=0.3,
                random_state=rs,
                n_jobs=-1,
            )
        ),
        "Ratio+HistGBM": _tree(
            HistGradientBoostingRegressor(
                max_depth=3,
                learning_rate=0.05,
                max_iter=300,
                min_samples_leaf=20,
                l2_regularization=1.0,
                random_state=rs,
            )
        ),
    }
    if include_slow and HAS_LGBM:
        ratio_bases["Ratio+LightGBM"] = _tree(
            LGBMRegressor(
                n_estimators=500,
                num_leaves=15,
                learning_rate=0.03,
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=0.5,
                reg_lambda=2.0,
                min_child_samples=20,
                random_state=rs,
                n_jobs=-1,
                verbose=-1,
            )
        )
    for name, base in ratio_bases.items():
        zoo[name] = PhysicsRatioRegressor(base_estimator=base, anchor=anchor)

    # Additive correction on top of the calibration. Volume has a genuine offset relative
    # to its anchor (V_CT ~ 0.89 V_NM + 6.0), so a gain-only ratio model is the wrong shape
    # for it; a heavily shrunk residual is what actually helps.
    anchor_model = (
        ProportionalCalibration(cfg.NM_COUNTS_COLUMN)
        if target == IS_ACTIVITY
        else AffineCalibration(cfg.NM_VOLUME_COLUMN)
    )
    residual_base = ratio_bases.get("Ratio+LightGBM") or ratio_bases["Ratio+HistGBM"]
    for shrinkage in (0.3, 0.5):
        zoo[f"Residual+GBM (shrink {shrinkage})"] = PhysicsResidualRegressor(
            base_estimator=residual_base,
            anchor_model=anchor_model,
            shrinkage=shrinkage,
        )

    return zoo


# --------------------------------------------------------------------------------------
# Hyper-parameter search spaces (used by scripts/tune.py)
# --------------------------------------------------------------------------------------
def search_space(name: str) -> dict:
    """Randomised-search distributions for the tunable members of the zoo."""
    spaces = {
        "Ridge": {"model__alpha": np.logspace(-3, 4, 40)},
        "ElasticNet": {
            "model__alpha": np.logspace(-4, 1, 40),
            "model__l1_ratio": np.linspace(0.05, 1.0, 20),
        },
        "RandomForest": {
            "model__n_estimators": [300, 500, 800],
            "model__min_samples_leaf": [1, 2, 4, 8, 16],
            "model__max_features": [0.2, 0.3, 0.5, 0.8, 1.0],
            "model__max_depth": [None, 6, 10, 16],
        },
        "HistGBM": {
            "model__learning_rate": [0.02, 0.05, 0.1],
            "model__max_depth": [2, 3, 4, 6],
            "model__min_samples_leaf": [5, 10, 20, 40],
            "model__l2_regularization": [0.0, 0.5, 2.0, 10.0],
            "model__max_iter": [200, 400, 800],
        },
        "XGBoost": {
            "model__learning_rate": [0.02, 0.05, 0.1],
            "model__max_depth": [2, 3, 4, 6],
            "model__min_child_weight": [1, 5, 10, 20],
            "model__subsample": [0.6, 0.8, 1.0],
            "model__colsample_bytree": [0.3, 0.5, 0.8],
            "model__reg_lambda": [0.5, 2.0, 10.0],
            "model__n_estimators": [300, 600, 1000],
        },
        "LightGBM": {
            "model__learning_rate": [0.02, 0.05, 0.1],
            "model__num_leaves": [7, 15, 31, 63],
            "model__min_child_samples": [5, 15, 30, 50],
            "model__subsample": [0.6, 0.8, 1.0],
            "model__colsample_bytree": [0.3, 0.5, 0.8],
            "model__reg_lambda": [0.5, 2.0, 10.0],
            "model__n_estimators": [300, 600, 1000],
        },
        "MLP": {
            "model__hidden_layer_sizes": [(32,), (64, 32), (128, 64), (64, 64, 32)],
            "model__alpha": np.logspace(-4, 1, 20),
            "model__learning_rate_init": [1e-3, 3e-3, 1e-2],
        },
    }
    return spaces.get(name, {})
