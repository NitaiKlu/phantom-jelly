"""Fit the manuscript models, write paper figures, and check every cited number.

Run with ``python scripts/make_paper_figures.py``.

Writes
    paper/figures/fig3_activity.png
    paper/figures/fig4_volume.png
    reports/paper_numbers.md

and exits non-zero if a manuscript inventory claim no longer matches the CSVs.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone

from phantom_jelly import config as cfg
from phantom_jelly.data import (
    build_dataset,
    experimental_inventory,
    load_raw,
    radiomics_bin_widths,
)
from phantom_jelly.evaluation import compute_metrics
from phantom_jelly.models import AffineCalibration, model_zoo

FIGURE_DIR = cfg.PROJECT_ROOT / "paper" / "figures"


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 200,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
        }
    )


def _identity_limits(*arrays, pad: float = 0.08) -> tuple[float, float]:
    lo = min(float(np.min(a)) for a in arrays)
    hi = max(float(np.max(a)) for a in arrays)
    span = hi - lo or 1.0
    return lo - pad * span, hi + pad * span


def fit_reported(dataset, name: str):
    model = model_zoo(dataset.target)[name]
    fitted = clone(model).fit(dataset.X_train, dataset.y_train)
    pred = fitted.predict(dataset.X_test)
    return fitted, pred, compute_metrics(dataset.y_test, pred)


def figure_activity(dataset, y_pred, metrics, path: Path) -> None:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4))

    x_tr = dataset.X_train[cfg.NM_COUNTS_COLUMN].to_numpy() / 1000.0
    x_te = dataset.X_test[cfg.NM_COUNTS_COLUMN].to_numpy() / 1000.0
    axes[0].plot(x_tr, dataset.y_train, ".b", markersize=4, label="train data", alpha=0.7)
    axes[0].plot(x_te, dataset.y_test, ".r", markersize=4, label="test data", alpha=0.8)
    axes[0].set_xlabel(r"Mean intensity counts ($\times 10^{3}$)")
    axes[0].set_ylabel(r"Activity density ($\mu$Ci/ml)")
    axes[0].legend(loc="upper left")

    y_true = np.asarray(dataset.y_test, dtype=float)
    lo, hi = _identity_limits(y_true, y_pred)
    axes[1].plot(y_true, y_pred, ".r", markersize=5)
    axes[1].plot([lo, hi], [lo, hi], "k--", linewidth=1)
    axes[1].set_xlim(lo, hi)
    axes[1].set_ylim(lo, hi)
    axes[1].set_xlabel(r"True activity density ($\mu$Ci/ml)")
    axes[1].set_ylabel(r"Predicted activity density ($\mu$Ci/ml)")
    axes[1].text(
        0.05,
        0.95,
        f"RMSE: {metrics.rmse:.2f}\n$R^2$: {metrics.r2:.2f}",
        transform=axes[1].transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def figure_volume(dataset, y_pred, metrics, path: Path) -> None:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4))

    axes[0].plot(
        dataset.X_train[cfg.NM_VOLUME_COLUMN],
        dataset.y_train,
        ".b",
        markersize=4,
        label="train data",
        alpha=0.7,
    )
    axes[0].plot(
        dataset.X_test[cfg.NM_VOLUME_COLUMN],
        dataset.y_test,
        ".r",
        markersize=4,
        label="test data",
        alpha=0.8,
    )
    axes[0].set_xlabel(r"Volume NM$_{40}$ (ml)")
    axes[0].set_ylabel(r"Volume CT (ml)")
    axes[0].legend(loc="upper left")

    y_true = np.asarray(dataset.y_test, dtype=float)
    lo, hi = _identity_limits(y_true, y_pred)
    axes[1].plot(y_true, y_pred, ".r", markersize=5)
    axes[1].plot([lo, hi], [lo, hi], "k--", linewidth=1)
    axes[1].set_xlim(lo, hi)
    axes[1].set_ylim(lo, hi)
    axes[1].set_xlabel(r"True volume CT (ml)")
    axes[1].set_ylabel(r"Predicted volume (ml)")
    axes[1].text(
        0.05,
        0.95,
        f"RMSE: {metrics.rmse:.2f}\n$R^2$: {metrics.r2:.2f}",
        transform=axes[1].transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def format_activity_range(lo: float, hi: float) -> str:
    return f"{lo:.1f}--{hi:.1f}"


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    cfg.REPORTS_DIR.mkdir(exist_ok=True)

    inventory = experimental_inventory("full")
    n_segments = int(inventory["segments"].sum())
    n_scans = int(inventory["scans"].sum())
    n_jars = int(inventory["jars"].sum())
    n_sessions = int(len(inventory))
    activity_min = float(inventory["activity_min"].min())
    activity_max = float(inventory["activity_max"].max())
    bin_widths = radiomics_bin_widths("full")

    errors = []
    if n_segments != cfg.EXPECTED_VALIDATED_SEGMENTS:
        errors.append(f"segments {n_segments} != {cfg.EXPECTED_VALIDATED_SEGMENTS}")
    if n_scans != cfg.EXPECTED_SCANS:
        errors.append(f"scans {n_scans} != {cfg.EXPECTED_SCANS}")
    if n_jars != cfg.EXPECTED_JARS:
        errors.append(f"jars {n_jars} != {cfg.EXPECTED_JARS}")
    if n_sessions != cfg.EXPECTED_SESSIONS:
        errors.append(f"sessions {n_sessions} != {cfg.EXPECTED_SESSIONS}")
    if bin_widths != {cfg.RADIOMICS_BIN_WIDTH}:
        errors.append(f"binWidth {bin_widths} != {{{cfg.RADIOMICS_BIN_WIDTH}}}")

    expected_rows = {
        1: (386, 11, 5.5, 7.3),
        2: (327, 13, 3.4, 4.7),
        3: (190, 10, 3.0, 4.0),
        4: (583, 15, 0.6, 0.9),
        5: (482, 13, 1.7, 11.8),
        0: (334, 13, 3.6, 10.2),
    }
    for session, (segs, scans, alo, ahi) in expected_rows.items():
        row = inventory.loc[inventory["session"] == session].iloc[0]
        if int(row["segments"]) != segs:
            errors.append(f"session {session} segments {row['segments']} != {segs}")
        if int(row["scans"]) != scans:
            errors.append(f"session {session} scans {row['scans']} != {scans}")
        if abs(row["activity_min"] - alo) > 0.15:
            errors.append(f"session {session} activity min {row['activity_min']} !~ {alo}")
        if abs(row["activity_max"] - ahi) > 0.15:
            errors.append(f"session {session} activity max {row['activity_max']} !~ {ahi}")

    red_train, _ = load_raw("reduced")
    activity_per_scan = red_train.groupby(cfg.SCAN_KEY)["Activity_Density_uCiml_NM_40"]
    n_constant_scans = int((activity_per_scan.nunique() == 1).sum())
    n_train_scans = int(red_train[cfg.SCAN_KEY].nunique())
    if (n_constant_scans, n_train_scans) != (51, 67):
        errors.append(
            f"constant-activity scans {n_constant_scans}/{n_train_scans} != 51/67"
        )

    activity_ds = build_dataset("Activity_Density_uCiml_NM_40", "reduced")
    volume_ds = build_dataset("Volume_[ml]_CT", "reduced")

    activity_name = "Ratio+LightGBM"
    volume_name = "Physics: a x V_NM + b"
    act_model, act_pred, act_metrics = fit_reported(activity_ds, activity_name)
    vol_model, vol_pred, vol_metrics = fit_reported(volume_ds, volume_name)

    physics_act = clone(model_zoo(activity_ds.target)["Physics: k x counts"])
    physics_act.fit(activity_ds.X_train, activity_ds.y_train)
    physics_act_metrics = compute_metrics(
        activity_ds.y_test, physics_act.predict(activity_ds.X_test)
    )

    raw_lgbm = clone(model_zoo(activity_ds.target)["LightGBM"])
    raw_lgbm.fit(activity_ds.X_train, activity_ds.y_train)
    raw_lgbm_metrics = compute_metrics(activity_ds.y_test, raw_lgbm.predict(activity_ds.X_test))

    affine = AffineCalibration(cfg.NM_VOLUME_COLUMN)
    affine.fit(volume_ds.X_train, volume_ds.y_train)

    figure_activity(activity_ds, act_pred, act_metrics, FIGURE_DIR / "fig3_activity.png")
    figure_volume(volume_ds, vol_pred, vol_metrics, FIGURE_DIR / "fig4_volume.png")

    # Published benchmark (reports/results.md) — the paper cites these rounded values.
    if abs(act_metrics.r2 - 0.949) > 0.01:
        errors.append(f"activity Ratio+LightGBM R^2 {act_metrics.r2:.3f} drifted from 0.949")
    if abs(vol_metrics.r2 - 0.748) > 0.01:
        errors.append(f"volume physics R^2 {vol_metrics.r2:.3f} drifted from 0.748")

    md = [
        "# Manuscript numbers (generated)\n",
        "Every quantity below is computed by this script from the CSVs / fitted models.\n",
        "## Experimental inventory (full dataset)\n",
        inventory.round(3).to_markdown(index=False),
        "",
        f"- validated 1:1 segments: **{n_segments}**",
        f"- SPECT/CT scans: **{n_scans}**",
        f"- jars (experimental log): **{n_jars}**",
        f"- sessions: **{n_sessions}**",
        f"- activity density range: **{activity_min:.2f}--{activity_max:.2f} µCi/ml**",
        f"- PyRadiomics binWidth: **{sorted(bin_widths)}**",
        "",
        "## Modelling split (reduced NM-40 table)\n",
        f"- activity train/test rows: **{len(activity_ds.X_train)}** / **{len(activity_ds.X_test)}**",
        f"- volume train/test rows: **{len(volume_ds.X_train)}** / **{len(volume_ds.X_test)}**",
        f"- modelling features: **{len(activity_ds.X_train.columns)}**",
        f"- training acquisitions / sessions: **{activity_ds.n_groups}** / **{activity_ds.n_sessions}**",
        f"- training scans with a single activity-density value: **{n_constant_scans}/{n_train_scans}**",
        "",
        "## Held-out test metrics\n",
        f"- Ratio+LightGBM activity: R²={act_metrics.r2:.3f}, RMSE={act_metrics.rmse:.2f} µCi/ml, "
        f"MAPE={act_metrics.mape:.1f}%",
        f"- Physics k×counts activity: R²={physics_act_metrics.r2:.3f}, "
        f"RMSE={physics_act_metrics.rmse:.2f} µCi/ml",
        f"- LightGBM on raw activity: R²={raw_lgbm_metrics.r2:.3f}, "
        f"RMSE={raw_lgbm_metrics.rmse:.2f} µCi/ml",
        f"- Physics a·V_NM+b volume: R²={vol_metrics.r2:.3f}, RMSE={vol_metrics.rmse:.2f} ml, "
        f"MAPE={vol_metrics.mape:.1f}%",
        f"- fitted affine: V_CT = {affine.a_:.3f}·V_NM + {affine.b_:.2f}",
        "",
        "## Figures\n",
        f"- `{FIGURE_DIR / 'fig3_activity.png'}`",
        f"- `{FIGURE_DIR / 'fig4_volume.png'}`",
    ]
    (cfg.REPORTS_DIR / "paper_numbers.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))

    if errors:
        print("\n[verify FAILED]")
        for err in errors:
            print(" -", err)
        sys.exit(1)
    print("\n[verify OK] manuscript inventory and headline metrics match the repo.")
    _ = (act_model, vol_model)


if __name__ == "__main__":
    main()
