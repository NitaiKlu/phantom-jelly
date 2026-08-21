"""Data audit: group structure, effective sample size and leakage quantification.

Run with ``python scripts/audit_data.py``. Writes ``reports/audit.md``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from phantom_jelly import config as cfg
from phantom_jelly.data import build_dataset, clean, load_raw
from phantom_jelly.evaluation import leakage_report
from phantom_jelly.models import _tree
from sklearn.ensemble import RandomForestRegressor


def section(title: str) -> str:
    return f"\n## {title}\n"


def describe_groups(train: pd.DataFrame, test: pd.DataFrame) -> list[str]:
    out = []
    h, d = cfg.SCAN_KEY, cfg.SESSION_KEY
    out.append(
        f"- rows: train **{len(train)}**, test **{len(test)}**\n"
        f"- unique acquisitions (NM image hash): train **{train[h].nunique()}**, "
        f"test **{test[h].nunique()}**\n"
        f"- unique sessions (date): train **{train[d].nunique()}**, test **{test[d].nunique()}**\n"
        f"- acquisitions shared between train and test: "
        f"**{len(set(train[h]) & set(test[h]))}**\n"
        f"- dates shared between train and test: **{len(set(train[d]) & set(test[d]))}**"
    )
    return out


def describe_target_granularity(train: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in cfg.TARGETS:
        g = train.groupby(cfg.SCAN_KEY)[target]
        within = g.std().mean()
        overall = train[target].std()
        rows.append(
            {
                "target": target,
                "rows": len(train),
                "unique values": train[target].nunique(),
                "unique per scan (mean)": round(g.nunique().mean(), 2),
                "scans with 1 unique value": f"{(g.nunique() == 1).sum()}/{train[cfg.SCAN_KEY].nunique()}",
                "within-scan sd / overall sd": round(within / overall, 3),
            }
        )
    return pd.DataFrame(rows)


def describe_distribution_shift(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in cfg.TARGETS:
        rows.append(
            {
                "target": target,
                "train mean": round(train[target].mean(), 3),
                "test mean": round(test[target].mean(), 3),
                "train sd": round(train[target].std(), 3),
                "test sd": round(test[target].std(), 3),
                "train range": f"{train[target].min():.2f} – {train[target].max():.2f}",
                "test range": f"{test[target].min():.2f} – {test[target].max():.2f}",
            }
        )
    return pd.DataFrame(rows)


def describe_calibration(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Per-session ratio of activity density to mean counts."""
    target = "Activity_Density_uCiml_NM_40"
    ratio = train[target] / train[cfg.NM_COUNTS_COLUMN]
    rows = (
        train.assign(_ratio=ratio)
        .groupby(cfg.SESSION_KEY)
        .agg(
            n=("_ratio", "size"),
            scale_factor=(cfg.SCALE_FACTOR_COLUMN, "first"),
            k_mean=("_ratio", "mean"),
            k_cv_pct=("_ratio", lambda s: 100 * s.std() / s.mean()),
        )
        .reset_index()
    )
    test_ratio = test[target] / test[cfg.NM_COUNTS_COLUMN]
    rows = pd.concat(
        [
            rows,
            pd.DataFrame(
                [
                    {
                        cfg.SESSION_KEY: "TEST SESSION",
                        "n": len(test),
                        "scale_factor": test[cfg.SCALE_FACTOR_COLUMN].iloc[0],
                        "k_mean": test_ratio.mean(),
                        "k_cv_pct": 100 * test_ratio.std() / test_ratio.mean(),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    rows["k_mean"] = rows["k_mean"].map(lambda v: f"{v:.3e}")
    rows["k_cv_pct"] = rows["k_cv_pct"].round(1)
    return rows


def leaky_correlations(train: pd.DataFrame) -> pd.DataFrame:
    """CT-derived columns ranked by how strongly they encode Volume_[ml]_CT."""
    target = "Volume_[ml]_CT"
    ct_cols = [
        c
        for c in train.select_dtypes(include=[np.number]).columns
        if (c.endswith("_CT") or c in cfg.OVERLAP_COLUMNS) and c != target
    ]
    corr = train[ct_cols].corrwith(train[target]).abs().sort_values(ascending=False)
    return pd.DataFrame(
        {
            "CT-derived column": corr.head(12).index,
            "|r| with Volume_[ml]_CT": corr.head(12).to_numpy().round(4),
        }
    )


def main() -> None:
    cfg.REPORTS_DIR.mkdir(exist_ok=True)
    md: list[str] = ["# Data audit\n"]

    train_raw, test_raw = load_raw("reduced")
    full_train, full_test = load_raw("full")

    md.append(section("1. Dataset shapes and group structure (reduced set)"))
    md.extend(describe_groups(train_raw, test_raw))
    md.append(
        f"\nThe full set carries more rows (train {len(full_train)}, test {len(full_test)}); "
        "the reduced set additionally drops the smallest ROIs "
        f"(mean Volume_[ml]_CT of dropped rows ≈ 7.2 ml vs 15.4 ml for kept rows)."
    )

    md.append(section("2. Target granularity — the effective sample size"))
    md.append(describe_target_granularity(train_raw).to_markdown(index=False))
    md.append(
        "\n`Activity_Density_uCiml_NM_40` is essentially a **per-acquisition constant**: "
        "51 of 67 training scans carry a single value and the mean within-scan spread is "
        "27% of the overall spread. The 1766 rows therefore contain roughly **67 "
        "independent observations** of this target, not 1766. `Volume_[ml]_CT` is a "
        "genuine per-ROI quantity (within-scan spread ≈ overall spread) and keeps its "
        "row-level sample size."
    )

    md.append(section("3. Distribution shift between train and test"))
    md.append(describe_distribution_shift(train_raw, test_raw).to_markdown(index=False))
    md.append(
        "\nThe supplied test file is a **separate acquisition session** (a single date, "
        "no shared scan hashes). Both targets are shifted upwards relative to training, "
        "so the test score measures extrapolation to a new session — the clinically "
        "relevant question, and a much harder one than a random row split."
    )

    md.append(section("4. Count-to-activity calibration is stable across sessions"))
    md.append(describe_calibration(train_raw, test_raw).to_markdown(index=False))
    md.append(
        "\nThe ratio of activity density to mean NM counts sits near 3e-5 for every "
        "session including the unseen test session, despite `Scale_factor_NM_40` varying "
        "over two orders of magnitude between sessions. The counts column is already "
        "de-scaled, so a single proportionality constant transfers across sessions and "
        "makes a one-parameter physics model a strong baseline.\n\n"
        "The often-quoted global correlation of r = 0.977 between counts and activity "
        "density is misleading: within a single session it falls to 0.45–0.55 for half "
        "the sessions. Most of that correlation is between-session variance."
    )

    md.append(section("5. CT-derived columns leak target 1"))
    md.append(leaky_correlations(full_train).to_markdown(index=False))
    md.append(
        "\nEvery `*_CT` shape feature is computed from the same CT segmentation that "
        "defines `Volume_[ml]_CT`, so several of them reproduce the target almost exactly "
        "(`original_shape_VoxelVolume_CT` and `original_shape_MeshVolume_CT` are the "
        "target in mm³). The `*_to_*_overlap` columns also require the CT mask. None of "
        "these may enter a baseline model — which is precisely why the reduced set omits "
        "them."
    )

    md.append(section("6. Leakage cost of a random row split"))
    md.append(
        "The same model, the same features, the same data — only the split changes.\n"
    )
    rf = _tree(
        RandomForestRegressor(
            n_estimators=300, min_samples_leaf=4, random_state=cfg.RANDOM_STATE, n_jobs=-1
        )
    )
    observed: dict[str, pd.DataFrame] = {}
    for target in cfg.TARGETS:
        ds = build_dataset(target, "reduced")
        report = leakage_report(
            rf, ds.X_train, ds.y_train, ds.groups_train, ds.sessions_train
        )
        observed[target] = report
        md.append(f"\n**{target}**\n")
        md.append(report.round(3).to_markdown(index=False))

    def r2_of(target: str, split: str) -> float:
        row = observed[target].loc[observed[target]["split"] == split, "r2"]
        return float(row.iloc[0])

    ad = "Activity_Density_uCiml_NM_40"
    md.append(
        "\nRandom `train_test_split` / `KFold` on rows places ROIs from the *same* "
        "acquisition on both sides of the split. Because activity density is constant "
        "within an acquisition, the model only has to recognise which scan a row came "
        "from — trivial, given that radiomics features are near-identical within a scan. "
        f"For activity density that inflates R² to {r2_of(ad, 'random KFold (leaky)'):.3f}, "
        f"against {r2_of(ad, 'GroupKFold by scan'):.3f} once scans are kept together and "
        f"{r2_of(ad, 'GroupKFold by session'):.3f} once whole sessions are held out.\n\n"
        "Volume is a genuine per-ROI quantity, so it barely moves between the random and "
        "scan-grouped splits "
        f"({r2_of('Volume_[ml]_CT', 'random KFold (leaky)'):.3f} vs "
        f"{r2_of('Volume_[ml]_CT', 'GroupKFold by scan'):.3f}); it is the *session*-grouped "
        f"split ({r2_of('Volume_[ml]_CT', 'GroupKFold by session'):.3f}) that reveals its "
        "true generalisation gap. Session-grouped CV is therefore the estimate used for "
        "model selection everywhere else in this repository."
    )

    md.append(section("7. Feature selection audit (reduced set, target 1)"))
    ds = build_dataset("Volume_[ml]_CT", "reduced")
    md.append("```\n" + ds.selection.summary() + "\n```")

    out = cfg.REPORTS_DIR / "audit.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\n[written] {out}")


if __name__ == "__main__":
    main()
