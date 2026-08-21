"""Test the conditional feature exception for CT radiomics.

Trains an NM-only surrogate for every non-circular CT radiomics feature, keeps those with
cross-validated R² > 0.85, and measures whether feeding the accepted surrogates into the
main models actually improves the held-out session score.

Run with ``python scripts/run_surrogate.py``. Writes ``reports/surrogate.md``.
"""

from __future__ import annotations

import copy
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from phantom_jelly import config as cfg
from phantom_jelly.data import add_physics_features, clean, load_raw, select_features
from phantom_jelly.data import Dataset, FeatureSelection
from phantom_jelly.evaluation import evaluate
from phantom_jelly.models import model_zoo
from phantom_jelly.surrogate import (
    CIRCULARITY_THRESHOLD,
    R2_ACCEPTANCE,
    circular_features,
    fit_surrogates,
)

warnings.filterwarnings("ignore")


def build_full_frames():
    train_raw, test_raw = load_raw("full")
    train, test = clean(train_raw), clean(test_raw)
    train, test = add_physics_features(train), add_physics_features(test)
    return train_raw, test_raw, train, test


def main() -> None:
    cfg.REPORTS_DIR.mkdir(exist_ok=True)
    md = ["# Conditional feature exception: CT radiomics surrogates\n"]

    train_raw, test_raw, train, test = build_full_frames()
    sessions = train_raw[cfg.SESSION_KEY].astype(str).reset_index(drop=True)
    scans = train_raw[cfg.SCAN_KEY].reset_index(drop=True)
    sessions_test = test_raw[cfg.SESSION_KEY].astype(str).reset_index(drop=True)
    scans_test = test_raw[cfg.SCAN_KEY].reset_index(drop=True)

    # NM-only predictor block (the legal features).
    nm_selection = select_features(train, allow_ct_radiomics=False, allow_nm80=True)
    nm_features = [c for c in nm_selection.features if c in test.columns]
    X_nm_train, X_nm_test = train[nm_features], test[nm_features]

    # Candidate CT block.
    ct_cols = [
        c
        for c in train.select_dtypes("number").columns
        if (c.endswith("_CT") or c in cfg.OVERLAP_COLUMNS) and c not in cfg.TARGETS
    ]
    ct_cols = [c for c in ct_cols if c in test.columns and train[c].nunique() > 1]
    targets = train[list(cfg.TARGETS)]

    md.append(
        f"\nNM-only predictor block: **{len(nm_features)}** features. "
        f"Candidate CT-derived columns: **{len(ct_cols)}**.\n"
    )

    md.append("\n## 1. Circularity screen\n")
    circ = circular_features(train[ct_cols], targets, CIRCULARITY_THRESHOLD)
    md.append(
        f"CT columns correlating |r| > {CIRCULARITY_THRESHOLD} with a target are removed "
        "before any surrogate is trained. Predicting these and feeding them back would "
        "reintroduce the target through the back door.\n"
    )
    md.append(circ.to_markdown(index=False) if len(circ) else "_none_")
    ct_cols = [c for c in ct_cols if c not in set(circ.get("feature", []))]
    md.append(f"\n**{len(ct_cols)}** CT columns survive the screen.\n")

    md.append("\n## 2. Surrogate validation (session-grouped CV)\n")
    result = fit_surrogates(X_nm_train, train[ct_cols], X_nm_test, sessions)
    n_pass = int(result.scores["accepted"].sum())
    md.append(
        f"Surrogates reaching the R² > {R2_ACCEPTANCE} bar: **{n_pass} / {len(result.scores)}**.\n"
    )
    md.append("\nTop 20 by cross-validated R²:\n")
    md.append(result.scores.head(20).round(4).to_markdown(index=False))

    if n_pass == 0:
        md.append(
            "\n**No CT radiomics feature clears the bar.** The exception cannot be "
            "invoked, and the baseline NM-only feature set stands as the final one."
        )
        (cfg.REPORTS_DIR / "surrogate.md").write_text("\n".join(md), encoding="utf-8")
        print("\n".join(md))
        return

    md.append("\n## 3. Do the accepted surrogates improve the final models?\n")
    md.append(
        "Training rows use **out-of-fold** surrogate values so the downstream model never "
        "sees a surrogate fitted on its own row; test rows use surrogates fitted on the "
        "full training set.\n"
    )

    rows = []
    for target in cfg.TARGETS:
        base = Dataset(
            X_train=X_nm_train,
            y_train=train[target],
            groups_train=scans,
            sessions_train=sessions,
            X_test=X_nm_test,
            y_test=test[target],
            groups_test=scans_test,
            sessions_test=sessions_test,
            selection=FeatureSelection(nm_features, {}),
            target=target,
        )
        augmented = copy.copy(base)
        augmented.X_train = pd.concat([X_nm_train, result.oof_train], axis=1)
        augmented.X_test = pd.concat([X_nm_test, result.pred_test], axis=1)

        for name, model in model_zoo(target).items():
            if name.startswith("Physics"):
                continue
            for label, ds in (("NM only", base), ("NM + surrogates", augmented)):
                try:
                    r = evaluate(name, model, ds)
                    r["features"] = label
                    r["target"] = target
                    rows.append(r)
                except Exception as exc:
                    print(f"  {target} / {name} / {label} failed: {exc}")

    comparison = pd.DataFrame(rows)
    for target in cfg.TARGETS:
        sub = comparison[comparison["target"] == target]
        pivot = sub.pivot_table(
            index="model", columns="features", values=["cv_session_r2", "test_r2"]
        ).round(4)
        md.append(f"\n**{target}**\n")
        md.append(pivot.to_markdown())

    comparison.to_csv(cfg.REPORTS_DIR / "surrogate_comparison.csv", index=False)
    (cfg.REPORTS_DIR / "surrogate.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\n[written] {cfg.REPORTS_DIR / 'surrogate.md'}")


if __name__ == "__main__":
    main()
