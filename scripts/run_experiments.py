"""Benchmark the model zoo on both targets under leakage-safe validation.

Run with ``python scripts/run_experiments.py``. Writes ``reports/results.md`` and
``reports/results.csv``.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from phantom_jelly import config as cfg
from phantom_jelly.data import build_dataset
from phantom_jelly.evaluation import evaluate, results_table
from phantom_jelly.models import model_zoo

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def run_target(target: str, variant: str, n_splits: int) -> tuple[pd.DataFrame, object]:
    dataset = build_dataset(target, variant)
    rows = []
    for name, model in model_zoo(target).items():
        print(f"  [{target}] {name} ...", flush=True)
        try:
            rows.append(evaluate(name, model, dataset, n_splits=n_splits))
        except Exception as exc:  # keep the benchmark going if one model misbehaves
            print(f"    failed: {exc}")
    return results_table(rows), dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="reduced", choices=sorted(cfg.DATASETS))
    parser.add_argument("--n-splits", type=int, default=5)
    args = parser.parse_args()

    cfg.REPORTS_DIR.mkdir(exist_ok=True)
    md = [
        "# Model benchmark\n",
        f"Dataset variant: **{args.variant}** (no CT-derived features).\n",
        "`cv_scan_r2` groups folds by acquisition; `cv_session_r2` groups by acquisition "
        "date and is the honest analogue of the held-out test session. `test_*` columns "
        "are measured on the supplied test file, an acquisition session never seen "
        "during training or model selection.\n",
    ]
    frames = []

    for target in cfg.TARGETS:
        table, dataset = run_target(target, args.variant, args.n_splits)
        table.insert(0, "target", target)
        frames.append(table)
        md.append(f"\n## {target}\n")
        md.append(
            f"{len(dataset.X_train)} training rows / {dataset.n_groups} acquisitions / "
            f"{dataset.n_sessions} sessions, {len(dataset.X_train.columns)} features.\n"
        )
        md.append(table.drop(columns="target").round(4).to_markdown(index=False))

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(cfg.REPORTS_DIR / "results.csv", index=False)
    (cfg.REPORTS_DIR / "results.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\n[written] {cfg.REPORTS_DIR / 'results.md'}")


if __name__ == "__main__":
    main()
