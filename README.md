# Jelly phantom — SPECT/CT quantification

Predicting two quantities for gelatine-phantom inserts imaged with SPECT/CT:

| Target | Meaning | Granularity |
| --- | --- | --- |
| `Volume_[ml]_CT` | insert volume from the CT segmentation | per ROI |
| `Activity_Density_uCiml_NM_40` | activity concentration (µCi/ml) | **per acquisition** |

The headline result of the audit is that these two targets need completely different
treatment, because they live at different granularities and only one of them is genuinely
a per-row quantity.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The feature-extraction stack (PyRadiomics, SimpleITK, pydicom) is deliberately *not* a
dependency — the pipeline starts from the extracted CSVs. See the note in
`requirements.txt` if you need to regenerate features from raw images.

## Run

```bash
python scripts/audit_data.py       # -> reports/audit.md
python scripts/run_experiments.py  # -> reports/results.md, reports/results.csv
python scripts/run_surrogate.py    # -> reports/surrogate.md   (conditional CT exception)
```

`data/raw/Jelly_Phantom.zip` is unpacked automatically on first use.

## Layout

```
src/phantom_jelly/
  config.py      column taxonomy, leakage blacklist, physical constants
  data.py        loading, cleaning, leakage-safe feature selection
  models.py      physics baselines, physics-anchored meta-estimators, model zoo
  evaluation.py  grouped CV schemes, metrics, leakage quantification
  surrogate.py   NM-only surrogates for CT radiomics (conditional exception)
scripts/         entry points, each writing a report
reports/         generated results
  issues.md      audit of the original code: 15 issues, quotes, and the fix for each
```

## Dataset variants

| Variant | Files | Columns | Contents |
| --- | --- | --- | --- |
| reduced | `train.csv`, `test.csv` | 134 | NM-40 radiomics only; smallest ROIs already dropped |
| full | `df_train.csv`, `df_test.csv` | 390 | adds NM-80 and CT radiomics plus mask-overlap columns |

CT-derived columns are **blacklisted for baseline training** (`data.select_features` with
`allow_ct_radiomics=False`, the default). They are a direct restatement of target 1: `original_shape_VoxelVolume_CT` and
`diagnostics_Mask-original_VoxelNum_CT` correlate with `Volume_[ml]_CT` at |r| = 1.000.
They may only re-enter through a validated NM-only surrogate — see
`scripts/run_surrogate.py`.

## Validation protocol

Three splits, in increasing strictness:

| Split | Purpose |
| --- | --- |
| random `KFold` | **diagnostic only** — reproduces the original code's split to quantify its bias |
| `GroupKFold` by acquisition | minimum needed to stop a per-scan-constant target leaking |
| `GroupKFold` by session (date) | matches deployment: the supplied test file is one unseen session |

Model selection uses **session-grouped CV**. On this dataset it is the only scheme whose
ranking agrees with held-out test performance; scan-grouped CV prefers models that then
collapse on the unseen session.

## Results

Full tables in `reports/results.md`. Test columns are the supplied test file — a single
acquisition session, no scans and no dates shared with training.

| Target | Model | session-CV R² | **test R²** | test RMSE | test MAPE |
| --- | --- | --- | --- | --- | --- |
| `Activity_Density` | LightGBM (raw target) — *baseline* | 0.578 | 0.429 | 1.72 | 23.8 % |
| `Activity_Density` | physics only, `k·counts` | 0.924 | 0.928 | 0.61 | 7.2 % |
| `Activity_Density` | **Ratio + LightGBM** | **0.934** | **0.949** | **0.51** | 6.7 % |
| `Volume_[ml]_CT` | LightGBM (raw target) — *baseline* | 0.406 | 0.722 | 3.96 | 15.6 % |
| `Volume_[ml]_CT` | **physics only, `a·V_NM + b`** | **0.701** | **0.748** | **3.78** | 14.5 % |
| `Volume_[ml]_CT` | Residual + GBM, shrinkage 0.3 | 0.674 | 0.758 | 3.70 | 14.4 % |

Two conclusions worth stating plainly:

- **Activity density**: anchoring the learner on the count-rate calibration takes test R²
  from 0.43 to 0.95. Trees applied to the raw target fail because the test session sits
  above the training range and a tree cannot predict outside it; the dimensionless ratio
  occupies the same range in every session, and multiplying the anchor back in supplies
  the extrapolation.
- **Volume**: a two-parameter affine calibration `V_CT ≈ 0.89·V_NM + 6.0` beats every
  79-feature model. Radiomics buys about one point of R² and only via a shrunk residual
  correction. This is a real finding, not a modelling failure — under the reduced feature
  set the NM→CT volume relationship is physics-limited, not model-limited.

## Data caveats

- Activity density is constant within an acquisition for 51 of 67 training scans, so the
  1766 rows carry roughly **67 independent observations** of that target. Any confidence
  interval must be computed at the scan level.
- The test session is shifted upward on both targets (activity density mean 6.76 vs 4.08;
  volume 19.2 vs 15.4 ml). Test scores measure extrapolation, not interpolation.
- Only 6 training sessions exist, so session-grouped CV has 6 effective folds and its
  variance is high. Treat differences below ~0.05 R² as noise.
- `Scale_factor_NM_40` varies over two orders of magnitude across sessions, but
  `Mean_Intensity_counts_NM_40` is already de-scaled — dividing by it *degrades* the
  count/activity correlation from 0.98 to 0.76. Do not "correct" for it.

`jelly_ml_functions.py` is the original submission, kept for reference; it is superseded
by `src/phantom_jelly/` and is not imported by anything.
