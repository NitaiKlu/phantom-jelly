# Code and methodology audit: `jelly_ml_functions.py`

Audit of the original submission for the jelly-phantom SPECT/CT study, with the
corresponding fixes in the replacement package `src/phantom_jelly/`.

Targets under study:

- `Volume_[ml]_CT` — insert volume from the CT segmentation
- `Activity_Density_uCiml_NM_40` — activity concentration in µCi/ml

Every quantitative claim below was measured on the supplied data. The commands that
reproduce each measurement are listed in [Appendix A](#appendix-a-reproducing-the-evidence).
Line numbers refer to the original `jelly_ml_functions.py` (351 lines, 6 public functions).

---

## Summary

| # | Issue | Severity | Effect |
|---|---|---|---|
| [1](#1-the-validation-split-ignores-the-nested-structure-of-the-data) | Validation split ignores the nested data structure | **Critical** | Activity-density R² inflated from 0.497 to 0.999 |
| [2](#2-no-protection-against-ct-derived-features) | No guard against CT-derived features | **Critical** | 9 columns correlate \|r\| ≥ 0.95 with target 1; two at \|r\| = 1.000 |
| [3](#3-hyperparameter-selection-leaks-through-the-inner-cv) | Hyperparameter selection leaks through the inner CV | **Critical** | `ElasticNetCV(cv=5)` selects α on contaminated folds |
| [4](#4-the-elasticnet-weight-vectors-are-misaligned-and-the-functions-raise) | ElasticNet weight vectors misaligned; functions raise | **High** | `ValueError`; both ElasticNet functions cannot return |
| [5](#5-plot_linear_regression-cannot-return) | `plot_linear_regression` cannot return | **High** | `TypeError` on the final line |
| [6](#6-a-training-score-is-reported-as-the-model-score) | A training score is reported as the model score | **High** | Mislabeled metric; misleading for any flexible model |
| [7](#7-mse_low-is-computed-over-a-vacuous-mask) | `mse_low` computed over a vacuous mask | **Medium** | Mask is all-`True`; reported subgroup metric is the global one |
| [8](#8-random_forest_volume-discards-its-own-result) | `random_forest_Volume` discards its own result | **Medium** | Feature importances computed, never returned |
| [9](#9-variables-named-mse-hold-absolute-error) | Variables named `mse` hold absolute error | **Medium** | Axis labelled "MSE (%)" shows neither MSE nor a percentage |
| [10](#10-no-random-seeds-anywhere) | No random seeds anywhere | **Medium** | Reported test R² varies by 0.24 between identical runs |
| [11](#11-elasticnet-search-grid-is-both-invalid-and-wasteful) | ElasticNet search grid invalid and wasteful | **Medium** | `l1_ratio = 0` silently accepted; 12,500 fits for 2 parameters |
| [12](#12-a-commented-out-block-fits-a-scaler-on-the-test-set) | Commented-out block fits a scaler on test data | **Low** | Dead code, but documents the intended (incorrect) approach |
| [13](#13-the-module-cannot-be-imported) | The module cannot be imported | **Low** | Fails at line 11; 33 of 43 imported names unused |
| [14](#14-impurity-based-feature-importances-on-correlated-radiomics) | Impurity-based importances on correlated radiomics | **Low** | Systematically biased rankings |
| [15](#15-plotting-defects) | Plotting defects | **Low** | Mismatched axes, unsorted line plots |

---

## What the original code got right

Two decisions in the original are sound and were kept.

`elasticNet_Volume` and `elasticNet_activityDensity` fit the scaler on training data only
and apply it to both splits (lines 260-264) — correct, and a genuine improvement over the
commented-out block above it (issue 12).

More importantly, `linearRegression_fo_randomForest_so_activity_density` (lines 146-211)
implements a **two-stage physics-then-residual model**: a linear fit on the single most
physically meaningful predictor, followed by a random forest on what the linear fit leaves
over.

```python
# lines 147-162
    # === שלב 1: רגרסיה ליניארית על הפיצ'ר הראשי ===
    model_lin = LinearRegression()
    model_lin.fit(X_train[[main_feature]], y_train)
    ...
    # === שלב 2: חישוב השאריות ===
    residuals_train = y_train - y_train_lin
    ...
    model_rf = RandomForestRegressor(n_estimators=n)
    model_rf.fit(X_train_rf, residuals_train)
```

This is the right instinct for a dataset with ~67 independent observations: constrain the
leading term by physics and spend the limited statistical power on the correction. It is
the basis of `models.PhysicsResidualRegressor`, which generalises it in three ways — the
anchor becomes a pluggable calibration (through the origin for activity density, affine for
volume), the residual is shrunk to control session-specific overfitting, and a
multiplicative variant (`PhysicsRatioRegressor`) is added for the extrapolation reasons
discussed at the end of this document. That family produces the best results reported here.

---

## Critical: validation and leakage

### 1. The validation split ignores the nested structure of the data

Each row is one region of interest (ROI); many ROIs come from the same acquisition. The
two targets do **not** live at the same granularity, and this is the single most
consequential fact about the dataset:

| Target | Distinct values in 1766 rows | Distinct values per scan (mean) | Scans with exactly 1 value | Within-scan SD ÷ overall SD |
|---|---|---|---|---|
| `Volume_[ml]_CT` | 1521 | 26.3 | 0 / 67 | 0.959 |
| `Activity_Density_uCiml_NM_40` | 83 | 1.24 | **51 / 67** | **0.269** |

`Activity_Density_uCiml_NM_40` is essentially a **per-acquisition constant**. The 1766
training rows therefore contain approximately **67 independent observations** of that
target, not 1766. This is expected physically — a single activity concentration is
prepared per acquisition — but it has a severe statistical consequence.

The original code splits on rows:

```python
# lines 44-45, plot_linear_regression
    # Split into training and testing
    train_df, test_df = train_test_split(df_clean, test_size=test_size)
```

The remaining five functions receive `X_train, X_test, y_train, y_test` already split by
the caller, so the same row-wise split propagates through the whole pipeline:

```python
# line 114
def random_forest_activity_density(X_train, X_test, y_train, y_test, n=100):
# line 214
def random_forest_Volume(X_train, X_test, y_train, y_test):
# line 244
def elasticNet_Volume(X_train, X_test, y_train, y_test, alpha_values=None, ...):
```

A row-wise split places ROIs from the same acquisition on **both** sides of the split.
Because radiomics features are nearly identical between ROIs of one acquisition, the model
only has to recognise *which scan a row came from* in order to recover a per-scan-constant
target. It never has to learn the underlying physics.

Measured cost — identical model, identical features, only the split changes:

| Split | Activity density R² | Volume R² |
|---|---|---|
| random `KFold` (as in the original) | **0.999** | 0.833 |
| `GroupKFold` by acquisition | 0.975 | 0.843 |
| `GroupKFold` by session (date) | **0.497** | **0.405** |

An R² of 0.999 on activity density is not a result; it is an artifact of the split. Note
that volume, being a genuine per-ROI quantity, barely moves between the first two rows —
it is the session-level split that reveals its true generalisation gap.

**Fix.** Three splitters in `src/phantom_jelly/evaluation.py`, with grouping keys carried
alongside the data in `data.Dataset`. The random splitter is retained **only** as a
diagnostic, to quantify the bias above:

```python
def make_cv(scheme: str, groups: pd.Series, n_splits: int = 5):
    """``"random"`` exists only to quantify how badly row-wise splitting inflates
    scores on this dataset; never use it to select a model."""
    if scheme == "random":
        return KFold(n_splits=n_splits, shuffle=True, random_state=cfg.RANDOM_STATE)
    if scheme == "group":
        return GroupKFold(n_splits=min(n_splits, groups.nunique()))
```

Model selection uses **session-grouped** CV, grouping on acquisition date. Two reasons:
the supplied test file is a single unseen session (13 acquisitions on one date, zero scan
or date overlap with training), and empirically it is the only scheme whose ranking agrees
with held-out performance.

Scan-grouped CV is *not* sufficient for model selection, even though it removes the worst
of the leak. Ranking the 13 candidate models for activity density by scan-grouped CV and
comparing against their actual test-session scores:

| Rank by scan-CV | Model | scan-CV R² | test R² |
|---|---|---|---|
| 1 | MLP | 0.9887 | 0.821 |
| 3 | LightGBM | 0.9823 | **0.429** |
| 7 | HistGBM | 0.9787 | **0.259** |
| 11 | Ratio+Ridge | 0.9731 | **0.950** |
| 12 | ElasticNet | 0.9686 | 0.943 |
| 13 | physics only, `k·counts` | 0.9487 | 0.928 |

The ordering is close to *inverted*. Scan-grouped CV puts the MLP first and the
one-parameter physics model last, spanning a range of only 0.04 R², while the true test
spread across the same models is 0.69. The models it ranks 11th, 12th and 13th are the
three that actually generalise. Session-grouped CV recovers the correct ordering
(`reports/results.md`), because it is the only scheme that forces a model to extrapolate
to acquisition conditions it has never seen.

> **Caveat to state in any write-up.** Only 6 training sessions exist, so session-grouped
> CV has at most 6 effective folds and high variance. Differences below roughly 0.05 R²
> should be treated as noise. Any confidence interval for activity density must be
> computed at the scan level (n ≈ 67), not the row level (n = 1766).

### 2. No protection against CT-derived features

Nothing in the original code distinguishes CT-derived columns from NM-derived ones. Every
function accepts whatever `X_train` the caller supplies. Applied to the full dataset
(`df_train.csv`, 390 columns) this is fatal for target 1, because CT shape features are
computed from the *same segmentation mask* that defines `Volume_[ml]_CT`:

| CT-derived column | \|r\| with `Volume_[ml]_CT` |
|---|---|
| `diagnostics_Mask-original_VoxelNum_CT` | 1.0000 |
| `original_shape_VoxelVolume_CT` | 1.0000 |
| `original_shape_MeshVolume_CT` | 0.9999 |
| `original_gldm_GrayLevelNonUniformity_CT` | 0.9858 |
| `original_glrlm_GrayLevelNonUniformity_CT` | 0.9789 |
| `original_firstorder_Energy_CT` | 0.9783 |
| `original_firstorder_TotalEnergy_CT` | 0.9783 |
| `original_gldm_DependenceNonUniformity_CT` | 0.9742 |
| `original_glrlm_RunLengthNonUniformity_CT` | 0.9565 |

`original_shape_VoxelVolume_CT` *is* the target, expressed in mm³. The
`CT_to_NM40_overlap` / `CT_to_NM80_overlap` columns are equally inadmissible for a
different reason: they are computed against the CT mask, which is unavailable at inference
time in the intended application.

**Fix.** A leakage blacklist in `config.py`, enforced by `data.select_features`, which
defaults to `allow_ct_radiomics=False`. Selection returns an audit trail recording why
each column was removed, so the exclusion is inspectable rather than implicit:

```
kept 79 features
  dropped   2  (target)
  dropped   3  (timestamp / acquisition key)
  dropped  10  (pyradiomics bookkeeping)
  dropped   6  (absolute position in phantom)
  dropped   3  (non-numeric)
  dropped  12  (constant)
  dropped  40  (|r| > 0.995 with an earlier feature)
```

Three exclusion categories deserve comment, as they are not obvious:

- **Timestamps.** `CT_Scan_Time` and `NM_Scan_Time` have exactly 67 distinct values across
  67 acquisitions — they are unique acquisition identifiers. Combined with a
  per-scan-constant target they act as a lookup key, not a feature.
- **Absolute position.** `diagnostics_Mask-original_CenterOfMass*` and the first three
  elements of the bounding box encode position inside the phantom. Since the phantom
  layout is fixed, these let a model memorise which insert sits where rather than learn
  imaging physics. The bounding-box *extent* (last three elements) is retained.
- **Segmentation label IDs** (`NM_40`, `NM_80`, `CT`, `Label_CT`) are arbitrary
  connected-component indices.

CT radiomics may re-enter only through the conditional exception: a surrogate trained on
NM-only inputs, validated to cross-validated R² > 0.85, and screened for circularity
(`src/phantom_jelly/surrogate.py`). That screen is essential — a surrogate for
`original_shape_VoxelVolume_CT` would simply be the volume model in disguise, and feeding
it back would make the pipeline appear self-consistent while adding no information.

### 3. Hyperparameter selection leaks through the inner CV

```python
# line 268, elasticNet_Volume  (identical at line 319, elasticNet_activityDensity)
    model = ElasticNetCV(alphas=alpha_values, l1_ratio=l1_ratio_vals, cv=5, random_state=random_state)
```

`cv=5` is an unstratified, ungrouped `KFold`. Even had the outer split been corrected,
α and `l1_ratio` would still be chosen on folds contaminated in exactly the manner of
issue 1 — the selected regularisation strength is tuned to a leaky objective and will be
systematically too weak.

**Fix.** All hyperparameter search passes group labels to a `GroupKFold`. Regularised
linear models are expressed as explicit pipelines whose search spaces live in
`models.search_space`, so the splitter is always supplied by the caller and never hidden
inside the estimator.

---

## High: bugs that crash or mislead

### 4. The ElasticNet weight vectors are misaligned and the functions raise

Both ElasticNet functions fit the model on the **full** feature matrix but build the
coefficient index from a **de-duplicated subset**:

```python
# lines 260-273, elasticNet_Volume
    scaler = StandardScaler()
    scaler.fit(X_train)  # מחשב ממוצע וסטיית תקן רק על train

    X_scaled_train = scaler.transform(X_train)
    X_scaled_test = scaler.transform(X_test)  #
    X_numeric=X_train.select_dtypes(include=[np.number])
    X_encoded = X_numeric.loc[:, ~X_numeric.T.duplicated()]#משתמש באותם פרמטרים
    feature_names = X_encoded.columns.tolist()
    model = ElasticNetCV(alphas=alpha_values, l1_ratio=l1_ratio_vals, cv=5, random_state=random_state)
    model.fit(X_scaled_train, y_train)
    y_pred = model.predict(X_scaled_test)
    weights = pd.Series(model.coef_/ scaler.scale_, index=feature_names)
```

`model.fit` receives `X_scaled_train`, derived from all of `X_train`. But
`feature_names` comes from `X_encoded`, which has had duplicate-valued columns removed.
The two lengths differ whenever any duplicate exists.

On the reduced dataset's numeric feature block this is not hypothetical:

```
X_train numeric block: 108 columns
feature_names length:  107
model.coef_ length:    108
-> ValueError: Length of values (108) does not match length of index (107)
```

The single duplicated column is `Mean_Intensity_counts_NM_40`, an **exact** copy of
`original_firstorder_Mean_NM_40` (maximum absolute difference 0.0). Two consequences:

1. Both ElasticNet functions raise `ValueError` on this dataset and cannot return.
2. The column that de-duplication removes is the most physically meaningful feature in
   the set — the ROI mean count rate, which is the direct physical predictor of activity
   concentration. Had the lengths happened to match, every coefficient after that position
   would have been reported against the wrong feature name.

The same `.T.duplicated()` idiom appears in the dead block at lines 252-257, so the
misalignment was present in both the old and the rewritten version of this function.

**Fix.** De-duplication is a feature-selection step performed *before* fitting, so one
column list governs the matrix, the scaler and the coefficient index. Redundancy removal
uses a correlation threshold rather than exact equality, since radiomics packs contain
many near-duplicates as well as exact ones, and the physically interpretable columns are
protected so a near-duplicate can never displace them:

```python
# src/phantom_jelly/data.py
PROTECTED_FEATURES = (cfg.NM_COUNTS_COLUMN, cfg.NM_VOLUME_COLUMN)

    if corr_threshold is not None and len(candidates) > 1:
        # Order protected columns first so the filter keeps them and drops their duplicates.
        ordered = [c for c in protected if c in candidates]
        ordered += [c for c in candidates if c not in ordered]
        corr = train[ordered].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        redundant = [
            c for c in upper.columns if c not in protected and (upper[c] > corr_threshold).any()
        ]
```

Coefficient recovery in original units is handled once, inside the pipeline, rather than
re-derived by hand in each function.

### 5. `plot_linear_regression` cannot return

```python
# lines 108-111
    weights = pd.Series(model.coef_, index=y_col)
    intercept_unscaled = model.intercept_
    weights['__intercept__'] = intercept_unscaled
    return weights
```

`y_col` is the *name of the target column*, a string — not a list of feature names. The
model has one feature, so `model.coef_` has length 1, while pandas attempts to build an
index from the string:

```
TypeError: Index(...) must be called with a collection of some kind,
           'Volume_[ml]_CT' was passed
```

The function raises after producing its figure, so it always returns `None` in effect.
The intended expression is almost certainly `index=[x_col]` — the *predictor* name.

**Fix.** Coefficient extraction is not the responsibility of a plotting routine. Fitting,
metric computation and figure generation are separated in the new package;
`ProportionalCalibration` and `AffineCalibration` expose fitted parameters as ordinary
attributes (`k_`, `a_`, `b_`).

### 6. A training score is reported as the model score

```python
# lines 57-66
    # Predict on train and test
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    ...
    # Evaluate
    slope = model.coef_[0]
    intercept = model.intercept_
    r2 = r2_score(y_train, y_pred_train)
```

`y_pred_test` is computed but never used for scoring. The `r2` that reaches the annotation
box at line 98 is a **training** score, presented without qualification as the model's R².

To be precise about the magnitude: for this specific one-feature linear fit, train and
test R² are close, and the test value is sometimes higher —

```
seed 0: reported(train) R2=0.7311   actual(test) R2=0.6750
seed 1: reported(train) R2=0.7096   actual(test) R2=0.7752
seed 2: reported(train) R2=0.7138   actual(test) R2=0.7452
seed 3: reported(train) R2=0.7304   actual(test) R2=0.6625
seed 4: reported(train) R2=0.7088   actual(test) R2=0.7569
```

So the practical distortion is small *here*. The defect is nonetheless serious: the
reported quantity is not the one its label claims, and the same pattern applied to any
higher-capacity model (the random forests in this file reach training R² above 0.98) would
be grossly optimistic. It also makes the deliberate train/test split pointless.

**Fix.** `evaluation.compute_metrics` is only ever called on held-out predictions —
out-of-fold predictions from `cross_val_predict` for CV scores, and the untouched test
session for final scores. Reported dictionaries name the split explicitly
(`cv_scan_r2`, `cv_session_r2`, `test_r2`).

### 7. `mse_low` is computed over a vacuous mask

```python
# line 43
    df_clean = df_clean[df_clean[x_col] <upper_threshold]
...
# lines 68-69
    low_mask = test_df[x_col] < upper_threshold  # לדוגמה
    mse_low = mean_squared_error(y_test[low_mask], y_pred_test[low_mask])
```

`df_clean` is filtered to `x_col < upper_threshold` at line 43, *before* the split at line
45. Every row of `test_df` therefore already satisfies the same condition, and `low_mask`
is all-`True` (verified: 354 of 354 rows retained). `mse_low` is the global test MSE, but
it is displayed as `mse (<{upper_threshold})`, implying a low-value subgroup that was
never actually isolated.

**Fix.** Stratified error reporting is done by explicit quantile binning of the target
after evaluation, not by a filter that duplicates an earlier one.

---

## Medium: correctness and reproducibility

### 8. `random_forest_Volume` discards its own result

```python
# lines 218-220
    importances = model.feature_importances_
    features = X_train.columns
    iif=pd.DataFrame([importances,features]).T
```

The function has no `return` statement (compare `random_forest_activity_density`, which
returns `iif` at line 142). The importances are computed and thrown away.

Separately, `pd.DataFrame([importances, features]).T` builds a 2-row frame and transposes
it, yielding columns named `0` and `1` and an `object` dtype for the importances. The
sibling function uses the correct idiom:

```python
# line 120
    iif=pd.DataFrame({'importances':importances,'features':features}).sort_values(by='importances', ascending=False)
```

**Fix.** Model fitting returns metrics as structured records (`evaluation.evaluate`);
importance extraction is separate and typed.

### 9. Variables named `mse` hold absolute error

```python
# lines 61-62
    mse_train=np.abs(y_pred_train-y_train)
    mse_test=np.abs(y_pred_test-y_test)
```

These are per-sample absolute errors, not mean squared errors. They are then divided by
the predictor and plotted under a percentage label:

```python
# lines 80-87
    ax2.plot(X_train.ravel()[sorted_idx_train],
    (mse_train / X_train.ravel())[sorted_idx_train],
    color='blue', marker='x', label='Training MSE per X')
    ...
    ax2.set_ylabel('MSE (%)')
```

The plotted quantity is |error| ÷ *x*. It is not MSE, and it is not a percentage — a
percentage would require division by *y* and multiplication by 100. Note also that
dividing the error by the predictor rather than the target makes the quantity
uninterpretable as a relative error.

**Fix.** `evaluation.Metrics` computes and names RMSE, MAE and MAPE separately, with MAPE
guarded against near-zero denominators.

### 10. No random seeds anywhere

```python
# line 45  — no random_state
    train_df, test_df = train_test_split(df_clean, test_size=test_size)
# line 115 — no random_state
    model = RandomForestRegressor(n_estimators=n)
# line 160 — no random_state
    model_rf = RandomForestRegressor(n_estimators=n)
# line 215 — no random_state
    model = RandomForestRegressor(n_estimators=100)
```

`plot_linear_regression` accepts no `random_state` parameter at all, so its result cannot
be reproduced even in principle. Over 30 identical unseeded runs of that function's split
and fit, the test R² ranged from **0.585 to 0.821** — a spread of 0.236, which is larger
than most of the model-to-model differences the study is trying to resolve.

**Fix.** A single `config.RANDOM_STATE = 42` is threaded through every splitter and
estimator.

### 11. ElasticNet search grid is both invalid and wasteful

```python
# lines 246-249
    if alpha_values is None:
        alpha_values = np.logspace(-4, 0, 50)
    if l1_ratio_vals is None:
        l1_ratio_vals = np.linspace(0, 1,50)[0:]
```

Three problems:

- `np.linspace(0, 1, 50)` includes `l1_ratio = 0`, which is pure L2 and outside the valid
  domain for `ElasticNet`. scikit-learn accepts it **silently** rather than raising —
  verified — so the invalid point contributes to selection without any warning surfacing.
- 50 alphas × 50 `l1_ratio` values × 5 folds = 12,500 fits to tune two parameters. A
  log-spaced grid of ~10 × 5 is statistically indistinguishable here, particularly given
  only 67 independent observations.
- `[0:]` is a no-op slice, suggesting an edit (probably `[1:]`, which would have excluded
  the invalid zero) that was reverted and left behind.

**Fix.** `models.search_space` uses `l1_ratio` ∈ [0.05, 1.0] and a compact log-spaced
α grid, searched under `GroupKFold`. Pure L2 is available as an explicit `Ridge` entry in
the zoo rather than as a degenerate corner of the ElasticNet grid.

### 12. A commented-out block fits a scaler on the test set

```python
# lines 251-258
    scaler = StandardScaler()
#    X_numeric=X_train.select_dtypes(include=[np.number])
#    X_encoded = X_numeric.loc[:, ~X_numeric.T.duplicated()]
#    feature_names = X_encoded.columns.tolist()
#    X_scaled = scaler.transform(X_encoded)
#    X_numeric1=X_test.select_dtypes(include=[np.number])
#    X_encoded1 = X_numeric1.loc[:, ~X_numeric1.T.duplicated()]
#    X_scaled_test = scaler.fit_transform(X_encoded1)
```

This block calls `scaler.transform` before any `fit` (which would raise
`NotFittedError`), and then `scaler.fit_transform(X_test)` — fitting normalisation
statistics on the test set, and fitting them *separately* from the training set, so the
two matrices end up in different coordinate systems.

The block is dead, and the live code below it is correct in this respect (`scaler.fit`
on train only, `transform` on both). It is recorded here because it documents the
originally intended approach, and because the surrounding live code still performs
scaling manually — outside any cross-validation loop — which is the pattern that makes
this class of error easy to reintroduce.

**Fix.** Scaling is never manual. Every estimator is a `Pipeline` beginning with
imputation and, where relevant, scaling, so all preprocessing statistics are fitted inside
each CV fold by construction:

```python
# src/phantom_jelly/models.py
def _scaled(estimator) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", estimator),
        ]
    )
```

This makes the error structurally impossible rather than merely absent.

---

## Low: hygiene and presentation

### 13. The module cannot be imported

Lines 7-36 import **43 names**, of which an AST analysis confirms **33 are never used**.
The 10 that are actually used are `np`, `pd`, `plt`, `train_test_split`,
`LinearRegression`, `ElasticNetCV`, `RandomForestRegressor`, `StandardScaler`,
`r2_score` and `mean_squared_error`.

The 33 unused names are: `Calendar`, `ElasticNet`, `KFold`, `OneHotEncoder`, `Tk`,
`VarianceThreshold`, `ast`, `chi2_contingency`, `curve_fit`, `date`, `datetime`,
`dendrogram`, `deque`, `fcluster`, `featureextractor`, `filedialog`,
`generate_binary_structure`, `getTestCase`, `itertools`, `label`, `linkage`,
`maximum_position`, `measure`, `ndi_sum`, `nrrd`, `os`, `pydicom`, `resize`,
`simpledialog`, `sns`, `tk`, `ttest_ind`, `ttk`.

This matters because five of the imported *packages* are unavailable on a current
interpreter, so the module cannot be imported at all — it fails at line 11, before
reaching any function definition:

```
File "jelly_ml_functions.py", line 11, in <module>
    from skimage import measure
ModuleNotFoundError: No module named 'skimage'
```

The unavailable packages are `skimage` (lines 11, 15), `pydicom` (13), `nrrd` (16),
`tkcalendar` (20) and `radiomics` (21) — **every one of them unused**. `radiomics` is the
hard constraint: PyRadiomics publishes no wheels for Python ≥ 3.10, so this import list
pins the entire analysis to Python 3.8/3.9 in exchange for nothing. The GUI imports
(`tkinter`, `ttk`, `Calendar`, `filedialog`, `simpledialog`) additionally make the module
unimportable in any headless environment such as a compute cluster or CI runner.

There is also no dependency manifest, so the working environment cannot be reconstructed.

**Fix.** Modelling code depends only on numpy, pandas, scipy, scikit-learn, LightGBM and
XGBoost, pinned in `requirements.txt`. The image-processing stack is documented there as
optional, needed only to regenerate features from raw DICOM/NRRD, with a note that it
requires a separate Python 3.8/3.9 environment. The new package imports cleanly on
Python 3.12.

### 14. Impurity-based feature importances on correlated radiomics

```python
# lines 118-120
    importances = model.feature_importances_
    features = X_train.columns
    iif=pd.DataFrame({'importances':importances,'features':features}).sort_values(...)
```

Mean-decrease-in-impurity importances are biased toward high-cardinality continuous
features and split arbitrarily among correlated ones. A radiomics pack is the worst case
for both: the feature set here contains an exact duplicate pair (issue 4) and 40 columns
correlated above |r| = 0.995. Rankings from this method are not interpretable as physical
importance, and are unstable across reruns given issue 10.

**Fix.** Redundancy is removed before fitting. Where importance is reported it is
permutation importance, computed on held-out folds, which is defined with respect to
predictive performance rather than tree structure.

### 15. Plotting defects

**Mismatched axis limits with an identity line.** In `random_forest_Volume`:

```python
# lines 230-237
    max_t=max(y_test)
    max_p=max(y_pred)
    ax2.plot(y_test, y_pred,'.r')
    ax2.plot([0, max_p], [0, max_p],'--r')
    ...
    ax2.set_xlim(0, max_t*1.1)
    ax2.set_ylim(0, max_p*1.1)
```

The identity line spans `[0, max_p]` on both axes, but the axes are scaled by *different*
maxima (`max_t` for x, `max_p` for y). The reference line therefore does not render at
45°, and any visual read of bias against it is wrong. The same pattern appears at lines
135-136 and 197-198.

**Unsorted line plot.** In `linearRegression_fo_randomForest_so_activity_density`:

```python
# line 185
    ax1.plot(X_test[main_feature], y_test_lin, 'k-', label='Linear fit', alpha=0.7)
```

`X_test[main_feature]` is in row order, not sorted, so a solid line connecting successive
points renders as a zig-zag rather than the intended straight fit. Lines 77-86 of
`plot_linear_regression` show the author was aware of this and applied `np.argsort` there.

**Fix.** Parity plots use shared, symmetric limits with a true 45° reference; line
overlays are sorted on the abscissa.

---

## What the corrected pipeline shows

Rebuilt under session-grouped validation with CT features excluded, on the reduced feature
set (79 features, 1766 rows, 67 acquisitions, 6 sessions). Test columns refer to the
supplied test file — one acquisition session, no scans or dates shared with training.

| Target | Model | session-CV R² | **test R²** | test RMSE | test MAPE |
|---|---|---|---|---|---|
| Activity density | LightGBM on raw target *(baseline)* | 0.578 | 0.429 | 1.72 | 23.8% |
| Activity density | physics only, `k·counts` | 0.924 | 0.928 | 0.61 | 7.2% |
| Activity density | **Ratio + LightGBM** | **0.934** | **0.949** | **0.51** | 6.7% |
| Volume | LightGBM on raw target *(baseline)* | 0.406 | 0.722 | 3.96 | 15.6% |
| Volume | **physics only, `a·V_NM + b`** | **0.701** | **0.748** | **3.78** | 14.5% |
| Volume | Residual + GBM, shrinkage 0.3 | 0.674 | 0.758 | 3.70 | 14.4% |

Three findings follow.

**Activity density is a calibration problem.** The ratio of activity density to mean NM
count rate lies between 2.8×10⁻⁵ and 3.4×10⁻⁵ for all six training sessions *and* for the
unseen test session (2.899×10⁻⁵), even though `Scale_factor_NM_40` varies over two orders
of magnitude between sessions. A one-parameter model through the origin therefore reaches
test R² = 0.928 on its own; machine learning adds 0.02 on top of it.

A related caution: the global correlation between mean counts and activity density is
r = 0.977, but *within* a single session it falls to 0.45–0.55 for half the sessions. Most
of that global correlation is between-session variance, so it should not be quoted as
evidence of a within-session predictive relationship. Note also that
`Mean_Intensity_counts_NM_40` is already de-scaled — dividing it by `Scale_factor_NM_40`
*degrades* the correlation with activity density from 0.977 to 0.758.

**Raw-target tree ensembles fail on this data, and the reason is structural.** The test
session lies above the training range on both targets (activity density mean 6.76 vs 4.08;
volume 19.2 vs 15.4 ml). A tree can only output values it observed in training, so it is
capped below what it needs to predict. Re-expressing the problem as a *dimensionless
ratio* to an NM anchor removes the ceiling — the ratio occupies the same range in every
session, and multiplying the anchor back in supplies the extrapolation. This takes
LightGBM from test R² 0.429 to 0.949, and is implemented as
`models.PhysicsRatioRegressor`.

**Volume is physics-limited rather than model-limited.** A two-parameter affine
calibration, `V_CT ≈ 0.89·V_NM + 6.0`, outperforms every 79-feature model. Radiomics buys
roughly one point of R², and only through a heavily shrunk residual correction. Restricting
the residual model to shape features only — on the hypothesis that texture carries
session-specific count noise — made it worse (test R² 0.731 vs 0.758). Under the reduced
feature set this relationship appears close to its ceiling.

### Known limitations of the corrected pipeline

- 6 training sessions gives session-grouped CV at most 6 folds; its variance is high.
  Several rows in `reports/results.md` show session-CV values of −118, −975 and −7058,
  which are single pathological folds rather than meaningful scores.
- The 0.949 vs 0.928 gap between the anchored model and the pure physics baseline has not
  been shown to be statistically significant. A per-scan bootstrap is needed.
- The conditional CT exception (`scripts/run_surrogate.py`) is implemented but has not yet
  been executed, so no CT feature is currently approved for use.
- `PhysicsRatioRegressor` and `PhysicsResidualRegressor` are new code and are not yet
  covered by unit tests.

---

## Appendix A: reproducing the evidence

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/audit_data.py       # issues 1, 2 -> reports/audit.md
python scripts/run_experiments.py  # results table -> reports/results.md
```

`reports/audit.md` section 2 reproduces the target-granularity table (issue 1), section 5
the CT correlation table (issue 2), and section 6 the three-way split comparison.

The bug reproductions in issues 4, 5, 6, 7, 10, 11 and 13 were produced by driving the
original functions' code paths directly against `data/Jelly_Phantom/train.csv`: the
duplicate-column identity check and the `ValueError`, the `TypeError` from
`pd.Series(model.coef_, index=y_col)`, the train-versus-test R² table, the all-`True`
`low_mask`, the 30-run unseeded R² spread, `ElasticNetCV` silently accepting
`l1_ratio = 0`, and `import jelly_ml_functions` failing at line 11.

The unused-import count in issue 13 comes from an AST pass over the original file,
comparing every imported binding against the set of `Name` and `Attribute` nodes in the
module body.

## Appendix B: file map

| Original | Replacement |
|---|---|
| `plot_linear_regression` | `models.AffineCalibration` + `evaluation.evaluate` |
| `random_forest_activity_density` | `models.model_zoo` (`RandomForest`) |
| `linearRegression_fo_randomForest_so_activity_density` | `models.PhysicsResidualRegressor` (generalised, with shrinkage) |
| `random_forest_Volume` | `models.model_zoo` (`RandomForest`) |
| `elasticNet_Volume` | `models.model_zoo` (`ElasticNet`) + `models.search_space` |
| `elasticNet_activityDensity` | `models.model_zoo` (`ElasticNet`) + `models.search_space` |
| — | `data.select_features` (leakage blacklist, audit trail) |
| — | `evaluation.leakage_report` (quantifies issue 1) |
| — | `models.PhysicsRatioRegressor` (extrapolation fix) |
| — | `surrogate` (conditional CT exception) |

The original file is retained unmodified at `jelly_ml_functions.py` for reference and is
imported by nothing.
