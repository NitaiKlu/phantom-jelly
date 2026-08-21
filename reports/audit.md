# Data audit


## 1. Dataset shapes and group structure (reduced set)

- rows: train **1766**, test **315**
- unique acquisitions (NM image hash): train **67**, test **13**
- unique sessions (date): train **6**, test **1**
- acquisitions shared between train and test: **0**
- dates shared between train and test: **0**

The full set carries more rows (train 1968, test 334); the reduced set additionally drops the smallest ROIs (mean Volume_[ml]_CT of dropped rows ≈ 7.2 ml vs 15.4 ml for kept rows).

## 2. Target granularity — the effective sample size

| target                       |   rows |   unique values |   unique per scan (mean) | scans with 1 unique value   |   within-scan sd / overall sd |
|:-----------------------------|-------:|----------------:|-------------------------:|:----------------------------|------------------------------:|
| Volume_[ml]_CT               |   1766 |            1521 |                    26.3  | 0/67                        |                         0.959 |
| Activity_Density_uCiml_NM_40 |   1766 |              83 |                     1.24 | 51/67                       |                         0.269 |

`Activity_Density_uCiml_NM_40` is essentially a **per-acquisition constant**: 51 of 67 training scans carry a single value and the mean within-scan spread is 27% of the overall spread. The 1766 rows therefore contain roughly **67 independent observations** of this target, not 1766. `Volume_[ml]_CT` is a genuine per-ROI quantity (within-scan spread ≈ overall spread) and keeps its row-level sample size.

## 3. Distribution shift between train and test

| target                       |   train mean |   test mean |   train sd |   test sd | train range   | test range   |
|:-----------------------------|-------------:|------------:|-----------:|----------:|:--------------|:-------------|
| Volume_[ml]_CT               |       15.414 |      19.199 |      6.835 |     7.529 | 1.90 – 71.80  | 6.75 – 51.42 |
| Activity_Density_uCiml_NM_40 |        4.081 |       6.762 |      3.182 |     2.278 | 0.59 – 11.78  | 3.66 – 10.23 |

The supplied test file is a **separate acquisition session** (a single date, no shared scan hashes). Both targets are shifted upwards relative to training, so the test score measures extrapolation to a new session — the clinically relevant question, and a much harder one than a random row split.

## 4. Count-to-activity calibration is stable across sessions

| date         |   n |   scale_factor |    k_mean |   k_cv_pct |
|:-------------|----:|---------------:|----------:|-----------:|
| 01/07/2025   | 164 |           0.1  | 3.276e-05 |       12.8 |
| 08/07/2025   | 533 |           1    | 2.903e-05 |       17.1 |
| 10/06/2025   | 248 |           0.1  | 3.276e-05 |       11.5 |
| 10/08/2025   |  20 |           0.1  | 3.17e-05  |       31.2 |
| 20/05/2025   | 357 |           0.1  | 3.38e-05  |       13.4 |
| 21/07/2025   | 444 |           0.01 | 2.819e-05 |       11.3 |
| TEST SESSION | 315 |           0.1  | 2.899e-05 |        8.1 |

The ratio of activity density to mean NM counts sits near 3e-5 for every session including the unseen test session, despite `Scale_factor_NM_40` varying over two orders of magnitude between sessions. The counts column is already de-scaled, so a single proportionality constant transfers across sessions and makes a one-parameter physics model a strong baseline.

The often-quoted global correlation of r = 0.977 between counts and activity density is misleading: within a single session it falls to 0.45–0.55 for half the sessions. Most of that correlation is between-session variance.

## 5. CT-derived columns leak target 1

| CT-derived column                                |   |r| with Volume_[ml]_CT |
|:-------------------------------------------------|--------------------------:|
| diagnostics_Mask-original_VoxelNum_CT            |                    1      |
| original_shape_VoxelVolume_CT                    |                    1      |
| original_shape_MeshVolume_CT                     |                    0.9999 |
| original_gldm_GrayLevelNonUniformity_CT          |                    0.9858 |
| original_glrlm_GrayLevelNonUniformity_CT         |                    0.9789 |
| original_firstorder_Energy_CT                    |                    0.9783 |
| original_firstorder_TotalEnergy_CT               |                    0.9783 |
| original_gldm_DependenceNonUniformity_CT         |                    0.9742 |
| original_glrlm_RunLengthNonUniformity_CT         |                    0.9565 |
| original_glszm_SizeZoneNonUniformity_CT          |                    0.7872 |
| original_shape_SurfaceArea_CT                    |                    0.7664 |
| original_glszm_LargeAreaHighGrayLevelEmphasis_CT |                    0.7641 |

Every `*_CT` shape feature is computed from the same CT segmentation that defines `Volume_[ml]_CT`, so several of them reproduce the target almost exactly (`original_shape_VoxelVolume_CT` and `original_shape_MeshVolume_CT` are the target in mm³). The `*_to_*_overlap` columns also require the CT mask. None of these may enter a baseline model — which is precisely why the reduced set omits them.

## 6. Leakage cost of a random row split

The same model, the same features, the same data — only the split changes.


**Volume_[ml]_CT**

| split                 |    r2 |   rmse |   mae |   mape |    n |
|:----------------------|------:|-------:|------:|-------:|-----:|
| random KFold (leaky)  | 0.833 |  2.79  | 1.674 | 12.305 | 1766 |
| GroupKFold by scan    | 0.843 |  2.711 | 1.664 | 12.147 | 1766 |
| GroupKFold by session | 0.405 |  5.272 | 3.75  | 26.891 | 1766 |

**Activity_Density_uCiml_NM_40**

| split                 |    r2 |   rmse |   mae |    mape |    n |
|:----------------------|------:|-------:|------:|--------:|-----:|
| random KFold (leaky)  | 0.999 |  0.08  | 0.023 |   0.583 | 1766 |
| GroupKFold by scan    | 0.975 |  0.506 | 0.307 |   8.182 | 1766 |
| GroupKFold by session | 0.497 |  2.256 | 1.987 | 141.524 | 1766 |

Random `train_test_split` / `KFold` on rows places ROIs from the *same* acquisition on both sides of the split. Because activity density is constant within an acquisition, the model only has to recognise which scan a row came from — trivial, given that radiomics features are near-identical within a scan. For activity density that inflates R² to 0.999, against 0.975 once scans are kept together and 0.497 once whole sessions are held out.

Volume is a genuine per-ROI quantity, so it barely moves between the random and scan-grouped splits (0.833 vs 0.843); it is the *session*-grouped split (0.405) that reveals its true generalisation gap. Session-grouped CV is therefore the estimate used for model selection everywhere else in this repository.

## 7. Feature selection audit (reduced set, target 1)

```
kept 79 features
  dropped   2  (target)
  dropped   3  (timestamp / acquisition key)
  dropped  10  (pyradiomics bookkeeping)
  dropped   6  (absolute position in phantom)
  dropped   3  (non-numeric)
  dropped  12  (constant)
  dropped  41  (|r| > 0.995 with an earlier feature)
```