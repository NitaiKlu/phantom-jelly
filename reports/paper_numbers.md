# Manuscript numbers (generated)

Every quantity below is computed by this script from the CSVs / fitted models.

## Experimental inventory (full dataset)

|   session | split    | date       |   segments |   scans |   jars |   activity_min |   activity_max |
|----------:|:---------|:-----------|-----------:|--------:|-------:|---------------:|---------------:|
|         0 | held-out | 2025-09-02 |        334 |      13 |      4 |          3.656 |         10.23  |
|         1 | train    | 2025-05-20 |        386 |      11 |      3 |          5.526 |          7.263 |
|         2 | train    | 2025-06-10 |        327 |      13 |      3 |          3.366 |          4.734 |
|         3 | train    | 2025-07-01 |        190 |      10 |      3 |          3.045 |          3.963 |
|         4 | train    | 2025-07-08 |        583 |      15 |      4 |          0.594 |          0.902 |
|         5 | train    | 2025-07-21 |        482 |      13 |      4 |          1.746 |         11.777 |

- validated 1:1 segments: **2302**
- SPECT/CT scans: **75**
- jars (experimental log): **21**
- sessions: **6**
- activity density range: **0.59--11.78 µCi/ml**
- PyRadiomics binWidth: **[2000]**

## Modelling split (reduced NM-40 table)

- activity train/test rows: **1766** / **315**
- volume train/test rows: **1766** / **315**
- modelling features: **79**
- training acquisitions / sessions: **67** / **6**
- training scans with a single activity-density value: **51/67**

## Held-out test metrics

- Ratio+LightGBM activity: R²=0.949, RMSE=0.51 µCi/ml, MAPE=6.7%
- Physics k×counts activity: R²=0.927, RMSE=0.61 µCi/ml
- LightGBM on raw activity: R²=0.429, RMSE=1.72 µCi/ml
- Physics a·V_NM+b volume: R²=0.748, RMSE=3.78 ml, MAPE=14.5%
- fitted affine: V_CT = 0.893·V_NM + 6.00

## Figures

- `/workspace/paper/figures/fig3_activity.png`
- `/workspace/paper/figures/fig4_volume.png`
