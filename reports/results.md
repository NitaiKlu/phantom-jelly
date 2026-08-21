# Model benchmark

Dataset variant: **reduced** (no CT-derived features).

`cv_scan_r2` groups folds by acquisition; `cv_session_r2` groups by acquisition date and is the honest analogue of the held-out test session. `test_*` columns are measured on the supplied test file, an acquisition session never seen during training or model selection.


## Volume_[ml]_CT

1766 training rows / 67 acquisitions / 6 sessions, 79 features.

| model                 |   cv_scan_r2 |   cv_session_r2 |   cv_session_rmse |   test_r2 |   test_rmse |   test_mae |   test_mape |
|:----------------------|-------------:|----------------:|------------------:|----------:|------------:|-----------:|------------:|
| Physics: a x V_NM + b |       0.7204 |          0.7007 |            3.7382 |    0.7477 |      3.7754 |     2.8022 |     14.4958 |
| Ratio+LightGBM        |       0.8877 |          0.3087 |            5.6814 |    0.7339 |      3.8777 |     2.8611 |     14.7545 |
| LightGBM              |       0.8913 |          0.4062 |            5.2657 |    0.7221 |      3.9625 |     3.0569 |     15.6323 |
| HistGBM               |       0.8749 |          0.3515 |            5.503  |    0.7173 |      3.9968 |     2.918  |     14.661  |
| XGBoost               |       0.8812 |          0.4106 |            5.2462 |    0.6952 |      4.15   |     3.0544 |     15.4719 |
| ElasticNet            |       0.8104 |        -10.7439 |           23.4173 |    0.6726 |      4.3008 |     3.3528 |     18.4586 |
| Ratio+HistGBM         |       0.8646 |          0.2998 |            5.718  |    0.6571 |      4.4015 |     3.2308 |     16.7329 |
| Ratio+RandomForest    |       0.8499 |         -0.2193 |            7.5455 |    0.6424 |      4.4948 |     3.1777 |     15.9756 |
| RandomForest          |       0.8472 |          0.3968 |            5.307  |    0.6407 |      4.5055 |     3.3607 |     16.6574 |
| Ridge                 |       0.8304 |       -117.658  |           74.4352 |    0.6151 |      4.6634 |     3.6271 |     19.9262 |
| MLP                   |       0.9065 |      -7057.96   |          574.116  |    0.597  |      4.772  |     3.6251 |     19.0446 |
| Huber                 |       0.8479 |       -974.538  |          213.428  |    0.3695 |      5.9686 |     4.7007 |     26.0183 |
| Ratio+Ridge           |       0.8041 |       -457.402  |          146.303  |    0.2751 |      6.3997 |     4.7204 |     25.0751 |

## Activity_Density_uCiml_NM_40

1766 training rows / 67 acquisitions / 6 sessions, 79 features.

| model               |   cv_scan_r2 |   cv_session_r2 |   cv_session_rmse |   test_r2 |   test_rmse |   test_mae |   test_mape |
|:--------------------|-------------:|----------------:|------------------:|----------:|------------:|-----------:|------------:|
| Ratio+Ridge         |       0.9731 |         -0.2726 |            3.5884 |    0.9498 |      0.5096 |     0.4029 |      6.3206 |
| Ratio+LightGBM      |       0.983  |          0.9337 |            0.8192 |    0.9491 |      0.5133 |     0.4126 |      6.6978 |
| Ratio+HistGBM       |       0.9793 |          0.9195 |            0.9022 |    0.9452 |      0.5325 |     0.4207 |      6.9186 |
| ElasticNet          |       0.9686 |          0.6906 |            1.7695 |    0.9432 |      0.5423 |     0.4393 |      7.2256 |
| Ridge               |       0.9735 |         -3.8067 |            6.9739 |    0.9333 |      0.5873 |     0.4903 |      8.6233 |
| Ratio+RandomForest  |       0.9791 |          0.9424 |            0.7636 |    0.9309 |      0.598  |     0.4742 |      7.9239 |
| Physics: k x counts |       0.9487 |          0.9238 |            0.8781 |    0.9275 |      0.6125 |     0.4736 |      7.2346 |
| Huber               |       0.9753 |       -253.306  |           50.726  |    0.9226 |      0.6327 |     0.5129 |      9.1123 |
| MLP                 |       0.9887 |       -737.543  |           86.4451 |    0.8206 |      0.9634 |     0.7609 |     13.9657 |
| RandomForest        |       0.9818 |          0.5643 |            2.0996 |    0.5935 |      1.4503 |     1.2462 |     21.4987 |
| LightGBM            |       0.9823 |          0.5777 |            2.067  |    0.429  |      1.7189 |     1.4678 |     23.8152 |
| XGBoost             |       0.9782 |          0.5444 |            2.1471 |    0.3701 |      1.8054 |     1.5902 |     25.3805 |
| HistGBM             |       0.9787 |          0.5161 |            2.2127 |    0.2594 |      1.9576 |     1.7306 |     29.5023 |