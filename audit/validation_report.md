# Validation Report

## Findings
- Headline metrics below are speaker-level and use unseen test speakers only.
- Mean predictor and pooled MLP baselines were recomputed directly from audited feature artifacts.
- Legacy and omega speaker pooling are shown side by side for the same run.
- The first matched no-physics omega replay did not beat legacy pooling and should not be promoted.

## Honest Evaluation Table
| model | MAE | RMSE | median AE | MAE omega | RMSE omega | median AE omega | notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| mean_height_baseline | 9.052 | 11.152 | 7.524 | 9.052 | 11.152 | 7.524 | predicts train speaker mean height |
| speaker_pooled_mlp | 18.315 | 22.615 | 14.654 | 18.315 | 22.615 | 14.654 | speaker-level pooled feature baseline |
| v2_small_physics | nan | nan | nan | nan | nan | nan | strict model metrics not available yet |
| v2_small_no_physics | 6.157 | 8.147 | 4.671 | 6.312 | 8.192 | 5.195 | strict speaker-level metric file; val-train gap=2.02cm; cal_mae=5.31 |

## Fixes
- Standardized the baseline comparison ladder around speaker-level evaluation.
- Added train-vs-val overfit gap slots to the strict metrics JSON format.
- Replayed the matched no-physics checkpoint through the Omega ladder and recorded the kill decision for omega pooling.

## Remaining Risks
- A canonical learned-model result still requires strict multi-seed retraining.
- Omega pooling is currently diagnostic-only because the replayed no-physics checkpoint was worse under omega than under legacy pooling.
