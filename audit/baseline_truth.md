# Baseline Truth

## Current Strict Frontier
- Best strict validation speaker-level height MAE to beat: `4.562 cm`
- Line: `v2_small_no_physics`
- Status: real but still not publishable as a new frontier claim because the learned model has not completed the required multi-seed ladder

## Canonical Baselines
- `mean_height_baseline`: `9.052 cm` test speaker MAE
- `speaker_pooled_mlp`: `18.315 cm` test speaker MAE

## Omega Replay Truth
- Stage 0 and Stage 1 were replayed on the matched no-physics seed-11 checkpoint:
  - `outputs/ablations_canonical_resume_safe/v2_small_no_physics/seed_11/ckpts/best.ckpt`
- Legacy pooling stayed better than omega pooling on both validation and test:
  - validation speaker MAE: legacy `4.611 cm`, omega `4.645 cm`
  - test speaker MAE: legacy `6.157 cm`, omega `6.312 cm`
- Replay decisions:
  - `stage0_baseline_truth`: `kill`
  - `stage1_aggregation_only`: `kill`

## Omega Rule
- Every Omega stage must beat the current strict no-physics frontier honestly.
- If omega pooling or speaker-level losses fail to improve speaker MAE, they are diagnostic only and must not be promoted.
