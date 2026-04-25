# Ablation Plan

## Canonical Ladder
- `mean_height_baseline`
- `speaker_pooled_mlp`
- `v2_small_no_physics`
- `v2_small_physics`

## Mandatory Protocol
- Use `data/features_audited/` only.
- Evaluate on unseen speakers only.
- Aggregate predictions per speaker.
- Run at least 3 seeds for each trainable model.
- Select checkpoints only by `height_mae_speaker` unless a later stage earns a monitor change.

## Omega Ladder
- `stage0_baseline_truth`
- `stage1_aggregation_only`
- `stage2_speaker_structured_no_physics`
- `stage3_speaker_alignment`
- `stage3c_height_only_regularized`
- `stage3d_height_only_slice_aligned`
- `stage3e_height_only_stable_bin_weighted`
- `stage3f_height_only_long_stable`
- `stage4_proper_v2_height_first`
- `stage4_learned_reliability`
- `stage5_physics_smart`
- `stage6_flagship`

## Current Replay Outcome
- `stage0_baseline_truth`: killed
- `stage1_aggregation_only`: killed
- Evidence from the matched no-physics seed-11 checkpoint:
  - validation speaker MAE: legacy `4.611 cm`, omega `4.645 cm`
  - test speaker MAE: legacy `6.157 cm`, omega `6.312 cm`

## Promotion Rules
- Promote only if mean validation speaker MAE improves under the same strict protocol.
- Kill a line if it helps clip metrics only, worsens speaker MAE, or collapses a critical slice.
- Keep the no-physics line as the flagship default until physics beats it across the same seeds.

## Immediate Next Line
- Stage 2 should keep `height_mae_speaker` as the primary monitor.
- Omega pooling should stay logged as a diagnostic-only metric until a redesigned pooling method proves a real lift.
- Stop spending time on pooling tricks or speaker-batching structures.
- Next stage: `stage3e_height_only_stable_bin_weighted`
  - hypothesis: Stage 3c preserved the stronger strict-test behavior, while Stage 3d improved short-slice pressure and the gap but became unstable under the speaker-batched regime; Stage 3e keeps the Stage 3c backbone and plain shuffled batches, adds only gentle hard-slice weighting plus smoothing, and removes restart-heavy scheduler pressure
  - why it could beat `4.611 / 6.157`: it preserves the best evidence-backed no-physics training path, attacks short-height brittleness directly, and avoids the instability pattern that made Stage 3d unpromotable
  - early kill rule: kill if seed 11 is still above `4.8 cm` validation speaker MAE by epoch 5, still above a `+2.0 cm` train/val speaker gap by epoch 5, or shows no material short-height speaker improvement
- Long-run candidate: `stage3f_height_only_long_stable`
  - hypothesis: the best current evidence still comes from the simpler Stage 3c objective, so the highest-confidence 50-epoch shot is to keep that objective intact and only remove the warm-restart schedule that can destabilize late epochs
  - why it could beat `4.611 / 6.157`: it keeps the cleanest no-physics line and gives it a longer, restart-free decay path instead of adding more speaker-level machinery that has not earned promotion
