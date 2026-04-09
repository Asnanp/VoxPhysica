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
- Speaker-structured batching and speaker-level losses are now the highest-value next experiments.
