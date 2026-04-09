# Current Frontier Summary

## Strong
- Canonical speaker-pure splits
- Fail-closed audited features
- Correct speaker-level aggregation by true `speaker_id`
- Crash-safe training and checkpoint recovery
- Compact V2 path with honest speaker-level monitoring

## Limiting
- Best strict validation speaker MAE is still far from `2.0-2.5 cm`
- Train-vs-val speaker gap remains large
- Physics remains unproven against the no-physics frontier
- The first omega pooling replay did not improve the last mile
- The first grouped-speaker Stage 2 line degraded the frontier
- The first one-epoch Stage 3 speaker-alignment smoke run degraded it even more

## Confirmed Results
- Replay frontier remains the best current no-physics reference:
  - validation speaker MAE: `4.611 cm`
  - test speaker MAE: `6.157 cm`
- `stage2_speaker_structured_no_physics`:
  - validation speaker MAE: `5.334 cm`
  - test speaker MAE: `6.951 cm`
  - decision: `kill`
- `stage3_speaker_alignment` one-epoch exploratory:
  - validation speaker MAE: `7.794 cm`
  - test speaker MAE: `9.629 cm`
  - decision: `kill`

## Current Direction
- Keep legacy speaker aggregation as the primary monitor.
- Keep omega pooling diagnostic-only until a redesigned pooling line proves a lift.
- Redesign the Stage 2 batching strategy before spending more ladder seeds.
- Do not promote the current Stage 3 alignment line.
- Active redesign:
  - `stage2b_hybrid_speaker_structured`
  - hybrid batch shape = `4` paired speakers + `8` singleton speakers
  - legacy aggregation remains primary
  - running as a `10`-epoch single-seed smoke on CUDA
