# Status

## Analyzed
- Repository mapping for data loading, feature extraction, model architecture, training, validation, inference, and configs.
- Historical split/feature mismatch and silent Praat feature collapse.
- Checkpoint selection logic, speaker aggregation, and evaluation honesty.
- Omega replay behavior and the first Stage 2/Stage 3 no-physics experiments.

## Fixed
- Added audited split and feature contract utilities.
- Rebuilt the canonical audited feature store at `data/features_audited/`.
- Hardened `scripts/train.py` and `scripts/predict.py` around audited feature-contract validation.
- Expanded evaluation metrics and added train-vs-val overfit-gap reporting.
- Fixed speaker aggregation to respect stored `speaker_id`.
- Added crash-safe training recovery, automatic logging, and strict evaluation scripts.
- Implemented the Omega first wave:
  - shared `omega_robust_reliability_pool`
  - handcrafted clip reliability composer
  - metadata-reliability tower scaffolding
  - deterministic grouped-speaker batch sampler
  - explicit speaker-level alignment losses
  - omega metrics logged beside legacy metrics
  - `scripts/run_omega_ladder.py`
- Added replay compatibility for older no-physics checkpoints:
  - model-only eval restore
  - forward-compatible missing-key handling for `reliability_tower.*`
  - Windows-safe worker seeding for replay/eval workers
- Fixed Omega ladder bookkeeping so later invocations no longer wipe `experiment_registry.md` and `run_decision_log.md`.
- Realigned Stage 3+ ladder monitoring back to legacy `height_mae_speaker`.
- Added a softer Stage 2b redesign path:
  - hybrid speaker sampler with paired and singleton speakers in the same batch
  - explicit `training.speaker_batching.mode`
  - omega runner epoch override for controlled smoke tests

## Evidence Update
- `stage0_baseline_truth`: `kill`
  - validation speaker MAE: legacy `4.611 cm`, omega `4.645 cm`
  - test speaker MAE: legacy `6.157 cm`, omega `6.312 cm`
- `stage1_aggregation_only`: `kill`
  - same matched no-physics replay result; omega remains diagnostic-only
- `stage2_speaker_structured_no_physics`: `kill`
  - validation speaker MAE: legacy `5.334 cm`, omega `5.349 cm`
  - test speaker MAE: legacy `6.951 cm`, omega `7.004 cm`
  - grouped-speaker batching reduced the apparent gap but materially worsened the frontier
- `stage3_speaker_alignment` one-epoch exploratory smoke: `kill`
  - validation speaker MAE: legacy `7.794 cm`, omega `7.823 cm`
  - test speaker MAE: legacy `9.629 cm`, omega `9.670 cm`
  - this was exploratory only and is not promotable evidence

## Still Risky
- No Omega training stage has improved the frontier yet.
- Only one matched seed has been replayed and one seed has been trained for Stage 2, so broader seed stability is still unknown.
- Learned reliability mode is still scaffolding until a strict training stage proves it helps.
- Physics remains unearned against the no-physics line.
- Stage 2b hybrid batching is live but not yet judged; it must beat the replay frontier before it earns more seeds.

## Next Experiments
- Do not promote Stage 2 or Stage 3 as currently designed.
- Redesign Stage 2 before spending more seeds:
  - revisit grouped-speaker batching shape and sampler pressure
  - reduce aggression before reintroducing speaker-level losses
- Active run:
  - `stage2b_hybrid_speaker_structured`
  - single-seed smoke on `seed 11`
  - `10` epochs on CUDA
  - hybrid batch shape = `4` paired speakers with `2` clips each + `8` singleton speakers
- Keep legacy `height_mae_speaker` as the primary monitor.
- Keep omega pooling diagnostic-only until a redesigned pooling method proves a real gain.
