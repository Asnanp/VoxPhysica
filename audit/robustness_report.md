# Robustness Report

## Findings
- The prior repo did not have a raw-audio perturbation harness tied to the actual inference rejection logic.
- Speaker-level helper code had a broken underscore-based grouping rule that could corrupt speaker aggregation.

## Fixes
- `src/inference/speaker_inference.py` now reads stored `speaker_id` metadata instead of truncating IDs with `split('_')[0]`.
- `scripts/predict.py` now validates the audited feature contract, computes a simple feature-space OOD score, and surfaces confidence plus rejection state.
- `scripts/run_robustness_audit.py` now stress-tests:
  - additive noise
  - clipping
  - band-limit degradation
  - short clips
  - silence padding
  - far-microphone degradation

## Remaining Risks
- Robustness numbers are not publishable until a strict audited checkpoint exists.
- Rejection thresholds may need calibration after the first canonical validation run.
