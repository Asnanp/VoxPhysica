# Architecture Audit

## Findings
- The legacy large V2 path is approximately 12.3M parameters and includes shoulder/waist branches that are outside the visible release scope.
- Base configuration previously left several nonessential branches active by default, increasing memorization risk and making attribution harder.

## Fixes
- `configs/pibnn_base.yaml` is now a strict small V2 baseline using the audited feature path and speaker-level checkpoint selection.
- Nonessential branches are disabled by default in the strict base config:
  - domain adversarial loss
  - diversity loss
  - ranking loss
  - speaker consistency loss
  - uncertainty calibration loss
  - shoulder head
  - waist head
- The legacy large model remains available only as an experimental path, not the default release candidate.

## Remaining Risks
- The model code still supports large-capacity settings, so config discipline remains important.
- Architecture claims are still provisional until strict multi-seed audited training runs are completed.
