# Leakage Report

## Findings
- Canonical feature root: `C:\Users\USER\Downloads\VoxPhysica-main21\VoxPhysica-main\data\features_vtl_external_fast`
- Speaker overlap counts: `{"train_test": 0, "train_val": 0, "val_test": 0}`
- Audio-path overlap counts: `{"train_test": 0, "train_val": 0, "val_test": 0}`
- Audio-hash overlap counts: `{"train_test": 0, "train_val": 0, "val_test": 0}`

## Fixes
- Canonicalized the build around `train_clean.csv`, `val_clean.csv`, and `test_clean.csv`.
- Added hard leakage assertions for speaker overlap, raw audio path overlap, and duplicate audio-content hashes.
- Added audited `feature_contract.json` and `build_manifest.json` artifacts so training can reject stale features.

## Remaining Risks
- If the raw split manifests change, the audited feature contract becomes invalid and the training CLI now fails closed.
