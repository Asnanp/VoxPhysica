# Leakage Report

## Findings
- Canonical feature root: `C:\Users\USER\OneDrive\Desktop\VoiceMorph\data/features_audited`
- Speaker overlap counts: `{"train_test": 0, "train_val": 0, "val_test": 0}`
- Audio-path overlap counts: `{"train_test": 0, "train_val": 0, "val_test": 0}`
- Audio-hash overlap counts: `{"train_test": 0, "train_val": 0, "val_test": 0}`
- Canonical manifests realized exactly as expected at the speaker level: `775` train, `97` val, `97` test.
- No manifest drift remained after materialization: `missing_expected_speakers=0`, `unexpected_speakers=0` for all splits.
- Realized clip attrition was limited to load/too-short failures: `124` train, `14` val, `19` test.

## Fixes
- Canonicalized the build around `train_clean.csv`, `val_clean.csv`, and `test_clean.csv`.
- Added hard leakage assertions for speaker overlap, raw audio path overlap, and duplicate audio-content hashes.
- Added audited `feature_contract.json` and `build_manifest.json` artifacts so training can reject stale features.

## Remaining Risks
- If the raw split manifests change, the audited feature contract becomes invalid and the training CLI now fails closed.
- Clip-level attrition is non-zero, so any future rebuild must continue reporting exact skipped-file counts rather than assuming all manifest-listed audio survives preprocessing.
