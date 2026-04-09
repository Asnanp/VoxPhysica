# Run Decision Log

## stage0_baseline_truth / seed 11
- Variant: `baseline_truth`
- Mode: `replay`
- Decision: `kill`
- Notes: matched no-physics replay showed omega pooling was worse than legacy on both validation and test
- Config diff: `C:\Users\USER\OneDrive\Desktop\VoiceMorph\outputs\omega\stage0_baseline_truth\baseline_truth\seed_11\config_diff.json`
- Legacy val/test speaker MAE: `4.611` / `6.157`
- Omega val/test speaker MAE: `4.645` / `6.312`
- Train->val gaps: legacy `2.018`, omega `2.150`
- Calibration: `cal=5.313`, `corr=0.030`, `p68=0.282`, `p95=0.480`

## stage1_aggregation_only / seed 11
- Variant: `aggregation_only`
- Mode: `replay`
- Decision: `kill`
- Notes: aggregation-only replay failed promotion; omega remains diagnostic-only
- Config diff: `C:\Users\USER\OneDrive\Desktop\VoiceMorph\outputs\omega\stage1_aggregation_only\aggregation_only\seed_11\config_diff.json`
- Legacy val/test speaker MAE: `4.611` / `6.157`
- Omega val/test speaker MAE: `4.645` / `6.312`
- Train->val gaps: legacy `2.018`, omega `2.150`
- Calibration: `cal=5.313`, `corr=0.030`, `p68=0.282`, `p95=0.480`

## stage2_speaker_structured_no_physics / seed 11
- Variant: `speaker_structured_no_physics`
- Mode: `train`
- Decision: `kill`
- Notes: grouped-speaker batching reduced the apparent overfit gap but made the frontier materially worse
- Config diff: `C:\Users\USER\OneDrive\Desktop\VoiceMorph\outputs\omega\stage2_speaker_structured_no_physics\speaker_structured_no_physics\seed_11\config_diff.json`
- Legacy val/test speaker MAE: `5.334` / `6.951`
- Omega val/test speaker MAE: `5.349` / `7.004`
- Train->val gaps: legacy `-0.594`, omega `-0.561`
- Calibration: `cal=4.813`, `corr=-0.068`, `p68=0.410`, `p95=0.649`

## stage3_speaker_alignment / seed 11
- Variant: `speaker_alignment_no_physics`
- Mode: `train`
- Decision: `kill`
- Notes: one-epoch exploratory smoke run only; much worse than the frontier and not promotable
- Config diff: `C:\Users\USER\OneDrive\Desktop\VoiceMorph\outputs\omega\stage3_speaker_alignment\speaker_alignment_no_physics\seed_11\config_diff.json`
- Legacy val/test speaker MAE: `7.794` / `9.629`
- Omega val/test speaker MAE: `7.823` / `9.670`
- Train->val gaps: legacy `-0.631`, omega `-0.603`
- Calibration: `cal=7.328`, `corr=-0.139`, `p68=0.340`, `p95=0.568`
