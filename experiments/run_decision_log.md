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

## stage2b_hybrid_speaker_structured / seed 11
- Variant: `hybrid_speaker_structured_no_physics`
- Mode: `train`
- Decision: `kill`
- Notes: killed after epoch 1; the eval-only score from the saved best checkpoint still missed the frontier badly, so the line did not earn more epochs or more seeds.
- Config diff: `C:\Users\USER\OneDrive\Desktop\VoiceMorph\outputs\omega\stage2b_hybrid_speaker_structured\hybrid_speaker_structured_no_physics\seed_11\config_diff.json`
- Legacy val/test speaker MAE: `7.815` / `9.591`
- Omega val/test speaker MAE: `7.833` / `9.537`
- Train->val gaps: legacy `-0.531`, omega `-0.508`
- Calibration: `cal=7.050`, `corr=-0.098`, `p68=0.352`, `p95=0.556`

## stage3b_height_focused_control / seed 11
- Variant: `height_focused_no_physics`
- Mode: `train`
- Decision: `kill`
- Notes: killed after epoch 4; the line was unstable and still failed to beat the frontier on both validation and strict test.
- Config diff: `C:\Users\USER\OneDrive\Desktop\VoiceMorph\outputs\omega\stage3b_height_focused_control\height_focused_no_physics\seed_11\config_diff.json`
- Legacy val/test speaker MAE: `5.842` / `6.365`
- Omega val/test speaker MAE: `6.285` / `6.791`
- Train->val gaps: legacy `2.789`, omega `2.861`
- Calibration: `cal=6.745`, `corr=0.074`, `p68=0.192`, `p95=0.397`
## stage3c_height_only_regularized / seed 11
- Variant: `height_only_regularized_no_physics`
- Mode: `train`
- Decision: `hold`
- Notes: this is the first strict single-seed line to beat the old no-physics frontier on both validation and test, but it is still not promotable because the train-to-val speaker gap remained large and the validation medium-quality slice was severely unstable.
- Config diff: `C:\Users\USER\OneDrive\Desktop\VoiceMorph\outputs\omega\stage3c_height_only_regularized\height_only_regularized_no_physics\seed_11\config_diff.json`

- Legacy val/test speaker MAE: `4.361` / `5.889`
- Omega val/test speaker MAE: `4.303` / `5.970`
- Train->val gaps: legacy `2.325`, omega `2.423`
- Critical slices: legacy[src=0.583,gender=0.550,height=0.967,quality=9.231]; omega[src=1.003,gender=0.423,height=0.373,quality=11.448]; max_deg=2.135
- Calibration: cal=5.737, corr=0.108, p68=0.206, p95=0.406
- Promotion block:
  - only one seed so far
  - legacy train->val speaker gap is still `+2.325 cm`
  - validation `height_quality_medium_speaker_mae = 13.497 cm`
  - omega is still not promotable because it only improved validation while worsening strict test

## stage3c_height_only_regularized / seed 17
- Variant: `height_only_regularized_no_physics`
- Mode: `train`
- Decision: `hold`
- Notes: replication weakened confidence. Validation missed the old frontier, test only barely beat it, the train-to-val speaker gap stayed large, and the short-height test slice remained bad.
- Config diff: `C:\Users\USER\OneDrive\Desktop\VoiceMorph\outputs/omega\stage3c_height_only_regularized\height_only_regularized_no_physics\seed_17\config_diff.json`

- Legacy val/test speaker MAE: `4.918` / `6.060`
- Omega val/test speaker MAE: `5.033` / `5.992`
- Train->val gaps: legacy `2.316`, omega `2.592`
- Critical slices: legacy[src=0.327,gender=0.566,height=2.595,quality=4.801]; omega[src=0.523,gender=0.428,height=1.063,quality=3.217]; max_deg=0.319
- Calibration: cal=5.332, corr=-0.017, p68=0.223, p95=0.417
- Failure mode snapshot:
  - validation missed the old `4.611 cm` frontier by `+0.307 cm`
  - test only beat the old `6.157 cm` frontier by `-0.097 cm`
  - `height_heightbin_short_speaker_mae = 12.330 cm`

## stage3c_height_only_regularized / seed 23
- Variant: `height_only_regularized_no_physics`
- Mode: `train`
- Decision: `hold`
- Notes: replication stayed mixed. Test beat the old frontier again, but validation still missed it and the speaker gap stayed large.
- Config diff: `C:\Users\USER\OneDrive\Desktop\VoiceMorph\outputs/omega\stage3c_height_only_regularized\height_only_regularized_no_physics\seed_23\config_diff.json`

- Legacy val/test speaker MAE: `4.736` / `5.974`
- Omega val/test speaker MAE: `4.845` / `5.977`
- Train->val gaps: legacy `2.359`, omega `2.591`
- Critical slices: legacy[src=0.889,gender=0.744,height=1.931,quality=4.379]; omega[src=1.201,gender=0.207,height=1.468,quality=5.493]; max_deg=1.212
- Calibration: cal=5.310, corr=0.088, p68=0.260, p95=0.473
- Failure mode snapshot:
  - validation missed the old `4.611 cm` frontier by `+0.125 cm`
  - test beat the old `6.157 cm` frontier by `-0.183 cm`
  - `height_heightbin_short_speaker_mae = 10.731 cm`

## stage3c_height_only_regularized / replication verdict
- Decision: `hold`
- Seeds evaluated: `11`, `17`, `23`
- Historical frontier: `4.611 cm` validation / `6.157 cm` test
- Stage 3c legacy mean across seeds: `4.671 cm` validation / `5.974 cm` test
- Stage 3c legacy spread across seeds: `0.284 cm` validation stdev / `0.086 cm` test stdev
- Why not promote:
  - mean validation is still slightly worse than the old frontier
  - all three seeds kept a large train-to-val speaker gap around `+2.3 cm`
  - short-height test speaker MAE stayed extremely high across all seeds
  - seed-level epoch traces still oscillated sharply, especially on seeds `17` and `23`
- Why not kill:
  - all three seeds beat the old strict test frontier
  - the line did not collapse under replication; it remained mixed rather than fraudulent

## stage3d_height_only_slice_aligned / seed 11
- Variant: `height_only_slice_aligned_no_physics`
- Mode: `train`
- Decision: `hold`
- Notes: mixed first evidence; validation and gap improved materially, but strict test still missed the old frontier and late-epoch stability was poor
- Config diff: `C:\Users\USER\OneDrive\Desktop\VoiceMorph\outputs/omega\stage3d_height_only_slice_aligned\height_only_slice_aligned_no_physics\seed_11\config_diff.json`

- Legacy val/test speaker MAE: `4.324` / `6.223`
- Omega val/test speaker MAE: `4.372` / `6.326`
- Train->val gaps: legacy `0.806`, omega `0.808`
- Critical slices: legacy[src=0.171,gender=0.949,height=1.327,quality=7.680]; omega[src=0.094,gender=0.763,height=1.944,quality=7.275]; max_deg=0.529
- Calibration: cal=5.292, corr=0.084, p68=0.208, p95=0.423
- Best checkpoint: `epoch_0006_metric_4.3243.ckpt`
- Validation short-height speaker MAE: `5.447 cm`
- Test short-height speaker MAE: `11.987 cm`
- Validation quality-medium speaker MAE: `11.925 cm`
- Epoch trace:
  - `4.88 -> 4.52 -> 5.71 -> 6.38 -> 4.52 -> 4.32 -> 4.37 -> 4.52 -> 10.29 -> 8.19`
- Readout:
  - validation improved vs the old `4.611 cm` frontier
  - strict test missed the old `6.157 cm` frontier by `+0.066 cm`
  - best-checkpoint train-to-val speaker gap dropped sharply from the Stage 3c `~+2.3 cm` range to `+0.806 cm`
  - short-height validation improved relative to Stage 3c seed 11, but short-height strict test remained badly unresolved
  - late-epoch collapse means the line is still seed-11 unstable and not promotable from first evidence

## stage3f_height_only_long_stable / seed 11
- Variant: `height_only_long_stable_no_physics`
- Mode: `train`
- Decision: `hold`
- Notes: real but limited positive single-seed result; it beat the old frontier on this seed, but the gain peaked early and did not consolidate
- Config diff: `C:\Users\USER\OneDrive\Desktop\VoiceMorph\outputs/omega\stage3f_height_only_long_stable\height_only_long_stable_no_physics\seed_11\config_diff.json`

- Legacy val/test speaker MAE: `4.527` / `5.802`
- Omega val/test speaker MAE: `4.589` / `5.957`
- Train->val gaps: legacy `1.917`, omega `2.107`
- Critical slices: legacy[src=0.490,gender=0.771,height=1.586,quality=6.946]; omega[src=0.449,gender=0.673,height=1.038,quality=8.271]; max_deg=1.375
- Calibration: cal=4.948, corr=0.061, p68=0.269, p95=0.500
- Best checkpoint: `epoch_0005_metric_4.5267.ckpt`
- Final epoch: `15` via early stop after `10` non-improving epochs
- Final train_eval / val gap: `2.412 cm` / `4.720 cm` -> `+2.309 cm`
- Best validation short-height speaker MAE: `3.513 cm` at epoch `5`
- Final validation short-height speaker MAE: `5.937 cm` at epoch `15`
- Validation quality-medium speaker MAE: best `7.658 cm` at epoch `2`, `11.401 cm` at the best overall epoch, final `9.693 cm`
- Top 5 validation epochs:
  - epoch `5`: val `4.527`, train_eval `2.609`, gap `+1.917`, short `3.513`, quality-medium `11.401`, lr `2.929e-4`
  - epoch `7`: val `4.546`, train_eval `1.902`, gap `+2.644`, short `3.016`, quality-medium `12.322`, lr `2.862e-4`
  - epoch `15`: val `4.720`, train_eval `2.412`, gap `+2.309`, short `5.937`, quality-medium `9.693`, lr `2.402e-4`
  - epoch `9`: val `4.758`, train_eval `2.600`, gap `+2.158`, short `5.279`, quality-medium `14.128`, lr `2.774e-4`
  - epoch `6`: val `4.836`, train_eval `3.114`, gap `+1.722`, short `8.450`, quality-medium `13.624`, lr `2.898e-4`
- Epoch trace:
  - `6.44 -> 5.35 -> 9.32 -> 5.34 -> 4.53 -> 4.84 -> 4.55 -> 4.84 -> 4.76 -> 9.47 -> 10.85 -> 7.03 -> 5.69 -> 7.16 -> 4.72`
- Short-height validation trace:
  - `13.48 -> 8.00 -> 18.47 -> 3.30 -> 3.51 -> 8.45 -> 3.02 -> 3.76 -> 5.28 -> 18.12 -> 21.27 -> 15.51 -> 11.22 -> 3.99 -> 5.94`
- Quality-medium validation trace:
  - `12.19 -> 7.66 -> 18.13 -> 8.58 -> 11.40 -> 13.62 -> 12.32 -> 10.05 -> 14.13 -> 16.29 -> 17.74 -> 14.13 -> 13.36 -> 7.76 -> 9.69`
- LR / warmup readout:
  - one-epoch warmup plus monotonic cosine decay did not lead to a later consolidation
  - the best validation point arrived early at epoch `5` while lr was still close to the base rate (`2.929e-4`)
  - later lower-lr epochs did not recover a cleaner frontier
- Readout:
  - this is a real but limited positive result, not a breakthrough
  - on this seed, legacy validation beat the old `4.611 cm` frontier by `0.084 cm` and strict legacy test beat the old `6.157 cm` frontier by `0.355 cm`
  - short-height validation improved materially versus the earlier height-only lines, especially in epochs `4-8`
  - the line still failed to consolidate: the train speaker error kept falling much faster than validation, the final gap reopened to `+2.309 cm`, and validation quality-medium speakers remained unstable
  - decision stays `hold`; promising, but not promotable from this single seed
## stage3f_height_only_long_stable / seed 11
- Variant: `height_only_long_stable_no_physics`
- Mode: `train`
- Decision: `hold`
- Notes: training completed; metrics captured; promotion requires ladder-stage review
- Config diff: `C:\Users\USER\OneDrive\Desktop\VoxPhysica\outputs/omega_best_push\stage3f_height_only_long_stable\height_only_long_stable_no_physics\seed_11\config_diff.json`

- Legacy val/test speaker MAE: `4.482` / `6.124`
- Omega val/test speaker MAE: `4.448` / `6.239`
- Train->val gaps: legacy `-0.063`, omega `-0.127`
- Critical slices: legacy[src=0.384,gender=0.732,height=2.207,quality=4.324]; omega[src=0.300,gender=0.832,height=2.644,quality=4.039]; max_deg=0.365
- Calibration: cal=4.355, corr=0.121, p68=0.362, p95=0.621

