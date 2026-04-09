# Feature Audit

## Findings
- Feature config fingerprint: `ddf9b6737588abdf5085ea9adfcfbbfab6b474e7e550d17ddeb1fea0d70b0a2d`
- Strict backend verification passed in the evidence environment with `Python 3.12.4` and `praat-parselmouth 0.4.7`.
- `train`: files=26056, speakers=775, zero_praat_tail_files=0, zero_voiced_f0_files=0, zero_variance_dims=0
- `val`: files=1158, speakers=97, zero_praat_tail_files=0, zero_voiced_f0_files=0, zero_variance_dims=0
- `test`: files=1155, speakers=97, zero_praat_tail_files=0, zero_voiced_f0_files=0, zero_variance_dims=0
- Short/load-failed clip attrition during audited rebuild was `124` train, `14` val, and `19` test.
- Train-vs-other scalar drift: `{"test_vs_train": {"capture_quality_score": {"abs_mean_shift": 0.010053038597106934, "other_mean": 0.8252978324890137, "reference_mean": 0.8353508710861206, "std_units": 0.1600473892555491}, "duration_s": {"abs_mean_shift": 0.8265609741210938, "other_mean": 5.065757751464844, "reference_mean": 4.23919677734375, "std_units": 0.3356257632868177}, "f0_mean": {"abs_mean_shift": 13.680877685546875, "other_mean": 170.7060089111328, "reference_mean": 157.02513122558594, "std_units": 0.2965004961709722}, "formant_spacing_mean": {"abs_mean_shift": 66.4932861328125, "other_mean": 1138.06201171875, "reference_mean": 1071.5687255859375, "std_units": 0.4905938175801454}, "snr_db_estimate": {"abs_mean_shift": 0.5551624298095703, "other_mean": 31.083229064941406, "reference_mean": 30.528066635131836, "std_units": 0.0880450684030762}, "speech_ratio": {"abs_mean_shift": 0.003111720085144043, "other_mean": 0.9859793782234192, "reference_mean": 0.9890910983085632, "std_units": 0.08553650013440343}, "vtl_mean": {"abs_mean_shift": 1427.76171875, "other_mean": 9452.6142578125, "reference_mean": 8024.8525390625, "std_units": 0.398823493227257}}, "val_vs_train": {"capture_quality_score": {"abs_mean_shift": 0.007488429546356201, "other_mean": 0.8278624415397644, "reference_mean": 0.8353508710861206, "std_units": 0.11921804407110619}, "duration_s": {"abs_mean_shift": 0.7085976600646973, "other_mean": 4.947794437408447, "reference_mean": 4.23919677734375, "std_units": 0.28772666260387114}, "f0_mean": {"abs_mean_shift": 12.443130493164062, "other_mean": 169.46826171875, "reference_mean": 157.02513122558594, "std_units": 0.26967526864456576}, "formant_spacing_mean": {"abs_mean_shift": 52.033447265625, "other_mean": 1123.6021728515625, "reference_mean": 1071.5687255859375, "std_units": 0.38390774498511626}, "snr_db_estimate": {"abs_mean_shift": 0.14786148071289062, "other_mean": 30.380205154418945, "reference_mean": 30.528066635131836, "std_units": 0.023449847260039073}, "speech_ratio": {"abs_mean_shift": 0.002193629741668701, "other_mean": 0.9868974685668945, "reference_mean": 0.9890910983085632, "std_units": 0.06029957886921905}, "vtl_mean": {"abs_mean_shift": 1808.779296875, "other_mean": 9833.6318359375, "reference_mean": 8024.8525390625, "std_units": 0.5052549512872484}}}`

## Fixes
- Strict feature builds now fail closed when required Praat/parselmouth features are unavailable.
- Training and inference can validate the exact audited feature contract before using any artifacts.
- Per-clip capture-quality and duration metadata are now persisted into every `.npz` artifact.

## Remaining Risks
- Train/test drift in duration and VTL-linked aggregates remains visible, especially `vtl_mean` (`~0.40-0.51` train-to-holdout std units), so generalization still has to be proven by strict speaker-level model evaluation.
- Build manifest status: `canonical`
