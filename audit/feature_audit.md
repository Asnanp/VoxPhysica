# Feature Audit

## Findings
- Feature config fingerprint: `ddf9b6737588abdf5085ea9adfcfbbfab6b474e7e550d17ddeb1fea0d70b0a2d`
- `train`: files=5394, speakers=272, zero_praat_tail_files=0, zero_voiced_f0_files=0, zero_variance_dims=0
- `val`: files=678, speakers=34, zero_praat_tail_files=0, zero_voiced_f0_files=0, zero_variance_dims=0
- `test`: files=678, speakers=34, zero_praat_tail_files=0, zero_voiced_f0_files=0, zero_variance_dims=0
- Train-vs-other scalar drift: `{"test_vs_train": {"capture_quality_score": {"abs_mean_shift": 0.0035178065299987793, "other_mean": 0.5527186393737793, "reference_mean": 0.5562364459037781, "std_units": 0.10191420464087024}, "duration_s": {"abs_mean_shift": 0.25081443786621094, "other_mean": 6.015645980834961, "reference_mean": 5.76483154296875, "std_units": 0.11664718836003007}, "f0_mean": {"abs_mean_shift": 19.318801879882812, "other_mean": 187.2189483642578, "reference_mean": 167.900146484375, "std_units": 0.4107853472109339}, "formant_spacing_mean": {"abs_mean_shift": 32.0347900390625, "other_mean": 1200.3758544921875, "reference_mean": 1168.341064453125, "std_units": 0.2866003739918152}, "snr_db_estimate": {"abs_mean_shift": 0.08949661254882812, "other_mean": 41.72273635864258, "reference_mean": 41.812232971191406, "std_units": 0.01894812696844203}, "speech_ratio": {"abs_mean_shift": 0.003430604934692383, "other_mean": 0.47793975472450256, "reference_mean": 0.48137035965919495, "std_units": 0.03888915305906388}, "vtl_mean": {"abs_mean_shift": 412.783203125, "other_mean": 11401.07421875, "reference_mean": 11813.857421875, "std_units": 0.21905546485251076}}, "val_vs_train": {"capture_quality_score": {"abs_mean_shift": 0.00047576427459716797, "other_mean": 0.5567122101783752, "reference_mean": 0.5562364459037781, "std_units": 0.01378334403231885}, "duration_s": {"abs_mean_shift": 0.14640235900878906, "other_mean": 5.911233901977539, "reference_mean": 5.76483154296875, "std_units": 0.06808788079719867}, "f0_mean": {"abs_mean_shift": 13.5194091796875, "other_mean": 181.4195556640625, "reference_mean": 167.900146484375, "std_units": 0.2874699594982494}, "formant_spacing_mean": {"abs_mean_shift": 13.290283203125, "other_mean": 1181.63134765625, "reference_mean": 1168.341064453125, "std_units": 0.11890198536741324}, "snr_db_estimate": {"abs_mean_shift": 0.9346466064453125, "other_mean": 42.74687957763672, "reference_mean": 41.812232971191406, "std_units": 0.19788237861949273}, "speech_ratio": {"abs_mean_shift": 0.0019960403442382812, "other_mean": 0.4833664000034332, "reference_mean": 0.48137035965919495, "std_units": 0.022627006005315368}, "vtl_mean": {"abs_mean_shift": 307.845703125, "other_mean": 12121.703125, "reference_mean": 11813.857421875, "std_units": 0.16336731507089444}}}`

## Fixes
- Strict feature builds now fail closed when required Praat/parselmouth features are unavailable.
- Training and inference can validate the exact audited feature contract before using any artifacts.
- Per-clip capture-quality and duration metadata are now persisted into every `.npz` artifact.

## Remaining Risks
- The current environment still needs `parselmouth` installed before a canonical audited rebuild can complete.
- Build manifest status: `manifest_drift_allowed`
