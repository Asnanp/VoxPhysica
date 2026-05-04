# Feature Audit

## Findings
- Feature config fingerprint: `ddf9b6737588abdf5085ea9adfcfbbfab6b474e7e550d17ddeb1fea0d70b0a2d`
- `train`: files=7944, speakers=1986, zero_praat_tail_files=0, zero_voiced_f0_files=0, zero_variance_dims=0
- `val`: files=388, speakers=97, zero_praat_tail_files=0, zero_voiced_f0_files=0, zero_variance_dims=0
- `test`: files=388, speakers=97, zero_praat_tail_files=0, zero_voiced_f0_files=0, zero_variance_dims=0
- Train-vs-other scalar drift: `{"test_vs_train": {"duration_s": {"abs_mean_shift": 1.1465692520141602, "other_mean": 4.832215785980225, "reference_mean": 5.978785037994385, "std_units": 0.46215609610305114}, "f0_mean": {"abs_mean_shift": 6.1573638916015625, "other_mean": 163.595458984375, "reference_mean": 157.43809509277344, "std_units": 0.13009181371868692}, "formant_spacing_mean": {"abs_mean_shift": 21.9874267578125, "other_mean": 1246.378173828125, "reference_mean": 1268.3656005859375, "std_units": 0.2746541396692107}, "vtl_mean": {"abs_mean_shift": 0.25716304779052734, "other_mean": 13.697687149047852, "reference_mean": 13.440524101257324, "std_units": 0.26856817175096676}}, "val_vs_train": {"duration_s": {"abs_mean_shift": 1.2600650787353516, "other_mean": 4.718719959259033, "reference_mean": 5.978785037994385, "std_units": 0.5079036932144434}, "f0_mean": {"abs_mean_shift": 5.3175201416015625, "other_mean": 162.755615234375, "reference_mean": 157.43809509277344, "std_units": 0.11234772735295726}, "formant_spacing_mean": {"abs_mean_shift": 31.635009765625, "other_mean": 1236.7305908203125, "reference_mean": 1268.3656005859375, "std_units": 0.39516613227682856}, "vtl_mean": {"abs_mean_shift": 0.3988656997680664, "other_mean": 13.83938980102539, "reference_mean": 13.440524101257324, "std_units": 0.4165553048202188}}}`

## Fixes
- Strict feature builds now fail closed when required Praat/parselmouth features are unavailable.
- Training and inference can validate the exact audited feature contract before using any artifacts.
- Per-clip capture-quality and duration metadata are now persisted into every `.npz` artifact.

## Remaining Risks
- The current environment still needs `parselmouth` installed before a canonical audited rebuild can complete.
- Build manifest status: `canonical`
