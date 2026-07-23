# VoxPhysica 3 cm MAE Research Plan

**Status:** strict run and first short-tail support pass completed; target not met  
**Measured result:** 4.951 cm speaker-level MAE, 95% CI 4.040–5.888 cm  
**Public tail support:** 80 speakers, 3,140 valid clips, negligible validation delta (-0.007 cm; paired 95% CI -0.051 to +0.033 cm)  
**Target:** speaker-level height MAE at or below 3.0 cm

## 1. Non-negotiable validity rules

1. A speaker may occur in exactly one of train, validation, or test.
2. Candidate models, feature choices, ensemble weights, and correction rules may use train out-of-fold predictions and validation labels only.
3. Test labels are opened once after the complete recipe is frozen.
4. A test result is not called sealed again after it has influenced another development decision.
5. Voice-only and metadata-assisted systems are labeled separately.
6. The historical 1.683 cm all-data cross-validation result is retained only as an audit artifact. It is not a held-out result.

## 2. Current evidence baseline

The repository contains 775 training, 97 validation, and 97 test speakers. The strongest validation-safe historical candidate in the Phase 22 audit is the Phase 12 baseline:

- validation MAE: 4.091 cm
- historical fixed-test MAE: 4.951 cm
- historical fixed-test within 3 cm: 44.3%

The Phase 22 global oracle reaches about 1.946 cm only by choosing predictions with knowledge unavailable at inference time. It establishes complementarity among models, not a deployable 3 cm system.

The completed strict search evaluated 65 configurations without failures. Validation retained the voice-only Phase 12 reference. The one-pass historical-test result was 4.951 cm MAE, 44.3% within 3 cm, and a 95% bootstrap interval of 4.040–5.888 cm. Both the 3 cm and 4 cm point targets failed. Short speakers below 160 cm remain the dominant blocker at 9.410 cm MAE.

## 3. Implemented experiment

The strict pipeline performs the following sequence:

1. Verify speaker disjointness and target agreement across split metadata and every complete WavLM cache.
2. Build independent WavLM views plus mean, difference, and concatenated fusion views.
3. Add a declared metadata-assisted track using gender, source, age, weight, language family, TIMIT dialect region, missingness flags, and clip count.
4. Generate source/gender-stratified out-of-fold predictions on training speakers.
5. Compare regularized ridge, PCA-SVR, absolute-loss histogram boosting, extra trees, and a shrunken hierarchical prior.
6. Select a small error-diverse shortlist using training OOF MAE and prediction correlation.
7. Learn non-negative convex ensemble weights from OOF predictions only.
8. Compare a limited set of predeclared recipes on validation, including the frozen Phase 12 reference.
9. Refit selected components on train plus validation.
10. Estimate residual offsets from development OOF predictions, not test labels.
11. Optionally snap TIMIT predictions to the corpus's 2.54 cm label grid if that rule wins on validation.
12. Evaluate test once and report MAE, median error, RMSE, 90th-percentile error, within-3/4 cm rates, subgroup results, and a bootstrap confidence interval.
13. Save predictions, model bundle, data hashes, full candidate metrics, and a human-readable report.

## 4. Acceptance gates

A result may be described as meeting the target only if all checks pass:

- speaker overlap count is zero;
- no test column enters candidate selection or postprocessing;
- every selected model has complete finite OOF predictions;
- point test MAE is at most 3.0 cm;
- result scope is stated as voice-only or metadata-assisted;
- the exact split and feature hashes are saved;
- the prediction CSV reproduces the reported MAE;
- the test set is not used for a subsequent tuning round.

The stronger publication gate additionally requires the bootstrap upper 95% bound to be at most 3.0 cm.

## 5. If the strict result remains above 3 cm

Do not tune against the 97 historical test labels. The next valid iteration is data work:

1. Create a new untouched external test set with measured, not estimated, height.
2. Add short male speakers and rare height tails, which dominate the existing error budget.
3. Balance source, language, gender, age, microphone, and height distributions.
4. Record repeated sessions to measure within-speaker stability and channel shift.
5. Train phonetic or vowel-aware pooling because selected phones and formants carry useful height cues.
6. Compare frozen WavLM features with supervised contrastive speaker-height adaptation.
7. Use nested cross-validation for architecture selection, then evaluate once on the new external set.
8. Report calibration and uncertainty so uncertain estimates can be abstained from rather than presented as precise measurements.

### 5.1 Completed short-tail data action

The first data-expansion pass is complete:

- audited the local HeightCeleb/VoxCeleb1 support manifest;
- found 80 additional unique speakers below 160 cm, with zero ID overlap against train, validation, or historical test;
- quality-controlled 3,168 candidate files and retained 3,140 clips (6.92 hours);
- rejected 28 clips for duplicate content, near-silence, or duration above 30 seconds;
- created `data/splits/train_plus_short_support.csv` with 855 unique training speakers;
- restricted all public estimated-height rows to train support;
- added adult consent, repeated-height measurement, pseudonymous intake, and sealed-test role controls; and
- predeclared a prospective pilot quota of 120 measured development speakers and 80 measured sealed-test speakers, including at least 45 short male speakers across both roles.

The public support set contains 79 female and only 1 male speaker below 160 cm. It therefore does not solve the critical short-male gap.

A development-only 2,007-candidate ridge/gating comparison selected its recipe from target-training out-of-fold predictions and then evaluated validation once. Validation MAE changed from 4.887 cm without public support to 4.879 cm with support; short-slice MAE changed from 4.778 to 4.768 cm. The 0.007 cm overall change is negligible. No historical-test labels or features were loaded.

## 6. Reproducibility commands

The frozen experiment has already consumed this historical test split. Re-running is reproducible verification, not a new sealed result:

    RUN_STRICT_3CM_RESEARCH.bat

Equivalent Python command:

    python scripts/run_strict_3cm_research.py --output-dir outputs/strict_3cm_research

Short-tail data audit and development-only comparison:

    python scripts/collect_short_speaker_data.py
    python scripts/evaluate_short_support_dev.py

Focused tests:

    python -m pytest tests/test_strict_height_pipeline.py -q

Expected outputs:

- outputs/strict_3cm_research/strict_results.json
- outputs/strict_3cm_research/STRICT_REPORT.md
- outputs/strict_3cm_research/predictions_validation_frozen.csv
- outputs/strict_3cm_research/predictions_test_once.csv
- outputs/strict_3cm_research/strict_model_bundle.joblib

## 7. Publication checklist

- Keep the paper result synchronized only from strict_results.json; it is currently 4.951 cm.
- Include both point MAE and confidence interval.
- State that the historical test set has been repeatedly inspected in prior work.
- Do not compare MAE values across TIMIT studies without noting partition and phone-selection differences.
- Include data licensing and consent statements before public release.
- Avoid forensic-identification claims; height from voice is an uncertain soft-biometric estimate.
