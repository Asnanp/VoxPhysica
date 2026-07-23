# Leakage-Resistant Speaker Height Estimation with Multi-View Self-Supervised Speech Representations

**VoxPhysica Research Project**  
Research manuscript and frozen evaluation report  
22 July 2026

## Abstract

Estimating human height from speech is a soft-biometric regression problem with weak physiological signal, substantial demographic confounding, and high sensitivity to dataset partitioning. This study audits the VoxPhysica speaker-height repository and introduces a leakage-resistant pipeline aimed at a speaker-level mean absolute error (MAE) of 3.0 cm. The data comprise 969 speaker-disjoint records drawn from TIMIT and NISP: 775 training, 97 validation, and 97 historical test speakers. The proposed method combines multiple cached WavLM representation views, supervised feature selection, regularized regression, nonlinear metadata models, out-of-fold convex ensembling, and validation-selected postprocessing. All architecture, blend, and correction choices are made without test labels. A repository audit found that a previously reported 1.683 cm result was not a valid held-out estimate because it cross-validated a concatenation of train, validation, and test speakers and included a neural prediction feature that was in-sample for most training speakers. The strongest validation-safe historical fixed-split candidate achieved 4.951 cm test MAE and 44.3% of speakers within 3 cm. The strict run evaluated 65 candidate configurations without failures. Validation retained the frozen Phase 12 voice-only model; its one-pass historical-test result was 4.951 cm MAE, with a 95% speaker-bootstrap interval of 4.040–5.888 cm and 44.3% of speakers within 3 cm. The 3 cm and 4 cm point targets were therefore not met. A follow-up short-tail data audit accepted 80 additional HeightCeleb speakers below 160 cm and 3,140 quality-controlled clips (6.92 hours) as noisy train-only support. A development-only comparison selected from 2,007 training-OOF candidates and changed validation MAE from 4.887 to 4.879 cm, a negligible 0.007 cm reduction. The work demonstrates that leakage controls, subgroup error analysis, label provenance, and uncertainty intervals are central to credible physical-trait inference from voice.

**Keywords:** speaker height estimation, WavLM, HeightCeleb, data collection, soft biometrics, data leakage, ensemble regression, TIMIT, NISP

## 1. Introduction

A speaker's voice contains information about the vocal source, vocal-tract geometry, speaking style, language, recording channel, and demographic background. Vocal-tract length and the resulting formant structure provide a plausible connection to body size, but that connection is noisy. Fundamental frequency is also associated with perceived body size, yet the mapping from acoustic cues to measured height is neither direct nor invariant across gender, age, language, phonetic content, and recording conditions.

Prior work has therefore treated speaker-height estimation as a statistical soft-biometric task rather than a physical measurement. Hansen, Williams, and Bořil reported 4.89 cm MAE for male speakers and 4.55 cm for female speakers on selected TIMIT phones, while warning that different data partitions prevent straightforward absolute comparison across studies. More recent work has used gender-specific transformer encoders and self-supervised representations for joint speaker age and height estimation.

VoxPhysica was developed to investigate whether richer speech representations and model fusion can reduce this error. Exploratory experiments in the repository produced many prediction views, including WavLM embeddings, physics-inspired acoustic features, neural regressors, and post-hoc ensembles. However, extensive reuse of a small test set created a central scientific risk: an apparently excellent score could reflect selection against test labels rather than generalization to unseen speakers.

This study has three objectives:

1. audit historical results and identify invalid evaluation paths;
2. implement a strict, reproducible speaker-level modeling pipeline; and
3. test the hypothesis that the frozen system can achieve MAE at or below 3.0 cm.

The 3.0 cm value is an experimental target, not a presumed outcome.

## 2. Related work

### 2.1 Acoustic height cues

Height-related speech studies commonly examine fundamental frequency, formants, formant spacing, vocal-tract length estimates, subglottal resonances, MFCCs, and LPC-derived features. Hansen et al. combined phone-conditioned spectral regression with statistical acoustic models and showed that complementary acoustic systems can improve TIMIT height estimation. Their reported MAEs remain an important reference, but their selected-phone protocol and partition must not be treated as identical to the present mixed-corpus setup.

### 2.2 Self-supervised speech representations

WavLM is a large-scale self-supervised model trained with masked speech prediction and denoising. Its representations contain linguistic, speaker, and paralinguistic information useful beyond automatic speech recognition. VoxPhysica uses already-extracted WavLM speaker embeddings as high-dimensional acoustic views. It does not infer that a general speech representation must encode height; the usefulness of each view is measured using out-of-fold speaker predictions.

Gupta et al. proposed gender-specific transformer encoders above wav2vec 2.0 features for age and height estimation. Their design supports the broader idea that male and female acoustic-height mappings can differ, although this study uses stratification and declared gender metadata rather than an end-to-end bi-encoder.

### 2.3 Evaluation leakage

Speaker-level regression requires speaker-level partitioning. Clip-level random splitting is invalid when several clips from the same person can enter both training and evaluation. A subtler failure occurs when test speakers are included in cross-validation or when an in-sample prediction becomes an input to a second-stage model. Both mechanisms reduce apparent error without demonstrating out-of-speaker generalization.

## 3. Data

### 3.1 Corpora and splits

The repository combines TIMIT and NISP speaker metadata. TIMIT is a phonetically rich American English corpus published by NIST. NISP contributes Indian-language and English recordings with speaker metadata. The frozen repository partitions contain:

| Split | Speakers |
|---|---:|
| Train | 775 |
| Validation | 97 |
| Historical test | 97 |
| Total | 969 |

The historical test split contains 63 TIMIT and 34 NISP speakers. The pipeline verifies that no speaker identifier occurs in more than one split and that height labels agree between split CSV files and every loaded embedding cache.

TIMIT heights are largely quantized in one-inch increments, represented as 2.54 cm multiples. NISP heights are recorded with different precision. This label-policy difference is modeled only through a validation-selected TIMIT grid rule; test labels are never used to decide whether that rule is active.

### 3.2 Input views

The pipeline discovers every complete cached WavLM directory containing train, validation, and test arrays. For the two complete views present in the repository, it builds:

- each original WavLM embedding;
- the elementwise mean;
- the elementwise difference;
- a concatenation of both views and their difference.

A metadata-assisted variant appends gender, corpus source, age, weight, missingness indicators, number of clips, NISP language family, and TIMIT dialect region. Results using these fields must be labeled **metadata-assisted**. A result that uses only acoustic views or the frozen Phase 12 acoustic system is labeled **voice-only**. The two scopes are not interchangeable.

### 3.3 Short-tail data expansion

The strict test error analysis identified speakers below 160 cm as the dominant failure slice. The original training split contains 78 such speakers: 73 female and 5 male. The local HeightCeleb/VoxCeleb1 manifest contains another 80 speakers below 160 cm, of whom 79 are female and 1 is male. Every public-support ID was checked against train, validation, and historical test; no speaker overlap was found.

HeightCeleb enriches VoxCeleb1 with height estimates collected from public web sources. Its authors explicitly describe measurement, rounding, temporal, and reporting uncertainty and recommend accurate measured data for testing. The present study therefore restricts HeightCeleb to noisy train support. It does not treat these labels as ground truth and does not redistribute VoxCeleb audio.

An automated PCM-WAV audit examined 3,168 candidate clips from the 80 short HeightCeleb speakers. It retained 3,140 clips totaling 6.92 hours and rejected 28 clips for duplicate content, near-silence, or duration above 30 seconds. All 80 speakers retained at least five valid clips. The resulting support manifest contains 855 training speakers: the original 775 plus 80 public short-tail speakers.

For prospective collection, the repository now includes an adult informed-consent template, direct-identifier prohibition, two-measurement height protocol, audio quality checks, participant-level role assignment, and separate development and sealed-test quotas. The pilot target is 120 measured development speakers and 80 measured sealed-test speakers below 160 cm, including at least 45 short male speakers across both roles.

## 4. Leakage audit

The file formerly named `scripts/final_ensemble.py` reported approximately 1.683 cm MAE. The audit found two disqualifying properties:

1. it concatenated train, validation, and test speakers before five-fold cross-validation; and
2. it added a VoxHeightNet prediction feature whose training-speaker values were generated by the same fitted network, making those values in-sample.

Consequently, the reported number answers neither the fixed-test question nor a fully out-of-fold all-speaker question. The original script is preserved as `archive/final_ensemble_all_data_cv_legacy.py` for provenance, and the public entry point now invokes the strict pipeline.

The Phase 22 reality gauntlet provides a more credible historical map. Its best validation-safe paired prediction was the Phase 12 baseline:

| Historical candidate | Validation MAE | Test MAE | Test within 3 cm | Interpretation |
|---|---:|---:|---:|---|
| Phase 12 baseline | 4.091 | 4.951 | 44.3% | Best validation-safe historical candidate |
| Phase 22 selected gate | 3.807 | 5.023 | 42.3% | Validation winner that did not transfer |
| Global per-speaker oracle | — | 1.946 | 75.3% | Diagnostic only; uses unavailable target knowledge |
| Former “final ensemble” | — | 1.683 | — | Invalid all-data/in-sample CV |

These test values are historical and repeatedly inspected. They are useful for audit and error analysis but are no longer pristine sealed-test estimates.

## 5. Method

### 5.1 Candidate models

For each acoustic view, the pipeline evaluates ridge regressors with supervised univariate feature selection over multiple feature counts and regularization strengths. It also evaluates PCA followed by an RBF support-vector regressor on the mean WavLM view.

The metadata-only family contains:

- robust-scaled ridge regression;
- histogram gradient boosting with absolute-error loss;
- extra-trees regression with leaf-size regularization; and
- a hierarchical median prior with shrinkage across gender, source, language, and dialect groups.

The full search uses approximately sixty model-view configurations. A reduced mode exists for smoke testing, but it still consumes a test evaluation and therefore is not used before the main frozen run.

### 5.2 Out-of-fold model selection

Training speakers are divided into source-and-gender-stratified folds. Each candidate produces one prediction for every training speaker from a model that did not train on that speaker. Candidate quality is measured using these out-of-fold predictions.

A shortlist is built by OOF MAE and prediction diversity. Highly correlated candidates are retained only when they contribute a new estimator family. Non-negative ensemble weights are then optimized under a sum-to-one constraint:

[
hat{h}_i = sum_{m=1}^{M} w_m hat{h}_{im},
qquad
w_m ge 0,
qquad
sum_{m=1}^{M} w_m = 1.
]

The objective is OOF MAE with a small penalty toward an inverse-error prior to reduce unstable weight concentration.

### 5.3 Frozen validation recipe

Only a limited, declared recipe set is compared on validation:

- the OOF convex ensemble;
- the three strongest OOF single models;
- an equal-weight top-three ensemble;
- raw, global-offset, group-offset, and TIMIT-grid postprocessors learned from training OOF residuals;
- the frozen Phase 12 prediction; and
- three fixed Phase 12/acoustic blend ratios.

The winner is frozen before test evaluation. If recipes lie within 0.03 cm validation MAE, the lower-complexity recipe is preferred.

### 5.4 Final fitting and evaluation

After recipe selection, train and validation are combined as development data. Selected components generate new development OOF predictions, final convex weights are estimated with stronger prior regularization, and postprocessing parameters are fitted from development OOF residuals. Components are then fitted on all development speakers and applied to test speakers.

For target height (h_i) and estimate (hat{h}_i), the primary metric is:

[
operatorname{MAE} =
rac{1}{N}sum_{i=1}^{N}|h_i-hat{h}_i|.
]

The pipeline also reports median absolute error, 90th-percentile absolute error, RMSE, and proportions within 3 cm and 4 cm. A nonparametric speaker bootstrap estimates a 95% confidence interval for MAE. Results are sliced by source, gender, and true-height range.

## 6. Frozen hypotheses and acceptance gates

The primary hypothesis is:

[
H_1: operatorname{MAE}_{test} le 3.0	ext{ cm}.
]

The point target is met when test MAE is at most 3.0 cm. A stronger publication gate requires the upper endpoint of the 95% bootstrap interval also to be at most 3.0 cm.

A positive claim additionally requires zero speaker overlap, finite predictions for all speakers, saved split/feature hashes, exact agreement between the prediction CSV and JSON metrics, and explicit voice-only or metadata-assisted scope.

## 7. Results

### 7.1 Historical error budget

The Phase 22 selected system accumulated approximately 487.3 cm of absolute error over 97 test speakers. A 3.0 cm MAE would permit 291.0 cm, requiring a reduction of about 196.3 cm. Fifteen of the worst speakers would need nearly perfect repair to close this gap under that model. Rare short speakers dominate several failures; examples include TIMIT_BCG1, TIMIT_DPK0, NISP_Tam_0012, and NISP_Mal_0008.

If all speakers below 160 cm were predicted perfectly, the selected model's MAE would fall to approximately 2.835 cm. This counterfactual identifies tail coverage as a major bottleneck, but it is not an achievable result.

### 7.2 Strict experiment

The frozen run evaluated 65 candidate model-view configurations and recorded zero candidate failures. The best WavLM-derived candidates remained weaker on validation than the existing Phase 12 reference, so the selector retained Phase 12 without a learned offset or test-informed correction.

| System | Scope | Frozen validation MAE | One-pass historical-test MAE | 95% MAE CI | Within 3 cm | 3 cm point gate |
|---|---|---:|---:|---:|---:|---|
| Strict frozen selector | Voice-only | 4.091 | 4.951 | 4.040–5.888 | 44.3% | No |

The prediction-level error profile was:

| Slice | Speakers | MAE (cm) | Within 3 cm |
|---|---:|---:|---:|
| All | 97 | 4.951 | 44.3% |
| NISP | 34 | 4.656 | 47.1% |
| TIMIT | 63 | 5.110 | 42.9% |
| Female | 37 | 5.114 | 37.8% |
| Male | 60 | 4.850 | 48.3% |
| Short, below 160 cm | 18 | 9.410 | 16.7% |
| Medium, 160–175 cm | 39 | 4.655 | 43.6% |
| Tall, at least 175 cm | 40 | 3.233 | 57.5% |

Overall median absolute error was 3.745 cm, RMSE was 6.767 cm, and the 90th-percentile absolute error was 10.225 cm. The independent verifier reproduced MAE from the 97-row prediction CSV and matched every recorded split and feature hash. The result is a one-pass evaluation under the new script, but the split remains a historically inspected test set rather than a newly sealed external benchmark.

### 7.3 Development-only public-support experiment

A separate experiment tested whether the 80 public short-tail speakers improve a lightweight cached-feature regressor. For every target-training fold, models were fitted to the remaining target speakers with or without weighted HeightCeleb support. The search evaluated 2,007 ridge regularization, public-label weight, prediction-gate, threshold, scale, and blend configurations. Selection used only target-training OOF predictions. The selected recipe was frozen before one validation evaluation, and the script had no historical-test input.

| Development system | Train OOF MAE | Train OOF short MAE | Validation MAE | Validation short MAE | Validation within 3 cm |
|---|---:|---:|---:|---:|---:|
| Target-only ridge | 5.649 | 9.201 | 4.887 | 4.778 | 34.0% |
| Weighted/gated HeightCeleb support | 5.638 | 9.000 | 4.879 | 4.768 | 36.1% |

The selected public-support recipe reduced validation MAE by only 0.007 cm and validation short-slice MAE by 0.010 cm. A 10,000-sample paired speaker bootstrap placed the 95% interval for the overall MAE difference at -0.051 to +0.033 cm, crossing zero. This magnitude is negligible relative to sampling uncertainty and does not alter the 4.951 cm historical-test result. It does not support a claim that the 3 cm target was met.

## 8. Discussion

The audit changes the interpretation of VoxPhysica's historical numbers. The 1.683 cm result shows that prediction errors from different systems are complementary, but it does not establish generalization. The validation-to-test reversal of the Phase 22 gate, from 3.807 cm to 5.023 cm, illustrates the variance produced by selecting among many recipes on only 97 validation speakers. In the strict run, refusing that unstable validation gain caused the selector to retain Phase 12 and reproduce 4.951 cm rather than manufacture an apparent improvement.

A credible 3 cm result is difficult for three reasons. First, physiological cues are indirect and affected by phonetic content. Second, the height distribution has sparse tails, so regression models shrink rare short and tall speakers toward group means. Third, corpus and label differences can become easier to learn than height itself. Metadata may improve numerical MAE, especially when weight is available, but such a system answers a broader anthropometric question and must not be marketed as voice-only estimation.

The public-support experiment sharpens the data diagnosis. More short clips are not equivalent to more reliable biological variation: 79 of the 80 added people are female, labels are web estimates, and VoxCeleb's interview audio differs from TIMIT and NISP. A 0.007 cm validation change after adding 80 people is consistent with a support set that is real but poorly matched to the critical subgroup and target domains.

The most valuable next step after a failed strict run is not another test-guided blend. It is a new untouched external set with accurately measured heights, deliberate tail sampling, repeated recording sessions, and balanced channel/language conditions. Nested cross-validation can then be used for model development before a single external evaluation.

## 9. Limitations and ethics

The historical test set has been inspected by many prior experiments and cannot provide the same evidential strength as a newly collected sealed set. NISP and TIMIT differ in language, recording process, metadata availability, and height-label precision. Weight is missing for TIMIT, making missingness itself source-informative. TIMIT label-grid snapping may improve corpus-specific MAE without improving continuous real-world height estimation.

Height inferred from voice is uncertain sensitive-trait inference. It should not be used to identify a person, establish guilt, make employment or insurance decisions, or present a precise physical measurement. Any operational system should report uncertainty, support abstention, obtain appropriate consent, and undergo demographic and cross-domain bias evaluation.

The public HeightCeleb metadata are licensed separately from VoxCeleb audio, and the height labels are estimates rather than measured clinical-grade values. Prospective VoxPhysica recruitment must be adult-only, institutionally approved, pseudonymous, voluntary, and governed by a documented withdrawal and retention process. Synthetic transformations and repeated clips must never be reported as additional people.

## 10. Reproducibility

Run the focused tests:

    python -m pytest tests/test_strict_height_pipeline.py -q

Run the full frozen experiment once:

    python scripts/run_strict_3cm_research.py --output-dir outputs/strict_3cm_research

Audit public short-tail support and run the development-only comparison:

    python scripts/collect_short_speaker_data.py
    python scripts/evaluate_short_support_dev.py

The strict pipeline writes a full metric JSON, validation and test predictions, a model bundle, a Markdown report, and SHA-256 hashes for all split and WavLM input files. The short-data pipeline writes accepted/rejected speaker and clip manifests, audio hashes and QC metrics, a combined train-support manifest, and a development-only result. The companion protocols are `research/VOXPHYSICA_3CM_RESEARCH_PLAN.md` and `research/SHORT_SPEAKER_COLLECTION_PROTOCOL.md`.

## 11. Conclusion

VoxPhysica's verified strict result is 4.951 cm MAE, not 1.683 cm. The 3.0 cm target was not met: its point estimate missed by 1.951 cm, and even the lower endpoint of the 95% interval exceeded 4.0 cm. The follow-up collection audit added 80 real short speakers and 3,140 valid train-support clips, but the public labels and extreme 79-to-1 female/male imbalance yielded only a negligible 0.007 cm validation reduction. The revised system supplies a materially stronger research basis through speaker-disjoint folds, out-of-fold selection, label-provenance controls, participant-level data accounting, audio QC, uncertainty intervals, subgroup analysis, and reproducibility hashes. A credible next attempt requires consented measured short-tail data—especially short male speakers—and a newly sealed external test set, not another round of selection against the 97 known test labels.

## References

1. J. H. L. Hansen, K. Williams, and H. Bořil, “Speaker height estimation from speech: Fusing spectral regression and statistical acoustic models,” *Journal of the Acoustical Society of America*, 138(2), 1052–1067, 2015. [doi:10.1121/1.4927554](https://doi.org/10.1121/1.4927554)

2. S. Chen et al., “WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing,” *IEEE Journal of Selected Topics in Signal Processing*, 2022. [doi:10.1109/JSTSP.2022.3188113](https://doi.org/10.1109/JSTSP.2022.3188113)

3. T. Gupta, T. D. Truong, T. T. Anh, and E. S. Chng, “Estimation of speaker age and height from speech signal using bi-encoder transformer mixture model,” *Interspeech 2022*, pp. 1978–1982. [doi:10.21437/Interspeech.2022-567](https://doi.org/10.21437/Interspeech.2022-567)

4. J. S. Garofolo, L. F. Lamel, W. M. Fisher, J. G. Fiscus, D. S. Pallett, and N. L. Dahlgren, “DARPA TIMIT Acoustic-Phonetic Continuous Speech Corpus CD-ROM,” NISTIR 4930, 1993. [NIST publication](https://www.nist.gov/publications/darpa-timit-acoustic-phonetic-continuous-speech-corpus-cd-rom-timit)

5. S. Kacprzak and K. Kowalczyk, “HeightCeleb—An Enrichment of VoxCeleb Dataset With Speaker Height Information,” *2024 IEEE Spoken Language Technology Workshop*, pp. 857–862, 2024. [doi:10.1109/SLT61566.2024.10832224](https://doi.org/10.1109/SLT61566.2024.10832224)

6. A. Nagrani, J. S. Chung, and A. Zisserman, “VoxCeleb: A Large-Scale Speaker Identification Dataset,” *Interspeech 2017*, pp. 2616–2620. [Official dataset page](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/)
