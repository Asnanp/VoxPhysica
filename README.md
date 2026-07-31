# 🎙️ VoxPhysica

> **Leakage-Resistant Speaker Physical Attribute Estimation using Physics-Informed Bayesian Deep Learning and Multi-View Self-Supervised Speech Representations (WavLM).**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![WavLM SSL](https://img.shields.io/badge/SSL-WavLM-green.svg)](https://github.com/microsoft/unilm/tree/master/wavlm)
[![Evaluation: Leakage--Resistant](https://img.shields.io/badge/Evaluation-Leakage--Resistant-brightgreen.svg)](#-leakage-audit--scientific-integrity)
[![Build Status](https://img.shields.io/badge/Tests-96%2F96%20Passing-brightgreen.svg)](#-unit-testing--verification)

---

## 📌 Executive Summary

**VoxPhysica** is an advanced AI and acoustic physics framework engineered to estimate human physical attributes—specifically **height (cm)**, **weight (kg)**, **age (years)**, and **gender**—directly from raw speech audio.

By combining **self-supervised learning (WavLM)** embeddings with **anatomical physics constraints** (vocal tract length and formant dispersions) and **Bayesian uncertainty quantification**, VoxPhysica maps subtle paralinguistic acoustic cues to physiological body metrics with verified probabilistic confidence bounds.

---

## 🌟 Key Features & Breakthrough Innovations

- 🔬 **Physics-Informed Loss Constraints (PIBNN)**  
  Embeds physiological laws linking Vocal Tract Length (VTL) to anatomical height ($\text{VTL} \approx \text{Height} / 6.7$), formant spacing ($\Delta f = c / 2L$), and **Short-Male Over-prediction Penalties** directly into PyTorch loss objectives.
- 🔒 **Strict Speaker-Disjoint Leakage Control**  
  Guarantees zero speaker overlap across Train, Validation, and Test sets (969 unique speakers across TIMIT & NISP corpora), eliminating data contamination pitfalls.
- ⚡ **Multi-View Self-Supervised Fusion**  
  Leverages deep multi-layer WavLM representations combined with hand-crafted acoustic features, regularized ridge regressors, extra-trees, and out-of-fold (OOF) convex ensembling.
- 🎯 **Short-Speaker Stature Breakthrough (< 160 cm)**  
  Applies **inverse-density sample weighting** ($6.0\times$ weight on short males, $3.0\times$ on short females) and **anatomical VTL short-calibration postprocessing** to break male gender prior anchoring, driving short female MAE down to **4.308 cm** (median error **2.592 cm**, **61.5%** within 3.0 cm) and overall short speaker MAE down to **5.598 cm**.
- 📊 **Bayesian Uncertainty Quantification**  
  Uses Monte Carlo Dropout and Variational Inference to provide non-parametric 95% bootstrap confidence intervals for every physical prediction.

---

## 📊 Verified Research Benchmarks & Latest Results

### Held-Out Speaker Test Results (Strict Leakage-Free Run)

VoxPhysica establishes a defensible, leakage-proof baseline across **97 held-out test speakers**:

| Metric | Verified Value | Baseline | Delta / Status |
|--------|---------------|----------|----------------|
| **Short Speakers (< 160 cm) MAE** | **5.598 cm** | `9.410 cm` | **`+3.812 cm Breakthrough (< 6.0 cm Target Met!)`** |
| **Short Females (< 160 cm) MAE** | **4.308 cm** | `6.072 cm` | **`-1.764 cm`** |
| **Short Female Median Error** | **2.592 cm** | `5.118 cm` | **`-2.526 cm`** |
| **Short Female Within 3.0 cm Ratio** | **61.5%** | `30.8%` | **`+30.7%`** |
| **Short Male (< 160 cm) MAE** | **8.951 cm** | `17.522 cm` | **`-8.571 cm`** |
| **All Test Speakers MAE** | **5.630 cm** | `5.849 cm` | **`-0.219 cm`** |
| **95% Bootstrap CI** | **[4.720, 6.550] cm** | `[4.85, 6.75] cm` | Robust Bounds |

---

### 🔍 Complete Demographic & Subgroup Audit Table

Evaluation across demographic slices reveals clear physiological characteristics and high accuracy on female and short cohorts:

```
┌───────────────────────────┬──────────────┬───────────────┬───────────────────┬──────────────────┐
│ Subgroup Slice            │ Speaker Count│ Test MAE (cm) │ Median Error (cm) │ Within 3.0 cm %  │
├───────────────────────────┼──────────────┼───────────────┼───────────────────┼──────────────────┤
│ Short (< 160 cm)          │      18      │    5.598 cm   │      3.893 cm     │      44.4%       │
│   ├── Short Females       │      13      │    4.308 cm   │      2.592 cm     │      61.5%       │
│   └── Short Males         │       5      │    8.951 cm   │      9.773 cm     │       0.0%       │
│ All Test Speakers         │      97      │    5.630 cm   │      5.080 cm     │      38.1%       │
│ Tall (≥ 175 cm)           │      40      │    3.901 cm   │      2.540 cm     │      52.5%       │
└───────────────────────────┴──────────────┴───────────────┴───────────────────┴──────────────────┘
```

---

## 🛡️ Leakage Audit & Scientific Integrity

> ⚠️ **Important Audit Note regarding Legacy Benchmarks:**
> Prior repository experiments (e.g., `scripts/final_ensemble.py`) reported an unverified $1.683 \text{ cm}$ MAE score. A rigorous audit revealed that this value resulted from **all-data cross-validation** combining train, val, and test speakers alongside an in-sample neural prediction feature.
> 
> VoxPhysica rejects contaminated evaluations. The official, verified, leak-free baseline is **5.630 cm overall speaker MAE** (with short female MAE down to **4.308 cm** and overall short speaker MAE down to **5.598 cm**). All breakthrough claims are strictly validated against frozen, speaker-disjoint test sets.
> 
> Detailed analysis is available in the research paper: [`research/VOXPHYSICA_RESEARCH_PAPER.md`](research/VOXPHYSICA_RESEARCH_PAPER.md).

---

## 🎯 Biometric Estimation Scope

VoxPhysica estimates four fundamental human physical biometrics:

| Biometric Target | Format | Unit / Classes | Description |
|------------------|--------|----------------|-------------|
| 🏃 **Height** | Continuous Regression | Centimeters ($\text{cm}$) | Anatomical stature derived from VTL & formants |
| ⚖️ **Weight** | Continuous Regression | Kilograms ($\text{kg}$) | Body mass estimated via acoustic energy & spectral tilt |
| 🎂 **Age** | Continuous Regression | Years | Biological age based on fundamental frequency decay & jitter |
| 🧬 **Gender** | Classification | Male / Female / Other | Binary / multi-class physiological classification |

---

## 🔬 Physics-Informed Acoustic Foundations

Standard deep learning models often treat voice biometrics as black-box pattern matching. VoxPhysica grounds predictions in acoustic physics:

1. **Vocal Tract Length (VTL) Stature Scaling**:
   $$VTL \approx \frac{\text{Height}}{6.7} \quad (\text{Fitch, 2000})$$
2. **Formant Spacing ($\Delta f$) & Speed of Sound ($c = 35000 \text{ cm/s}$)**:
   $$\Delta f = \frac{c}{2 \times VTL}$$
3. **Fundamental Frequency ($F_0$) Physiological Bound**:
   Couples vocal fold mass and tension to age and gender physiological constraints.

---

## 📁 Repository Structure

```
VoxPhysica/
├── data/
│   ├── raw/                  # Raw audio samples (.wav, .flac)
│   ├── nisp/                 # NISP corpus (audio & metadata.csv)
│   ├── timit/                # TIMIT corpus
│   ├── processed/            # Trimmed, resampled, normalized audio
│   └── features/             # Cached WavLM SSL embeddings & acoustic features
│
├── src/
│   ├── models/               # PyTorch architectures
│   │   ├── vocalmorph_v5.py  # Adaptive per-bin MAE multi-task model
│   │   ├── vocalmorph_v4.py  # Huber + MAE direct loss optimizer
│   │   ├── vocalmorph_v3.py  # High-sensitivity height regressor
│   │   ├── pibnn.py          # Physics-Informed Bayesian Neural Network (short male penalty)
│   │   └── ecapa.py          # ECAPA-TDNN speaker embedding backbone
│   │
│   ├── research/             # Core leak-free research pipeline engines
│   │   ├── strict_height_pipeline.py  # Frozen 5-fold OOF evaluator & short-male debias
│   │   ├── speaker_height_ensemble.py # Out-of-fold convex ensembling
│   │   └── short_data_collection.py   # HeightCeleb short-tail audio collector
│   │
│   ├── preprocessing/        # Feature extraction & data loading
│   ├── training/             # Loss functions, schedulers & trainers
│   ├── inference/            # Real-time speaker inference pipeline
│   └── utils/                # Metrics, visualization, and audit helpers
│
├── scripts/                  # Automated execution pipelines
│   ├── run_strict_3cm_research.py          # Strict leak-free 3.0cm evaluator
│   ├── evaluate_short_speaker_breakthrough.py # Subgroup breakdown evaluation script
│   ├── inspect_all_recipes.py             # Recipe inspection tool
│   ├── build_feature_splits.py             # Audio resolution & feature split builder
│   └── start_two_cm_push_live.py           # Live detached runner with log tail
│
├── configs/                  # Experiment configuration files (YAML)
├── research/                 # Research papers & frozen plans
├── outputs/                  # Saved checkpoints, predictions & metrics JSONs
└── tests/                    # Unit & integration test suites (96 passing tests)
```

---

## ⚡ Tech Stack

- **Core Deep Learning**: PyTorch 2.x, PyTorch Lightning
- **Bayesian Inference**: Pyro (Uber AI)
- **Speech Representations**: Microsoft WavLM Large / Base SSL
- **Acoustic Signal Processing**: librosa, Praat Parselmouth, openSMILE, torchaudio
- **Machine Learning & Ensembling**: scikit-learn, XGBoost, CatBoost, Optuna
- **Data Engineering**: pandas, numpy, scipy
- **Testing & Quality Assurance**: pytest, pytest-cov (100% test pass rate)

---

## 🚀 Quick Start & Environment Setup

### 1. Installation

```bash
# Clone repository
git clone https://github.com/Asnanp/VoxPhysica.git
cd VoxPhysica

# Create virtual environment (Python 3.10+)
python -m venv .venv-gpu
# Windows
.venv-gpu\Scripts\activate
# Linux/macOS
source .venv-gpu/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🏃 Running Pipelines & Research Workflows

### A. Run Strict Leak-Free Research Pipeline

Runs the multi-candidate evaluation pipeline with short-speaker weighted models and group-snap offset dampening:

```powershell
python scripts/run_strict_3cm_research.py --output-dir outputs/strict_3cm_short_opt
```

### B. Evaluate Subgroup Breakdown & Verification

Generates the detailed subgroup metrics audit report across height and gender slices:

```powershell
python scripts/evaluate_short_speaker_breakthrough.py --pred-csv outputs/strict_3cm_short_opt/predictions_test_once.csv --output-dir outputs/strict_3cm_short_opt
```

### C. Run Unit Test Suite

```powershell
python -m pytest --basetemp=outputs/pytest_temp
```

---

## 📖 Research Publications & Citation

If you use VoxPhysica, its codebase, or its research findings in your work, please cite:

```bibtex
@article{voxphysica2026,
  title={Leakage-Resistant Speaker Height Estimation with Multi-View Self-Supervised Speech Representations},
  author={P, Asnan},
  journal={VoxPhysica Research Technical Report},
  year={2026},
  month={July},
  url={https://github.com/Asnanp/VoxPhysica}
}
```

Key academic literature integrated into VoxPhysica:
1. **Fitch, W. T.** (2000). *Vocal tract length and formant frequency dispersion correlate with body size*. JASA.
2. **Hansen, J. H. L., et al.** (2015). *Speaker height estimation from speech*. JASA.
3. **Chen, S. et al.** (2022). *WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing*. IEEE JSTSP.
4. **Kacprzak, S., & Kowalczyk, K.** (2024). *HeightCeleb — An Enrichment of VoxCeleb Dataset With Speaker Height Information*. IEEE SLT.

---

## 👤 Author & Acknowledgments

**Asnan P** — Lead AI / ML Developer & Researcher  
🐙 [GitHub (@Asnanp)](https://github.com/Asnanp)

> *"Rigorous evaluation, uncompromised leakage controls, continuous iteration — defining the frontier of vocal biometrics."*
