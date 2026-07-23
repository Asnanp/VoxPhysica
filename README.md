# 🎙️ VoxPhysica

> **Leakage-Resistant Speaker Physical Attribute Estimation using Physics-Informed Bayesian Deep Learning and Multi-View Self-Supervised Speech Representations (WavLM).**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![WavLM SSL](https://img.shields.io/badge/SSL-WavLM-green.svg)](https://github.com/microsoft/unilm/tree/master/wavlm)
[![Evaluation: Leakage--Resistant](https://img.shields.io/badge/Evaluation-Leakage--Resistant-brightgreen.svg)](#-leakage-audit--scientific-integrity)

---

## 📌 Executive Summary

**VoxPhysica** is an advanced AI and acoustic physics framework engineered to estimate human physical attributes—specifically **height (cm)**, **weight (kg)**, **age (years)**, and **gender**—directly from raw speech audio.

By combining **self-supervised learning (WavLM)** embeddings with **anatomical physics constraints** (vocal tract length and formant dispersions) and **Bayesian uncertainty quantification**, VoxPhysica maps subtle paralinguistic acoustic cues to physiological body metrics with verified probabilistic confidence bounds.

---

## 🌟 Key Features & Core Innovations

- 🔬 **Physics-Informed Loss Constraints (PIBNN)**  
  Embeds physiological laws linking Vocal Tract Length (VTL) to anatomical height ($\text{VTL} \approx \text{Height} / 6.7$) and formant spacing ($\Delta f = c / 2L$) directly into PyTorch loss objectives.
- 🔒 **Strict Speaker-Disjoint Leakage Control**  
  Guarantees zero speaker overlap across Train, Validation, and Test sets (969 unique speakers across TIMIT & NISP corpora), eliminating data contamination pitfalls.
- ⚡ **Multi-View Self-Supervised Fusion**  
  Leverages deep multi-layer WavLM representations combined with hand-crafted acoustic features, regularized ridge regressors, extra-trees, and out-of-fold (OOF) convex ensembling.
- 📊 **Bayesian Uncertainty Quantification**  
  Uses Monte Carlo Dropout and Variational Inference to provide non-parametric 95% bootstrap confidence intervals for every physical prediction.
- 🎯 **Tail-Data Subgroup Optimization**  
  Integrates a quality-controlled short-tail speaker expansion pipeline (**80 HeightCeleb/VoxCeleb1 speakers < 160 cm**, 3,140 valid clips, 6.92 hours of audio) to tackle tail estimation errors.

---

## 📊 Verified Research Benchmarks & Current Status

### Held-Out Speaker Test Results (Frozen Run)

VoxPhysica establishes a defensible, leakage-proof baseline on held-out test speakers:

| Metric | Verified Value | Target Goal | Status |
|--------|---------------|-------------|--------|
| **Speaker-Level MAE** | **4.951 cm** | $< 3.0 \text{ cm}$ | Active Research |
| **95% Bootstrap CI** | **4.040 – 5.888 cm** | $< 3.0 \text{ cm}$ | Active Research |
| **Within 3.0 cm Ratio** | **44.3%** | $> 70.0\%$ | In Progress |
| **Within 4.0 cm Ratio** | **57.7%** | $> 80.0\%$ | In Progress |
| **Median Absolute Error** | **3.745 cm** | $< 2.5 \text{ cm}$ | In Progress |
| **Overall RMSE** | **6.767 cm** | $< 4.5 \text{ cm}$ | In Progress |

### 🔍 Error Breakdown by Subgroup & Corpus

Evaluation across demographic slices reveals clear physiological and corpus characteristics:

```
┌───────────────────────────┬──────────────┬───────────────┬──────────────────┐
│ Subgroup Slice            │ Speaker Count│ Test MAE (cm) │ Within 3.0 cm %  │
├───────────────────────────┼──────────────┼───────────────┼──────────────────┤
│ All Test Speakers         │      97      │    4.951 cm   │      44.3%       │
│ NISP Corpus               │      34      │    4.656 cm   │      47.1%       │
│ TIMIT Corpus              │      63      │    5.110 cm   │      42.9%       │
│ Male Speakers             │      60      │    4.850 cm   │      48.3%       │
│ Female Speakers           │      37      │    5.114 cm   │      37.8%       │
│ Short (< 160 cm)          │      18      │    9.410 cm   │      16.7%       │
│ Medium (160 – 175 cm)     │      39      │    4.655 cm   │      43.6%       │
│ Tall (≥ 175 cm)           │      40      │    3.233 cm   │      57.5%       │
└───────────────────────────┴──────────────┴───────────────┴──────────────────┘
```

---

## 🛡️ Leakage Audit & Scientific Integrity

> ⚠️ **Important Audit Note regarding Legacy Benchmarks:**
> Prior repository experiments (e.g., `scripts/final_ensemble.py`) reported an unverified $1.683 \text{ cm}$ MAE score. A rigorous audit revealed that this value resulted from **all-data cross-validation** combining train, val, and test speakers alongside an in-sample neural prediction feature.
> 
> VoxPhysica rejects contaminated evaluations. The official, verified, leak-free benchmark is **4.951 cm speaker MAE**. All future 3.0 cm and 4.0 cm breakthrough claims are strictly validated against frozen, speaker-disjoint test sets.
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

*Note: Waist circumference and shoulder width are explicitly out of scope.*

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
│   │   ├── pibnn.py          # Physics-Informed Bayesian Neural Network
│   │   └── ecapa.py          # ECAPA-TDNN speaker embedding backbone
│   │
│   ├── research/             # Core leak-free research pipeline engines
│   │   ├── strict_height_pipeline.py  # Frozen 5-fold OOF evaluator
│   │   ├── speaker_height_ensemble.py # Out-of-fold convex ensembling
│   │   └── short_data_collection.py   # HeightCeleb short-tail audio collector
│   │
│   ├── preprocessing/        # Feature extraction & data loading
│   ├── training/             # Loss functions, schedulers & trainers
│   ├── inference/            # Real-time speaker inference pipeline
│   └── utils/                # Metrics, visualization, and audit helpers
│
├── scripts/                  # Automated execution pipelines
│   ├── run_4cm_breakthrough_pipeline.py # 4.0cm MAE breakthrough executor
│   ├── run_strict_3cm_research.py       # Strict leak-free 3.0cm evaluator
│   ├── start_two_cm_push_live.py        # Live detached runner with log tail
│   ├── collect_short_speaker_data.py    # Short-speaker data expansion script
│   └── evaluate_short_support_dev.py    # Support development evaluator
│
├── configs/                  # Experiment configuration files (YAML)
├── research/                 # Research papers & frozen plans
│   ├── VOXPHYSICA_RESEARCH_PAPER.md
│   ├── VOXPHYSICA_3CM_RESEARCH_PLAN.md
│   └── SHORT_SPEAKER_COLLECTION_PROTOCOL.md
│
├── outputs/                  # Saved checkpoints, predictions & metrics JSONs
└── tests/                    # Unit & integration test suites
```

---

## ⚡ Tech Stack

- **Core Deep Learning**: PyTorch 2.x, PyTorch Lightning
- **Bayesian Inference**: Pyro (Uber AI)
- **Speech Representations**: Microsoft WavLM Large / Base SSL
- **Acoustic Signal Processing**: librosa, Praat Parselmouth, openSMILE, torchaudio
- **Machine Learning & Ensembling**: scikit-learn, XGBoost, CatBoost, Optuna
- **Data Engineering**: pandas, numpy, scipy
- **Testing & Quality Assurance**: pytest, pytest-cov

---

## 🚀 Quick Start & Environment Setup

### 1. Installation

```bash
# Clone repository
git clone https://github.com/Asnanp/VoxPhysica.git
cd VoxPhysica/VoxPhysica-main

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

### A. Run 4.0 cm MAE Breakthrough Pipeline

Executes multi-SSL feature extraction, 5-fold OOF cross-validation, convex meta-ensembling, and residual calibration evaluated on the sealed test set:

```powershell
python scripts/run_4cm_breakthrough_pipeline.py --output-dir outputs/4cm_breakthrough
```

### B. Run Strict Leak-Free 3.0 cm Research Gauntlet

Runs the 65-configuration leak-resistant evaluation pipeline:

```powershell
python scripts/run_strict_3cm_research.py --output-dir outputs/strict_3cm_research
```

### C. Launch Live 2.0 cm Push (Detached Process)

Launches background training with live log tailing:

```powershell
python scripts/start_two_cm_push_live.py --seed 11 --device cuda
```

### D. Short-Speaker Data Expansion Audit

Audits and collects 3,140 clips from 80 short-tail speakers (< 160 cm):

```powershell
python scripts/collect_short_speaker_data.py
python scripts/evaluate_short_support_dev.py
```

### E. Run Unit Tests

```bash
pytest tests/ -v
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
