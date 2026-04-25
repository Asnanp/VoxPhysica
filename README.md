# 🎙️ VocalMorph

> **Predicting human physical attributes — height, weight, age, and gender — directly from voice using Physics-Informed Bayesian Neural Networks.**

Built by [Asnan P](https://asnanp.netlify.app) | GitHub: [@Asnanp](https://github.com/Asnanp)

---

## 🧠 What is VocalMorph?

VocalMorph is a deep learning system that extracts physical biometric information from raw audio. By analyzing acoustic features like formant frequencies, pitch, vocal tract resonances, and spectral characteristics, VocalMorph estimates:

| Target | Description |
|--------|-------------|
| 🏃 **Height** | Estimated in cm (continuous regression) |
| ⚖️ **Weight** | Estimated in kg (continuous regression) |
| 🎂 **Age** | Estimated in years (continuous regression) |
| 🧬 **Gender** | Male / Female / Other (classification) |

> ⚠️ **Scope Note:** VocalMorph does **NOT** predict waist circumference or shoulder width. The model is scoped to the four targets above only.

---

## 🔬 Why Physics-Informed Bayesian Neural Networks?

Standard neural networks for this task suffer from two major problems:
1. **Black-box predictions** — no uncertainty quantification
2. **Ignoring acoustic physics** — vocal tract length is directly correlated with height via physical laws

VocalMorph solves this by embedding **acoustic physics as hard constraints** into the loss function, combined with **Bayesian inference** for probabilistic output — so every prediction comes with a confidence interval.

### Physics Embedded:
- **Vocal Tract Length (VTL) ↔ Height**: `VTL ≈ Height / 6.7` (Fitch, 2000)
- **Formant spacing ↔ VTL**: `Δf = c / (2 × VTL)` where c = speed of sound
- **Fundamental frequency (F0) ↔ Age/Gender**: hormonal physiology constraints

---

## 📁 Project Structure

```
VocalMorph/
├── data/
│   ├── raw/            # Original audio files (.wav, .flac)
│   ├── nisp/           # NISP dataset (primary training corpus)
│   ├── processed/      # Cleaned, normalized audio
│   ├── augmented/      # Time-stretch, pitch-shift, noise augmented
│   └── features/       # Extracted MFCC, formants, spectral features
│
├── src/
│   ├── preprocessing/  # Audio loading, feature extraction pipeline
│   ├── models/         # PIBNN architecture definitions
│   ├── training/       # Training loops, loss functions, schedulers
│   ├── inference/      # Real-time prediction pipeline
│   └── utils/          # Helpers, metrics, visualization
│
├── configs/            # YAML config files for experiments
├── notebooks/          # EDA, prototyping, result visualization
├── experiments/        # Experiment tracking logs
├── outputs/
│   ├── checkpoints/    # Saved model weights
│   ├── logs/           # Training logs (TensorBoard/W&B)
│   └── predictions/    # Inference output CSVs
├── docs/               # Architecture diagrams, research notes
├── tests/              # Unit tests
└── scripts/            # Training/eval/inference run scripts
```

---

## 🗃️ Dataset: NISP (National Institute of Speech and Perception)

The **NISP dataset** is the primary training corpus for VocalMorph.

- Contains labeled audio recordings with speaker metadata
- Metadata includes: **height, weight, age, gender**
- Audio format: `.wav`, 16kHz sampling rate
- Located in: `data/nisp/`

### NISP Data Structure Expected:
```
data/nisp/
├── audio/
│   ├── speaker_001.wav
│   ├── speaker_002.wav
│   └── ...
└── metadata.csv   # columns: speaker_id, height_cm, weight_kg, age, gender
```

---

## ⚙️ Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.10+ |
| Deep Learning | PyTorch 2.x |
| Bayesian Inference | Pyro (Uber) |
| Audio Processing | librosa, torchaudio |
| Feature Extraction | openSMILE, parselmouth (Praat) |
| Experiment Tracking | Weights & Biases (W&B) |
| Config Management | Hydra + OmegaConf |
| Data Processing | pandas, numpy, scipy |
| Visualization | matplotlib, seaborn, plotly |
| Testing | pytest |

---

## 🛣️ Project Roadmap

### ✅ Phase 0 — Setup (Current)
- [x] Project structure initialized
- [x] README written
- [x] Config system setup
- [ ] Environment setup (`requirements.txt`)
- [ ] NISP dataset downloaded and verified

### 🔄 Phase 1 — Data Pipeline
- [ ] Audio loading & validation script
- [ ] EDA on NISP dataset (distribution of height/weight/age/gender)
- [ ] Feature extraction pipeline:
  - MFCCs (13-40 coefficients)
  - Formant frequencies (F1-F4)
  - Fundamental frequency (F0/pitch)
  - Spectral centroid, rolloff, flux
  - Vocal tract length estimation
  - Jitter, shimmer (voice quality)
- [ ] Data augmentation pipeline
- [ ] Train/val/test split (70/15/15 stratified by gender)

### 🔄 Phase 2 — Baseline Models
- [ ] Simple CNN baseline on spectrograms
- [ ] LSTM baseline on temporal features
- [ ] XGBoost baseline on handcrafted features
- [ ] Establish baseline MAE for each target

### 🔄 Phase 3 — PIBNN Architecture
- [ ] Design physics-informed loss function
- [ ] Implement Bayesian layers (MC Dropout or Variational Inference)
- [ ] Build multi-task output head (3 regression + 1 classification)
- [ ] Physics constraint integration (VTL-height, formant-VTL)
- [ ] Uncertainty calibration

### 🔄 Phase 4 — Training & Optimization
- [ ] Hyperparameter sweep (Optuna/W&B Sweeps)
- [ ] Learning rate scheduling
- [ ] Regularization (dropout, weight decay, physics penalty weight tuning)
- [ ] Model checkpointing
- [ ] Target: Height MAE < 3cm, Weight MAE < 5kg, Age MAE < 5yr, Gender Acc > 95%

### 🔄 Phase 5 — Inference Pipeline
- [ ] Real-time voice capture (microphone input)
- [ ] End-to-end prediction with uncertainty bounds
- [ ] REST API (FastAPI)
- [ ] Demo UI

### 🔄 Phase 6 — Evaluation & Paper
- [ ] Comprehensive evaluation on held-out test set
- [ ] Ablation studies (with/without physics constraints)
- [ ] Uncertainty calibration plots
- [ ] Write research paper / technical report

---

## 🚀 Quick Start

```bash
# Clone the repo
git clone https://github.com/Asnanp/VocalMorph.git
cd VocalMorph

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Place NISP data in data/nisp/
# Then run feature extraction
python scripts/extract_features.py

# Train the model
python scripts/train.py --config configs/pibnn_base.yaml
```

---

## 📊 Target Metrics

| Target | Metric | Goal |
|--------|--------|------|
| Height | MAE (cm) | < 3.0 cm |
| Weight | MAE (kg) | < 5.0 kg |
| Age | MAE (years) | < 5.0 yrs |
| Gender | Accuracy | > 95% |

---

## 📚 Key References

- Fitch, W. T. (2000). Vocal tract length and formant frequency dispersion correlate with body size in rhesus macaques. *JASA*
- Raine, C. et al. — Speaker physical attribute estimation from voice
- NISP Dataset documentation
- Pyro: Deep Universal Probabilistic Programming (Bingham et al., 2019)

---

## 👤 Author

**Asnan P** — ML Developer, Kerala, India  
🌐 [asnanp.netlify.app](https://asnanp.netlify.app) | 🐙 [GitHub](https://github.com/Asnanp)

> *"Strategic silence, long-term planning, controlled output — then dominate."*
