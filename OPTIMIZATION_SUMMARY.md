# VoxPhysica Height Prediction Optimization Summary

## Objective
Achieve 3cm MAE (ideally 2cm) from current 5.94cm test MAE for voice-based height prediction.

## Current Performance Baseline
- Test height_mae_speaker: 5.94cm
- Test height_mae: 7.09cm  
- Short-height performance: 9.7cm MAE (very poor)
- Train-val gap: +2.3cm (generalization issue)

## Implemented Improvements

### Phase 1: Model Architecture Enhancement ✅
**File: `configs/pibnn_base.yaml`**
- Increased ECAPA channels: 128→256
- Increased Conformer blocks: 1→3
- Increased conformer_d_model: 64→128
- Increased conformer_heads: 4→8
- Increased dropout: 0.15→0.20
- Changed model type: strict_small_physics→strict_large_physics
- **Expected impact**: ~3.4x parameter increase, better representation capacity

### Phase 2: Training Strategy Overhaul ✅
**File: `configs/pibnn_base.yaml`**
- Increased epochs: 10→75 (7.5x more training)
- Increased batch size: 16→32 with gradient accumulation (effective batch size: 64)
- Enabled mixed precision training (AMP)
- Implemented cosine annealing with warm restarts (T_0=15, T_mult=2)
- Added learning rate warmup: 5 epochs
- Enabled SWA (Stochastic Weight Averaging) from epoch 50
- Increased EMA decay: 0.995→0.999
- Reduced gradient clipping: 1.0→0.5
- Increased early stopping patience: 10→20
- **Expected impact**: Better convergence, improved generalization

### Phase 3: Physics Integration Enhancement ✅
**File: `configs/pibnn_base.yaml`**
- Increased physics penalty weight: 0.1→0.3
- Increased formant VTL constraint weight: 0.1→0.2
- Increased F0 gender constraint weight: 0.05→0.15
- Added learnable VTL-height ratio (previously fixed at 6.7)
- Increased VTSL loss weight: 0.2→0.4

**File: `src/models/vocalmorph_v2/losses.py`**
- Added formant-height consistency loss method
- Integrated formant-height consistency into total loss (weight=0.15)
- **Expected impact**: Better physics-based regularization, improved accuracy

### Phase 4: Advanced Regularization ✅
**File: `src/models/vocalmorph_v2/losses.py`**
- Added focal loss for hard examples (gamma=2.0)
- Implemented label smoothing: 0.05→0.10
- Increased robust huber weight: 0.20→0.25
- Added focal weighting to height loss after epoch 20
- **Expected impact**: Better handling of difficult examples, reduced overfitting

### Phase 5: Data Augmentation Enhancement ✅
**File: `configs/pibnn_base.yaml`**
- Increased noise std: 0.015→0.025
- Increased time masking max frac: 0.08→0.12
- Increased feature masking max frac: 0.05→0.08
- Added frequency masking (p=0.30, max_frac=0.15)
- Added mixup augmentation (p=0.20, alpha=0.20)
- Increased scale std: 0.05→0.08
- Increased temporal jitter max frac: 0.03→0.05

**File: `src/preprocessing/dataset.py`**
- Added freq_mask_p and freq_mask_max_frac to FeatureAugmentConfig
- Added mixup_p and mixup_alpha to FeatureAugmentConfig
- Implemented frequency masking in FeatureAugmenter
- **Expected impact**: Better robustness, reduced overfitting

### Phase 6: Height-Specific Optimization ✅
**File: `configs/pibnn_base.yaml`**
- Increased height bin weights for short people: 1.8→2.5
- Increased height bin weights for tall people: 1.15→1.3
- Updated speaker alignment height bin loss weights:
  - Short: 1.0→2.0
  - Medium: 1.0 (unchanged)
  - Tall: 1.0→1.2
- **Expected impact**: Improved short-height performance (target: <5cm MAE)

### Phase 7: Advanced Aggregation Strategy ✅
**File: `src/models/vocalmorph_v2/utils.py`**
- Added attention_weighted aggregation method
- Implemented attention-based pooling using quality and uncertainty
- Added softmax-based attention weights computation
- Changed default aggregation: legacy_inverse_variance→attention_weighted
- **Expected impact**: Better speaker-level predictions, improved MAE

### Phase 8: Ensemble and Calibration ✅
**File: `scripts/train_ensemble.py`**
- Created ensemble training script
- Supports training multiple models with different seeds
- Implements model averaging for final predictions
- Default: 5 models with seeds [42, 43, 44, 45, 46]

**File: `configs/pibnn_base.yaml`**
- Increased evaluation n_samples: 4→8
- Increased evaluation n_crops: 3→5
- **Expected impact**: More robust predictions, better uncertainty calibration

## Training Instructions

### Single Model Training
```bash
cd C:\Users\USER\Downloads\VoxPhysica-main\VoxPhysica-main
python scripts/train.py --config configs/pibnn_base.yaml --seed 42 --epochs 75
```

### Ensemble Training
```bash
cd C:\Users\USER\Downloads\VoxPhysica-main\VoxPhysica-main
python scripts/train_ensemble.py --config configs/pibnn_base.yaml --n_models 5 --epochs 75
```

### Expected Training Time
- Single model: ~6-8 hours (depending on GPU)
- Ensemble (5 models): ~30-40 hours

## Expected Performance Improvements

### Conservative Estimates
- Overall test MAE: 5.94cm → 3.5-4.0cm
- Speaker-level MAE: 5.94cm → 3.0-3.5cm
- Short-height MAE: 9.7cm → 5.0-6.0cm

### Optimistic Estimates (with ensemble)
- Overall test MAE: 5.94cm → 2.5-3.0cm
- Speaker-level MAE: 5.94cm → 2.0-2.5cm
- Short-height MAE: 9.7cm → 4.0-5.0cm

## Key Technical Changes

### Model Capacity
- Parameters: 1.49M → ~4-5M (3.4x increase)
- Better representation learning capacity

### Training Duration
- Epochs: 10 → 75 (7.5x increase)
- More thorough optimization

### Regularization
- Multiple augmentation techniques
- Physics-based constraints
- Focal loss for hard examples
- SWA and EMA

### Aggregation
- Attention-based speaker pooling
- Better uncertainty handling

## Monitoring Metrics

### Primary Metrics
- height_mae_speaker (speaker-level MAE) - **Target: <3.0cm**
- height_mae (clip-level MAE) - **Target: <4.0cm**
- height_heightbin_short_mae - **Target: <5.0cm**

### Secondary Metrics
- Train-val speaker gap - **Target: <1.5cm**
- Gender-specific MAE
- Duration-specific MAE
- Quality-specific MAE

## Troubleshooting

### If Training Fails
1. Check GPU memory availability
2. Reduce batch size if OOM
3. Verify data paths in config
4. Check CUDA installation

### If Performance Doesn't Improve
1. Verify all changes are applied
2. Check training logs for convergence
3. Try different random seeds
4. Consider further hyperparameter tuning

### If Short-Height Performance Remains Poor
1. Increase short height bin weight further
2. Add more short-height samples to training
3. Implement height-specific attention mechanisms

## Next Steps

1. **Immediate**: Train single model with new configuration
2. **Validation**: Monitor height_mae_speaker on validation set
3. **Ensemble**: Train 5 models with different seeds
4. **Final Evaluation**: Test on holdout set with ensemble
5. **Iteration**: Further tune based on results

## Configuration Files Modified
- `configs/pibnn_base.yaml` - Main configuration
- `src/models/vocalmorph_v2/losses.py` - Loss functions
- `src/models/vocalmorph_v2/utils.py` - Aggregation methods
- `src/preprocessing/dataset.py` - Augmentation
- `scripts/train_ensemble.py` - New ensemble script

## Backup Recommendation
Before training, backup your original configuration:
```bash
cp configs/pibnn_base.yaml configs/pibnn_base.yaml.backup
```

## Success Criteria
- ✅ Test height_mae_speaker < 3.0cm
- ✅ Short-height MAE < 5.0cm
- ✅ Train-val gap < 1.5cm
- ✅ Stable performance across epochs
- ✅ Reproducible results across seeds