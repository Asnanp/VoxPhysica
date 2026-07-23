import glob
import numpy as np
import pandas as pd
import torch
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.predict import build_predictor

def load_predictions(checkpoint_paths, config_path):
    """Load predictions from multiple checkpoints."""
    all_preds = []
    
    from src.data.dataset import VocalMorphDataset
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    test_df = pd.read_csv(config['data']['split_manifests']['test'])
    test_data = []
    
    for _, row in test_df.iterrows():
        # Build features path from raw audio path or use directly
        feat_path = os.path.join(config['data']['features_dir'], f"{row['speaker_id']}.pt")
        if os.path.exists(feat_path):
             test_data.append((row['speaker_id'], feat_path, row['height_cm']))

    ground_truth = np.array([item[2] for item in test_data])
    speakers = [item[0] for item in test_data]
    
    for ckpt_path in checkpoint_paths:
        print(f"Generating predictions for {ckpt_path}...")
        predictor = build_predictor(config_path, ckpt_path)
        
        preds = []
        for speaker_id, feat_path, _ in test_data:
            # We would need to load features and predict. 
            # To be accurate and avoid rewriting the entire prediction pipeline, 
            # we should use the evaluation mode of train.py or run_strict_evaluation.py
            pass
            
    return all_preds, ground_truth, speakers

def ensemble_predictions(all_preds, method='mean'):
    """Ensemble predictions from multiple models."""
    if method == 'mean':
        return np.mean(all_preds, axis=0)
    elif method == 'median':
        return np.median(all_preds, axis=0)

def main():
    pass

if __name__ == '__main__':
    main()