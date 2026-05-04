"""
Ensemble training script for VoxPhysica height prediction.

Trains multiple models with different random seeds and combines predictions
for improved accuracy and robustness.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any

def train_single_model(
    config_path: str,
    seed: int,
    output_dir: str,
    epochs: int = 75,
) -> str:
    """Train a single model with the given seed."""
    seed_output_dir = os.path.join(output_dir, f"seed_{seed}")
    os.makedirs(seed_output_dir, exist_ok=True)
    
    cmd = [
        sys.executable,
        "scripts/train.py",
        "--config", config_path,
        "--seed", str(seed),
        "--epochs", str(epochs),
        "--output_dir", seed_output_dir,
    ]
    
    print(f"Training model with seed {seed}...")
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    
    if result.returncode != 0:
        print(f"Training failed for seed {seed}")
        return None
    
    checkpoint_path = os.path.join(seed_output_dir, "best.ckpt")
    if os.path.exists(checkpoint_path):
        return checkpoint_path
    else:
        print(f"Checkpoint not found for seed {seed}")
        return None

def ensemble_predictions(
    checkpoint_paths: List[str],
    test_data_dir: str,
    output_path: str,
    n_samples: int = 8,
    n_crops: int = 5,
) -> Dict[str, Any]:
    """Generate ensemble predictions from multiple checkpoints."""
    # This would load each checkpoint and average their predictions
    # For now, return a placeholder
    return {
        "checkpoints": checkpoint_paths,
        "n_samples": n_samples,
        "n_crops": n_crops,
        "output_path": output_path,
    }

def main():
    parser = argparse.ArgumentParser(description="Train ensemble of VoxPhysica models")
    parser.add_argument("--config", type=str, default="configs/pibnn_base.yaml",
                        help="Path to configuration file")
    parser.add_argument("--n_models", type=int, default=5,
                        help="Number of models to train in ensemble")
    parser.add_argument("--seeds", type=int, nargs='+', default=None,
                        help="Random seeds for each model (default: 0,1,2,3,4)")
    parser.add_argument("--epochs", type=int, default=75,
                        help="Number of training epochs per model")
    parser.add_argument("--output_dir", type=str, default="outputs/ensemble",
                        help="Output directory for ensemble models")
    parser.add_argument("--base_seed", type=int, default=42,
                        help="Base random seed")
    
    args = parser.parse_args()
    
    if args.seeds is None:
        args.seeds = [args.base_seed + i for i in range(args.n_models)]
    else:
        args.n_models = len(args.seeds)
    
    print(f"Training ensemble of {args.n_models} models with seeds: {args.seeds}")
    
    # Train each model
    checkpoint_paths = []
    for i, seed in enumerate(args.seeds):
        print(f"\n=== Training model {i+1}/{args.n_models} with seed {seed} ===")
        checkpoint = train_single_model(
            args.config,
            seed,
            args.output_dir,
            args.epochs,
        )
        if checkpoint:
            checkpoint_paths.append(checkpoint)
        else:
            print(f"Failed to train model with seed {seed}")
    
    print(f"\n=== Ensemble Training Complete ===")
    print(f"Successfully trained {len(checkpoint_paths)}/{args.n_models} models")
    print(f"Checkpoints: {checkpoint_paths}")
    
    # Save ensemble configuration
    ensemble_config = {
        "n_models": len(checkpoint_paths),
        "seeds": args.seeds,
        "checkpoint_paths": checkpoint_paths,
        "config": args.config,
        "epochs": args.epochs,
    }
    
    config_path = os.path.join(args.output_dir, "ensemble_config.json")
    with open(config_path, 'w') as f:
        json.dump(ensemble_config, f, indent=2)
    
    print(f"Ensemble configuration saved to {config_path}")

if __name__ == "__main__":
    main()