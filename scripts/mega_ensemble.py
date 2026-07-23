import os
import sys
import glob
import numpy as np
import torch
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.diagnose_height_bias import (
    build_eval_stack,
    collect_speaker_records,
    configure_cuda_math,
    seed_everything,
    _prepare_config
)

def _records_by_speaker(records):
    return {str(row["speaker_id"]): row for row in records}

def _speaker_truth(speakers, records_by_speaker):
    return np.asarray([float(records_by_speaker[s]["height_true"]) for s in speakers], dtype=np.float32)
from scripts.train import _torch_load_checkpoint
import argparse

def get_config_for_checkpoint(ckpt_dir):
    # Try to map the checkpoint dir to a config file
    name = os.path.basename(ckpt_dir).replace("checkpoints_", "")
    # handle nuclear specially
    if name == "rtx3060_nuclear":
        return "configs/pibnn_rtx3060_3cm_NUCLEAR.yaml"
    elif name == "rtx3060_3cm_power":
        return "configs/pibnn_rtx3060_3cm_power.yaml"
    elif name == "rtx3060_3cm_current_fullpower12gb":
        return "configs/pibnn_rtx3060_3cm_current_fullpower12gb.yaml"
    elif name == "rtx3060_3cm_extpretrain_headreset_finetune12gb":
        return "configs/pibnn_rtx3060_3cm_extpretrain_headreset_finetune12gb.yaml"
    elif name == "rtx3060_15ep_elite":
        return "configs/pibnn_rtx3060_15ep_elite.yaml"
    elif name == "rtx3060_3cm_finetune_lrfix12gb":
        return "configs/pibnn_rtx3060_3cm_finetune_lrfix12gb.yaml"
    elif name == "rtx3060_3cm_ssl":
        return "configs/pibnn_rtx3060_3cm_ssl.yaml"
    elif name == "rtx3060_3cm_optimized":
        return "configs/pibnn_rtx3060_3cm_OPTIMIZED.yaml"
    return None

def main():
    print("Starting Mega Ensemble Evaluation...")
    
    ckpt_dirs = glob.glob("outputs/checkpoints_*")
    
    all_val_preds = {}
    all_test_preds = {}
    val_truth = None
    test_truth = None
    val_speakers = None
    test_speakers = None
    
    for ckpt_dir in ckpt_dirs:
        config_path = get_config_for_checkpoint(ckpt_dir)
        if not config_path or not os.path.exists(config_path):
            continue
            
        best_ckpt = os.path.join(ckpt_dir, "best.ckpt")
        if not os.path.exists(best_ckpt):
            # Try to find epoch_*_metric_*.ckpt if best doesn't exist
            epoch_ckpts = glob.glob(os.path.join(ckpt_dir, "epoch_*_metric_*.ckpt"))
            if not epoch_ckpts:
                continue
            best_ckpt = sorted(epoch_ckpts)[0] # just grab one
            
        print(f"Loading {best_ckpt} with config {config_path}")
        
        try:
            prep_args = argparse.Namespace(
                config=config_path,
                checkpoint=None,
                device="cuda",
                batch_size=64,
                num_workers=0,
                output_dir="outputs/mega_ensemble",
                include_train=False,
            )
            config = _prepare_config(prep_args)
            seed_everything(42)
            configure_cuda_math(config["training"])
            trainer, _, val_loader, test_loader = build_eval_stack(config)
            
            payload = _torch_load_checkpoint(best_ckpt)
            # Remove keys that might cause size mismatch if head reset was used
            # We must load exact architecture
            trainer._load_model_checkpoint_state(payload["model_state_dict"])
            if payload.get("ema_state_dict") is not None:
                trainer.load_ema_state_dict(payload["ema_state_dict"])
                
            val_records, val_summary = collect_speaker_records(trainer, val_loader, "val")
            test_records, test_summary = collect_speaker_records(trainer, test_loader, "test")
            
            val_dict = _records_by_speaker(val_records)
            test_dict = _records_by_speaker(test_records)
            
            if val_speakers is None:
                val_speakers = sorted(list(val_dict.keys()))
                test_speakers = sorted(list(test_dict.keys()))
                val_truth = _speaker_truth(val_speakers, val_dict)
                test_truth = _speaker_truth(test_speakers, test_dict)
                
            # Filter common speakers just in case
            val_preds = [float(val_dict[s]["pred_omega"]) for s in val_speakers if s in val_dict]
            test_preds = [float(test_dict[s]["pred_omega"]) for s in test_speakers if s in test_dict]
            
            if len(val_preds) == len(val_speakers):
                all_val_preds[ckpt_dir] = val_preds
                all_test_preds[ckpt_dir] = test_preds
                print(f"Success! Test MAE for {ckpt_dir}: {test_summary.get('omega_mae', 'N/A')}")
        except Exception as e:
            print(f"Failed to evaluate {ckpt_dir}: {e}")
            
    if not all_test_preds:
        print("No models evaluated.")
        return
        
    print("\nEnsemble Results:")
    test_matrix = np.array(list(all_test_preds.values())).T # Shape: (speakers, models)
    
    mean_preds = np.mean(test_matrix, axis=1)
    median_preds = np.median(test_matrix, axis=1)
    
    mean_mae = np.mean(np.abs(mean_preds - test_truth))
    median_mae = np.mean(np.abs(median_preds - test_truth))
    
    print(f"Number of models in ensemble: {test_matrix.shape[1]}")
    print(f"Mean Ensemble MAE: {mean_mae:.4f} cm")
    print(f"Median Ensemble MAE: {median_mae:.4f} cm")
    
    # Save predictions
    os.makedirs("outputs/mega_ensemble", exist_ok=True)
    with open("outputs/mega_ensemble/ensemble_metrics.txt", "w") as f:
        f.write(f"Models: {list(all_test_preds.keys())}\n")
        f.write(f"Mean MAE: {mean_mae:.4f}\n")
        f.write(f"Median MAE: {median_mae:.4f}\n")
        
    # Is it below 3.0?
    if mean_mae <= 3.0 or median_mae <= 3.0:
        print("🎉🎉🎉 ACHIEVED 3.0 CM MAE!!! 🎉🎉🎉")
        
if __name__ == "__main__":
    main()