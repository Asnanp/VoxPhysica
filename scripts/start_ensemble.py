import os
import subprocess
import yaml
import shutil

def main():
    base_config_path = "configs/pibnn_rtx3060_3cm_NUCLEAR.yaml"
    seeds = [17, 23, 29, 31, 37, 41]
    
    with open(base_config_path, "r") as f:
        base_config = yaml.safe_load(f)
        
    for seed in seeds:
        config = base_config.copy()
        
        # Override settings for ensemble seed
        config["training"]["seed"] = seed
        
        # Set distinct logging and checkpoint dirs
        config["logging"]["tensorboard"]["log_dir"] = f"outputs/logs_rtx3060_nuclear/seed_{seed}/"
        config["logging"]["checkpoint"]["dir"] = f"outputs/checkpoints_rtx3060_nuclear/seed_{seed}/"
        config["inference"]["output_dir"] = f"outputs/predictions_rtx3060_nuclear/seed_{seed}/"
        
        tmp_config_path = f"configs/tmp_nuclear_seed_{seed}.yaml"
        with open(tmp_config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
        print(f"========================================")
        print(f"Training ensemble model with seed {seed}")
        print(f"========================================")
        
        # Start training
        cmd = ["python", "scripts/train.py", "--config", tmp_config_path]
        
        # In a real environment, this blocks.
        # But we will run this python script in the background.
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"Error training seed {seed}")
            
    print("All training complete. Ready for ensemble evaluation.")

if __name__ == "__main__":
    main()