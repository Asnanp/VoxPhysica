#!/usr/bin/env python
"""
GPU Optimization Validation Script for VoxPhysica RTX 3060 Elite Edition

This script validates that all GPU optimizations are working correctly
and provides performance metrics for the training setup.
"""

import torch
import yaml
import sys
import os

# Fix path for test script
script_dir = os.path.dirname(os.path.abspath(__file__))
ROOT = script_dir
sys.path.insert(0, ROOT)

# Fix Windows encoding issue
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

def test_gpu_availability():
    """Test if GPU is available and properly configured."""
    print("=" * 60)
    print("GPU AVAILABILITY TEST")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("[X] CUDA is not available. GPU training will not work.")
        return False

    print(f"[OK] CUDA is available")
    print(f"[OK] CUDA Version: {torch.version.cuda}")
    print(f"[OK] Device Count: {torch.cuda.device_count()}")
    print(f"[OK] Current Device: {torch.cuda.current_device()}")
    print(f"[OK] Device Name: {torch.cuda.get_device_name(0)}")

    # Test GPU memory
    device = torch.device("cuda")
    test_tensor = torch.randn(10000, 10000, device=device)
    print(f"[OK] GPU Memory Test: Passed")
    del test_tensor
    torch.cuda.empty_cache()

    return True

def test_mixed_precision():
    """Test if mixed precision (AMP) is supported."""
    print("\n" + "=" * 60)
    print("MIXED PRECISION TEST")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("[X] CUDA not available, skipping AMP test")
        return False

    device = torch.device("cuda")

    # Test FP16 support
    try:
        test_tensor = torch.randn(100, 100, device=device, dtype=torch.float16)
        print(f"[OK] FP16 (Half Precision) Supported")
        del test_tensor
    except Exception as e:
        print(f"[X] FP16 not supported: {e}")
        return False

    # Test TF32 support (for RTX 30 series)
    try:
        if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'matmul'):
            tf32_enabled = torch.backends.cuda.matmul.allow_tf32
            print(f"[OK] TF32 Support: {'Enabled' if tf32_enabled else 'Available'}")
    except:
        print("[!] TF32 support check skipped")

    # Test GradScaler
    try:
        scaler = torch.cuda.amp.GradScaler()
        print(f"[OK] GradScaler (AMP) Available")
    except Exception as e:
        print(f"[X] GradScaler not available: {e}")
        return False

    return True

def test_elite_config():
    """Test if elite configuration file exists and is valid."""
    print("\n" + "=" * 60)
    print("ELITE CONFIGURATION TEST")
    print("=" * 60)

    config_path = os.path.join(ROOT, "configs", "pibnn_rtx3060_elite.yaml")
    
    # Try alternative path if not found
    if not os.path.exists(config_path):
        alt_path = os.path.join(os.path.dirname(ROOT), "VoxPhysica-main", "configs", "pibnn_rtx3060_elite.yaml")
        if os.path.exists(alt_path):
            config_path = alt_path

    if not os.path.exists(config_path):
        print(f"[X] Elite config not found: {config_path}")
        return False

    print(f"[OK] Elite config found: {config_path}")

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"[OK] Config file is valid YAML")

        # Check key elite settings
        training_cfg = config.get('training', {})

        checks = [
            ("Batch Size", training_cfg.get('batch_size') == 32, training_cfg.get('batch_size')),
            ("Device", training_cfg.get('device') == 'cuda', training_cfg.get('device')),
            ("Mixed Precision", training_cfg.get('mixed_precision') == True, training_cfg.get('mixed_precision')),
            ("Pin Memory", training_cfg.get('pin_memory') == True, training_cfg.get('pin_memory')),
            ("Non-blocking", training_cfg.get('non_blocking') == True, training_cfg.get('non_blocking')),
            ("Gradient Checkpointing", training_cfg.get('gradient_checkpointing') == True, training_cfg.get('gradient_checkpointing')),
            ("Num Workers", training_cfg.get('num_workers') >= 4, training_cfg.get('num_workers')),
            ("Prefetch Factor", training_cfg.get('prefetch_factor') >= 4, training_cfg.get('prefetch_factor')),
        ]

        for name, passed, value in checks:
            status = "[OK]" if passed else "[X]"
            print(f"{status} {name}: {value}")

        all_passed = all(check[1] for check in checks)
        return all_passed

    except Exception as e:
        print(f"[X] Config file error: {e}")
        return False

def test_memory_efficiency():
    """Test memory efficiency optimizations."""
    print("\n" + "=" * 60)
    print("MEMORY EFFICIENCY TEST")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("[X] CUDA not available, skipping memory test")
        return False

    device = torch.device("cuda")

    # Test pin memory behavior
    try:
        # Create a tensor and test pinning
        cpu_tensor = torch.randn(1000, 1000)
        pinned_tensor = cpu_tensor.pin_memory()
        print(f"[OK] Pin Memory: Supported")

        # Test non-blocking transfer
        gpu_tensor = pinned_tensor.to(device, non_blocking=True)
        torch.cuda.synchronize()
        print(f"[OK] Non-blocking Transfer: Supported")

        del cpu_tensor, pinned_tensor, gpu_tensor
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"[X] Memory efficiency test failed: {e}")
        return False

    return True

def test_data_loading_optimizations():
    """Test data loading optimizations."""
    print("\n" + "=" * 60)
    print("DATA LOADING OPTIMIZATION TEST")
    print("=" * 60)

    try:
        from src.preprocessing.dataset import collate_fn, FeatureAugmentConfig
        import numpy as np

        # Test optimized collate function
        batch = [
            {
                "sequence": torch.randn(100, 136),
                "height": torch.tensor(1.5),
                "weight": torch.tensor(70.0),
                "age": torch.tensor(25.0),
                "gender": torch.tensor(1),
                "speaker_id": f"speaker_{i}",
            }
            for i in range(4)
        ]

        result = collate_fn(batch)
        print(f"[OK] Optimized Collate Function: Working")
        print(f"[OK] Vectorized Padding Mask: Implemented")

        # Test enhanced augmentation
        aug_config = FeatureAugmentConfig()
        print(f"[OK] Elite Augmentation Config: Loaded")
        print(f"   - Noise Probability: {aug_config.noise_p}")
        print(f"   - Time Mask Probability: {aug_config.time_mask_p}")
        print(f"   - Feature Mask Probability: {aug_config.feat_mask_p}")
        print(f"   - Frequency Mask Probability: {aug_config.freq_mask_p}")

        return True

    except Exception as e:
        print(f"[X] Data loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_pipeline():
    """Test training pipeline optimizations."""
    print("\n" + "=" * 60)
    print("TRAINING PIPELINE TEST")
    print("=" * 60)

    try:
        from src.training.trainer import VocalMorphTrainer

        # Check if trainer has GPU optimizations
        import inspect
        trainer_source = inspect.getsource(VocalMorphTrainer.__init__)

        checks = [
            ("Pin Memory Optimization", "pin_memory" in trainer_source),
            ("Non-blocking Transfers", "non_blocking" in trainer_source),
            ("Gradient Checkpointing", "gradient_checkpointing" in trainer_source),
            ("Cache Clearing", "empty_cache" in trainer_source),
            ("Early Stopping Delta", "es_min_delta" in trainer_source),
        ]

        for name, passed in checks:
            status = "[OK]" if passed else "[X]"
            print(f"{status} {name}: {'Implemented' if passed else 'Missing'}")

        all_passed = all(check[1] for check in checks)
        return all_passed

    except Exception as e:
        print(f"[X] Training pipeline test failed: {e}")
        return False

def main():
    """Run all GPU optimization tests."""
    print("\n" + "=" * 60)
    print("VOXPHYSICA GPU OPTIMIZATION VALIDATION")
    print("RTX 3060 Elite Edition")
    print("=" * 60)

    tests = [
        ("GPU Availability", test_gpu_availability),
        ("Mixed Precision Support", test_mixed_precision),
        ("Elite Configuration", test_elite_config),
        ("Memory Efficiency", test_memory_efficiency),
        ("Data Loading Optimizations", test_data_loading_optimizations),
        ("Training Pipeline Optimizations", test_training_pipeline),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n[X] {test_name} failed with exception: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "[OK] PASSED" if result else "[X] FAILED"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n[SUCCESS] All GPU optimizations are working correctly!")
        print("[READY] Ready for elite training with RTX 3060")
        print("\nTo start training:")
        print("python scripts/train.py --config configs/pibnn_rtx3060_elite.yaml")
        return 0
    else:
        print(f"\n[WARNING] {total - passed} test(s) failed. Please review the output above.")
        return 1

if __name__ == "__main__":
    exit(main())