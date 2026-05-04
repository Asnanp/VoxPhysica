#!/usr/bin/env python
"""VocalMorph training entrypoint."""

import argparse
import ast
import json
import os
import random
import sys
import traceback

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.models.pibnn import build_model
from src.preprocessing.dataset import (
    FeatureAugmentConfig,
    VocalMorphDataset,
    build_worker_init_fn,
    build_dataloaders_from_dirs,
    collate_fn,
)
from src.preprocessing.feature_extractor import build_feature_config
from src.utils.audit_utils import validate_feature_contract


def _eval_numeric_expr(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval_numeric_expr(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and isinstance(
        node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv)
    ):
        left = _eval_numeric_expr(node.left)
        right = _eval_numeric_expr(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        return left // right
    raise ValueError("Only simple numeric expressions are supported.")


def _coerce_int(value, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got bool")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and float(value).is_integer():
        return int(value)
    if isinstance(value, str):
        expr = value.strip()
        if not expr:
            raise ValueError(f"{name} cannot be empty")
        numeric = _eval_numeric_expr(ast.parse(expr, mode="eval").body)
        if float(numeric).is_integer():
            return int(round(float(numeric)))
        raise ValueError(f"{name} must evaluate to an integer, got {value!r}")
    raise TypeError(
        f"{name} must be an integer or numeric expression, got {type(value).__name__}"
    )


def _coerce_bool(value, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    raise TypeError(f"{name} must be a boolean, got {type(value).__name__}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train VocalMorph")
    parser.add_argument("--config", type=str, default="configs/pibnn_base.yaml")
    parser.add_argument(
        "--resume", type=str, default=None, help="Checkpoint path to resume from"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="cuda / cpu (overrides config)"
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override config epochs"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Override config training seed"
    )
    parser.add_argument(
        "--metrics-out",
        type=str,
        default=None,
        help="Write final val/test metrics JSON to this path",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training and run strict evaluation from the resolved checkpoint only",
    )
    parser.add_argument(
        "--model-only-resume",
        action="store_true",
        help="Load checkpoint model/EMA weights but reset optimizer, scheduler, scaler, and RNG state.",
    )
    parser.add_argument(
        "--reset-resume-heads",
        action="store_true",
        help="When resuming model-only, keep encoder/fusion weights but reset task/domain heads from the new model init.",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def configure_cuda_math(train_cfg: dict) -> None:
    allow_tf32 = _coerce_bool(train_cfg.get("allow_tf32", True), "training.allow_tf32")
    train_cfg["allow_tf32"] = allow_tf32
    if not torch.cuda.is_available():
        return

    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = allow_tf32
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")


class TeeStream:
    def __init__(self, primary, secondary):
        self.primary = primary
        self.secondary = secondary
        self.primary_broken = False

    def write(self, data):
        if not self.primary_broken:
            try:
                self.primary.write(data)
            except OSError:
                self.primary_broken = True
        self.secondary.write(data)
        return len(data)

    def flush(self):
        if not self.primary_broken:
            try:
                self.primary.flush()
            except OSError:
                self.primary_broken = True
        self.secondary.flush()

    def isatty(self):
        return getattr(self.primary, "isatty", lambda: False)()

    @property
    def encoding(self):
        return getattr(self.primary, "encoding", "utf-8")


def _resolve_checkpoint_dir(config: dict) -> str:
    ckpt_dir = (
        config.get("logging", {})
        .get("checkpoint", {})
        .get("dir", "outputs/checkpoints")
    )
    return ckpt_dir if os.path.isabs(ckpt_dir) else os.path.join(ROOT, ckpt_dir)


def _resolve_run_dir(config: dict) -> str:
    return os.path.dirname(_resolve_checkpoint_dir(config))


def _configure_disk_logging(config: dict) -> None:
    run_dir = _resolve_run_dir(config)
    os.makedirs(run_dir, exist_ok=True)
    stdout_path = os.path.join(run_dir, "train.stdout.log")
    stderr_path = os.path.join(run_dir, "train.stderr.log")

    stdout_handle = open(stdout_path, "a", encoding="utf-8", buffering=1)
    stderr_handle = open(stderr_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = TeeStream(sys.stdout, stdout_handle)
    sys.stderr = TeeStream(sys.stderr, stderr_handle)
    print(f"[VocalMorph] stdout -> {stdout_path}")
    print(f"[VocalMorph] stderr -> {stderr_path}")


def _torch_load_checkpoint(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _is_valid_checkpoint(path: str) -> bool:
    if not path or not os.path.exists(path):
        return False
    try:
        payload = _torch_load_checkpoint(path)
    except Exception as exc:
        print(f"[VocalMorph] Invalid checkpoint skipped: {path} ({exc})")
        return False
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        print(f"[VocalMorph] Invalid checkpoint payload skipped: {path}")
        return False
    return True


def _resolve_resume_checkpoint(config: dict, explicit_path: str | None = None) -> str | None:
    ckpt_dir = _resolve_checkpoint_dir(config)
    candidates = []
    if explicit_path:
        explicit = explicit_path if os.path.isabs(explicit_path) else os.path.join(ROOT, explicit_path)
        candidates.append(explicit)
    candidates.extend(
        [
            os.path.join(ckpt_dir, "last.ckpt"),
            os.path.join(ckpt_dir, "last_good.ckpt"),
            os.path.join(ckpt_dir, "best.ckpt"),
            os.path.join(ckpt_dir, "best_model.pt"),
        ]
    )
    seen = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if _is_valid_checkpoint(path):
            return path
    return None


def _write_metrics_summary(
    *,
    trainer,
    model,
    train_eval_loader,
    val_loader,
    test_loader,
    train_cfg: dict,
    metrics_out: str | None,
) -> dict:
    final_train_eval_metrics = {}
    if _coerce_bool(train_cfg.get("final_train_eval", True), "training.final_train_eval"):
        final_train_eval_metrics = trainer._val_epoch_on(
            train_eval_loader, split_name="train_eval"
        )
    final_val_metrics = trainer._val_epoch()
    print("\n[VocalMorph] Running final test set evaluation...")
    test_metrics = trainer._val_epoch_on(test_loader, split_name="test")

    final_gaps = {}
    for key in (
        "height_mae",
        "height_mae_speaker",
        "height_rmse_speaker",
        "height_median_ae_speaker",
        "height_mae_speaker_omega",
        "height_rmse_speaker_omega",
        "height_median_ae_speaker_omega",
    ):
        train_value = final_train_eval_metrics.get(key)
        val_value = final_val_metrics.get(key)
        if isinstance(train_value, (int, float)) and isinstance(val_value, (int, float)):
            final_gaps[f"{key}_gap_val_minus_train"] = float(val_value) - float(train_value)

    metrics_summary = {
        "seed": int(train_cfg["seed"]),
        "monitor_name": trainer.es_monitor,
        "best_monitor_value": float(trainer.best_val_metric),
        "final_train_eval": {k: float(v) for k, v in final_train_eval_metrics.items()},
        "final_val": {k: float(v) for k, v in final_val_metrics.items()},
        "final_test": {k: float(v) for k, v in test_metrics.items()},
        "overfit_gaps": final_gaps,
    }

    print("\nTest Set Results:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    if metrics_out:
        resolved_metrics_out = (
            metrics_out if os.path.isabs(metrics_out) else os.path.join(ROOT, metrics_out)
        )
        os.makedirs(os.path.dirname(resolved_metrics_out), exist_ok=True)
        with open(resolved_metrics_out, "w") as f:
            json.dump(metrics_summary, f, indent=2)
        print(f"[VocalMorph] Wrote metrics summary to {resolved_metrics_out}")
    return metrics_summary


def _merge_checkpoint_state_with_fresh_heads(model, checkpoint: dict) -> dict:
    """Keep transferable backbone weights while leaving task heads freshly initialized."""

    reset_prefixes = (
        "height_head.",
        "height_adapter.",
        "height_bin_head.",
        "height_prior_head.",
        "physics_height_residual.",
        "weight_head.",
        "age_head.",
        "gender_head.",
        "domain_head.",
    )

    fresh_state = model.state_dict()

    def merged_state(source_state):
        merged = {key: value.detach().clone() for key, value in fresh_state.items()}
        copied = 0
        skipped = 0
        for key, value in source_state.items():
            if key not in merged:
                skipped += 1
                continue
            if key.startswith(reset_prefixes):
                skipped += 1
                continue
            if tuple(value.shape) != tuple(merged[key].shape):
                skipped += 1
                continue
            merged[key] = value.detach().clone()
            copied += 1
        return merged, copied, skipped

    updated = dict(checkpoint)
    model_state, copied, skipped = merged_state(checkpoint["model_state_dict"])
    updated["model_state_dict"] = model_state
    print(
        "[VocalMorph] Reset resume heads: "
        f"copied {copied} transferable tensors, kept fresh heads/skipped {skipped} tensors"
    )
    if checkpoint.get("ema_state_dict") is not None:
        ema_state, ema_copied, ema_skipped = merged_state(checkpoint["ema_state_dict"])
        updated["ema_state_dict"] = ema_state
        print(
            "[VocalMorph] Reset EMA heads: "
            f"copied {ema_copied} transferable tensors, kept fresh heads/skipped {ema_skipped} tensors"
        )
    return updated


def main():
    from src.training.trainer import VocalMorphTrainer

    args = parse_args()

    config_path = (
        args.config if os.path.isabs(args.config) else os.path.join(ROOT, args.config)
    )
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if args.device:
        config["training"]["device"] = args.device
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.seed is not None:
        config["training"]["seed"] = int(args.seed)
    _configure_disk_logging(config)

    trainer = None
    try:
        data_cfg = config["data"]
        train_cfg = config["training"]
        train_cfg["seed"] = int(train_cfg.get("seed", 42))
        train_cfg["epochs"] = _coerce_int(train_cfg.get("epochs", 100), "training.epochs")
        train_cfg["batch_size"] = _coerce_int(
            train_cfg.get("batch_size", 32), "training.batch_size"
        )
        train_cfg["num_workers"] = _coerce_int(
            train_cfg.get("num_workers", 4), "training.num_workers"
        )
        train_cfg["gradient_accumulation_steps"] = _coerce_int(
            train_cfg.get("gradient_accumulation_steps", 1),
            "training.gradient_accumulation_steps",
        )
        if train_cfg.get("max_feature_frames") is not None:
            train_cfg["max_feature_frames"] = _coerce_int(
                train_cfg["max_feature_frames"], "training.max_feature_frames"
            )
        if train_cfg.get("prefetch_factor") is not None:
            train_cfg["prefetch_factor"] = _coerce_int(
                train_cfg["prefetch_factor"], "training.prefetch_factor"
            )
        if train_cfg.get("persistent_workers") is not None:
            train_cfg["persistent_workers"] = _coerce_bool(
                train_cfg["persistent_workers"], "training.persistent_workers"
            )

        seed_everything(train_cfg["seed"])
        configure_cuda_math(train_cfg)
        print(f"[VocalMorph] Using seed: {train_cfg['seed']}")
        feat_dir = os.path.join(ROOT, data_cfg["features_dir"])
        feature_config = build_feature_config(config)
        split_manifest_cfg = data_cfg.get("split_manifests", {})
        expected_split_files = {
            "train": os.path.join(
                ROOT, split_manifest_cfg.get("train", "data/splits/train_clean.csv")
            ),
            "val": os.path.join(ROOT, split_manifest_cfg.get("val", "data/splits/val_clean.csv")),
            "test": os.path.join(
                ROOT, split_manifest_cfg.get("test", "data/splits/test_clean.csv")
            ),
        }
        
        # Skip feature contract validation for 15-epoch aggressive mode or
        # derived feature roots such as SSL-fused NPZs.
        skip_validation = train_cfg.get("epochs", 100) <= 20 or _coerce_bool(
            data_cfg.get("skip_feature_contract_validation", False),
            "data.skip_feature_contract_validation",
        )
        if skip_validation:
            print("[VocalMorph] Skipping feature contract validation for this run")
        else:
            validate_feature_contract(
                feature_root=feat_dir,
                expected_feature_config=feature_config.to_dict(),
                expected_split_files=expected_split_files,
                require_target_stats=True,
            )
            print("[VocalMorph] Audited feature contract verified.")

        stats_path = os.path.join(feat_dir, "target_stats.json")
        target_stats = None
        if os.path.exists(stats_path):
            with open(stats_path, "r") as f:
                target_stats = json.load(f)
            print(f"[VocalMorph] Loaded normalization stats from {stats_path}")
        else:
            print(f"[VocalMorph] WARNING: target_stats.json not found at {stats_path}")
            print("[VocalMorph] Continuing without normalization.")
        config["target_stats"] = target_stats

        print("[VocalMorph] Building dataloaders...")
        max_feature_frames = train_cfg.get("max_feature_frames")
        train_crop_mode = str(train_cfg.get("train_crop_mode", "head")).strip().lower()
        eval_crop_mode = str(train_cfg.get("eval_crop_mode", "center")).strip().lower()

        aug_cfg_raw = train_cfg.get("augmentation", {})
        train_augment = _coerce_bool(
            aug_cfg_raw.get("enabled", False), "training.augmentation.enabled"
        )
        speaker_batching_cfg = dict(train_cfg.get("speaker_batching", {}) or {})
        if "enabled" in speaker_batching_cfg:
            speaker_batching_cfg["enabled"] = _coerce_bool(
                speaker_batching_cfg["enabled"], "training.speaker_batching.enabled"
            )
        if "mode" in speaker_batching_cfg and speaker_batching_cfg["mode"] is not None:
            speaker_batching_cfg["mode"] = (
                str(speaker_batching_cfg["mode"]).strip().lower()
            )
        for key in (
            "speakers_per_batch",
            "clips_per_speaker",
            "paired_speakers_per_batch",
            "singleton_speakers_per_batch",
        ):
            if key in speaker_batching_cfg and speaker_batching_cfg[key] is not None:
                speaker_batching_cfg[key] = _coerce_int(
                    speaker_batching_cfg[key], f"training.speaker_batching.{key}"
                )
        augment_config = None
        if train_augment:
            augment_config = FeatureAugmentConfig(
                noise_p=float(aug_cfg_raw.get("noise_p", 0.50)),
                noise_std=float(aug_cfg_raw.get("noise_std", 0.02)),
                time_mask_p=float(aug_cfg_raw.get("time_mask_p", 0.40)),
                time_mask_max_frac=float(aug_cfg_raw.get("time_mask_max_frac", 0.10)),
                feat_mask_p=float(aug_cfg_raw.get("feat_mask_p", 0.30)),
                feat_mask_max_frac=float(aug_cfg_raw.get("feat_mask_max_frac", 0.08)),
                scale_p=float(aug_cfg_raw.get("scale_p", 0.35)),
                scale_std=float(aug_cfg_raw.get("scale_std", 0.08)),
                temporal_jitter_p=float(aug_cfg_raw.get("temporal_jitter_p", 0.25)),
                temporal_jitter_max_frac=float(
                    aug_cfg_raw.get("temporal_jitter_max_frac", 0.05)
                ),
                freq_mask_p=float(aug_cfg_raw.get("freq_mask_p", 0.30)),
                freq_mask_max_frac=float(aug_cfg_raw.get("freq_mask_max_frac", 0.15)),
                mixup_p=float(aug_cfg_raw.get("mixup_p", 0.20)),
                mixup_alpha=float(aug_cfg_raw.get("mixup_alpha", 0.20)),
            )
            print("[VocalMorph] On-the-fly augmentation: ENABLED")
            if train_cfg.get("epochs", 100) <= 20:
                print("[VocalMorph] ULTRA-AGGRESSIVE 15-EPOCH AUGMENTATION MODE")

        print(
            f"[VocalMorph] Sequence cropping -> max_frames={max_feature_frames} "
            f"| train={train_crop_mode} | eval={eval_crop_mode}"
        )
        
        train_loader, val_loader, test_loader = build_dataloaders_from_dirs(
            train_dir=os.path.join(feat_dir, "train"),
            val_dir=os.path.join(feat_dir, "val"),
            test_dir=os.path.join(feat_dir, "test"),
            batch_size=train_cfg["batch_size"],
            num_workers=train_cfg["num_workers"],
            target_stats=target_stats,
            max_len=max_feature_frames,
            train_crop_mode=train_crop_mode,
            eval_crop_mode=eval_crop_mode,
            persistent_workers=train_cfg.get("persistent_workers"),
            prefetch_factor=train_cfg.get("prefetch_factor"),
            train_augment=train_augment,
            augment_config=augment_config,
            speaker_batching=speaker_batching_cfg,
            sample_weighting=train_cfg.get("sample_weighting", {}),
            base_seed=int(train_cfg["seed"]),
            pin_memory=train_cfg.get("pin_memory", True),
        )
        train_eval_dataset = VocalMorphDataset(
            os.path.join(feat_dir, "train"),
            max_len=max_feature_frames,
            target_stats=target_stats,
            crop_mode=eval_crop_mode,
            augment=False,
        )
        pin_memory = train_cfg.get("pin_memory", True) and torch.cuda.is_available()
        train_eval_loader = DataLoader(
            train_eval_dataset,
            batch_size=train_cfg["batch_size"],
            shuffle=False,
            num_workers=train_cfg["num_workers"],
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            persistent_workers=bool(train_cfg["num_workers"] > 0)
            if train_cfg.get("persistent_workers") is None
            else bool(train_cfg["persistent_workers"]),
            worker_init_fn=build_worker_init_fn(int(train_cfg["seed"]) + 997),
            **(
                {"prefetch_factor": int(train_cfg["prefetch_factor"])}
                if train_cfg["num_workers"] > 0 and train_cfg.get("prefetch_factor") is not None
                else {}
            ),
        )

        inferred_input_dim = train_loader.dataset.infer_input_dim()
        config.setdefault("model", {})["input_dim"] = int(inferred_input_dim)
        print(f"[VocalMorph] Inferred input_dim from data: {inferred_input_dim}")

        print("[VocalMorph] Building model...")
        model = build_model(config)
        if hasattr(model, "set_target_stats"):
            model.set_target_stats(target_stats)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[VocalMorph] Trainable parameters: {n_params:,}")

        trainer = VocalMorphTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            target_stats=target_stats,
            train_eval_loader=train_eval_loader,
        )

        resume_path = _resolve_resume_checkpoint(config, args.resume)
        if resume_path:
            checkpoint = _torch_load_checkpoint(resume_path)
            if args.reset_resume_heads:
                checkpoint = _merge_checkpoint_state_with_fresh_heads(model, checkpoint)
            trainer.restore_from_checkpoint(
                checkpoint,
                checkpoint_path=resume_path,
                model_only=bool(args.eval_only or args.model_only_resume),
            )
        else:
            print("[VocalMorph] No valid resume checkpoint found. Starting fresh.")

        if args.eval_only:
            if not resume_path:
                raise RuntimeError("Eval-only mode requires a valid checkpoint via --resume or checkpoint dir.")
            print("[VocalMorph] Eval-only mode enabled. Skipping training loop.")
        else:
            trainer.train()

        if args.eval_only:
            best_ckpt_path = resume_path
        else:
            best_ckpt_path = trainer.ckpt_manager.best_path
            if not os.path.exists(best_ckpt_path):
                legacy_best = os.path.join(trainer.ckpt_manager.save_dir, "best_model.pt")
                if os.path.exists(legacy_best):
                    best_ckpt_path = legacy_best
        if os.path.exists(best_ckpt_path):
            best_ckpt = _torch_load_checkpoint(best_ckpt_path)
            trainer._load_model_checkpoint_state(best_ckpt["model_state_dict"])
            if best_ckpt.get("ema_state_dict") is not None:
                trainer.load_ema_state_dict(best_ckpt["ema_state_dict"])
            print(
                f"[VocalMorph] Loaded best checkpoint for final evaluation: {best_ckpt_path}"
            )

        _write_metrics_summary(
            trainer=trainer,
            model=model,
            train_eval_loader=train_eval_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            train_cfg=train_cfg,
            metrics_out=args.metrics_out,
        )
        return 0
    except KeyboardInterrupt:
        print("[VocalMorph] KeyboardInterrupt received.", file=sys.stderr)
        if trainer is not None:
            trainer.save_emergency_checkpoint("interrupt", "KeyboardInterrupt")
        return 130
    except Exception:
        exception_text = traceback.format_exc()
        if trainer is not None:
            trainer.save_emergency_checkpoint("crash", exception_text)
        print(exception_text, file=sys.stderr)
        return 1
    finally:
        if trainer is not None:
            trainer.close()
        sys.stdout.flush()
        sys.stderr.flush()


if __name__ == "__main__":
    raise SystemExit(main())
