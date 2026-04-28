#!/usr/bin/env python
"""
VocalMorph V3 — Unified Height Training Script
================================================

Clean end-to-end training for the V3 nuclear model.
Height-focused with Huber + ordinal ranking loss.
No short/tall segmentation — everyone in one training phase.

Usage:
    python scripts/train_v3.py
    python scripts/train_v3.py --config configs/v3_nuclear.yaml --epochs 120 --seed 42
"""

import argparse
import json
import math
import os
import random
import sys
import time
import traceback
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.models.vocalmorph_v3 import VocalMorphV3, V3HeightLoss, build_v3_model, build_v3_loss
from src.preprocessing.dataset import (
    FeatureAugmentConfig,
    VocalMorphDataset,
    build_worker_init_fn,
    build_dataloaders_from_dirs,
    collate_fn,
)
from src.preprocessing.feature_extractor import build_feature_config
from src.utils.audit_utils import validate_feature_contract


# ────────────────────────────────────────────────────────────
# Utilities
# ────────────────────────────────────────────────────────────


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def configure_cuda(allow_tf32: bool = True) -> None:
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


# ────────────────────────────────────────────────────────────
# EMA
# ────────────────────────────────────────────────────────────


class EMAWeights:
    """Exponential Moving Average of model weights."""

    def __init__(self, model: nn.Module, decay: float = 0.999, warmup_steps: int = 200):
        self.model = model
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.step_count = 0
        self.shadow: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

    def update(self) -> None:
        self.step_count += 1
        decay = self.decay if self.step_count > self.warmup_steps else 0.0
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.shadow:
                    self.shadow[name].mul_(decay).add_(param.detach(), alpha=1.0 - decay)

    def swap_in(self) -> Dict[str, torch.Tensor]:
        """Swap in EMA weights and return backup of current weights."""
        backup = {}
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.shadow:
                    backup[name] = param.detach().clone()
                    param.copy_(self.shadow[name])
        return backup

    def restore(self, backup: Dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in backup:
                    param.copy_(backup[name])

    def state_dict(self) -> Dict[str, Any]:
        return {
            "step_count": self.step_count,
            "shadow": {k: v.cpu().clone() for k, v in self.shadow.items()},
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.step_count = int(state.get("step_count", 0))
        for k, v in state.get("shadow", {}).items():
            if k in self.shadow:
                self.shadow[k] = v.to(self.shadow[k].device)


# ────────────────────────────────────────────────────────────
# Metrics
# ────────────────────────────────────────────────────────────


def compute_height_metrics(
    all_preds: List[torch.Tensor],
    all_targets: List[torch.Tensor],
    all_speaker_ids: List[List[str]],
    target_stats: Optional[Dict],
) -> Dict[str, float]:
    """Compute clip-level and speaker-level height MAE in cm."""
    preds = torch.cat(all_preds, dim=0).cpu()
    targets = torch.cat(all_targets, dim=0).cpu()
    speaker_ids = []
    for batch_ids in all_speaker_ids:
        speaker_ids.extend(batch_ids)

    # Denormalize to cm
    if target_stats and "height" in target_stats:
        mean = float(target_stats["height"]["mean"])
        std = float(target_stats["height"]["std"])
        preds_cm = preds * std + mean
        targets_cm = targets * std + mean
    else:
        preds_cm = preds
        targets_cm = targets

    # Clip-level MAE
    clip_mae = (preds_cm - targets_cm).abs().mean().item()

    # Speaker-level MAE (average predictions per speaker, then compute MAE)
    speaker_preds = defaultdict(list)
    speaker_targets = defaultdict(list)
    for i, sid in enumerate(speaker_ids):
        speaker_preds[sid].append(preds_cm[i].item())
        speaker_targets[sid].append(targets_cm[i].item())

    speaker_maes = []
    speaker_short_maes = []
    speaker_tall_maes = []
    for sid in speaker_preds:
        pred_mean = np.mean(speaker_preds[sid])
        target_mean = np.mean(speaker_targets[sid])
        mae = abs(pred_mean - target_mean)
        speaker_maes.append(mae)
        if target_mean < 165:
            speaker_short_maes.append(mae)
        elif target_mean >= 175:
            speaker_tall_maes.append(mae)

    speaker_mae = float(np.mean(speaker_maes)) if speaker_maes else 0.0
    speaker_rmse = float(np.sqrt(np.mean(np.array(speaker_maes) ** 2))) if speaker_maes else 0.0
    speaker_median = float(np.median(speaker_maes)) if speaker_maes else 0.0

    short_mae = float(np.mean(speaker_short_maes)) if speaker_short_maes else 0.0
    tall_mae = float(np.mean(speaker_tall_maes)) if speaker_tall_maes else 0.0

    # Gender accuracy if available
    metrics = {
        "height_mae_clip": clip_mae,
        "height_mae_speaker": speaker_mae,
        "height_rmse_speaker": speaker_rmse,
        "height_median_ae_speaker": speaker_median,
        "height_mae_short_speaker": short_mae,
        "height_mae_tall_speaker": tall_mae,
        "n_speakers": len(speaker_preds),
        "n_short_speakers": len(speaker_short_maes),
        "n_tall_speakers": len(speaker_tall_maes),
    }
    return metrics


# ────────────────────────────────────────────────────────────
# Trainer
# ────────────────────────────────────────────────────────────


class V3Trainer:
    """Clean trainer for VocalMorphV3."""

    def __init__(
        self,
        model: VocalMorphV3,
        criterion: V3HeightLoss,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        config: dict,
        target_stats: Optional[Dict],
    ):
        self.model = model
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.target_stats = target_stats

        train_cfg = config.get("training", {})
        self.epochs = int(train_cfg.get("epochs", 120))
        self.device = self._resolve_device(train_cfg.get("device", "auto"))
        self.model = self.model.to(self.device)

        # AMP
        amp_enabled = train_cfg.get("amp", True)
        self.use_amp = bool(amp_enabled) and self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # Optimizer
        opt_cfg = train_cfg.get("optimizer", {})
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(opt_cfg.get("lr", 3e-4)),
            weight_decay=float(opt_cfg.get("weight_decay", 0.05)),
            betas=tuple(opt_cfg.get("betas", [0.9, 0.98])),
        )

        # Scheduler
        sched_cfg = train_cfg.get("scheduler", {})
        sched_type = sched_cfg.get("type", "cosine_annealing")
        eta_min = float(sched_cfg.get("eta_min", 1e-6))
        
        if sched_type == "cosine_annealing_warm_restarts":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, 
                T_0=int(sched_cfg.get("T_0", 25)), 
                T_mult=int(sched_cfg.get("T_mult", 2)), 
                eta_min=eta_min
            )
        else:
            t_max = int(sched_cfg.get("T_max", self.epochs))
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=t_max, eta_min=eta_min
            )

        # Gradient clipping
        gc_cfg = train_cfg.get("gradient_clipping", {})
        self.grad_clip = float(gc_cfg.get("max_norm", 1.0)) if gc_cfg.get("enabled", True) else None

        # Gradient accumulation
        self.grad_accum_steps = int(train_cfg.get("gradient_accumulation_steps", 1))

        # LR warmup
        self.warmup_epochs = int(train_cfg.get("lr_warmup_epochs", 5))
        self.warmup_start = float(train_cfg.get("lr_warmup_start_factor", 0.01))
        self.base_lr = float(opt_cfg.get("lr", 3e-4))

        # EMA
        ema_cfg = train_cfg.get("ema", {})
        self.use_ema = bool(ema_cfg.get("enabled", True))
        if self.use_ema:
            self.ema = EMAWeights(
                self.model,
                decay=float(ema_cfg.get("decay", 0.999)),
                warmup_steps=int(ema_cfg.get("warmup_steps", 200)),
            )

        # Early stopping
        es_cfg = train_cfg.get("early_stopping", {})
        self.patience = int(es_cfg.get("patience", 30))
        self.best_val_metric = float("inf")
        self.es_counter = 0

        # Checkpointing
        log_cfg = config.get("logging", {})
        ckpt_cfg = log_cfg.get("checkpoint", {})
        self.ckpt_dir = os.path.join(ROOT, ckpt_cfg.get("dir", "outputs/v3/checkpoints/"))
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.run_dir = os.path.dirname(self.ckpt_dir)

        # TensorBoard
        tb_dir = os.path.join(ROOT, log_cfg.get("tensorboard", {}).get("log_dir", "outputs/v3/logs/"))
        self.writer = self._build_writer(tb_dir)

        # Metrics log
        self.metrics_log_path = os.path.join(self.run_dir, "metrics.jsonl")
        os.makedirs(self.run_dir, exist_ok=True)

        # Feature augmentation params
        self.feature_smoothing_std = float(train_cfg.get("feature_smoothing_std", 0.0))

        print(f"\n{'='*60}")
        print(f"[V3 Trainer] NUCLEAR HEIGHT ESTIMATION TRAINING")
        print(f"{'='*60}")
        print(f"  Device: {self.device} | AMP: {self.use_amp}")
        print(f"  Parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        eff_bs = int(train_cfg.get('batch_size', 64)) * self.grad_accum_steps
        print(f"  Epochs: {self.epochs} | Batch: {train_cfg.get('batch_size', 64)} x {self.grad_accum_steps} = {eff_bs}")
        print(f"  LR: {self.base_lr} | WD: {opt_cfg.get('weight_decay', 0.05)}")
        print(f"  Warmup: {self.warmup_epochs} epochs | Patience: {self.patience}")
        print(f"  EMA: {self.use_ema}")
        print(f"  Checkpoint dir: {self.ckpt_dir}")
        print(f"{'='*60}\n")

    def _resolve_device(self, d):
        if d == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(d)

    def _build_writer(self, log_dir):
        try:
            from torch.utils.tensorboard import SummaryWriter
            return SummaryWriter(log_dir=log_dir)
        except Exception:
            return None

    def _set_warmup_lr(self, epoch: int, step: int, total_steps: int) -> None:
        if epoch > self.warmup_epochs:
            return
        total_warmup_steps = self.warmup_epochs * total_steps
        current_step = (epoch - 1) * total_steps + step
        progress = min(1.0, current_step / max(1, total_warmup_steps))
        factor = self.warmup_start + (1.0 - self.warmup_start) * progress
        for group in self.optimizer.param_groups:
            group["lr"] = self.base_lr * factor

    def _to_device(self, batch):
        out = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(self.device)
            else:
                out[k] = v
        return out

    def train(self) -> None:
        """Full training loop."""
        print("[V3] Starting training...\n")

        for epoch in range(1, self.epochs + 1):
            epoch_start = time.time()

            # ── Train epoch ──
            train_losses = self._train_epoch(epoch)

            # ── Validate (with EMA if enabled) ──
            ema_backup = None
            if self.use_ema:
                ema_backup = self.ema.swap_in()

            val_metrics = self._eval_epoch(self.val_loader, "val")

            if self.use_ema and ema_backup:
                self.ema.restore(ema_backup)

            # ── Step scheduler (after warmup) ──
            if epoch > self.warmup_epochs:
                self.scheduler.step()

            # ── Checkpointing ──
            current_metric = val_metrics.get("height_mae_speaker", float("inf"))
            is_best = current_metric < self.best_val_metric

            if is_best:
                self.best_val_metric = current_metric
                self.es_counter = 0
                self._save_checkpoint(epoch, current_metric, is_best=True)
                best_marker = " ** NEW BEST **"
            else:
                self.es_counter += 1
                self._save_checkpoint(epoch, current_metric, is_best=False)
                best_marker = ""

            # ── Epoch summary ──
            elapsed = time.time() - epoch_start
            lr = self.optimizer.param_groups[0]["lr"]
            rank_loss = train_losses.get('height_ranking', 0)
            iso_loss = train_losses.get('height_iso', 0)
            wing_loss = train_losses.get('height_wing', 0)
            calib_loss = train_losses.get('calibration_reg', 0)
            print(
                f"[Epoch {epoch:3d}/{self.epochs}] "
                f"loss={train_losses['total']:.4f} "
                f"wing={wing_loss:.4f} rank={rank_loss:.4f} iso={iso_loss:.4f} "
                f"calib={calib_loss:.4f} | "
                f"val_speaker_MAE={val_metrics['height_mae_speaker']:.3f}cm | "
                f"short={val_metrics.get('height_mae_short_speaker', 0):.3f}cm | "
                f"tall={val_metrics.get('height_mae_tall_speaker', 0):.3f}cm | "
                f"lr={lr:.2e} | "
                f"es={self.es_counter}/{self.patience} | "
                f"{elapsed:.0f}s{best_marker}"
            )

            # ── Log ──
            self._log_epoch(epoch, train_losses, val_metrics, lr)

            # ── Early stopping ──
            if self.es_counter >= self.patience:
                print(f"\n[V3] Early stopping triggered at epoch {epoch}")
                break

        # ── Final evaluation ──
        print(f"\n{'='*60}")
        print("[V3] FINAL EVALUATION")
        print(f"{'='*60}")

        # Load best model
        best_path = os.path.join(self.ckpt_dir, "best.ckpt")
        if os.path.exists(best_path):
            ckpt = torch.load(best_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model_state_dict"])
            if self.use_ema and "ema_state" in ckpt:
                self.ema.load_state_dict(ckpt["ema_state"])
                ema_backup = self.ema.swap_in()
            print(f"[V3] Loaded best checkpoint from epoch {ckpt.get('epoch', '?')}")

        val_final = self._eval_epoch(self.val_loader, "val")
        test_final = self._eval_epoch(self.test_loader, "test")

        print(f"\n  VAL  speaker MAE: {val_final['height_mae_speaker']:.3f} cm")
        print(f"  VAL  short MAE:   {val_final.get('height_mae_short_speaker', 0):.3f} cm")
        print(f"  VAL  tall MAE:    {val_final.get('height_mae_tall_speaker', 0):.3f} cm")
        print(f"\n  TEST speaker MAE: {test_final['height_mae_speaker']:.3f} cm")
        print(f"  TEST short MAE:   {test_final.get('height_mae_short_speaker', 0):.3f} cm")
        print(f"  TEST tall MAE:    {test_final.get('height_mae_tall_speaker', 0):.3f} cm")
        print(f"  TEST RMSE:        {test_final.get('height_rmse_speaker', 0):.3f} cm")
        print(f"  TEST median AE:   {test_final.get('height_median_ae_speaker', 0):.3f} cm")

        # Save final metrics
        metrics_out = os.path.join(self.run_dir, "final_metrics.json")
        final = {"val": val_final, "test": test_final, "best_val_metric": self.best_val_metric}
        with open(metrics_out, "w") as f:
            json.dump(final, f, indent=2)
        print(f"\n[V3] Metrics saved to {metrics_out}")

        if self.use_ema and ema_backup:
            self.ema.restore(ema_backup)

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_steps = len(self.train_loader)
        epoch_losses = defaultdict(float)
        n_batches = 0

        self.optimizer.zero_grad(set_to_none=True)

        import time
        start_time = time.time()
        for step, batch in enumerate(self.train_loader):
            if step % 20 == 0 and step > 0:
                elapsed = time.time() - start_time
                print(f"[Epoch {epoch}] Step {step}/{total_steps} | elapsed: {elapsed:.1f}s | memory: {torch.cuda.memory_allocated() / 1024**2:.0f}MB", flush=True)
            # Warmup LR
            self._set_warmup_lr(epoch, step, total_steps)

            batch = self._to_device(batch)
            features = batch["sequence"]
            padding_mask = batch.get("padding_mask")

            # Feature smoothing (training-only noise)
            if self.feature_smoothing_std > 0.0:
                noise = torch.randn_like(features) * self.feature_smoothing_std
                if padding_mask is not None:
                    noise = noise.masked_fill(padding_mask.unsqueeze(-1), 0.0)
                features = features + noise

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                preds = self.model(
                    features,
                    padding_mask=padding_mask,
                    height_targets=batch["height"],
                    gender_targets=batch["gender"],
                )
                # Use mixed targets from mixup if available
                height_target = preds.pop("mixed_height_targets", batch["height"])
                preds.pop("mixup_lambda", None)
                targets = {
                    "height": height_target,
                    "gender": batch["gender"],
                    "height_cm": batch.get("height_raw"),
                }
                losses = self.criterion(preds, targets)

            # Scale loss by accumulation steps
            loss = losses["total"] / self.grad_accum_steps
            self.scaler.scale(loss).backward()

            # Step optimizer every grad_accum_steps
            if (step + 1) % self.grad_accum_steps == 0 or (step + 1) == total_steps:
                if self.grad_clip:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

                # EMA update (only on actual optimizer steps)
                if self.use_ema:
                    self.ema.update()

            for k, v in losses.items():
                epoch_losses[k] += float(v.item()) if isinstance(v, torch.Tensor) else float(v)
            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in epoch_losses.items()}

    @torch.no_grad()
    def _eval_epoch(
        self, loader: DataLoader, split_name: str
    ) -> Dict[str, float]:
        self.model.eval()
        all_preds = []
        all_targets = []
        all_speaker_ids = []
        all_gender_preds = []
        all_gender_targets = []

        for batch in loader:
            batch = self._to_device(batch)
            features = batch["sequence"]
            padding_mask = batch.get("padding_mask")

            preds = self.model(features, padding_mask=padding_mask)

            all_preds.append(preds["height"].detach().cpu())
            all_targets.append(batch["height"].detach().cpu())
            all_speaker_ids.append(batch["speaker_id"])

            if "gender_logits" in preds:
                all_gender_preds.append(preds["gender_logits"].argmax(dim=-1).detach().cpu())
                all_gender_targets.append(batch["gender"].detach().cpu())

        metrics = compute_height_metrics(
            all_preds, all_targets, all_speaker_ids, self.target_stats
        )

        # Gender accuracy
        if all_gender_preds and all_gender_targets:
            g_pred = torch.cat(all_gender_preds)
            g_true = torch.cat(all_gender_targets)
            metrics["gender_accuracy"] = float((g_pred == g_true).float().mean().item())

        return metrics

    def _save_checkpoint(self, epoch: int, metric: float, is_best: bool) -> None:
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict() if self.use_amp else None,
            "best_metric": self.best_val_metric,
            "metric_val": metric,
            "es_counter": self.es_counter,
        }
        if self.use_ema:
            state["ema_state"] = self.ema.state_dict()

        # Always save last
        last_path = os.path.join(self.ckpt_dir, "last.ckpt")
        torch.save(state, last_path)

        if is_best:
            best_path = os.path.join(self.ckpt_dir, "best.ckpt")
            torch.save(state, best_path)

    def _log_epoch(
        self,
        epoch: int,
        train_losses: Dict[str, float],
        val_metrics: Dict[str, float],
        lr: float,
    ) -> None:
        record = {
            "epoch": epoch,
            "lr": lr,
            "train": train_losses,
            "val": val_metrics,
        }

        with open(self.metrics_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, allow_nan=True) + "\n")

        if self.writer:
            for k, v in train_losses.items():
                self.writer.add_scalar(f"train/{k}", v, epoch)
            for k, v in val_metrics.items():
                self.writer.add_scalar(f"val/{k}", v, epoch)
            self.writer.add_scalar("lr", lr, epoch)
            self.writer.flush()

    def close(self) -> None:
        if self.writer:
            self.writer.flush()
            self.writer.close()


# ────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(description="Train VocalMorph V3")
    parser.add_argument("--config", type=str, default="configs/v3_nuclear.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    return parser.parse_args()


def main():
    args = parse_args()

    config_path = args.config if os.path.isabs(args.config) else os.path.join(ROOT, args.config)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    train_cfg = config["training"]
    if args.seed is not None:
        train_cfg["seed"] = args.seed
    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs
    if args.device is not None:
        train_cfg["device"] = args.device

    seed = int(train_cfg.get("seed", 42))

    # Disk logging
    log_cfg = config.get("logging", {})
    ckpt_dir = os.path.join(ROOT, log_cfg.get("checkpoint", {}).get("dir", "outputs/v3/checkpoints/"))
    run_dir = os.path.dirname(ckpt_dir)
    os.makedirs(run_dir, exist_ok=True)
    stdout_handle = open(os.path.join(run_dir, "train.stdout.log"), "a", encoding="utf-8", buffering=1)
    stderr_handle = open(os.path.join(run_dir, "train.stderr.log"), "a", encoding="utf-8", buffering=1)
    sys.stdout = TeeStream(sys.stdout, stdout_handle)
    sys.stderr = TeeStream(sys.stderr, stderr_handle)

    trainer = None
    try:
        seed_everything(seed)
        configure_cuda(bool(train_cfg.get("allow_tf32", True)))

        print(f"\n[V3] Seed: {seed}")
        print(f"[V3] Config: {config_path}")

        # Load data
        data_cfg = config["data"]
        feat_dir = os.path.join(ROOT, data_cfg["features_dir"])
        feature_config = build_feature_config(config)

        split_manifest_cfg = data_cfg.get("split_manifests", {})
        expected_split_files = {
            "train": os.path.join(ROOT, split_manifest_cfg.get("train", "data/splits/train_clean.csv")),
            "val": os.path.join(ROOT, split_manifest_cfg.get("val", "data/splits/val_clean.csv")),
            "test": os.path.join(ROOT, split_manifest_cfg.get("test", "data/splits/test_clean.csv")),
        }
        validate_feature_contract(
            feature_root=feat_dir,
            expected_feature_config=feature_config.to_dict(),
            expected_split_files=expected_split_files,
            require_target_stats=True,
        )
        print("[V3] Feature contract verified.")

        # Target stats
        stats_path = os.path.join(feat_dir, "target_stats.json")
        target_stats = None
        if os.path.exists(stats_path):
            with open(stats_path, "r") as f:
                target_stats = json.load(f)
            print(f"[V3] Target stats: height mean={target_stats['height']['mean']:.1f}, std={target_stats['height']['std']:.1f}")

        # Augmentation
        aug_cfg = train_cfg.get("augmentation", {})
        train_augment = bool(aug_cfg.get("enabled", True))
        augment_config = None
        if train_augment:
            augment_config = FeatureAugmentConfig(
                noise_p=float(aug_cfg.get("noise_p", 0.50)),
                noise_std=float(aug_cfg.get("noise_std", 0.015)),
                time_mask_p=float(aug_cfg.get("time_mask_p", 0.30)),
                time_mask_max_frac=float(aug_cfg.get("time_mask_max_frac", 0.12)),
                feat_mask_p=float(aug_cfg.get("feat_mask_p", 0.25)),
                feat_mask_max_frac=float(aug_cfg.get("feat_mask_max_frac", 0.10)),
                scale_p=float(aug_cfg.get("scale_p", 0.40)),
                scale_std=float(aug_cfg.get("scale_std", 0.05)),
                temporal_jitter_p=float(aug_cfg.get("temporal_jitter_p", 0.25)),
                temporal_jitter_max_frac=float(aug_cfg.get("temporal_jitter_max_frac", 0.05)),
            )
            print("[V3] Augmentation: ENABLED (strong)")

        # Build dataloaders
        max_feature_frames = train_cfg.get("max_feature_frames", 960)
        num_workers = int(train_cfg.get("num_workers", 4))
        prefetch_factor = train_cfg.get("prefetch_factor", 2)
        if num_workers == 0:
            prefetch_factor = None
            persistent_workers = False
        else:
            prefetch_factor = int(prefetch_factor) if prefetch_factor is not None else 2
            persistent_workers = bool(train_cfg.get("persistent_workers", True))

        train_loader, val_loader, test_loader = build_dataloaders_from_dirs(
            train_dir=os.path.join(feat_dir, "train"),
            val_dir=os.path.join(feat_dir, "val"),
            test_dir=os.path.join(feat_dir, "test"),
            batch_size=int(train_cfg.get("batch_size", 64)),
            num_workers=num_workers,
            target_stats=target_stats,
            max_len=max_feature_frames,
            train_crop_mode=str(train_cfg.get("train_crop_mode", "random")),
            eval_crop_mode=str(train_cfg.get("eval_crop_mode", "center")),
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            train_augment=train_augment,
            augment_config=augment_config,
            speaker_batching={"enabled": False},
            base_seed=seed,
        )

        # Input dim
        input_dim = train_loader.dataset.infer_input_dim()
        config.setdefault("model", {})["input_dim"] = int(input_dim)
        print(f"[V3] Input dim: {input_dim}")
        print(f"[V3] Train: {len(train_loader.dataset)} clips | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")

        # Build model + loss
        model = build_v3_model(config)
        criterion = build_v3_loss(config)

        # Build trainer
        trainer = V3Trainer(
            model=model,
            criterion=criterion,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            config=config,
            target_stats=target_stats,
        )

        # Resume if requested
        if args.resume:
            resume_path = args.resume if os.path.isabs(args.resume) else os.path.join(ROOT, args.resume)
            if os.path.exists(resume_path):
                ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
                model.load_state_dict(ckpt["model_state_dict"])
                trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                if ckpt.get("scheduler_state_dict"):
                    trainer.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                print(f"[V3] Resumed from {resume_path}")

        # Train
        trainer.train()
        return 0

    except KeyboardInterrupt:
        print("\n[V3] Interrupted by user.", file=sys.stderr)
        return 130
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        if trainer:
            trainer.close()
        sys.stdout.flush()
        sys.stderr.flush()


if __name__ == "__main__":
    raise SystemExit(main())
