"""
VocalMorph trainer.

Adds:
- gender-weighted classification loss
- NISP-only masked weight loss
- empty split safety for validation/test
"""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import sys
import tempfile
import time
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml


class _NoOpSummaryWriter:
    """No-op fallback when TensorBoard is unavailable locally."""

    def add_scalar(self, *args, **kwargs):
        pass

    def flush(self):
        pass

    def close(self):
        pass


def _build_summary_writer(log_dir: str, enabled: bool = True):
    if not enabled:
        return _NoOpSummaryWriter()
    try:
        from torch.utils.tensorboard import SummaryWriter as TorchSummaryWriter

        return TorchSummaryWriter(log_dir=log_dir)
    except Exception as exc:  # pragma: no cover - environment-specific dependency fault
        print(
            "[Trainer] TensorBoard unavailable; continuing without SummaryWriter "
            f"({exc})"
        )
        return _NoOpSummaryWriter()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(ROOT))

from src.models.pibnn import VocalMorphLoss  # noqa: E402
from src.utils.metrics import compute_metrics  # noqa: E402
from src.utils.audit_utils import duration_bin, height_bin, quality_bucket  # noqa: E402


def _torch_load_checkpoint(path: str) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _capture_rng_state() -> Dict[str, Any]:
    numpy_state = np.random.get_state()
    payload: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "state": numpy_state[1].tolist(),
            "pos": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch": torch.get_rng_state().cpu(),
    }
    if torch.cuda.is_available():
        payload["torch_cuda"] = [state.cpu() for state in torch.cuda.get_rng_state_all()]
    return payload


def _restore_rng_state(state: Optional[Dict[str, Any]]) -> None:
    if not state:
        return
    python_state = state.get("python")
    if python_state is not None:
        random.setstate(python_state)

    numpy_state = state.get("numpy")
    if isinstance(numpy_state, dict):
        np.random.set_state(
            (
                numpy_state.get("bit_generator", "MT19937"),
                np.asarray(numpy_state.get("state", []), dtype=np.uint32),
                int(numpy_state.get("pos", 0)),
                int(numpy_state.get("has_gauss", 0)),
                float(numpy_state.get("cached_gaussian", 0.0)),
            )
        )

    torch_state = state.get("torch")
    if isinstance(torch_state, torch.Tensor):
        torch.set_rng_state(torch_state.cpu())

    cuda_states = state.get("torch_cuda")
    if torch.cuda.is_available() and isinstance(cuda_states, list) and cuda_states:
        torch.cuda.set_rng_state_all(
            [item.cpu() if isinstance(item, torch.Tensor) else torch.tensor(item, dtype=torch.uint8) for item in cuda_states]
        )


class StochasticWeightAveraging:
    """Averages model weights over the last N checkpoints for better generalization."""

    def __init__(self, model, start_epoch: int, anneal_epochs: int = 10):
        self.model = model
        self.start_epoch = int(start_epoch)
        self.anneal_epochs = int(anneal_epochs)
        self.swa_state: Dict[str, torch.Tensor] = {}
        self.n_averaged: int = 0
        self.enabled: bool = False

    def update(self, epoch: int) -> None:
        if epoch < self.start_epoch:
            return
        self.enabled = True
        decay = 1.0 / (self.n_averaged + 1)
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                if name not in self.swa_state:
                    self.swa_state[name] = param.detach().clone()
                else:
                    self.swa_state[name].mul_(1.0 - decay).add_(
                        param.detach(), alpha=decay
                    )
        self.n_averaged += 1

    def apply(self) -> Dict[str, torch.Tensor]:
        if not self.enabled or not self.swa_state:
            return {}
        backup: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.swa_state:
                    backup[name] = param.detach().clone()
                    param.copy_(self.swa_state[name])
        return backup

    def restore(self, backup: Dict[str, torch.Tensor]) -> None:
        if not backup:
            return
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in backup:
                    param.copy_(backup[name])

    def state_dict(self) -> Dict[str, Any]:
        return {
            "start_epoch": self.start_epoch,
            "anneal_epochs": self.anneal_epochs,
            "n_averaged": self.n_averaged,
            "enabled": self.enabled,
            "swa_state": {name: tensor.detach().clone() for name, tensor in self.swa_state.items()},
        }

    def load_state_dict(self, state: Optional[Dict[str, Any]]) -> None:
        if not state:
            return
        self.start_epoch = int(state.get("start_epoch", self.start_epoch))
        self.anneal_epochs = int(state.get("anneal_epochs", self.anneal_epochs))
        self.n_averaged = int(state.get("n_averaged", 0))
        self.enabled = bool(state.get("enabled", False))
        swa_state = state.get("swa_state", {})
        loaded: Dict[str, torch.Tensor] = {}
        for name, tensor in swa_state.items():
            if isinstance(tensor, torch.Tensor):
                loaded[name] = tensor.detach().clone()
        self.swa_state = loaded


class EpochMetricsLogger:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def append(self, record: Dict[str, Any]) -> None:
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, allow_nan=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


class CheckpointManager:
    def __init__(self, save_dir: str, top_k: int = 3, mode: str = "min"):
        self.save_dir = save_dir
        self.top_k = top_k
        self.mode = mode
        self.checkpoints: List[Tuple[float, str]] = []
        self.last_path = os.path.join(save_dir, "last.ckpt")
        self.last_good_path = os.path.join(save_dir, "last_good.ckpt")
        self.best_path = os.path.join(save_dir, "best.ckpt")
        os.makedirs(save_dir, exist_ok=True)

    @staticmethod
    def _atomic_torch_save(payload: Dict[str, Any], path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            prefix=f"{os.path.basename(path)}.",
            suffix=".tmp",
            dir=os.path.dirname(path),
        )
        os.close(fd)
        try:
            torch.save(payload, tmp_path)
            _torch_load_checkpoint(tmp_path)
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    @staticmethod
    def _atomic_copy(src: str, dst: str) -> None:
        fd, tmp_path = tempfile.mkstemp(
            prefix=f"{os.path.basename(dst)}.",
            suffix=".tmp",
            dir=os.path.dirname(dst),
        )
        os.close(fd)
        try:
            shutil.copy2(src, tmp_path)
            os.replace(tmp_path, dst)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    @staticmethod
    def is_valid_checkpoint(path: str) -> bool:
        if not path or not os.path.exists(path):
            return False
        try:
            payload = _torch_load_checkpoint(path)
        except Exception:
            return False
        return isinstance(payload, dict) and "model_state_dict" in payload

    def save(self, state: Dict[str, Any], metric_val: float, is_best: bool) -> str:
        epoch = int(state.get("epoch", 0))
        archive_path = os.path.join(
            self.save_dir, f"epoch_{epoch:04d}_metric_{metric_val:.4f}.ckpt"
        )

        if self.top_k > 0:
            self._atomic_torch_save(state, archive_path)
            self.checkpoints.append((metric_val, archive_path))
            reverse = self.mode == "max"
            self.checkpoints.sort(key=lambda x: x[0], reverse=reverse)
            while len(self.checkpoints) > self.top_k:
                _, old_path = self.checkpoints.pop()
                if os.path.exists(old_path):
                    os.remove(old_path)

        self._atomic_torch_save(state, self.last_path)
        self._atomic_copy(self.last_path, self.last_good_path)

        if is_best or not os.path.exists(self.best_path):
            source = archive_path if os.path.exists(archive_path) else self.last_path
            self._atomic_copy(source, self.best_path)

        return archive_path if os.path.exists(archive_path) else self.last_path

    def save_emergency(self, state: Dict[str, Any], filename: str) -> str:
        path = os.path.join(self.save_dir, filename)
        self._atomic_torch_save(state, path)
        return path

    def best_metric(self):
        return self.checkpoints[0][0] if self.checkpoints else None


class VocalMorphTrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        config,
        target_stats=None,
        train_eval_loader=None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_eval_loader = train_eval_loader
        self.config = config
        self.target_stats = target_stats
        self.last_train_eval_metrics: Dict[str, float] = {}
        self.start_epoch = 1
        self.last_completed_epoch = 0

        train_cfg = config.get("training", {})
        self.progress_log_interval_steps = max(
            1, int(train_cfg.get("progress_log_interval_steps", 100))
        )
        self.epochs = int(train_cfg.get("epochs", 100))
        self.device = self._resolve_device(train_cfg.get("device", "auto"))
        amp_enabled = train_cfg.get("amp", train_cfg.get("mixed_precision", True))
        self.use_amp = bool(amp_enabled) and self.device.type == "cuda"
        self.gradient_accumulation_steps = max(
            1, int(train_cfg.get("gradient_accumulation_steps", 1))
        )
        self.model = self.model.to(self.device)
        self.loss_type = str(train_cfg.get("loss", {}).get("type", "default")).lower()
        self.grl_lambda_max = float(
            train_cfg.get("loss", {}).get("domain_grl_lambda_max", 0.10)
        )
        self.grl_lambda_min = float(
            train_cfg.get("loss", {}).get("domain_grl_lambda_min", 0.00)
        )

        gender_weights = None
        if train_cfg.get("loss", {}).get("use_gender_class_weights", True):
            gender_weights = self._compute_gender_class_weights()

        lw = train_cfg.get("loss", {}).get("task_weights", {})
        loss_cfg = train_cfg.get("loss", {})
        self.use_native_v2_loss = self.loss_type == "vtsl_v2" and hasattr(
            self.model, "loss_module"
        )
        if self.use_native_v2_loss:
            self.criterion = self.model.loss_module
            if hasattr(self.criterion, "set_target_stats"):
                self.criterion.set_target_stats(target_stats)
        elif self.loss_type == "vtsl_v2":
            from src.models.vocalmorphv2 import (  # noqa: E402
                SpeakerAlignmentConfig,
                VocalTractSimulatorLossV2,
                dataclass_from_mapping,
            )

            vtl_ratio = (
                config.get("physics", {})
                .get("vtl_height_constraint", {})
                .get("ratio", 6.7)
            )
            self.criterion = VocalTractSimulatorLossV2(
                vtl_height_ratio=vtl_ratio,
                robust_huber_weight=float(
                    loss_cfg.get("robust_huber_weight", 0.20)
                ),
                focal_after_epoch=int(loss_cfg.get("focal_after_epoch", 20)),
                focal_ema_decay=float(loss_cfg.get("focal_ema_decay", 0.95)),
                target_stats=target_stats,
                speaker_alignment=dataclass_from_mapping(
                    SpeakerAlignmentConfig,
                    train_cfg.get("speaker_alignment"),
                ),
            )
        else:
            self.criterion = VocalMorphLoss(
                height_weight=lw.get("height", 1.0),
                weight_weight=lw.get("weight", 1.0),
                age_weight=lw.get("age", 1.0),
                gender_weight=lw.get("gender", 2.0),
                physics_weight=loss_cfg.get("physics_penalty_weight", 0.2),
                gender_class_weights=gender_weights,
            )

        focal_target = getattr(self.criterion, "base_loss", self.criterion)
        if hasattr(focal_target, "focal_after_epoch"):
            focal_target.focal_after_epoch = int(
                loss_cfg.get("focal_after_epoch", focal_target.focal_after_epoch)
            )
        if hasattr(focal_target, "focal_ema_decay"):
            focal_target.focal_ema_decay = float(
                loss_cfg.get("focal_ema_decay", focal_target.focal_ema_decay)
            )

        opt_cfg = train_cfg.get("optimizer", {})
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=opt_cfg.get("lr", 3e-5),
            weight_decay=opt_cfg.get("weight_decay", 0.01),
            betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
        )
        self.base_lrs = [float(group["lr"]) for group in self.optimizer.param_groups]

        sched_cfg = train_cfg.get("scheduler", {})
        self.scheduler_type = str(
            sched_cfg.get("type", "cosine_annealing_warm_restarts")
        ).strip()
        self.scheduler = self._build_scheduler(sched_cfg)

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        gc_cfg = train_cfg.get("gradient_clipping", {})
        self.grad_clip = (
            gc_cfg.get("max_norm", 1.0) if gc_cfg.get("enabled", True) else None
        )
        self.feature_smoothing_std = float(train_cfg.get("feature_smoothing_std", 0.0))
        self.lr_warmup_epochs = max(0, int(train_cfg.get("lr_warmup_epochs", 0)))
        self.lr_warmup_start_factor = float(
            train_cfg.get("lr_warmup_start_factor", 0.25)
        )

        es_cfg = train_cfg.get("early_stopping", {})
        self.early_stopping_enabled = es_cfg.get("enabled", True)
        self.patience = int(es_cfg.get("patience", 15))
        self.es_counter = 0
        self.es_monitor = self._normalize_monitor_key(
            es_cfg.get("monitor", "val_total_loss")
        )
        self.es_mode = str(es_cfg.get("mode", "min")).lower()
        self.best_val_metric = float("inf") if self.es_mode == "min" else float("-inf")

        log_cfg = config.get("logging", {})
        tb_dir = log_cfg.get("tensorboard", {}).get("log_dir", "outputs/logs")
        resolved_tb_dir = os.path.join(ROOT, tb_dir) if not os.path.isabs(tb_dir) else tb_dir
        self.writer = _build_summary_writer(
            log_dir=resolved_tb_dir,
            enabled=bool(log_cfg.get("tensorboard", {}).get("enabled", True)),
        )

        ckpt_cfg = log_cfg.get("checkpoint", {})
        ckpt_dir = ckpt_cfg.get("dir", "outputs/checkpoints")
        self.ckpt_monitor = self._normalize_monitor_key(
            ckpt_cfg.get("monitor", self.es_monitor)
        )
        self.ckpt_mode = str(ckpt_cfg.get("mode", self.es_mode)).lower()
        self.ckpt_manager = CheckpointManager(
            save_dir=os.path.join(ROOT, ckpt_dir)
            if not os.path.isabs(ckpt_dir)
            else ckpt_dir,
            top_k=int(ckpt_cfg.get("save_top_k", 3)),
            mode=self.ckpt_mode,
        )
        self.run_dir = os.path.dirname(self.ckpt_manager.save_dir)
        self.metrics_logger = EpochMetricsLogger(
            os.path.join(self.run_dir, "metrics.jsonl")
        )

        ema_cfg = train_cfg.get("ema", {})
        self.use_ema = bool(ema_cfg.get("enabled", True))
        self.ema_decay = float(ema_cfg.get("decay", 0.998))
        self.ema_update_every = max(1, int(ema_cfg.get("update_every", 1)))
        self.ema_warmup_steps = max(0, int(ema_cfg.get("warmup_steps", 0)))
        self.ema_use_for_eval = bool(ema_cfg.get("use_for_eval", True))
        self.optimizer_step_count = 0
        self.ema_state: Dict[str, torch.Tensor] = {}
        if self.use_ema:
            self._init_ema_state()

        # Stochastic Weight Averaging
        swa_cfg = train_cfg.get("swa", {})
        self.use_swa = bool(swa_cfg.get("enabled", True))
        swa_start_frac = float(swa_cfg.get("start_frac", 0.75))
        swa_start_epoch = max(1, int(self.epochs * swa_start_frac))
        self.swa = StochasticWeightAveraging(
            self.model,
            start_epoch=swa_start_epoch,
            anneal_epochs=int(swa_cfg.get("anneal_epochs", 10)),
        )
        self.swa_use_for_eval = bool(swa_cfg.get("use_for_eval", True))

        eval_cfg = config.get("evaluation", {}).get("inference", {})
        self.eval_use_ensemble = bool(eval_cfg.get("use_ensemble", True))
        self.eval_deterministic = bool(eval_cfg.get("deterministic", True))
        self.eval_n_samples = max(1, int(eval_cfg.get("n_samples", 8)))
        self.eval_n_crops = max(1, int(eval_cfg.get("n_crops", 3)))
        self.eval_crop_size = eval_cfg.get("crop_size")
        if self.eval_crop_size is not None:
            self.eval_crop_size = int(self.eval_crop_size)

        print(f"[Trainer] Device: {self.device} | AMP: {self.use_amp}")
        print(
            f"[Trainer] Parameters: {sum(p.numel() for p in self.model.parameters()):,}"
        )
        if gender_weights is not None:
            print(f"[Trainer] Gender class weights: {gender_weights.tolist()}")
        if self.loss_type == "vtsl_v2":
            print(
                f"[Trainer] Loss: VTSL v2 ({'native' if self.use_native_v2_loss else 'external'}) "
                f"| GRL lambda in [{self.grl_lambda_min}, {self.grl_lambda_max}]"
            )
        print(f"[Trainer] Grad accumulation: {self.gradient_accumulation_steps}")
        if self.feature_smoothing_std > 0.0:
            print(
                f"[Trainer] Feature smoothing: std={self.feature_smoothing_std:.4f}"
            )
        if self.lr_warmup_epochs > 0:
            print(
                f"[Trainer] LR warmup: epochs={self.lr_warmup_epochs} "
                f"| start_factor={self.lr_warmup_start_factor:.3f}"
            )
        print(f"[Trainer] Scheduler: {self.scheduler_type}")
        print(f"[Trainer] Early stop monitor: {self.es_monitor} ({self.es_mode})")
        print(f"[Trainer] Checkpoint monitor: {self.ckpt_monitor} ({self.ckpt_mode})")
        print(
            f"[Trainer] EMA: {self.use_ema} | decay={self.ema_decay} | eval={self.ema_use_for_eval}"
        )
        if self.use_swa:
            print(
                f"[Trainer] SWA: enabled | start_epoch={self.swa.start_epoch} "
                f"| anneal={self.swa.anneal_epochs} | eval={self.swa_use_for_eval}"
            )
        print(
            f"[Trainer] Eval ensemble: {self.eval_use_ensemble} "
            f"| deterministic={self.eval_deterministic} | crops={self.eval_n_crops} | crop_size={self.eval_crop_size}"
        )
        print(f"[Trainer] Train-eval loader: {'enabled' if self.train_eval_loader is not None else 'disabled'}")

    def _compute_gender_class_weights(self) -> Optional[torch.Tensor]:
        ds = getattr(self.train_loader, "dataset", None)
        if ds is None or not hasattr(ds, "gender_counts"):
            return None

        counts = ds.gender_counts()
        n0 = max(float(counts.get(0, 0)), 1.0)
        n1 = max(float(counts.get(1, 0)), 1.0)
        total = n0 + n1
        w0 = total / (2.0 * n0)
        w1 = total / (2.0 * n1)
        return torch.tensor([w0, w1], dtype=torch.float32)

    def _named_trainable_parameters(self):
        if hasattr(self.model, "named_parameters_for_ema") and callable(
            getattr(self.model, "named_parameters_for_ema")
        ):
            yield from self.model.named_parameters_for_ema()
            return
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                yield name, param

    def _init_ema_state(self) -> None:
        self.ema_state = {
            name: param.detach().clone()
            for name, param in self._named_trainable_parameters()
        }
        if not self.ema_state:
            self.use_ema = False

    def ema_state_dict(self) -> Dict[str, torch.Tensor]:
        return {name: value.detach().clone() for name, value in self.ema_state.items()}

    def load_ema_state_dict(self, state: Optional[Dict[str, torch.Tensor]]) -> None:
        if not state:
            return
        loaded: Dict[str, torch.Tensor] = {}
        for name, param in self._named_trainable_parameters():
            if name not in state:
                continue
            loaded[name] = (
                state[name].detach().clone().to(device=param.device, dtype=param.dtype)
            )
        if loaded:
            self.ema_state = loaded
            self.use_ema = True

    def _update_ema(self) -> None:
        if not self.use_ema or not self.ema_state:
            return
        if self.optimizer_step_count % self.ema_update_every != 0:
            return
        decay = (
            self.ema_decay if self.optimizer_step_count > self.ema_warmup_steps else 0.0
        )
        with torch.no_grad():
            for name, param in self._named_trainable_parameters():
                shadow = self.ema_state.get(name)
                if shadow is None:
                    self.ema_state[name] = param.detach().clone()
                    continue
                shadow.mul_(decay).add_(param.detach(), alpha=1.0 - decay)

    def _swap_in_ema_weights(self) -> Optional[Dict[str, torch.Tensor]]:
        if not self.use_ema or not self.ema_use_for_eval or not self.ema_state:
            return None
        backup: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for name, param in self._named_trainable_parameters():
                shadow = self.ema_state.get(name)
                if shadow is None:
                    continue
                backup[name] = param.detach().clone()
                param.copy_(shadow)
        return backup

    def _restore_weights(self, backup: Optional[Dict[str, torch.Tensor]]) -> None:
        if not backup:
            return
        with torch.no_grad():
            for name, param in self._named_trainable_parameters():
                if name in backup:
                    param.copy_(backup[name])

    def _load_model_checkpoint_state(self, state_dict: Mapping[str, Any]) -> None:
        try:
            self.model.load_state_dict(state_dict)
            return
        except RuntimeError as exc:
            incompatible = self.model.load_state_dict(state_dict, strict=False)
            allowed_missing_prefixes = ("reliability_tower.",)
            disallowed_missing = [
                key
                for key in incompatible.missing_keys
                if not key.startswith(allowed_missing_prefixes)
            ]
            if disallowed_missing or incompatible.unexpected_keys:
                raise RuntimeError(
                    "Checkpoint/model mismatch is not safely recoverable. "
                    f"Missing={disallowed_missing or incompatible.missing_keys}, "
                    f"Unexpected={list(incompatible.unexpected_keys)}"
                ) from exc
            if incompatible.missing_keys:
                print(
                    "[Trainer] Loaded checkpoint with forward-compatible missing keys: "
                    f"{list(incompatible.missing_keys)}"
                )

    def restore_from_checkpoint(
        self,
        checkpoint: Dict[str, Any],
        checkpoint_path: Optional[str] = None,
        model_only: bool = False,
    ) -> None:
        self._load_model_checkpoint_state(checkpoint["model_state_dict"])
        self.best_val_metric = float(
            checkpoint.get("best_metric", checkpoint.get("metric_val", self.best_val_metric))
        )
        self.last_completed_epoch = int(checkpoint.get("epoch", 0))
        self.start_epoch = self.last_completed_epoch + 1
        if model_only:
            if checkpoint_path:
                print(
                    f"[Trainer] Loaded model-only replay checkpoint from {checkpoint_path} "
                    f"(epoch={self.last_completed_epoch})"
                )
            return
        optimizer_state = checkpoint.get("optimizer_state_dict")
        if optimizer_state:
            self.optimizer.load_state_dict(optimizer_state)
        scheduler_state = checkpoint.get("scheduler_state_dict")
        if scheduler_state:
            self.scheduler.load_state_dict(scheduler_state)
        scaler_state = checkpoint.get("scaler_state_dict")
        if scaler_state and self.use_amp:
            self.scaler.load_state_dict(scaler_state)
        if checkpoint.get("ema_state_dict") is not None:
            self.load_ema_state_dict(checkpoint.get("ema_state_dict"))
        if checkpoint.get("swa_state_dict") is not None:
            self.swa.load_state_dict(checkpoint.get("swa_state_dict"))

        self.es_counter = int(checkpoint.get("es_counter", self.es_counter))
        self.optimizer_step_count = int(
            checkpoint.get("optimizer_step_count", checkpoint.get("global_step", 0))
        )
        self._restore_sampler_state(checkpoint.get("sampler_state"))
        _restore_rng_state(checkpoint.get("rng_state"))

        if checkpoint_path:
            print(
                f"[Trainer] Resumed from {checkpoint_path} "
                f"(epoch={self.last_completed_epoch}, next_epoch={self.start_epoch})"
            )

    def _build_checkpoint_state(
        self,
        *,
        epoch: int,
        metric_val: float,
        train_losses: Optional[Dict[str, float]] = None,
        train_eval_metrics: Optional[Dict[str, float]] = None,
        val_metrics: Optional[Dict[str, float]] = None,
        termination_reason: Optional[str] = None,
        exception_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        return {
            "epoch": int(epoch),
            "global_step": int(self.optimizer_step_count),
            "optimizer_step_count": int(self.optimizer_step_count),
            "best_metric": float(self.best_val_metric),
            "metric_val": float(metric_val),
            "es_counter": int(self.es_counter),
            "monitor_name": self.es_monitor,
            "monitor_mode": self.es_mode,
            "checkpoint_monitor": self.ckpt_monitor,
            "checkpoint_mode": self.ckpt_mode,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict()
            if self.scheduler is not None
            else None,
            "scaler_state_dict": self.scaler.state_dict()
            if self.scaler is not None and self.use_amp
            else None,
            "ema_state_dict": self.ema_state_dict() if self.use_ema and self.ema_state else None,
            "swa_state_dict": self.swa.state_dict() if self.use_swa else None,
            "config": self.config,
            "sampler_state": self._sampler_state(),
            "rng_state": _capture_rng_state(),
            "train_losses": train_losses or {},
            "train_eval_metrics": train_eval_metrics or {},
            "val_metrics": val_metrics or {},
            "termination_reason": termination_reason,
            "exception_text": exception_text,
        }

    def save_emergency_checkpoint(
        self, reason: str, exception_text: Optional[str] = None
    ) -> Optional[str]:
        filename = "interrupt.ckpt" if reason == "interrupt" else "crash.ckpt"
        metric_val = (
            float(self.best_val_metric)
            if np.isfinite(self.best_val_metric)
            else float("nan")
        )
        state = self._build_checkpoint_state(
            epoch=self.last_completed_epoch,
            metric_val=metric_val,
            train_eval_metrics=self.last_train_eval_metrics,
            termination_reason=reason,
            exception_text=exception_text,
        )
        path = self.ckpt_manager.save_emergency(state, filename)
        print(f"[Trainer] Saved emergency checkpoint: {path}")
        return path

    def close(self) -> None:
        if getattr(self, "writer", None) is not None:
            self.writer.flush()
            self.writer.close()

    def _resolve_device(self, d):
        if d == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(d)

    @staticmethod
    def _normalize_monitor_key(name: str) -> str:
        key = str(name or "").strip()
        return key[4:] if key.startswith("val_") else key

    def _to_device(self, batch):
        out = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(self.device)
            else:
                out[k] = v
        return out

    def _apply_feature_smoothing(
        self, sequence: torch.Tensor, padding_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if self.feature_smoothing_std <= 0.0:
            return sequence
        smoothed = sequence.clone()
        noise = torch.randn_like(smoothed) * float(self.feature_smoothing_std)
        if isinstance(padding_mask, torch.Tensor):
            noise = noise.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return smoothed + noise

    def _set_step_learning_rate(self, epoch: int, step_idx: int, n_steps: int) -> None:
        if self.lr_warmup_epochs <= 0 or epoch > self.lr_warmup_epochs:
            return
        total_steps = max(1, self.lr_warmup_epochs * max(1, n_steps))
        current_step = ((max(1, int(epoch)) - 1) * max(1, n_steps)) + int(step_idx) + 1
        progress = min(1.0, max(0.0, float(current_step) / float(total_steps)))
        factor = self.lr_warmup_start_factor + (
            (1.0 - self.lr_warmup_start_factor) * progress
        )
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            group["lr"] = float(base_lr) * float(factor)

    def _build_scheduler(
        self, sched_cfg: Mapping[str, Any]
    ) -> torch.optim.lr_scheduler._LRScheduler | torch.optim.lr_scheduler.CosineAnnealingWarmRestarts:
        scheduler_type = str(
            sched_cfg.get("type", "cosine_annealing_warm_restarts")
        ).strip().lower()
        eta_min = float(sched_cfg.get("min_lr", sched_cfg.get("eta_min", 1e-5)))
        if scheduler_type in {
            "cosine_annealing_warm_restarts",
            "warm_restarts",
            "cosine_warm_restarts",
        }:
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=max(1, int(sched_cfg.get("T_0", 10))),
                T_mult=max(1, int(sched_cfg.get("T_mult", 2))),
                eta_min=eta_min,
            )
        if scheduler_type in {"cosine_annealing", "cosine", "cosine_decay"}:
            t_max = max(
                1,
                int(
                    sched_cfg.get(
                        "T_max", sched_cfg.get("T_0", max(1, int(self.epochs)))
                    )
                ),
            )
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=t_max,
                eta_min=eta_min,
            )
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    def _clip_metadata_from_batch(self, batch: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
        metadata: Dict[str, torch.Tensor] = {}
        for key in (
            "duration_s",
            "speech_ratio",
            "snr_db_estimate",
            "capture_quality_score",
            "voiced_ratio",
            "clipped_ratio",
            "distance_cm_estimate",
            "distance_confidence",
            "quality_ok",
            "feature_drift_zscore",
            "ood_zscore",
        ):
            value = batch.get(key)
            if isinstance(value, torch.Tensor):
                metadata[key] = value
        valid_frames = batch.get("padding_mask")
        if isinstance(valid_frames, torch.Tensor):
            metadata["valid_frames"] = (~valid_frames).sum(dim=1).to(dtype=torch.float32)
        return metadata

    def _train_sampler(self):
        return getattr(self.train_loader, "batch_sampler", None)

    def _set_train_epoch(self, epoch: int) -> None:
        sampler = self._train_sampler()
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(int(epoch))

    def _sampler_state(self) -> Optional[Dict[str, Any]]:
        sampler = self._train_sampler()
        if sampler is not None and hasattr(sampler, "state_dict"):
            return sampler.state_dict()
        return None

    def _restore_sampler_state(self, state: Optional[Dict[str, Any]]) -> None:
        sampler = self._train_sampler()
        if state and sampler is not None and hasattr(sampler, "load_state_dict"):
            sampler.load_state_dict(state)

    def _scheduled_grl_lambda(self, epoch: int, step_idx: int, n_steps: int) -> float:
        # Standard domain-adversarial schedule:
        # lambda(p) = lambda_max * (2 / (1 + exp(-10p)) - 1), p in [0,1]
        if n_steps <= 0:
            return self.grl_lambda_max
        p_epoch = (epoch - 1) / max(1, self.epochs - 1)
        p_step = step_idx / max(1, n_steps - 1)
        p = 0.5 * p_epoch + 0.5 * p_step
        base = 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0
        return self.grl_lambda_min + (self.grl_lambda_max - self.grl_lambda_min) * base

    def _build_targets(
        self, batch, epoch: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        targets: Dict[str, torch.Tensor] = {}
        for key in (
            "height",
            "weight",
            "age",
            "shoulder",
            "waist",
            "height_raw",
            "weight_raw",
            "age_raw",
            "shoulder_raw",
            "waist_raw",
            "gender",
            "weight_mask",
            "shoulder_mask",
            "waist_mask",
            "f0_mean",
            "formant_spacing_mean",
            "vtl_mean",
            "duration_s",
            "speech_ratio",
            "snr_db_estimate",
            "capture_quality_score",
            "voiced_ratio",
            "clipped_ratio",
            "distance_cm_estimate",
            "distance_confidence",
            "quality_ok",
        ):
            value = batch.get(key)
            if isinstance(value, torch.Tensor):
                targets[key] = value
        if isinstance(batch.get("source_id"), torch.Tensor):
            targets["domain"] = batch["source_id"]
        speaker_ids = batch.get("speaker_id")
        if isinstance(speaker_ids, (list, tuple)) and len(speaker_ids) > 0:
            speaker_lookup: Dict[str, int] = {}
            speaker_idx = []
            for raw_speaker_id in speaker_ids:
                if raw_speaker_id is None:
                    speaker_idx.append(-1)
                    continue
                speaker_key = str(raw_speaker_id)
                if speaker_key not in speaker_lookup:
                    speaker_lookup[speaker_key] = len(speaker_lookup)
                speaker_idx.append(speaker_lookup[speaker_key])
            targets["speaker_idx"] = torch.tensor(
                speaker_idx, device=self.device, dtype=torch.long
            )
        elif isinstance(speaker_ids, torch.Tensor):
            targets["speaker_idx"] = speaker_ids.to(
                device=self.device, dtype=torch.long
            )
        if epoch is not None:
            targets["epoch"] = int(epoch)

        # Target label smoothing: add small noise to regression targets during training
        if epoch is not None:
            tls_cfg = self.config.get("training", {}).get("target_label_smoothing", {})
            if tls_cfg.get("enabled", False):
                tls_std = float(tls_cfg.get("std", 0.05))
                tls_decay = float(tls_cfg.get("decay", 0.95))
                # Decay noise over training: stronger early, weaker later
                current_std = tls_std * (tls_decay ** (epoch - 1))
                if current_std > 1e-4:
                    for key in ("height", "weight", "age"):
                        if key in targets:
                            noise = torch.randn_like(targets[key]) * current_std
                            targets[key] = targets[key] + noise

        return targets

    def _forward_model(
        self,
        batch,
        epoch: Optional[int] = None,
        step_idx: int = 0,
        n_steps: int = 1,
        train_mode: bool = True,
        targets: Optional[Dict[str, torch.Tensor]] = None,
    ):
        sequence = batch["sequence"]
        if train_mode and self.feature_smoothing_std > 0.0:
            sequence = self._apply_feature_smoothing(
                sequence, batch.get("padding_mask")
            )
        kwargs = {
            "padding_mask": batch["padding_mask"],
            "clip_metadata": self._clip_metadata_from_batch(batch),
        }
        if getattr(self.model, "expects_domain", False):
            kwargs["domain"] = batch.get("source_id")
            lam = self.grl_lambda_max
            if train_mode and epoch is not None:
                lam = self._scheduled_grl_lambda(
                    epoch=epoch, step_idx=step_idx, n_steps=n_steps
                )
            kwargs["lambda_grl"] = lam
        if self.use_native_v2_loss:
            kwargs["return_aux"] = False
            if targets is not None:
                kwargs["targets"] = targets
                if epoch is not None:
                    kwargs["current_epoch"] = int(epoch)
        return self.model(sequence, **kwargs)

    def _ensemble_predictions(self, batch) -> Optional[Dict[str, torch.Tensor]]:
        if not self.eval_use_ensemble or not hasattr(
            self.model, "predict_with_uncertainty"
        ):
            return None

        result = self.model.predict_with_uncertainty(
            batch["sequence"],
            padding_mask=batch.get("padding_mask"),
            domain=batch.get("source_id")
            if getattr(self.model, "expects_domain", False)
            else None,
            speaker_ids=batch.get("speaker_id"),
            clip_metadata=self._clip_metadata_from_batch(batch),
            deterministic=self.eval_deterministic,
            n_samples=self.eval_n_samples,
            crop_size=self.eval_crop_size,
            n_crops=self.eval_n_crops,
        )
        return {
            "height": result["height"]["mean"].detach(),
            "weight": result["weight"]["mean"].detach(),
            "age": result["age"]["mean"].detach(),
            "height_var": result["height"]["var"].detach(),
            "weight_var": result["weight"]["var"].detach(),
            "age_var": result["age"]["var"].detach(),
            "gender_pred": result["gender"]["pred"].detach(),
            "gender_probs": result["gender"]["probs"].detach(),
            "quality_score": result.get("utterance", {})
            .get("quality_score", torch.ones_like(result["height"]["mean"]))
            .detach(),
        }

    def _compute_losses(
        self,
        preds: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ):
        if self.use_native_v2_loss and isinstance(preds.get("losses"), dict):
            return preds["losses"]

        physics_inputs = {
            "vtl_estimated": batch.get("vtl_mean"),
            "formant_spacing": batch.get("formant_spacing_mean"),
            "f0_mean": batch.get("f0_mean"),
        }
        physics_module = getattr(self.model, "physics_loss", None)
        if self.loss_type == "vtsl_v2":
            return self.criterion(preds, targets)
        return self.criterion(preds, targets, physics_inputs, physics_module)

    def _metric_value(
        self,
        metrics: Dict[str, float],
        key: str,
        mode: str,
        fallback_key: str = "total",
    ) -> float:
        raw = metrics.get(key)
        if isinstance(raw, (int, float)) and math.isfinite(float(raw)):
            return float(raw)
        fallback = metrics.get(fallback_key)
        if isinstance(fallback, (int, float)) and math.isfinite(float(fallback)):
            return float(fallback)
        return float("inf") if mode == "min" else float("-inf")

    def _denorm_numpy(self, values: np.ndarray, key: str) -> np.ndarray:
        if self.target_stats is None:
            return values
        stats = self.target_stats.get(key, {})
        return values * float(stats.get("std", 1.0)) + float(stats.get("mean", 0.0))

    @staticmethod
    def _safe_corrcoef(x: np.ndarray, y: np.ndarray) -> float:
        if x.size < 2 or y.size < 2:
            return float("nan")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            return float("nan")
        if np.allclose(x, x[0]) or np.allclose(y, y[0]):
            return float("nan")
        return float(np.corrcoef(x, y)[0, 1])

    def _calibration_metrics(
        self,
        pred_means: Dict[str, List[torch.Tensor]],
        pred_vars: Dict[str, List[torch.Tensor]],
        targets: Dict[str, List[torch.Tensor]],
    ) -> Dict[str, float]:
        height_pred = self._denorm_numpy(
            torch.cat(pred_means["height"]).float().numpy(), "height"
        )
        height_true = torch.cat(targets["height_raw"]).float().numpy()
        height_var = torch.cat(pred_vars["height"]).float().numpy()
        height_std_scale = (
            float(self.target_stats.get("height", {}).get("std", 1.0))
            if self.target_stats
            else 1.0
        )
        height_std = (
            np.sqrt(np.clip(height_var, a_min=0.0, a_max=None)) * height_std_scale
        )

        valid = (
            np.isfinite(height_pred)
            & np.isfinite(height_true)
            & np.isfinite(height_std)
        )
        if not valid.any():
            return {}

        abs_error = np.abs(height_true[valid] - height_pred[valid])
        pred_std = np.clip(height_std[valid], a_min=1e-6, a_max=None)
        return {
            "height_calibration_mae": float(np.mean(np.abs(pred_std - abs_error))),
            "height_uncertainty_error_corr": self._safe_corrcoef(pred_std, abs_error),
            "height_interval_68": float(np.mean(abs_error <= pred_std)),
            "height_interval_95": float(np.mean(abs_error <= 1.96 * pred_std)),
            "height_pred_std_mean": float(np.mean(pred_std)),
        }

    def _speaker_level_metrics(
        self,
        speaker_ids: List[str],
        pred_means: Dict[str, List[torch.Tensor]],
        pred_vars: Dict[str, List[torch.Tensor]],
        gender_probs: List[torch.Tensor],
        quality_scores: List[torch.Tensor],
        targets: Dict[str, List[torch.Tensor]],
    ) -> Dict[str, float]:
        if not speaker_ids or not hasattr(self.model, "aggregate_by_speaker"):
            return {}

        preds = {
            "height": torch.cat(pred_means["height"]).float(),
            "weight": torch.cat(pred_means["weight"]).float(),
            "age": torch.cat(pred_means["age"]).float(),
            "gender_probs": torch.cat(gender_probs).float(),
        }
        variances = {
            "height": torch.cat(pred_vars["height"]).float(),
            "weight": torch.cat(pred_vars["weight"]).float(),
            "age": torch.cat(pred_vars["age"]).float(),
            "gender_probs": None,
        }
        quality = torch.cat(quality_scores).float()
        metadata = {
            key: torch.cat(targets[key]).float()
            for key in (
                "duration_s",
                "speech_ratio",
                "snr_db_estimate",
                "capture_quality_score",
                "voiced_ratio",
                "clipped_ratio",
                "distance_cm_estimate",
            )
            if targets.get(key)
        }
        metadata["valid_frames"] = torch.cat(
            targets["valid_frames"]
            if targets.get("valid_frames")
            else [torch.ones_like(torch.cat(pred_means["height"]).float())]
        ).float()

        aggregated_legacy = self.model.aggregate_by_speaker(
            speaker_ids=speaker_ids,
            preds=preds,
            variances=variances,
            quality=quality,
            metadata=metadata,
            method="legacy_inverse_variance",
        )
        aggregated_omega = self.model.aggregate_by_speaker(
            speaker_ids=speaker_ids,
            preds=preds,
            variances=variances,
            quality=quality,
            metadata=metadata,
            method="omega_robust_reliability_pool",
        )

        height_true = torch.cat(targets["height_raw"]).float().numpy()
        weight_true = torch.cat(targets["weight_raw"]).float().numpy()
        age_true = torch.cat(targets["age_raw"]).float().numpy()
        gender_true = torch.cat(targets["gender"]).long().numpy()
        source_id = (
            torch.cat(targets["source_id"]).long().numpy()
            if targets.get("source_id")
            else np.zeros_like(gender_true, dtype=np.int64)
        )
        duration_s = (
            torch.cat(targets["duration_s"]).float().numpy()
            if targets.get("duration_s")
            else np.full_like(height_true, np.nan, dtype=np.float32)
        )
        capture_quality = (
            torch.cat(targets["capture_quality_score"]).float().numpy()
            if targets.get("capture_quality_score")
            else np.full_like(height_true, np.nan, dtype=np.float32)
        )

        truth_by_speaker: Dict[str, Dict[str, float]] = {}
        for idx, speaker_id in enumerate(speaker_ids):
            entry = truth_by_speaker.setdefault(str(speaker_id), {})
            if "height" not in entry and np.isfinite(height_true[idx]):
                entry["height"] = float(height_true[idx])
            if "weight" not in entry and np.isfinite(weight_true[idx]):
                entry["weight"] = float(weight_true[idx])
            if "age" not in entry and np.isfinite(age_true[idx]):
                entry["age"] = float(age_true[idx])
            if "gender" not in entry:
                entry["gender"] = int(gender_true[idx])
            if "source_id" not in entry:
                entry["source_id"] = int(source_id[idx])
            entry.setdefault("duration_values", []).append(float(duration_s[idx]))
            entry.setdefault("quality_values", []).append(float(capture_quality[idx]))
        def summarize(aggregated: Dict[str, Any], *, suffix: str = "") -> Dict[str, float]:
            metrics_local: Dict[str, float] = {
                f"n_speakers_eval{suffix}": float(len(aggregated.get("speaker", {})))
            }
            speaker_preds_height: List[float] = []
            speaker_true_height: List[float] = []
            speaker_preds_weight: List[float] = []
            speaker_true_weight: List[float] = []
            speaker_preds_age: List[float] = []
            speaker_true_age: List[float] = []
            speaker_preds_gender: List[int] = []
            speaker_true_gender: List[int] = []
            speaker_sources: List[int] = []
            speaker_genders: List[int] = []
            speaker_durations: List[float] = []
            speaker_qualities: List[float] = []
            speaker_height_stds: List[float] = []

            for speaker_id, entry in aggregated.get("speaker", {}).items():
                truth = truth_by_speaker.get(str(speaker_id))
                if truth is None:
                    continue
                if "height" in truth:
                    speaker_true_height.append(truth["height"])
                    speaker_preds_height.append(
                        float(self._denorm_numpy(entry["height"].view(1).cpu().numpy(), "height")[0])
                    )
                    if "height_std" in entry:
                        height_std_scale = (
                            float(self.target_stats.get("height", {}).get("std", 1.0))
                            if self.target_stats
                            else 1.0
                        )
                        speaker_height_stds.append(
                            float(entry["height_std"].view(1).cpu().numpy()[0]) * height_std_scale
                        )
                if "weight" in truth:
                    speaker_true_weight.append(truth["weight"])
                    speaker_preds_weight.append(
                        float(self._denorm_numpy(entry["weight"].view(1).cpu().numpy(), "weight")[0])
                    )
                if "age" in truth:
                    speaker_true_age.append(truth["age"])
                    speaker_preds_age.append(
                        float(self._denorm_numpy(entry["age"].view(1).cpu().numpy(), "age")[0])
                    )
                if "gender" in truth:
                    speaker_true_gender.append(int(truth["gender"]))
                    speaker_preds_gender.append(int(entry["gender_pred"]))
                speaker_sources.append(int(truth.get("source_id", 0)))
                speaker_genders.append(int(truth.get("gender", 0)))
                duration_values = np.asarray(truth.get("duration_values", []), dtype=np.float32)
                quality_values = np.asarray(truth.get("quality_values", []), dtype=np.float32)
                duration_values = duration_values[np.isfinite(duration_values)]
                quality_values = quality_values[np.isfinite(quality_values)]
                speaker_durations.append(float(np.mean(duration_values)) if duration_values.size else float("nan"))
                speaker_qualities.append(float(np.mean(quality_values)) if quality_values.size else float("nan"))

            if speaker_true_height:
                height_true_arr = np.asarray(speaker_true_height, dtype=np.float32)
                height_pred_arr = np.asarray(speaker_preds_height, dtype=np.float32)
                height_abs_err = np.abs(height_true_arr - height_pred_arr)
                metrics_local[f"height_mae_speaker{suffix}"] = float(np.mean(height_abs_err))
                metrics_local[f"height_rmse_speaker{suffix}"] = float(
                    np.sqrt(np.mean((height_true_arr - height_pred_arr) ** 2))
                )
                metrics_local[f"height_median_ae_speaker{suffix}"] = float(np.median(height_abs_err))
                if speaker_height_stds:
                    metrics_local[f"height_pred_std_speaker_mean{suffix}"] = float(
                        np.mean(np.asarray(speaker_height_stds, dtype=np.float32))
                    )
                source_arr = np.asarray(speaker_sources, dtype=np.int64)
                gender_arr = np.asarray(speaker_genders, dtype=np.int64)
                duration_arr = np.asarray(speaker_durations, dtype=np.float32)
                quality_arr = np.asarray(speaker_qualities, dtype=np.float32)
                subgroup_masks = {
                    f"height_source_nisp_speaker_mae{suffix}": source_arr == 1,
                    f"height_source_timit_speaker_mae{suffix}": source_arr == 0,
                    f"height_gender_female_speaker_mae{suffix}": gender_arr == 0,
                    f"height_gender_male_speaker_mae{suffix}": gender_arr == 1,
                }
                for label, mask in subgroup_masks.items():
                    if np.any(mask):
                        metrics_local[label] = float(np.mean(np.abs(height_true_arr[mask] - height_pred_arr[mask])))
                height_bin_metrics: Dict[str, float] = {}
                for bin_label in ("short", "medium", "tall"):
                    mask = np.array([height_bin(float(value)) == bin_label for value in height_true_arr], dtype=bool)
                    if np.any(mask):
                        metric_value = float(
                            np.mean(np.abs(height_true_arr[mask] - height_pred_arr[mask]))
                        )
                        metrics_local[f"height_heightbin_{bin_label}_speaker_mae{suffix}"] = metric_value
                        height_bin_metrics[bin_label] = metric_value
                extreme_values = [
                    height_bin_metrics[label]
                    for label in ("short", "tall")
                    if label in height_bin_metrics
                ]
                if extreme_values:
                    metrics_local[f"height_heightbin_extreme_speaker_mae{suffix}"] = float(
                        np.mean(np.asarray(extreme_values, dtype=np.float32))
                    )
                    metrics_local[
                        f"height_heightbin_extreme_worst_speaker_mae{suffix}"
                    ] = float(np.max(np.asarray(extreme_values, dtype=np.float32)))
                for bin_label in ("short", "medium", "long"):
                    mask = np.array([duration_bin(float(value)) == bin_label for value in duration_arr], dtype=bool)
                    if np.any(mask):
                        metrics_local[f"height_duration_{bin_label}_speaker_mae{suffix}"] = float(
                            np.mean(np.abs(height_true_arr[mask] - height_pred_arr[mask]))
                        )
                for bucket in ("low", "medium", "high"):
                    mask = np.array([quality_bucket(float(value)) == bucket for value in quality_arr], dtype=bool)
                    if np.any(mask):
                        metrics_local[f"height_quality_{bucket}_speaker_mae{suffix}"] = float(
                            np.mean(np.abs(height_true_arr[mask] - height_pred_arr[mask]))
                        )
            if speaker_true_weight:
                weight_true_arr = np.asarray(speaker_true_weight, dtype=np.float32)
                weight_pred_arr = np.asarray(speaker_preds_weight, dtype=np.float32)
                weight_abs_err = np.abs(weight_true_arr - weight_pred_arr)
                metrics_local[f"weight_mae_speaker{suffix}"] = float(np.mean(weight_abs_err))
                metrics_local[f"weight_rmse_speaker{suffix}"] = float(
                    np.sqrt(np.mean((weight_true_arr - weight_pred_arr) ** 2))
                )
                metrics_local[f"weight_median_ae_speaker{suffix}"] = float(np.median(weight_abs_err))
            if speaker_true_age:
                age_true_arr = np.asarray(speaker_true_age, dtype=np.float32)
                age_pred_arr = np.asarray(speaker_preds_age, dtype=np.float32)
                age_abs_err = np.abs(age_true_arr - age_pred_arr)
                metrics_local[f"age_mae_speaker{suffix}"] = float(np.mean(age_abs_err))
                metrics_local[f"age_rmse_speaker{suffix}"] = float(
                    np.sqrt(np.mean((age_true_arr - age_pred_arr) ** 2))
                )
                metrics_local[f"age_median_ae_speaker{suffix}"] = float(np.median(age_abs_err))
            if speaker_true_gender:
                metrics_local[f"gender_acc_speaker{suffix}"] = float(
                    np.mean(
                        np.asarray(speaker_true_gender, dtype=np.int64)
                        == np.asarray(speaker_preds_gender, dtype=np.int64)
                    )
                )
            return metrics_local

        metrics = summarize(aggregated_legacy, suffix="")
        metrics.update(summarize(aggregated_omega, suffix="_omega"))
        return metrics

    def _train_epoch(self, epoch):
        self.model.train()
        self._set_train_epoch(epoch)
        total_losses: Dict[str, float] = {}
        n = 0
        epoch_start = time.time()

        # R-Drop config
        rdrop_enabled = bool(
            self.config.get("training", {}).get("rdrop", {}).get("enabled", False)
        )
        rdrop_weight = float(
            self.config.get("training", {}).get("rdrop", {}).get("weight", 0.5)
        )
        rdrop_keys = ("height", "weight", "age")

        n_steps = len(self.train_loader)
        self.optimizer.zero_grad(set_to_none=True)
        for step_idx, batch in enumerate(self.train_loader):
            self._set_step_learning_rate(epoch, step_idx, n_steps)
            batch = self._to_device(batch)
            targets = self._build_targets(batch, epoch=epoch)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                preds = self._forward_model(
                    batch=batch,
                    epoch=epoch,
                    step_idx=step_idx,
                    n_steps=n_steps,
                    train_mode=True,
                    targets=targets if self.use_native_v2_loss else None,
                )
                losses = self._compute_losses(preds, batch, targets)

                # R-Drop: second forward pass with different dropout mask
                if rdrop_enabled:
                    preds2 = self._forward_model(
                        batch=batch,
                        epoch=epoch,
                        step_idx=step_idx,
                        n_steps=n_steps,
                        train_mode=True,
                        targets=targets if self.use_native_v2_loss else None,
                    )
                    rdrop_loss = torch.tensor(0.0, device=self.device)
                    n_rdrop = 0
                    for key in rdrop_keys:
                        if key in preds and key in preds2:
                            rdrop_loss = rdrop_loss + torch.mean(
                                (preds[key] - preds2[key]) ** 2
                            )
                            n_rdrop += 1
                    if n_rdrop > 0:
                        rdrop_loss = rdrop_loss / n_rdrop
                        losses["total"] = losses["total"] + rdrop_weight * rdrop_loss
                        losses["rdrop"] = rdrop_loss.detach()

                total_loss = losses["total"] / float(self.gradient_accumulation_steps)

            self.scaler.scale(total_loss).backward()
            should_step = ((step_idx + 1) % self.gradient_accumulation_steps == 0) or (
                (step_idx + 1) == n_steps
            )
            if should_step:
                if self.grad_clip:
                    self.scaler.unscale_(self.optimizer)
                    if hasattr(self.model, "clip_gradients") and callable(
                        getattr(self.model, "clip_gradients")
                    ):
                        self.model.clip_gradients()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), float(self.grad_clip)
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.optimizer_step_count += 1
                self._update_ema()

            for k, v in losses.items():
                total_losses[k] = total_losses.get(k, 0.0) + float(v.item())
            n += 1

            current_step = step_idx + 1
            if (
                current_step % self.progress_log_interval_steps == 0
                or current_step == n_steps
            ):
                avg_total = total_losses.get("total", 0.0) / max(1, n)
                elapsed = time.time() - epoch_start
                avg_step_time = elapsed / max(1, current_step)
                eta_seconds = avg_step_time * max(0, n_steps - current_step)
                progress = 100.0 * float(current_step) / float(max(1, n_steps))
                print(
                    f"[Trainer] Epoch {epoch:03d} step {current_step:04d}/{n_steps:04d} "
                    f"({progress:5.1f}%) | loss={avg_total:.4f} "
                    f"| lr={self.optimizer.param_groups[0]['lr']:.6g} "
                    f"| elapsed={elapsed / 60.0:.1f}m | eta={eta_seconds / 60.0:.1f}m",
                    flush=True,
                )

        if n == 0:
            return {"total": float("inf")}

        self.scheduler.step()
        return {k: v / n for k, v in total_losses.items()}

    @torch.no_grad()
    def _val_epoch(self):
        return self._val_epoch_on(self.val_loader, split_name="val")

    @torch.no_grad()
    def _val_epoch_on(self, loader, split_name: str = "eval"):
        self.model.eval()
        weight_backup = self._swap_in_ema_weights()

        try:
            all_preds = {"height": [], "weight": [], "age": [], "gender_pred": []}
            all_pred_vars = {"height": [], "weight": [], "age": []}
            all_gender_probs: List[torch.Tensor] = []
            all_quality_scores: List[torch.Tensor] = []
            all_targets = {
                "height_raw": [],
                "weight_raw": [],
                "age_raw": [],
                "gender": [],
                "weight_mask": [],
                "source_id": [],
                "duration_s": [],
                "speech_ratio": [],
                "snr_db_estimate": [],
                "capture_quality_score": [],
                "voiced_ratio": [],
                "clipped_ratio": [],
                "distance_cm_estimate": [],
                "valid_frames": [],
            }
            speaker_ids: List[str] = []

            total_losses: Dict[str, float] = {}
            n = 0

            n_steps = len(loader)
            for step_idx, batch in enumerate(loader):
                batch = self._to_device(batch)
                targets = self._build_targets(batch, epoch=1)

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    preds = self._forward_model(
                        batch=batch,
                        epoch=1,
                        step_idx=step_idx,
                        n_steps=n_steps,
                        train_mode=False,
                        targets=targets if self.use_native_v2_loss else None,
                    )
                    losses = self._compute_losses(preds, batch, targets)

                ensemble = self._ensemble_predictions(batch)
                metric_height = preds["height"].detach()
                metric_weight = preds["weight"].detach()
                metric_age = preds["age"].detach()
                metric_height_var = preds.get(
                    "height_var", torch.ones_like(preds["height"])
                ).detach()
                metric_weight_var = preds.get(
                    "weight_var", torch.ones_like(preds["weight"])
                ).detach()
                metric_age_var = preds.get(
                    "age_var", torch.ones_like(preds["age"])
                ).detach()
                metric_gender_pred = preds["gender_logits"].argmax(-1).detach()
                metric_gender_probs = torch.softmax(
                    preds["gender_logits"], dim=-1
                ).detach()
                metric_quality = preds.get(
                    "quality_score", torch.ones_like(preds["height"])
                ).detach()

                if ensemble is not None:
                    metric_height = ensemble["height"]
                    metric_weight = ensemble["weight"]
                    metric_age = ensemble["age"]
                    metric_height_var = ensemble["height_var"]
                    metric_weight_var = ensemble["weight_var"]
                    metric_age_var = ensemble["age_var"]
                    metric_gender_pred = ensemble["gender_pred"]
                    metric_gender_probs = ensemble["gender_probs"]
                    metric_quality = ensemble["quality_score"]

                for k, v in losses.items():
                    total_losses[k] = total_losses.get(k, 0.0) + float(v.item())
                n += 1

                all_preds["height"].append(metric_height.cpu())
                all_preds["weight"].append(metric_weight.cpu())
                all_preds["age"].append(metric_age.cpu())
                all_pred_vars["height"].append(metric_height_var.cpu())
                all_pred_vars["weight"].append(metric_weight_var.cpu())
                all_pred_vars["age"].append(metric_age_var.cpu())
                all_preds["gender_pred"].append(metric_gender_pred.cpu())
                all_gender_probs.append(metric_gender_probs.cpu())
                all_quality_scores.append(metric_quality.cpu())

                all_targets["height_raw"].append(batch["height_raw"].detach().cpu())
                all_targets["weight_raw"].append(batch["weight_raw"].detach().cpu())
                all_targets["age_raw"].append(batch["age_raw"].detach().cpu())
                all_targets["gender"].append(batch["gender"].detach().cpu())
                all_targets["weight_mask"].append(
                    batch.get("weight_mask", torch.ones_like(batch["height"]))
                    .detach()
                    .cpu()
                )
                for key in ("source_id", "duration_s", "speech_ratio", "snr_db_estimate", "capture_quality_score"):
                    value = batch.get(key)
                    if isinstance(value, torch.Tensor):
                        all_targets[key].append(value.detach().cpu())
                for key in ("voiced_ratio", "clipped_ratio", "distance_cm_estimate"):
                    value = batch.get(key)
                    if isinstance(value, torch.Tensor):
                        all_targets[key].append(value.detach().cpu())
                padding_mask = batch.get("padding_mask")
                if isinstance(padding_mask, torch.Tensor):
                    all_targets["valid_frames"].append(
                        (~padding_mask).sum(dim=1).to(dtype=torch.float32).detach().cpu()
                    )
                speaker_ids.extend(batch.get("speaker_id", []))

            if n == 0:
                return {
                    "total": float("inf"),
                    "height_mae": float("nan"),
                    "weight_mae": float("nan"),
                    "age_mae": float("nan"),
                    "gender_acc": float("nan"),
                }

            avg_losses = {k: v / n for k, v in total_losses.items()}
            metrics = compute_metrics(all_preds, all_targets, self.target_stats)
            metrics.update(
                self._calibration_metrics(
                    pred_means={k: all_preds[k] for k in ("height", "weight", "age")},
                    pred_vars=all_pred_vars,
                    targets=all_targets,
                )
            )
            metrics.update(
                self._speaker_level_metrics(
                    speaker_ids=speaker_ids,
                    pred_means={k: all_preds[k] for k in ("height", "weight", "age")},
                    pred_vars=all_pred_vars,
                    gender_probs=all_gender_probs,
                    quality_scores=all_quality_scores,
                    targets=all_targets,
                )
            )
            avg_losses.update(metrics)
            return avg_losses
        finally:
            self._restore_weights(weight_backup)

    def train(self):
        print(f"\n{'=' * 60}")
        print(f"  VocalMorph Training - {self.epochs} epochs")
        print(f"{'=' * 60}\n", flush=True)

        for epoch in range(self.start_epoch, self.epochs + 1):
            t0 = time.time()
            train_losses = self._train_epoch(epoch)

            # Update SWA weights after each epoch
            if self.use_swa:
                self.swa.update(epoch)

            train_eval_metrics: Dict[str, float] = {}
            if self.train_eval_loader is not None:
                train_eval_metrics = self._val_epoch_on(
                    self.train_eval_loader, split_name="train_eval"
                )
                self.last_train_eval_metrics = dict(train_eval_metrics)

            val_metrics = self._val_epoch()
            elapsed = time.time() - t0

            for k, v in train_losses.items():
                self.writer.add_scalar(f"train/{k}", v, epoch)
            for k, v in train_eval_metrics.items():
                if isinstance(v, (int, float)):
                    self.writer.add_scalar(f"train_eval/{k}", v, epoch)
            for k, v in val_metrics.items():
                if isinstance(v, (int, float)):
                    self.writer.add_scalar(f"val/{k}", v, epoch)
            if train_eval_metrics:
                for key in (
                    "height_mae",
                    "height_mae_speaker",
                    "height_rmse_speaker",
                    "height_mae_speaker_omega",
                    "height_rmse_speaker_omega",
                ):
                    train_value = train_eval_metrics.get(key)
                    val_value = val_metrics.get(key)
                    if isinstance(train_value, (int, float)) and isinstance(val_value, (int, float)):
                        self.writer.add_scalar(
                            f"gap/{key}",
                            float(val_value) - float(train_value),
                            epoch,
                        )
            self.writer.add_scalar("lr", self.optimizer.param_groups[0]["lr"], epoch)
            self.writer.flush()

            val_loss = float(val_metrics.get("total", float("inf")))
            speaker_height = val_metrics.get("height_mae_speaker", float("nan"))
            speaker_height_omega = val_metrics.get("height_mae_speaker_omega", float("nan"))
            gap = float("nan")
            if train_eval_metrics:
                gap = float(val_metrics.get("height_mae_speaker", float("nan"))) - float(
                    train_eval_metrics.get("height_mae_speaker", float("nan"))
                )

            monitored = self._metric_value(val_metrics, self.es_monitor, self.es_mode)
            improved = (
                monitored < self.best_val_metric
                if self.es_mode == "min"
                else monitored > self.best_val_metric
            )
            if improved:
                self.best_val_metric = monitored
                self.es_counter = 0
            else:
                self.es_counter += 1

            self.last_completed_epoch = epoch
            epoch_record: Dict[str, Any] = {
                "epoch": int(epoch),
                "elapsed_seconds": float(elapsed),
                "lr": float(self.optimizer.param_groups[0]["lr"]),
                "train": {k: float(v) for k, v in train_losses.items()},
                "train_eval": {
                    k: float(v) for k, v in train_eval_metrics.items() if isinstance(v, (int, float))
                },
                "val": {k: float(v) for k, v in val_metrics.items() if isinstance(v, (int, float))},
                "monitor_name": self.es_monitor,
                "monitor_value": float(monitored),
                "best_metric_so_far": float(self.best_val_metric),
                "improved": bool(improved),
                "es_counter": int(self.es_counter),
            }
            if train_eval_metrics:
                epoch_record["gap_height_mae_speaker_val_minus_train"] = float(gap)
                omega_gap = float(val_metrics.get("height_mae_speaker_omega", float("nan"))) - float(
                    train_eval_metrics.get("height_mae_speaker_omega", float("nan"))
                )
                epoch_record["gap_height_mae_speaker_omega_val_minus_train"] = float(omega_gap)
            self.metrics_logger.append(epoch_record)

            print(
                f"Epoch [{epoch:3d}/{self.epochs}] "
                f"loss={train_losses.get('total', 0):.4f} "
                f"val={val_loss:.4f} | "
                f"h={val_metrics.get('height_mae', float('nan')):.2f}cm "
                f"h_spk={speaker_height:.2f}cm "
                f"h_spk_omega={speaker_height_omega:.2f}cm "
                f"w={val_metrics.get('weight_mae', float('nan')):.2f}kg "
                f"age={val_metrics.get('age_mae', float('nan')):.1f}yr "
                f"gender={val_metrics.get('gender_acc', float('nan')) * 100:.1f}% "
                f"[{elapsed:.1f}s]",
                flush=True,
            )
            if train_eval_metrics:
                print(
                    f"              train_eval_h_spk={train_eval_metrics.get('height_mae_speaker', float('nan')):.2f}cm "
                    f"| gap={gap:.2f}cm",
                    flush=True,
                )
            short_height_speaker = val_metrics.get(
                "height_heightbin_short_speaker_mae", float("nan")
            )
            tall_height_speaker = val_metrics.get(
                "height_heightbin_tall_speaker_mae", float("nan")
            )
            extreme_height_speaker = val_metrics.get(
                "height_heightbin_extreme_speaker_mae", float("nan")
            )
            medium_quality_speaker = val_metrics.get(
                "height_quality_medium_speaker_mae", float("nan")
            )
            if (
                math.isfinite(float(short_height_speaker))
                or math.isfinite(float(tall_height_speaker))
                or math.isfinite(float(extreme_height_speaker))
                or math.isfinite(float(medium_quality_speaker))
            ):
                print(
                    "              "
                    f"val_short_h_spk={float(short_height_speaker):.2f}cm "
                    f"| val_tall_h_spk={float(tall_height_speaker):.2f}cm "
                    f"| val_edge_h_spk={float(extreme_height_speaker):.2f}cm "
                    f"| val_qmed_h_spk={float(medium_quality_speaker):.2f}cm",
                    flush=True,
                )

            ckpt_metric = self._metric_value(
                val_metrics, self.ckpt_monitor, self.ckpt_mode
            )
            checkpoint_state = self._build_checkpoint_state(
                epoch=epoch,
                metric_val=float(ckpt_metric),
                train_losses=train_losses,
                train_eval_metrics=train_eval_metrics,
                val_metrics=val_metrics,
            )
            self.ckpt_manager.save(checkpoint_state, float(ckpt_metric), is_best=improved)

            if (not improved) and self.early_stopping_enabled and self.es_counter >= self.patience:
                print(f"\n[Early Stop] No improvement for {self.patience} epochs.")
                break

        # Final SWA evaluation if enabled
        if self.use_swa and self.swa.enabled and self.swa_use_for_eval:
            swa_backup = self.swa.apply()
            if swa_backup:
                swa_val = self._val_epoch()
                swa_monitor = self._metric_value(swa_val, self.es_monitor, self.es_mode)
                print(
                    f"\n[SWA] Final eval with averaged weights ({self.swa.n_averaged} epochs):"
                )
                print(
                    f"  {self.es_monitor}: {swa_monitor:.4f} (best single: {self.best_val_metric:.4f})"
                )
                if (self.es_mode == "min" and swa_monitor < self.best_val_metric) or (
                    self.es_mode == "max" and swa_monitor > self.best_val_metric
                ):
                    print(
                        "  [SWA] Outperforms best single checkpoint - keeping SWA weights"
                    )
                    self.best_val_metric = swa_monitor
                else:
                    print(
                        "  [SWA] Single checkpoint was better - restoring best weights"
                    )
                    self.swa.restore(swa_backup)

        print(
            f"\nTraining complete. Best {self.es_monitor}: {self.best_val_metric:.4f}"
        )


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
