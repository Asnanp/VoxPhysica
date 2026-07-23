#!/usr/bin/env python
"""
VocalMorph V4 Training Script
==============================
Clean, stable training for 3cm MAE height estimation.
Usage: python scripts/train_v4.py --config configs/v4_ssl.yaml
"""
import argparse, json, math, os, random, sys, time, traceback
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, yaml
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.models.vocalmorph_v4 import VocalMorphV4, V4Loss, build_v4_model, build_v4_loss
from src.preprocessing.dataset import (
    FeatureAugmentConfig, VocalMorphDataset, build_dataloaders_from_dirs, collate_fn,
)

def seed_everything(seed: int, deterministic: bool = True):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)

def configure_cuda(allow_tf32=True):
    if not torch.cuda.is_available(): return
    if hasattr(torch.backends.cuda, 'matmul'): torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    if hasattr(torch, 'set_float32_matmul_precision'): torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")

class TeeStream:
    def __init__(self, primary, secondary):
        self.primary, self.secondary, self.broken = primary, secondary, False
    def write(self, data):
        if not self.broken:
            try: self.primary.write(data)
            except OSError: self.broken = True
        self.secondary.write(data); return len(data)
    def flush(self):
        if not self.broken:
            try: self.primary.flush()
            except OSError: self.broken = True
        self.secondary.flush()
    def isatty(self): return getattr(self.primary, 'isatty', lambda: False)()
    @property
    def encoding(self): return getattr(self.primary, 'encoding', 'utf-8')

class EMAWeights:
    def __init__(self, model, decay=0.999, warmup_steps=200):
        self.model, self.decay, self.warmup_steps, self.step_count = model, decay, warmup_steps, 0
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    def update(self):
        self.step_count += 1
        d = self.decay if self.step_count > self.warmup_steps else 0.0
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if n in self.shadow: self.shadow[n].mul_(d).add_(p.detach(), alpha=1-d)
    def swap_in(self):
        backup = {}
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if n in self.shadow: backup[n] = p.detach().clone(); p.copy_(self.shadow[n])
        return backup
    def restore(self, backup):
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if n in backup: p.copy_(backup[n])
    def state_dict(self): return {"step_count": self.step_count, "shadow": {k: v.cpu().clone() for k, v in self.shadow.items()}}
    def load_state_dict(self, s):
        self.step_count = int(s.get("step_count", 0))
        for k, v in s.get("shadow", {}).items():
            if k in self.shadow: self.shadow[k] = v.to(self.shadow[k].device)

def compute_height_metrics(all_preds, all_targets, all_speaker_ids, target_stats):
    preds = torch.cat(all_preds, 0).cpu(); targets = torch.cat(all_targets, 0).cpu()
    speaker_ids = [sid for batch in all_speaker_ids for sid in batch]
    if target_stats and "height" in target_stats:
        m, s = float(target_stats["height"]["mean"]), float(target_stats["height"]["std"])
        preds_cm, targets_cm = preds * s + m, targets * s + m
    else: preds_cm, targets_cm = preds, targets
    clip_mae = (preds_cm - targets_cm).abs().mean().item()
    sp, st = defaultdict(list), defaultdict(list)
    for i, sid in enumerate(speaker_ids): sp[sid].append(preds_cm[i].item()); st[sid].append(targets_cm[i].item())
    maes, short_m, medium_m, tall_m = [], [], [], []
    for sid in sp:
        pm, tm = np.mean(sp[sid]), np.mean(st[sid]); mae = abs(pm - tm); maes.append(mae)
        if tm < 160: short_m.append(mae)
        elif tm < 175: medium_m.append(mae)
        elif tm >= 175: tall_m.append(mae)
    overall = float(np.mean(maes)) if maes else 0.0
    short_mae = float(np.mean(short_m)) if short_m else float("nan")
    medium_mae = float(np.mean(medium_m)) if medium_m else float("nan")
    tall_mae = float(np.mean(tall_m)) if tall_m else float("nan")
    balanced_values = [overall] + [
        value for value in (short_mae, medium_mae, tall_mae) if np.isfinite(value)
    ]
    balanced = float(np.mean(balanced_values)) if balanced_values else overall
    guarded = overall
    if np.isfinite(short_mae):
        guarded += 0.35 * max(0.0, short_mae - overall)
    if np.isfinite(tall_mae):
        guarded += 0.10 * max(0.0, tall_mae - overall)
    return {
        "height_mae_clip": clip_mae,
        "height_mae_speaker": overall,
        "height_rmse_speaker": float(np.sqrt(np.mean(np.array(maes)**2))) if maes else 0.0,
        "height_median_ae_speaker": float(np.median(maes)) if maes else 0.0,
        "height_mae_short_speaker": short_mae,
        "height_mae_medium_speaker": medium_mae,
        "height_mae_tall_speaker": tall_mae,
        "height_mae_speaker_balanced": balanced,
        "height_mae_speaker_guarded": guarded,
        "n_speakers": len(sp), "n_short": len(short_m), "n_medium": len(medium_m), "n_tall": len(tall_m),
    }

class V4Trainer:
    def __init__(self, model, criterion, train_loader, val_loader, test_loader, config, target_stats):
        self.model, self.criterion = model, criterion
        self.train_loader, self.val_loader, self.test_loader = train_loader, val_loader, test_loader
        self.config, self.target_stats = config, target_stats
        tc = config.get("training", {})
        self.epochs = int(tc.get("epochs", 100))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if tc.get("device","auto")=="auto" else torch.device(tc["device"])
        self.model = self.model.to(self.device)
        self.use_amp = bool(tc.get("amp", True)) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        oc = tc.get("optimizer", {})
        self.base_lr = float(oc.get("lr", 5e-4))
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.base_lr,
            weight_decay=float(oc.get("weight_decay", 0.01)), betas=tuple(oc.get("betas", [0.9,0.999])))
        sc = tc.get("scheduler", {})
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=int(sc.get("T_max", self.epochs)), eta_min=float(sc.get("eta_min", 1e-6)))
        gc = tc.get("gradient_clipping", {})
        self.grad_clip = float(gc.get("max_norm", 1.0)) if gc.get("enabled", True) else None
        self.grad_accum = int(tc.get("gradient_accumulation_steps", 1))
        self.warmup_epochs = int(tc.get("lr_warmup_epochs", 5))
        self.warmup_start = float(tc.get("lr_warmup_start_factor", 0.01))
        ec = tc.get("ema", {})
        self.use_ema = bool(ec.get("enabled", True))
        if self.use_ema: self.ema = EMAWeights(self.model, float(ec.get("decay", 0.9995)), int(ec.get("warmup_steps", 300)))
        self.patience = int(tc.get("early_stopping", {}).get("patience", 25))
        self.monitor = str(tc.get("early_stopping", {}).get("monitor", "height_mae_speaker_balanced"))
        self.best_val, self.es_counter = float("inf"), 0
        lc = config.get("logging", {}); cc = lc.get("checkpoint", {})
        self.ckpt_dir = os.path.normpath(os.path.join(ROOT, cc.get("dir", "outputs/v4/checkpoints/"))); os.makedirs(self.ckpt_dir, exist_ok=True)
        self.run_dir = os.path.dirname(self.ckpt_dir); os.makedirs(self.run_dir, exist_ok=True)
        self.metrics_path = os.path.join(self.run_dir, "metrics.jsonl")
        tb_dir = os.path.join(ROOT, lc.get("tensorboard", {}).get("log_dir", "outputs/v4/logs/"))
        try:
            from torch.utils.tensorboard import SummaryWriter; self.writer = SummaryWriter(tb_dir)
        except: self.writer = None
        self.feat_smooth = float(tc.get("feature_smoothing_std", 0.0))
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\n{'='*60}\n[V4] TRAINING — Target: 3cm MAE\n{'='*60}")
        print(f"  Device: {self.device} | AMP: {self.use_amp} | Params: {n_params:,}")
        print(f"  Epochs: {self.epochs} | Batch: {tc.get('batch_size',64)} x {self.grad_accum}")
        print(f"  LR: {self.base_lr} | WD: {oc.get('weight_decay',0.01)} | Warmup: {self.warmup_epochs}")
        print(f"  EMA: {self.use_ema} | Monitor: {self.monitor} | Patience: {self.patience}\n{'='*60}\n")

    def _metadata_features(self, batch):
        source_id = batch.get("source_id")
        if source_id is None:
            source_oh = torch.zeros((batch["sequence"].shape[0], 3), device=self.device)
        else:
            source_oh = F.one_hot(source_id.clamp(min=0, max=2).long(), num_classes=3).float()

        def clean(key, scale=1.0, clamp_min=None, clamp_max=None, transform=None):
            value = batch.get(key)
            if value is None:
                value = torch.zeros(batch["sequence"].shape[0], device=self.device)
            value = torch.nan_to_num(value.float(), nan=0.0, posinf=0.0, neginf=0.0)
            if clamp_min is not None or clamp_max is not None:
                value = value.clamp(
                    min=-float("inf") if clamp_min is None else clamp_min,
                    max=float("inf") if clamp_max is None else clamp_max,
                )
            if transform == "log1p":
                value = torch.log1p(value.clamp(min=0.0))
            return (value / scale).unsqueeze(1)

        scalar = torch.cat(
            [
                clean("f0_mean", 250.0, 0.0, 500.0),
                clean("formant_spacing_mean", 1200.0, 300.0, 2000.0),
                clean("vtl_mean", 20.0, 5.0, 40.0),
                clean("duration_s", 2.5, 0.0, 20.0, transform="log1p"),
                clean("voiced_ratio", 1.0, 0.0, 1.0),
                clean("speech_ratio", 1.0, 0.0, 1.0),
                clean("snr_db_estimate", 40.0, -10.0, 80.0),
                clean("capture_quality_score", 1.0, 0.0, 1.0),
                clean("clipped_ratio", 0.10, 0.0, 0.50),
                clean("distance_cm_estimate", 100.0, 0.0, 300.0),
            ],
            dim=1,
        )
        return torch.cat([scalar, source_oh], dim=1)

    def _warmup_lr(self, epoch, step, total):
        if epoch > self.warmup_epochs: return
        progress = min(1.0, ((epoch-1)*total + step) / max(1, self.warmup_epochs*total))
        factor = self.warmup_start + (1 - self.warmup_start) * progress
        for g in self.optimizer.param_groups: g["lr"] = self.base_lr * factor

    def _to_dev(self, batch):
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    def train(self):
        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            train_loss = self._train_epoch(epoch)
            ema_bk = self.ema.swap_in() if self.use_ema else None
            val_m = self._eval_epoch(self.val_loader)
            if self.use_ema and ema_bk: self.ema.restore(ema_bk)
            if epoch > self.warmup_epochs: self.scheduler.step()
            cur = val_m.get(self.monitor, val_m.get("height_mae_speaker", float("inf")))
            is_best = cur < self.best_val
            if is_best: self.best_val = cur; self.es_counter = 0; self._save(epoch, cur, True); tag=" ** BEST **"
            else: self.es_counter += 1; self._save(epoch, cur, False); tag=""
            lr = self.optimizer.param_groups[0]["lr"]
            print(f"[E{epoch:3d}/{self.epochs}] loss={train_loss['total']:.4f} | "
                  f"spk={val_m.get('height_mae_speaker',0):.3f}cm bal={val_m.get('height_mae_speaker_balanced',0):.3f} "
                  f"short={val_m.get('height_mae_short_speaker',0):.3f} med={val_m.get('height_mae_medium_speaker',0):.3f} "
                  f"tall={val_m.get('height_mae_tall_speaker',0):.3f} | "
                  f"lr={lr:.2e} es={self.es_counter}/{self.patience} {time.time()-t0:.0f}s{tag}")
            self._log(epoch, train_loss, val_m, lr)
            if self.es_counter >= self.patience: print(f"\n[V4] Early stopping at epoch {epoch}"); break

        # Final eval
        print(f"\n{'='*60}\n[V4] FINAL EVALUATION\n{'='*60}")
        best_path = os.path.join(self.ckpt_dir, "best.ckpt")
        if os.path.exists(best_path):
            ckpt = torch.load(best_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model_state_dict"])
            if self.use_ema and "ema_state" in ckpt:
                self.ema.load_state_dict(ckpt["ema_state"]); self.ema.swap_in()
            print(f"[V4] Loaded best from epoch {ckpt.get('epoch','?')}")
        val_f = self._eval_epoch(self.val_loader); test_f = self._eval_epoch(self.test_loader)
        print(f"\n  VAL  MAE: {val_f['height_mae_speaker']:.3f}cm | short: {val_f.get('height_mae_short_speaker',0):.3f} | tall: {val_f.get('height_mae_tall_speaker',0):.3f}")
        print(f"  TEST MAE: {test_f['height_mae_speaker']:.3f}cm | short: {test_f.get('height_mae_short_speaker',0):.3f} | tall: {test_f.get('height_mae_tall_speaker',0):.3f}")
        print(f"  TEST RMSE: {test_f.get('height_rmse_speaker',0):.3f}cm | median: {test_f.get('height_median_ae_speaker',0):.3f}cm")
        with open(os.path.join(self.run_dir, "final_metrics.json"), "w") as f:
            json.dump({"val": val_f, "test": test_f, "best_val": self.best_val}, f, indent=2)

    def _train_epoch(self, epoch):
        train_batch_sampler = getattr(self.train_loader, "batch_sampler", None)
        if hasattr(train_batch_sampler, "set_epoch"):
            train_batch_sampler.set_epoch(epoch)

        self.model.train(); losses = defaultdict(float); n = 0; total = len(self.train_loader)
        self.optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(self.train_loader):
            self._warmup_lr(epoch, step, total); batch = self._to_dev(batch)
            feats = batch["sequence"]; mask = batch.get("padding_mask")
            if self.feat_smooth > 0:
                noise = torch.randn_like(feats) * self.feat_smooth
                if mask is not None: noise = noise.masked_fill(mask.unsqueeze(-1), 0)
                feats = feats + noise
            metadata = self._metadata_features(batch) if getattr(self.model, "meta_dim", 0) > 0 else None
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                preds = self.model(feats, padding_mask=mask, metadata=metadata)
                targets = {
                    "height": batch["height"],
                    "height_raw": batch["height_raw"],
                    "gender": batch["gender"],
                    "source_id": batch["source_id"],
                }
                loss_dict = self.criterion(preds, targets)
            loss = loss_dict["total"] / self.grad_accum
            self.scaler.scale(loss).backward()
            if (step+1) % self.grad_accum == 0 or (step+1) == total:
                if self.grad_clip: self.scaler.unscale_(self.optimizer); nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer); self.scaler.update(); self.optimizer.zero_grad(set_to_none=True)
                if self.use_ema: self.ema.update()
            for k, v in loss_dict.items(): losses[k] += (v.item() if isinstance(v, torch.Tensor) else float(v))
            n += 1
        return {k: v/max(n,1) for k, v in losses.items()}

    @torch.no_grad()
    def _eval_epoch(self, loader):
        self.model.eval(); ap, at, asi = [], [], []
        for batch in loader:
            batch = self._to_dev(batch)
            metadata = self._metadata_features(batch) if getattr(self.model, "meta_dim", 0) > 0 else None
            preds = self.model(batch["sequence"], padding_mask=batch.get("padding_mask"), metadata=metadata)
            ap.append(preds["height"].cpu()); at.append(batch["height"].cpu()); asi.append(batch["speaker_id"])
        return compute_height_metrics(ap, at, asi, self.target_stats)

    def _save(self, epoch, metric, is_best):
        state = {"epoch": epoch, "model_state_dict": self.model.state_dict(),
                 "optimizer_state_dict": self.optimizer.state_dict(),
                 "scheduler_state_dict": self.scheduler.state_dict(),
                 "best_metric": self.best_val, "metric_val": metric}
        if self.use_ema: state["ema_state"] = self.ema.state_dict()
        if self.use_amp: state["scaler_state_dict"] = self.scaler.state_dict()
        torch.save(state, os.path.join(self.ckpt_dir, "last.ckpt"))
        if is_best: torch.save(state, os.path.join(self.ckpt_dir, "best.ckpt"))

    def _log(self, epoch, tl, vm, lr):
        rec = {"epoch": epoch, "lr": lr, "train": dict(tl), "val": dict(vm)}
        with open(self.metrics_path, "a") as f: f.write(json.dumps(rec, allow_nan=True)+"\n")
        if self.writer:
            for k, v in tl.items(): self.writer.add_scalar(f"train/{k}", v, epoch)
            for k, v in vm.items(): self.writer.add_scalar(f"val/{k}", v, epoch)
            self.writer.add_scalar("lr", lr, epoch); self.writer.flush()

    def close(self):
        if self.writer: self.writer.flush(); self.writer.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v4_ssl.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg_path = args.config if os.path.isabs(args.config) else os.path.join(ROOT, args.config)
    with open(cfg_path) as f: config = yaml.safe_load(f)
    tc = config["training"]
    if args.seed: tc["seed"] = args.seed
    if args.epochs: tc["epochs"] = args.epochs
    if args.device: tc["device"] = args.device
    seed = int(tc.get("seed", 42))

    # Logging
    lc = config.get("logging", {}); cc = lc.get("checkpoint", {})
    ckpt_dir_for_logs = os.path.normpath(os.path.join(ROOT, cc.get("dir", "outputs/v4/checkpoints/")))
    run_dir = os.path.dirname(ckpt_dir_for_logs)
    os.makedirs(run_dir, exist_ok=True)
    original_stdout, original_stderr = sys.stdout, sys.stderr
    stdout_h = open(os.path.join(run_dir, "train.stdout.log"), "a", encoding="utf-8", buffering=1)
    stderr_h = open(os.path.join(run_dir, "train.stderr.log"), "a", encoding="utf-8", buffering=1)
    sys.stdout = TeeStream(sys.stdout, stdout_h); sys.stderr = TeeStream(sys.stderr, stderr_h)

    trainer = None
    try:
        seed_everything(seed, bool(tc.get("deterministic", True)))
        configure_cuda(bool(tc.get("allow_tf32", True)))
        print(f"[V4] Seed: {seed} | Config: {cfg_path}")

        # Data
        dc = config.get("data", {})
        feat_dir = os.path.join(ROOT, dc.get("features_dir", "data/features_v4"))

        # Auto-detect if SSL or handcrafted
        ssl_info_path = os.path.join(feat_dir, "ssl_info.json")
        if os.path.exists(ssl_info_path):
            with open(ssl_info_path) as f: ssl_info = json.load(f)
            input_dim = int(ssl_info.get("input_dim", 768))
            print(f"[V4] SSL info: mode={ssl_info.get('mode')} input_dim={input_dim}")
        else:
            input_dim = None  # will auto-detect

        # Target stats
        stats_path = os.path.join(feat_dir, "target_stats.json"); target_stats = None
        if os.path.exists(stats_path):
            with open(stats_path) as f: target_stats = json.load(f)
            print(f"[V4] Height stats: mean={target_stats['height']['mean']:.1f}, std={target_stats['height']['std']:.1f}")

        # Augmentation
        ac = tc.get("augmentation", {}); do_aug = bool(ac.get("enabled", True))
        aug_cfg = FeatureAugmentConfig(
            noise_p=float(ac.get("noise_p", 0.4)), noise_std=float(ac.get("noise_std", 0.012)),
            time_mask_p=float(ac.get("time_mask_p", 0.3)), time_mask_max_frac=float(ac.get("time_mask_max_frac", 0.1)),
            feat_mask_p=float(ac.get("feat_mask_p", 0.25)), feat_mask_max_frac=float(ac.get("feat_mask_max_frac", 0.08)),
            scale_p=float(ac.get("scale_p", 0.35)), scale_std=float(ac.get("scale_std", 0.04)),
            temporal_jitter_p=float(ac.get("temporal_jitter_p", 0.2)),
            temporal_jitter_max_frac=float(ac.get("temporal_jitter_max_frac", 0.05)),
        ) if do_aug else None

        nw = int(tc.get("num_workers", 4)); pf = tc.get("prefetch_factor", 2)
        if nw == 0: pf = None; pw = False
        else: pf = int(pf) if pf else 2; pw = bool(tc.get("persistent_workers", True))

        speaker_batching_cfg = dict(tc.get("speaker_batching", {}) or {})
        sample_weighting_cfg = dict(tc.get("sample_weighting", {}) or {})
        print(f"[V4] Speaker batching: {speaker_batching_cfg}")
        if sample_weighting_cfg:
            print(f"[V4] Sample weighting: {sample_weighting_cfg}")

        train_loader, val_loader, test_loader = build_dataloaders_from_dirs(
            train_dir=os.path.join(feat_dir, "train"), val_dir=os.path.join(feat_dir, "val"),
            test_dir=os.path.join(feat_dir, "test"), batch_size=int(tc.get("batch_size", 64)),
            num_workers=nw, target_stats=target_stats,
            max_len=tc.get("max_feature_frames", 640),
            train_crop_mode=str(tc.get("train_crop_mode", "random")),
            eval_crop_mode=str(tc.get("eval_crop_mode", "center")),
            persistent_workers=pw, prefetch_factor=pf,
            train_augment=do_aug, augment_config=aug_cfg,
            speaker_batching=speaker_batching_cfg,
            sample_weighting=sample_weighting_cfg,
            base_seed=seed,
            pin_memory=bool(tc.get("pin_memory", True)),
        )

        if input_dim is None: input_dim = train_loader.dataset.infer_input_dim()
        config.setdefault("model", {})["input_dim"] = int(input_dim)
        print(f"[V4] Input dim: {input_dim} | Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")

        model = build_v4_model(config); criterion = build_v4_loss(config)
        trainer = V4Trainer(model, criterion, train_loader, val_loader, test_loader, config, target_stats)
        trainer.train()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        if trainer: trainer.close()
        try:
            sys.stdout.flush(); sys.stderr.flush()
        finally:
            sys.stdout, sys.stderr = original_stdout, original_stderr
        stdout_h.close(); stderr_h.close()

if __name__ == "__main__":
    main()
