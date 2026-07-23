#!/usr/bin/env python
"""Phase 1 CUDA-only speaker height trainer.

This is the foundation pass for the 3cm MAE push. It deliberately refuses CPU
training, builds speaker-level targets, pools clips with reliability weights,
uses robust feature scaling, and trains a stronger residual/gated model with
tail-aware losses. Disk reads still happen on the host because Windows has to
read .npz files, but all vector aggregation and optimization use torch CUDA.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
CACHE_VERSION = 2

SCALAR_KEYS: Tuple[str, ...] = (
    "f0_mean",
    "formant_spacing_mean",
    "vtl_mean",
    "jitter",
    "shimmer",
    "hnr",
    "duration_s",
    "voiced_ratio",
    "invalid_spacing_rate",
    "invalid_vtl_rate",
    "speech_ratio",
    "snr_db_estimate",
    "capture_quality_score",
    "distance_cm_estimate",
    "distance_confidence",
    "clipped_ratio",
)


@dataclass
class ClipRecord:
    speaker_id: str
    height_cm: float
    gender: int
    source: str
    vector: torch.Tensor
    quality: float
    augmented: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Phase 1 CUDA-only speaker height model.")
    parser.add_argument("--features-root", default="data/features_v4_combo_full_ssl")
    parser.add_argument("--output-dir", default="outputs/speaker_gpu_phase1_fullpower")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--original-only", action="store_true", help="Ignore static augmented train files.")
    parser.add_argument("--max-clips-per-speaker", type=int, default=0)
    parser.add_argument("--max-speaker-height-range-cm", type=float, default=2.0)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--patience", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--lr", type=float, default=6.0e-4)
    parser.add_argument("--weight-decay", type=float, default=4.0e-3)
    parser.add_argument("--dropout", type=float, default=0.22)
    parser.add_argument("--hidden", type=int, default=768)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--seeds", default="", help="Comma-separated seed list. Default uses seed, seed+13, seed+29, seed+43.")
    parser.add_argument("--target-mae-cm", type=float, default=3.0)
    parser.add_argument("--short-weight", type=float, default=2.60)
    parser.add_argument("--tall-weight", type=float, default=1.45)
    parser.add_argument("--rank-weight", type=float, default=0.08)
    parser.add_argument("--bin-weight", type=float, default=0.16)
    parser.add_argument("--nll-weight", type=float, default=0.12)
    parser.add_argument("--val-short-penalty", type=float, default=0.55)
    parser.add_argument("--calibration-shrinkage", type=float, default=12.0)
    parser.add_argument("--cache-name", default="speaker_phase1_cache.pt")
    return parser.parse_args()


def resolve(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def parse_seeds(args: argparse.Namespace) -> List[int]:
    if str(args.seeds).strip():
        out = [int(part.strip()) for part in str(args.seeds).split(",") if part.strip()]
        return out or [int(args.seed)]
    base = int(args.seed)
    return [base, base + 13, base + 29, base + 43]


def decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return ""
        if value.shape == ():
            return str(value.item())
        return decode(value.reshape(-1)[0])
    return str(value)


def safe_float(data: Mapping[str, Any], key: str, default: float = math.nan) -> float:
    if key not in data:
        return float(default)
    try:
        value = data[key]
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return float(default)
            out = float(value.item() if value.shape == () else value.reshape(-1)[0])
        else:
            out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def source_id(source: str, speaker_id: str) -> int:
    src = str(source or "").upper()
    sid = str(speaker_id or "").upper()
    if src == "TIMIT" or sid.startswith("TIMIT_"):
        return 0
    if src == "NISP" or sid.startswith("NISP_"):
        return 1
    if src == "CELEB" or src == "VOXCELEB" or sid.startswith("CELEB_") or sid.startswith("VOX"):
        return 2
    return 3


def gender_id(data: Mapping[str, Any]) -> int:
    raw = data["gender"] if "gender" in data else 0
    if isinstance(raw, np.ndarray) and raw.shape == ():
        raw = raw.item()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="ignore")
    if isinstance(raw, str):
        text = raw.strip().lower()
        if text in {"m", "male", "1"}:
            return 1
        if text in {"f", "female", "0"}:
            return 0
    try:
        return int(raw)
    except Exception:
        return 0


def height_bin_id(height_cm: float) -> int:
    if height_cm < 160.0:
        return 0
    if height_cm < 175.0:
        return 1
    return 2


def is_augmented_path(path: Path) -> bool:
    stem = path.stem.lower()
    return "_aug" in stem or stem.endswith("-aug")


def clean_tensor(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)


def quantiles(x: torch.Tensor, qs: Sequence[float], dim: int = 0) -> torch.Tensor:
    return torch.quantile(clean_tensor(x), torch.tensor(qs, dtype=torch.float32, device=x.device), dim=dim)


def clip_quality(data: Mapping[str, Any], augmented: bool) -> float:
    base = safe_float(data, "capture_quality_score", 0.75)
    speech = safe_float(data, "speech_ratio", 0.75)
    voiced = safe_float(data, "voiced_ratio", 0.65)
    snr = safe_float(data, "snr_db_estimate", 18.0)
    clipped = safe_float(data, "clipped_ratio", 0.0)
    invalid_spacing = safe_float(data, "invalid_spacing_rate", 0.0)
    invalid_vtl = safe_float(data, "invalid_vtl_rate", 0.0)

    snr_score = max(0.0, min(1.0, (snr - 4.0) / 24.0))
    q = 0.30 * base + 0.20 * speech + 0.16 * voiced + 0.16 * snr_score
    q += 0.18 * (1.0 - min(1.0, 0.55 * clipped + 0.35 * invalid_spacing + 0.35 * invalid_vtl))
    if augmented:
        q *= 0.58
    return float(max(0.05, min(1.0, q)))


@torch.no_grad()
def sequence_stats(seq: torch.Tensor) -> torch.Tensor:
    seq = clean_tensor(seq)
    mean = seq.mean(dim=0)
    std = seq.std(dim=0, unbiased=False).clamp_min(1e-5)
    qs = quantiles(seq, (0.05, 0.25, 0.50, 0.75, 0.95), dim=0)
    centered = (seq - mean) / std
    skew = torch.clamp((centered**3).mean(dim=0), -8.0, 8.0)
    kurt = torch.clamp((centered**4).mean(dim=0) - 3.0, -8.0, 8.0)
    half = max(1, seq.shape[0] // 2)
    early = seq[:half].mean(dim=0)
    late = seq[-half:].mean(dim=0)
    trend = late - early
    iqr = qs[3] - qs[1]
    return torch.cat([mean, std, qs.reshape(-1), iqr, skew, kurt, trend], dim=0)


@torch.no_grad()
def clip_vector(path: Path, device: torch.device) -> ClipRecord:
    augmented = is_augmented_path(path)
    with np.load(path, allow_pickle=True) as data:
        speaker_id = decode(data["speaker_id"]) if "speaker_id" in data else path.stem.rsplit("_", 1)[0]
        height_cm = safe_float(data, "height_cm")
        gender = gender_id(data)
        source = decode(data["source"]).upper() if "source" in data else ""
        if source not in {"TIMIT", "NISP", "CELEB", "VOXCELEB"}:
            source = "UNKNOWN"

        seq_np = np.asarray(data["sequence"], dtype=np.float32)
        if seq_np.ndim != 2 or seq_np.size == 0:
            raise ValueError(f"Bad sequence: {path}")
        seq = torch.from_numpy(seq_np).to(device=device, non_blocking=True)
        pieces = [sequence_stats(seq)]

        if "ssl_embedding" in data:
            ssl = torch.from_numpy(np.asarray(data["ssl_embedding"], dtype=np.float32).reshape(-1)).to(device)
            pieces.append(clean_tensor(ssl))

        scalars = [safe_float(data, key, 0.0) for key in SCALAR_KEYS]
        src_id = source_id(source, speaker_id)
        source_onehot = [1.0 if src_id == idx else 0.0 for idx in range(4)]
        scalars.extend(
            [
                float(seq.shape[0]),
                float(seq.shape[1]),
                float(gender),
                float(augmented),
                *source_onehot,
            ]
        )
        scalar_t = torch.tensor(scalars, dtype=torch.float32, device=device)
        vector = torch.cat([*pieces, clean_tensor(scalar_t)], dim=0)
        quality = clip_quality(data, augmented=augmented)
    return ClipRecord(str(speaker_id), float(height_cm), int(gender), str(source), vector, quality, augmented)


@torch.no_grad()
def speaker_pool(records: Sequence[ClipRecord], device: torch.device) -> Tuple[torch.Tensor, Dict[str, Any], bool]:
    heights = [rec.height_cm for rec in records if math.isfinite(rec.height_cm)]
    if not heights:
        raise ValueError("speaker has no height labels")
    vectors = torch.stack([rec.vector for rec in records], dim=0)
    qualities = torch.tensor([rec.quality for rec in records], dtype=torch.float32, device=device).clamp_min(0.03)
    weights = qualities / qualities.sum().clamp_min(1e-6)

    weighted_mean = (vectors * weights[:, None]).sum(dim=0)
    plain_mean = vectors.mean(dim=0)
    std = vectors.std(dim=0, unbiased=False)
    qs = quantiles(vectors, (0.25, 0.50, 0.75), dim=0)
    q25, q50, q75 = qs[0], qs[1], qs[2]
    top_k = max(1, int(math.ceil(0.40 * vectors.shape[0])))
    top_idx = torch.topk(qualities, k=top_k).indices
    top_mean = vectors[top_idx].mean(dim=0)
    reliability_gap = top_mean - plain_mean

    heights_np = np.asarray(heights, dtype=np.float32)
    height_range = float(np.max(heights_np) - np.min(heights_np)) if heights_np.size else 0.0
    height = float(np.median(heights_np))
    gender = int(round(float(np.median([rec.gender for rec in records]))))
    source = Counter(rec.source for rec in records).most_common(1)[0][0]
    augmented_count = sum(1 for rec in records if rec.augmented)
    source_idx = source_id(source, records[0].speaker_id)

    meta_features = torch.tensor(
        [
            float(vectors.shape[0]),
            float(augmented_count),
            float(qualities.mean().item()),
            float(qualities.std(unbiased=False).item()) if qualities.numel() > 1 else 0.0,
            float(qualities.max().item()),
            float(height_range),
            float(gender),
            *[1.0 if source_idx == idx else 0.0 for idx in range(4)],
        ],
        dtype=torch.float32,
        device=device,
    )
    speaker_vector = torch.cat(
        [
            weighted_mean,
            q50,
            std,
            q75 - q25,
            top_mean,
            reliability_gap,
            meta_features,
        ],
        dim=0,
    )
    metadata = {
        "speaker_id": records[0].speaker_id,
        "height_cm": height,
        "height_bin": height_bin_id(height),
        "gender": gender,
        "source": source,
        "source_id": source_idx,
        "n_clips": int(vectors.shape[0]),
        "n_augmented": int(augmented_count),
        "quality_mean": float(qualities.mean().item()),
        "quality_max": float(qualities.max().item()),
        "height_range_cm": height_range,
    }
    consistent = height_range <= 2.0
    return clean_tensor(speaker_vector), metadata, consistent


@torch.no_grad()
def build_split(
    split_dir: Path,
    *,
    split_name: str,
    device: torch.device,
    include_augmented: bool,
    max_clips_per_speaker: int,
    max_height_range_cm: float,
) -> Dict[str, Any]:
    paths = sorted(split_dir.glob("*.npz"))
    if not include_augmented:
        paths = [path for path in paths if not is_augmented_path(path)]

    grouped: Dict[str, List[ClipRecord]] = defaultdict(list)
    skipped = Counter()
    for idx, path in enumerate(paths, start=1):
        try:
            rec = clip_vector(path, device)
        except Exception:
            skipped["load_or_vector_failed"] += 1
            continue
        if not math.isfinite(rec.height_cm):
            skipped["missing_height"] += 1
            continue
        grouped[rec.speaker_id].append(rec)
        if idx % 2500 == 0:
            print(f"[speaker-phase1] {split_name}: vectorized {idx}/{len(paths)} clips", flush=True)

    rows: List[torch.Tensor] = []
    targets: List[float] = []
    speaker_ids: List[str] = []
    metadata: List[Dict[str, Any]] = []
    for speaker_id, records in sorted(grouped.items()):
        records = sorted(records, key=lambda row: (row.augmented, -row.quality))
        if max_clips_per_speaker > 0:
            records = records[:max_clips_per_speaker]
        try:
            speaker_vector, meta, consistent = speaker_pool(records, device)
        except Exception:
            skipped["speaker_pool_failed"] += 1
            continue
        if split_name == "train" and float(meta["height_range_cm"]) > float(max_height_range_cm):
            skipped["inconsistent_speaker_height"] += 1
            continue
        rows.append(speaker_vector)
        targets.append(float(meta["height_cm"]))
        speaker_ids.append(speaker_id)
        metadata.append(meta)

    if not rows:
        raise RuntimeError(f"No usable speaker rows built for split {split_name} from {split_dir}")
    x = torch.stack(rows, dim=0).float()
    y = torch.tensor(targets, dtype=torch.float32, device=device)
    print(
        f"[speaker-phase1] {split_name}: speakers={x.shape[0]} clips={len(paths)} "
        f"feature_dim={x.shape[1]} skipped={dict(skipped)}",
        flush=True,
    )
    return {"x": x, "y": y, "speaker_ids": speaker_ids, "metadata": metadata, "skipped": dict(skipped)}


def load_or_build_cache(args: argparse.Namespace, device: torch.device, output_dir: Path) -> Dict[str, Any]:
    cache_path = output_dir / str(args.cache_name)
    if cache_path.exists() and not args.rebuild_cache:
        payload = torch.load(cache_path, map_location=device, weights_only=False)
        legacy_ok = all(
            split in payload
            and isinstance(payload[split], Mapping)
            and "x" in payload[split]
            and "y" in payload[split]
            and "metadata" in payload[split]
            for split in ("train", "val", "test")
        )
        if int(payload.get("cache_version", -1)) == CACHE_VERSION or legacy_ok:
            label = "phase1" if int(payload.get("cache_version", -1)) == CACHE_VERSION else "legacy"
            print(f"[speaker-phase1] loading {label} cache {cache_path}", flush=True)
            for split in ("train", "val", "test"):
                payload[split]["x"] = payload[split]["x"].to(device)
                payload[split]["y"] = payload[split]["y"].to(device)
                payload[split].setdefault("skipped", {})
            return payload
        print(f"[speaker-phase1] cache version mismatch, rebuilding {cache_path}", flush=True)

    features_root = resolve(args.features_root)
    include_aug = not bool(args.original_only)
    payload: Dict[str, Any] = {
        "cache_version": CACHE_VERSION,
        "features_root": str(features_root),
        "include_augmented": include_aug,
        "max_clips_per_speaker": int(args.max_clips_per_speaker),
        "max_speaker_height_range_cm": float(args.max_speaker_height_range_cm),
    }
    for split in ("train", "val", "test"):
        payload[split] = build_split(
            features_root / split,
            split_name=split,
            device=device,
            include_augmented=include_aug if split == "train" else False,
            max_clips_per_speaker=int(args.max_clips_per_speaker),
            max_height_range_cm=float(args.max_speaker_height_range_cm),
        )

    cpu_payload: Dict[str, Any] = {}
    for key, value in payload.items():
        if key in {"train", "val", "test"}:
            cpu_payload[key] = {
                "x": value["x"].detach().cpu(),
                "y": value["y"].detach().cpu(),
                "speaker_ids": value["speaker_ids"],
                "metadata": value["metadata"],
                "skipped": value.get("skipped", {}),
            }
        else:
            cpu_payload[key] = value
    torch.save(cpu_payload, cache_path)
    print(f"[speaker-phase1] wrote cache {cache_path}", flush=True)
    return payload


class ResidualBlock(nn.Module):
    def __init__(self, hidden: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.Dropout(dropout * 0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ff(self.norm(x))


class Phase1HeightNet(nn.Module):
    def __init__(self, input_dim: int, hidden: int, dropout: float, blocks: int, experts: int):
        super().__init__()
        self.experts = max(2, int(experts))
        self.stem = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList([ResidualBlock(hidden, dropout) for _ in range(max(1, int(blocks)))])
        self.final_norm = nn.LayerNorm(hidden)
        self.expert_heads = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(self.experts)])
        self.gate = nn.Linear(hidden, self.experts)
        self.log_scale = nn.Linear(hidden, 1)
        self.bin_head = nn.Linear(hidden, 3)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.stem(x)
        for block in self.blocks:
            h = block(h)
        h = self.final_norm(h)
        expert_values = torch.cat([head(h) for head in self.expert_heads], dim=1)
        gates = torch.softmax(self.gate(h), dim=1)
        mean = (expert_values * gates).sum(dim=1)
        log_scale = torch.clamp(self.log_scale(h).squeeze(1), -3.0, 1.2)
        return {
            "mean": mean,
            "log_scale": log_scale,
            "bin_logits": self.bin_head(h),
            "gates": gates,
        }


def robust_standardize(train_x: torch.Tensor, *others: torch.Tensor) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, ...]]:
    center = torch.median(train_x, dim=0, keepdim=True).values
    q = quantiles(train_x, (0.25, 0.75), dim=0)
    scale = (q[1] - q[0]).reshape(1, -1) / 1.349
    std = train_x.std(dim=0, unbiased=False, keepdim=True)
    scale = torch.where(scale.abs() < 1e-5, std, scale).clamp_min(1e-4)
    scaled = tuple(clean_tensor((x - center) / scale).clamp(-8.0, 8.0) for x in (train_x,) + others)
    return (center, scale), scaled


def target_standardize(train_y: torch.Tensor, *others: torch.Tensor) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, ...]]:
    mean = train_y.mean()
    std = train_y.std(unbiased=False).clamp_min(1e-4)
    return (mean, std), tuple((y - mean) / std for y in (train_y,) + others)


def height_bins(y_cm: torch.Tensor) -> torch.Tensor:
    return torch.where(y_cm < 160.0, torch.zeros_like(y_cm, dtype=torch.long), torch.where(y_cm < 175.0, torch.ones_like(y_cm, dtype=torch.long), torch.full_like(y_cm, 2, dtype=torch.long)))


def sample_weights(y_cm: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    weights = torch.ones_like(y_cm)
    weights = torch.where(y_cm < 160.0, weights * float(args.short_weight), weights)
    weights = torch.where(y_cm >= 175.0, weights * float(args.tall_weight), weights)
    return weights / weights.mean().clamp_min(1e-6)


def pairwise_rank_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.numel() < 4:
        return pred.new_tensor(0.0)
    idx_a = torch.randint(0, pred.numel(), (min(256, pred.numel() * 4),), device=pred.device)
    idx_b = torch.randint(0, pred.numel(), (idx_a.numel(),), device=pred.device)
    dy = target[idx_a] - target[idx_b]
    mask = dy.abs() > 0.35
    if not mask.any():
        return pred.new_tensor(0.0)
    dp = pred[idx_a] - pred[idx_b]
    sign = dy.sign()
    return F.softplus(-(dp[mask] * sign[mask]) / 0.35).mean()


def loss_fn(out: Mapping[str, torch.Tensor], target_norm: torch.Tensor, target_cm: torch.Tensor, weights: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    mean = out["mean"]
    log_scale = out["log_scale"]
    scale = torch.exp(log_scale).clamp_min(0.05)
    err = mean - target_norm
    huber = F.huber_loss(mean, target_norm, delta=0.55, reduction="none")
    mae = err.abs()
    nll = 0.5 * (err / scale) ** 2 + log_scale
    bin_target = height_bins(target_cm)
    bin_loss = F.cross_entropy(out["bin_logits"], bin_target, reduction="none")
    loss = ((0.55 * huber + 0.33 * mae + float(args.nll_weight) * nll + float(args.bin_weight) * bin_loss) * weights).mean()
    loss = loss + float(args.rank_weight) * pairwise_rank_loss(mean, target_norm)
    return loss


@torch.no_grad()
def metrics(y_true: torch.Tensor, y_pred: torch.Tensor, metadata: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    err = y_pred - y_true
    abs_err = err.abs()
    out = {
        "mae": float(abs_err.mean().item()),
        "rmse": float(torch.sqrt((err * err).mean()).item()),
        "median_ae": float(abs_err.median().item()),
        "p90_ae": float(torch.quantile(abs_err, 0.90).item()),
        "bias": float(err.mean().item()),
        "within_3cm": float((abs_err <= 3.0).float().mean().item()),
        "within_5cm": float((abs_err <= 5.0).float().mean().item()),
        "pred_std": float(y_pred.std(unbiased=False).item()),
        "true_std": float(y_true.std(unbiased=False).item()),
        "n_speakers": float(y_true.numel()),
    }
    for label, idx in (("short", 0), ("medium", 1), ("tall", 2)):
        mask = torch.tensor([int(row["height_bin"]) == idx for row in metadata], dtype=torch.bool, device=y_true.device)
        if mask.any():
            out[f"{label}_mae"] = float(abs_err[mask].mean().item())
            out[f"{label}_n"] = float(mask.sum().item())
    for source in sorted({str(row["source"]) for row in metadata}):
        mask = torch.tensor([str(row["source"]) == source for row in metadata], dtype=torch.bool, device=y_true.device)
        if mask.any():
            out[f"source_{source.lower()}_mae"] = float(abs_err[mask].mean().item())
            out[f"source_{source.lower()}_n"] = float(mask.sum().item())
    return out


def val_score(m: Mapping[str, float], args: argparse.Namespace) -> float:
    short = float(m.get("short_mae", m["mae"]))
    p90 = float(m.get("p90_ae", m["mae"]))
    return float(m["mae"]) + float(args.val_short_penalty) * max(0.0, short - float(m["mae"])) + 0.08 * p90


def train_once(
    *,
    train_x: torch.Tensor,
    train_y_cm: torch.Tensor,
    val_x: torch.Tensor,
    val_y_cm: torch.Tensor,
    test_x: torch.Tensor,
    test_y_cm: torch.Tensor,
    val_meta: Sequence[Mapping[str, Any]],
    test_meta: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
    seed: int,
) -> Dict[str, Any]:
    seed_everything(seed)
    (y_mean, y_std), (train_y, val_y, test_y) = target_standardize(train_y_cm, val_y_cm, test_y_cm)
    model = Phase1HeightNet(
        input_dim=train_x.shape[1],
        hidden=int(args.hidden),
        dropout=float(args.dropout),
        blocks=int(args.blocks),
        experts=int(args.experts),
    ).to(train_x.device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=80, T_mult=2, eta_min=float(args.lr) * 0.02)
    weights_all = sample_weights(train_y_cm, args)
    draw_probs = weights_all / weights_all.sum().clamp_min(1e-6)
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_score = float("inf")
    best_epoch = 0
    bad_epochs = 0
    n = train_x.shape[0]
    batch_size = min(int(args.batch_size), n)
    steps_per_epoch = max(1, math.ceil(n / batch_size))

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        total_loss = 0.0
        for _ in range(steps_per_epoch):
            idx = torch.multinomial(draw_probs, num_samples=batch_size, replacement=True)
            out = model(train_x[idx])
            loss = loss_fn(out, train_y[idx], train_y_cm[idx], weights_all[idx], args)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += float(loss.detach().item())
        scheduler.step(epoch)

        model.eval()
        with torch.no_grad():
            val_out = model(val_x)
            val_pred_cm = val_out["mean"] * y_std + y_mean
            val_metrics = metrics(val_y_cm, val_pred_cm, val_meta)
            score = val_score(val_metrics, args)
        if score < best_score:
            best_score = score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if epoch == 1 or epoch % 25 == 0:
            print(
                f"[speaker-phase1][seed={seed}] e={epoch} loss={total_loss / steps_per_epoch:.4f} "
                f"val={val_metrics['mae']:.3f} short={val_metrics.get('short_mae', float('nan')):.3f} "
                f"medium={val_metrics.get('medium_mae', float('nan')):.3f} "
                f"tall={val_metrics.get('tall_mae', float('nan')):.3f} score={score:.3f} best={best_score:.3f}",
                flush=True,
            )
        if bad_epochs >= int(args.patience):
            break

    if best_state is None:
        raise RuntimeError("training failed to produce a checkpoint")
    model.load_state_dict({k: v.to(train_x.device) for k, v in best_state.items()})
    model.eval()
    with torch.no_grad():
        val_pred_cm = model(val_x)["mean"] * y_std + y_mean
        test_pred_cm = model(test_x)["mean"] * y_std + y_mean
    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "state_dict": best_state,
        "y_mean": float(y_mean.item()),
        "y_std": float(y_std.item()),
        "val_pred": val_pred_cm.detach().cpu(),
        "test_pred": test_pred_cm.detach().cpu(),
        "val": metrics(val_y_cm, val_pred_cm, val_meta),
        "test": metrics(test_y_cm, test_pred_cm, test_meta),
    }


def group_key(row: Mapping[str, Any]) -> str:
    return f"{str(row.get('source', 'UNKNOWN')).upper()}|g{int(row.get('gender', 0))}"


@torch.no_grad()
def fit_calibration(
    val_y: torch.Tensor,
    val_pred: torch.Tensor,
    val_meta: Sequence[Mapping[str, Any]],
    shrinkage: float,
) -> Dict[str, Any]:
    device = val_y.device
    x = val_pred.float()
    y = val_y.float()
    x_mean = x.mean()
    y_mean = y.mean()
    slope = ((x - x_mean) * (y - y_mean)).sum() / (((x - x_mean) ** 2).sum() + 1e-6)
    slope = torch.clamp(slope, 0.70, 1.30)
    intercept = y_mean - slope * x_mean
    base = slope * x + intercept
    residual = y - base
    offsets: Dict[str, float] = {}
    for key in sorted({group_key(row) for row in val_meta}):
        mask = torch.tensor([group_key(row) == key for row in val_meta], dtype=torch.bool, device=device)
        n = int(mask.sum().item())
        if n <= 0:
            continue
        offset = residual[mask].mean() * (float(n) / (float(n) + float(shrinkage)))
        offsets[key] = float(offset.item())
    return {"slope": float(slope.item()), "intercept": float(intercept.item()), "offsets": offsets, "shrinkage": float(shrinkage)}


@torch.no_grad()
def apply_calibration(pred: torch.Tensor, meta: Sequence[Mapping[str, Any]], calibration: Mapping[str, Any]) -> torch.Tensor:
    out = pred.float() * float(calibration.get("slope", 1.0)) + float(calibration.get("intercept", 0.0))
    offsets = dict(calibration.get("offsets", {}))
    if offsets:
        offset_t = torch.tensor([float(offsets.get(group_key(row), 0.0)) for row in meta], dtype=torch.float32, device=pred.device)
        out = out + offset_t
    return out


def make_jsonable(row: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in row.items():
        if key == "state_dict":
            continue
        if isinstance(value, torch.Tensor):
            continue
        if isinstance(value, np.generic):
            out[key] = value.item()
        else:
            out[key] = value
    return out


def write_predictions(path: Path, split_payload: Mapping[str, Any], pred: torch.Tensor, pred_calibrated: Optional[torch.Tensor] = None) -> None:
    rows = []
    pred_np = pred.detach().cpu().numpy()
    cal_np = pred_calibrated.detach().cpu().numpy() if pred_calibrated is not None else None
    for idx, row in enumerate(split_payload["metadata"]):
        true = float(row["height_cm"])
        item = {
            "speaker_id": row["speaker_id"],
            "source": row["source"],
            "gender": row["gender"],
            "height_cm": f"{true:.6f}",
            "height_bin": row["height_bin"],
            "n_clips": row["n_clips"],
            "n_augmented": row.get("n_augmented", 0),
            "quality_mean": f"{float(row.get('quality_mean', 0.0)):.6f}",
            "pred_cm": f"{float(pred_np[idx]):.6f}",
            "abs_error_cm": f"{abs(float(pred_np[idx]) - true):.6f}",
        }
        if cal_np is not None:
            item["pred_calibrated_cm"] = f"{float(cal_np[idx]):.6f}"
            item["abs_error_calibrated_cm"] = f"{abs(float(cal_np[idx]) - true):.6f}"
        rows.append(item)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if str(args.device).lower() != "cuda" or not torch.cuda.is_available():
        raise SystemExit("CUDA is required. Refusing CPU training.")
    device = torch.device("cuda")
    seed_everything(int(args.seed))
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = load_or_build_cache(args, device, output_dir)
    (x_center, x_scale), (train_x, val_x, test_x) = robust_standardize(
        payload["train"]["x"], payload["val"]["x"], payload["test"]["x"]
    )
    train_y = payload["train"]["y"]
    val_y = payload["val"]["y"]
    test_y = payload["test"]["y"]

    seeds = parse_seeds(args)
    print(f"[speaker-phase1] CUDA device: {torch.cuda.get_device_name(0)} | seeds={seeds}", flush=True)
    seed_results = []
    for seed in seeds:
        seed_results.append(
            train_once(
                train_x=train_x,
                train_y_cm=train_y,
                val_x=val_x,
                val_y_cm=val_y,
                test_x=test_x,
                test_y_cm=test_y,
                val_meta=payload["val"]["metadata"],
                test_meta=payload["test"]["metadata"],
                args=args,
                seed=seed,
            )
        )

    val_stack = torch.stack([row["val_pred"] for row in seed_results], dim=0).to(device)
    test_stack = torch.stack([row["test_pred"] for row in seed_results], dim=0).to(device)
    val_pred = val_stack.mean(dim=0)
    test_pred = test_stack.mean(dim=0)
    final_val_raw = metrics(val_y, val_pred, payload["val"]["metadata"])
    final_test_raw = metrics(test_y, test_pred, payload["test"]["metadata"])

    calibration = fit_calibration(val_y, val_pred, payload["val"]["metadata"], float(args.calibration_shrinkage))
    val_pred_cal = apply_calibration(val_pred, payload["val"]["metadata"], calibration)
    test_pred_cal = apply_calibration(test_pred, payload["test"]["metadata"], calibration)
    final_val_cal = metrics(val_y, val_pred_cal, payload["val"]["metadata"])
    final_test_cal = metrics(test_y, test_pred_cal, payload["test"]["metadata"])

    selected = min(seed_results, key=lambda row: row["best_score"])
    report = {
        "phase": "phase1_cuda_speaker_foundation",
        "features_root": str(resolve(args.features_root)),
        "output_dir": str(output_dir),
        "device": torch.cuda.get_device_name(0),
        "target_mae_cm": float(args.target_mae_cm),
        "target_met_raw": bool(final_test_raw["mae"] <= float(args.target_mae_cm)),
        "target_met_calibrated": bool(final_test_cal["mae"] <= float(args.target_mae_cm)),
        "cache_version": CACHE_VERSION,
        "include_augmented": not bool(args.original_only),
        "speaker_counts": {
            "train": int(train_x.shape[0]),
            "val": int(val_x.shape[0]),
            "test": int(test_x.shape[0]),
        },
        "feature_dim": int(train_x.shape[1]),
        "seeds": seeds,
        "calibration": calibration,
        "final_val_raw": final_val_raw,
        "final_test_raw": final_test_raw,
        "final_val_calibrated": final_val_cal,
        "final_test_calibrated": final_test_cal,
        "selected_single_seed": make_jsonable(selected),
        "seed_results": [make_jsonable(row) for row in seed_results],
        "skipped": {
            "train": payload["train"].get("skipped", {}),
            "val": payload["val"].get("skipped", {}),
            "test": payload["test"].get("skipped", {}),
        },
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=True), encoding="utf-8")
    write_predictions(output_dir / "predictions_val.csv", payload["val"], val_pred.detach().cpu(), val_pred_cal.detach().cpu())
    write_predictions(output_dir / "predictions_test.csv", payload["test"], test_pred.detach().cpu(), test_pred_cal.detach().cpu())
    torch.save(
        {
            "model_state_dict": selected["state_dict"],
            "selected_seed": selected["seed"],
            "metrics": report,
            "feature_center": x_center.detach().cpu(),
            "feature_scale": x_scale.detach().cpu(),
        },
        output_dir / "best_single_seed_model.pt",
    )
    print(
        "[speaker-phase1] final raw test "
        f"mae={final_test_raw['mae']:.3f} short={final_test_raw.get('short_mae', float('nan')):.3f} "
        f"medium={final_test_raw.get('medium_mae', float('nan')):.3f} tall={final_test_raw.get('tall_mae', float('nan')):.3f}",
        flush=True,
    )
    print(
        "[speaker-phase1] final calibrated test "
        f"mae={final_test_cal['mae']:.3f} short={final_test_cal.get('short_mae', float('nan')):.3f} "
        f"medium={final_test_cal.get('medium_mae', float('nan')):.3f} tall={final_test_cal.get('tall_mae', float('nan')):.3f} "
        f"within3={final_test_cal['within_3cm'] * 100:.1f}%",
        flush=True,
    )
    print(f"[speaker-phase1] wrote {output_dir / 'metrics.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
