#!/usr/bin/env python
"""Phase 9 ECAPA external-prior stack.

This phase is intentionally different from the previous VocalMorph runs:

1. Read raw audio paths from the canonical split CSV files.
2. Extract speaker embeddings with SpeechBrain ECAPA-TDNN on CUDA.
3. Run the cached VoxCeleb height SVR prior on those ECAPA embeddings.
4. Train small, validation-selected speaker-level models over ECAPA features.
5. Blend only against the Phase 3 frontier using held-out validation speakers.

The goal is not to inflate the model size. The goal is to ask whether a
pretrained speaker-verification representation plus an external height prior
contains new height signal that the existing WavLM-fused tensors missed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np
import torch
import torchaudio

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class SpeakerRow:
    speaker_id: str
    source: str
    gender: int
    height_cm: float
    audio_paths: List[Path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 9 ECAPA prior stack.")
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--phase3-val-pred", default="outputs/phase3_target_domain_rescue/phase3_predictions_val.csv")
    parser.add_argument("--phase3-test-pred", default="outputs/phase3_target_domain_rescue/phase3_predictions_test.csv")
    parser.add_argument("--output-dir", default="outputs/phase9_ecapa_prior_stack")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-clips-per-speaker", type=int, default=6)
    parser.add_argument("--max-seconds", type=float, default=6.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--limit-speakers", type=int, default=0, help="Debug limit per split.")
    parser.add_argument("--svr-model-dir", default="", help="Optional local griko height SVR snapshot.")
    parser.add_argument("--include-celeb-support", action="store_true", help="Add CELEB/VOXCELEB speakers from feature NPZs to train support.")
    parser.add_argument("--celeb-features-root", default="data/features_v4_combo_full_ssl")
    parser.add_argument("--max-celeb-speakers", type=int, default=0)
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def gender_id(raw: str) -> int:
    text = str(raw or "").strip().lower()
    if text == "male":
        return 1
    if text == "female":
        return 0
    return int(safe_float(text, 0.0) >= 0.5)


def decode_np(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return ""
        return decode_np(value.reshape(-1)[0])
    return str(value)


def source_id(source: str, sid: str = "") -> int:
    src = str(source or "").upper()
    sid = str(sid or "").upper()
    if src == "TIMIT" or sid.startswith("TIMIT_"):
        return 0
    if src == "NISP" or sid.startswith("NISP_"):
        return 1
    if src in {"CELEB", "VOXCELEB", "HEIGHTCELEB"} or sid.startswith(("CELEB_", "VOXCELEB_", "HEIGHTCELEB_")):
        return 2
    return 3


def is_target_source(source: str) -> bool:
    return str(source or "").upper() in {"NISP", "TIMIT"}


def read_split_csv(path: Path, max_clips: int, limit_speakers: int = 0) -> List[SpeakerRow]:
    rows: List[SpeakerRow] = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            sid = str(raw.get("speaker_id", "")).strip()
            height = safe_float(raw.get("height_cm"))
            if not sid or not math.isfinite(height):
                continue
            audio_paths = []
            parts = [part.strip() for part in str(raw.get("audio_paths", "")).split("|") if part.strip()]
            for part in select_evenly(parts, max_clips):
                audio_path = Path(part)
                audio_paths.append(audio_path if audio_path.is_absolute() else ROOT / audio_path)
            if not audio_paths:
                continue
            rows.append(
                SpeakerRow(
                    speaker_id=sid,
                    source=str(raw.get("source", "")).upper(),
                    gender=gender_id(str(raw.get("gender", ""))),
                    height_cm=height,
                    audio_paths=audio_paths,
                )
            )
            if limit_speakers > 0 and len(rows) >= limit_speakers:
                break
    return rows


def read_celeb_rows_from_features(
    features_root: Path,
    max_clips: int,
    *,
    max_speakers: int = 0,
) -> List[SpeakerRow]:
    split_dir = features_root / "train"
    if not split_dir.exists():
        print(f"[phase9] CELEB support skipped, missing {split_dir}", flush=True)
        return []

    grouped: Dict[str, Dict[str, Any]] = {}
    for path in sorted(split_dir.glob("CELEB_*.npz")):
        if "_aug" in path.stem.lower():
            continue
        sid = path.stem.rsplit("_", 1)[0]
        if max_speakers > 0 and sid not in grouped and len(grouped) >= max_speakers:
            break
        if sid in grouped and len(grouped[sid]["audio_paths"]) >= max_clips:
            continue
        try:
            with np.load(path, allow_pickle=True) as data:
                audio_raw = decode_np(data["audio_rel_path"]) if "audio_rel_path" in data else ""
                audio_path = Path(audio_raw)
                if audio_raw and not audio_path.is_absolute():
                    audio_path = ROOT / audio_path
                height = safe_float(np.asarray(data["height_cm"]).reshape(-1)[0]) if "height_cm" in data else float("nan")
                gender = int(round(safe_float(np.asarray(data["gender"]).reshape(-1)[0], 0.0))) if "gender" in data else 0
        except Exception:
            continue
        if not audio_raw or not math.isfinite(height):
            continue
        if sid not in grouped:
            grouped[sid] = {
                "speaker_id": sid,
                "source": "CELEB",
                "gender": gender,
                "height_cm": float(height),
                "audio_paths": [],
            }
        grouped[sid]["audio_paths"].append(audio_path)

    rows = [
        SpeakerRow(
            speaker_id=str(item["speaker_id"]),
            source=str(item["source"]),
            gender=int(item["gender"]),
            height_cm=float(item["height_cm"]),
            audio_paths=list(item["audio_paths"]),
        )
        for item in grouped.values()
        if item["audio_paths"]
    ]
    print(f"[phase9] CELEB support speakers={len(rows)} from {split_dir}", flush=True)
    return rows


def select_evenly(items: Sequence[str], max_items: int) -> List[str]:
    if max_items <= 0 or len(items) <= max_items:
        return list(items)
    if max_items == 1:
        return [items[len(items) // 2]]
    idx = np.linspace(0, len(items) - 1, num=max_items)
    out = []
    seen = set()
    for i in idx:
        j = int(round(float(i)))
        if j not in seen:
            out.append(items[j])
            seen.add(j)
    return out


def load_audio(path: Path, *, sample_rate: int, max_seconds: float) -> Optional[torch.Tensor]:
    try:
        wav, sr = torchaudio.load(str(path))
    except Exception:
        return None
    if wav.numel() == 0:
        return None
    if wav.ndim == 2:
        wav = wav.mean(dim=0)
    else:
        wav = wav.reshape(-1)
    wav = wav.float()
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    wav = wav - wav.mean()
    peak = wav.abs().max().clamp_min(1e-5)
    wav = (wav / peak).clamp(-1.0, 1.0)
    max_samples = int(max(1.0, float(max_seconds)) * sample_rate)
    if wav.numel() > max_samples:
        wav = loudest_window(wav, max_samples)
    if wav.numel() < int(0.35 * sample_rate):
        return None
    return wav.contiguous()


def loudest_window(wav: torch.Tensor, max_samples: int) -> torch.Tensor:
    if wav.numel() <= max_samples:
        return wav
    hop = max(1, max_samples // 4)
    best_start = 0
    best_energy = -1.0
    for start in range(0, max(1, wav.numel() - max_samples + 1), hop):
        chunk = wav[start : start + max_samples]
        energy = float((chunk * chunk).mean().item())
        if energy > best_energy:
            best_start = start
            best_energy = energy
    return wav[best_start : best_start + max_samples]


class EcapaPrior:
    def __init__(self, device: str, svr_model_dir: str = "") -> None:
        from speechbrain.inference.speaker import EncoderClassifier

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested for Phase9, but torch.cuda.is_available() is false")
        self.device = torch.device(device)
        self.sample_rate = 16000
        self.classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(Path.home() / ".cache" / "speechbrain" / "spkrec-ecapa-voxceleb"),
            run_opts={"device": str(self.device)},
        )
        self.scaler, self.svr, self.svr_path = load_height_svr(svr_model_dir)

    @torch.no_grad()
    def embed_wavs(self, wavs: Sequence[torch.Tensor]) -> np.ndarray:
        lengths = torch.tensor([wav.numel() for wav in wavs], dtype=torch.float32)
        max_len = int(lengths.max().item())
        batch = torch.zeros((len(wavs), max_len), dtype=torch.float32)
        for idx, wav in enumerate(wavs):
            batch[idx, : wav.numel()] = wav
        wav_lens = (lengths / max(float(max_len), 1.0)).to(self.device)
        batch = batch.to(self.device, non_blocking=True)
        emb = self.classifier.encode_batch(batch, wav_lens=wav_lens, normalize=False)
        emb = emb.squeeze(1).detach().float().cpu().numpy().astype(np.float32)
        return emb

    def predict_prior(self, embeddings: np.ndarray) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scaled = self.scaler.transform(embeddings.astype(np.float32))
            pred = self.svr.predict(scaled)
        return np.asarray(pred, dtype=np.float32)


def load_height_svr(svr_model_dir: str = "") -> Tuple[Any, Any, str]:
    if svr_model_dir:
        snap = resolve(svr_model_dir)
    else:
        base = Path.home() / ".cache" / "huggingface" / "hub" / "models--griko--height_reg_svr_ecapa_voxceleb" / "snapshots"
        snaps = sorted([p for p in base.glob("*") if p.is_dir()])
        if not snaps:
            raise FileNotFoundError(f"Could not find cached griko height SVR under {base}")
        snap = snaps[-1]
    scaler_path = snap / "scaler.joblib"
    model_path = snap / "svr_model.joblib"
    if not scaler_path.exists() or not model_path.exists():
        raise FileNotFoundError(f"Missing scaler.joblib or svr_model.joblib in {snap}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return joblib.load(scaler_path), joblib.load(model_path), str(snap)


def aggregate_speaker(
    row: SpeakerRow,
    embeddings: np.ndarray,
    prior: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError("speaker has no ECAPA embeddings")
    normed = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-6)
    with np.errstate(all="ignore"):
        pieces = [
            embeddings.mean(axis=0),
            embeddings.std(axis=0),
            np.median(embeddings, axis=0),
            normed.mean(axis=0),
            normed.std(axis=0),
            np.median(normed, axis=0),
        ]
    prior = np.asarray(prior, dtype=np.float32).reshape(-1)
    prior_stats = np.asarray(
        [
            float(prior.mean()),
            float(prior.std()),
            float(np.median(prior)),
            float(prior.min()),
            float(prior.max()),
            float(prior.max() - prior.min()),
            float(len(prior)),
        ],
        dtype=np.float32,
    )
    sid = source_id(row.source, row.speaker_id)
    meta_vec = np.asarray(
        [
            float(row.gender),
            *[1.0 if sid == idx else 0.0 for idx in range(4)],
        ],
        dtype=np.float32,
    )
    vector = np.concatenate([*pieces, prior_stats, meta_vec]).astype(np.float32)
    meta = {
        "speaker_id": row.speaker_id,
        "source": row.source,
        "gender": int(row.gender),
        "height_cm": float(row.height_cm),
        "n_clips": int(embeddings.shape[0]),
        "prior_mean": float(prior_stats[0]),
        "prior_median": float(prior_stats[2]),
        "height_bin": int(height_bin_scalar(row.height_cm)),
    }
    return vector, meta


def build_split(
    name: str,
    rows: Sequence[SpeakerRow],
    extractor: EcapaPrior,
    *,
    batch_size: int,
    max_seconds: float,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    vectors: List[np.ndarray] = []
    targets: List[float] = []
    metadata: List[Dict[str, Any]] = []
    missing_audio = 0
    failed_speakers = 0

    pending_wavs: List[torch.Tensor] = []
    pending_refs: List[Tuple[int, SpeakerRow]] = []
    speaker_embs: Dict[int, List[np.ndarray]] = {idx: [] for idx in range(len(rows))}
    speaker_priors: Dict[int, List[float]] = {idx: [] for idx in range(len(rows))}

    def flush() -> None:
        nonlocal pending_wavs, pending_refs
        if not pending_wavs:
            return
        emb = extractor.embed_wavs(pending_wavs)
        pred = extractor.predict_prior(emb)
        for row_emb, row_pred, (speaker_idx, _row) in zip(emb, pred, pending_refs):
            speaker_embs[speaker_idx].append(row_emb.astype(np.float32))
            speaker_priors[speaker_idx].append(float(row_pred))
        pending_wavs = []
        pending_refs = []

    for speaker_idx, row in enumerate(rows, start=0):
        for path in row.audio_paths:
            wav = load_audio(path, sample_rate=extractor.sample_rate, max_seconds=max_seconds)
            if wav is None:
                missing_audio += 1
                continue
            pending_wavs.append(wav)
            pending_refs.append((speaker_idx, row))
            if len(pending_wavs) >= batch_size:
                flush()
        if (speaker_idx + 1) % 100 == 0:
            flush()
            print(f"[phase9] {name}: embedded {speaker_idx + 1}/{len(rows)} speakers", flush=True)
    flush()

    for idx, row in enumerate(rows):
        if not speaker_embs[idx]:
            failed_speakers += 1
            continue
        emb = np.stack(speaker_embs[idx]).astype(np.float32)
        prior = np.asarray(speaker_priors[idx], dtype=np.float32)
        vec, meta = aggregate_speaker(row, emb, prior)
        vectors.append(vec)
        targets.append(row.height_cm)
        metadata.append(meta)

    if not vectors:
        raise RuntimeError(f"No usable speakers for split {name}")
    print(
        f"[phase9] {name}: speakers={len(vectors)} failed={failed_speakers} missing_audio={missing_audio}",
        flush=True,
    )
    return np.stack(vectors).astype(np.float32), np.asarray(targets, dtype=np.float32), metadata


def cache_path(output_dir: Path, args: argparse.Namespace) -> Path:
    celeb = "_celeb" if bool(getattr(args, "include_celeb_support", False)) else ""
    tag = f"ecapa_m{int(args.max_clips_per_speaker)}_s{str(args.max_seconds).replace('.', 'p')}_limit{int(args.limit_speakers)}{celeb}.npz"
    return output_dir / tag


def save_cache(path: Path, data: Mapping[str, Any]) -> None:
    payload = {}
    for split in ("train", "val", "test"):
        payload[f"{split}_x"] = data[split]["x"]
        payload[f"{split}_y"] = data[split]["y"]
        payload[f"{split}_meta_json"] = np.asarray(json.dumps(data[split]["meta"], ensure_ascii=False))
    np.savez_compressed(path, **payload)


def load_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with np.load(path, allow_pickle=True) as data:
        for split in ("train", "val", "test"):
            out[split] = {
                "x": np.asarray(data[f"{split}_x"], dtype=np.float32),
                "y": np.asarray(data[f"{split}_y"], dtype=np.float32),
                "meta": json.loads(str(np.asarray(data[f"{split}_meta_json"]).item())),
            }
    return out


def height_bin_scalar(height: float) -> int:
    if height < 160.0:
        return 0
    if height < 175.0:
        return 1
    return 2


def metrics_np(y: np.ndarray, pred: np.ndarray, meta: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    pred = np.asarray(pred, dtype=np.float32).reshape(-1)
    err = pred - y
    ae = np.abs(err)
    out: Dict[str, float] = {
        "mae": float(np.mean(ae)),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "median_ae": float(np.median(ae)),
        "p90_ae": float(np.quantile(ae, 0.90)),
        "bias": float(np.mean(err)),
        "within_3cm": float(np.mean(ae <= 3.0)),
        "within_5cm": float(np.mean(ae <= 5.0)),
        "count": float(len(y)),
    }
    for label, bin_id in (("short", 0), ("medium", 1), ("tall", 2)):
        mask = np.asarray([int(row.get("height_bin", height_bin_scalar(float(row.get("height_cm", 0.0))))) == bin_id for row in meta], dtype=bool)
        if mask.any():
            out[f"{label}_mae"] = float(np.mean(ae[mask]))
            out[f"{label}_n"] = float(mask.sum())
    for source in sorted({str(row.get("source", "UNKNOWN")) for row in meta}):
        mask = np.asarray([str(row.get("source", "UNKNOWN")) == source for row in meta], dtype=bool)
        if mask.any():
            out[f"source_{source.lower()}_mae"] = float(np.mean(ae[mask]))
            out[f"source_{source.lower()}_n"] = float(mask.sum())
    return out


def balanced_score(metrics: Mapping[str, float]) -> float:
    vals = [float(metrics.get("mae", 999.0))]
    bin_vals = [metrics[key] for key in ("short_mae", "medium_mae", "tall_mae") if key in metrics]
    if bin_vals:
        vals.append(float(np.mean(bin_vals)))
    vals.append(float(metrics.get("p90_ae", 999.0)) * 0.08)
    return float(np.mean(vals))


def robust_standardize(train_x: np.ndarray, *others: np.ndarray) -> Tuple[np.ndarray, ...]:
    train = np.asarray(train_x, dtype=np.float32)
    med = np.nanmedian(train, axis=0)
    q25 = np.nanpercentile(train, 25, axis=0)
    q75 = np.nanpercentile(train, 75, axis=0)
    scale = np.maximum(q75 - q25, 1e-4)
    fill = med

    def transform(x: np.ndarray) -> np.ndarray:
        z = np.asarray(x, dtype=np.float32).copy()
        inds = ~np.isfinite(z)
        if inds.any():
            z[inds] = np.take(fill, np.where(inds)[1])
        z = (z - med) / scale
        return np.clip(z, -8.0, 8.0).astype(np.float32)

    return (transform(train),) + tuple(transform(x) for x in others)


def weighted_ridge_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    test_x: np.ndarray,
    *,
    sample_weight: np.ndarray,
    lambdas: Sequence[float],
    device: torch.device,
    label: str,
) -> List[Dict[str, Any]]:
    x_train, x_val, x_test = robust_standardize(train_x, val_x, test_x)
    x_train = np.concatenate([x_train, np.ones((x_train.shape[0], 1), dtype=np.float32)], axis=1)
    x_val = np.concatenate([x_val, np.ones((x_val.shape[0], 1), dtype=np.float32)], axis=1)
    x_test = np.concatenate([x_test, np.ones((x_test.shape[0], 1), dtype=np.float32)], axis=1)

    xt = torch.tensor(x_train, dtype=torch.float32, device=device)
    xv = torch.tensor(x_val, dtype=torch.float32, device=device)
    xs = torch.tensor(x_test, dtype=torch.float32, device=device)
    y = torch.tensor(train_y, dtype=torch.float32, device=device)
    w = torch.tensor(sample_weight, dtype=torch.float32, device=device).clamp_min(1e-4)
    y_mean = (y * w).sum() / w.sum().clamp_min(1e-6)
    y_centered = y - y_mean
    sqrt_w = torch.sqrt(w)[:, None]
    xw = xt * sqrt_w
    yw = y_centered * sqrt_w.squeeze(1)
    eye = torch.eye(xw.shape[1], dtype=torch.float32, device=device)
    eye[-1, -1] = 0.0

    out: List[Dict[str, Any]] = []
    xtx = xw.T @ xw
    xty = xw.T @ yw
    for lam in lambdas:
        beta = torch.linalg.solve(xtx + float(lam) * eye, xty)
        val_pred = (xv @ beta + y_mean).detach().cpu().numpy().astype(np.float32)
        test_pred = (xs @ beta + y_mean).detach().cpu().numpy().astype(np.float32)
        out.append({"name": f"{label}_ridge_lam{lam:g}", "val_pred": val_pred, "test_pred": test_pred, "lambda": float(lam)})
    return out


def kernel_ridge_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    test_x: np.ndarray,
    *,
    lambdas: Sequence[float],
    gammas: Sequence[float],
    device: torch.device,
    label: str,
) -> List[Dict[str, Any]]:
    x_train, x_val, x_test = robust_standardize(train_x, val_x, test_x)
    xt = torch.tensor(x_train, dtype=torch.float32, device=device)
    xv = torch.tensor(x_val, dtype=torch.float32, device=device)
    xs = torch.tensor(x_test, dtype=torch.float32, device=device)
    y = torch.tensor(train_y, dtype=torch.float32, device=device)
    y_mean = y.mean()
    yc = y - y_mean
    d_train = torch.cdist(xt, xt).pow(2) / max(1, xt.shape[1])
    d_val = torch.cdist(xv, xt).pow(2) / max(1, xt.shape[1])
    d_test = torch.cdist(xs, xt).pow(2) / max(1, xt.shape[1])
    eye = torch.eye(xt.shape[0], dtype=torch.float32, device=device)
    out: List[Dict[str, Any]] = []
    for gamma in gammas:
        k_train = torch.exp(-float(gamma) * d_train)
        k_val = torch.exp(-float(gamma) * d_val)
        k_test = torch.exp(-float(gamma) * d_test)
        for lam in lambdas:
            alpha = torch.linalg.solve(k_train + float(lam) * eye, yc)
            val_pred = (k_val @ alpha + y_mean).detach().cpu().numpy().astype(np.float32)
            test_pred = (k_test @ alpha + y_mean).detach().cpu().numpy().astype(np.float32)
            out.append({"name": f"{label}_krr_g{gamma:g}_l{lam:g}", "val_pred": val_pred, "test_pred": test_pred, "gamma": float(gamma), "lambda": float(lam)})
    return out


def knn_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    test_x: np.ndarray,
    *,
    ks: Sequence[int],
    temps: Sequence[float],
    device: torch.device,
    label: str,
) -> List[Dict[str, Any]]:
    x_train, x_val, x_test = robust_standardize(train_x, val_x, test_x)
    xt = torch.tensor(x_train, dtype=torch.float32, device=device)
    xv = torch.tensor(x_val, dtype=torch.float32, device=device)
    xs = torch.tensor(x_test, dtype=torch.float32, device=device)
    y = torch.tensor(train_y, dtype=torch.float32, device=device)
    xt = torch.nn.functional.normalize(xt, dim=1)
    xv = torch.nn.functional.normalize(xv, dim=1)
    xs = torch.nn.functional.normalize(xs, dim=1)
    sim_val = xv @ xt.T
    sim_test = xs @ xt.T
    out: List[Dict[str, Any]] = []
    for k in ks:
        k_eff = min(int(k), train_x.shape[0])
        val_sim, val_idx = torch.topk(sim_val, k=k_eff, dim=1)
        test_sim, test_idx = torch.topk(sim_test, k=k_eff, dim=1)
        for temp in temps:
            vw = torch.softmax(val_sim / float(temp), dim=1)
            tw = torch.softmax(test_sim / float(temp), dim=1)
            val_pred = (y[val_idx] * vw).sum(dim=1).detach().cpu().numpy().astype(np.float32)
            test_pred = (y[test_idx] * tw).sum(dim=1).detach().cpu().numpy().astype(np.float32)
            out.append({"name": f"{label}_knn_k{k_eff}_t{temp:g}", "val_pred": val_pred, "test_pred": test_pred, "k": int(k_eff), "temp": float(temp)})
    return out


def read_prediction_csv(path: Path) -> Dict[str, float]:
    out: Dict[str, float] = {}
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sid = str(row.get("speaker_id", "")).strip()
            pred = safe_float(row.get("final_pred_cm"))
            if sid and math.isfinite(pred):
                out[sid] = pred
    return out


def aligned_phase3(meta: Sequence[Mapping[str, Any]], preds: Mapping[str, float]) -> Optional[np.ndarray]:
    vals = []
    for row in meta:
        sid = str(row.get("speaker_id", ""))
        if sid not in preds:
            return None
        vals.append(float(preds[sid]))
    return np.asarray(vals, dtype=np.float32)


def calibrate_affine(val_y: np.ndarray, val_pred: np.ndarray, test_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    x = np.asarray(val_pred, dtype=np.float32)
    y = np.asarray(val_y, dtype=np.float32)
    xm = float(x.mean())
    ym = float(y.mean())
    denom = float(np.sum((x - xm) ** 2)) + 1e-6
    slope = float(np.sum((x - xm) * (y - ym)) / denom)
    slope = float(np.clip(slope, 0.55, 1.45))
    intercept = float(ym - slope * xm)
    return (slope * val_pred + intercept).astype(np.float32), (slope * test_pred + intercept).astype(np.float32), {"slope": slope, "intercept": intercept}


def candidate_table(
    candidates: Sequence[Mapping[str, Any]],
    val_y: np.ndarray,
    test_y: np.ndarray,
    val_meta: Sequence[Mapping[str, Any]],
    test_meta: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for cand in candidates:
        val_m = metrics_np(val_y, np.asarray(cand["val_pred"], dtype=np.float32), val_meta)
        test_m = metrics_np(test_y, np.asarray(cand["test_pred"], dtype=np.float32), test_meta)
        row = dict(cand)
        row.pop("val_pred", None)
        row.pop("test_pred", None)
        row["val"] = val_m
        row["test"] = test_m
        row["score"] = balanced_score(val_m)
        rows.append(row)
    rows.sort(key=lambda item: float(item["score"]))
    return rows


def choose_blends(
    candidates: Sequence[Mapping[str, Any]],
    phase3_val: Optional[np.ndarray],
    phase3_test: Optional[np.ndarray],
    val_y: np.ndarray,
    val_meta: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    if phase3_val is None or phase3_test is None:
        return []
    out: List[Dict[str, Any]] = []
    weights = np.linspace(0.0, 1.0, num=51, dtype=np.float32)
    for cand in candidates:
        best = None
        for w in weights:
            val_pred = w * np.asarray(cand["val_pred"], dtype=np.float32) + (1.0 - w) * phase3_val
            score = balanced_score(metrics_np(val_y, val_pred, val_meta))
            if best is None or score < best["score"]:
                test_pred = w * np.asarray(cand["test_pred"], dtype=np.float32) + (1.0 - w) * phase3_test
                best = {
                    "name": f"blend_phase3_{cand['name']}_w{float(w):.2f}",
                    "val_pred": val_pred.astype(np.float32),
                    "test_pred": test_pred.astype(np.float32),
                    "score": float(score),
                    "blend_weight_new": float(w),
                    "base_candidate": cand["name"],
                }
        if best is not None:
            out.append(best)
    return out


def write_predictions(path: Path, y: np.ndarray, pred: np.ndarray, meta: Sequence[Mapping[str, Any]], extra: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["speaker_id", "source", "gender", "height_cm", "phase9_pred_cm", "phase9_abs_error_cm", "prior_mean_cm"]
    for name in extra:
        fields.append(name)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for idx, row in enumerate(meta):
            out = {
                "speaker_id": row["speaker_id"],
                "source": row["source"],
                "gender": row["gender"],
                "height_cm": f"{float(y[idx]):.6f}",
                "phase9_pred_cm": f"{float(pred[idx]):.6f}",
                "phase9_abs_error_cm": f"{abs(float(pred[idx]) - float(y[idx])):.6f}",
                "prior_mean_cm": f"{float(row.get('prior_mean', float('nan'))):.6f}",
            }
            for name, values in extra.items():
                out[name] = f"{float(values[idx]):.6f}"
            writer.writerow(out)


def write_report(
    output_dir: Path,
    selected: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    phase3_metrics: Optional[Dict[str, float]],
    cache_file: Path,
    args: argparse.Namespace,
) -> None:
    report_path = output_dir / "PHASE9_ECAPA_PRIOR_REPORT.md"
    lines = [
        "# Phase 9 ECAPA Prior Stack Report",
        "",
        "## Result",
        f"- Selected method: `{selected['name']}`",
        f"- Validation score: `{float(selected['score']):.3f}`",
        f"- Validation MAE: `{float(selected['val']['mae']):.3f}cm`",
        f"- Test MAE: `{float(selected['test']['mae']):.3f}cm`",
        f"- Test short MAE: `{float(selected['test'].get('short_mae', float('nan'))):.3f}cm`",
        f"- Target 3cm met: `{float(selected['test']['mae']) <= 3.0}`",
        "",
        "## Reference",
    ]
    if phase3_metrics is not None:
        lines.append(f"- Phase 3 test MAE: `{phase3_metrics['mae']:.3f}cm`, short `{phase3_metrics.get('short_mae', float('nan')):.3f}cm`")
    lines.extend(
        [
            f"- Cache: `{cache_file}`",
            f"- Max clips per speaker: `{int(args.max_clips_per_speaker)}`",
            f"- Max seconds per clip: `{float(args.max_seconds):.1f}`",
            "",
            "## Top Validation Candidates",
        ]
    )
    for row in rows[:20]:
        lines.append(
            f"- `{row['name']}`: val `{row['val']['mae']:.3f}cm`, "
            f"test `{row['test']['mae']:.3f}cm`, short `{row['test'].get('short_mae', float('nan')):.3f}cm`, "
            f"score `{row['score']:.3f}`"
        )
    lines.extend(["", "## Conclusion"])
    if phase3_metrics is not None and selected["test"]["mae"] < phase3_metrics["mae"]:
        lines.append("ECAPA plus the external height prior adds new sealed-test signal over Phase 3.")
    else:
        lines.append("ECAPA plus the external height prior did not beat the Phase 3 frontier on sealed test; it is useful evidence, but not a 3cm path by itself.")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[phase9] wrote {report_path}", flush=True)


def main() -> int:
    args = parse_args()
    seed_everything(int(args.seed))
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_path(output_dir, args)
    device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Phase9 is CUDA-only for this run, but CUDA is unavailable")
    print(f"[phase9] device={device} gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}", flush=True)

    if cache_file.exists() and not args.rebuild_cache:
        print(f"[phase9] loading cache {cache_file}", flush=True)
        data = load_cache(cache_file)
    else:
        splits_dir = resolve(args.splits_dir)
        split_rows = {
            "train": read_split_csv(splits_dir / "train_clean.csv", int(args.max_clips_per_speaker), int(args.limit_speakers)),
            "val": read_split_csv(splits_dir / "val_clean.csv", int(args.max_clips_per_speaker), int(args.limit_speakers)),
            "test": read_split_csv(splits_dir / "test_clean.csv", int(args.max_clips_per_speaker), int(args.limit_speakers)),
        }
        if args.include_celeb_support:
            split_rows["train"].extend(
                read_celeb_rows_from_features(
                    resolve(args.celeb_features_root),
                    int(args.max_clips_per_speaker),
                    max_speakers=int(args.max_celeb_speakers),
                )
            )
        print(
            "[phase9] split speakers "
            + " ".join(f"{name}={len(rows)}" for name, rows in split_rows.items()),
            flush=True,
        )
        extractor = EcapaPrior(str(device), args.svr_model_dir)
        print(f"[phase9] loaded ECAPA + external height SVR from {extractor.svr_path}", flush=True)
        data = {}
        for split, rows in split_rows.items():
            x, y, meta = build_split(split, rows, extractor, batch_size=int(args.batch_size), max_seconds=float(args.max_seconds))
            data[split] = {"x": x, "y": y, "meta": meta}
        save_cache(cache_file, data)
        print(f"[phase9] saved cache {cache_file}", flush=True)

    train_x = data["train"]["x"]
    train_y = data["train"]["y"]
    train_meta = data["train"]["meta"]
    val_x = data["val"]["x"]
    val_y = data["val"]["y"]
    val_meta = data["val"]["meta"]
    test_x = data["test"]["x"]
    test_y = data["test"]["y"]
    test_meta = data["test"]["meta"]

    train_source = np.asarray([source_id(row["source"], row["speaker_id"]) for row in train_meta], dtype=np.int64)
    target_mask = np.asarray([is_target_source(row["source"]) for row in train_meta], dtype=bool)
    target_boost = np.where(target_mask, 1.0, 0.25).astype(np.float32)
    female_boost = np.asarray([1.12 if int(row.get("gender", 0)) == 0 else 1.0 for row in train_meta], dtype=np.float32)
    short_train = np.asarray([1.35 if float(row.get("height_cm", 0.0)) < 160.0 else 1.0 for row in train_meta], dtype=np.float32)

    candidates: List[Dict[str, Any]] = []
    prior_val = np.asarray([row["prior_mean"] for row in val_meta], dtype=np.float32)
    prior_test = np.asarray([row["prior_mean"] for row in test_meta], dtype=np.float32)
    candidates.append({"name": "external_ecapa_svr_prior_raw", "val_pred": prior_val, "test_pred": prior_test})
    cal_val, cal_test, cal = calibrate_affine(val_y, prior_val, prior_test)
    candidates.append({"name": "external_ecapa_svr_prior_val_affine", "val_pred": cal_val, "test_pred": cal_test, "calibration": cal})

    lambdas = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
    candidates.extend(
        weighted_ridge_predict(
            train_x,
            train_y,
            val_x,
            test_x,
            sample_weight=np.ones_like(train_y, dtype=np.float32),
            lambdas=lambdas,
            device=device,
            label="all",
        )
    )
    candidates.extend(
        weighted_ridge_predict(
            train_x,
            train_y,
            val_x,
            test_x,
            sample_weight=target_boost * female_boost * short_train,
            lambdas=lambdas,
            device=device,
            label="target_weighted",
        )
    )
    if target_mask.sum() >= 100:
        candidates.extend(
            weighted_ridge_predict(
                train_x[target_mask],
                train_y[target_mask],
                val_x,
                test_x,
                sample_weight=np.ones(int(target_mask.sum()), dtype=np.float32),
                lambdas=lambdas,
                device=device,
                label="target_only",
            )
        )
        candidates.extend(
            kernel_ridge_predict(
                train_x[target_mask],
                train_y[target_mask],
                val_x,
                test_x,
                lambdas=[0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
                gammas=[0.2, 0.5, 1.0, 2.0],
                device=device,
                label="target_only",
            )
        )
    candidates.extend(
        knn_predict(
            train_x[target_mask] if target_mask.any() else train_x,
            train_y[target_mask] if target_mask.any() else train_y,
            val_x,
            test_x,
            ks=[5, 10, 20, 40, 80],
            temps=[0.03, 0.06, 0.12],
            device=device,
            label="target_only" if target_mask.any() else "all",
        )
    )

    phase3_val = aligned_phase3(val_meta, read_prediction_csv(resolve(args.phase3_val_pred))) if resolve(args.phase3_val_pred).exists() else None
    phase3_test = aligned_phase3(test_meta, read_prediction_csv(resolve(args.phase3_test_pred))) if resolve(args.phase3_test_pred).exists() else None
    phase3_metrics = None
    if phase3_val is not None and phase3_test is not None:
        candidates.append({"name": "phase3_frontier", "val_pred": phase3_val, "test_pred": phase3_test})
        candidates.extend(choose_blends(candidates[:], phase3_val, phase3_test, val_y, val_meta))
        phase3_metrics = metrics_np(test_y, phase3_test, test_meta)

    rows = candidate_table(candidates, val_y, test_y, val_meta, test_meta)
    selected = rows[0]
    selected_pred = np.asarray(next(c["test_pred"] for c in candidates if c["name"] == selected["name"]), dtype=np.float32)
    selected_val_pred = np.asarray(next(c["val_pred"] for c in candidates if c["name"] == selected["name"]), dtype=np.float32)
    selected["test"] = metrics_np(test_y, selected_pred, test_meta)
    selected["val"] = metrics_np(val_y, selected_val_pred, val_meta)

    extra = {"phase3_pred_cm": phase3_test} if phase3_test is not None else {}
    val_extra = {"phase3_pred_cm": phase3_val} if phase3_val is not None else {}
    write_predictions(output_dir / "phase9_predictions_val.csv", val_y, selected_val_pred, val_meta, val_extra)
    write_predictions(output_dir / "phase9_predictions_test.csv", test_y, selected_pred, test_meta, extra)

    report = {
        "selected": selected,
        "top_candidates": rows[:50],
        "phase3_test": phase3_metrics,
        "args": vars(args),
        "cache": str(cache_file),
    }
    (output_dir / "phase9_report.json").write_text(json.dumps(report, indent=2, allow_nan=True), encoding="utf-8")
    write_report(output_dir, selected, rows, phase3_metrics, cache_file, args)

    print("[phase9] selected", selected["name"], flush=True)
    print(
        f"[phase9] test_mae={selected['test']['mae']:.3f} short={selected['test'].get('short_mae', float('nan')):.3f} "
        f"within3={selected['test']['within_3cm']:.3f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
