#!/usr/bin/env python
"""Fuse pretrained speech embeddings into existing VocalMorph feature NPZs.

The current 136-dim handcrafted sequence is hitting a ceiling. This script keeps
those features intact and appends an utterance-level SSL embedding to every frame:

    fused_sequence[t] = [handcrafted_sequence[t], ssl_embedding]

That lets the existing VocalMorphV2 model consume stronger speaker information
without rewriting the trainer. The script requires the original audio paths
stored in each NPZ under ``audio_rel_path``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np
import torch


os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SSL-fused VocalMorph features.")
    parser.add_argument("--input-root", default="data/features_vtl_fixed")
    parser.add_argument("--output-root", default="data/features_vtl_ssl")
    parser.add_argument("--model-name", default="microsoft/wavlm-base-plus")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--max-seconds", type=float, default=10.0)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--projection-seed", type=int, default=1337)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--limit", type=int, default=0, help="Debug limit per split.")
    parser.add_argument(
        "--skip-augmented",
        action="store_true",
        help="Skip *_aug*.npz files so SSL training starts from original clips only.",
    )
    parser.add_argument(
        "--allow-missing-audio",
        action="store_true",
        help="Copy original features when audio is missing instead of failing.",
    )
    return parser.parse_args()


def _resolve(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _decode_np_value(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return ""
        if value.ndim == 0:
            return str(value.item())
        return str(value[0])
    return str(value)


def _audio_path_from_npz(npz: Mapping[str, Any]) -> Optional[Path]:
    if "audio_rel_path" not in npz:
        return None
    rel = _decode_np_value(npz["audio_rel_path"]).strip()
    if not rel:
        return None
    path = Path(rel)
    return path if path.is_absolute() else ROOT / path


def _load_audio(path: Path, sample_rate: int, max_seconds: float) -> torch.Tensor:
    import torchaudio

    wav, sr = torchaudio.load(str(path))
    if wav.ndim == 2:
        wav = wav.mean(dim=0)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    max_samples = int(max(1.0, max_seconds) * sample_rate)
    if wav.numel() > max_samples:
        wav = wav[:max_samples]
    return wav.contiguous()


def _build_projection(input_dim: int, output_dim: int, seed: int) -> Optional[np.ndarray]:
    if output_dim <= 0 or output_dim >= input_dim:
        return None
    rng = np.random.default_rng(seed)
    matrix = rng.standard_normal((input_dim, output_dim), dtype=np.float32)
    matrix /= math.sqrt(float(output_dim))
    return matrix.astype(np.float32)


class SslEmbedder:
    def __init__(
        self,
        *,
        model_name: str,
        sample_rate: int,
        max_seconds: float,
        projection_dim: int,
        projection_seed: int,
        device: str,
    ) -> None:
        from transformers import AutoFeatureExtractor, AutoModel

        self.model_name = model_name
        self.sample_rate = int(sample_rate)
        self.max_seconds = float(max_seconds)
        self.device = torch.device(device)
        self.extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        hidden_size = int(getattr(self.model.config, "hidden_size", 0) or 0)
        if hidden_size <= 0:
            raise ValueError(f"Could not determine hidden size for {model_name}")
        self.projection = _build_projection(hidden_size, projection_dim, projection_seed)
        self.output_dim = int(projection_dim if self.projection is not None else hidden_size)

    @torch.no_grad()
    def embed_many(self, audio_paths: Iterable[Path]) -> list[np.ndarray]:
        wavs = [
            _load_audio(audio_path, self.sample_rate, self.max_seconds).numpy()
            for audio_path in audio_paths
        ]
        inputs = self.extractor(
            wavs,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        output = self.model(**inputs).last_hidden_state
        mask = inputs.get("attention_mask")
        if mask is not None:
            frame_mask = torch.nn.functional.interpolate(
                mask[:, None].float(),
                size=output.shape[1],
                mode="nearest",
            ).squeeze(1)
            pooled = (output * frame_mask[..., None]).sum(dim=1) / frame_mask.sum(
                dim=1, keepdim=True
            ).clamp(min=1.0)
        else:
            pooled = output.mean(dim=1)
        embeddings = pooled.detach().cpu().float().numpy().astype(np.float32)
        if self.projection is not None:
            embeddings = embeddings @ self.projection
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-6)
        return [row.astype(np.float32) for row in embeddings]

    @torch.no_grad()
    def embed(self, audio_path: Path) -> np.ndarray:
        return self.embed_many([audio_path])[0]


def _copy_root_metadata(input_root: Path, output_root: Path, args: argparse.Namespace) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    for name in (
        "target_stats.json",
        "feature_contract.json",
        "feature_diagnostics.json",
        "build_manifest.json",
        "vtl_repair_summary.json",
    ):
        src = input_root / name
        if src.exists():
            shutil.copy2(src, output_root / name)
    manifest = {
        "ssl_fusion": True,
        "model_name": args.model_name,
        "projection_dim": int(args.projection_dim),
        "projection_seed": int(args.projection_seed),
        "sample_rate": int(args.sample_rate),
        "max_seconds": float(args.max_seconds),
        "source_feature_root": str(input_root),
    }
    with open(output_root / "ssl_fusion_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def _write_fused_npz(
    *,
    input_path: Path,
    output_path: Path,
    ssl_embedding: Optional[np.ndarray],
    args: argparse.Namespace,
) -> None:
    with np.load(input_path, allow_pickle=True) as data:
        payload: Dict[str, Any] = {key: data[key] for key in data.files}
        sequence = np.asarray(data["sequence"], dtype=np.float32)
        if ssl_embedding is not None:
            repeated = np.repeat(ssl_embedding.reshape(1, -1), sequence.shape[0], axis=0)
            payload["sequence"] = np.concatenate([sequence, repeated], axis=1).astype(np.float32)
            payload["ssl_embedding"] = ssl_embedding.astype(np.float32)
            payload["ssl_embedding_dim"] = np.asarray([ssl_embedding.shape[0]], dtype=np.int32)
            payload["ssl_model_name"] = np.asarray(args.model_name)
            payload["ssl_fused_tag"] = np.asarray("ssl_fused_v1")
        else:
            payload["sequence"] = sequence
            payload["ssl_fused_tag"] = np.asarray("ssl_missing_audio_original_sequence")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, **payload)


def iter_npz_files(split_dir: Path, limit: int) -> Iterable[Path]:
    files = sorted(split_dir.glob("*.npz"))
    if limit > 0:
        files = files[:limit]
    return files


def _is_augmented_feature(path: Path) -> bool:
    return "_aug" in path.stem.lower()


def main() -> int:
    args = parse_args()
    input_root = _resolve(args.input_root)
    output_root = _resolve(args.output_root)
    if not input_root.exists():
        raise FileNotFoundError(input_root)

    splits = [part.strip() for part in str(args.splits).split(",") if part.strip()]
    _copy_root_metadata(input_root, output_root, args)

    embedder: Optional[SslEmbedder] = None
    cache: Dict[str, np.ndarray] = {}
    pending: list[tuple[Path, Path, Path, str]] = []
    batch_size = max(1, int(args.batch_size))
    missing_audio = 0
    processed = 0

    def flush_pending() -> None:
        nonlocal embedder, processed
        if not pending:
            return
        if embedder is None:
            embedder = SslEmbedder(
                model_name=args.model_name,
                sample_rate=args.sample_rate,
                max_seconds=args.max_seconds,
                projection_dim=args.projection_dim,
                projection_seed=args.projection_seed,
                device=args.device,
            )
        unique: Dict[str, Path] = {}
        for _input_path, _output_path, audio_path, cache_key in pending:
            if cache_key not in cache:
                unique.setdefault(cache_key, audio_path)
        if unique:
            embeddings = embedder.embed_many(unique.values())
            for cache_key, embedding in zip(unique.keys(), embeddings):
                cache[cache_key] = embedding
        for input_path, output_path, _audio_path, cache_key in pending:
            _write_fused_npz(
                input_path=input_path,
                output_path=output_path,
                ssl_embedding=cache[cache_key],
                args=args,
            )
            processed += 1
            if processed % 100 == 0:
                print(f"[SSL Fusion] processed={processed} cached_audio={len(cache)}")
        pending.clear()

    for split in splits:
        split_input = input_root / split
        if not split_input.exists():
            raise FileNotFoundError(split_input)
        for input_path in iter_npz_files(split_input, int(args.limit)):
            if args.skip_augmented and _is_augmented_feature(input_path):
                continue
            rel = input_path.relative_to(input_root)
            output_path = output_root / rel
            with np.load(input_path, allow_pickle=True) as data:
                audio_path = _audio_path_from_npz(data)
            if audio_path is None or not audio_path.exists():
                missing_audio += 1
                if not args.allow_missing_audio:
                    raise FileNotFoundError(
                        f"Missing audio for {input_path}: {audio_path}. "
                        "Restore data/audio_clean or rerun with --allow-missing-audio."
                    )
                _write_fused_npz(
                    input_path=input_path,
                    output_path=output_path,
                    ssl_embedding=None,
                    args=args,
                )
                continue

            cache_key = str(audio_path.resolve()).lower()
            if cache_key in cache:
                _write_fused_npz(
                    input_path=input_path,
                    output_path=output_path,
                    ssl_embedding=cache[cache_key],
                    args=args,
                )
                processed += 1
                if processed % 100 == 0:
                    print(f"[SSL Fusion] processed={processed} cached_audio={len(cache)}")
                continue
            pending.append((input_path, output_path, audio_path, cache_key))
            if len(pending) >= batch_size:
                flush_pending()

    flush_pending()

    print(
        f"[SSL Fusion] done processed={processed} missing_audio={missing_audio} "
        f"output={output_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
