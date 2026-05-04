#!/usr/bin/env python
"""Evaluate the public ECAPA+SVR height teacher on VocalMorph splits.

The teacher is `griko/height_reg_svr_ecapa_voxceleb`: SpeechBrain ECAPA
embeddings plus an SVR regressor. Calibration is fit on validation speakers
only; test labels are never used for fitting.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import joblib
import numpy as np
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from speechbrain.inference.speaker import EncoderClassifier

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ECAPA height teacher.")
    parser.add_argument("--train-csv", default="data/splits/train_clean.csv")
    parser.add_argument("--val-csv", default="data/splits/val_clean.csv")
    parser.add_argument("--test-csv", default="data/splits/test_clean.csv")
    parser.add_argument("--output-dir", default="outputs/diagnostics/ecapa_height_teacher")
    parser.add_argument("--max-clips-per-speaker", type=int, default=6)
    parser.add_argument("--max-duration", type=float, default=8.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--teacher-repo", default="griko/height_reg_svr_ecapa_voxceleb")
    parser.add_argument("--embedding-repo", default="speechbrain/spkrec-ecapa-voxceleb")
    return parser.parse_args()


def _resolve(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _audio_paths(row: Mapping[str, str], max_clips: int) -> List[Path]:
    paths = []
    for part in str(row.get("audio_paths", "") or "").split("|"):
        part = part.strip()
        if not part:
            continue
        path = _resolve(part)
        if path.is_file():
            paths.append(path)
        if max_clips > 0 and len(paths) >= max_clips:
            break
    return paths


def _safe_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _load_audio(path: Path, sample_rate: int, max_duration: float) -> torch.Tensor | None:
    try:
        wav, sr = torchaudio.load(str(path))
    except Exception:
        return None
    if wav.numel() == 0:
        return None
    wav = wav.float()
    if wav.ndim == 2 and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if wav.ndim == 2:
        wav = wav.squeeze(0)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    max_samples = int(float(max_duration) * sample_rate)
    if max_samples > 0 and wav.numel() > max_samples:
        wav = wav[:max_samples]
    if wav.numel() < int(0.5 * sample_rate):
        return None
    peak = wav.abs().max().clamp(min=1e-6)
    return (wav / peak).contiguous()


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(valid):
        return {"count": 0.0, "mae": float("nan"), "rmse": float("nan"), "median_ae": float("nan")}
    err = y_true[valid] - y_pred[valid]
    return {
        "count": float(valid.sum()),
        "mae": float(mean_absolute_error(y_true[valid], y_pred[valid])),
        "rmse": float(math.sqrt(mean_squared_error(y_true[valid], y_pred[valid]))),
        "median_ae": float(np.median(np.abs(err))),
    }


def _height_bin(value: float) -> str:
    if value < 160.0:
        return "short"
    if value >= 175.0:
        return "tall"
    return "medium"


def _slice_metrics(records: Sequence[Mapping[str, Any]], pred_key: str) -> Dict[str, Dict[str, float]]:
    y_true = np.asarray([float(row["height_true"]) for row in records], dtype=np.float32)
    y_pred = np.asarray([float(row[pred_key]) for row in records], dtype=np.float32)
    out: Dict[str, Dict[str, float]] = {}
    for group_name, labels in {
        "heightbin": [_height_bin(float(row["height_true"])) for row in records],
        "gender": [str(row.get("gender", "")).lower() for row in records],
        "source": [str(row.get("source", "")).upper() for row in records],
    }.items():
        for label in sorted(set(labels)):
            mask = np.asarray([item == label for item in labels], dtype=bool)
            out[f"{group_name}_{label}"] = _metrics(y_true[mask], y_pred[mask])
    return out


def _teacher_assets(repo: str) -> Tuple[Any, Any]:
    scaler_path = hf_hub_download(repo, "scaler.joblib")
    model_path = hf_hub_download(repo, "svr_model.joblib")
    return joblib.load(scaler_path), joblib.load(model_path)


@torch.no_grad()
def _speaker_prediction(
    row: Mapping[str, str],
    *,
    classifier: EncoderClassifier,
    scaler: Any,
    svr: Any,
    device: torch.device,
    sample_rate: int,
    max_clips: int,
    max_duration: float,
) -> Dict[str, Any] | None:
    paths = _audio_paths(row, max_clips)
    clip_preds: List[float] = []
    clip_embs: List[np.ndarray] = []
    used_paths: List[str] = []
    for path in paths:
        wav = _load_audio(path, sample_rate, max_duration)
        if wav is None:
            continue
        wav = wav.unsqueeze(0).to(device)
        emb = classifier.encode_batch(wav).detach().cpu().numpy().reshape(-1)
        if emb.size == 0:
            continue
        pred = float(svr.predict(scaler.transform(emb.reshape(1, -1)))[0])
        clip_preds.append(pred)
        clip_embs.append(emb.astype(np.float32))
        used_paths.append(str(path))

    if not clip_preds:
        return None
    emb_mean = np.stack(clip_embs).mean(axis=0)
    pred_from_mean_embedding = float(svr.predict(scaler.transform(emb_mean.reshape(1, -1)))[0])
    return {
        "speaker_id": str(row.get("speaker_id", "")),
        "source": str(row.get("source", "")),
        "gender": str(row.get("gender", "")),
        "height_true": _safe_float(row.get("height_cm")),
        "n_used_clips": len(clip_preds),
        "pred_clip_mean": float(np.mean(clip_preds)),
        "pred_clip_median": float(np.median(clip_preds)),
        "pred_clip_std": float(np.std(clip_preds)),
        "pred_embedding_mean": pred_from_mean_embedding,
        "used_audio_paths": "|".join(used_paths),
    }


def _collect_split(
    split_name: str,
    rows: Sequence[Mapping[str, str]],
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    records = []
    for idx, row in enumerate(rows, start=1):
        rec = _speaker_prediction(row, **kwargs)
        if rec is not None and math.isfinite(float(rec["height_true"])):
            rec["split"] = split_name
            records.append(rec)
        if idx % 25 == 0:
            print(f"[ecapa-teacher] {split_name}: {idx}/{len(rows)} rows, kept={len(records)}")
    print(f"[ecapa-teacher] {split_name}: kept={len(records)} / {len(rows)}")
    return records


def _fit_calibrators(val_records: Sequence[Mapping[str, Any]], test_records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    y_val = np.asarray([float(row["height_true"]) for row in val_records], dtype=np.float32)
    y_test = np.asarray([float(row["height_true"]) for row in test_records], dtype=np.float32)
    results: List[Dict[str, Any]] = []
    for base_key in ("pred_clip_mean", "pred_clip_median", "pred_embedding_mean"):
        x_val = np.asarray([[float(row[base_key]), float(row.get("pred_clip_std", 0.0)), float(row.get("n_used_clips", 1.0))] for row in val_records], dtype=np.float32)
        x_test = np.asarray([[float(row[base_key]), float(row.get("pred_clip_std", 0.0)), float(row.get("n_used_clips", 1.0))] for row in test_records], dtype=np.float32)
        raw_pred = x_test[:, 0]
        results.append(
            {
                "name": base_key,
                "fit": "raw_teacher",
                "val": _metrics(y_val, x_val[:, 0]),
                "test": _metrics(y_test, raw_pred),
                "test_slices": _slice_metrics(test_records, base_key),
            }
        )
        calibrators = {
            f"val_linear_{base_key}": Pipeline([("impute", SimpleImputer(strategy="median")), ("model", LinearRegression())]),
            f"val_ridge_{base_key}": Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", RidgeCV(alphas=np.logspace(-3, 3, 13)))]),
            f"val_huber_{base_key}": Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", HuberRegressor(alpha=0.03, epsilon=1.35, max_iter=1000))]),
        }
        for name, model in calibrators.items():
            try:
                model.fit(x_val, y_val)
                pred_val = model.predict(x_val).astype(np.float32)
                pred_test = model.predict(x_test).astype(np.float32)
                for row, pred in zip(test_records, pred_test):
                    row[name] = float(pred)
                results.append(
                    {
                        "name": name,
                        "fit": "validation_only",
                        "val": _metrics(y_val, pred_val),
                        "test": _metrics(y_test, pred_test),
                        "test_slices": _slice_metrics(test_records, name),
                    }
                )
            except Exception as exc:
                results.append({"name": name, "error": str(exc)})
    results.sort(key=lambda row: row.get("val", {}).get("mae", float("inf")))
    return results


def _write_records(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    if not records:
        return
    fields = sorted({key for row in records for key in row.keys() if key != "used_audio_paths"})
    fields.append("used_audio_paths")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow(row)


def main() -> int:
    args = parse_args()
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    print(f"[ecapa-teacher] device={device}")

    scaler, svr = _teacher_assets(str(args.teacher_repo))
    classifier = EncoderClassifier.from_hparams(
        source=str(args.embedding_repo),
        savedir=str(output_dir / "speechbrain_ecapa"),
        run_opts={"device": str(device)},
    )
    classifier.eval()

    kwargs = {
        "classifier": classifier,
        "scaler": scaler,
        "svr": svr,
        "device": device,
        "sample_rate": 16000,
        "max_clips": int(args.max_clips_per_speaker),
        "max_duration": float(args.max_duration),
    }
    val_records = _collect_split("val", _read_rows(_resolve(args.val_csv)), **kwargs)
    test_records = _collect_split("test", _read_rows(_resolve(args.test_csv)), **kwargs)
    results = _fit_calibrators(val_records, test_records)

    summary = {
        "teacher_repo": str(args.teacher_repo),
        "embedding_repo": str(args.embedding_repo),
        "device": str(device),
        "max_clips_per_speaker": int(args.max_clips_per_speaker),
        "max_duration": float(args.max_duration),
        "val_speakers": len(val_records),
        "test_speakers": len(test_records),
        "selected_by_val": results[0] if results else {},
        "best_test_report_only_do_not_select_on_this": min(
            results,
            key=lambda row: row.get("test", {}).get("mae", float("inf")),
        )
        if results
        else {},
        "results": results,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
    _write_records(output_dir / "val_teacher_predictions.csv", val_records)
    _write_records(output_dir / "test_teacher_predictions.csv", test_records)
    best = summary["selected_by_val"]
    print(
        "[ecapa-teacher] selected by val: "
        f"{best.get('name')} val={best.get('val', {}).get('mae', float('nan')):.3f} "
        f"test={best.get('test', {}).get('mae', float('nan')):.3f}"
    )
    print(f"[ecapa-teacher] wrote {output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
