#!/usr/bin/env python
"""Phase 5 label/domain audit for the 3cm path.

After the model zoo plateaued around 5.83cm, the next defensible path is not a
larger network. It is finding label, identity, and split-domain problems that
hide the remaining height signal. This script builds a concrete correction
queue from the canonical split CSVs and the best speaker-level predictions.

It uses CUDA for feature-space nearest-neighbor diagnostics when the speaker
cache is available. File existence checks are normal Windows disk I/O.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 5 height-label/domain audit.")
    parser.add_argument("--train-csv", default="data/splits/train_clean.csv")
    parser.add_argument("--val-csv", default="data/splits/val_clean.csv")
    parser.add_argument("--test-csv", default="data/splits/test_clean.csv")
    parser.add_argument("--speaker-cache", default="outputs/speaker_gpu_combo_full_ssl_cuda/speaker_gpu_cache.pt")
    parser.add_argument("--pred-val", default="outputs/phase3_target_domain_rescue/phase3_predictions_val.csv")
    parser.add_argument("--pred-test", default="outputs/phase3_target_domain_rescue/phase3_predictions_test.csv")
    parser.add_argument("--pred-val-column", default="final_pred_cm")
    parser.add_argument("--pred-test-column", default="final_pred_cm")
    parser.add_argument("--output-dir", default="outputs/phase5_label_domain_audit")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--review-top-k", type=int, default=80)
    parser.add_argument("--audio-path-check-limit", type=int, default=0, help="0 checks every path.")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def as_float(value: Any, default: float = float("nan")) -> float:
    try:
        text = str(value).strip()
        if not text:
            return default
        return float(text)
    except Exception:
        return default


def split_audio_paths(value: str) -> List[str]:
    return [part.strip() for part in str(value or "").split("|") if part.strip()]


def height_bin(height_cm: float) -> str:
    if height_cm < 160.0:
        return "short"
    if height_cm < 175.0:
        return "medium"
    return "tall"


def source_id(source: str) -> int:
    value = str(source or "").upper()
    if value == "TIMIT":
        return 0
    if value == "NISP":
        return 1
    if value in {"CELEB", "VOXCELEB"}:
        return 2
    return 3


def row_key(split: str, speaker_id: str) -> str:
    return f"{split}:{speaker_id}"


def summarize_split(split: str, rows: Sequence[Mapping[str, str]]) -> Dict[str, Any]:
    heights = [as_float(row.get("height_cm")) for row in rows]
    finite = np.asarray([h for h in heights if math.isfinite(h)], dtype=np.float32)
    return {
        "split": split,
        "speakers": len(rows),
        "source_counts": dict(Counter(str(row.get("source", "UNKNOWN")).upper() for row in rows)),
        "gender_counts": dict(Counter(str(row.get("gender", "UNKNOWN")).strip().lower() for row in rows)),
        "height_bin_counts": dict(Counter(height_bin(as_float(row.get("height_cm"))) for row in rows if math.isfinite(as_float(row.get("height_cm"))))),
        "height_mean": float(finite.mean()) if finite.size else float("nan"),
        "height_std": float(finite.std()) if finite.size else float("nan"),
        "height_min": float(finite.min()) if finite.size else float("nan"),
        "height_max": float(finite.max()) if finite.size else float("nan"),
    }


def path_audit(split: str, rows: Sequence[Mapping[str, str]], limit: int) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    checked = 0
    missing_rows: List[Dict[str, Any]] = []
    per_speaker_missing = Counter()
    total_paths = 0
    for row in rows:
        paths = split_audio_paths(str(row.get("audio_paths", "")))
        total_paths += len(paths)
        for rel in paths:
            if limit > 0 and checked >= limit:
                break
            checked += 1
            full = resolve(rel)
            if not full.exists():
                per_speaker_missing[str(row.get("speaker_id", ""))] += 1
                if len(missing_rows) < 5000:
                    missing_rows.append(
                        {
                            "split": split,
                            "speaker_id": row.get("speaker_id", ""),
                            "source": row.get("source", ""),
                            "height_cm": row.get("height_cm", ""),
                            "missing_audio_path": rel,
                        }
                    )
        if limit > 0 and checked >= limit:
            break
    return (
        {
            "split": split,
            "speaker_count": len(rows),
            "total_audio_paths_listed": total_paths,
            "paths_checked": checked,
            "missing_paths": int(sum(per_speaker_missing.values())),
            "speakers_with_missing_paths": int(len(per_speaker_missing)),
        },
        missing_rows,
    )


def read_predictions(path: Path, pred_column: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return out
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sid = str(row.get("speaker_id", "")).strip()
            if not sid:
                continue
            pred_value = row.get(pred_column)
            if pred_value is None:
                pred_value = row.get("final_pred_cm") or row.get("phase4_pred_cm") or row.get("pred_cm")
            out[sid] = {
                "pred_cm": as_float(pred_value),
                "abs_error_cm": as_float(row.get("final_abs_error_cm", row.get("phase4_abs_error_cm", row.get("abs_error_cm", "")))),
            }
    return out


def parse_timit_height(text: str) -> float:
    value = str(text or "").strip().replace(" ", "")
    if "'" not in value:
        return float("nan")
    feet_text, rest = value.split("'", 1)
    inch_text = rest.replace('"', "")
    try:
        feet = int(feet_text)
        inches = int(inch_text)
    except Exception:
        return float("nan")
    return float((feet * 12 + inches) * 2.54)


def load_timit_spkrinfo() -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for path in sorted((ROOT / "data" / "audio_clean").glob("*/data/TIMIT/DOC/SPKRINFO.TXT")):
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for raw in handle:
                line = raw.rstrip("\n")
                if not line or line.lstrip().startswith(";"):
                    continue
                parts = line.split()
                if len(parts) < 7:
                    continue
                sid = parts[0].upper()
                height_raw = parts[6]
                height_cm = parse_timit_height(height_raw)
                if math.isfinite(height_cm):
                    out[sid] = {
                        "timit_height_raw": height_raw,
                        "timit_height_cm": height_cm,
                        "timit_spkrinfo_path": str(path),
                    }
    return out


def robust_group_stats(rows: Sequence[Mapping[str, str]]) -> Dict[Tuple[str, str], Dict[str, float]]:
    buckets: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for row in rows:
        h = as_float(row.get("height_cm"))
        if math.isfinite(h):
            buckets[(str(row.get("source", "")).upper(), str(row.get("gender", "")).strip().lower())].append(h)
    stats = {}
    for key, values in buckets.items():
        arr = np.asarray(values, dtype=np.float32)
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)))
        stats[key] = {"median": med, "mad": max(mad, 1.0), "count": float(arr.size)}
    return stats


def nearest_train_support(
    cache_path: Path,
    device: torch.device,
) -> Dict[str, Dict[str, Any]]:
    if not cache_path.exists():
        return {}
    payload = torch.load(cache_path, map_location=device, weights_only=False)
    for split in ("train", "val", "test"):
        payload[split]["x"] = payload[split]["x"].to(device).float()
        payload[split]["y"] = payload[split]["y"].to(device).float()
    train_x = payload["train"]["x"]
    center = torch.quantile(train_x, 0.50, dim=0)
    q25 = torch.quantile(train_x, 0.25, dim=0)
    q75 = torch.quantile(train_x, 0.75, dim=0)
    scale = (q75 - q25).clamp_min(1e-3)
    train_z = F.normalize(torch.nan_to_num((train_x - center) / scale).clamp(-8, 8), dim=1)
    train_meta = payload["train"]["metadata"]
    support: Dict[str, Dict[str, Any]] = {}
    for split in ("val", "test"):
        q = F.normalize(torch.nan_to_num((payload[split]["x"] - center) / scale).clamp(-8, 8), dim=1)
        sim, idx = torch.max(q @ train_z.T, dim=1)
        for i, meta in enumerate(payload[split]["metadata"]):
            nrow = train_meta[int(idx[i].item())]
            support[str(meta["speaker_id"])] = {
                "nearest_train_speaker": str(nrow.get("speaker_id", "")),
                "nearest_train_source": str(nrow.get("source", "")),
                "nearest_train_height_cm": float(nrow.get("height_cm", float("nan"))),
                "nearest_train_similarity": float(sim[i].item()),
                "nearest_height_delta_cm": float(abs(float(meta.get("height_cm", float("nan"))) - float(nrow.get("height_cm", float("nan"))))),
            }
    return support


def make_review_queue(
    split_name: str,
    rows: Sequence[Mapping[str, str]],
    predictions: Mapping[str, Mapping[str, Any]],
    group_stats: Mapping[Tuple[str, str], Mapping[str, float]],
    support: Mapping[str, Mapping[str, Any]],
    timit_info: Mapping[str, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    queue = []
    for row in rows:
        sid = str(row.get("speaker_id", "")).strip()
        h = as_float(row.get("height_cm"))
        pred = predictions.get(sid, {})
        pred_cm = as_float(pred.get("pred_cm"))
        err = abs(pred_cm - h) if math.isfinite(pred_cm) and math.isfinite(h) else float("nan")
        key = (str(row.get("source", "")).upper(), str(row.get("gender", "")).strip().lower())
        stat = group_stats.get(key, {"median": float("nan"), "mad": 1.0, "count": 0.0})
        z = abs(h - float(stat["median"])) / max(float(stat["mad"]), 1.0) if math.isfinite(h) and math.isfinite(float(stat["median"])) else 0.0
        paths = split_audio_paths(str(row.get("audio_paths", "")))
        support_row = support.get(sid, {})
        timit_sid = sid.replace("TIMIT_", "").upper() if sid.upper().startswith("TIMIT_") else sid.upper()
        timit_row = timit_info.get(timit_sid, {})
        timit_h = as_float(timit_row.get("timit_height_cm"))
        timit_delta = abs(timit_h - h) if math.isfinite(timit_h) and math.isfinite(h) else float("nan")
        nearest_delta = as_float(support_row.get("nearest_height_delta_cm"), 0.0)
        score = 0.0
        score += 3.0 * max(0.0, err - 6.0) if math.isfinite(err) else 0.0
        score += 1.5 * max(0.0, nearest_delta - 12.0)
        score += 4.0 * max(0.0, z - 3.0)
        score += 5.0 if len(paths) < 3 else 0.0
        score += 8.0 if h < 145.0 or h > 205.0 else 0.0
        action = "verify_height_label"
        if len(paths) < 3:
            action = "verify_audio_count_or_paths"
        if z >= 4.0:
            action = "verify_height_outlier"
        if math.isfinite(err) and err >= 10.0:
            action = "verify_height_and_identity"
        queue.append(
            {
                "priority_score": f"{score:.3f}",
                "split": split_name,
                "speaker_id": sid,
                "source": row.get("source", ""),
                "gender": row.get("gender", ""),
                "height_cm": f"{h:.3f}" if math.isfinite(h) else "",
                "height_bin": height_bin(h) if math.isfinite(h) else "",
                "best_pred_cm": f"{pred_cm:.3f}" if math.isfinite(pred_cm) else "",
                "abs_error_cm": f"{err:.3f}" if math.isfinite(err) else "",
                "group_median_cm": f"{float(stat['median']):.3f}" if math.isfinite(float(stat["median"])) else "",
                "group_mad_cm": f"{float(stat['mad']):.3f}",
                "group_outlier_mad_z": f"{z:.3f}",
                "n_audio_paths": len(paths),
                "nearest_train_speaker": support_row.get("nearest_train_speaker", ""),
                "nearest_train_source": support_row.get("nearest_train_source", ""),
                "nearest_train_height_cm": f"{as_float(support_row.get('nearest_train_height_cm')):.3f}" if support_row else "",
                "nearest_train_similarity": f"{as_float(support_row.get('nearest_train_similarity')):.5f}" if support_row else "",
                "nearest_height_delta_cm": f"{nearest_delta:.3f}" if support_row else "",
                "timit_official_height_cm": f"{timit_h:.3f}" if math.isfinite(timit_h) else "",
                "timit_height_delta_cm": f"{timit_delta:.3f}" if math.isfinite(timit_delta) else "",
                "timit_height_verified": "yes" if math.isfinite(timit_delta) and timit_delta < 0.75 else ("mismatch" if math.isfinite(timit_delta) else ""),
                "suggested_action": action,
                "first_audio_path": paths[0] if paths else "",
            }
        )
    queue.sort(key=lambda r: float(r["priority_score"]), reverse=True)
    return queue


def write_markdown(path: Path, report: Mapping[str, Any]) -> None:
    lines = [
        "# Phase 5 Label/Domain Audit Report",
        "",
        "## Result",
        f"- Review rows written: `{report['review_rows_written']}`",
        f"- Missing audio paths found: `{report['missing_audio_paths']}`",
        f"- Correction template: `{report['correction_template']}`",
        "",
        "## Split Summary",
    ]
    for split, summary in report["split_summary"].items():
        lines.append(
            f"- `{split}`: speakers `{summary['speakers']}`, sources `{summary['source_counts']}`, bins `{summary['height_bin_counts']}`"
        )
    lines.extend(["", "## Audio Path Audit"])
    for split, row in report["path_audit"].items():
        lines.append(
            f"- `{split}`: checked `{row['paths_checked']}` / listed `{row['total_audio_paths_listed']}`, missing `{row['missing_paths']}`"
        )
    lines.extend(["", "## Top Review Targets"])
    for row in report["top_review_rows"][:20]:
        lines.append(
            f"- `{row['split']}:{row['speaker_id']}` source `{row['source']}` height `{row['height_cm']}` "
            f"pred `{row['best_pred_cm']}` err `{row['abs_error_cm']}` action `{row['suggested_action']}`"
        )
    lines.extend(
        [
            "",
            "## How This Helps 3cm",
            "The current model frontier is not limited by GPU strength anymore. This audit creates the label-repair queue needed to build a cleaner training/evaluation set. After verified corrections are filled into the template, run the correction script and rebuild features/training from those corrected splits.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if str(args.device).lower() == "cuda" and torch.cuda.is_available() else "cpu")
    if str(args.device).lower() == "cuda" and device.type != "cuda":
        raise SystemExit("CUDA requested but unavailable.")

    splits = {
        "train": read_csv_rows(resolve(args.train_csv)),
        "val": read_csv_rows(resolve(args.val_csv)),
        "test": read_csv_rows(resolve(args.test_csv)),
    }
    split_summary = {name: summarize_split(name, rows) for name, rows in splits.items()}
    path_summaries = {}
    missing_all = []
    for split, rows in splits.items():
        summary, missing = path_audit(split, rows, int(args.audio_path_check_limit))
        path_summaries[split] = summary
        missing_all.extend(missing)

    group_stats = robust_group_stats([*splits["train"], *splits["val"], *splits["test"]])
    val_pred = read_predictions(resolve(args.pred_val), str(args.pred_val_column))
    test_pred = read_predictions(resolve(args.pred_test), str(args.pred_test_column))
    support = nearest_train_support(resolve(args.speaker_cache), device)
    timit_info = load_timit_spkrinfo()
    queue = [
        *make_review_queue("val", splits["val"], val_pred, group_stats, support, timit_info),
        *make_review_queue("test", splits["test"], test_pred, group_stats, support, timit_info),
    ]
    queue.sort(key=lambda r: float(r["priority_score"]), reverse=True)
    review_rows = queue[: max(1, int(args.review_top_k))]

    review_fields = [
        "priority_score",
        "split",
        "speaker_id",
        "source",
        "gender",
        "height_cm",
        "height_bin",
        "best_pred_cm",
        "abs_error_cm",
        "group_median_cm",
        "group_mad_cm",
        "group_outlier_mad_z",
        "n_audio_paths",
        "nearest_train_speaker",
        "nearest_train_source",
        "nearest_train_height_cm",
        "nearest_train_similarity",
        "nearest_height_delta_cm",
        "timit_official_height_cm",
        "timit_height_delta_cm",
        "timit_height_verified",
        "suggested_action",
        "first_audio_path",
    ]
    write_csv(output_dir / "height_label_review_queue.csv", review_rows, review_fields)
    write_csv(output_dir / "missing_audio_paths.csv", missing_all, ["split", "speaker_id", "source", "height_cm", "missing_audio_path"])
    correction_rows = [
        {
            "split": row["split"],
            "speaker_id": row["speaker_id"],
            "current_height_cm": row["height_cm"],
            "corrected_height_cm": "",
            "decision": "",
            "notes": "",
        }
        for row in review_rows
    ]
    correction_template = output_dir / "height_label_corrections_template.csv"
    write_csv(correction_template, correction_rows, ["split", "speaker_id", "current_height_cm", "corrected_height_cm", "decision", "notes"])

    report = {
        "phase": "phase5_label_domain_audit",
        "device_for_nn_diagnostics": str(device),
        "split_summary": split_summary,
        "path_audit": path_summaries,
        "missing_audio_paths": len(missing_all),
        "timit_spkrinfo_entries": len(timit_info),
        "review_rows_written": len(review_rows),
        "correction_template": str(correction_template),
        "top_review_rows": review_rows[:30],
    }
    (output_dir / "phase5_report.json").write_text(json.dumps(report, indent=2, allow_nan=True), encoding="utf-8")
    write_markdown(output_dir / "PHASE5_LABEL_AUDIT_REPORT.md", report)
    print(f"[phase5] review queue: {output_dir / 'height_label_review_queue.csv'}", flush=True)
    print(f"[phase5] correction template: {correction_template}", flush=True)
    print(f"[phase5] report: {output_dir / 'PHASE5_LABEL_AUDIT_REPORT.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
