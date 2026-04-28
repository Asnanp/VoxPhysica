#!/usr/bin/env python
"""Build strict audited train/val/test feature splits from canonical split CSVs."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np
import yaml
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.preprocessing.augmentation import (  # noqa: E402
    AUDIOMENTATIONS_AVAILABLE,
    AugmentationConfig,
    apply_augmentations,
    build_augmenter,
)
from src.preprocessing.audio_enhancement import MicrophoneEnhancementConfig  # noqa: E402
from src.preprocessing.feature_extractor import (  # noqa: E402
    FeatureConfig,
    build_feature_config,
    extract_all_features,
    load_audio,
)
from src.utils.audit_utils import (  # noqa: E402
    CANONICAL_SPLITS,
    assert_no_split_leakage,
    compare_scalar_drift,
    compute_feature_diagnostics,
    compute_target_stats_from_feature_dir,
    feature_contract_payload,
    file_sha256,
    json_fingerprint,
    leakage_report_for_rows,
    normalize_path,
    read_csv_rows,
    safe_float,
    summarize_split_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build strict VocalMorph feature splits.")
    parser.add_argument("--config", default="configs/pibnn_base.yaml")
    parser.add_argument("--train_csv", default="data/splits/train_clean.csv")
    parser.add_argument("--val_csv", default="data/splits/val_clean.csv")
    parser.add_argument("--test_csv", default="data/splits/test_clean.csv")
    parser.add_argument("--output_dir", default="data/features_audited")
    parser.add_argument("--sample_rate", type=int, default=None)
    parser.add_argument("--n_mfcc", type=int, default=None)
    parser.add_argument("--max_utterances_per_speaker", type=int, default=20)
    parser.add_argument("--augment_train_factor", type=int, default=1)
    parser.add_argument("--augment_timit_factor", type=int, default=2)
    parser.add_argument("--disable_enhancement", action="store_true")
    parser.add_argument("--allow_manifest_drift", action="store_true")
    parser.add_argument("--skip_audio_hash_check", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume_existing", action="store_true")
    return parser.parse_args()


def _resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def _resolve_audio_path(path: str) -> str:
    resolved = _resolve(path)
    if os.path.isfile(resolved):
        return resolved
    normalized = str(path or "").replace("\\", "/")
    nisp_marker = "data/NISP-Dataset/"
    marker_idx = normalized.find(nisp_marker)
    if marker_idx >= 0:
        raw_candidate = _resolve(normalized[marker_idx:])
        if os.path.isfile(raw_candidate):
            return raw_candidate
    return resolved


def _iter_audio_paths(audio_paths_field: str, max_n: int) -> Iterable[str]:
    parts = [part.strip() for part in str(audio_paths_field or "").split("|") if part.strip()]
    return parts[: max(0, int(max_n))]


def _prepare_output_dir(output_dir: str, overwrite: bool, resume_existing: bool = False) -> None:
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        return
    stale = [os.path.join(output_dir, split_name) for split_name in CANONICAL_SPLITS if os.path.isdir(os.path.join(output_dir, split_name))]
    if stale and resume_existing:
        os.makedirs(output_dir, exist_ok=True)
        return
    if stale and not overwrite:
        raise RuntimeError(
            f"{output_dir} already contains split artifacts. Re-run with --overwrite to rebuild audited features."
        )
    for split_dir in stale:
        shutil.rmtree(split_dir)
    os.makedirs(output_dir, exist_ok=True)


_RESUME_FILE_RE = re.compile(r"^(?P<speaker>.+)_(?P<clip>\d{3})(?:_aug\d{2})?\.npz$", re.IGNORECASE)


def _scan_existing_split(split_dir: str) -> Dict[str, Any]:
    existing_files: set[str] = set()
    realized_speakers: set[str] = set()
    speaker_clip_counts: Counter[str] = Counter()

    if not os.path.isdir(split_dir):
        return {
            "existing_files": existing_files,
            "realized_speakers": realized_speakers,
            "speaker_clip_counts": speaker_clip_counts,
        }

    for name in os.listdir(split_dir):
        if not name.lower().endswith(".npz"):
            continue
        existing_files.add(name)
        match = _RESUME_FILE_RE.match(name)
        if not match:
            continue
        speaker_id = match.group("speaker")
        realized_speakers.add(speaker_id)
        speaker_clip_counts[speaker_id] += 1

    return {
        "existing_files": existing_files,
        "realized_speakers": realized_speakers,
        "speaker_clip_counts": speaker_clip_counts,
    }


def _total_aug_variants(
    *,
    split_name: str,
    source: str,
    augment_train_factor: int,
    augment_timit_factor: int,
) -> int:
    total_aug_variants = 0
    if split_name == "train":
        total_aug_variants += max(0, int(augment_train_factor))
        if source == "TIMIT":
            total_aug_variants += max(0, int(augment_timit_factor))
    elif augment_train_factor > 0 or augment_timit_factor > 0:
        total_aug_variants = 0

    if split_name != "train":
        total_aug_variants = 0
    return total_aug_variants


def _load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _build_audio_augmentation_config(config: Mapping[str, Any]) -> AugmentationConfig:
    aug_cfg = dict(config.get("training", {}).get("augmentation", {}) or {})
    speed_rates = tuple(
        float(rate)
        for rate in aug_cfg.get("speed_perturb_rates", (1.0,))
        if float(rate) > 0.0
    )
    return AugmentationConfig(
        speed_perturb_p=float(aug_cfg.get("speed_perturb_p", 0.0)),
        speed_perturb_rates=speed_rates or (1.0,),
    )


def _build_audio_hash_index(
    split_rows: Mapping[str, List[Dict[str, str]]],
    *,
    verify_audio_hashes: bool,
) -> Dict[str, str]:
    if not verify_audio_hashes:
        return {}
    all_paths = set()
    for rows in split_rows.values():
        for row in rows:
            for raw_path in _iter_audio_paths(row.get("audio_paths", ""), max_n=10**9):
                full_path = _resolve_audio_path(raw_path)
                if os.path.isfile(full_path):
                    all_paths.add(normalize_path(full_path))
    hash_index: Dict[str, str] = {}
    for full_path in tqdm(sorted(all_paths), desc="Hashing raw audio", leave=False):
        hash_index[full_path] = file_sha256(full_path)
    return hash_index


def _save_npz(
    out_path: str,
    feature_dict: Mapping[str, Any],
    row: Mapping[str, str],
    *,
    split_name: str,
    rel_audio_path: str,
    load_metadata: Optional[Mapping[str, Any]],
    is_augmented: bool,
    augmentation_tag: str,
    audio_sha256: str,
    feature_fingerprint: str,
) -> None:
    source = str(row.get("source", "") or "").upper()
    gender_raw = str(row.get("gender", "") or "").strip().lower()
    gender_id = 1 if gender_raw == "male" else 0
    enhancement_meta = dict((load_metadata or {}).get("enhancement") or {})

    np.savez(
        out_path,
        sequence=np.asarray(feature_dict["sequence"], dtype=np.float32),
        f0_mean=np.float32(feature_dict["f0_mean"]),
        formant_spacing_mean=np.float32(feature_dict["formant_spacing_mean"]),
        vtl_mean=np.float32(feature_dict["vtl_mean"]),
        jitter=np.float32(feature_dict["jitter"]),
        shimmer=np.float32(feature_dict["shimmer"]),
        hnr=np.float32(feature_dict["hnr"]),
        duration_s=np.float32(feature_dict.get("duration_s", safe_float(load_metadata.get("duration_s") if load_metadata else np.nan))),
        voiced_ratio=np.float32(feature_dict.get("voiced_ratio", np.nan)),
        invalid_spacing_rate=np.float32(feature_dict.get("invalid_spacing_rate", np.nan)),
        invalid_vtl_rate=np.float32(feature_dict.get("invalid_vtl_rate", np.nan)),
        speech_ratio=np.float32(safe_float(enhancement_meta.get("speech_ratio"))),
        snr_db_estimate=np.float32(safe_float(enhancement_meta.get("snr_db_estimate"))),
        capture_quality_score=np.float32(safe_float(enhancement_meta.get("capture_quality_score"))),
        distance_cm_estimate=np.float32(safe_float(enhancement_meta.get("distance_cm_estimate"))),
        distance_confidence=np.float32(safe_float(enhancement_meta.get("distance_confidence"))),
        clipped_ratio=np.float32(safe_float(enhancement_meta.get("clipped_ratio"))),
        quality_ok=np.bool_(bool(enhancement_meta.get("quality_ok", True))),
        height_cm=np.float32(safe_float(row.get("height_cm"))),
        weight_kg=np.float32(safe_float(row.get("weight_kg"))),
        age=np.float32(safe_float(row.get("age"))),
        gender=np.int64(gender_id),
        speaker_id=np.array(str(row.get("speaker_id", "")), dtype=object),
        source=np.array(source, dtype=object),
        split=np.array(split_name, dtype=object),
        audio_rel_path=np.array(str(rel_audio_path), dtype=object),
        audio_sha256=np.array(audio_sha256 or "", dtype=object),
        is_augmented=np.int64(1 if is_augmented else 0),
        augmentation_tag=np.array(augmentation_tag, dtype=object),
        feature_config_fingerprint=np.array(feature_fingerprint, dtype=object),
    )


def _build_split(
    *,
    split_name: str,
    rows: List[Dict[str, str]],
    out_dir: str,
    feature_config: FeatureConfig,
    augment_train_factor: int,
    augment_timit_factor: int,
    max_utterances_per_speaker: int,
    enhance_audio: bool,
    audio_hashes: Mapping[str, str],
    resume_existing: bool,
    augmentation_config: AugmentationConfig,
) -> Dict[str, Any]:
    split_dir = os.path.join(out_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)

    augmenter = build_augmenter() if split_name == "train" else None
    aug_cfg = augmentation_config
    enhancement_cfg = MicrophoneEnhancementConfig(enabled=enhance_audio)
    feature_fingerprint = json_fingerprint(feature_config.to_dict())

    resume_state = _scan_existing_split(split_dir) if resume_existing else {
        "existing_files": set(),
        "realized_speakers": set(),
        "speaker_clip_counts": Counter(),
    }
    existing_files: set[str] = set(resume_state["existing_files"])
    processed = int(len(existing_files))
    skipped = 0
    speaker_clip_counts: Counter[str] = Counter(resume_state["speaker_clip_counts"])
    skipped_reasons: Counter[str] = Counter()
    realized_speakers: set[str] = set(resume_state["realized_speakers"])

    for row in tqdm(rows, desc=f"{split_name} feature extraction"):
        speaker_id = str(row.get("speaker_id", "") or "").strip()
        source = str(row.get("source", "") or "").upper()
        speaker_saved = speaker_id in realized_speakers

        for clip_idx, rel_audio_path in enumerate(
            _iter_audio_paths(row.get("audio_paths", ""), max_utterances_per_speaker)
        ):
            original_name = f"{speaker_id}_{clip_idx:03d}.npz"
            total_aug_variants = _total_aug_variants(
                split_name=split_name,
                source=source,
                augment_train_factor=augment_train_factor,
                augment_timit_factor=augment_timit_factor,
            )
            aug_names = [
                f"{speaker_id}_{clip_idx:03d}_aug{aug_idx:02d}.npz"
                for aug_idx in range(total_aug_variants)
            ]
            original_exists = original_name in existing_files
            missing_aug_names = [name for name in aug_names if name not in existing_files]
            if resume_existing and original_exists and not missing_aug_names:
                speaker_saved = True
                continue

            full_audio_path = _resolve_audio_path(rel_audio_path)
            loaded = load_audio(
                full_audio_path,
                target_sr=feature_config.sample_rate,
                max_duration=10.0,
                min_duration=1.5,
                enhance=enhance_audio,
                enhancement_config=enhancement_cfg,
                return_metadata=True,
            )
            if loaded is None:
                skipped += 1
                skipped_reasons["load_failed_or_too_short"] += 1
                continue

            audio, metadata = loaded
            if not original_exists:
                feats = extract_all_features(audio, feature_config)
                out_path = os.path.join(split_dir, original_name)
                _save_npz(
                    out_path,
                    feats,
                    row,
                    split_name=split_name,
                    rel_audio_path=rel_audio_path,
                    load_metadata=metadata,
                    is_augmented=False,
                    augmentation_tag="original",
                    audio_sha256=audio_hashes.get(normalize_path(full_audio_path), ""),
                    feature_fingerprint=feature_fingerprint,
                )
                processed += 1
                speaker_clip_counts[speaker_id] += 1
                realized_speakers.add(speaker_id)
                existing_files.add(original_name)
                speaker_saved = True

            if total_aug_variants > 0 and missing_aug_names:
                augmented_wavs = apply_augmentations(
                    audio,
                    feature_config.sample_rate,
                    augmenter,
                    total_aug_variants,
                    config=aug_cfg,
                )
                for aug_idx, aug_audio in enumerate(augmented_wavs):
                    aug_name = f"{speaker_id}_{clip_idx:03d}_aug{aug_idx:02d}.npz"
                    if aug_name in existing_files:
                        continue
                    aug_feats = extract_all_features(aug_audio, feature_config)
                    aug_metadata = dict(metadata or {})
                    aug_metadata["duration_s"] = float(
                        len(aug_audio) / max(feature_config.sample_rate, 1)
                    )
                    aug_path = os.path.join(split_dir, aug_name)
                    _save_npz(
                        aug_path,
                        aug_feats,
                        row,
                        split_name=split_name,
                        rel_audio_path=rel_audio_path,
                        load_metadata=aug_metadata,
                        is_augmented=True,
                        augmentation_tag=f"aug{aug_idx:02d}",
                        audio_sha256=audio_hashes.get(normalize_path(full_audio_path), ""),
                        feature_fingerprint=feature_fingerprint,
                    )
                    processed += 1
                    speaker_clip_counts[speaker_id] += 1
                    realized_speakers.add(speaker_id)
                    existing_files.add(aug_name)
                    speaker_saved = True

        if not speaker_saved:
            skipped_reasons["speaker_realized_zero_clips"] += 1

    return {
        "split_dir": split_dir,
        "processed_files": int(processed),
        "skipped_files": int(skipped),
        "skipped_reasons": dict(sorted(skipped_reasons.items())),
        "realized_speakers": sorted(realized_speakers),
        "per_speaker_clip_count": dict(sorted(speaker_clip_counts.items())),
    }


def _write_markdown_report(
    *,
    leakage_report: Mapping[str, Any],
    diagnostics: Mapping[str, Mapping[str, Any]],
    drift_report: Mapping[str, Any],
    output_dir: str,
    feature_config: FeatureConfig,
    manifest: Mapping[str, Any],
) -> None:
    audit_dir = os.path.join(ROOT, "audit")
    os.makedirs(audit_dir, exist_ok=True)

    leakage_path = os.path.join(audit_dir, "leakage_report.md")
    with open(leakage_path, "w", encoding="utf-8") as handle:
        handle.write("# Leakage Report\n\n")
        handle.write("## Findings\n")
        handle.write(f"- Canonical feature root: `{output_dir}`\n")
        handle.write(
            f"- Speaker overlap counts: `{json.dumps(leakage_report.get('speaker_overlap_counts', {}), sort_keys=True)}`\n"
        )
        handle.write(
            f"- Audio-path overlap counts: `{json.dumps(leakage_report.get('audio_path_overlap_counts', {}), sort_keys=True)}`\n"
        )
        handle.write(
            f"- Audio-hash overlap counts: `{json.dumps(leakage_report.get('audio_hash_overlap_counts', {}), sort_keys=True)}`\n"
        )
        handle.write("\n## Fixes\n")
        handle.write("- Canonicalized the build around `train_clean.csv`, `val_clean.csv`, and `test_clean.csv`.\n")
        handle.write("- Added hard leakage assertions for speaker overlap, raw audio path overlap, and duplicate audio-content hashes.\n")
        handle.write("- Added audited `feature_contract.json` and `build_manifest.json` artifacts so training can reject stale features.\n")
        handle.write("\n## Remaining Risks\n")
        handle.write("- If the raw split manifests change, the audited feature contract becomes invalid and the training CLI now fails closed.\n")

    feature_path = os.path.join(audit_dir, "feature_audit.md")
    with open(feature_path, "w", encoding="utf-8") as handle:
        handle.write("# Feature Audit\n\n")
        handle.write("## Findings\n")
        handle.write(
            f"- Feature config fingerprint: `{json_fingerprint(feature_config.to_dict())}`\n"
        )
        for split_name in CANONICAL_SPLITS:
            diag = diagnostics[split_name]
            handle.write(
                f"- `{split_name}`: files={diag['file_count']}, speakers={diag['unique_speakers']}, "
                f"zero_praat_tail_files={diag['zero_praat_tail_files']}, zero_voiced_f0_files={diag['zero_voiced_f0_files']}, "
                f"zero_variance_dims={diag['zero_variance_dims_count']}\n"
            )
        handle.write(
            f"- Train-vs-other scalar drift: `{json.dumps(drift_report, sort_keys=True)}`\n"
        )
        handle.write("\n## Fixes\n")
        handle.write("- Strict feature builds now fail closed when required Praat/parselmouth features are unavailable.\n")
        handle.write("- Training and inference can validate the exact audited feature contract before using any artifacts.\n")
        handle.write("- Per-clip capture-quality and duration metadata are now persisted into every `.npz` artifact.\n")
        handle.write("\n## Remaining Risks\n")
        handle.write("- The current environment still needs `parselmouth` installed before a canonical audited rebuild can complete.\n")
        handle.write(f"- Build manifest status: `{manifest.get('status', 'unknown')}`\n")


def main() -> int:
    args = parse_args()
    config_path = _resolve(args.config)
    config = _load_config(config_path)
    feature_config = build_feature_config(config)
    augmentation_config = _build_audio_augmentation_config(config)
    if args.sample_rate is not None:
        feature_config.sample_rate = int(args.sample_rate)
    if args.n_mfcc is not None:
        feature_config.n_mfcc = int(args.n_mfcc)

    split_paths = {
        "train": _resolve(args.train_csv),
        "val": _resolve(args.val_csv),
        "test": _resolve(args.test_csv),
    }
    split_rows = {split_name: read_csv_rows(path) for split_name, path in split_paths.items()}
    split_summaries = {
        split_name: summarize_split_rows(rows) for split_name, rows in split_rows.items()
    }

    leakage_report = leakage_report_for_rows(
        split_rows,
        verify_audio_hashes=not args.skip_audio_hash_check,
        root_dir=ROOT,
    )
    assert_no_split_leakage(leakage_report)
    print("[VocalMorph] Split leakage audit passed.")

    if args.resume_existing and args.overwrite:
        raise RuntimeError("Use either --overwrite or --resume_existing, not both.")
    output_dir = _resolve(args.output_dir)
    _prepare_output_dir(output_dir, overwrite=args.overwrite, resume_existing=args.resume_existing)

    audio_hashes = _build_audio_hash_index(
        split_rows,
        verify_audio_hashes=not args.skip_audio_hash_check,
    )

    if (args.augment_train_factor > 0 or args.augment_timit_factor > 0) and not AUDIOMENTATIONS_AVAILABLE:
        print("[WARN] audiomentations is not installed. Falling back to built-in robust audio augmentations.")

    build_results: Dict[str, Dict[str, Any]] = {}
    for split_name in CANONICAL_SPLITS:
        build_results[split_name] = _build_split(
            split_name=split_name,
            rows=split_rows[split_name],
            out_dir=output_dir,
            feature_config=feature_config,
            augment_train_factor=args.augment_train_factor,
            augment_timit_factor=args.augment_timit_factor,
            max_utterances_per_speaker=args.max_utterances_per_speaker,
            enhance_audio=not args.disable_enhancement,
            audio_hashes=audio_hashes,
            resume_existing=args.resume_existing,
            augmentation_config=augmentation_config,
        )

    expected_speakers = {
        split_name: sorted({str(row.get("speaker_id", "") or "").strip() for row in rows if str(row.get("speaker_id", "") or "").strip()})
        for split_name, rows in split_rows.items()
    }
    drift_flags = []
    for split_name in CANONICAL_SPLITS:
        realized = set(build_results[split_name]["realized_speakers"])
        expected = set(expected_speakers[split_name])
        missing = sorted(expected - realized)
        unexpected = sorted(realized - expected)
        build_results[split_name]["missing_expected_speakers"] = missing
        build_results[split_name]["unexpected_speakers"] = unexpected
        if missing or unexpected:
            drift_flags.append(split_name)

    if drift_flags and not args.allow_manifest_drift:
        raise RuntimeError(
            "Manifest drift detected after feature materialization. "
            f"Splits with mismatch: {', '.join(drift_flags)}. "
            "Re-run only if you intentionally allow drift with --allow_manifest_drift."
        )

    train_dir = os.path.join(output_dir, "train")
    target_stats = compute_target_stats_from_feature_dir(train_dir)
    stats_path = os.path.join(output_dir, "target_stats.json")
    with open(stats_path, "w", encoding="utf-8") as handle:
        json.dump(target_stats, handle, indent=2)

    diagnostics = {
        split_name: compute_feature_diagnostics(os.path.join(output_dir, split_name))
        for split_name in CANONICAL_SPLITS
    }
    drift_report = {
        "val_vs_train": compare_scalar_drift(
            diagnostics["train"],
            diagnostics["val"],
            keys=("duration_s", "speech_ratio", "snr_db_estimate", "capture_quality_score", "f0_mean", "formant_spacing_mean", "vtl_mean"),
        ),
        "test_vs_train": compare_scalar_drift(
            diagnostics["train"],
            diagnostics["test"],
            keys=("duration_s", "speech_ratio", "snr_db_estimate", "capture_quality_score", "f0_mean", "formant_spacing_mean", "vtl_mean"),
        ),
    }
    diagnostics_path = os.path.join(output_dir, "feature_diagnostics.json")
    with open(diagnostics_path, "w", encoding="utf-8") as handle:
        json.dump({"splits": diagnostics, "drift": drift_report}, handle, indent=2)

    contract = feature_contract_payload(
        feature_config=feature_config.to_dict(),
        split_files=split_paths,
        target_stats=target_stats,
    )
    contract_path = os.path.join(output_dir, "feature_contract.json")
    with open(contract_path, "w", encoding="utf-8") as handle:
        json.dump(contract, handle, indent=2)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": normalize_path(config_path),
        "output_dir": normalize_path(output_dir),
        "status": "manifest_drift_allowed" if drift_flags else "canonical",
        "feature_config": feature_config.to_dict(),
        "feature_config_fingerprint": json_fingerprint(feature_config.to_dict()),
        "split_files": {key: normalize_path(value) for key, value in split_paths.items()},
        "split_summaries": split_summaries,
        "leakage_report": leakage_report,
        "build_results": build_results,
        "target_stats": target_stats,
        "diagnostics_path": normalize_path(diagnostics_path),
        "feature_contract_path": normalize_path(contract_path),
        "config_fingerprint": json_fingerprint(config),
    }
    manifest_path = os.path.join(output_dir, "build_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    _write_markdown_report(
        leakage_report=leakage_report,
        diagnostics=diagnostics,
        drift_report=drift_report,
        output_dir=output_dir,
        feature_config=feature_config,
        manifest=manifest,
    )

    print("\nStrict audited feature build complete")
    for split_name in CANONICAL_SPLITS:
        result = build_results[split_name]
        print(
            f"  {split_name}: processed={result['processed_files']}, "
            f"skipped={result['skipped_files']}, speakers={len(result['realized_speakers'])}"
        )
    print(f"  target_stats: {stats_path}")
    print(f"  diagnostics:  {diagnostics_path}")
    print(f"  manifest:     {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
