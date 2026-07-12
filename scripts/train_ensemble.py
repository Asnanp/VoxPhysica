#!/usr/bin/env python
"""GPU long-epoch multi-seed training ensemble for VocalMorph.

This replaces the old placeholder ensemble runner. It creates one config per
member, trains each member with a different seed and controlled hyperparameter
diversity, saves top-k checkpoints, and writes a manifest that can be evaluated
by ``scripts/evaluate_checkpoint_ensemble.py``.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import yaml

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a powerful GPU checkpoint ensemble.")
    parser.add_argument("--config", default="configs/pibnn_rtx3060_3cm_OPTIMIZED.yaml")
    parser.add_argument("--output-dir", default="outputs/epoch_ensemble_gpu")
    parser.add_argument("--n-models", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--base-seed", type=int, default=11)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--patience", type=int, default=45)
    parser.add_argument("--degradation-patience", type=int, default=10)
    parser.add_argument("--degradation-delta", type=float, default=2.50)
    parser.add_argument("--save-top-k", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=None, help="Base loader batch size; train uses speaker batch when enabled.")
    parser.add_argument("--eval-batch-size", type=int, default=None, help="Validation/test batch size. Defaults to a safe smaller value for TTA runs.")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--prefetch-factor", type=int, default=None)
    parser.add_argument("--eval-num-workers", type=int, default=None)
    parser.add_argument("--eval-prefetch-factor", type=int, default=None)
    parser.add_argument("--speakers-per-batch", type=int, default=None)
    parser.add_argument("--clips-per-speaker", type=int, default=None)
    parser.add_argument("--max-feature-frames", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--capacity-profile", choices=("base", "big", "max"), default="base")
    parser.add_argument("--no-auto-resume", action="store_true", help="Do not continue from per-seed last/interruption checkpoints.")
    parser.add_argument("--resume-from", default=None, help="Optional checkpoint to warm-start every member.")
    parser.add_argument("--model-only-resume", action="store_true", help="Warm-start weights but reset optimizer/scheduler.")
    parser.add_argument("--reset-resume-heads", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--eval-crops", type=int, default=5)
    parser.add_argument("--eval-samples", type=int, default=1)
    parser.add_argument("--eval-crop-size", type=int, default=128)
    parser.add_argument("--run-ensemble-eval", action="store_true")
    parser.add_argument(
        "--max-hours",
        type=float,
        default=None,
        help="Stop each member after this many hours and skip remaining members once the budget is used.",
    )
    return parser.parse_args()


def resolve(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(payload), handle, sort_keys=False, allow_unicode=False)


def ensure_mapping(parent: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        value = {}
        parent[key] = value
    return value


def scale_prob(value: Any, scale: float, cap: float = 0.95) -> float:
    try:
        raw = float(value)
    except Exception:
        raw = 0.0
    return float(max(0.0, min(cap, raw * float(scale))))


def scale_float(value: Any, scale: float, min_value: float = 0.0) -> float:
    try:
        raw = float(value)
    except Exception:
        raw = min_value
    return float(max(min_value, raw * float(scale)))


def member_seeds(args: argparse.Namespace) -> List[int]:
    if args.seeds:
        return [int(seed) for seed in args.seeds]
    defaults = [11, 17, 23, 31, 47, 59, 71, 83]
    if int(args.base_seed) == 11:
        return defaults[: int(args.n_models)]
    return [int(args.base_seed) + i * 6 for i in range(int(args.n_models))]


def variant(index: int) -> Dict[str, float]:
    variants = [
        {"lr": 0.85, "wd": 1.10, "aug": 0.90, "short": 2.00, "ema": 0.9985, "clip": 0.50},
        {"lr": 1.00, "wd": 1.00, "aug": 1.00, "short": 2.25, "ema": 0.9990, "clip": 0.55},
        {"lr": 1.15, "wd": 0.90, "aug": 1.10, "short": 1.85, "ema": 0.9980, "clip": 0.60},
        {"lr": 0.70, "wd": 1.25, "aug": 0.80, "short": 2.45, "ema": 0.9992, "clip": 0.45},
        {"lr": 1.30, "wd": 0.80, "aug": 1.20, "short": 2.10, "ema": 0.9988, "clip": 0.65},
        {"lr": 0.95, "wd": 1.15, "aug": 1.05, "short": 2.35, "ema": 0.9991, "clip": 0.50},
        {"lr": 1.08, "wd": 0.95, "aug": 0.95, "short": 1.95, "ema": 0.9986, "clip": 0.55},
        {"lr": 0.78, "wd": 1.05, "aug": 1.15, "short": 2.60, "ema": 0.9993, "clip": 0.45},
    ]
    return variants[int(index) % len(variants)]


def apply_capacity_profile(v2: Dict[str, Any], tc: Dict[str, Any], profile: str) -> None:
    if profile == "base":
        return
    if profile == "big":
        v2["ecapa_channels"] = max(int(v2.get("ecapa_channels", 192)), 224)
        v2["conformer_d_model"] = max(int(v2.get("conformer_d_model", 128)), 160)
        v2["conformer_blocks"] = max(int(v2.get("conformer_blocks", 3)), 4)
        v2["regression_hidden_dim"] = max(int(v2.get("regression_hidden_dim", 256)), 320)
        v2["dropout"] = max(float(v2.get("dropout", 0.35)), 0.38)
        v2["branch_dropout"] = max(float(v2.get("branch_dropout", 0.15)), 0.18)
        return
    v2["ecapa_channels"] = max(int(v2.get("ecapa_channels", 192)), 256)
    v2["conformer_d_model"] = max(int(v2.get("conformer_d_model", 128)), 192)
    v2["conformer_blocks"] = max(int(v2.get("conformer_blocks", 3)), 4)
    v2["regression_hidden_dim"] = max(int(v2.get("regression_hidden_dim", 256)), 384)
    v2["dropout"] = max(float(v2.get("dropout", 0.35)), 0.42)
    v2["branch_dropout"] = max(float(v2.get("branch_dropout", 0.15)), 0.20)
    # Keep this off by default in base/big. In max, turn it on so the profile
    # can survive on 12GB if the user's batch settings are also aggressive.
    tc["gradient_checkpointing"] = True


def patch_member_config(
    base: Mapping[str, Any],
    *,
    seed: int,
    index: int,
    args: argparse.Namespace,
    checkpoint_dir: Path,
    log_dir: Path,
) -> Dict[str, Any]:
    cfg = copy.deepcopy(dict(base))
    tc = ensure_mapping(cfg, "training")
    lc = ensure_mapping(cfg, "logging")
    ec = ensure_mapping(cfg, "evaluation")
    inf = ensure_mapping(ec, "inference")
    infer_top = ensure_mapping(cfg, "inference")
    model = ensure_mapping(cfg, "model")
    v2 = ensure_mapping(model, "v2")
    toggles = ensure_mapping(v2, "toggles")
    loss_weights = ensure_mapping(v2, "loss_weights")
    opt = ensure_mapping(tc, "optimizer")
    sched = ensure_mapping(tc, "scheduler")
    es = ensure_mapping(tc, "early_stopping")
    aug = ensure_mapping(tc, "augmentation")
    ema = ensure_mapping(tc, "ema")
    swa = ensure_mapping(tc, "swa")
    clipping = ensure_mapping(tc, "gradient_clipping")
    speaker_batching = ensure_mapping(tc, "speaker_batching")

    v = variant(index)
    apply_capacity_profile(v2, tc, str(args.capacity_profile))
    tc["seed"] = int(seed)
    tc["epochs"] = int(args.epochs)
    if args.max_hours is not None and float(args.max_hours) > 0:
        tc["max_hours"] = float(args.max_hours)
    tc["device"] = str(args.device)
    tc["mixed_precision"] = True
    tc["allow_tf32"] = True
    tc["pin_memory"] = True
    # Eval/test use TTA crops and can briefly hold much larger batches than
    # training. Keeping pin_memory off there avoids PyTorch's pin-memory thread
    # eating CUDA memory while the actual model still runs on GPU.
    tc["eval_pin_memory"] = False
    tc["non_blocking"] = True
    tc["gradient_accumulation_steps"] = max(1, int(args.gradient_accumulation_steps))
    if args.batch_size is not None:
        tc["batch_size"] = int(args.batch_size)
    base_batch_size = int(tc.get("batch_size", 32))
    if args.eval_batch_size is not None:
        tc["eval_batch_size"] = int(args.eval_batch_size)
    else:
        safe_eval_batch = 48 if int(args.eval_crops) >= 6 else 64
        tc["eval_batch_size"] = min(base_batch_size, safe_eval_batch)
    if args.num_workers is not None:
        tc["num_workers"] = int(args.num_workers)
    if args.prefetch_factor is not None:
        tc["prefetch_factor"] = int(args.prefetch_factor)
    if args.eval_num_workers is not None:
        tc["eval_num_workers"] = int(args.eval_num_workers)
    elif args.num_workers is not None:
        tc["eval_num_workers"] = min(int(args.num_workers), 4)
    if args.eval_prefetch_factor is not None:
        tc["eval_prefetch_factor"] = int(args.eval_prefetch_factor)
    elif args.prefetch_factor is not None:
        tc["eval_prefetch_factor"] = min(int(args.prefetch_factor), 2)
    if args.max_feature_frames is not None:
        tc["max_feature_frames"] = int(args.max_feature_frames)
    tc["train_eval_frequency"] = 0
    tc["final_train_eval"] = False
    tc["lr_warmup_epochs"] = max(int(tc.get("lr_warmup_epochs", 5)), 8)
    tc["lr_warmup_start_factor"] = min(float(tc.get("lr_warmup_start_factor", 0.1)), 0.08)

    opt["lr"] = float(opt.get("lr", 9e-5)) * float(v["lr"])
    opt["weight_decay"] = float(opt.get("weight_decay", 0.05)) * float(v["wd"])
    sched["type"] = sched.get("type", "cosine_annealing")
    sched["T_max"] = max(int(args.epochs), int(sched.get("T_max", 1)))
    sched["eta_min"] = min(float(sched.get("eta_min", 1e-5)), float(opt["lr"]) * 0.12)

    es["enabled"] = True
    es["patience"] = int(args.patience)
    es["monitor"] = es.get("monitor", "height_mae_speaker_balanced")
    es["mode"] = es.get("mode", "min")
    es["min_delta"] = min(float(es.get("min_delta", 0.001)), 0.0005)
    es["degradation_patience"] = int(args.degradation_patience)
    es["degradation_delta"] = float(args.degradation_delta)

    aug["enabled"] = True
    for key in ("noise_p", "time_mask_p", "feat_mask_p", "scale_p", "temporal_jitter_p"):
        if key in aug:
            aug[key] = scale_prob(aug[key], float(v["aug"]))
    for key in ("noise_std", "time_mask_max_frac", "feat_mask_max_frac", "scale_std", "temporal_jitter_max_frac"):
        if key in aug:
            aug[key] = scale_float(aug[key], float(v["aug"]), 0.0)

    ema["enabled"] = True
    ema["decay"] = float(v["ema"])
    ema["update_every"] = 1
    ema["warmup_steps"] = max(int(ema.get("warmup_steps", 0)), 800)
    ema["use_for_eval"] = True

    swa["enabled"] = True
    swa["start_frac"] = float(swa.get("start_frac", 0.68))
    swa["start_frac"] = min(max(swa["start_frac"], 0.58), 0.74)
    swa["anneal_epochs"] = max(int(swa.get("anneal_epochs", 10)), 14)
    swa["use_for_eval"] = True

    clipping["enabled"] = True
    clipping["max_norm"] = float(v["clip"])

    toggles["use_uncertainty_calibration"] = True
    toggles["use_height_adapter"] = True
    toggles["use_speaker_consistency"] = False
    toggles["use_domain_adv"] = False
    loss_weights["height"] = max(float(loss_weights.get("height", 5.0)), 5.0)
    loss_weights["height_bin"] = max(float(loss_weights.get("height_bin", 0.0)), 0.20)
    loss_weights["uncertainty_calibration"] = max(float(loss_weights.get("uncertainty_calibration", 0.0)), 0.03)

    hbw = ensure_mapping(speaker_batching, "height_bin_weights")
    ghw = ensure_mapping(speaker_batching, "gender_height_weights")
    speaker_batching["enabled"] = True
    speaker_batching["mode"] = speaker_batching.get("mode", "height_balanced")
    if args.speakers_per_batch is not None:
        speaker_batching["speakers_per_batch"] = int(args.speakers_per_batch)
    if args.clips_per_speaker is not None:
        speaker_batching["clips_per_speaker"] = int(args.clips_per_speaker)
    hbw["short"] = float(v["short"])
    hbw["medium"] = 1.0
    hbw["tall"] = 1.0
    ghw["male_short"] = max(float(ghw.get("male_short", 1.0)), float(v["short"]))
    ghw["female_short"] = max(float(ghw.get("female_short", 1.0)), min(float(v["short"]), 1.45))

    inf["use_ensemble"] = True
    inf["deterministic"] = True
    inf["n_samples"] = int(args.eval_samples)
    inf["n_crops"] = int(args.eval_crops)
    inf["crop_size"] = int(args.eval_crop_size)

    tb = ensure_mapping(lc, "tensorboard")
    ckpt = ensure_mapping(lc, "checkpoint")
    tb["enabled"] = True
    tb["log_dir"] = str(log_dir.as_posix())
    ckpt["dir"] = str(checkpoint_dir.as_posix())
    ckpt["save_top_k"] = int(args.save_top_k)
    ckpt["monitor"] = ckpt.get("monitor", es["monitor"])
    ckpt["mode"] = ckpt.get("mode", es["mode"])

    infer_top["checkpoint"] = str((checkpoint_dir / "best.ckpt").as_posix())
    infer_top["deterministic_ensemble"] = True
    infer_top["n_crops"] = int(args.eval_crops)
    infer_top["crop_size"] = int(args.eval_crop_size)
    infer_top["output_dir"] = str((Path(args.output_dir) / "predictions" / f"seed_{seed}").as_posix())
    return cfg


def stream_command(cmd: List[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        return int(proc.wait())


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def find_resume_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    for name in ("last.ckpt", "last_good.ckpt", "interrupt.ckpt", "crash.ckpt", "best.ckpt", "best_model.pt"):
        path = checkpoint_dir / name
        if path.exists():
            return path
    return None


def summarize_member(metrics: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not metrics:
        return {}
    val = metrics.get("final_val", {}) if isinstance(metrics.get("final_val"), dict) else {}
    test = metrics.get("final_test", {}) if isinstance(metrics.get("final_test"), dict) else {}
    return {
        "best_monitor_value": metrics.get("best_monitor_value"),
        "val_height_mae_speaker_balanced": val.get("height_mae_speaker_balanced"),
        "val_height_mae_speaker": val.get("height_mae_speaker"),
        "test_height_mae_speaker": test.get("height_mae_speaker"),
        "test_height_mae_speaker_balanced": test.get("height_mae_speaker_balanced"),
        "test_height_mae": test.get("height_mae"),
    }


def main() -> int:
    args = parse_args()
    if str(args.device).lower() == "cuda":
        try:
            import torch

            if not torch.cuda.is_available():
                raise SystemExit("CUDA requested but torch.cuda.is_available() is false.")
        except ImportError as exc:
            raise SystemExit(f"PyTorch import failed: {exc}") from exc

    config_path = resolve(args.config)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base = load_yaml(config_path)
    seeds = member_seeds(args)
    members: List[Dict[str, Any]] = []

    print("=" * 64)
    print("  VocalMorph Long-Epoch GPU Ensemble")
    print("=" * 64)
    print(f"Base config : {config_path}")
    print(f"Output     : {output_dir}")
    print(f"Seeds      : {seeds}")
    print(f"Epochs     : {int(args.epochs)} per member")
    if args.max_hours is not None and float(args.max_hours) > 0:
        print(f"Time limit : {float(args.max_hours):.1f} hours per member")
    print(f"Capacity   : {args.capacity_profile}")
    if args.speakers_per_batch and args.clips_per_speaker:
        effective_batch = int(args.speakers_per_batch) * int(args.clips_per_speaker)
        print(
            f"Train batch: {args.speakers_per_batch} speakers x "
            f"{args.clips_per_speaker} clips = {effective_batch} clips"
        )
    if args.batch_size:
        safe_eval_batch = int(args.eval_batch_size) if args.eval_batch_size else min(int(args.batch_size), 48 if int(args.eval_crops) >= 6 else 64)
        print(f"Base batch : {int(args.batch_size)}")
        print(f"Eval batch : {safe_eval_batch} (pin_memory off)")
    print()

    started_at = time.time()
    max_total_seconds = (
        float(args.max_hours) * 3600.0
        if args.max_hours is not None and float(args.max_hours) > 0
        else None
    )
    for index, seed in enumerate(seeds):
        if max_total_seconds is not None and (time.time() - started_at) >= max_total_seconds:
            print(
                f"[ensemble] Reached total time budget of {float(args.max_hours):.1f}h; "
                "skipping remaining members."
            )
            break
        checkpoint_dir = output_dir / "checkpoints" / f"seed_{seed}"
        log_dir = output_dir / "logs" / f"seed_{seed}"
        member_config_path = output_dir / "configs" / f"seed_{seed}.yaml"
        metrics_path = output_dir / "metrics" / f"seed_{seed}.json"
        train_log = output_dir / "train_logs" / f"seed_{seed}.log"
        best_ckpt = checkpoint_dir / "best.ckpt"
        cfg = patch_member_config(
            base,
            seed=seed,
            index=index,
            args=args,
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir,
        )
        write_yaml(member_config_path, cfg)
        member = {
            "index": index,
            "seed": seed,
            "config": str(member_config_path),
            "checkpoint_dir": str(checkpoint_dir),
            "best_checkpoint": str(best_ckpt),
            "metrics": str(metrics_path),
            "train_log": str(train_log),
            "variant": variant(index),
            "status": "pending",
        }

        if args.skip_existing and best_ckpt.exists():
            print(f"[ensemble] seed {seed}: existing checkpoint found, skipping training.")
            member["status"] = "skipped_existing"
            member["summary"] = summarize_member(read_json(metrics_path))
            members.append(member)
            continue

        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "train.py"),
            "--config",
            str(member_config_path),
            "--device",
            str(args.device),
            "--epochs",
            str(int(args.epochs)),
            "--seed",
            str(seed),
            "--metrics-out",
            str(metrics_path),
        ]
        if args.max_hours is not None and float(args.max_hours) > 0:
            remaining_hours = max(
                0.01,
                (max_total_seconds - (time.time() - started_at)) / 3600.0,
            )
            cmd.extend(["--max-hours", f"{remaining_hours:.4f}"])
        resume_checkpoint = None
        if args.resume_from:
            resume_checkpoint = resolve(args.resume_from)
        elif not args.no_auto_resume:
            resume_checkpoint = find_resume_checkpoint(checkpoint_dir)
        if resume_checkpoint is not None:
            member["resume_checkpoint"] = str(resume_checkpoint)
            print(f"[ensemble] seed {seed}: resume checkpoint -> {resume_checkpoint}")
            cmd.extend(["--resume", str(resume_checkpoint)])
        if args.resume_from:
            member["warm_start_all_members"] = True
        if args.model_only_resume:
            cmd.append("--model-only-resume")
        if args.reset_resume_heads:
            cmd.append("--reset-resume-heads")
        member["command"] = cmd
        if args.dry_run:
            print(f"[dry-run] seed {seed}: {' '.join(cmd)}")
            member["status"] = "dry_run"
            members.append(member)
            continue

        print("\n" + "=" * 64)
        print(f"  Training member {index + 1}/{len(seeds)} | seed={seed}")
        print("=" * 64)
        return_code = stream_command(cmd, train_log)
        member["return_code"] = return_code
        member["status"] = "ok" if return_code == 0 and best_ckpt.exists() else "failed"
        member["summary"] = summarize_member(read_json(metrics_path))
        members.append(member)
        manifest_path = output_dir / "ensemble_manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "base_config": str(config_path),
                    "output_dir": str(output_dir),
                    "epochs": int(args.epochs),
                    "members": members,
                    "elapsed_minutes": (time.time() - started_at) / 60.0,
                },
                indent=2,
                allow_nan=True,
            ),
            encoding="utf-8",
        )
        if return_code != 0:
            print(f"[ensemble] seed {seed} failed with return code {return_code}; continuing to next member.")

    manifest = {
        "base_config": str(config_path),
        "output_dir": str(output_dir),
        "epochs": int(args.epochs),
        "members": members,
        "completed_members": [m for m in members if m.get("status") in {"ok", "skipped_existing"} and Path(m["best_checkpoint"]).exists()],
        "elapsed_minutes": (time.time() - started_at) / 60.0,
    }
    manifest_path = output_dir / "ensemble_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=True), encoding="utf-8")
    print(f"\n[ensemble] wrote manifest: {manifest_path}")

    if args.run_ensemble_eval and not args.dry_run:
        eval_cmd = [
            sys.executable,
            str(ROOT / "scripts" / "evaluate_checkpoint_ensemble.py"),
            "--manifest",
            str(manifest_path),
            "--device",
            str(args.device),
            "--output-dir",
            str(output_dir / "ensemble_eval"),
        ]
        print(f"[ensemble] running evaluation: {' '.join(eval_cmd)}")
        return stream_command(eval_cmd, output_dir / "ensemble_eval.log")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
