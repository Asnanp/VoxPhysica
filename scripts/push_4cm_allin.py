#!/usr/bin/env python
"""All-in 4cm MAE push: export GPU preds, refresh CPU stacks, gauntlet blend, meta-stack."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
OUT = ROOT / "outputs" / "push_4cm_allin"
OUT.mkdir(parents=True, exist_ok=True)

EXPORTS = [
    (
        "4cm_seed11",
        ROOT / "outputs/4cm_full_power_ensemble/checkpoints/seed_11/best.ckpt",
        ROOT / "configs/pibnn_rtx3060_4cm_FULL_POWER.yaml",
    ),
    ("v4_combo_ssl", ROOT / "outputs/v4_combo_full_ssl/checkpoints/best.ckpt", None),
    ("v4_target_ssl", ROOT / "outputs/v4_target_ssl/checkpoints/best.ckpt", None),
    (
        "fullpower_seed17",
        ROOT / "outputs/full_power_ensemble/checkpoints/seed_17/best.ckpt",
        ROOT / "outputs/full_power_ensemble/configs/seed_17.yaml",
    ),
    (
        "fullpower_seed23",
        ROOT / "outputs/full_power_ensemble/checkpoints/seed_23/best.ckpt",
        ROOT / "outputs/full_power_ensemble/configs/seed_23.yaml",
    ),
    (
        "epoch3cm_seed11",
        ROOT / "outputs/epoch_ensemble_gpu_3cm_push/checkpoints/seed_11/best.ckpt",
        ROOT / "outputs/epoch_ensemble_gpu_3cm_push/configs/seed_11.yaml",
    ),
]


def run(cmd: list[str], log_name: str) -> int:
    log_path = OUT / log_name
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
        code = int(proc.wait())
    print(f"[exit {code}] log -> {log_path}", flush=True)
    return code


def export_checkpoints() -> None:
    export_root = OUT / "gpu_exports"
    export_root.mkdir(parents=True, exist_ok=True)
    for label, ckpt, cfg in EXPORTS:
        if not ckpt.exists():
            print(f"[skip export] missing {ckpt}", flush=True)
            continue
        out_dir = export_root / label
        cmd = [
            PY,
            str(ROOT / "scripts/export_checkpoint_predictions.py"),
            "--checkpoint",
            str(ckpt),
            "--output-dir",
            str(out_dir),
            "--device",
            "cuda",
            "--use-ema",
        ]
        if cfg is not None:
            cmd.extend(["--config", str(cfg)])
        run(cmd, f"export_{label}.log")


def main() -> int:
    print("=" * 64)
    print("  VocalMorph 4CM ALL-IN PUSH")
    print("=" * 64)

    steps = [
        # Skip re-export if 4cm preds already exist; gauntlet auto-discovers all CSVs.
        ("nuclear_3cm_push", [PY, str(ROOT / "scripts/nuclear_3cm_push.py")]),
        ("blend_3cm_posthoc", [PY, str(ROOT / "scripts/blend_3cm_posthoc.py")]),
        (
            "stacking_meta",
            [
                PY,
                str(ROOT / "scripts/stacking_meta_ensemble.py"),
                "--output-dir",
                str(ROOT / "outputs/stacking_meta_ensemble"),
                "--device",
                "cuda",
            ],
        ),
        (
            "phase22_gauntlet",
            [
                PY,
                str(ROOT / "scripts/phase22_3cm_reality_gauntlet.py"),
                "--output-dir",
                str(ROOT / "outputs/phase22_4cm_gauntlet"),
                "--target-mae",
                "4.0",
                "--device",
                "cuda",
                "--blend-probes",
                "200000",
            ],
        ),
    ]

    # Only export the working 4cm checkpoint; others use per-seed member configs.
    export_checkpoints()

    results: dict[str, int] = {}
    for name, cmd in steps:
        results[name] = run(cmd, f"{name}.log")

    summary_path = ROOT / "outputs/phase22_4cm_gauntlet/phase22_summary.json"
    gauntlet_best = None
    if summary_path.exists():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        deploy = payload.get("deployable", {})
        gauntlet_best = {
            "val_mae": deploy.get("val_mae"),
            "test_mae": deploy.get("test_mae"),
            "method": deploy.get("method"),
        }

    stack_path = ROOT / "outputs/stacking_meta_ensemble/stacking_report.json"
    stack_best = None
    if stack_path.exists():
        payload = json.loads(stack_path.read_text(encoding="utf-8"))
        stack_best = {"val_mae": payload.get("val", {}).get("mae"), "test_mae": payload.get("test", {}).get("mae")}

    blend_path = ROOT / "outputs/blend_3cm_posthoc/summary.json"
    blend_best = None
    if blend_path.exists():
        payload = json.loads(blend_path.read_text(encoding="utf-8"))
        blend_best = {
            "val_mae": payload.get("val", {}).get("blend_mae_cm"),
            "test_mae": payload.get("test", {}).get("blend_mae_cm"),
        }

    nuclear_path = ROOT / "outputs/nuclear_3cm_push/summary.json"
    nuclear_best = None
    if nuclear_path.exists():
        payload = json.loads(nuclear_path.read_text(encoding="utf-8"))
        nuclear_best = {"val_mae": payload.get("val", {}).get("final_mae"), "test_mae": payload.get("test", {}).get("final_mae")}

    report = {
        "target_cm": 4.0,
        "step_exit_codes": results,
        "best_sources": {
            "phase22_gauntlet": gauntlet_best,
            "stacking_meta": stack_best,
            "blend_posthoc": blend_best,
            "nuclear": nuclear_best,
        },
    }
    report_path = OUT / "PUSH_4CM_ALLIN_REPORT.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 64)
    print("  4CM ALL-IN SUMMARY")
    print("=" * 64)
    for key, val in report["best_sources"].items():
        if val:
            print(f"  {key:22s}  test={val.get('test_mae', '?'):>7}  val={val.get('val_mae', '?'):>7}")
    print(f"\nReport: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())