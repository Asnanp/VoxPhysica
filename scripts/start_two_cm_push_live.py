#!/usr/bin/env python
"""Generate config, launch detached training, and open a live log watcher."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Iterable

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNNER = os.path.join(ROOT, "scripts", "run_two_cm_push.py")
TRAIN = os.path.join(ROOT, "scripts", "train.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the real-world two-cm push in a detached process and open a live log tail."
    )
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="outputs/two_cm_push_realworld")
    parser.add_argument("--python", default=os.path.join(ROOT, ".venv-gpu", "Scripts", "python.exe"))
    parser.add_argument("--no-watch", action="store_true")
    return parser.parse_args()


def _remove_if_exists(paths: Iterable[str]) -> None:
    for path in paths:
        if os.path.exists(path):
            os.remove(path)


def _creation_flags() -> int:
    if os.name != "nt":
        return 0
    return int(getattr(subprocess, "DETACHED_PROCESS", 0)) | int(
        getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    )


def _new_console_flag() -> int:
    if os.name != "nt":
        return 0
    return int(getattr(subprocess, "CREATE_NEW_CONSOLE", 0))


def _resolve_python(path: str) -> str:
    candidate = path if os.path.isabs(path) else os.path.join(ROOT, path)
    candidate = os.path.abspath(candidate)
    if not os.path.exists(candidate):
        raise FileNotFoundError(f"Python executable not found: {candidate}")
    return candidate


def _run_dir(output_dir: str, seed: int) -> str:
    base = output_dir if os.path.isabs(output_dir) else os.path.join(ROOT, output_dir)
    return os.path.join(os.path.abspath(base), f"seed_{seed}")


def main() -> int:
    args = parse_args()
    python_exe = _resolve_python(args.python)
    run_dir = _run_dir(args.output_dir, args.seed)
    os.makedirs(run_dir, exist_ok=True)

    generator_cmd = [
        python_exe,
        RUNNER,
        "--seeds",
        str(args.seed),
        "--python",
        python_exe,
        "--device",
        args.device,
        "--output-dir",
        args.output_dir,
    ]
    subprocess.run(generator_cmd, check=True, cwd=ROOT)

    config_path = os.path.join(run_dir, "config.yaml")
    train_stdout = os.path.join(run_dir, "train.stdout.log")
    train_stderr = os.path.join(run_dir, "train.stderr.log")
    metrics_json = os.path.join(run_dir, "metrics.json")
    metrics_jsonl = os.path.join(run_dir, "metrics.jsonl")
    launcher_stdout = os.path.join(run_dir, "launcher.stdout.log")
    launcher_stderr = os.path.join(run_dir, "launcher.stderr.log")
    launch_info = os.path.join(run_dir, "live_launch.json")

    _remove_if_exists(
        [
            train_stdout,
            train_stderr,
            metrics_json,
            metrics_jsonl,
            launcher_stdout,
            launcher_stderr,
            launch_info,
        ]
    )

    stdout_handle = open(launcher_stdout, "ab", buffering=0)
    stderr_handle = open(launcher_stderr, "ab", buffering=0)

    train_cmd = [
        python_exe,
        TRAIN,
        "--config",
        config_path,
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--metrics-out",
        metrics_json,
    ]
    proc = subprocess.Popen(
        train_cmd,
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stdout=stdout_handle,
        stderr=stderr_handle,
        creationflags=_creation_flags(),
        close_fds=True,
    )
    stdout_handle.close()
    stderr_handle.close()

    payload = {
        "pid": int(proc.pid),
        "seed": int(args.seed),
        "device": args.device,
        "python": python_exe,
        "config": config_path,
        "train_stdout": train_stdout,
        "train_stderr": train_stderr,
        "launcher_stdout": launcher_stdout,
        "launcher_stderr": launcher_stderr,
    }
    with open(launch_info, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"[LiveLaunch] PID: {proc.pid}")
    print(f"[LiveLaunch] Config: {config_path}")
    print(f"[LiveLaunch] Train log: {train_stdout}")
    print(f"[LiveLaunch] Error log: {train_stderr}")

    if not args.no_watch and os.name == "nt":
        watch_cmd = (
            f"Get-Content '{train_stdout}' -Tail 40 -Wait"
        )
        subprocess.Popen(
            ["powershell.exe", "-NoExit", "-Command", watch_cmd],
            cwd=ROOT,
            creationflags=_new_console_flag(),
            close_fds=True,
        )
        print("[LiveLaunch] Opened a PowerShell log watcher window.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
