#!/usr/bin/env python
"""Clean regenerable artifacts, rebuild audited features, and run live research."""

from __future__ import annotations

import os
import stat
import shutil
import subprocess
import sys
import time
from typing import List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GPU_PYTHON = r"C:\Users\USER\anaconda3\python.exe"

SAFE_REMOVE_DIRS = (
    "outputs",
    os.path.join("src", "outputs"),
    "Trash",
    ".pytest_cache",
    os.path.join("data", "features_audited"),
    os.path.join("data", "audio_clean"),
    os.path.join("data", "audio_quarantine"),
    os.path.join("data", "normalized"),
)


def _resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.abspath(os.path.join(ROOT, path))


def _ensure_under_root(path: str) -> str:
    resolved = _resolve(path)
    root_norm = os.path.normcase(os.path.abspath(ROOT))
    path_norm = os.path.normcase(resolved)
    if path_norm != root_norm and not path_norm.startswith(root_norm + os.sep):
        raise RuntimeError(f"Refusing to touch path outside workspace: {resolved}")
    return resolved


def _dir_size_bytes(path: str) -> int:
    total = 0
    if not os.path.exists(path):
        return 0
    for dirpath, _, filenames in os.walk(path):
        for name in filenames:
            full_path = os.path.join(dirpath, name)
            try:
                total += os.path.getsize(full_path)
            except OSError:
                continue
    return total


def _format_gb(size_bytes: int) -> str:
    return f"{size_bytes / (1024 ** 3):.2f} GB"


def _remove_dir(path: str) -> int:
    resolved = _ensure_under_root(path)
    if not os.path.exists(resolved):
        return 0
    size_bytes = _dir_size_bytes(resolved)
    shutil.rmtree(resolved, ignore_errors=False, onexc=_on_rmtree_error)
    return size_bytes


def _remove_path(path: str) -> None:
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path, ignore_errors=False, onexc=_on_rmtree_error)
    else:
        try:
            os.chmod(path, stat.S_IWRITE)
        except OSError:
            pass
        os.remove(path)


def _clear_dir(path: str) -> int:
    resolved = _ensure_under_root(path)
    if not os.path.exists(resolved):
        return 0
    size_bytes = _dir_size_bytes(resolved)
    for name in os.listdir(resolved):
        child = os.path.join(resolved, name)
        _remove_path(child)
    return size_bytes


def _on_rmtree_error(func, path: str, excinfo) -> None:
    try:
        os.chmod(path, stat.S_IWRITE)
    except OSError:
        pass
    func(path)


def _remove_pycache_dirs(root: str) -> int:
    total = 0
    resolved_root = _ensure_under_root(root)
    for dirpath, dirnames, _ in os.walk(resolved_root):
        if "__pycache__" in dirnames:
            cache_dir = os.path.join(dirpath, "__pycache__")
            total += _remove_dir(cache_dir)
            dirnames[:] = [name for name in dirnames if name != "__pycache__"]
    return total


def _run(cmd: List[str]) -> None:
    print("\n[LiveResearch] Running:")
    print("  " + " ".join(cmd))
    sys.stdout.flush()
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> int:
    if not os.path.exists(GPU_PYTHON):
        print(f"[LiveResearch] Missing GPU python: {GPU_PYTHON}", file=sys.stderr)
        return 2

    print("[LiveResearch] Workspace:", ROOT)
    print("[LiveResearch] Python:", GPU_PYTHON)
    print("[LiveResearch] Cleaning regenerable artifacts to free storage...")
    freed = 0
    for rel_path in SAFE_REMOVE_DIRS:
        resolved = _resolve(rel_path)
        if os.path.exists(resolved):
            size_bytes = _dir_size_bytes(resolved)
            print(f"  remove {resolved} ({_format_gb(size_bytes)})")
            freed += _clear_dir(rel_path)
        else:
            print(f"  skip   {resolved} (missing)")
    pycache_freed = _remove_pycache_dirs(ROOT)
    if pycache_freed:
        print(f"  remove __pycache__ dirs ({_format_gb(pycache_freed)})")
        freed += pycache_freed
    print(f"[LiveResearch] Freed about {_format_gb(freed)}")
    sys.stdout.flush()

    time.sleep(1.0)

    try:
        build_cmd = [
            GPU_PYTHON,
            os.path.join(ROOT, "scripts", "build_feature_splits.py"),
            "--config",
            os.path.join(ROOT, "configs", "pibnn_base.yaml"),
            "--output_dir",
            os.path.join(ROOT, "data", "features_audited"),
            "--overwrite",
            "--allow_manifest_drift",
            "--skip_audio_hash_check",
            "--augment_train_factor",
            "0",
            "--augment_timit_factor",
            "0",
        ]
        _run(build_cmd)

        research_cmd = [
            GPU_PYTHON,
            os.path.join(ROOT, "scripts", "run_research_height_ensemble.py"),
            "--features-dir",
            os.path.join(ROOT, "data", "features_audited"),
            "--output-dir",
            os.path.join(ROOT, "outputs", "research_height_ensemble", "seed_11"),
            "--seed",
            "11",
            "--target-mae-cm",
            "3.0",
            "--ensemble-trials",
            "5000",
        ]
        _run(research_cmd)
    except subprocess.CalledProcessError as exc:
        print(f"\n[LiveResearch] Pipeline stopped with exit code {exc.returncode}.", file=sys.stderr)
        return int(exc.returncode or 1)

    print("\n[LiveResearch] Finished.")
    print("[LiveResearch] Notebook: notebooks\\voxphysica_3cm_research_pipeline.ipynb")
    print("[LiveResearch] Metrics: outputs\\research_height_ensemble\\seed_11\\metrics.json")
    print("[LiveResearch] Summary: outputs\\research_height_ensemble\\seed_11\\summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
