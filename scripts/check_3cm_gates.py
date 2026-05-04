#!/usr/bin/env python
"""Check whether a VocalMorph run is plausibly on a 3cm-MAE path."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate 3cm research gates from a metrics JSON.")
    parser.add_argument("metrics_json")
    parser.add_argument("--target-mae", type=float, default=3.0)
    parser.add_argument("--max-short-mae", type=float, default=5.5)
    parser.add_argument("--max-balanced-mae", type=float, default=4.2)
    parser.add_argument("--max-val-test-gap", type=float, default=0.7)
    return parser.parse_args()


def _get(mapping: Mapping[str, Any], key: str, default: float = math.nan) -> float:
    try:
        return float(mapping.get(key, default))
    except Exception:
        return float(default)


def _fmt(value: float) -> str:
    return "nan" if not math.isfinite(value) else f"{value:.3f}"


def _nanmin(*values: float) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return min(finite) if finite else float("nan")


def main() -> int:
    args = parse_args()
    path = Path(args.metrics_json)
    payload = json.loads(path.read_text(encoding="utf-8"))
    val = payload.get("final_val", payload.get("val", {})) or {}
    test = payload.get("final_test", payload.get("test", {})) or {}

    val_spk = _nanmin(
        _get(val, "height_mae_speaker"),
        _get(val, "height_mae_speaker_omega"),
        _get(val, "height_mae_speaker_balanced"),
        _get(val, "height_mae_speaker_balanced_omega"),
    )
    test_spk = _nanmin(_get(test, "height_mae_speaker"), _get(test, "height_mae_speaker_omega"))
    test_bal = _nanmin(
        _get(test, "height_mae_speaker_balanced"),
        _get(test, "height_mae_speaker_balanced_omega"),
    )
    test_short = _nanmin(
        _get(test, "height_heightbin_short_speaker_mae"),
        _get(test, "height_heightbin_short_speaker_mae_omega"),
    )
    gap = test_spk - val_spk

    checks = [
        ("sealed_test_speaker_mae", test_spk <= float(args.target_mae), test_spk, float(args.target_mae)),
        ("short_tail_speaker_mae", test_short <= float(args.max_short_mae), test_short, float(args.max_short_mae)),
        ("balanced_speaker_mae", test_bal <= float(args.max_balanced_mae), test_bal, float(args.max_balanced_mae)),
        ("val_test_gap", gap <= float(args.max_val_test_gap), gap, float(args.max_val_test_gap)),
    ]

    print(f"[3cm-gates] {path}")
    for name, ok, value, threshold in checks:
        verdict = "PASS" if ok else "BLOCK"
        print(f"  {verdict:5s} {name}: value={_fmt(value)} threshold={_fmt(threshold)}")

    if all(ok for _, ok, _, _ in checks):
        print("[3cm-gates] RUN IS ON A DEFENSIBLE 3CM PATH")
        return 0
    print("[3cm-gates] NOT ON A DEFENSIBLE 3CM PATH")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
