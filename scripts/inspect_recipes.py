"""Inspect winning recipe selection from VoxPhysica strict research results."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    path = Path("outputs/strict_3cm_short_opt/strict_results.json")
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        winner = data.get("selection", {}).get("frozen_recipe", {}).get("name")
        print("Selection info:")
        print(f"Winner: {winner}")
    else:
        print(f"File not found: {path}")


if __name__ == "__main__":
    main()
