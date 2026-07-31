"""Inspect and rank all candidate recipes by MAE, RMSE, and 3cm error ratio."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    results_path = Path("outputs/strict_3cm_short_opt/strict_results.json")
    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        return

    with results_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    rec_metrics = data.get("selection", {}).get("validation_recipe_metrics", {})
    for name, m in sorted(rec_metrics.items(), key=lambda item: item[1]["mae_cm"]):
        mae = m.get("mae_cm", float("nan"))
        rmse = m.get("rmse_cm", float("nan"))
        w3 = m.get("within_3cm", 0.0) * 100.0
        print(f"{name:35s} | MAE: {mae:.3f} cm | RMSE: {rmse:.3f} cm | within3: {w3:.1f}%")


if __name__ == "__main__":
    main()
