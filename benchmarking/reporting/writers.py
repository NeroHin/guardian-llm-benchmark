from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_rows_csv(df: pd.DataFrame, path: Path) -> None:
    _ensure_parent(path)
    df.to_csv(path, index=False, encoding="utf-8")


def write_metrics_json(payload: dict[str, Any], path: Path) -> None:
    _ensure_parent(path)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_leaderboard_csv(rows: list[dict[str, Any]], path: Path) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if not df.empty and {"tpr", "fpr", "overhead_latency_ms_avg", "cost_usd_total"}.issubset(df.columns):
        df = df.sort_values(
            by=["tpr", "fpr", "overhead_latency_ms_avg", "cost_usd_total"],
            ascending=[False, True, True, True],
        ).reset_index(drop=True)
    _ensure_parent(path)
    df.to_csv(path, index=False, encoding="utf-8")
    return df

def write_run_manifest(payload: dict[str, Any], path: Path) -> None:
    write_metrics_json(payload, path)
