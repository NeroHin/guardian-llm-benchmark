"""
比較多個模型的評測結果，產生對照表與報告。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


# =============================================================================
# 模型參數對照（可依專案擴充）
# =============================================================================

MODEL_PARAMS: dict[str, float] = {
    "granite4:1b-h": 1.0,
    "granite4:micro-h": 3.0,
    "granite4:tiny-h": 7.0,
    "granite4:1b": 1.0,
    "granite4:micro": 3.0,
    "granite4:tiny": 7.0,
}


def load_model_params(model_id: str) -> float | None:
    """取得模型參數量（B），未知則回傳 None。"""
    return MODEL_PARAMS.get(model_id)


def register_model_params(model_id: str, params_b: float) -> None:
    """註冊模型參數量。"""
    MODEL_PARAMS[model_id] = params_b


# =============================================================================
# 載入與彙整
# =============================================================================


def load_metrics_json(path: str | Path) -> dict[str, Any]:
    """從 _metrics.json 載入指標。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"找不到: {p}")

    with open(p, encoding="utf-8") as f:
        return json.load(f)


def load_result_metrics(
    result_dir: str | Path,
    *,
    pattern: str = "*_metrics.json",
) -> list[dict[str, Any]]:
    """
    從目錄載入多個模型的结果 metrics JSON。

    預期檔名含 model_id，例如：granite4_1b_metrics.json
    """
    result_dir = Path(result_dir)
    if not result_dir.is_dir():
        raise NotADirectoryError(f"不是目錄: {result_dir}")

    loaded = []
    for f in result_dir.glob(pattern):
        try:
            data = load_metrics_json(f)
            data["_source_file"] = str(f.name)
            loaded.append(data)
        except (json.JSONDecodeError, Exception) as e:
            print(f"警告: 跳過 {f}: {e}")
    return loaded


def metrics_to_comparison_rows(
    metrics_list: list[dict[str, Any]],
    *,
    model_id_key: str = "model_id",
) -> list[dict[str, Any]]:
    """
    將多個 metrics 轉成比較表的一列一列資料。
    """
    rows = []
    for m in metrics_list:
        model_id = m.get(model_id_key, "unknown")
        params = m.get("model_params_b") or load_model_params(model_id)

        row = {
            "model_id": model_id,
            "accuracy": m.get("accuracy"),
            "precision": m.get("precision"),
            "recall": m.get("recall"),
            "f1": m.get("f1"),
            "execution_time_sec": m.get("execution_time_sec"),
            "model_params_b": params,
            "total": m.get("total"),
            "unparseable": m.get("unparseable"),
        }
        rows.append(row)
    return rows


def build_comparison_table(
    metrics_list: list[dict[str, Any]],
    *,
    model_id_key: str = "model_id",
    sort_by: str | None = "f1",
) -> pd.DataFrame:
    """
    建立模型比較 DataFrame。
    """
    rows = metrics_to_comparison_rows(metrics_list, model_id_key=model_id_key)
    df = pd.DataFrame(rows)

    if sort_by and sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=False).reset_index(drop=True)

    return df


def format_comparison_report(
    df: pd.DataFrame,
    *,
    columns: list[str] | None = None,
) -> str:
    """
    將比較表格式化為可讀文字報告。
    """
    if columns is None:
        columns = [
            "model_id",
            "accuracy",
            "recall",
            "f1",
            "execution_time_sec",
            "model_params_b",
        ]

    available = [c for c in columns if c in df.columns]
    sub = df[available] if available else df

    return sub.to_string(index=False)


def save_comparison(
    metrics_list: list[dict[str, Any]],
    output_path: str | Path,
    *,
    model_id_key: str = "model_id",
    sort_by: str | None = "f1",
) -> pd.DataFrame:
    """
    建立比較表並存成 CSV，回傳 DataFrame。
    """
    df = build_comparison_table(
        metrics_list,
        model_id_key=model_id_key,
        sort_by=sort_by,
    )
    df.to_csv(output_path, index=False, encoding="utf-8")
    return df
