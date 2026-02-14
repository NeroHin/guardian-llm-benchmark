"""
處理模型推論結果，計算評估指標。

支援任務：pii_binary（二分類：是否含 PII）
輸出指標：Accuracy、Precision、Recall、F1、執行時間
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


# =============================================================================
# 輸出解析
# =============================================================================

# 用於判斷模型回覆是否為「是」（含有 PII）的關鍵字
PII_YES_PATTERNS = (
    r"^是\s*$",
    r"^是[。.，,]",
    r"[：:]\s*是",
    r"含有\s*PII",
    r"包含\s*PII",
    r"有\s*PII",
    r"存在\s*PII",
    r"yes",
    r"true",
    r"^有$",
)

# 用於判斷模型回覆是否為「否」（不含 PII）的關鍵字
PII_NO_PATTERNS = (
    r"^否\s*$",
    r"^否[。.，,]",
    r"[：:]\s*否",
    r"不含有\s*PII",
    r"不包含\s*PII",
    r"無\s*PII",
    r"沒有\s*PII",
    r"no",
    r"false",
    r"^無$",
)


def parse_pii_binary_output(raw: str) -> bool | None:
    """
    解析 pii_binary 任務的模型輸出，提取「是」或「否」。

    Args:
        raw: 模型原始輸出文字

    Returns:
        True = 判定有 PII（是）
        False = 判定無 PII（否）
        None = 無法解析
    """
    if not raw or not isinstance(raw, str):
        return None

    text = raw.strip().lower()
    if not text:
        return None

    # 擷取首句或前 50 字，避免長篇解釋干擾
    first_part = text[:80].strip()

    # 優先檢查「否」
    for pattern in PII_NO_PATTERNS:
        if re.search(pattern, first_part, re.IGNORECASE):
            return False

    # 再檢查「是」
    for pattern in PII_YES_PATTERNS:
        if re.search(pattern, first_part, re.IGNORECASE):
            return True

    # 若首字為是/否
    if first_part.startswith("是"):
        return True
    if first_part.startswith("否"):
        return False

    return None


# =============================================================================
# 指標計算
# =============================================================================


@dataclass
class BinaryMetrics:
    """二分類任務的評估指標。"""

    accuracy: float
    precision: float
    recall: float
    f1: float
    tp: int
    tn: int
    fp: int
    fn: int
    total: int
    unparseable: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "tp": self.tp,
            "tn": self.tn,
            "fp": self.fp,
            "fn": self.fn,
            "total": self.total,
            "unparseable": self.unparseable,
        }


def compute_binary_metrics(
    predictions: list[bool | None],
    ground_truths: list[bool],
    *,
    treat_unparseable_as: bool | None = False,
) -> BinaryMetrics:
    """
    計算二分類指標。

    Args:
        predictions: 模型預測（True/False/None）
        ground_truths: 真實標籤（True/False）
        treat_unparseable_as: 無法解析的輸出視為該值，預設 False（計入 FN）
    """
    if len(predictions) != len(ground_truths):
        raise ValueError(
            f"predictions ({len(predictions)}) 與 ground_truths ({len(ground_truths)}) 長度不一致"
        )

    tp = tn = fp = fn = unparseable = 0
    resolved = list[bool]()

    for pred, true in zip(predictions, ground_truths):
        if pred is None:
            unparseable += 1
            pred = treat_unparseable_as if treat_unparseable_as is not None else False
        resolved.append(pred)

        if true and pred:
            tp += 1
        elif true and not pred:
            fn += 1
        elif not true and pred:
            fp += 1
        else:
            tn += 1

    total = len(predictions)
    correct = tp + tn
    accuracy = correct / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return BinaryMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        total=total,
        unparseable=unparseable,
    )


# =============================================================================
# 結果處理
# =============================================================================


def process_pii_binary_results(
    df: pd.DataFrame,
    model_outputs: list[str],
    *,
    content_column: str = "naturalParagraph",
    ground_truth_column: str | None = None,
) -> tuple[pd.DataFrame, BinaryMetrics]:
    """
    處理 pii_binary 任務的模型輸出，計算指標。

    當資料集全為含 PII 樣本（ground_truth_column=None）時，
    Recall = TP / total = 偵測率，為比較各模型偵測能力的主要指標。

    Args:
        df: 原始資料（含 content_column）
        model_outputs: 每筆樣本對應的模型 raw 輸出
        content_column: 輸入文字欄位名稱
        ground_truth_column: 真實標籤欄位；若 None，假設全部為含 PII 正樣本

    Returns:
        (處理後的 DataFrame, BinaryMetrics)
    """
    if len(model_outputs) != len(df):
        raise ValueError(
            f"model_outputs ({len(model_outputs)}) 與 df ({len(df)}) 長度不一致"
        )

    predictions = [parse_pii_binary_output(out) for out in model_outputs]

    if ground_truth_column and ground_truth_column in df.columns:
        gt_raw = df[ground_truth_column]
        ground_truths = [
            str(v).strip().lower() in ("是", "yes", "true", "1", "有")
            for v in gt_raw
        ]
    else:
        # 預設：data_person 資料集全部為含 PII 的正樣本
        ground_truths = [True] * len(df)

    metrics = compute_binary_metrics(predictions, ground_truths)

    result_df = df[[content_column]].copy() if content_column in df.columns else df.copy()
    result_df["model_output_raw"] = model_outputs
    result_df["prediction"] = [
        "是" if p else "否" if p is False else "無法解析"
        for p in predictions
    ]
    result_df["prediction_bool"] = predictions
    result_df["ground_truth"] = ground_truths

    return result_df, metrics


def process_model_run(
    df: pd.DataFrame,
    model_outputs: list[str],
    task_id: str,
    *,
    content_column: str = "naturalParagraph",
    ground_truth_column: str | None = None,
) -> tuple[pd.DataFrame, BinaryMetrics | None]:
    """
    依任務類型處理模型輸出（可擴展至其他任務）。

    Returns:
        (result_df, metrics)
        task 不支援時 metrics 為 None
    """
    if task_id == "pii_binary":
        return process_pii_binary_results(
            df,
            model_outputs,
            content_column=content_column,
            ground_truth_column=ground_truth_column,
        )
    raise ValueError(f"尚未支援的任務: {task_id}")


# =============================================================================
# 便捷匯出
# =============================================================================


def save_processed_results(
    result_df: pd.DataFrame,
    metrics: BinaryMetrics | None,
    output_path: str,
    *,
    extra_info: dict[str, Any] | None = None,
) -> None:
    """
    將處理後的結果與指標存成 CSV 與 JSON meta。
    """
    result_df.to_csv(output_path, index=False, encoding="utf-8")
    if metrics is not None:
        import json

        meta_path = (
            output_path.replace(".csv", "_metrics.json")
            if output_path.endswith(".csv")
            else output_path + "_metrics.json"
        )
        meta = {**metrics.to_dict(), **(extra_info or {})}
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
