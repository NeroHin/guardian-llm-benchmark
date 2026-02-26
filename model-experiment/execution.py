"""
Granite 4 模型 PII 二分類評測實驗。

1. 載入 dataset/data_person_1000_target.csv 的資料於比較 pii_binary 任務的準確率
2. 比較的模型：granite4:1b-h, granite4:micro-h, granite4:tiny-h
3. 比較的結果：Accuracy, Recall, F1 Score, Execution Time, Model Parameters
4. 使用 ollama 載入模型，並使用 build_prompts 組裝 pii_binary 任務的 prompt
5. 使用 utils/result_process.py, utils/result_compare.py 處理結果
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import ollama
import pandas as pd

# 專案根目錄加入 path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from prompt import build_prompts
from utils.result_compare import (
    build_comparison_table,
    format_comparison_report,
    load_model_params,
)
from utils.result_process import (
    process_model_run,
    save_processed_results,
)

# =============================================================================
# 設定
# =============================================================================

DATASET_PATH = ROOT / "dataset" / "data_person_1000_target.csv"
RESULTS_DIR = ROOT / "model-experiment" / "results"
TASK_ID = "pii_binary"
CONTENT_COLUMN = "naturalParagraph"

MODELS = [
    "granite4:1b-h",
    # "granite4:micro-h",
    # "granite4:tiny-h",
]

PII_BINARY_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "contains_pii": {"type": "boolean"},
        "label": {"type": "string", "enum": ["是", "否"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string"},
    },
    "required": ["contains_pii", "label", "confidence", "reason"],
    "additionalProperties": False,
}


def _extract_json_object(raw_text: str) -> dict[str, Any] | None:
    """
    從模型輸出中擷取 JSON 物件。

    支援：
    1) 純 JSON 文字
    2) 包含 ```json ... ``` 的輸出
    3) 文字中夾帶第一個 {...} JSON 片段
    """
    if not raw_text:
        return None

    text = raw_text.strip()
    if not text:
        return None

    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    cleaned = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.IGNORECASE).strip()

    try:
        payload = json.loads(cleaned)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return None

    try:
        payload = json.loads(match.group(0))
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        return None


def _extract_partial_value(raw_text: str, key: str) -> str | None:
    """從不完整 JSON 文字中抽取欄位值（字串型）。"""
    if not raw_text:
        return None
    pattern = rf'"{re.escape(key)}"\s*:\s*"([^"\n\r}}]*)'
    match = re.search(pattern, raw_text, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1).strip()


def _extract_partial_number(raw_text: str, key: str) -> float | None:
    """從不完整 JSON 文字中抽取數值欄位。"""
    if not raw_text:
        return None
    pattern = rf'"{re.escape(key)}"\s*:\s*([+-]?\d+(?:\.\d+)?)'
    match = re.search(pattern, raw_text, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _extract_partial_contains_pii(raw_text: str) -> bool | None:
    """從不完整 JSON 或文字中推斷 contains_pii。"""
    if not raw_text:
        return None

    # 優先抓 JSON 欄位值
    match = re.search(r'"contains_pii"\s*:\s*(true|false|1|0)', raw_text, flags=re.IGNORECASE)
    if match:
        return _coerce_bool(match.group(1))

    # 次優先抓 label 欄位
    label = _extract_partial_value(raw_text, "label")
    parsed = _coerce_bool(label)
    if parsed is not None:
        return parsed

    # 最後以整段文字關鍵字推斷
    lowered = raw_text.lower()
    if "contains_pii" in lowered and "true" in lowered:
        return True
    if "contains_pii" in lowered and "false" in lowered:
        return False
    return None


def _coerce_bool(value: Any) -> bool | None:
    """將常見布林語意值轉成 bool。"""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"是", "有", "yes", "true", "1", "y"}:
            return True
        if text in {"否", "無", "no", "false", "0", "n"}:
            return False
    return None


def _coerce_contains_pii(payload: dict[str, Any] | None) -> bool | None:
    """從 JSON payload 中推斷 contains_pii。"""
    if not payload:
        return None

    for key in ("contains_pii", "has_pii", "pii", "prediction", "label", "result"):
        if key in payload:
            parsed = _coerce_bool(payload.get(key))
            if parsed is not None:
                return parsed
    return None


def normalize_pii_binary_json(raw_text: str) -> dict[str, Any]:
    """
    將模型輸出正規化為統一 JSON。
    """
    payload = _extract_json_object(raw_text)
    contains_pii = _coerce_contains_pii(payload)
    if contains_pii is None:
        contains_pii = _extract_partial_contains_pii(raw_text)

    confidence = None
    if payload is not None and "confidence" in payload:
        try:
            confidence_val = float(payload["confidence"])
            confidence = min(1.0, max(0.0, confidence_val))
        except (TypeError, ValueError):
            confidence = None
    if confidence is None:
        partial_conf = _extract_partial_number(raw_text, "confidence")
        if partial_conf is not None:
            confidence = min(1.0, max(0.0, partial_conf))

    reason = ""
    if payload is not None and "reason" in payload:
        reason = str(payload["reason"]).strip()
    if not reason:
        reason = _extract_partial_value(raw_text, "reason") or ""

    normalized = {
        "contains_pii": contains_pii,
        "label": "是" if contains_pii is True else "否" if contains_pii is False else "無法解析",
        "confidence": confidence,
        "reason": reason,
        "raw_text": raw_text,
    }
    return normalized


def run_single_model(
    df: pd.DataFrame,
    model_id: str,
    *,
    sample_limit: int | None = None,
) -> tuple[list[str], float, dict]:
    """
    對單一模型執行 pii_binary 推論。

    Returns:
        (model_outputs, execution_time_sec, extra_info)
    """
    data = df if sample_limit is None else df.head(sample_limit)
    outputs: list[str] = []
    start = time.perf_counter()

    for idx, row in data.iterrows():
        content = row[CONTENT_COLUMN]
        if pd.isna(content):
            outputs.append(
                json.dumps(
                    {
                        "contains_pii": None,
                        "label": "無法解析",
                        "confidence": None,
                        "reason": "empty content",
                        "raw_text": "",
                    },
                    ensure_ascii=False,
                )
            )
            continue

        prompts = build_prompts(
            model_id=model_id,
            task_id=TASK_ID,
            content=str(content).strip(),
        )
        messages = [
            {"role": "system", "content": prompts["system"]},
            {"role": "user", "content": prompts["user"]},
        ]

        try:
            try:
                response = ollama.chat(
                    model=model_id,
                    messages=messages,
                    stream=False,
                    format=PII_BINARY_JSON_SCHEMA,
                )
            except TypeError:
                response = ollama.chat(
                    model=model_id,
                    messages=messages,
                    stream=False,
                    format="json",
                )
            text = response.message.content if response.message else ""
            normalized = normalize_pii_binary_json(text or "")
            outputs.append(json.dumps(normalized, ensure_ascii=False))
        except Exception as e:
            outputs.append(
                json.dumps(
                    {
                        "contains_pii": None,
                        "label": "無法解析",
                        "confidence": None,
                        "reason": f"model_error: {e}",
                        "raw_text": "",
                    },
                    ensure_ascii=False,
                )
            )

    elapsed = time.perf_counter() - start
    extra = {
        "model_id": model_id,
        "execution_time_sec": round(elapsed, 2),
        "model_params_b": load_model_params(model_id),
    }
    return outputs, elapsed, extra


def main(
    *,
    sample_limit: int | None = None,
) -> None:
    """主流程：載入資料、跑模型、處理結果、產出比較報告。"""
    # 1. 載入資料
    df = pd.read_csv(DATASET_PATH)
    data = df if sample_limit is None else df.head(sample_limit)
    print(f"載入 {len(data)} 筆資料")

    # 2. 建立輸出目錄
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 3. 對每個模型執行推論
    all_metrics: list[dict] = []

    for model_id in MODELS:
        print(f"\n執行模型: {model_id} ...")
        outputs, elapsed, extra = run_single_model(
            df,
            model_id,
            sample_limit=sample_limit,
        )

        # 4. 處理結果、計算指標
        result_df, metrics = process_model_run(
            data,
            outputs,
            TASK_ID,
            content_column=CONTENT_COLUMN,
        )

        if metrics:
            meta = {**metrics.to_dict(), **extra}
            all_metrics.append(meta)
            # Recall = 偵測率：含 PII 樣本中被正確判定的比例
            print(f"  偵測率 (Recall): {metrics.recall:.4f} ({metrics.tp}/{metrics.total})")
            print(f"  Execution: {elapsed:.2f}s")

        # 5. 儲存單一模型結果
        safe_name = model_id.replace("/", "_").replace(":", "_")
        out_csv = RESULTS_DIR / f"{safe_name}_pii_binary_results.csv"
        save_processed_results(
            result_df,
            metrics,
            str(out_csv),
            extra_info=extra,
        )
        print(f"  結果已存: {out_csv}")

    # 6. 比較多模型、產出報告
    # 資料集全為含 PII 樣本，主要比較：各模型能否正確偵測（Recall = 正確「是」數 / 總數）
    if all_metrics:
        compare_df = build_comparison_table(
            all_metrics,
            model_id_key="model_id",
            sort_by="recall",
        )
        compare_path = RESULTS_DIR / "granite4_pii_binary_comparison.csv"
        compare_df.to_csv(compare_path, index=False, encoding="utf-8")
        print(f"\n比較表已存: {compare_path}")
        print("\n【PII 偵測能力比較】Recall = 正確判定「是」的比例（越高越好）")
        print(format_comparison_report(
            compare_df,
            columns=["model_id", "recall", "tp", "total", "accuracy", "f1", "execution_time_sec", "model_params_b"],
        ))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Granite 4 PII 二分類評測")
    parser.add_argument(
        "-n",
        "--sample-limit",
        type=int,
        default=None,
        help="限制樣本數（用於快速測試，預設不限制）",
    )
    args = parser.parse_args()
    main(sample_limit=args.sample_limit)
