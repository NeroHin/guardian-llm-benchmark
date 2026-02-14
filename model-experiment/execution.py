"""
Granite 4 模型 PII 二分類評測實驗。

1. 載入 dataset/data_person_1000_target.csv 的資料於比較 pii_binary 任務的準確率
2. 比較的模型：granite4:1b-h, granite4:micro-h, granite4:tiny-h
3. 比較的結果：Accuracy, Recall, F1 Score, Execution Time, Model Parameters
4. 使用 ollama 載入模型，並使用 build_prompts 組裝 pii_binary 任務的 prompt
5. 使用 utils/result_process.py, utils/result_compare.py 處理結果
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

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
            outputs.append("")
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
            response = ollama.chat(
                model=model_id,
                messages=messages,
                stream=False,
            )
            text = response.message.content if response.message else ""
            outputs.append(text or "")
        except Exception as e:
            outputs.append(f"[Error: {e}]")

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
