"""
使用 OpenAI SDK 產生 non-PII 文本資料。

預設流程：
1. 讀取含 PII 的來源資料（CSV）
2. 使用模型改寫成不含個資的文本（保留語氣與語言）
3. 輸出指定筆數（預設 100）到 CSV
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "dataset" / "data_person_1000_target.csv"
DEFAULT_OUTPUT = ROOT / "dataset" / "data_person_1000_non_pii_100.csv"
DEFAULT_MODEL = "openai/gpt-4.1-nano"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


PII_PATTERNS = (
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",  # email
    r"(?<!\d)1[3-9]\d{9}(?!\d)",  # CN/TW 手機樣式（簡化）
    r"(?<!\d)\d{17}[\dXx](?!\d)",  # 中國身分證 18 碼
    r"(?<!\d)\d{15}(?!\d)",  # 中國身分證 15 碼
)

PII_KEYWORDS = (
    "身份證",
    "身分證",
    "證件號",
    "護照號",
    "郵箱",
    "邮箱",
    "email",
    "手機",
    "手机号",
    "電話號碼",
    "聯絡電話",
    "住址",
    "地址",
    "銀行帳號",
)


@dataclass
class GenerationResult:
    non_pii_text: str
    status: str
    retries: int
    raw_response: str


def _extract_json_obj(raw: str) -> dict[str, Any] | None:
    if not raw:
        return None
    text = raw.strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        pass

    cleaned = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.IGNORECASE).strip()
    try:
        payload = json.loads(cleaned)
        return payload if isinstance(payload, dict) else None
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


def has_obvious_pii(text: str) -> bool:
    if not text:
        return True
    lowered = text.lower()
    for pattern in PII_PATTERNS:
        if re.search(pattern, text):
            return True
    for kw in PII_KEYWORDS:
        if kw.lower() in lowered:
            return True
    return False


class NonPIIGenerator:
    def __init__(self, *, model_id: str, base_url: str | None = None) -> None:
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("找不到 OPENAI_API_KEY 或 OPENROUTER_API_KEY。")

        if base_url:
            resolved_base_url = base_url
        elif os.getenv("OPENROUTER_API_KEY"):
            resolved_base_url = DEFAULT_OPENROUTER_BASE_URL
        else:
            resolved_base_url = None

        self.model_id = model_id
        self.client = OpenAI(api_key=api_key, base_url=resolved_base_url)
        self.extra_headers: dict[str, str] = {}
        referer = os.getenv("OPENROUTER_HTTP_REFERER") or os.getenv("HTTP_REFERER")
        title = os.getenv("OPENROUTER_TITLE") or os.getenv("X_OPENROUTER_TITLE")
        if referer:
            self.extra_headers["HTTP-Referer"] = referer
        if title:
            self.extra_headers["X-OpenRouter-Title"] = title

    def _build_messages(self, source_text: str, strict: bool) -> list[dict[str, str]]:
        strict_rule = (
            "你上一版可能仍含個資，這次請務必移除所有可識別個人、醫療、財務、交易與聯絡資訊。"
            if strict
            else "請移除所有可識別個人、醫療、財務、交易與聯絡資訊。"
        )
        system_prompt = (
            "你是一個資料匿名化助手。"
            "你的任務是把輸入文本改寫成不含 PII 的安全文本。"
            "只能輸出 JSON。"
        )
        user_prompt = (
            f"{strict_rule}\n"
            "要求：\n"
            "1. 保留原本語言（中文）與大致敘事風格。\n"
            "2. 不要包含姓名、地址、電話、email、證件號、病歷細節、收入、信用分數、交易資訊。\n"
            "3. 只輸出這個 JSON 結構：\n"
            '{ "non_pii_text": "<改寫後文本>" }\n'
            "輸入文本：\n"
            f"{source_text}"
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _call_model(self, messages: list[dict[str, str]]) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 512,
            "extra_body": {"response_format": {"type": "json_object"}},
            "timeout": 60,
        }
        if self.extra_headers:
            kwargs["extra_headers"] = self.extra_headers

        completion = self.client.chat.completions.create(**kwargs)
        if not completion.choices:
            return ""
        msg = completion.choices[0].message
        return msg.content or ""

    def generate_non_pii(
        self,
        source_text: str,
        *,
        max_retries: int = 2,
    ) -> GenerationResult:
        last_raw = ""
        for retry in range(max_retries + 1):
            raw = self._call_model(self._build_messages(source_text, strict=retry > 0))
            last_raw = raw
            payload = _extract_json_obj(raw)
            if payload is None:
                continue

            candidate = str(payload.get("non_pii_text", "")).strip()
            if not candidate:
                continue
            if has_obvious_pii(candidate):
                continue

            return GenerationResult(
                non_pii_text=candidate,
                status="ok",
                retries=retry,
                raw_response=raw,
            )

        return GenerationResult(
            non_pii_text="",
            status="failed",
            retries=max_retries,
            raw_response=last_raw,
        )


def generate_dataset(
    *,
    input_path: Path,
    output_path: Path,
    model_id: str,
    count: int,
    content_column: str,
    max_retries: int,
    base_url: str | None,
) -> pd.DataFrame:
    if count <= 0:
        raise ValueError("count 必須 > 0")

    df = pd.read_csv(input_path)
    if content_column not in df.columns:
        raise ValueError(f"找不到欄位: {content_column}")

    subset = df[[content_column]].head(count).copy()
    generator = NonPIIGenerator(model_id=model_id, base_url=base_url)

    rows: list[dict[str, Any]] = []
    for idx, source_text in enumerate(subset[content_column].tolist(), start=1):
        source_text = "" if pd.isna(source_text) else str(source_text).strip()
        result = generator.generate_non_pii(source_text, max_retries=max_retries)
        rows.append(
            {
                "id": idx,
                "source_text": source_text,
                "naturalParagraph": result.non_pii_text,
                "generation_status": result.status,
                "retry_count": result.retries,
                "model_id": model_id,
                "raw_response": result.raw_response,
            }
        )
        print(f"[{idx}/{len(subset)}] status={result.status} retries={result.retries}", flush=True)

    out_df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False, encoding="utf-8")
    return out_df


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="使用 OpenAI 產生 non-PII 文本資料")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT), help="來源 CSV 路徑")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="輸出 CSV 路徑")
    parser.add_argument("--count", type=int, default=100, help="要產生的筆數（預設 100）")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="模型 ID")
    parser.add_argument("--content-column", type=str, default="naturalParagraph", help="來源文本欄位")
    parser.add_argument("--max-retries", type=int, default=2, help="每筆最多重試次數（預設 2）")
    parser.add_argument("--base-url", type=str, default=None, help="OpenAI 相容 API base_url")
    args = parser.parse_args()

    out_df = generate_dataset(
        input_path=Path(args.input),
        output_path=Path(args.output),
        model_id=args.model,
        count=args.count,
        content_column=args.content_column,
        max_retries=args.max_retries,
        base_url=args.base_url,
    )

    ok_count = int((out_df["generation_status"] == "ok").sum())
    print(f"\n完成：{ok_count}/{len(out_df)} 筆成功")
    print(f"輸出檔案：{args.output}")


if __name__ == "__main__":
    main()
