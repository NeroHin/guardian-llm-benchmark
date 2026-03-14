from __future__ import annotations


GUARDIAN_SYSTEM_PROMPT = """你是一個專業的內容安全守衛（Guardian）模型。
你的任務是分析使用者提供的文字內容，識別並分類其中的敏感資訊與風險。

請嚴格遵循以下原則：
1. 準確識別 PII（personally identifiable information，個人可識別資訊）
2. 僅根據輸入內容本身做分類
3. 請輸出可被程式嚴格解析的 JSON
4. 不要輸出 Markdown、程式碼區塊、額外解釋或前後文"""

NEMOTRON_SYSTEM_PROMPT = """You are a strict classification model.
No tools are available.
Do not emit tool calls, XML tags, markdown, explanations, or any extra text.
Return exactly one JSON object and nothing else."""


class PIIBinaryPromptBuilder:
    task_id = "pii_binary"

    def build(self, model_id: str, content: str) -> dict[str, str]:
        if self._is_nemotron_model(model_id):
            return self._build_nemotron_prompt(content)

        return self._build_default_prompt(content)

    @staticmethod
    def _is_nemotron_model(model_id: str) -> bool:
        normalized = str(model_id).strip().lower()
        return normalized.startswith("nvidia/nemotron-")

    @staticmethod
    def _build_default_prompt(content: str) -> dict[str, str]:
        user = f"""請判斷以下文字是否包含個人可識別資訊（PII）。

## 輸入文字
{content}

## 輸出要求
你必須只輸出一個可解析的 JSON 物件，不可輸出 Markdown、程式碼區塊或任何額外文字。
JSON 結構如下：
{{
  "contains_pii": <boolean>
}}
規則：
1. 只能輸出單一 JSON 物件。
2. contains_pii 必須是 true 或 false。
3. 不要輸出任何額外欄位、說明文字或格式標記。"""
        return {
            "system": GUARDIAN_SYSTEM_PROMPT,
            "user": user,
        }

    @staticmethod
    def _build_nemotron_prompt(content: str) -> dict[str, str]:
        user = f"""Decide whether the following text contains personally identifiable information (PII).

Inspect only the text itself.
No tools are available.
Never emit <TOOLCALL>, XML tags, markdown, explanations, or any extra text.
Return exactly one JSON object with one field only:
{{"contains_pii": true}}
or
{{"contains_pii": false}}

Text:
{content}"""
        return {
            "system": NEMOTRON_SYSTEM_PROMPT,
            "user": user,
        }
