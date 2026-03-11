from __future__ import annotations


GUARDIAN_SYSTEM_PROMPT = """你是一個專業的內容安全守衛（Guardian）模型。
你的任務是分析使用者提供的文字內容，識別並分類其中的敏感資訊與風險。

請嚴格遵循以下原則：
1. 準確識別 PII（personally identifiable information，個人可識別資訊）
2. 僅根據輸入內容本身做分類
3. 請輸出可被程式嚴格解析的 JSON
4. 不要輸出 Markdown、程式碼區塊、額外解釋或前後文"""


class PIIBinaryPromptBuilder:
    task_id = "pii_binary"

    def build(self, model_id: str, content: str) -> dict[str, str]:
        del model_id
        user = f"""請判斷以下文字是否包含個人可識別資訊（PII）。

## 輸入文字
{content}

## 輸出要求
你必須只輸出一個可解析的 JSON 物件，不可輸出 Markdown、程式碼區塊或任何額外文字。
JSON 結構如下：
{{
  "contains_pii": <boolean>,
  "label": "是" 或 "否",
  "confidence": <0 到 1 的小數>,
  "reason": "<簡短理由>"
}}
規則：
1. contains_pii=true 時，label 必須是「是」；contains_pii=false 時，label 必須是「否」。
2. confidence 必須介於 0 和 1。
3. reason 請保持一句話且不超過 20 字。"""
        return {
            "system": GUARDIAN_SYSTEM_PROMPT,
            "user": user,
        }
