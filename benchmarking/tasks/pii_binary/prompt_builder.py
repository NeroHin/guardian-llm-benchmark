from __future__ import annotations


GUARDIAN_SYSTEM_PROMPT = """你是一個專業的內容安全守衛（Guardian）模型。
你的任務是分析使用者提供的文字內容，判斷其中是否出現明確的個人可識別資訊（PII）。

請嚴格遵循以下原則：
1. 僅根據輸入文字本身做二元分類
2. 只有明確、直接、可回溯到個人的識別子才算 PII
3. 匿名化、泛化、占位符與模糊描述不算 PII
4. 請輸出可被程式嚴格解析的 JSON
5. 不要輸出 Markdown、程式碼區塊、額外解釋或前後文"""

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

判定標準：
- 只有文字中出現「明確、直接、可回溯到個人的識別子」時，才回傳 true。
- 只要長文本中的任何一個位置出現一個明確識別子，就必須回傳 true。
- 若只有泛化描述、匿名化資訊、占位符、背景敘事、疾病/職業/年齡/收入/信用分數/交易類型等語意，但沒有明確識別子，必須回傳 false。
- 若不確定，但文字裡沒有明確識別子，回傳 false。

以下情況應判定為 true：
- 真實或合成的姓名
- 電話號碼、電子郵件
- 精確地址
- 身分證號、護照號、病歷號、病人編號
- 銀行帳號、卡號、可識別個人的帳務編號
- 其他可直接識別、聯絡、定位或回溯到個人的明確字串

以下情況應判定為 false：
- 「某城市」、「某銀行」、「某帳戶」、「某醫師」、「一名年輕男性」這類泛化或占位符描述
- 已匿名化、已去識別化、已遮罩的資訊
- 只有疾病、職業、年齡區間、收入水平、信用評分、交易存在與否等敘述，但沒有明確識別子
- 與隱私或醫療或金融相關的主題文字，但沒有直接識別子

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
Return true only when the text contains at least one explicit, directly identifying personal identifier.
One explicit identifier anywhere in a long text is sufficient for true.
If the text only contains anonymized, generalized, masked, or placeholder information, return false.
If the text only mentions demographics, health conditions, occupations, income, credit scores, transactions, or generic locations without a direct identifier, return false.
If uncertain and there is no explicit identifier, return false.

Count as true:
- a real or synthetic personal name
- a phone number or email address
- an exact street address
- a government ID, passport number, patient ID, or medical record number
- a bank account number, card number, or another exact personal account identifier

Count as false:
- placeholders such as 某城市, 某銀行, 某帳戶, 某醫師
- anonymized or de-identified descriptions
- generic narrative text about health, finance, work, or privacy without an explicit identifier

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
