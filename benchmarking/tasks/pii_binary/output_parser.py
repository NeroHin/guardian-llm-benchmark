from __future__ import annotations

import json
from typing import Any


class PIIBinaryStrictJSONParser:
    def parse(self, raw_text: str) -> dict[str, Any]:
        base = {
            "raw_output": raw_text,
            "parse_status": "unparseable",
            "contains_pii": None,
            "label": None,
            "confidence": None,
            "reason": "",
            "error": "",
        }
        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            return {**base, "error": f"invalid_json: {exc.msg}"}

        if not isinstance(payload, dict):
            return {**base, "error": "schema_error: output must be an object"}

        required = ("contains_pii", "label", "confidence", "reason")
        missing = [key for key in required if key not in payload]
        if missing:
            return {**base, "error": f"schema_error: missing fields {missing}"}

        allowed_keys = set(required) | {"raw_text"}
        unknown = sorted(set(payload.keys()) - allowed_keys)
        if unknown:
            return {**base, "error": f"schema_error: unknown fields {unknown}"}

        contains_pii = payload.get("contains_pii")
        label = payload.get("label")
        confidence = payload.get("confidence")
        reason = payload.get("reason")

        if not isinstance(contains_pii, bool):
            return {**base, "error": "schema_error: contains_pii must be boolean"}
        if label not in {"是", "否"}:
            return {**base, "error": "schema_error: label must be '是' or '否'"}
        if (contains_pii and label != "是") or ((not contains_pii) and label != "否"):
            return {**base, "error": "schema_error: label does not match contains_pii"}
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
            return {**base, "error": "schema_error: confidence must be number"}
        if not 0 <= float(confidence) <= 1:
            return {**base, "error": "schema_error: confidence out of range"}
        if not isinstance(reason, str):
            return {**base, "error": "schema_error: reason must be string"}

        return {
            "raw_output": raw_text,
            "parse_status": "parsed",
            "contains_pii": contains_pii,
            "label": label,
            "confidence": float(confidence),
            "reason": reason.strip(),
            "error": "",
        }
