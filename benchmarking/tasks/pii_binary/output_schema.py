from __future__ import annotations

PII_BINARY_OUTPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "contains_pii": {"type": "boolean"},
        "label": {"type": "string", "enum": ["是", "否"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string"},
        "raw_text": {"type": "string"},
    },
    "required": ["contains_pii"],
    "additionalProperties": False,
}
