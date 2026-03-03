from __future__ import annotations

from utils.generate_non_pii_with_openai import _extract_json_obj, has_obvious_pii


def test_extract_json_obj_handles_plain_json() -> None:
    raw = '{"non_pii_text":"這是一段匿名化文本。"}'
    payload = _extract_json_obj(raw)
    assert payload is not None
    assert payload["non_pii_text"] == "這是一段匿名化文本。"


def test_extract_json_obj_handles_fenced_json() -> None:
    raw = '```json\n{"non_pii_text":"安全文本"}\n```'
    payload = _extract_json_obj(raw)
    assert payload is not None
    assert payload["non_pii_text"] == "安全文本"


def test_has_obvious_pii_detects_email() -> None:
    assert has_obvious_pii("請聯絡 test@example.com") is True


def test_has_obvious_pii_for_safe_text() -> None:
    assert has_obvious_pii("這是一段一般流程說明，不涉及任何個人資料。") is False

