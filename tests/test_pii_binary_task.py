from __future__ import annotations

from benchmarking.tasks import get_task_definition


def test_pii_binary_prompt_builder_requires_single_json_object() -> None:
    task = get_task_definition("pii_binary")

    prompts = task.prompt_builder.build("fake/model", "王小明身分證字號是A123456789")

    assert "只輸出一個可解析的 JSON 物件" in prompts["user"]
    assert "只有文字中出現「明確、直接、可回溯到個人的識別子」時，才回傳 true" in prompts["user"]
    assert "若不確定，但文字裡沒有明確識別子，回傳 false" in prompts["user"]
    assert '"contains_pii"' in prompts["user"]
    assert '"label"' not in prompts["user"]


def test_pii_binary_prompt_builder_uses_nemotron_specific_non_tool_prompt() -> None:
    task = get_task_definition("pii_binary")

    prompts = task.prompt_builder.build(
        "nvidia/Nemotron-H-4B-Instruct-128K",
        "測試內容：王小明的手機是0912345678",
    )

    assert "No tools are available." in prompts["system"]
    assert "Never emit <TOOLCALL>" in prompts["user"]
    assert "One explicit identifier anywhere in a long text is sufficient for true." in prompts["user"]
    assert "If uncertain and there is no explicit identifier, return false." in prompts["user"]
    assert '{"contains_pii": true}' in prompts["user"]
    assert '{"contains_pii": false}' in prompts["user"]
    assert "只輸出一個可解析的 JSON 物件" not in prompts["user"]


def test_pii_binary_strict_parser_accepts_minimal_json() -> None:
    task = get_task_definition("pii_binary")

    parsed = task.output_parser.parse('{"contains_pii": true}')

    assert parsed["parse_status"] == "parsed"
    assert parsed["contains_pii"] is True
    assert parsed["label"] == "是"
    assert parsed["confidence"] is None
    assert parsed["reason"] == ""


def test_pii_binary_strict_parser_accepts_full_json_backward_compatible() -> None:
    task = get_task_definition("pii_binary")

    parsed = task.output_parser.parse(
        '{"contains_pii": true, "label": "是", "confidence": 0.91, "reason": "包含身分證資訊"}'
    )

    assert parsed["parse_status"] == "parsed"
    assert parsed["contains_pii"] is True
    assert parsed["label"] == "是"
    assert parsed["confidence"] == 0.91
    assert parsed["reason"] == "包含身分證資訊"


def test_pii_binary_strict_parser_rejects_non_json_missing_required_and_invalid_optional_types() -> None:
    task = get_task_definition("pii_binary")

    invalid_text = task.output_parser.parse("這不是 JSON")
    missing_required = task.output_parser.parse('{"label": "是"}')
    invalid_optional = task.output_parser.parse('{"contains_pii": true, "confidence": "high"}')

    assert invalid_text["parse_status"] == "unparseable"
    assert "invalid_json" in invalid_text["error"]
    assert missing_required["parse_status"] == "unparseable"
    assert "missing fields" in missing_required["error"]
    assert invalid_optional["parse_status"] == "unparseable"
    assert "confidence must be number" in invalid_optional["error"]
