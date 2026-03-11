from __future__ import annotations

from benchmarking.tasks import get_task_definition


def test_pii_binary_prompt_builder_requires_single_json_object() -> None:
    task = get_task_definition("pii_binary")

    prompts = task.prompt_builder.build("fake/model", "王小明身分證字號是A123456789")

    assert "只輸出一個可解析的 JSON 物件" in prompts["user"]
    assert '"contains_pii"' in prompts["user"]


def test_pii_binary_strict_parser_accepts_valid_json() -> None:
    task = get_task_definition("pii_binary")

    parsed = task.output_parser.parse(
        '{"contains_pii": true, "label": "是", "confidence": 0.91, "reason": "包含身分證資訊"}'
    )

    assert parsed["parse_status"] == "parsed"
    assert parsed["contains_pii"] is True


def test_pii_binary_strict_parser_rejects_non_json_and_mismatched_label() -> None:
    task = get_task_definition("pii_binary")

    invalid_text = task.output_parser.parse("這不是 JSON")
    invalid_label = task.output_parser.parse(
        '{"contains_pii": true, "label": "否", "confidence": 0.91, "reason": "錯誤"}'
    )

    assert invalid_text["parse_status"] == "unparseable"
    assert "invalid_json" in invalid_text["error"]
    assert invalid_label["parse_status"] == "unparseable"
    assert "label does not match" in invalid_label["error"]
