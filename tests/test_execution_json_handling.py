from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from types import ModuleType


def _load_execution_module() -> ModuleType:
    root = Path(__file__).resolve().parents[1]
    execution_path = root / "model-experiment" / "execution.py"

    # 測試環境若未安裝 ollama，提供最小替身避免 import 失敗
    if "ollama" not in sys.modules:
        sys.modules["ollama"] = types.SimpleNamespace(chat=None)

    module_name = "execution_module_for_tests"
    spec = importlib.util.spec_from_file_location(module_name, execution_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"無法載入模組: {execution_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_normalize_pii_binary_json_with_plain_json() -> None:
    execution = _load_execution_module()
    raw = '{"contains_pii": true, "label": "是", "confidence": 0.91, "reason": "包含身分證資訊"}'

    normalized = execution.normalize_pii_binary_json(raw)

    assert normalized["contains_pii"] is True
    assert normalized["label"] == "是"
    assert normalized["confidence"] == 0.91
    assert normalized["reason"] == "包含身分證資訊"


def test_normalize_pii_binary_json_with_fenced_json_and_label_only() -> None:
    execution = _load_execution_module()
    raw = """```json
{"label":"否","confidence":"0.2","reason":"未發現可識別資訊"}
```"""

    normalized = execution.normalize_pii_binary_json(raw)

    assert normalized["contains_pii"] is False
    assert normalized["label"] == "否"
    assert normalized["confidence"] == 0.2


def test_normalize_pii_binary_json_with_invalid_text() -> None:
    execution = _load_execution_module()
    normalized = execution.normalize_pii_binary_json("這不是 JSON")

    assert normalized["contains_pii"] is None
    assert normalized["label"] == "無法解析"
    assert normalized["confidence"] is None


def test_normalize_pii_binary_json_with_truncated_json() -> None:
    execution = _load_execution_module()
    raw = '{\n  "contains_pii": true,\n  "label": "是",\n  "confidence": 0.98,\n  "reason": "包含身份證號碼'

    normalized = execution.normalize_pii_binary_json(raw)

    assert normalized["contains_pii"] is True
    assert normalized["label"] == "是"
    assert normalized["confidence"] == 0.98
    assert "包含身份證號碼" in normalized["reason"]


def test_normalized_json_can_be_serialized() -> None:
    execution = _load_execution_module()
    normalized = execution.normalize_pii_binary_json('{"contains_pii": false, "confidence": 0.4}')
    serialized = json.dumps(normalized, ensure_ascii=False)

    assert '"contains_pii": false' in serialized


def test_parse_qwen_text_response_extracts_label_categories_refusal() -> None:
    execution = _load_execution_module()
    content = (
        "Safety: Unsafe\n"
        "Category: PII, Unethical Acts\n"
        "Refusal: No"
    )
    label, categories, refusal = execution.parse_qwen_text_response(content)

    assert label == "Unsafe"
    assert "PII" in categories
    assert "Unethical Acts" in categories
    assert refusal == "No"


def test_contains_pii_category_detection() -> None:
    execution = _load_execution_module()
    assert execution._contains_pii_category("PII, Unethical Acts") is True
    assert execution._contains_pii_category("Violent", ["PII", "Violent"]) is True
    assert execution._contains_pii_category("None") is False
