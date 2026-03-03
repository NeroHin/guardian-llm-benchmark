from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_execution_module() -> ModuleType:
    root = Path(__file__).resolve().parents[1]
    execution_path = root / "model-experiment" / "execution.py"
    module_name = "execution_module_model_config_tests"
    spec = importlib.util.spec_from_file_location(module_name, execution_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"無法載入模組: {execution_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_load_model_specs_from_yaml_enabled_only(tmp_path: Path) -> None:
    execution = _load_execution_module()
    yaml_path = tmp_path / "models.yaml"
    yaml_path.write_text(
        """
version: 1
models:
  - key: a-openrouter
    model_id: a/model
    provider: openrouter
    enabled: true
    settings:
      max_tokens: 64
  - key: b-disabled
    model_id: b/model
    provider: openrouter
    enabled: false
  - key: c-hf
    model_id: c/model
    provider: huggingface
    enabled: true
    profile: granite_guard_json
    settings:
      torch_dtype: float16
      device_map: auto
""".strip(),
        encoding="utf-8",
    )

    specs = execution.load_model_specs_from_yaml(yaml_path)

    assert [s.key for s in specs] == ["a-openrouter", "c-hf"]
    assert specs[0].settings["max_tokens"] == 64
    assert specs[1].profile == "granite_guard_json"


def test_build_model_specs_fallback_to_legacy_when_yaml_missing(tmp_path: Path) -> None:
    execution = _load_execution_module()
    missing_path = tmp_path / "missing.yaml"

    specs = execution.build_model_specs(models_config_path=missing_path)

    assert len(specs) == len(execution.LEGACY_DEFAULT_MODEL_SPECS)
    assert [(s.model_id, s.provider) for s in specs] == list(execution.LEGACY_DEFAULT_MODEL_SPECS)


def test_build_model_specs_supports_filter_by_key_and_model_id(tmp_path: Path) -> None:
    execution = _load_execution_module()
    yaml_path = tmp_path / "models.yaml"
    yaml_path.write_text(
        """
version: 1
models:
  - key: granite-or
    model_id: ibm-granite/granite-4.0-h-micro
    provider: openrouter
    enabled: true
  - key: qwen-stream
    model_id: Qwen/Qwen3Guard-Stream-4B
    provider: qwen_stream
    enabled: true
""".strip(),
        encoding="utf-8",
    )

    by_key = execution.build_model_specs(["granite-or"], models_config_path=yaml_path)
    by_model_id = execution.build_model_specs(
        ["Qwen/Qwen3Guard-Stream-4B"],
        models_config_path=yaml_path,
    )

    assert len(by_key) == 1
    assert by_key[0].provider == "openrouter"
    assert len(by_model_id) == 1
    assert by_model_id[0].key == "qwen-stream"


def test_build_model_specs_unknown_selector_raises_readable_error(tmp_path: Path) -> None:
    execution = _load_execution_module()
    yaml_path = tmp_path / "models.yaml"
    yaml_path.write_text(
        """
version: 1
models:
  - key: only-one
    model_id: only/model
    provider: openrouter
    enabled: true
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as exc:
        execution.build_model_specs(["not-found"], models_config_path=yaml_path)

    message = str(exc.value)
    assert "可用 key" in message
    assert "only-one" in message
