from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_execution_module() -> ModuleType:
    root = Path(__file__).resolve().parents[1]
    execution_path = root / "model-experiment" / "execution.py"
    module_name = "execution_module_runner_factory_tests"
    spec = importlib.util.spec_from_file_location(module_name, execution_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"無法載入模組: {execution_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_create_runner_dispatches_by_provider(monkeypatch) -> None:
    execution = _load_execution_module()

    monkeypatch.setitem(execution.RUNNER_FACTORIES, "openrouter", lambda spec: f"or:{spec.key}")
    monkeypatch.setitem(execution.RUNNER_FACTORIES, "huggingface", lambda spec: f"hf:{spec.key}")
    monkeypatch.setitem(execution.RUNNER_FACTORIES, "qwen_stream", lambda spec: f"qs:{spec.key}")

    s1 = execution.ModelSpec(key="k1", model_id="m1", provider="openrouter")
    s2 = execution.ModelSpec(key="k2", model_id="m2", provider="huggingface")
    s3 = execution.ModelSpec(key="k3", model_id="m3", provider="qwen_stream")

    assert execution.create_runner(s1) == "or:k1"
    assert execution.create_runner(s2) == "hf:k2"
    assert execution.create_runner(s3) == "qs:k3"


def test_create_runner_unknown_provider_raises() -> None:
    execution = _load_execution_module()
    spec = execution.ModelSpec(key="k", model_id="m", provider="unknown")

    with pytest.raises(ValueError) as exc:
        execution.create_runner(spec)

    assert "未知 provider" in str(exc.value)


def test_create_huggingface_runner_uses_profile(monkeypatch) -> None:
    execution = _load_execution_module()

    class DummyGranite:
        def __init__(self, spec):
            self.spec = spec

    class DummyQwen:
        def __init__(self, spec):
            self.spec = spec

    monkeypatch.setattr(execution, "GraniteHuggingFaceRunner", DummyGranite)
    monkeypatch.setattr(execution, "QwenGuardStreamRunner", DummyQwen)

    granite_spec = execution.ModelSpec(
        key="granite",
        model_id="ibm-granite/granite-4.0-h-micro",
        provider="huggingface",
        profile="granite_guard_json",
    )
    qwen_spec = execution.ModelSpec(
        key="qwen",
        model_id="Qwen/Qwen3Guard-Stream-4B",
        provider="huggingface",
        profile="qwen_stream",
    )
    qwen_instruct_spec = execution.ModelSpec(
        key="qwen-instruct",
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        provider="huggingface",
        profile="qwen_instruct_json",
    )

    assert isinstance(execution._create_huggingface_runner(granite_spec), DummyGranite)
    assert isinstance(execution._create_huggingface_runner(qwen_spec), DummyQwen)
    assert isinstance(execution._create_huggingface_runner(qwen_instruct_spec), DummyGranite)


def test_resolve_hf_runtime_settings_honors_config() -> None:
    execution = _load_execution_module()

    runtime = execution._resolve_hf_runtime_settings(
        {
            "trust_remote_code": False,
            "torch_dtype": "float16",
            "device_map": "auto",
        },
        cuda_available=True,
    )
    assert runtime["trust_remote_code"] is False
    assert runtime["torch_dtype_name"] == "float16"
    assert runtime["device_map"] == "auto"

    runtime_cpu = execution._resolve_hf_runtime_settings({}, cuda_available=False)
    assert runtime_cpu["torch_dtype_name"] == "float32"
    assert runtime_cpu["device_map"] is None


def test_resolve_hf_runtime_settings_fallbacks_bf16_to_fp16_when_unsupported() -> None:
    execution = _load_execution_module()
    runtime = execution._resolve_hf_runtime_settings(
        {"torch_dtype": "bfloat16"},
        cuda_available=True,
        bf16_available=False,
    )
    assert runtime["torch_dtype_name"] == "float16"
