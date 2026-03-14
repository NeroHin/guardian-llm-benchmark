from __future__ import annotations

import os

import pytest

import benchmarking.runtime as runtime


def test_create_runner_dispatches_by_provider(monkeypatch) -> None:
    monkeypatch.setitem(runtime.RUNNER_FACTORIES, "openrouter", lambda spec: f"or:{spec.key}")
    monkeypatch.setitem(runtime.RUNNER_FACTORIES, "openai_compatible", lambda spec: f"oc:{spec.key}")
    monkeypatch.setitem(runtime.RUNNER_FACTORIES, "vllm_offline", lambda spec: f"vo:{spec.key}")
    monkeypatch.setitem(runtime.RUNNER_FACTORIES, "huggingface", lambda spec: f"hf:{spec.key}")
    monkeypatch.setitem(runtime.RUNNER_FACTORIES, "qwen_stream", lambda spec: f"qs:{spec.key}")

    s1 = runtime.ModelSpec(key="k1", model_id="m1", provider="openrouter")
    s2 = runtime.ModelSpec(key="k2", model_id="m2", provider="openai_compatible")
    s3 = runtime.ModelSpec(key="k3", model_id="m3", provider="vllm_offline")
    s4 = runtime.ModelSpec(key="k4", model_id="m4", provider="huggingface")
    s5 = runtime.ModelSpec(key="k5", model_id="m5", provider="qwen_stream")

    assert runtime.create_runner(s1) == "or:k1"
    assert runtime.create_runner(s2) == "oc:k2"
    assert runtime.create_runner(s3) == "vo:k3"
    assert runtime.create_runner(s4) == "hf:k4"
    assert runtime.create_runner(s5) == "qs:k5"


def test_create_runner_unknown_provider_raises() -> None:
    spec = runtime.ModelSpec(key="k", model_id="m", provider="unknown")

    with pytest.raises(ValueError) as exc:
        runtime.create_runner(spec)

    assert "未知 provider" in str(exc.value)


def test_create_huggingface_runner_uses_profile(monkeypatch) -> None:
    class DummyGranite:
        def __init__(self, spec):
            self.spec = spec

    class DummyQwen:
        def __init__(self, spec):
            self.spec = spec

    monkeypatch.setattr(runtime, "GraniteHuggingFaceRunner", DummyGranite)
    monkeypatch.setattr(runtime, "QwenGuardStreamRunner", DummyQwen)

    granite_spec = runtime.ModelSpec(
        key="granite",
        model_id="ibm-granite/granite-4.0-h-micro",
        provider="huggingface",
        profile="granite_guard_json",
    )
    qwen_spec = runtime.ModelSpec(
        key="qwen",
        model_id="Qwen/Qwen3Guard-Stream-4B",
        provider="huggingface",
        profile="qwen_stream",
    )
    qwen_instruct_spec = runtime.ModelSpec(
        key="qwen-instruct",
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        provider="huggingface",
        profile="qwen_instruct_json",
    )

    assert isinstance(runtime._create_huggingface_runner(granite_spec), DummyGranite)
    assert isinstance(runtime._create_huggingface_runner(qwen_spec), DummyQwen)
    assert isinstance(runtime._create_huggingface_runner(qwen_instruct_spec), DummyGranite)


def test_resolve_hf_runtime_settings_honors_config() -> None:
    resolved = runtime._resolve_hf_runtime_settings(
        {
            "trust_remote_code": False,
            "torch_dtype": "float16",
            "device_map": "auto",
        },
        cuda_available=True,
    )
    assert resolved["trust_remote_code"] is False
    assert resolved["torch_dtype_name"] == "float16"
    assert resolved["device_map"] == "auto"

    runtime_cpu = runtime._resolve_hf_runtime_settings({}, cuda_available=False)
    assert runtime_cpu["torch_dtype_name"] == "float32"
    assert runtime_cpu["device_map"] is None


def test_resolve_hf_runtime_settings_fallbacks_bf16_to_fp16_when_unsupported() -> None:
    resolved = runtime._resolve_hf_runtime_settings(
        {"torch_dtype": "bfloat16"},
        cuda_available=True,
        bf16_available=False,
    )
    assert resolved["torch_dtype_name"] == "float16"


def test_configure_hf_xet_env_applies_expected_values(monkeypatch) -> None:
    for key in (
        "HF_XET_HIGH_PERFORMANCE",
        "HF_XET_NUM_CONCURRENT_RANGE_GETS",
        "HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY",
    ):
        monkeypatch.delenv(key, raising=False)

    applied = runtime._configure_hf_xet_env(
        {
            "hf_xet": {
                "enabled": True,
                "high_performance": True,
                "num_concurrent_range_gets": 48,
                "reconstruct_write_sequentially": False,
            }
        }
    )

    assert os.environ["HF_XET_HIGH_PERFORMANCE"] == "1"
    assert os.environ["HF_XET_NUM_CONCURRENT_RANGE_GETS"] == "48"
    assert os.environ["HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY"] == "0"
    assert applied["HF_XET_HIGH_PERFORMANCE"] == "1"
    assert applied["HF_XET_NUM_CONCURRENT_RANGE_GETS"] == "48"
    assert applied["HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY"] == "0"


def test_configure_hf_xet_env_respects_existing_env_unless_override(monkeypatch) -> None:
    monkeypatch.setenv("HF_XET_HIGH_PERFORMANCE", "0")
    monkeypatch.setenv("HF_XET_NUM_CONCURRENT_RANGE_GETS", "16")

    applied_without_override = runtime._configure_hf_xet_env(
        {
            "hf_xet": {
                "enabled": True,
                "high_performance": True,
                "num_concurrent_range_gets": 64,
                "override_env": False,
            }
        }
    )
    assert os.environ["HF_XET_HIGH_PERFORMANCE"] == "0"
    assert os.environ["HF_XET_NUM_CONCURRENT_RANGE_GETS"] == "16"
    assert applied_without_override == {}

    applied_with_override = runtime._configure_hf_xet_env(
        {
            "hf_xet": {
                "enabled": True,
                "high_performance": True,
                "num_concurrent_range_gets": 64,
                "override_env": True,
            }
        }
    )
    assert os.environ["HF_XET_HIGH_PERFORMANCE"] == "1"
    assert os.environ["HF_XET_NUM_CONCURRENT_RANGE_GETS"] == "64"
    assert applied_with_override["HF_XET_HIGH_PERFORMANCE"] == "1"
    assert applied_with_override["HF_XET_NUM_CONCURRENT_RANGE_GETS"] == "64"
