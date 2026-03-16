from __future__ import annotations

import sys
import types

import benchmarking.runtime as runtime


class _FakeTokenizer:
    def __init__(self) -> None:
        self.pad_token = None
        self.eos_token = "<eos>"

    @classmethod
    def from_pretrained(cls, model_id: str, trust_remote_code: bool = True):
        tokenizer = cls()
        tokenizer.model_id = model_id
        tokenizer.trust_remote_code = trust_remote_code
        return tokenizer

    def apply_chat_template(
        self,
        messages,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        enable_thinking: bool = False,
    ) -> str:
        parts = [f"{item['role']}:{item['content']}" for item in messages]
        if add_generation_prompt:
            parts.append("assistant:")
        return "\n".join(parts)

    def __call__(self, text: str):
        length = max(1, len(text.split()))
        return {"input_ids": [[idx for idx in range(length)]]}


class _FakeSamplingParams:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class _FakeStructuredOutputsParams:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class _FakeLLM:
    init_calls: list[dict[str, object]] = []
    generate_calls: list[tuple[list[str], dict[str, object]]] = []

    def __init__(self, **kwargs) -> None:
        self.init_calls.append(kwargs)

    def generate(self, prompts, sampling_params):
        self.generate_calls.append((list(prompts), dict(sampling_params.kwargs)))
        template_outputs = [
            types.SimpleNamespace(
                prompt_token_ids=[1, 2, 3, 4],
                outputs=[
                    types.SimpleNamespace(
                        text='{"contains_pii": true, "label": "是", "confidence": 0.91, "reason": "pii"}',
                        token_ids=[10, 11],
                    )
                ],
            ),
            types.SimpleNamespace(
                prompt_token_ids=[1, 2],
                outputs=[
                    types.SimpleNamespace(
                        text='{"contains_pii": false, "label": "否", "confidence": 0.88, "reason": "safe"}',
                        token_ids=[12],
                    )
                ],
            ),
        ]
        return template_outputs[: len(prompts)]


class _FakeTorchLowMem:
    class cuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def mem_get_info() -> tuple[int, int]:
            total = 100 * 1024**3
            free = 60 * 1024**3
            return free, total


def test_vllm_offline_runner_predict_batch(monkeypatch) -> None:
    _FakeLLM.init_calls.clear()
    _FakeLLM.generate_calls.clear()
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(AutoTokenizer=_FakeTokenizer),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm",
        types.SimpleNamespace(
            LLM=_FakeLLM,
            SamplingParams=_FakeSamplingParams,
            sampling_params=types.SimpleNamespace(
                StructuredOutputsParams=_FakeStructuredOutputsParams,
            ),
        ),
    )

    spec = runtime.ModelSpec(
        key="local-vllm-offline",
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        provider="vllm_offline",
        settings={
            "trust_remote_code": False,
            "tensor_parallel_size": 2,
            "gpu_memory_utilization": 0.85,
            "torch_dtype": "float16",
            "temperature": 0,
            "max_tokens": 64,
            "top_p": 0.95,
        },
    )

    runner = runtime.VLLMOfflineRunner(spec)
    results = runner.predict_batch(["第一筆", "第二筆"])
    runner.close()

    assert len(results) == 2
    assert results[0].contains_pii is True
    assert results[0].prompt_tokens == 4
    assert results[0].completion_tokens == 2
    assert results[0].cost_usd == 0.0
    assert results[1].contains_pii is False
    assert results[1].prompt_tokens == 2
    assert results[1].completion_tokens == 1

    assert _FakeLLM.init_calls == [
        {
            "model": "Qwen/Qwen2.5-0.5B-Instruct",
            "trust_remote_code": False,
            "dtype": "float16",
            "tensor_parallel_size": 2,
            "gpu_memory_utilization": 0.85,
        }
    ]
    prompts, sampling_kwargs = _FakeLLM.generate_calls[0]
    assert len(prompts) == 2
    assert "system:" in prompts[0]
    assert "user:" in prompts[0]
    assert sampling_kwargs["temperature"] == 0.0
    assert sampling_kwargs["max_tokens"] == 64
    assert sampling_kwargs["top_p"] == 0.95
    assert isinstance(sampling_kwargs["structured_outputs"], _FakeStructuredOutputsParams)
    assert sampling_kwargs["structured_outputs"].kwargs["json"]["required"] == ["contains_pii"]


def test_vllm_offline_runner_raises_with_clear_error_when_gpu_memory_insufficient(monkeypatch) -> None:
    _FakeLLM.init_calls.clear()
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(AutoTokenizer=_FakeTokenizer),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm",
        types.SimpleNamespace(
            LLM=_FakeLLM,
            SamplingParams=_FakeSamplingParams,
            sampling_params=types.SimpleNamespace(
                StructuredOutputsParams=_FakeStructuredOutputsParams,
            ),
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", _FakeTorchLowMem)

    spec = runtime.ModelSpec(
        key="local-vllm-offline-low-mem",
        model_id="nvidia/Nemotron-H-4B-Instruct-128K",
        provider="vllm_offline",
        settings={
            "gpu_memory_utilization": 0.75,
            "torch_dtype": "auto",
        },
    )

    try:
        runtime.VLLMOfflineRunner(spec)
        raise AssertionError("預期顯存不足時應拋出 RuntimeError")
    except RuntimeError as exc:
        message = str(exc)
        assert "GPU 可用記憶體不足" in message
        assert "gpu_memory_utilization=0.75" in message
        assert "降低 gpu_memory_utilization" in message

    assert _FakeLLM.init_calls == []


def test_vllm_offline_runner_predict_wraps_single_item(monkeypatch) -> None:
    _FakeLLM.init_calls.clear()
    _FakeLLM.generate_calls.clear()
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(AutoTokenizer=_FakeTokenizer),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm",
        types.SimpleNamespace(
            LLM=_FakeLLM,
            SamplingParams=_FakeSamplingParams,
            sampling_params=types.SimpleNamespace(
                StructuredOutputsParams=_FakeStructuredOutputsParams,
            ),
        ),
    )

    spec = runtime.ModelSpec(
        key="local-vllm-offline-single",
        model_id="ibm-granite/granite-4.0-h-micro",
        provider="vllm_offline",
        settings={"max_new_tokens": 32},
    )
    runner = runtime.VLLMOfflineRunner(spec)

    result = runner.predict("單筆內容")

    assert result.contains_pii is True
    assert result.prompt_tokens == 4
    assert result.completion_tokens == 2
    assert _FakeLLM.generate_calls[0][1]["max_tokens"] == 32
    assert isinstance(_FakeLLM.generate_calls[0][1]["structured_outputs"], _FakeStructuredOutputsParams)
