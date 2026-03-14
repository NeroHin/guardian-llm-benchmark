from __future__ import annotations

import asyncio
import sys
import types

import benchmarking.runtime as runtime


def _fake_completion(raw_text: str):
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=raw_text))],
        usage=types.SimpleNamespace(prompt_tokens=12, completion_tokens=3),
    )


def test_openai_compatible_runner_predicts_with_cost_zero(monkeypatch) -> None:
    sync_calls: list[dict[str, object]] = []
    async_calls: list[dict[str, object]] = []
    client_inits: list[tuple[str, str, str]] = []

    class FakeOpenAI:
        def __init__(self, *, base_url: str, api_key: str) -> None:
            client_inits.append(("sync", base_url, api_key))
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=self._create)
            )

        def _create(self, **kwargs):
            sync_calls.append(kwargs)
            return _fake_completion(
                '{"contains_pii": true, "label": "是", "confidence": 0.95, "reason": "fake"}'
            )

        def close(self) -> None:
            return None

    class FakeAsyncOpenAI:
        def __init__(self, *, base_url: str, api_key: str) -> None:
            client_inits.append(("async", base_url, api_key))
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=self._create)
            )

        async def _create(self, **kwargs):
            async_calls.append(kwargs)
            return _fake_completion(
                '{"contains_pii": false, "label": "否", "confidence": 0.9, "reason": "fake-async"}'
            )

        async def close(self) -> None:
            return None

    monkeypatch.setitem(
        sys.modules,
        "openai",
        types.SimpleNamespace(OpenAI=FakeOpenAI, AsyncOpenAI=FakeAsyncOpenAI),
    )

    spec = runtime.ModelSpec(
        key="local-vllm",
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        provider="openai_compatible",
        settings={
            "base_url": "http://127.0.0.1:8000/v1",
            "api_key": "direct-key",
            "concurrency": 3,
            "temperature": 0,
            "max_tokens": 123,
            "response_format": "json_object",
        },
    )
    runner = runtime.OpenAICompatibleRunner(spec)

    sync_result = runner.predict("我的身分證字號是A123456789")
    async_result = asyncio.run(runner.predict_async("這是安全內容"))
    asyncio.run(runner.aclose_async_client())
    runner.close()

    assert runner.concurrency == 3
    assert sync_result.cost_usd == 0.0
    assert sync_result.prompt_tokens == 12
    assert sync_result.completion_tokens == 3
    assert async_result.cost_usd == 0.0
    assert len(sync_calls) == 1
    assert len(async_calls) == 1
    assert sync_calls[0]["extra_body"] == {"response_format": {"type": "json_object"}}
    assert sync_calls[0]["max_tokens"] == 123
    assert client_inits == [
        ("sync", "http://127.0.0.1:8000/v1", "direct-key"),
        ("async", "http://127.0.0.1:8000/v1", "direct-key"),
    ]


def test_openai_compatible_runner_api_key_resolution(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_COMPAT_API_KEY", raising=False)
    monkeypatch.delenv("MY_OPENAI_KEY", raising=False)

    assert runtime.OpenAICompatibleRunner._resolve_api_key({}) == "EMPTY"

    monkeypatch.setenv("MY_OPENAI_KEY", "from-custom-env")
    assert (
        runtime.OpenAICompatibleRunner._resolve_api_key({"api_key_env": "MY_OPENAI_KEY"})
        == "from-custom-env"
    )

    monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "from-default-env")
    assert runtime.OpenAICompatibleRunner._resolve_api_key({}) == "from-default-env"
    assert (
        runtime.OpenAICompatibleRunner._resolve_api_key({"api_key": "from-settings"})
        == "from-settings"
    )
