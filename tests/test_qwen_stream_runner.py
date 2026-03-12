from __future__ import annotations

import json

import torch

import benchmarking.runtime as runtime
from benchmarking.tasks.pii_binary.output_parser import PIIBinaryStrictJSONParser


class FakeTokenizer:
    def __init__(self) -> None:
        self._token_map = {
            "<|im_start|>": 1001,
            "user": 1002,
            "<|im_end|>": 1003,
            "</think>": 1004,
        }

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    ) -> str:
        if len(messages) == 1 and messages[0]["role"] == "user":
            return "USER_ONLY"
        if len(messages) == 2 and messages[0]["role"] == "user" and messages[1]["role"] == "assistant":
            return "USER_ASSISTANT"
        raise AssertionError(f"unexpected messages: {messages!r}")

    def __call__(self, text: str, return_tensors: str = "pt"):
        if return_tensors != "pt":
            raise AssertionError(f"unexpected return_tensors: {return_tensors}")
        if text == "USER_ONLY":
            input_ids = torch.tensor([[11, 12, 13, 14]], dtype=torch.long)
        elif text == "USER_ASSISTANT":
            input_ids = torch.tensor(
                [[1001, 1002, 21, 22, 1003, 41, 42, 1004, 43, 31, 32, 33, 34]],
                dtype=torch.long,
            )
        else:
            raise AssertionError(f"unexpected text: {text!r}")
        return type("FakeBatch", (), {"input_ids": input_ids})()

    def convert_tokens_to_ids(self, token: str) -> int | None:
        return self._token_map.get(token)

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        if skip_special_tokens:
            raise AssertionError("unexpected skip_special_tokens=True")
        token_id = ids[0]
        if token_id in {41, 42, 43}:
            return "\n"
        if token_id == 1004:
            return "</think>"
        return f"tok-{token_id}"


class FakeStreamModel:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []
        self.closed_states: list[object] = []

    def stream_moderate_from_ids(self, token_input, role: str, stream_state=None):
        self.calls.append((role, token_input))
        if role == "user":
            return (
                {
                    "risk_level": ["Safe"],
                    "risk_prob": [0.91],
                    "category": ["None"],
                    "category_prob": [0.88],
                    "content": [""],
                },
                {"assistant_step": 0},
            )
        if role != "assistant":
            raise AssertionError(f"unexpected role: {role}")

        next_step = int(stream_state["assistant_step"]) + 1
        next_state = {"assistant_step": next_step}
        if next_step >= 3:
            return (
                {
                    "risk_level": ["Unsafe"],
                    "risk_prob": [0.86],
                    "category": ["PII"],
                    "category_prob": [0.97],
                    "content": [""],
                },
                next_state,
            )
        return (
            {
                "risk_level": ["Safe"],
                "risk_prob": [0.9],
                "category": ["None"],
                "category_prob": [0.85],
                "content": [""],
            },
            next_state,
        )

    def close_stream(self, stream_state) -> None:
        self.closed_states.append(stream_state)


def _build_runner(execution, *, assistant_stream_simulation: bool):
    runner = execution.QwenGuardStreamRunner.__new__(execution.QwenGuardStreamRunner)
    runner.model_id = "Qwen/Qwen3Guard-Stream-0.6B"
    runner.provider = "qwen_stream"
    runner.profile = "qwen_stream"
    runner._settings = {
        "assistant_stream_simulation": assistant_stream_simulation,
        "assistant_stream_user_message": "Please continue.",
        "enable_thinking": False,
    }
    runner._tokenizer = FakeTokenizer()
    runner._model = FakeStreamModel()
    runner._torch = torch
    return runner


def test_qwen_stream_predict_uses_assistant_stream_simulation() -> None:
    runner = _build_runner(runtime, assistant_stream_simulation=True)

    inference = runner.predict("我的身分證字號是A123456789")
    parsed = PIIBinaryStrictJSONParser().parse(inference.output_json)
    payload = json.loads(inference.output_json)
    inner = json.loads(payload["raw_text"])

    assert inference.stream_mode == "assistant_token_stream_api"
    assert inference.early_stopped is True
    assert inference.contains_pii is True
    assert inference.fallback_used is False
    assert inference.processed_tokens == 3
    assert inference.total_tokens == 4
    assert inference.detection_latency_ms is not None
    assert parsed["parse_status"] == "parsed"
    assert parsed["contains_pii"] is True
    assert parsed["confidence"] == 0.97
    assert inner["category_prob"] == 0.97
    assert [role for role, _ in runner._model.calls] == ["user", "assistant", "assistant", "assistant"]
    assert runner._model.closed_states[-1] == {"assistant_step": 3}


def test_prepare_assistant_stream_tokens_strips_template_prefix() -> None:
    runner = _build_runner(runtime, assistant_stream_simulation=True)

    _, assistant_ids = runner._prepare_assistant_stream_tokens("測試內容")

    assert assistant_ids.tolist() == [31, 32, 33, 34]


def test_qwen_stream_predict_defaults_to_user_full_pass() -> None:
    runner = _build_runner(runtime, assistant_stream_simulation=False)

    inference = runner.predict("我的身分證字號是A123456789")

    assert inference.stream_mode == "user_full_pass_stream_api"
    assert inference.early_stopped is False
    assert inference.contains_pii is False
    assert inference.fallback_used is False
    assert inference.processed_tokens == 4
    assert inference.total_tokens == 4
    assert [role for role, _ in runner._model.calls] == ["user"]
    assert runner._model.closed_states[-1] == {"assistant_step": 0}


def test_qwen_stream_predict_defaults_to_assistant_stream_when_setting_missing() -> None:
    runner = _build_runner(runtime, assistant_stream_simulation=True)
    runner._settings.pop("assistant_stream_simulation")

    inference = runner.predict("我的身分證字號是A123456789")

    assert inference.stream_mode == "assistant_token_stream_api"
    assert inference.early_stopped is True
    assert [role for role, _ in runner._model.calls] == ["user", "assistant", "assistant", "assistant"]
