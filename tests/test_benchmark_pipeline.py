from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd


def _load_execution_module() -> ModuleType:
    root = Path(__file__).resolve().parents[1]
    execution_path = root / "model-experiment" / "execution.py"
    module_name = "execution_module_pipeline_tests"
    spec = importlib.util.spec_from_file_location(module_name, execution_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"無法載入模組: {execution_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_build_eval_dataset_creates_positive_and_negative_samples() -> None:
    execution = _load_execution_module()
    src = pd.DataFrame(
        {
            "naturalParagraph": [
                "王小明是一位35歲工程師，可透過test@example.com或手機0912345678聯絡，身份證A123456789。",
                "李小美，28歲，可透過mail@abc.com聯絡。",
            ]
        }
    )

    eval_df = execution.build_eval_dataset(src, sample_limit=2, include_negative=True)

    assert len(eval_df) == 4
    assert set(eval_df["sample_type"].unique().tolist()) == {"positive", "negative"}
    assert set(eval_df["ground_truth"].unique().tolist()) == {True, False}


def test_build_eval_dataset_prefers_external_non_pii_dataset() -> None:
    execution = _load_execution_module()
    src = pd.DataFrame({"naturalParagraph": ["含個資文本A", "含個資文本B"]})
    non_pii = pd.DataFrame({"naturalParagraph": ["安全文本1", "安全文本2"]})

    eval_df = execution.build_eval_dataset(
        src,
        sample_limit=2,
        include_negative=True,
        non_pii_df=non_pii,
    )

    negatives = eval_df[eval_df["ground_truth"] == False]  # noqa: E712
    assert len(negatives) == 2
    assert set(negatives["naturalParagraph"].tolist()) == {"安全文本1", "安全文本2"}


def test_compute_guardrail_metrics_returns_expected_values() -> None:
    execution = _load_execution_module()
    metrics = execution.compute_guardrail_metrics(
        predictions=[True, False, True, None],
        ground_truths=[True, True, False, False],
        latencies_ms=[10.0, 20.0, 30.0, 40.0],
        costs_usd=[0.01, 0.02, 0.03, 0.0],
    )

    assert metrics["tp"] == 1
    assert metrics["fn"] == 1
    assert metrics["fp"] == 1
    assert metrics["tn"] == 1
    assert metrics["tpr"] == 0.5
    assert metrics["fpr"] == 0.5
    assert metrics["unparseable"] == 1
    assert metrics["cost_usd_total"] == 0.06


def test_run_single_model_with_fake_runner(monkeypatch) -> None:
    execution = _load_execution_module()

    class FakeRunner:
        model_id = "fake/model"
        provider = "openrouter"

        def predict(self, content: str):
            contains = "匿名化文本" not in content
            payload = {
                "contains_pii": contains,
                "label": "是" if contains else "否",
                "confidence": 0.9,
                "reason": "fake",
                "raw_text": "",
            }
            return execution.InferenceResult(
                output_json=json.dumps(payload, ensure_ascii=False),
                contains_pii=contains,
                cost_usd=0.001,
                prompt_tokens=10,
                completion_tokens=2,
            )

        def close(self) -> None:
            return None

    monkeypatch.setattr(execution, "create_runner", lambda spec: FakeRunner())

    df = pd.DataFrame(
        {
            "naturalParagraph": ["PII 內容", "匿名化文本：安全內容"],
            "ground_truth": [True, False],
            "sample_type": ["positive", "negative"],
        }
    )
    spec = execution.ModelSpec(key="fake-openrouter", model_id="fake/model", provider="openrouter")

    result_df, meta = execution.run_single_model(df, spec)

    assert len(result_df) == 2
    assert meta["tp"] == 1
    assert meta["tn"] == 1
    assert meta["fp"] == 0
    assert meta["fn"] == 0
    assert meta["tpr"] == 1.0
    assert meta["fpr"] == 0.0


def test_run_single_model_openrouter_uses_async_predict(monkeypatch) -> None:
    execution = _load_execution_module()

    class AsyncOnlyRunner:
        model_id = "fake/async-model"
        provider = "openrouter"
        concurrency = 2

        def __init__(self) -> None:
            self.sync_called = False

        def predict(self, content: str):
            self.sync_called = True
            raise AssertionError("sync predict 不應在 openrouter async 路徑被呼叫")

        async def predict_async(self, content: str):
            contains = "匿名化文本" not in content
            payload = {
                "contains_pii": contains,
                "label": "是" if contains else "否",
                "confidence": 0.8,
                "reason": "async-fake",
                "raw_text": "",
            }
            return execution.InferenceResult(
                output_json=json.dumps(payload, ensure_ascii=False),
                contains_pii=contains,
                cost_usd=0.002,
                prompt_tokens=6,
                completion_tokens=1,
            )

        def close(self) -> None:
            return None

    runner = AsyncOnlyRunner()
    monkeypatch.setattr(execution, "create_runner", lambda spec: runner)

    df = pd.DataFrame(
        {
            "naturalParagraph": ["PII 內容", "匿名化文本：安全內容"],
            "ground_truth": [True, False],
            "sample_type": ["positive", "negative"],
        }
    )
    spec = execution.ModelSpec(key="fake-openrouter-async", model_id="fake/async-model", provider="openrouter")

    result_df, meta = execution.run_single_model(df, spec)

    assert runner.sync_called is False
    assert len(result_df) == 2
    assert meta["tp"] == 1
    assert meta["tn"] == 1
    assert meta["fp"] == 0
    assert meta["fn"] == 0
