from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import benchmarking.core as core
import benchmarking.runtime as runtime
from benchmarking.datasets.loader import load_dataset_frame
from benchmarking.specs import DatasetGroundTruthSpec, DatasetSpec
from benchmarking.specs import (
    BenchmarkDatasetSelection,
    BenchmarkOutputsSpec,
    BenchmarkRuntimeSpec,
    BenchmarkSpec,
    ModelSelectionSpec,
    ModelSpec,
)


def test_load_dataset_frame_creates_positive_labels(tmp_path) -> None:
    source = tmp_path / "positive.csv"
    pd.DataFrame({"content": ["含個資文本A", "含個資文本B"]}).to_csv(source, index=False)

    spec = DatasetSpec(
        key="positive",
        task="pii_binary",
        format="csv",
        path=source,
        content_column="content",
        ground_truth=DatasetGroundTruthSpec(mode="fixed", value=True),
    )
    eval_df = load_dataset_frame(spec, root=tmp_path)

    assert len(eval_df) == 2
    assert set(eval_df["sample_type"].unique().tolist()) == {"positive"}
    assert set(eval_df["ground_truth"].unique().tolist()) == {True}


def test_compute_guardrail_metrics_returns_expected_values() -> None:
    metrics = runtime.compute_guardrail_metrics(
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
            return runtime.InferenceResult(
                output_json=json.dumps(payload, ensure_ascii=False),
                contains_pii=contains,
                cost_usd=0.001,
                prompt_tokens=10,
                completion_tokens=2,
            )

        def close(self) -> None:
            return None

    vram_samples = iter([(100.0, 1), (120.0, 1), (110.0, 1), (130.0, 1)])

    monkeypatch.setattr(runtime, "create_runner", lambda spec: FakeRunner())
    monkeypatch.setattr(runtime, "_sample_total_vram_usage_mb", lambda torch_module=None: next(vram_samples))

    df = pd.DataFrame(
        {
            "content": ["PII 內容", "匿名化文本：安全內容"],
            "ground_truth": [True, False],
            "sample_type": ["positive", "negative"],
        }
    )
    spec = runtime.ModelSpec(key="fake-openrouter", model_id="fake/model", provider="openrouter")

    result_df, meta = runtime.run_single_model(df, spec, content_column="content")

    assert len(result_df) == 2
    assert meta["tp"] == 1
    assert meta["tn"] == 1
    assert meta["fp"] == 0
    assert meta["fn"] == 0
    assert meta["tpr"] == 1.0
    assert meta["fpr"] == 0.0
    assert meta["vram_used_mb_min"] == 100.0
    assert meta["vram_used_mb_max"] == 130.0
    assert meta["vram_used_mb_avg"] == 115.0
    assert meta["vram_sample_count"] == 4
    assert meta["vram_device_count"] == 1
    assert result_df["parse_status"].tolist() == ["parsed", "parsed"]


def test_run_single_model_openrouter_uses_async_predict(monkeypatch) -> None:
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
            return runtime.InferenceResult(
                output_json=json.dumps(payload, ensure_ascii=False),
                contains_pii=contains,
                cost_usd=0.002,
                prompt_tokens=6,
                completion_tokens=1,
            )

        def close(self) -> None:
            return None

    runner = AsyncOnlyRunner()
    monkeypatch.setattr(runtime, "create_runner", lambda spec: runner)

    df = pd.DataFrame(
        {
            "content": ["PII 內容", "匿名化文本：安全內容"],
            "ground_truth": [True, False],
            "sample_type": ["positive", "negative"],
        }
    )
    spec = runtime.ModelSpec(key="fake-openrouter-async", model_id="fake/async-model", provider="openrouter")

    result_df, meta = runtime.run_single_model(df, spec, content_column="content")

    assert runner.sync_called is False
    assert len(result_df) == 2
    assert meta["tp"] == 1
    assert meta["tn"] == 1
    assert meta["fp"] == 0
    assert meta["fn"] == 0


def test_run_single_model_openai_compatible_uses_async_predict(monkeypatch) -> None:
    class AsyncOnlyRunner:
        model_id = "fake/openai-compat-model"
        provider = "openai_compatible"
        concurrency = 2

        def __init__(self) -> None:
            self.sync_called = False

        def predict(self, content: str):
            self.sync_called = True
            raise AssertionError("sync predict 不應在 openai_compatible async 路徑被呼叫")

        async def predict_async(self, content: str):
            contains = "匿名化文本" not in content
            payload = {
                "contains_pii": contains,
                "label": "是" if contains else "否",
                "confidence": 0.8,
                "reason": "async-fake",
                "raw_text": "",
            }
            return runtime.InferenceResult(
                output_json=json.dumps(payload, ensure_ascii=False),
                contains_pii=contains,
                cost_usd=0.0,
                prompt_tokens=6,
                completion_tokens=1,
            )

        async def aclose_async_client(self) -> None:
            return None

        def close(self) -> None:
            return None

    runner = AsyncOnlyRunner()
    monkeypatch.setattr(runtime, "create_runner", lambda spec: runner)

    df = pd.DataFrame(
        {
            "content": ["PII 內容", "匿名化文本：安全內容"],
            "ground_truth": [True, False],
            "sample_type": ["positive", "negative"],
        }
    )
    spec = runtime.ModelSpec(
        key="fake-openai-compat-async",
        model_id="fake/openai-compat-model",
        provider="openai_compatible",
    )

    result_df, meta = runtime.run_single_model(df, spec, content_column="content")

    assert runner.sync_called is False
    assert len(result_df) == 2
    assert meta["tp"] == 1
    assert meta["tn"] == 1
    assert meta["fp"] == 0
    assert meta["fn"] == 0


def test_run_single_model_uses_predict_batch_when_available(monkeypatch) -> None:
    class BatchRunner:
        model_id = "fake/batch-model"
        provider = "huggingface"

        def __init__(self) -> None:
            self.predict_called = False
            self.batch_call_sizes: list[int] = []

        def predict(self, content: str):
            self.predict_called = True
            raise AssertionError("正常情況不應回退到逐筆 predict")

        def predict_batch(self, contents: list[str]):
            self.batch_call_sizes.append(len(contents))
            outputs = []
            for content in contents:
                contains = "匿名化文本" not in content
                payload = {
                    "contains_pii": contains,
                    "label": "是" if contains else "否",
                    "confidence": 0.77,
                    "reason": "batch-fake",
                    "raw_text": "",
                }
                outputs.append(
                    runtime.InferenceResult(
                        output_json=json.dumps(payload, ensure_ascii=False),
                        contains_pii=contains,
                        cost_usd=0.0,
                        prompt_tokens=4,
                        completion_tokens=1,
                    )
                )
            return outputs

        def close(self) -> None:
            return None

    runner = BatchRunner()
    monkeypatch.setattr(runtime, "create_runner", lambda spec: runner)

    df = pd.DataFrame(
        {
            "content": [
                "PII 內容 1",
                "PII 內容 2",
                "PII 內容 3",
                "PII 內容 4",
                "PII 內容 5",
                "PII 內容 6",
                "PII 內容 7",
                "PII 內容 8",
                "PII 內容 9",
                "",
            ],
            "ground_truth": [True, True, True, True, True, True, True, True, True, False],
            "sample_type": ["positive"] * 9 + ["negative"],
        }
    )
    spec = runtime.ModelSpec(key="fake-hf-batch", model_id="fake/batch-model", provider="huggingface")

    result_df, _ = runtime.run_single_model(df, spec, content_column="content")

    assert runner.predict_called is False
    assert runner.batch_call_sizes == [8, 1]
    assert len(result_df) == 10
    assert result_df["parse_status"].tolist().count("parsed") == 9


def test_run_benchmark_writes_benchmark_mode_to_leaderboard_and_manifest(monkeypatch, tmp_path: Path) -> None:
    benchmark = BenchmarkSpec(
        key="pii-qwen-compare",
        task="pii_binary",
        dataset=BenchmarkDatasetSelection(
            positive=("positive",),
            negative=("negative",),
        ),
        models=ModelSelectionSpec(include_keys=("qwen3guard06b-stream",)),
        runtime=BenchmarkRuntimeSpec(sample_limit=1, shuffle=False, random_seed=42),
        outputs=BenchmarkOutputsSpec(dir=Path("results/pii-qwen-compare")),
        source_path=tmp_path / "benchmark.yaml",
    )
    eval_df = pd.DataFrame(
        {
            "content": ["PII 內容", "安全內容"],
            "ground_truth": [True, False],
            "sample_type": ["positive", "negative"],
        }
    )
    selected_models = [
        ModelSpec(
            key="qwen3guard06b-stream-full-pass",
            model_id="Qwen/Qwen3Guard-Stream-0.6B",
            provider="qwen_stream",
            settings={"benchmark_mode": "full_pass"},
        ),
        ModelSpec(
            key="qwen3guard06b-stream-early-stop",
            model_id="Qwen/Qwen3Guard-Stream-0.6B",
            provider="qwen_stream",
            settings={"benchmark_mode": "early_stop"},
        ),
    ]

    monkeypatch.setattr(core, "ROOT", tmp_path)
    monkeypatch.setattr(core, "load_benchmark_spec", lambda path: benchmark)
    monkeypatch.setattr(core, "load_model_registry", lambda path: {})
    monkeypatch.setattr(core, "load_dataset_registry", lambda path: {})
    monkeypatch.setattr(core, "validate_benchmark_spec", lambda *args, **kwargs: None)
    monkeypatch.setattr(core, "build_eval_dataframe", lambda *args, **kwargs: eval_df)
    monkeypatch.setattr(core, "resolve_model_selection", lambda registry, selection: selected_models)
    monkeypatch.setattr(core, "_git_commit", lambda: "test-sha")

    def fake_run_single_model(df, spec, **kwargs):
        result_df = df.copy()
        result_df["raw_output"] = ['{"contains_pii": true}'] * len(df)
        meta = {
            "model_key": spec.key,
            "model_id": spec.model_id,
            "provider": spec.provider,
            "tp": 1,
            "tn": 1,
            "fp": 0,
            "fn": 0,
            "tpr": 1.0,
            "fpr": 0.0,
            "overhead_latency_ms_avg": 12.34,
            "cost_usd_total": 0.0,
            "vram_used_mb_min": 100.0,
            "vram_used_mb_max": 150.0,
            "vram_used_mb_avg": 120.0,
        }
        return result_df, meta

    monkeypatch.setattr(core, "run_single_model", fake_run_single_model)

    run_dir = core.run_benchmark(tmp_path / "benchmark.yaml", show_progress=False)

    leaderboard = pd.read_csv(run_dir / "leaderboard.csv")
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    full_pass_metrics = json.loads(
        (run_dir / "metrics" / "qwen3guard06b-stream-full-pass.json").read_text(encoding="utf-8")
    )

    assert leaderboard["benchmark_mode"].tolist() == ["full_pass", "early_stop"]
    assert leaderboard["vram_used_mb_avg"].tolist() == [120.0, 120.0]
    assert manifest["models"] == [
        "qwen3guard06b-stream-full-pass",
        "qwen3guard06b-stream-early-stop",
    ]
    assert manifest["model_runs"] == [
        {
            "model_key": "qwen3guard06b-stream-full-pass",
            "model_id": "Qwen/Qwen3Guard-Stream-0.6B",
            "provider": "qwen_stream",
            "benchmark_mode": "full_pass",
        },
        {
            "model_key": "qwen3guard06b-stream-early-stop",
            "model_id": "Qwen/Qwen3Guard-Stream-0.6B",
            "provider": "qwen_stream",
            "benchmark_mode": "early_stop",
        },
    ]
    assert full_pass_metrics["benchmark_mode"] == "full_pass"
    assert full_pass_metrics["vram_used_mb_min"] == 100.0
    assert full_pass_metrics["vram_used_mb_max"] == 150.0
    assert full_pass_metrics["vram_used_mb_avg"] == 120.0


def test_run_benchmark_keeps_userpass_qwen_in_default_mode(monkeypatch, tmp_path: Path) -> None:
    benchmark = BenchmarkSpec(
        key="pii-baseline",
        task="pii_binary",
        dataset=BenchmarkDatasetSelection(
            positive=("positive",),
            negative=("negative",),
        ),
        models=ModelSelectionSpec(include_keys=("qwen3guard06b-userpass",)),
        runtime=BenchmarkRuntimeSpec(sample_limit=1, shuffle=False, random_seed=42),
        outputs=BenchmarkOutputsSpec(dir=Path("results/pii-baseline")),
        source_path=tmp_path / "benchmark.yaml",
    )
    eval_df = pd.DataFrame(
        {
            "content": ["PII 內容", "安全內容"],
            "ground_truth": [True, False],
            "sample_type": ["positive", "negative"],
        }
    )
    selected_models = [
        ModelSpec(
            key="qwen3guard06b-userpass",
            model_id="Qwen/Qwen3Guard-Stream-0.6B",
            provider="qwen_stream",
            settings={"assistant_stream_simulation": False},
        ),
    ]

    monkeypatch.setattr(core, "ROOT", tmp_path)
    monkeypatch.setattr(core, "load_benchmark_spec", lambda path: benchmark)
    monkeypatch.setattr(core, "load_model_registry", lambda path: {})
    monkeypatch.setattr(core, "load_dataset_registry", lambda path: {})
    monkeypatch.setattr(core, "validate_benchmark_spec", lambda *args, **kwargs: None)
    monkeypatch.setattr(core, "build_eval_dataframe", lambda *args, **kwargs: eval_df)
    monkeypatch.setattr(core, "resolve_model_selection", lambda registry, selection: selected_models)
    monkeypatch.setattr(core, "_git_commit", lambda: "test-sha")

    def fake_run_single_model(df, spec, **kwargs):
        result_df = df.copy()
        result_df["raw_output"] = ['{"contains_pii": true}'] * len(df)
        meta = {
            "model_key": spec.key,
            "model_id": spec.model_id,
            "provider": spec.provider,
            "tp": 1,
            "tn": 1,
            "fp": 0,
            "fn": 0,
            "tpr": 1.0,
            "fpr": 0.0,
            "overhead_latency_ms_avg": 12.34,
            "cost_usd_total": 0.0,
        }
        return result_df, meta

    monkeypatch.setattr(core, "run_single_model", fake_run_single_model)

    run_dir = core.run_benchmark(tmp_path / "benchmark.yaml", show_progress=False)

    leaderboard = pd.read_csv(run_dir / "leaderboard.csv")
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))

    assert leaderboard["benchmark_mode"].tolist() == ["default"]
    assert manifest["models"] == ["qwen3guard06b-userpass"]
    assert manifest["model_runs"] == [
        {
            "model_key": "qwen3guard06b-userpass",
            "model_id": "Qwen/Qwen3Guard-Stream-0.6B",
            "provider": "qwen_stream",
            "benchmark_mode": "default",
        }
    ]


def test_run_benchmark_separates_content_and_assistantstream_outputs(monkeypatch, tmp_path: Path) -> None:
    eval_df = pd.DataFrame(
        {
            "content": ["PII 內容", "安全內容"],
            "ground_truth": [True, False],
            "sample_type": ["positive", "negative"],
        }
    )
    benchmarks = {
        "content": BenchmarkSpec(
            key="pii-baseline",
            task="pii_binary",
            dataset=BenchmarkDatasetSelection(positive=("positive",), negative=("negative",)),
            models=ModelSelectionSpec(include_keys=("qwen3guard06b-userpass",)),
            runtime=BenchmarkRuntimeSpec(sample_limit=1, shuffle=False, random_seed=42),
            outputs=BenchmarkOutputsSpec(dir=Path("results/pii-baseline")),
            source_path=tmp_path / "content.yaml",
        ),
        "stream": BenchmarkSpec(
            key="pii-assistantstream-baseline",
            task="pii_binary",
            dataset=BenchmarkDatasetSelection(positive=("positive",), negative=("negative",)),
            models=ModelSelectionSpec(include_keys=("qwen3guard06b-assistantstream",)),
            runtime=BenchmarkRuntimeSpec(sample_limit=1, shuffle=False, random_seed=42),
            outputs=BenchmarkOutputsSpec(dir=Path("results/pii-assistantstream-baseline")),
            source_path=tmp_path / "stream.yaml",
        ),
    }
    selected_models = {
        "content.yaml": [
            ModelSpec(
                key="qwen3guard06b-userpass",
                model_id="Qwen/Qwen3Guard-Stream-0.6B",
                provider="qwen_stream",
                settings={"assistant_stream_simulation": False},
            )
        ],
        "stream.yaml": [
            ModelSpec(
                key="qwen3guard06b-assistantstream-full-pass",
                model_id="Qwen/Qwen3Guard-Stream-0.6B",
                provider="qwen_stream",
                settings={"assistant_stream_simulation": True, "benchmark_mode": "full_pass"},
            ),
            ModelSpec(
                key="qwen3guard06b-assistantstream-early-stop",
                model_id="Qwen/Qwen3Guard-Stream-0.6B",
                provider="qwen_stream",
                settings={"assistant_stream_simulation": True, "benchmark_mode": "early_stop"},
            ),
        ],
    }

    monkeypatch.setattr(core, "ROOT", tmp_path)
    monkeypatch.setattr(
        core,
        "load_benchmark_spec",
        lambda path: benchmarks["content"] if path.name == "content.yaml" else benchmarks["stream"],
    )
    monkeypatch.setattr(core, "load_model_registry", lambda path: {})
    monkeypatch.setattr(core, "load_dataset_registry", lambda path: {})
    monkeypatch.setattr(core, "validate_benchmark_spec", lambda *args, **kwargs: None)
    monkeypatch.setattr(core, "build_eval_dataframe", lambda *args, **kwargs: eval_df)
    monkeypatch.setattr(
        core,
        "resolve_model_selection",
        lambda registry, selection: selected_models[
            "content.yaml"
            if selection.include_keys == ("qwen3guard06b-userpass",)
            else "stream.yaml"
        ],
    )
    monkeypatch.setattr(core, "_git_commit", lambda: "test-sha")

    def fake_run_single_model(df, spec, **kwargs):
        result_df = df.copy()
        result_df["raw_output"] = ['{"contains_pii": true}'] * len(df)
        meta = {
            "model_key": spec.key,
            "model_id": spec.model_id,
            "provider": spec.provider,
            "tp": 1,
            "tn": 1,
            "fp": 0,
            "fn": 0,
            "tpr": 1.0,
            "fpr": 0.0,
            "overhead_latency_ms_avg": 12.34,
            "cost_usd_total": 0.0,
        }
        return result_df, meta

    monkeypatch.setattr(core, "run_single_model", fake_run_single_model)

    content_run_dir = core.run_benchmark(tmp_path / "content.yaml", show_progress=False)
    stream_run_dir = core.run_benchmark(tmp_path / "stream.yaml", show_progress=False)

    assert content_run_dir.parent == tmp_path / "results/pii-baseline"
    assert stream_run_dir.parent == tmp_path / "results/pii-assistantstream-baseline"
    assert content_run_dir.parent != stream_run_dir.parent
    assert (content_run_dir / "leaderboard.csv").exists()
    assert (stream_run_dir / "leaderboard.csv").exists()
