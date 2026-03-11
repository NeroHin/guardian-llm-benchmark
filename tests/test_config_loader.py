from __future__ import annotations

from pathlib import Path

import pytest

from benchmarking.config.loader import (
    load_benchmark_spec,
    load_dataset_registry,
    load_model_registry,
    resolve_model_selection,
    validate_benchmark_spec,
)


def test_load_model_registry_reads_params(tmp_path: Path) -> None:
    config = tmp_path / "models.yaml"
    config.write_text(
        """
version: 1
models:
  - key: model-a
    model_id: vendor/model-a
    provider: huggingface
    params_b: 3.5
    settings:
      max_new_tokens: 64
""".strip(),
        encoding="utf-8",
    )

    registry = load_model_registry(config)

    assert registry["model-a"].params_b == 3.5


def test_load_dataset_registry_supports_fixed_and_column_modes(tmp_path: Path) -> None:
    config = tmp_path / "datasets.yaml"
    config.write_text(
        """
version: 1
datasets:
  - key: fixed-positive
    task: pii_binary
    format: csv
    path: dataset/positive.csv
    content_column: text
    ground_truth:
      mode: fixed
      value: true
  - key: mixed
    task: pii_binary
    format: csv
    path: dataset/mixed.csv
    content_column: text
    ground_truth:
      mode: column
      column: label
      positive_values: ["pii", 1]
      negative_values: ["non_pii", 0]
""".strip(),
        encoding="utf-8",
    )

    registry = load_dataset_registry(config)

    assert registry["fixed-positive"].ground_truth.mode == "fixed"
    assert registry["mixed"].ground_truth.column == "label"


def test_validate_benchmark_spec_rejects_missing_dataset_reference(tmp_path: Path) -> None:
    benchmark_path = tmp_path / "benchmark.yaml"
    benchmark_path.write_text(
        """
version: 1
benchmark:
  key: pii-baseline
  task: pii_binary
  dataset:
    positive: missing-positive
    negative: missing-negative
  models:
    include_keys: [model-a]
  runtime:
    sample_limit: 10
  outputs:
    dir: results/pii-baseline
""".strip(),
        encoding="utf-8",
    )
    model_path = tmp_path / "models.yaml"
    model_path.write_text(
        """
version: 1
models:
  - key: model-a
    model_id: vendor/model-a
    provider: huggingface
""".strip(),
        encoding="utf-8",
    )
    dataset_path = tmp_path / "datasets.yaml"
    dataset_path.write_text(
        """
version: 1
datasets:
  - key: fixed-positive
    task: pii_binary
    format: csv
    path: dataset/positive.csv
    content_column: text
    ground_truth:
      mode: fixed
      value: true
""".strip(),
        encoding="utf-8",
    )

    benchmark = load_benchmark_spec(benchmark_path)
    models = load_model_registry(model_path)
    datasets = load_dataset_registry(dataset_path)

    with pytest.raises(ValueError) as exc:
        validate_benchmark_spec(benchmark, model_registry=models, dataset_registry=datasets)

    assert "找不到 benchmark 指定 dataset" in str(exc.value)


def test_resolve_model_selection_uses_keys_then_all_models_fallback(tmp_path: Path) -> None:
    config = tmp_path / "models.yaml"
    config.write_text(
        """
version: 1
models:
  - key: model-a
    model_id: vendor/model-a
    provider: huggingface
  - key: model-b
    model_id: vendor/model-b
    provider: openrouter
""".strip(),
        encoding="utf-8",
    )

    registry = load_model_registry(config)
    explicit = resolve_model_selection(
        registry,
        load_benchmark_spec(
            _write_benchmark(
                tmp_path,
                """
version: 1
benchmark:
  key: pii-baseline
  task: pii_binary
  dataset:
    source: pii-source
  models:
    include_keys: [model-a]
  outputs:
    dir: results/pii-baseline
""",
            )
        ).models,
    )
    fallback = resolve_model_selection(
        registry,
        load_benchmark_spec(
            _write_benchmark(
                tmp_path,
                """
version: 1
benchmark:
  key: pii-baseline
  task: pii_binary
  dataset:
    source: pii-source
  models: {}
  outputs:
    dir: results/pii-baseline
""",
            )
        ).models,
    )

    assert [item.key for item in explicit] == ["model-a"]
    assert [item.key for item in fallback] == ["model-a", "model-b"]


def _write_benchmark(tmp_path: Path, content: str) -> Path:
    path = tmp_path / f"benchmark-{abs(hash(content))}.yaml"
    path.write_text(content.strip(), encoding="utf-8")
    return path
