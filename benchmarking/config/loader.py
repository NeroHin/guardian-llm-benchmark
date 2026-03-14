from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from benchmarking.specs import (
    BenchmarkDatasetSelection,
    BenchmarkOutputsSpec,
    BenchmarkRuntimeSpec,
    BenchmarkSpec,
    DatasetFilterSpec,
    DatasetGroundTruthSpec,
    DatasetSpec,
    ModelSelectionSpec,
    ModelSpec,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELS_DIR = ROOT / "configs" / "models"
DEFAULT_DATASETS_DIR = ROOT / "configs" / "datasets"
DEFAULT_BENCHMARKS_DIR = ROOT / "configs" / "benchmarks"


def _normalize_dataset_refs(raw: Any, *, path: Path, field_name: str) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        value = raw.strip()
        if not value:
            raise ValueError(f"benchmark 設定格式錯誤: {path} dataset.{field_name} 不可為空字串")
        return (value,)
    if isinstance(raw, list):
        values: list[str] = []
        for idx, item in enumerate(raw):
            if not isinstance(item, str) or not item.strip():
                raise ValueError(
                    f"benchmark 設定格式錯誤: {path} dataset.{field_name}[{idx}] 必須為非空字串"
                )
            values.append(item.strip())
        if not values:
            raise ValueError(f"benchmark 設定格式錯誤: {path} dataset.{field_name} 不可為空清單")
        return tuple(values)
    raise ValueError(f"benchmark 設定格式錯誤: {path} dataset.{field_name} 必須為字串或字串清單")


def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"設定格式錯誤: {path} 必須為 object")
    if payload.get("version") != 1:
        raise ValueError(f"設定 version 錯誤: {path} 目前只支援 1")
    return payload


def _iter_yaml_files(path_or_dir: Path) -> list[Path]:
    if path_or_dir.is_dir():
        return sorted(
            [p for p in path_or_dir.glob("*.yaml") if p.is_file()]
            + [p for p in path_or_dir.glob("*.yml") if p.is_file()]
        )
    if path_or_dir.is_file():
        return [path_or_dir]
    raise FileNotFoundError(f"找不到設定路徑: {path_or_dir}")


def load_model_registry(path_or_dir: Path = DEFAULT_MODELS_DIR) -> dict[str, ModelSpec]:
    specs: dict[str, ModelSpec] = {}
    for path in _iter_yaml_files(path_or_dir):
        payload = _read_yaml(path)
        models = payload.get("models")
        if not isinstance(models, list):
            raise ValueError(f"模型設定格式錯誤: {path} 的 models 必須為 list")
        for idx, item in enumerate(models):
            if not isinstance(item, dict):
                raise ValueError(f"模型設定格式錯誤: {path} models[{idx}] 必須為 object")
            key = str(item.get("key") or "").strip()
            model_id = str(item.get("model_id") or "").strip()
            provider = str(item.get("provider") or "").strip()
            if not key or not model_id or not provider:
                raise ValueError(f"模型設定格式錯誤: {path} models[{idx}] 缺少 key/model_id/provider")
            if key in specs:
                raise ValueError(f"模型設定重複 key: {key}")
            params_b = item.get("params_b")
            if params_b is not None:
                params_b = float(params_b)
            specs[key] = ModelSpec(
                key=key,
                model_id=model_id,
                provider=provider,
                profile=item.get("profile"),
                settings=dict(item.get("settings") or {}),
                params_b=params_b,
            )
    return specs


def _load_ground_truth(raw: Any, *, path: Path, idx: int) -> DatasetGroundTruthSpec:
    if not isinstance(raw, dict):
        raise ValueError(f"dataset 設定格式錯誤: {path} datasets[{idx}].ground_truth 必須為 object")
    mode = str(raw.get("mode") or "").strip()
    if mode not in {"fixed", "column"}:
        raise ValueError(f"dataset 設定格式錯誤: {path} datasets[{idx}].ground_truth.mode 必須為 fixed 或 column")
    if mode == "fixed":
        value = raw.get("value")
        if not isinstance(value, bool):
            raise ValueError(f"dataset 設定格式錯誤: {path} datasets[{idx}].ground_truth.value 必須為 bool")
        return DatasetGroundTruthSpec(mode="fixed", value=value)
    column = str(raw.get("column") or "").strip()
    if not column:
        raise ValueError(f"dataset 設定格式錯誤: {path} datasets[{idx}].ground_truth.column 必填")
    positives = tuple(raw.get("positive_values") or ())
    negatives = tuple(raw.get("negative_values") or ())
    if not positives or not negatives:
        raise ValueError(
            f"dataset 設定格式錯誤: {path} datasets[{idx}].ground_truth 需要 positive_values 與 negative_values"
        )
    return DatasetGroundTruthSpec(
        mode="column",
        column=column,
        positive_values=positives,
        negative_values=negatives,
    )


def load_dataset_registry(path_or_dir: Path = DEFAULT_DATASETS_DIR) -> dict[str, DatasetSpec]:
    specs: dict[str, DatasetSpec] = {}
    for path in _iter_yaml_files(path_or_dir):
        payload = _read_yaml(path)
        datasets = payload.get("datasets")
        if not isinstance(datasets, list):
            raise ValueError(f"dataset 設定格式錯誤: {path} 的 datasets 必須為 list")
        for idx, item in enumerate(datasets):
            if not isinstance(item, dict):
                raise ValueError(f"dataset 設定格式錯誤: {path} datasets[{idx}] 必須為 object")
            key = str(item.get("key") or "").strip()
            task = str(item.get("task") or "").strip()
            fmt = str(item.get("format") or "").strip()
            content_column = str(item.get("content_column") or "").strip()
            raw_path = str(item.get("path") or "").strip()
            if not key or not task or not fmt or not content_column or not raw_path:
                raise ValueError(
                    f"dataset 設定格式錯誤: {path} datasets[{idx}] 缺少 key/task/format/path/content_column"
                )
            if fmt not in {"csv", "jsonl"}:
                raise ValueError(f"dataset 設定格式錯誤: {path} datasets[{idx}].format 只支援 csv/jsonl")
            if key in specs:
                raise ValueError(f"dataset 設定重複 key: {key}")
            filters_raw = item.get("filters") or []
            filters: list[DatasetFilterSpec] = []
            if not isinstance(filters_raw, list):
                raise ValueError(f"dataset 設定格式錯誤: {path} datasets[{idx}].filters 必須為 list")
            for filter_idx, filter_item in enumerate(filters_raw):
                if not isinstance(filter_item, dict):
                    raise ValueError(
                        f"dataset 設定格式錯誤: {path} datasets[{idx}].filters[{filter_idx}] 必須為 object"
                    )
                column = str(filter_item.get("column") or "").strip()
                if not column or "equals" not in filter_item:
                    raise ValueError(
                        f"dataset 設定格式錯誤: {path} datasets[{idx}].filters[{filter_idx}] 需要 column 與 equals"
                    )
                filters.append(DatasetFilterSpec(column=column, equals=filter_item.get("equals")))
            sample_raw = item.get("sample") or {}
            if sample_raw and not isinstance(sample_raw, dict):
                raise ValueError(f"dataset 設定格式錯誤: {path} datasets[{idx}].sample 必須為 object")
            specs[key] = DatasetSpec(
                key=key,
                task=task,
                format=fmt,
                path=Path(raw_path),
                content_column=content_column,
                ground_truth=_load_ground_truth(item.get("ground_truth"), path=path, idx=idx),
                filters=tuple(filters),
                default_limit=sample_raw.get("default_limit"),
                shuffle=bool(sample_raw.get("shuffle", False)),
                random_seed=int(sample_raw.get("random_seed", 42)),
            )
    return specs


def load_benchmark_spec(path: Path) -> BenchmarkSpec:
    payload = _read_yaml(path)
    benchmark = payload.get("benchmark")
    if not isinstance(benchmark, dict):
        raise ValueError(f"benchmark 設定格式錯誤: {path} 的 benchmark 必須為 object")

    key = str(benchmark.get("key") or "").strip()
    task = str(benchmark.get("task") or "").strip()
    if not key or not task:
        raise ValueError(f"benchmark 設定格式錯誤: {path} 缺少 key/task")

    dataset_raw = benchmark.get("dataset") or {}
    if not isinstance(dataset_raw, dict):
        raise ValueError(f"benchmark 設定格式錯誤: {path} dataset 必須為 object")
    dataset = BenchmarkDatasetSelection(
        positive=_normalize_dataset_refs(dataset_raw.get("positive"), path=path, field_name="positive"),
        negative=_normalize_dataset_refs(dataset_raw.get("negative"), path=path, field_name="negative"),
        source=_normalize_dataset_refs(dataset_raw.get("source"), path=path, field_name="source"),
    )

    models_raw = benchmark.get("models") or {}
    if not isinstance(models_raw, dict):
        raise ValueError(f"benchmark 設定格式錯誤: {path} models 必須為 object")
    include_keys = tuple(str(item) for item in (models_raw.get("include_keys") or ()))
    models = ModelSelectionSpec(include_keys=include_keys)

    runtime_raw = benchmark.get("runtime") or {}
    if runtime_raw and not isinstance(runtime_raw, dict):
        raise ValueError(f"benchmark 設定格式錯誤: {path} runtime 必須為 object")
    batch_size = int(runtime_raw.get("batch_size", 1))
    if batch_size <= 0:
        raise ValueError(f"benchmark 設定格式錯誤: {path} runtime.batch_size 必須 > 0")
    runtime = BenchmarkRuntimeSpec(
        sample_limit=runtime_raw.get("sample_limit"),
        batch_size=batch_size,
        shuffle=bool(runtime_raw.get("shuffle", True)),
        random_seed=int(runtime_raw.get("random_seed", 42)),
        fail_fast=bool(runtime_raw.get("fail_fast", False)),
    )

    outputs_raw = benchmark.get("outputs") or {}
    if not isinstance(outputs_raw, dict):
        raise ValueError(f"benchmark 設定格式錯誤: {path} outputs 必須為 object")
    output_dir = str(outputs_raw.get("dir") or "").strip()
    if not output_dir:
        raise ValueError(f"benchmark 設定格式錯誤: {path} outputs.dir 必填")
    outputs = BenchmarkOutputsSpec(
        dir=Path(output_dir),
        save_rows=bool(outputs_raw.get("save_rows", True)),
        save_metrics_json=bool(outputs_raw.get("save_metrics_json", True)),
    )
    return BenchmarkSpec(
        key=key,
        task=task,
        dataset=dataset,
        models=models,
        runtime=runtime,
        outputs=outputs,
        source_path=path,
    )


def resolve_model_selection(
    registry: dict[str, ModelSpec],
    selection: ModelSelectionSpec,
) -> list[ModelSpec]:
    def expand(spec: ModelSpec) -> list[ModelSpec]:
        if spec.provider != "qwen_stream":
            return [spec]

        base_settings = dict(spec.settings)
        use_assistant_stream = bool(base_settings.get("assistant_stream_simulation", True))
        if not use_assistant_stream:
            return [spec]

        return [
            ModelSpec(
                key=f"{spec.key}-full-pass",
                model_id=spec.model_id,
                provider=spec.provider,
                profile=spec.profile,
                settings={
                    **base_settings,
                    "assistant_stream_simulation": True,
                    "benchmark_mode": "full_pass",
                },
                params_b=spec.params_b,
            ),
            ModelSpec(
                key=f"{spec.key}-early-stop",
                model_id=spec.model_id,
                provider=spec.provider,
                profile=spec.profile,
                settings={
                    **base_settings,
                    "assistant_stream_simulation": True,
                    "benchmark_mode": "early_stop",
                },
                params_b=spec.params_b,
            ),
        ]

    selected: list[ModelSpec] = []

    for key in selection.include_keys:
        try:
            spec = registry[key]
        except KeyError as exc:
            raise ValueError(f"找不到 benchmark 指定模型: {key}") from exc
        selected.extend(expand(spec))

    if not selected:
        for spec in registry.values():
            selected.extend(expand(spec))

    if not selected:
        raise ValueError("沒有任何模型被選取")
    return selected


def validate_benchmark_spec(
    benchmark: BenchmarkSpec,
    *,
    model_registry: dict[str, ModelSpec],
    dataset_registry: dict[str, DatasetSpec],
) -> None:
    if benchmark.task != "pii_binary":
        raise ValueError(f"V1 目前只支援 pii_binary，收到 {benchmark.task}")
    dataset = benchmark.dataset
    has_split = bool(dataset.positive and dataset.negative)
    has_source = bool(dataset.source)
    if has_split == has_source:
        raise ValueError("benchmark.dataset 必須擇一使用 positive/negative 或 source")

    refs = [*dataset.positive, *dataset.negative, *dataset.source]
    for ref in refs:
        if ref not in dataset_registry:
            raise ValueError(f"找不到 benchmark 指定 dataset: {ref}")
        if dataset_registry[ref].task != benchmark.task:
            raise ValueError(f"dataset 任務不相符: {ref} -> {dataset_registry[ref].task}")

    resolve_model_selection(model_registry, benchmark.models)
