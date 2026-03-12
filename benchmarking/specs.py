from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


GroundTruthMode = Literal["fixed", "column"]
DatasetFormat = Literal["csv", "jsonl"]


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_id: str
    provider: str
    profile: str | None = None
    settings: dict[str, Any] = field(default_factory=dict)
    params_b: float | None = None


@dataclass(frozen=True)
class DatasetGroundTruthSpec:
    mode: GroundTruthMode
    value: bool | None = None
    column: str | None = None
    positive_values: tuple[Any, ...] = ()
    negative_values: tuple[Any, ...] = ()


@dataclass(frozen=True)
class DatasetFilterSpec:
    column: str
    equals: Any


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    task: str
    format: DatasetFormat
    path: Path
    content_column: str
    ground_truth: DatasetGroundTruthSpec
    filters: tuple[DatasetFilterSpec, ...] = ()
    default_limit: int | None = None
    shuffle: bool = False
    random_seed: int = 42


@dataclass(frozen=True)
class BenchmarkDatasetSelection:
    positive: tuple[str, ...] = ()
    negative: tuple[str, ...] = ()
    source: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelSelectionSpec:
    include_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class BenchmarkRuntimeSpec:
    sample_limit: int | None = None
    shuffle: bool = True
    random_seed: int = 42
    fail_fast: bool = False


@dataclass(frozen=True)
class BenchmarkOutputsSpec:
    dir: Path
    save_rows: bool = True
    save_metrics_json: bool = True


@dataclass(frozen=True)
class BenchmarkSpec:
    key: str
    task: str
    dataset: BenchmarkDatasetSelection
    models: ModelSelectionSpec
    runtime: BenchmarkRuntimeSpec
    outputs: BenchmarkOutputsSpec
    source_path: Path
