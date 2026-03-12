from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from benchmarking.specs import BenchmarkSpec, DatasetSpec


def _normalize_label(value: Any) -> Any:
    if pd.isna(value):
        return None
    return value


def _resolve_dataset_path(path: Path, *, root: Path) -> Path:
    return path if path.is_absolute() else root / path


def load_dataset_frame(spec: DatasetSpec, *, root: Path) -> pd.DataFrame:
    dataset_path = _resolve_dataset_path(spec.path, root=root)
    if spec.format == "csv":
        df = pd.read_csv(dataset_path)
    elif spec.format == "jsonl":
        rows = []
        with open(dataset_path, "r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if text:
                    rows.append(json.loads(text))
        df = pd.DataFrame(rows)
    else:
        raise ValueError(f"尚未支援的 dataset format: {spec.format}")

    if spec.content_column not in df.columns:
        raise ValueError(f"dataset {spec.key} 找不到 content_column: {spec.content_column}")

    for filter_spec in spec.filters:
        if filter_spec.column not in df.columns:
            raise ValueError(f"dataset {spec.key} 找不到 filter 欄位: {filter_spec.column}")
        df = df[df[filter_spec.column] == filter_spec.equals].copy()

    records = pd.DataFrame(
        {
            "content": df[spec.content_column].map(lambda value: "" if pd.isna(value) else str(value).strip()),
            "source_dataset_key": spec.key,
        }
    )
    records = records[records["content"].astype(bool)].copy()

    gt = spec.ground_truth
    if gt.mode == "fixed":
        records["ground_truth"] = bool(gt.value)
    else:
        if gt.column not in df.columns:
            raise ValueError(f"dataset {spec.key} 找不到標籤欄位: {gt.column}")
        values = df.loc[records.index, gt.column]
        normalized: list[bool] = []
        positives = {_normalize_label(value) for value in gt.positive_values}
        negatives = {_normalize_label(value) for value in gt.negative_values}
        for value in values:
            current = _normalize_label(value)
            if current in positives:
                normalized.append(True)
            elif current in negatives:
                normalized.append(False)
            else:
                raise ValueError(f"dataset {spec.key} 標籤值無法映射: {value!r}")
        records["ground_truth"] = normalized

    records["sample_type"] = records["ground_truth"].map(lambda value: "positive" if value else "negative")
    return records.reset_index(drop=True)


def _sample_df(
    df: pd.DataFrame,
    *,
    limit: int | None,
    shuffle: bool,
    random_seed: int,
) -> pd.DataFrame:
    result = df.copy()
    if shuffle:
        result = result.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    if limit is not None:
        result = result.head(limit).reset_index(drop=True)
    return result


def _load_selected_datasets(
    dataset_keys: tuple[str, ...],
    dataset_registry: dict[str, DatasetSpec],
    *,
    root: Path,
    runtime_limit: int | None,
    runtime_shuffle: bool,
    runtime_seed: int,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for key in dataset_keys:
        spec = dataset_registry[key]
        frame = load_dataset_frame(spec, root=root)
        frame = _sample_df(
            frame,
            limit=runtime_limit or spec.default_limit,
            shuffle=runtime_shuffle or spec.shuffle,
            random_seed=runtime_seed if runtime_shuffle else spec.random_seed,
        )
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["content", "source_dataset_key", "ground_truth", "sample_type"])
    merged = pd.concat(frames, ignore_index=True)
    if runtime_shuffle:
        merged = merged.sample(frac=1.0, random_state=runtime_seed).reset_index(drop=True)
    return merged


def build_eval_dataframe(
    benchmark: BenchmarkSpec,
    dataset_registry: dict[str, DatasetSpec],
    *,
    root: Path,
) -> pd.DataFrame:
    runtime_limit = benchmark.runtime.sample_limit
    runtime_shuffle = benchmark.runtime.shuffle
    runtime_seed = benchmark.runtime.random_seed

    if benchmark.dataset.source:
        return _load_selected_datasets(
            benchmark.dataset.source,
            dataset_registry,
            root=root,
            runtime_limit=runtime_limit,
            runtime_shuffle=runtime_shuffle,
            runtime_seed=runtime_seed,
        )

    positive_df = _load_selected_datasets(
        benchmark.dataset.positive,
        dataset_registry,
        root=root,
        runtime_limit=runtime_limit,
        runtime_shuffle=runtime_shuffle,
        runtime_seed=runtime_seed,
    )
    negative_df = _load_selected_datasets(
        benchmark.dataset.negative,
        dataset_registry,
        root=root,
        runtime_limit=runtime_limit,
        runtime_shuffle=runtime_shuffle,
        runtime_seed=runtime_seed,
    )
    merged = pd.concat([positive_df, negative_df], ignore_index=True)
    if runtime_shuffle:
        merged = merged.sample(frac=1.0, random_state=runtime_seed).reset_index(drop=True)
    return merged
