from __future__ import annotations

from pathlib import Path

import pandas as pd

from benchmarking.datasets.loader import build_eval_dataframe, load_dataset_frame
from benchmarking.specs import (
    BenchmarkDatasetSelection,
    BenchmarkOutputsSpec,
    BenchmarkRuntimeSpec,
    BenchmarkSpec,
    DatasetGroundTruthSpec,
    DatasetSpec,
    ModelSelectionSpec,
)


def test_load_dataset_frame_supports_fixed_ground_truth(tmp_path: Path) -> None:
    source = tmp_path / "positive.csv"
    pd.DataFrame({"text": ["A", "B"]}).to_csv(source, index=False)
    spec = DatasetSpec(
        key="positive",
        task="pii_binary",
        format="csv",
        path=source,
        content_column="text",
        ground_truth=DatasetGroundTruthSpec(mode="fixed", value=True),
    )

    df = load_dataset_frame(spec, root=tmp_path)

    assert df["ground_truth"].tolist() == [True, True]
    assert df["sample_type"].tolist() == ["positive", "positive"]


def test_load_dataset_frame_supports_column_ground_truth(tmp_path: Path) -> None:
    source = tmp_path / "mixed.csv"
    pd.DataFrame({"text": ["A", "B"], "label": ["pii", "non_pii"]}).to_csv(source, index=False)
    spec = DatasetSpec(
        key="mixed",
        task="pii_binary",
        format="csv",
        path=source,
        content_column="text",
        ground_truth=DatasetGroundTruthSpec(
            mode="column",
            column="label",
            positive_values=("pii",),
            negative_values=("non_pii",),
        ),
    )

    df = load_dataset_frame(spec, root=tmp_path)

    assert df["ground_truth"].tolist() == [True, False]
    assert df["sample_type"].tolist() == ["positive", "negative"]


def test_build_eval_dataframe_merges_positive_negative_sources(tmp_path: Path) -> None:
    pos = tmp_path / "positive.csv"
    neg = tmp_path / "negative.csv"
    pd.DataFrame({"text": ["P1", "P2"]}).to_csv(pos, index=False)
    pd.DataFrame({"text": ["N1", "N2"]}).to_csv(neg, index=False)

    registry = {
        "pos": DatasetSpec(
            key="pos",
            task="pii_binary",
            format="csv",
            path=pos,
            content_column="text",
            ground_truth=DatasetGroundTruthSpec(mode="fixed", value=True),
        ),
        "neg": DatasetSpec(
            key="neg",
            task="pii_binary",
            format="csv",
            path=neg,
            content_column="text",
            ground_truth=DatasetGroundTruthSpec(mode="fixed", value=False),
        ),
    }
    benchmark = BenchmarkSpec(
        key="pii-baseline",
        task="pii_binary",
        dataset=BenchmarkDatasetSelection(positive=("pos",), negative=("neg",)),
        models=ModelSelectionSpec(),
        runtime=BenchmarkRuntimeSpec(sample_limit=1, shuffle=False, random_seed=42),
        outputs=BenchmarkOutputsSpec(dir=Path("results/pii-baseline")),
        source_path=tmp_path / "benchmark.yaml",
    )

    df = build_eval_dataframe(benchmark, registry, root=tmp_path)

    assert len(df) == 2
    assert set(df["sample_type"].tolist()) == {"positive", "negative"}


def test_build_eval_dataframe_merges_multiple_dataset_keys_per_split(tmp_path: Path) -> None:
    pos_a = tmp_path / "positive-a.csv"
    pos_b = tmp_path / "positive-b.csv"
    neg_a = tmp_path / "negative-a.csv"
    neg_b = tmp_path / "negative-b.csv"
    pd.DataFrame({"text": ["P1"]}).to_csv(pos_a, index=False)
    pd.DataFrame({"text": ["P2"]}).to_csv(pos_b, index=False)
    pd.DataFrame({"text": ["N1"]}).to_csv(neg_a, index=False)
    pd.DataFrame({"text": ["N2"]}).to_csv(neg_b, index=False)

    registry = {
        "pos-a": DatasetSpec(
            key="pos-a",
            task="pii_binary",
            format="csv",
            path=pos_a,
            content_column="text",
            ground_truth=DatasetGroundTruthSpec(mode="fixed", value=True),
        ),
        "pos-b": DatasetSpec(
            key="pos-b",
            task="pii_binary",
            format="csv",
            path=pos_b,
            content_column="text",
            ground_truth=DatasetGroundTruthSpec(mode="fixed", value=True),
        ),
        "neg-a": DatasetSpec(
            key="neg-a",
            task="pii_binary",
            format="csv",
            path=neg_a,
            content_column="text",
            ground_truth=DatasetGroundTruthSpec(mode="fixed", value=False),
        ),
        "neg-b": DatasetSpec(
            key="neg-b",
            task="pii_binary",
            format="csv",
            path=neg_b,
            content_column="text",
            ground_truth=DatasetGroundTruthSpec(mode="fixed", value=False),
        ),
    }
    benchmark = BenchmarkSpec(
        key="pii-long-context",
        task="pii_binary",
        dataset=BenchmarkDatasetSelection(
            positive=("pos-a", "pos-b"),
            negative=("neg-a", "neg-b"),
        ),
        models=ModelSelectionSpec(),
        runtime=BenchmarkRuntimeSpec(sample_limit=1, shuffle=False, random_seed=42),
        outputs=BenchmarkOutputsSpec(dir=Path("results/pii-long-context")),
        source_path=tmp_path / "benchmark.yaml",
    )

    df = build_eval_dataframe(benchmark, registry, root=tmp_path)

    assert len(df) == 4
    assert set(df["source_dataset_key"].tolist()) == {"pos-a", "pos-b", "neg-a", "neg-b"}
