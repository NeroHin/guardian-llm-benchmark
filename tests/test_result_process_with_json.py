from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from utils.result_process import BinaryMetrics, parse_pii_binary_output, process_model_run, save_processed_results


def test_parse_pii_binary_output_from_json_string() -> None:
    assert parse_pii_binary_output('{"contains_pii": true, "label": "是"}') is True
    assert parse_pii_binary_output('{"contains_pii": false, "label": "否"}') is False


def test_process_model_run_accepts_json_outputs() -> None:
    df = pd.DataFrame(
        {
            "naturalParagraph": ["樣本1", "樣本2", "樣本3"],
            "gt": ["是", "否", "是"],
        }
    )
    outputs = [
        '{"contains_pii": true, "label": "是", "confidence": 0.9, "reason": "有姓名"}',
        '{"contains_pii": false, "label": "否", "confidence": 0.8, "reason": "無個資"}',
        '{"contains_pii": null, "label": "無法解析", "confidence": null, "reason": "格式錯誤"}',
    ]

    result_df, metrics = process_model_run(
        df,
        outputs,
        "pii_binary",
        content_column="naturalParagraph",
        ground_truth_column="gt",
    )

    assert metrics is not None
    assert metrics.total == 3
    assert metrics.tp == 1
    assert metrics.tn == 1
    assert metrics.fn == 1
    assert metrics.unparseable == 1
    assert result_df["prediction"].tolist() == ["是", "否", "無法解析"]


def test_parse_pii_binary_output_from_nested_raw_text() -> None:
    wrapped = (
        '{"contains_pii": null, "label": "無法解析", "confidence": null, '
        '"reason": "", "raw_text": "{\\n  \\"contains_pii\\": true,\\n  \\"label\\": \\"是\\",\\n  \\"confidence\\": 0.98"}'
    )
    assert parse_pii_binary_output(wrapped) is True


def test_save_processed_results_supports_append_mode(tmp_path: Path) -> None:
    out_csv = tmp_path / "result.csv"
    df1 = pd.DataFrame({"a": [1], "b": ["x"]})
    df2 = pd.DataFrame({"a": [2], "b": ["y"]})

    metrics = BinaryMetrics(
        accuracy=1.0,
        precision=1.0,
        recall=1.0,
        f1=1.0,
        tp=1,
        tn=0,
        fp=0,
        fn=0,
        total=1,
    )

    save_processed_results(df1, metrics, str(out_csv), append=False)
    save_processed_results(df2, metrics, str(out_csv), append=True)

    merged = pd.read_csv(out_csv)
    assert len(merged) == 2
    assert merged["a"].tolist() == [1, 2]

    metrics_jsonl = out_csv.with_name("result_metrics.jsonl")
    lines = [line for line in metrics_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["accuracy"] == 1.0
