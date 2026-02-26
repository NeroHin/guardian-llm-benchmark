from __future__ import annotations

import pandas as pd

from utils.result_process import parse_pii_binary_output, process_model_run


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
