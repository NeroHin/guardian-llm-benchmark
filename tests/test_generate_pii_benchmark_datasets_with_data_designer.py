from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from utils.generate_pii_benchmark_datasets_with_data_designer import (
    build_long_context_variants,
    build_output_paths,
    build_needle_variants,
    contains_direct_identifier,
    extract_seed_identifiers,
    has_heavy_placeholders,
    insert_identifier_at_position,
    main,
    validate_clean_negative_text,
    validate_hard_negative_text,
)


def _sample_seed_row() -> dict[str, object]:
    return {
        "name": "王小明",
        "doctor": "李醫師",
        "location": "台北市信義區松仁路100號",
        "phoneNumbers": "0912-345-678",
        "emailAddress": "ming@example.com",
        "idCardNumbers": "A123456789",
        "occupation": "工程師",
        "symptoms": "疲倦",
        "transactionDetails": "一般帳務異動",
    }


def test_extract_seed_identifiers_cycles_identifier_types() -> None:
    row = _sample_seed_row()
    first = extract_seed_identifiers(row, 1)
    second = extract_seed_identifiers(row, 2)
    third = extract_seed_identifiers(row, 3)

    assert first.chosen_identifier_type == "name"
    assert second.chosen_identifier_type == "phone"
    assert third.chosen_identifier_type == "email"


def test_validate_clean_and_hard_negative_behaviour() -> None:
    seed = extract_seed_identifiers(_sample_seed_row(), 1)

    clean_ok = validate_clean_negative_text("這位個案近期因工作壓力而安排一般健康檢查，後續僅需持續休息。", seed)
    clean_bad = validate_clean_negative_text("某城市某銀行某帳戶的案例，王小明後續待追蹤。", seed)
    hard_ok = validate_hard_negative_text("個案近期因收入波動與就醫安排承受壓力，醫療與財務決策仍需追蹤。", seed)

    assert clean_ok.status == "passed"
    assert clean_bad.status == "failed"
    assert hard_ok.status == "passed"
    assert has_heavy_placeholders("某城市某銀行某帳戶某患者") is True
    assert contains_direct_identifier("聯絡電話是0912-345-678", seed) is True


def test_insert_identifier_positions_are_ordered() -> None:
    base = "。".join([f"第{i}句測試文字" for i in range(1, 21)]) + "。"
    snippet = "文件中明確寫出姓名為王小明。"

    _, start_pos = insert_identifier_at_position(base, snippet, "start")
    _, mid_pos = insert_identifier_at_position(base, snippet, "mid")
    _, end_pos = insert_identifier_at_position(base, snippet, "end")

    assert start_pos < mid_pos < end_pos


def test_long_and_needle_variants_match_expected_columns() -> None:
    seed = extract_seed_identifiers(_sample_seed_row(), 2)
    base = "這是一段沒有直接識別資訊的案例敘述。" * 30

    long_row = build_long_context_variants(base, seed)
    needle_row = build_needle_variants(base, seed)

    assert long_row["validation_status"] == "passed"
    assert long_row["1K_context_with_pii"].count(seed.chosen_identifier_snippet) == 1
    assert long_row["32k_context_with_pii"].count(seed.chosen_identifier_snippet) == 1
    assert needle_row["validation_status"] == "passed"
    assert needle_row["16k_needle_start_context"].count(seed.chosen_identifier_snippet) == 1
    assert needle_row["16k_needle_mid_context"].count(seed.chosen_identifier_snippet) == 1
    assert needle_row["16k_needle_end_context"].count(seed.chosen_identifier_snippet) == 1


def test_build_output_paths_uses_dataset_prefix(tmp_path: Path) -> None:
    paths = build_output_paths(tmp_path, "foo")

    assert paths.clean_negative.name == "data_person_1000_non_pii_clean_foo.csv"
    assert paths.needle.name == "harmless_long_context_needle_foo.csv"


class _FakePayload:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df


class _FakeBuilder:
    def __init__(self, model_configs: list[object]) -> None:
        self.model_configs = model_configs
        self.seed_dataset = None
        self.columns: list[object] = []

    def with_seed_dataset(self, seed_dataset: object) -> None:
        self.seed_dataset = seed_dataset

    def add_column(self, column: object) -> None:
        self.columns.append(column)


class _FakeDDModule:
    class ModelConfig:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    class ChatCompletionInferenceParams:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    class LocalFileSeedSource:
        def __init__(self, path: str) -> None:
            self.path = path

    class DataDesignerConfigBuilder(_FakeBuilder):
        pass

    class LLMTextColumnConfig:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs


class _FakeDataDesigner:
    def preview(self, builder: object, num_records: int) -> _FakePayload:
        return _FakePayload(pd.DataFrame([{"naturalParagraph": "preview text", "base_filler_case_report": "preview filler"}]))

    def create(self, builder: object, num_records: int, dataset_name: str) -> _FakePayload:
        return _FakePayload(pd.DataFrame([{"naturalParagraph": "產生的去識別化文本", "base_filler_case_report": "沒有明確識別資訊的 filler 文字。" * 5} for _ in range(num_records)]))


def test_main_preview_only_smoke(monkeypatch, tmp_path: Path) -> None:
    seed_path = tmp_path / "seed.json"
    seed_path.write_text(json.dumps([_sample_seed_row()], ensure_ascii=False), encoding="utf-8")
    positive_path = tmp_path / "positive.csv"
    pd.DataFrame([{"naturalParagraph": "王小明的電話是0912-345-678。"}]).to_csv(positive_path, index=False)

    import utils.generate_pii_benchmark_datasets_with_data_designer as module

    monkeypatch.setattr(module, "DEFAULT_POSITIVE_PATH", positive_path)
    monkeypatch.setattr(module, "_import_data_designer", lambda: (_FakeDDModule, _FakeDataDesigner))

    rc = main([
        "--seed-path",
        str(seed_path),
        "--output-dir",
        str(tmp_path),
        "--baseline-count",
        "2",
        "--long-count",
        "1",
        "--preview-only",
    ])

    assert rc == 0
    assert not (tmp_path / "data_person_1000_non_pii_clean_dd_v2.csv").exists()
