from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from benchmarking.env import ensure_dotenv_loaded


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SEED_PATH = ROOT / "dataset" / "data_person_1000_zh.json"
DEFAULT_OUTPUT_DIR = ROOT / "dataset"
DEFAULT_POSITIVE_PATH = ROOT / "dataset" / "data_person_1000_target.csv"
DEFAULT_PROVIDER = "nvidia"
DEFAULT_MODEL_ID = "nvidia/nemotron-3-nano-30b-a3b"
DEFAULT_MODEL_ALIAS = "nemotron-nano-v3"
DEFAULT_BASELINE_COUNT = 1000
DEFAULT_LONG_COUNT = 200
DEFAULT_PREVIEW_COUNT = 2
DEFAULT_DATASET_PREFIX = "dd_v2"

TARGET_LENGTHS = {
    "1K_context": 600,
    "4k_context": 2000,
    "8k_context": 4000,
    "16k_context": 8000,
    "32k_context": 18000,
}

EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_PATTERN = re.compile(r"(?:\+?886[-\s]?)?09\d{2}[-\s]?\d{3}[-\s]?\d{3}|(?<!\d)1[3-9]\d{9}(?!\d)")
GOV_ID_PATTERN = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)|\b[A-Z][12]\d{8}\b")
PLACEHOLDER_PATTERN = re.compile(r"某(?:城市|銀行|帳戶|醫院|公司|患者|先生|女士|地址|醫師)")
SENTENCE_BOUNDARY_PATTERN = re.compile(r"[。！？!?]\s*|\n+")


@dataclass(frozen=True)
class SeedIdentifiers:
    source_seed_id: int
    name: str
    doctor: str
    location: str
    phone: str
    email: str
    gov_id: str
    chosen_identifier_type: str
    chosen_identifier_value: str
    chosen_identifier_snippet: str
    topic_seed: str


@dataclass(frozen=True)
class OutputPaths:
    clean_negative: Path
    hard_negative: Path
    long_context: Path
    needle: Path


@dataclass(frozen=True)
class ValidationResult:
    status: str
    error: str


@dataclass(frozen=True)
class GenerationJob:
    label: str
    column_name: str
    prompt: str
    count: int
    dataset_name: str


def _resolve_repo_path(path_str: str | Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return ROOT / path


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def _normalize_space(text: str) -> str:
    collapsed = re.sub(r"\s+", " ", text or "").strip()
    return collapsed


def _topic_seed(row: Mapping[str, Any]) -> str:
    occupation = _safe_text(row.get("occupation"))
    symptoms = _safe_text(row.get("symptoms"))
    transaction = _safe_text(row.get("transactionDetails"))
    parts = [part for part in (occupation, symptoms, transaction) if part]
    if not parts:
        return "一般生活紀錄"
    return " / ".join(parts[:3])


def _available_identifier_options(row: Mapping[str, Any]) -> list[tuple[str, str, str]]:
    name = _safe_text(row.get("name"))
    phone = _safe_text(row.get("phoneNumbers"))
    email = _safe_text(row.get("emailAddress"))
    address = _safe_text(row.get("location"))
    gov_id = _safe_text(row.get("idCardNumbers"))

    options: list[tuple[str, str, str]] = []
    if name:
        options.append(("name", name, f"文件中明確寫出姓名為{name}。"))
    if phone:
        options.append(("phone", phone, f"聯絡電話明確記為{phone}。"))
    if email:
        options.append(("email", email, f"電子郵件地址明確記為{email}。"))
    if address:
        options.append(("address", address, f"通訊地址明確記為{address}。"))
    if gov_id:
        options.append(("gov_id", gov_id, f"政府核發證件號碼明確記為{gov_id}。"))
    return options


def extract_seed_identifiers(row: Mapping[str, Any], source_seed_id: int) -> SeedIdentifiers:
    options = _available_identifier_options(row)
    if not options:
        raise ValueError(f"seed row {source_seed_id} 找不到可用的 direct identifier")
    chosen_type, chosen_value, chosen_snippet = options[(source_seed_id - 1) % len(options)]
    return SeedIdentifiers(
        source_seed_id=source_seed_id,
        name=_safe_text(row.get("name")),
        doctor=_safe_text(row.get("doctor")),
        location=_safe_text(row.get("location")),
        phone=_safe_text(row.get("phoneNumbers")),
        email=_safe_text(row.get("emailAddress")),
        gov_id=_safe_text(row.get("idCardNumbers")),
        chosen_identifier_type=chosen_type,
        chosen_identifier_value=chosen_value,
        chosen_identifier_snippet=chosen_snippet,
        topic_seed=_topic_seed(row),
    )


def contains_direct_identifier(text: str, seed: SeedIdentifiers) -> bool:
    normalized = text or ""
    if EMAIL_PATTERN.search(normalized) or PHONE_PATTERN.search(normalized) or GOV_ID_PATTERN.search(normalized):
        return True
    for value in (seed.name, seed.doctor, seed.location, seed.phone, seed.email, seed.gov_id):
        if value and value in normalized:
            return True
    return False


def has_heavy_placeholders(text: str) -> bool:
    if not text:
        return False
    placeholder_hits = PLACEHOLDER_PATTERN.findall(text)
    return len(placeholder_hits) >= 2 or text.count("某") >= 4


def validate_clean_negative_text(text: str, seed: SeedIdentifiers) -> ValidationResult:
    candidate = _normalize_space(text)
    if not candidate:
        return ValidationResult("failed", "空白輸出")
    if contains_direct_identifier(candidate, seed):
        return ValidationResult("failed", "clean negative 含 direct identifier")
    if has_heavy_placeholders(candidate):
        return ValidationResult("failed", "clean negative 含過多 placeholder 語句")
    return ValidationResult("passed", "")


def validate_hard_negative_text(text: str, seed: SeedIdentifiers) -> ValidationResult:
    candidate = _normalize_space(text)
    if not candidate:
        return ValidationResult("failed", "空白輸出")
    if contains_direct_identifier(candidate, seed):
        return ValidationResult("failed", "hard negative 含 direct identifier")
    return ValidationResult("passed", "")


def validate_positive_text(text: str, identifier_snippet: str) -> ValidationResult:
    if not text:
        return ValidationResult("failed", "空白輸出")
    count = text.count(identifier_snippet)
    if count != 1:
        return ValidationResult("failed", f"expected 1 inserted identifier snippet, got {count}")
    return ValidationResult("passed", "")


def _trim_to_sentence_boundary(text: str, target_chars: int) -> str:
    if len(text) <= target_chars:
        return text.strip()
    window_end = min(len(text), target_chars + 120)
    boundary_positions = [match.end() for match in SENTENCE_BOUNDARY_PATTERN.finditer(text[:window_end])]
    if boundary_positions:
        usable = [pos for pos in boundary_positions if pos <= target_chars + 40]
        if usable:
            return text[: usable[-1]].strip()
    return text[:target_chars].strip()


def expand_text_to_target_length(base_text: str, target_chars: int) -> str:
    normalized = _normalize_space(base_text)
    if not normalized:
        raise ValueError("base_text 不可為空")
    chunks = [normalized]
    idx = 1
    while len("\n\n".join(chunks)) < target_chars + 120:
        chunks.append(f"補充說明{idx}：{normalized}")
        idx += 1
    return _trim_to_sentence_boundary("\n\n".join(chunks), target_chars)


def _sentence_boundary_offsets(text: str) -> list[int]:
    offsets = [0]
    offsets.extend(match.end() for match in SENTENCE_BOUNDARY_PATTERN.finditer(text))
    if offsets[-1] != len(text):
        offsets.append(len(text))
    deduped = sorted(set(offsets))
    return deduped


def insert_identifier_at_position(text: str, identifier_snippet: str, position: str) -> tuple[str, int]:
    if position not in {"start", "mid", "end"}:
        raise ValueError(f"未知插入位置: {position}")
    boundaries = _sentence_boundary_offsets(text)
    desired_ratio = {"start": 0.12, "mid": 0.5, "end": 0.88}[position]
    desired = int(len(text) * desired_ratio)
    insert_at = min(boundaries, key=lambda value: abs(value - desired))
    snippet = f"\n\n{identifier_snippet}\n\n"
    return f"{text[:insert_at]}{snippet}{text[insert_at:]}".strip(), insert_at


def build_long_context_variants(base_text: str, seed: SeedIdentifiers) -> dict[str, Any]:
    record: dict[str, Any] = {
        "source_seed_id": seed.source_seed_id,
        "topic_seed": seed.topic_seed,
        "needle_type": seed.chosen_identifier_type,
        "needle_value": seed.chosen_identifier_value,
        "pii_source_row_index": seed.source_seed_id,
        "pii_fields_used": seed.chosen_identifier_type,
        "pii_snippets_used": seed.chosen_identifier_snippet,
    }
    validation_errors: list[str] = []
    for key, target_chars in TARGET_LENGTHS.items():
        negative_text = expand_text_to_target_length(base_text, target_chars)
        positive_text, insert_position = insert_identifier_at_position(
            negative_text,
            seed.chosen_identifier_snippet,
            position="mid",
        )
        record[key] = negative_text
        positive_key = f"{key}_with_pii"
        record[positive_key] = positive_text
        record[f"{key}_pii_insert_positions"] = insert_position
        validation = validate_positive_text(positive_text, seed.chosen_identifier_snippet)
        if validation.status != "passed":
            validation_errors.append(f"{positive_key}: {validation.error}")
    record["validation_status"] = "passed" if not validation_errors else "failed"
    record["validation_error"] = " | ".join(validation_errors)
    return record


def build_needle_variants(base_text: str, seed: SeedIdentifiers) -> dict[str, Any]:
    record: dict[str, Any] = {
        "source_seed_id": seed.source_seed_id,
        "topic_seed": seed.topic_seed,
        "needle_type": seed.chosen_identifier_type,
        "needle_value": seed.chosen_identifier_value,
    }
    validation_errors: list[str] = []
    expanded: dict[str, str] = {}
    for key, target_chars in TARGET_LENGTHS.items():
        expanded[key] = expand_text_to_target_length(base_text, target_chars)
        record[key] = expanded[key]
    for base_key in ("16k_context", "32k_context"):
        for position in ("start", "mid", "end"):
            variant_key = f"{base_key.split('_')[0]}_needle_{position}_context"
            variant_text, insert_at = insert_identifier_at_position(
                expanded[base_key],
                seed.chosen_identifier_snippet,
                position=position,
            )
            record[variant_key] = variant_text
            record[f"{variant_key}_insert_position"] = insert_at
            validation = validate_positive_text(variant_text, seed.chosen_identifier_snippet)
            if validation.status != "passed":
                validation_errors.append(f"{variant_key}: {validation.error}")
    record["validation_status"] = "passed" if not validation_errors else "failed"
    record["validation_error"] = " | ".join(validation_errors)
    return record


def build_output_paths(output_dir: Path, dataset_prefix: str) -> OutputPaths:
    suffix = dataset_prefix.strip()
    return OutputPaths(
        clean_negative=output_dir / f"data_person_1000_non_pii_clean_{suffix}.csv",
        hard_negative=output_dir / f"data_person_1000_non_pii_hard_{suffix}.csv",
        long_context=output_dir / f"harmless_long_context_pii_{suffix}.csv",
        needle=output_dir / f"harmless_long_context_needle_{suffix}.csv",
    )


CLEAN_NEGATIVE_PROMPT = """
請根據以下個人背景欄位，撰寫一段自然、連貫、像真實案例描述的繁體中文正文。

要求：
1. 只能保留一般生活、工作、健康或制度脈絡，不得出現任何明確、可直接識別個人的資訊。
2. 不得出現姓名、醫師姓名、電話、電子郵件、精確地址、政府核發證件號碼、銀行帳號、卡號、病歷號。
3. 不得大量使用「某城市」「某銀行」「某帳戶」這種 placeholder 來硬湊匿名化語氣。
4. 可以提及一般情境，但要避免讓讀者能直接回推出特定個人。
5. 只輸出正文，不要條列、不要 JSON。

背景欄位：
姓名：{{ name }}
性別：{{ gender }}
年齡：{{ age }}
居住地：{{ location }}
職業：{{ occupation }}
症狀：{{ symptoms }}
診斷結果：{{ diagnosticOutcome }}
用藥：{{ medicationDetails }}
主治醫師：{{ doctor }}
交易資訊：{{ transactionDetails }}
信用分數：{{ creditScore }}
年收入：{{ income }}
""".strip()


HARD_NEGATIVE_PROMPT = """
請根據以下個人背景欄位，撰寫一段自然、連貫、帶有醫療與金融脈絡的繁體中文案例描述。

要求：
1. 可以保留敏感語意與個人背景脈絡，但不得出現任何直接識別子。
2. 必須移除或泛化姓名、醫師姓名、電話、電子郵件、精確地址、政府核發證件號碼、銀行帳號、卡號、病歷號。
3. 內容可提及就醫、費用、交易、收入、工作與生活壓力，但要維持去識別化。
4. 只輸出正文，不要條列、不要 JSON。

背景欄位：
姓名：{{ name }}
性別：{{ gender }}
年齡：{{ age }}
居住地：{{ location }}
職業：{{ occupation }}
症狀：{{ symptoms }}
診斷結果：{{ diagnosticOutcome }}
用藥：{{ medicationDetails }}
主治醫師：{{ doctor }}
交易資訊：{{ transactionDetails }}
信用分數：{{ creditScore }}
年收入：{{ income }}
""".strip()


LONG_CONTEXT_FILLER_PROMPT = """
請根據以下背景，生成一段繁體中文長篇 filler 文本，作為企業或個人案例敘事素材。

要求：
1. 只保留一般敘事、制度、流程、生活、工作、健康或財務情境。
2. 不得出現任何明確 direct identifier：姓名、醫師姓名、電話、電子郵件、精確地址、政府核發證件號碼、銀行帳號、卡號、病歷號。
3. 內容要自然、連貫、適合後續擴展成更長文本。
4. 請寫得比一般短文更完整，但仍只輸出正文，不要條列、不要 JSON。
5. 使用繁體中文、台灣可讀用語。

背景欄位：
姓名：{{ name }}
性別：{{ gender }}
年齡：{{ age }}
居住地：{{ location }}
職業：{{ occupation }}
症狀：{{ symptoms }}
診斷結果：{{ diagnosticOutcome }}
用藥：{{ medicationDetails }}
主治醫師：{{ doctor }}
交易資訊：{{ transactionDetails }}
信用分數：{{ creditScore }}
年收入：{{ income }}
""".strip()


def _import_data_designer() -> tuple[Any, Any]:
    try:
        import data_designer.config as dd
        from data_designer.interface import DataDesigner
    except ImportError as exc:  # pragma: no cover - exercised in real environment
        raise RuntimeError(
            "找不到 data_designer，請先在目標環境安裝 Data Designer 相關套件。"
        ) from exc
    return dd, DataDesigner


def _extract_dataframe(payload: Any) -> pd.DataFrame:
    for method_name in ("to_pandas", "to_dataframe"):
        method = getattr(payload, method_name, None)
        if callable(method):
            frame = method()
            if isinstance(frame, pd.DataFrame):
                return frame.copy()
    for attr_name in ("dataframe", "df"):
        frame = getattr(payload, attr_name, None)
        if isinstance(frame, pd.DataFrame):
            return frame.copy()
    records = getattr(payload, "records", None)
    if isinstance(records, list):
        return pd.DataFrame(records)
    if isinstance(payload, list):
        return pd.DataFrame(payload)
    raise TypeError(f"無法從 Data Designer 輸出轉成 DataFrame: {type(payload)!r}")


def _build_model_configs(dd: Any) -> list[Any]:
    return [
        dd.ModelConfig(
            alias=DEFAULT_MODEL_ALIAS,
            model=DEFAULT_MODEL_ID,
            provider=DEFAULT_PROVIDER,
            inference_parameters=dd.ChatCompletionInferenceParams(
                temperature=1.0,
                top_p=1.0,
                max_tokens=2048,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            ),
        )
    ]


def _build_generation_builder(dd: Any, seed_path: Path, *, column_name: str, prompt: str) -> Any:
    model_configs = _build_model_configs(dd)
    builder = dd.DataDesignerConfigBuilder(model_configs=model_configs)
    builder.with_seed_dataset(dd.LocalFileSeedSource(path=str(seed_path)))
    builder.add_column(
        dd.LLMTextColumnConfig(
            name=column_name,
            model_alias=DEFAULT_MODEL_ALIAS,
            prompt=prompt,
        )
    )
    return builder


def _run_generation_job(seed_path: Path, job: GenerationJob, *, preview_count: int, preview_only: bool) -> pd.DataFrame | None:
    dd, DataDesigner = _import_data_designer()
    app = DataDesigner()
    builder = _build_generation_builder(dd, seed_path, column_name=job.column_name, prompt=job.prompt)

    if preview_count > 0:
        preview_payload = app.preview(builder, num_records=preview_count)
        preview_df = _extract_dataframe(preview_payload)
        print(f"\n[preview] {job.label}")
        print(preview_df.head(preview_count).to_json(force_ascii=False, orient="records", indent=2))

    if preview_only:
        return None

    created_payload = app.create(builder, num_records=job.count, dataset_name=job.dataset_name)
    return _extract_dataframe(created_payload)


def _load_seed_rows(seed_path: Path) -> list[dict[str, Any]]:
    data = json.loads(seed_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"seed 檔案格式錯誤: {seed_path}")
    rows: list[dict[str, Any]] = []
    for row in data:
        if not isinstance(row, dict):
            raise ValueError(f"seed 檔案內容格式錯誤: {seed_path}")
        rows.append(dict(row))
    return rows


def _seed_for_output_row(row: Mapping[str, Any], fallback_seed: Mapping[str, Any], source_seed_id: int) -> SeedIdentifiers:
    merged = dict(fallback_seed)
    for key, value in row.items():
        if value is not None and not (isinstance(value, float) and pd.isna(value)):
            merged[key] = value
    return extract_seed_identifiers(merged, source_seed_id)


def _finalize_negative_dataframe(
    generated_df: pd.DataFrame,
    seed_rows: list[dict[str, Any]],
    *,
    negative_type: str,
) -> pd.DataFrame:
    finalized_rows: list[dict[str, Any]] = []
    validator = validate_clean_negative_text if negative_type == "clean" else validate_hard_negative_text
    for idx, row in generated_df.reset_index(drop=True).iterrows():
        source_seed_id = idx + 1
        seed = _seed_for_output_row(row.to_dict(), seed_rows[idx % len(seed_rows)], source_seed_id)
        paragraph = _safe_text(row.get("naturalParagraph"))
        validation = validator(paragraph, seed)
        generation_status = "ok" if validation.status == "passed" else "failed_validation"
        finalized_rows.append(
            {
                "id": source_seed_id,
                "source_seed_id": source_seed_id,
                "naturalParagraph": paragraph,
                "negative_type": negative_type,
                "generation_status": generation_status,
                "raw_response": paragraph,
                "validation_status": validation.status,
                "validation_error": validation.error,
                "chosen_identifier_type": seed.chosen_identifier_type,
                "chosen_identifier_value": seed.chosen_identifier_value,
                "topic_seed": seed.topic_seed,
            }
        )
    return pd.DataFrame(finalized_rows)


def _finalize_long_context_dataframe(generated_df: pd.DataFrame, seed_rows: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for idx, row in generated_df.reset_index(drop=True).iterrows():
        source_seed_id = idx + 1
        seed = _seed_for_output_row(row.to_dict(), seed_rows[idx % len(seed_rows)], source_seed_id)
        base_text = _safe_text(row.get("base_filler_case_report"))
        if not base_text:
            rows.append(
                {
                    "source_seed_id": source_seed_id,
                    "topic_seed": seed.topic_seed,
                    "validation_status": "failed",
                    "validation_error": "base_filler_case_report 為空",
                }
            )
            continue
        filler_validation = validate_hard_negative_text(base_text, seed)
        record = build_long_context_variants(base_text, seed)
        if filler_validation.status != "passed":
            record["validation_status"] = "failed"
            record["validation_error"] = (
                f"base_filler_case_report: {filler_validation.error}"
                + (f" | {record['validation_error']}" if record.get("validation_error") else "")
            )
        rows.append(record)
    return pd.DataFrame(rows)


def _finalize_needle_dataframe(generated_df: pd.DataFrame, seed_rows: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for idx, row in generated_df.reset_index(drop=True).iterrows():
        source_seed_id = idx + 1
        seed = _seed_for_output_row(row.to_dict(), seed_rows[idx % len(seed_rows)], source_seed_id)
        base_text = _safe_text(row.get("base_filler_case_report"))
        if not base_text:
            rows.append(
                {
                    "source_seed_id": source_seed_id,
                    "topic_seed": seed.topic_seed,
                    "validation_status": "failed",
                    "validation_error": "base_filler_case_report 為空",
                }
            )
            continue
        filler_validation = validate_hard_negative_text(base_text, seed)
        record = build_needle_variants(base_text, seed)
        if filler_validation.status != "passed":
            record["validation_status"] = "failed"
            record["validation_error"] = (
                f"base_filler_case_report: {filler_validation.error}"
                + (f" | {record['validation_error']}" if record.get("validation_error") else "")
            )
        rows.append(record)
    return pd.DataFrame(rows)


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"[write] {path} rows={len(df)}")


def _load_existing_positive(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"找不到既有 positive dataset: {path}")
    pd.read_csv(path, nrows=1)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 Data Designer 產生 direct-identifier-only 的 PII benchmark datasets")
    parser.add_argument("--seed-path", default=str(DEFAULT_SEED_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--baseline-count", type=int, default=DEFAULT_BASELINE_COUNT)
    parser.add_argument("--long-count", type=int, default=DEFAULT_LONG_COUNT)
    parser.add_argument("--preview-count", type=int, default=DEFAULT_PREVIEW_COUNT)
    parser.add_argument("--dataset-prefix", default=DEFAULT_DATASET_PREFIX)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-long-context", action="store_true")
    parser.add_argument("--skip-needle", action="store_true")
    parser.add_argument("--preview-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    ensure_dotenv_loaded()
    args = parse_args(argv)

    seed_path = _resolve_repo_path(args.seed_path)
    output_dir = _resolve_repo_path(args.output_dir)
    output_paths = build_output_paths(output_dir, args.dataset_prefix)

    if not seed_path.exists():
        raise FileNotFoundError(f"找不到 seed dataset: {seed_path}")
    _load_existing_positive(DEFAULT_POSITIVE_PATH)
    seed_rows = _load_seed_rows(seed_path)
    if not seed_rows:
        raise ValueError("seed dataset 不可為空")

    if not args.skip_baseline:
        clean_job = GenerationJob(
            label="baseline-clean-negative",
            column_name="naturalParagraph",
            prompt=CLEAN_NEGATIVE_PROMPT,
            count=args.baseline_count,
            dataset_name=f"{args.dataset_prefix}-baseline-clean",
        )
        hard_job = GenerationJob(
            label="baseline-hard-negative",
            column_name="naturalParagraph",
            prompt=HARD_NEGATIVE_PROMPT,
            count=args.baseline_count,
            dataset_name=f"{args.dataset_prefix}-baseline-hard",
        )
        clean_df = _run_generation_job(seed_path, clean_job, preview_count=args.preview_count, preview_only=args.preview_only)
        hard_df = _run_generation_job(seed_path, hard_job, preview_count=args.preview_count, preview_only=args.preview_only)
        if not args.preview_only:
            assert clean_df is not None and hard_df is not None
            _write_csv(_finalize_negative_dataframe(clean_df, seed_rows, negative_type="clean"), output_paths.clean_negative)
            _write_csv(_finalize_negative_dataframe(hard_df, seed_rows, negative_type="hard"), output_paths.hard_negative)

    filler_job = GenerationJob(
        label="long-context-filler",
        column_name="base_filler_case_report",
        prompt=LONG_CONTEXT_FILLER_PROMPT,
        count=args.long_count,
        dataset_name=f"{args.dataset_prefix}-long-filler",
    )

    if not args.skip_long_context:
        long_df = _run_generation_job(seed_path, filler_job, preview_count=args.preview_count, preview_only=args.preview_only)
        if not args.preview_only:
            assert long_df is not None
            _write_csv(_finalize_long_context_dataframe(long_df, seed_rows), output_paths.long_context)

    if not args.skip_needle:
        needle_df = _run_generation_job(seed_path, filler_job, preview_count=args.preview_count, preview_only=args.preview_only)
        if not args.preview_only:
            assert needle_df is not None
            _write_csv(_finalize_needle_dataframe(needle_df, seed_rows), output_paths.needle)

    if args.preview_only:
        print("preview-only 模式完成，未寫入 CSV。")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
