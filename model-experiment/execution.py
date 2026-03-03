"""
PII Guardrail Benchmark:
- OpenRouter: ibm-granite/granite-4.0-h-micro, openai/gpt-4.1-nano
- Local streaming guard model: Qwen/Qwen3Guard-Stream-4B

評估指標：TPR、FPR、Overhead Latency（ms）、Cost（USD）
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import gc
import asyncio
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol
from urllib import request
import numpy as np

import pandas as pd

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - fallback for environments without python-dotenv
    def load_dotenv(*args: Any, **kwargs: Any) -> bool:
        return False

try:
    import yaml
except Exception:  # pragma: no cover - fallback when pyyaml is unavailable
    yaml = None


# 專案根目錄加入 path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from prompt import build_prompts
from utils.result_compare import load_model_params
from utils.result_process import process_model_run, save_processed_results

# =============================================================================
# 設定
# =============================================================================

DATASET_PATH = ROOT / "dataset" / "data_person_1000_target.csv"
NON_PII_DATASET_PATH = ROOT / "dataset" / "data_person_1000_non_pii_100.csv"
RESULTS_DIR = ROOT / "model-experiment" / "results"
TASK_ID = "pii_binary"
CONTENT_COLUMN = "naturalParagraph"
GROUND_TRUTH_COLUMN = "ground_truth"

LEGACY_DEFAULT_MODEL_SPECS = (
    # ("ibm-granite/granite-4.0-h-micro", "openrouter"),
    # ("openai/gpt-4.1-nano", "openrouter"),
    ("Qwen/Qwen3Guard-Stream-4B", "qwen_stream"),
)
MODEL_SPECS_CONFIG_PATH = ROOT / "model-experiment" / "model_specs.yaml"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODELS_URL = f"{OPENROUTER_BASE_URL}/models"
QWEN_DEBUG_LOG_PATH = ROOT / "model-experiment" / "results" / "qwen_stream_debug.log"
QWEN_SAFETY_PATTERN = r"Safety:\s*(Safe|Unsafe|Controversial)"
QWEN_CATEGORY_PATTERN = (
    r"(Violent|Non-violent Illegal Acts|Sexual Content or Sexual Acts|PII|"
    r"Suicide & Self-Harm|Unethical Acts|Politically Sensitive Topics|"
    r"Copyright Violation|Jailbreak|None)"
)
QWEN_REFUSAL_PATTERN = r"Refusal:\s*(Yes|No)"

PII_BINARY_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "contains_pii": {"type": "boolean"},
        "label": {"type": "string", "enum": ["是", "否"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string"},
    },
    "required": ["contains_pii", "label", "confidence", "reason"],
    "additionalProperties": False,
}

_OPENROUTER_PRICING_CACHE: dict[str, tuple[float, float]] | None = None


def _append_qwen_debug(event: dict[str, Any]) -> None:
    """將 Qwen stream 偵錯事件追加寫入 JSONL。"""
    try:
        QWEN_DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            **event,
        }
        with open(QWEN_DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        # 偵錯寫檔失敗不應影響主流程
        return


def _release_cuda_memory(torch_module: Any | None = None) -> None:
    """
    盡可能釋放 CUDA 記憶體：
    1) 先 GC 讓 Python 物件解除參考
    2) 再由 torch CUDA allocator 釋放 cache/IPC 區塊
    """
    gc.collect()
    try:
        tm = torch_module
        if tm is None:
            import torch as tm  # type: ignore

        if tm.cuda.is_available():
            try:
                tm.cuda.synchronize()
            except Exception:
                pass
            tm.cuda.empty_cache()
            tm.cuda.ipc_collect()
    except Exception:
        # 記憶體回收失敗不應中斷主流程
        return


# =============================================================================
# 型別
# =============================================================================


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_id: str
    provider: str
    profile: str | None = None
    settings: dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceResult:
    output_json: str
    contains_pii: bool | None
    cost_usd: float
    prompt_tokens: int
    completion_tokens: int


class ModelRunner(Protocol):
    model_id: str
    provider: str

    def predict(self, content: str) -> InferenceResult:
        ...

    def close(self) -> None:
        ...


def _sanitize_model_key(text: str) -> str:
    sanitized = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return sanitized or "model"


def _build_legacy_default_specs() -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    for index, (model_id, provider) in enumerate(LEGACY_DEFAULT_MODEL_SPECS, start=1):
        key = f"legacy-{index}-{_sanitize_model_key(model_id)}"
        specs.append(ModelSpec(key=key, model_id=model_id, provider=provider))
    return specs


def load_model_specs_from_yaml(path: Path) -> list[ModelSpec]:
    """從 YAML 讀取模型設定；若檔案不存在則回傳 legacy defaults。"""
    if not path.exists():
        return _build_legacy_default_specs()

    if yaml is None:
        raise RuntimeError("缺少 pyyaml 套件，無法解析模型 YAML 設定")

    with open(path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}

    if not isinstance(payload, dict):
        raise ValueError(f"模型設定格式錯誤: {path} 需為 object")

    version = payload.get("version")
    if version != 1:
        raise ValueError(f"模型設定 version 錯誤: 目前僅支援 1，收到 {version!r}")

    models = payload.get("models")
    if not isinstance(models, list):
        raise ValueError(f"模型設定格式錯誤: {path} 的 models 必須為 list")

    seen_keys: set[str] = set()
    specs: list[ModelSpec] = []
    for idx, item in enumerate(models):
        if not isinstance(item, dict):
            raise ValueError(f"模型設定格式錯誤: models[{idx}] 必須為 object")

        enabled = item.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError(f"模型設定格式錯誤: models[{idx}].enabled 必須為 bool")
        if not enabled:
            continue

        key = item.get("key")
        model_id = item.get("model_id")
        provider = item.get("provider")
        profile = item.get("profile")
        settings = item.get("settings", {})

        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"模型設定格式錯誤: models[{idx}].key 必須為非空字串")
        if key in seen_keys:
            raise ValueError(f"模型設定格式錯誤: key 重複 {key!r}")
        seen_keys.add(key)

        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError(f"模型設定格式錯誤: models[{idx}].model_id 必須為非空字串")
        if not isinstance(provider, str) or not provider.strip():
            raise ValueError(f"模型設定格式錯誤: models[{idx}].provider 必須為非空字串")
        if profile is not None and not isinstance(profile, str):
            raise ValueError(f"模型設定格式錯誤: models[{idx}].profile 必須為字串或 null")
        if not isinstance(settings, dict):
            raise ValueError(f"模型設定格式錯誤: models[{idx}].settings 必須為 object")

        specs.append(
            ModelSpec(
                key=key.strip(),
                model_id=model_id.strip(),
                provider=provider.strip(),
                profile=profile.strip() if isinstance(profile, str) and profile.strip() else None,
                settings=settings,
            )
        )

    if not specs:
        raise ValueError(f"模型設定錯誤: {path} 沒有任何 enabled 模型")
    return specs


# =============================================================================
# JSON 解析工具
# =============================================================================


def _extract_json_object(raw_text: str) -> dict[str, Any] | None:
    """
    從模型輸出中擷取 JSON 物件。

    支援：純 JSON、code fence、文字中夾帶第一個 JSON object。
    """
    if not raw_text:
        return None

    text = raw_text.strip()
    if not text:
        return None

    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    cleaned = re.sub(r"^```(?:json)?\\s*", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"\\s*```$", "", cleaned, flags=re.IGNORECASE).strip()

    try:
        payload = json.loads(cleaned)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return None

    try:
        payload = json.loads(match.group(0))
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        return None


def _extract_partial_value(raw_text: str, key: str) -> str | None:
    """從不完整 JSON 文字中抽取欄位值（字串型）。"""
    if not raw_text:
        return None
    pattern = rf'"{re.escape(key)}"\s*:\s*"([^"\n\r}}]*)'
    match = re.search(pattern, raw_text, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1).strip()


def _extract_partial_number(raw_text: str, key: str) -> float | None:
    """從不完整 JSON 文字中抽取數值欄位。"""
    if not raw_text:
        return None
    pattern = rf'"{re.escape(key)}"\s*:\s*([+-]?\d+(?:\.\d+)?)'
    match = re.search(pattern, raw_text, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _coerce_bool(value: Any) -> bool | None:
    """將常見布林語意值轉成 bool。"""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"是", "有", "yes", "true", "1", "y"}:
            return True
        if text in {"否", "無", "no", "false", "0", "n"}:
            return False
    return None


def _extract_partial_contains_pii(raw_text: str) -> bool | None:
    """從不完整 JSON 或文字中推斷 contains_pii。"""
    if not raw_text:
        return None

    match = re.search(r'"contains_pii"\s*:\s*(true|false|1|0)', raw_text, flags=re.IGNORECASE)
    if match:
        return _coerce_bool(match.group(1))

    label = _extract_partial_value(raw_text, "label")
    parsed = _coerce_bool(label)
    if parsed is not None:
        return parsed

    lowered = raw_text.lower()
    if "contains_pii" in lowered and "true" in lowered:
        return True
    if "contains_pii" in lowered and "false" in lowered:
        return False
    return None


def _coerce_contains_pii(payload: dict[str, Any] | None) -> bool | None:
    if not payload:
        return None

    for key in ("contains_pii", "has_pii", "pii", "prediction", "label", "result"):
        if key in payload:
            parsed = _coerce_bool(payload.get(key))
            if parsed is not None:
                return parsed
    return None


def normalize_pii_binary_json(raw_text: str) -> dict[str, Any]:
    """將模型輸出正規化為統一 JSON。"""
    payload = _extract_json_object(raw_text)
    contains_pii = _coerce_contains_pii(payload)
    if contains_pii is None:
        contains_pii = _extract_partial_contains_pii(raw_text)

    confidence = None
    if payload is not None and "confidence" in payload:
        try:
            confidence_val = float(payload["confidence"])
            confidence = min(1.0, max(0.0, confidence_val))
        except (TypeError, ValueError):
            confidence = None
    if confidence is None:
        partial_conf = _extract_partial_number(raw_text, "confidence")
        if partial_conf is not None:
            confidence = min(1.0, max(0.0, partial_conf))

    reason = ""
    if payload is not None and "reason" in payload:
        reason = str(payload["reason"]).strip()
    if not reason:
        reason = _extract_partial_value(raw_text, "reason") or ""

    normalized = {
        "contains_pii": contains_pii,
        "label": "是" if contains_pii is True else "否" if contains_pii is False else "無法解析",
        "confidence": confidence,
        "reason": reason,
        "raw_text": raw_text,
    }
    return normalized


def parse_qwen_text_response(content: str) -> tuple[str | None, list[str], str | None]:
    """
    解析 Qwen Guard 文字輸出，例如：
    - Safety: Unsafe
    - Category: PII, ...
    - Refusal: Yes/No
    """
    if not content:
        return None, [], None

    safety_match = re.search(QWEN_SAFETY_PATTERN, content, re.IGNORECASE)
    refusal_match = re.search(QWEN_REFUSAL_PATTERN, content, re.IGNORECASE)
    categories = re.findall(QWEN_CATEGORY_PATTERN, content, re.IGNORECASE)

    label = safety_match.group(1) if safety_match else None
    refusal = refusal_match.group(1) if refusal_match else None
    normalized_categories = []
    for c in categories:
        c_norm = c.strip()
        if c_norm and c_norm not in normalized_categories:
            normalized_categories.append(c_norm)
    return label, normalized_categories, refusal


def _contains_pii_category(category_text: str | None, parsed_categories: list[str] | None = None) -> bool:
    """判斷分類結果是否包含 PII 類別。"""
    if parsed_categories:
        return any(c.strip().upper() == "PII" for c in parsed_categories)
    if not category_text:
        return False
    return "PII" in category_text.upper()


def _pick_indexed_value(value: Any, idx: int) -> str:
    """從 list-like 結果取指定 index，否則回傳字串值。"""
    if isinstance(value, list):
        if not value:
            return ""
        if idx < len(value):
            return str(value[idx])
        return str(value[-1])
    return str(value) if value is not None else ""


def _infer_first_pii_index(result: dict[str, Any]) -> int | None:
    """
    從 Qwen result 的序列欄位推估首次命中 PII category 的 token index。
    注意：PII 判斷僅依 category，不依賴 risk_level（Safe/Unsafe 皆可能含 PII）。
    若無法推估回傳 None。
    """
    cats = result.get("category")
    if not isinstance(cats, list):
        return None

    length = len(cats)
    if length <= 0:
        return None

    for i in range(length):
        c = _pick_indexed_value(cats, i)
        if _contains_pii_category(c):
            return i
    return None


# =============================================================================
# 資料準備與指標
# =============================================================================


def _redact_to_non_pii(text: str) -> str:
    """將含 PII 句子轉成去識別化句子，用於 FPR 測試。"""
    safe_templates = (
        "匿名化文本：這是一段一般流程說明，僅描述系統狀態與服務步驟。",
        "匿名化文本：本段為客服處理紀錄摘要，不含任何個人識別資訊。",
        "匿名化文本：這是產品使用回饋，內容僅有功能與體驗描述。",
        "匿名化文本：此段為營運報表文字，僅含統計結果與一般描述。",
    )
    idx = abs(hash(str(text))) % len(safe_templates)
    return safe_templates[idx]


def build_eval_dataset(
    source_df: pd.DataFrame,
    *,
    sample_limit: int,
    include_negative: bool,
    non_pii_df: pd.DataFrame | None = None,
    content_column: str = CONTENT_COLUMN,
) -> pd.DataFrame:
    """建立評測資料：PII 正樣本 + 負樣本（優先使用 non_pii_df）。"""
    if content_column not in source_df.columns:
        raise ValueError(f"找不到欄位: {content_column}")

    positives = source_df[[content_column]].head(sample_limit).copy()

    positives[GROUND_TRUTH_COLUMN] = True
    positives["sample_type"] = "positive"

    if not include_negative:
        return positives.reset_index(drop=True)

    negatives: pd.DataFrame
    if non_pii_df is not None and content_column in non_pii_df.columns:
        negatives = non_pii_df[[content_column]].dropna()
        if len(negatives) > sample_limit:
            negatives = negatives.sample(n=sample_limit, random_state=np.random.randint(0, 1_000_000))
        else:
            negatives = negatives.head(sample_limit).copy()
        if len(negatives) < sample_limit:
            # 若外部 non-PII 不足，補齊缺口（使用合成負樣本）
            needed = sample_limit - len(negatives)
            fallback = positives.head(needed).copy()
            fallback[content_column] = fallback[content_column].map(_redact_to_non_pii)
            negatives = pd.concat([negatives, fallback[[content_column]]], ignore_index=True)
    else:
        negatives = positives.copy()
        negatives[content_column] = negatives[content_column].map(_redact_to_non_pii)

    negatives[GROUND_TRUTH_COLUMN] = False
    negatives["sample_type"] = "negative"

    merged = pd.concat([positives, negatives], ignore_index=True)
    merged = merged.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return merged


def load_non_pii_dataset(path: Path, *, content_column: str = CONTENT_COLUMN) -> pd.DataFrame | None:
    """載入 non-PII 資料集；若不存在或格式不符則回傳 None。"""
    if not path.exists():
        return None

    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    if content_column not in df.columns:
        return None

    if "generation_status" in df.columns:
        df = df[df["generation_status"] == "ok"].copy()

    return df[[content_column]].dropna().reset_index(drop=True)


def compute_guardrail_metrics(
    predictions: list[bool | None],
    ground_truths: list[bool],
    latencies_ms: list[float],
    costs_usd: list[float],
) -> dict[str, Any]:
    """計算 TPR / FPR / Latency / Cost。"""
    if len(predictions) != len(ground_truths):
        raise ValueError("predictions 與 ground_truths 長度不一致")

    tp = tn = fp = fn = unparseable = 0
    for pred, truth in zip(predictions, ground_truths):
        if pred is None:
            unparseable += 1
            pred = False

        if truth and pred:
            tp += 1
        elif truth and not pred:
            fn += 1
        elif not truth and pred:
            fp += 1
        else:
            tn += 1

    pos_total = tp + fn
    neg_total = tn + fp

    tpr = tp / pos_total if pos_total else 0.0
    fpr = fp / neg_total if neg_total else 0.0

    lat_series = pd.Series(latencies_ms, dtype=float)
    avg_latency_ms = float(lat_series.mean()) if not lat_series.empty else 0.0
    p95_latency_ms = float(lat_series.quantile(0.95)) if not lat_series.empty else 0.0

    total_cost_usd = float(sum(costs_usd))
    avg_cost_usd = total_cost_usd / len(costs_usd) if costs_usd else 0.0

    return {
        "tpr": round(tpr, 4),
        "fpr": round(fpr, 4),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "unparseable": unparseable,
        "overhead_latency_ms_avg": round(avg_latency_ms, 2),
        "overhead_latency_ms_p95": round(p95_latency_ms, 2),
        "cost_usd_total": round(total_cost_usd, 6),
        "cost_usd_avg": round(avg_cost_usd, 6),
    }


# =============================================================================
# OpenRouter runner
# =============================================================================


def fetch_openrouter_pricing(api_key: str) -> dict[str, tuple[float, float]]:
    """從 OpenRouter models API 取得每個模型 token 單價（prompt/completion）。"""
    global _OPENROUTER_PRICING_CACHE
    if _OPENROUTER_PRICING_CACHE is not None:
        return _OPENROUTER_PRICING_CACHE

    req = request.Request(
        OPENROUTER_MODELS_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="GET",
    )

    pricing: dict[str, tuple[float, float]] = {}
    try:
        with request.urlopen(req, timeout=20) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        for item in payload.get("data", []):
            model_id = item.get("id")
            p = item.get("pricing") or {}
            prompt_price = p.get("prompt")
            completion_price = p.get("completion")
            if not model_id or prompt_price is None or completion_price is None:
                continue
            try:
                pricing[model_id] = (float(prompt_price), float(completion_price))
            except (TypeError, ValueError):
                continue
    except Exception:
        pricing = {}

    _OPENROUTER_PRICING_CACHE = pricing
    return pricing


class OpenRouterRunner:
    def __init__(self, spec: ModelSpec) -> None:
        try:
            from openai import OpenAI, AsyncOpenAI
        except ImportError as e:  # pragma: no cover - runtime dependency
            raise RuntimeError("缺少 openai 套件，請先安裝 requirements.txt") from e

        self.model_id = spec.model_id
        self.provider = "openrouter"
        self._settings = dict(spec.settings)

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("找不到 OPENROUTER_API_KEY，請確認 .env 或環境變數")

        self._client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
        self._async_client = AsyncOpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)

        headers: dict[str, str] = {}
        referer = os.getenv("OPENROUTER_HTTP_REFERER") or os.getenv("HTTP_REFERER")
        title = os.getenv("OPENROUTER_TITLE") or os.getenv("X_OPENROUTER_TITLE")
        if referer:
            headers["HTTP-Referer"] = referer
        if title:
            headers["X-OpenRouter-Title"] = title
        self._extra_headers = headers
        self._concurrency = max(
            1,
            int(self._settings.get("concurrency") or os.getenv("OPENROUTER_CONCURRENCY", "5")),
        )

        self._pricing = fetch_openrouter_pricing(api_key)

    @staticmethod
    def _extract_usage(completion: Any) -> tuple[int, int]:
        usage = getattr(completion, "usage", None)
        if usage is None:
            return 0, 0
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        return prompt_tokens, completion_tokens

    def _estimate_cost(
        self,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        completion: Any,
    ) -> float:
        # 優先使用 SDK 若有回 total_cost
        usage = getattr(completion, "usage", None)
        total_cost = getattr(usage, "total_cost", None) if usage else None
        if total_cost is not None:
            try:
                return float(total_cost)
            except (TypeError, ValueError):
                pass

        pricing = self._pricing.get(self.model_id)
        if not pricing:
            return 0.0

        prompt_price, completion_price = pricing
        return float(prompt_tokens * prompt_price + completion_tokens * completion_price)

    def _build_completion_kwargs(self, content: str) -> dict[str, Any]:
        prompts = build_prompts(
            model_id=self.model_id,
            task_id=TASK_ID,
            content=content,
        )

        messages = [
            {"role": "system", "content": prompts["system"]},
            {"role": "user", "content": prompts["user"]},
        ]

        temperature = self._settings.get("temperature", 0)
        max_tokens = self._settings.get("max_tokens", 128)
        response_format = self._settings.get("response_format", "json_object")

        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            kwargs["extra_body"] = {"response_format": {"type": str(response_format)}}
        if self._extra_headers:
            kwargs["extra_headers"] = self._extra_headers
        return kwargs

    def _completion_to_result(self, completion: Any) -> InferenceResult:
        raw_text = ""
        if getattr(completion, "choices", None):
            message = completion.choices[0].message
            raw_text = message.content if message and message.content else ""

        normalized = normalize_pii_binary_json(raw_text)
        prompt_tokens, completion_tokens = self._extract_usage(completion)
        cost_usd = self._estimate_cost(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            completion=completion,
        )

        return InferenceResult(
            output_json=json.dumps(normalized, ensure_ascii=False),
            contains_pii=normalized.get("contains_pii"),
            cost_usd=cost_usd,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    @property
    def concurrency(self) -> int:
        return self._concurrency

    async def aclose_async_client(self) -> None:
        client = getattr(self, "_async_client", None)
        if client is None:
            return

        try:
            close_fn = getattr(client, "close", None)
            if callable(close_fn):
                maybe_awaitable = close_fn()
                if hasattr(maybe_awaitable, "__await__"):
                    await maybe_awaitable
                return

            aclose_fn = getattr(client, "aclose", None)
            if callable(aclose_fn):
                maybe_awaitable = aclose_fn()
                if hasattr(maybe_awaitable, "__await__"):
                    await maybe_awaitable
        except Exception:
            # 關閉連線失敗不應影響主流程
            return

    def predict(self, content: str) -> InferenceResult:
        kwargs = self._build_completion_kwargs(content)
        completion = self._client.chat.completions.create(**kwargs)
        return self._completion_to_result(completion)

    async def predict_async(self, content: str) -> InferenceResult:
        kwargs = self._build_completion_kwargs(content)
        completion = await self._async_client.chat.completions.create(**kwargs)
        return self._completion_to_result(completion)

    def close(self) -> None:
        client = getattr(self, "_client", None)
        if client is not None:
            try:
                close_fn = getattr(client, "close", None)
                if callable(close_fn):
                    close_fn()
            except Exception:
                pass
        return None


def _resolve_hf_runtime_settings(
    settings: dict[str, Any],
    *,
    cuda_available: bool,
    bf16_available: bool = False,
) -> dict[str, Any]:
    trust_remote_code = bool(settings.get("trust_remote_code", True))

    dtype_cfg = str(settings.get("torch_dtype", "auto")).strip().lower()
    if dtype_cfg == "auto":
        if cuda_available:
            # Colab/T4 常不支援原生 bf16，auto 應優先選 float16 避免額外顯存壓力。
            torch_dtype_name = "bfloat16" if bf16_available else "float16"
        else:
            torch_dtype_name = "float32"
    elif dtype_cfg in {"bfloat16", "float16", "float32"}:
        torch_dtype_name = dtype_cfg
    else:
        raise ValueError(f"不支援的 torch_dtype: {dtype_cfg}")

    device_map_cfg = settings.get("device_map", "auto")
    if device_map_cfg in (None, "", "none"):
        device_map = None
    elif device_map_cfg == "auto":
        device_map = "auto" if cuda_available else None
    elif isinstance(device_map_cfg, str):
        device_map = device_map_cfg
    else:
        raise ValueError("device_map 必須為字串、'auto' 或 null")

    return {
        "trust_remote_code": trust_remote_code,
        "torch_dtype_name": torch_dtype_name,
        "device_map": device_map,
    }


# =============================================================================
# Hugging Face runners
# =============================================================================


class BaseHuggingFaceRunner:
    def __init__(self, spec: ModelSpec) -> None:
        self.model_id = spec.model_id
        self.provider = spec.provider
        self.profile = spec.profile
        self._settings = dict(spec.settings)

        try:
            import torch
            import transformers.configuration_utils as config_utils
        except ImportError as e:  # pragma: no cover - runtime dependency
            raise RuntimeError(
                "缺少 transformers/torch 套件，請先安裝 requirements.txt"
            ) from e

        if not hasattr(config_utils, "layer_type_validation"):
            # 兼容舊版 transformers，供部分 remote-code 模型匯入使用。
            def _layer_type_validation(*args: Any, **kwargs: Any) -> None:
                return None

            config_utils.layer_type_validation = _layer_type_validation

        self._torch = torch
        runtime = _resolve_hf_runtime_settings(
            self._settings,
            cuda_available=bool(torch.cuda.is_available()),
            bf16_available=bool(torch.cuda.is_bf16_supported()) if torch.cuda.is_available() else False,
        )
        self._trust_remote_code = runtime["trust_remote_code"]
        self._torch_dtype_name = runtime["torch_dtype_name"]
        self._device_map = runtime["device_map"]
        self._torch_dtype = getattr(torch, self._torch_dtype_name)

    def _tokenizer_kwargs(self) -> dict[str, Any]:
        return {"trust_remote_code": self._trust_remote_code}

    def _model_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "trust_remote_code": self._trust_remote_code,
            "torch_dtype": self._torch_dtype,
        }
        if self._device_map is not None:
            kwargs["device_map"] = self._device_map
            max_memory = self._settings.get("max_memory")
            if isinstance(max_memory, dict) and max_memory:
                kwargs["max_memory"] = max_memory
            else:
                cuda_gib = self._settings.get("max_memory_cuda_gib")
                cpu_gib = self._settings.get("max_memory_cpu_gib")
                if cuda_gib is not None or cpu_gib is not None:
                    mm: dict[Any, str] = {}
                    if cuda_gib is not None:
                        mm[0] = f"{int(cuda_gib)}GiB"
                    if cpu_gib is not None:
                        mm["cpu"] = f"{int(cpu_gib)}GiB"
                    if mm:
                        kwargs["max_memory"] = mm

            offload_folder = self._settings.get("offload_folder")
            if offload_folder:
                kwargs["offload_folder"] = str(offload_folder)
        return kwargs

    def close(self) -> None:
        # 釋放 HF 模型資源，避免多模型序列 benchmark 時 GPU OOM。
        for attr in ("_model", "_tokenizer", "_input_device"):
            if hasattr(self, attr):
                obj = getattr(self, attr)
                setattr(self, attr, None)
                try:
                    del obj
                except Exception:
                    pass

        _release_cuda_memory(getattr(self, "_torch", None))


class GraniteHuggingFaceRunner(BaseHuggingFaceRunner):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__(spec)
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:  # pragma: no cover - runtime dependency
            raise RuntimeError("缺少 transformers 套件，請先安裝 requirements.txt") from e

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            **self._tokenizer_kwargs(),
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            **self._model_kwargs(),
        ).eval()

        self._input_device = None
        if self._device_map is None:
            device_name = "cuda" if self._torch.cuda.is_available() else "cpu"
            self._model = self._model.to(device_name)
            self._input_device = self._torch.device(device_name)
        else:
            self._input_device = self._detect_input_device()

        if getattr(self._tokenizer, "pad_token", None) is None:
            eos_token = getattr(self._tokenizer, "eos_token", None)
            if eos_token is not None:
                self._tokenizer.pad_token = eos_token

    def _detect_input_device(self) -> Any:
        """在 device_map=auto 等情況下，找出模型實際接受 input_ids 的裝置。"""
        try:
            input_embeddings = self._model.get_input_embeddings()
            if input_embeddings is not None and hasattr(input_embeddings, "weight"):
                emb_device = input_embeddings.weight.device
                if str(emb_device) != "meta":
                    return emb_device
        except Exception:
            pass

        try:
            first_param_device = next(self._model.parameters()).device
            if str(first_param_device) != "meta":
                return first_param_device
        except Exception:
            pass

        return self._torch.device("cuda:0" if self._torch.cuda.is_available() else "cpu")

    def _build_prompt_text(self, messages: list[dict[str, str]]) -> str:
        apply_template = getattr(self._tokenizer, "apply_chat_template", None)
        if callable(apply_template):
            try:
                return self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                return self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

        return "\n".join(f"{m['role']}: {m['content']}" for m in messages)

    def _build_generation_kwargs(self) -> dict[str, Any]:
        do_sample_raw = self._settings.get("do_sample", False)
        do_sample = _coerce_bool(do_sample_raw)
        if do_sample is None:
            raise ValueError(
                f"settings.do_sample 必須為布林值（true/false），目前收到: {do_sample_raw!r}"
            )
        use_cache_raw = self._settings.get("use_cache", False)
        use_cache = _coerce_bool(use_cache_raw)
        if use_cache is None:
            raise ValueError(
                f"settings.use_cache 必須為布林值（true/false），目前收到: {use_cache_raw!r}"
            )

        kwargs: dict[str, Any] = {
            "max_new_tokens": int(self._settings.get("max_new_tokens", 128)),
            "do_sample": do_sample,
            "use_cache": use_cache,
        }
        temperature = self._settings.get("temperature")
        if kwargs["do_sample"] and temperature is not None:
            kwargs["temperature"] = float(temperature)
        return kwargs

    def predict(self, content: str) -> InferenceResult:
        prompts = build_prompts(
            model_id=self.model_id,
            task_id=TASK_ID,
            content=content,
        )
        messages = [
            {"role": "system", "content": prompts["system"]},
            {"role": "user", "content": prompts["user"]},
        ]
        prompt_text = self._build_prompt_text(messages)

        max_input_tokens = self._settings.get("max_input_tokens")
        token_kwargs: dict[str, Any] = {"return_tensors": "pt"}
        if max_input_tokens is not None:
            token_kwargs["truncation"] = True
            token_kwargs["max_length"] = int(max_input_tokens)

        model_inputs = self._tokenizer(prompt_text, **token_kwargs)
        if self._input_device is not None:
            model_inputs = {
                k: (v.to(self._input_device) if hasattr(v, "to") else v)
                for k, v in model_inputs.items()
            }

        with self._torch.no_grad():
            generated_ids = self._model.generate(
                **model_inputs,
                **self._build_generation_kwargs(),
            )

        input_len = int(model_inputs["input_ids"].shape[-1]) if "input_ids" in model_inputs else 0
        output_row = generated_ids[0]
        generated_part = output_row[input_len:] if input_len and len(output_row) >= input_len else output_row
        completion_tokens = int(generated_part.shape[0]) if hasattr(generated_part, "shape") else 0
        raw_text = self._tokenizer.decode(generated_part, skip_special_tokens=True).strip()

        normalized = normalize_pii_binary_json(raw_text)
        return InferenceResult(
            output_json=json.dumps(normalized, ensure_ascii=False),
            contains_pii=normalized.get("contains_pii"),
            cost_usd=0.0,
            prompt_tokens=input_len,
            completion_tokens=completion_tokens,
        )


class QwenGuardStreamRunner(BaseHuggingFaceRunner):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__(spec)

        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:  # pragma: no cover - runtime dependency
            raise RuntimeError("缺少 transformers 套件，請先安裝 requirements.txt") from e

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            **self._tokenizer_kwargs(),
        )
        self._model = AutoModel.from_pretrained(
            self.model_id,
            **self._model_kwargs(),
        ).eval()

    @staticmethod
    def _last_item(value: Any) -> str:
        if isinstance(value, list):
            return str(value[-1]) if value else ""
        return str(value) if value is not None else ""

    def _stream_moderate_single_token(
        self,
        token_ids: Any,
        token_index: int,
        stream_state: Any,
    ) -> tuple[dict[str, Any], Any]:
        """
        對單一 token 嘗試不同輸入形態，兼容不同 Qwen3Guard-Stream 版本實作差異。
        """
        token_1d = token_ids[token_index : token_index + 1]
        attempts: list[tuple[str, Any]] = [("tensor_1d", token_1d)]
        if hasattr(token_1d, "unsqueeze"):
            attempts.insert(0, ("tensor_2d", token_1d.unsqueeze(0)))
        token_0d = token_ids[token_index]
        attempts.append(("tensor_0d", token_0d))
        try:
            attempts.append(("int", int(token_0d.item())))
        except Exception:
            pass

        errors: list[str] = []
        for input_kind, token_input in attempts:
            try:
                result, next_state = self._model.stream_moderate_from_ids(
                    token_input,
                    role="user",
                    stream_state=stream_state,
                )
                return result, next_state
            except Exception as e:
                errors.append(f"{input_kind}: {e}")
                _append_qwen_debug(
                    {
                        "event": "token_input_attempt_failed",
                        "model_id": self.model_id,
                        "token_index": token_index,
                        "input_kind": input_kind,
                        "error": str(e),
                    }
                )

        raise RuntimeError("; ".join(errors))

    def predict(self, content: str) -> InferenceResult:
        messages = [{"role": "user", "content": content}]

        try:
            text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=bool(self._settings.get("enable_thinking", False)),
            )
        except TypeError:
            text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

        model_inputs = self._tokenizer(text, return_tensors="pt")
        token_ids = model_inputs.input_ids[0]

        stream_state = None
        result: dict[str, Any] = {}
        processed_tokens = 0
        early_stopped = False
        trigger_idx: int | None = None
        inferred_pii = False
        inferred_trigger_idx: int | None = None
        stream_mode = "token_by_token"
        token_count = int(token_ids.shape[0]) if len(token_ids.shape) > 0 else 0
        _append_qwen_debug(
            {
                "event": "start_predict",
                "model_id": self.model_id,
                "token_count": token_count,
                "content_preview": content[:120],
            }
        )
        try:
            # 逐 token moderation：命中 PII category 立即 early stop。
            for i in range(token_count):
                try:
                    result, stream_state = self._stream_moderate_single_token(
                        token_ids=token_ids,
                        token_index=i,
                        stream_state=stream_state,
                    )
                except Exception as e:
                    if i == 0:
                        # 若連第一個 token 都無法串流，回退為一次性 user moderation。
                        stream_mode = "full_sequence_fallback"
                        _append_qwen_debug(
                            {
                                "event": "fallback_to_full_sequence",
                                "model_id": self.model_id,
                                "reason": str(e),
                            }
                        )
                        result, stream_state = self._model.stream_moderate_from_ids(
                            token_ids,
                            role="user",
                            stream_state=stream_state,
                        )
                        processed_tokens = token_count
                        break
                    _append_qwen_debug(
                        {
                            "event": "token_step_failed",
                            "model_id": self.model_id,
                            "token_index": i,
                            "error": str(e),
                        }
                    )
                    raise RuntimeError(f"stream token step failed at index={i}: {e}") from e

                processed_tokens = i + 1
                step_category = self._last_item(result.get("category"))
                if _contains_pii_category(step_category):
                    early_stopped = True
                    trigger_idx = i
                    _append_qwen_debug(
                        {
                            "event": "early_stop_triggered_by_pii_category",
                            "model_id": self.model_id,
                            "token_index": i,
                            "category": step_category,
                        }
                    )
                    break

            # 防守式回補：若上方沒觸發，但序列中存在 PII，僅作推估標記。
            # 注意：這不代表真實 early stop，不應改寫 early_stopped。
            if not early_stopped and result:
                inferred_idx = _infer_first_pii_index(result)
                if inferred_idx is not None:
                    inferred_pii = True
                    inferred_trigger_idx = inferred_idx
                    _append_qwen_debug(
                        {
                            "event": "pii_inferred_from_result_without_early_stop",
                            "model_id": self.model_id,
                            "token_index": inferred_idx,
                            "stream_mode": stream_mode,
                        }
                    )
                elif processed_tokens == 0:
                    processed_tokens = token_count
        finally:
            if stream_state is not None:
                try:
                    self._model.close_stream(stream_state)
                except Exception:
                    pass

        risk_level = self._last_item(result.get("risk_level"))
        category = self._last_item(result.get("category"))
        text_output = self._last_item(result.get("content"))

        # 兼容 Qwen 文字回傳格式（Safety/Category/Refusal）
        parsed_label = None
        parsed_categories: list[str] = []
        parsed_refusal = None
        if text_output:
            parsed_label, parsed_categories, parsed_refusal = parse_qwen_text_response(text_output)
            if parsed_label:
                risk_level = parsed_label
            if parsed_categories:
                category = ", ".join(parsed_categories)

        contains_pii = None
        if parsed_categories:
            contains_pii = _contains_pii_category(category, parsed_categories)
        elif category:
            contains_pii = _contains_pii_category(category)

        reason_parts = []
        if risk_level:
            reason_parts.append(f"risk_level={risk_level}")
        if category:
            reason_parts.append(f"category={category}")
        if parsed_refusal:
            reason_parts.append(f"refusal={parsed_refusal}")

        raw_text = json.dumps(
            {
                "risk_level": risk_level,
                "category": category,
                "refusal": parsed_refusal,
                "text_output": text_output,
                "stream_mode": stream_mode,
                "early_stopped": early_stopped,
                "trigger_index": trigger_idx,
                "inferred_pii": inferred_pii,
                "inferred_trigger_index": inferred_trigger_idx,
                "processed_tokens": processed_tokens,
                "total_tokens": int(token_ids.shape[0]),
            },
            ensure_ascii=False,
        )

        normalized = {
            "contains_pii": contains_pii,
            "label": "是" if contains_pii is True else "否" if contains_pii is False else "無法解析",
            "confidence": None,
            "reason": "; ".join(reason_parts),
            "raw_text": raw_text,
            "risk_level": risk_level,
            "category": category,
            "refusal": parsed_refusal,
            "stream_mode": stream_mode,
            "early_stopped": early_stopped,
            "trigger_index": trigger_idx,
            "inferred_pii": inferred_pii,
            "inferred_trigger_index": inferred_trigger_idx,
            "processed_tokens": processed_tokens,
            "total_tokens": int(token_ids.shape[0]),
        }

        return InferenceResult(
            output_json=json.dumps(normalized, ensure_ascii=False),
            contains_pii=contains_pii,
            cost_usd=0.0,
            prompt_tokens=int(token_ids.shape[0]),
            completion_tokens=0,
        )

    def close(self) -> None:
        return None


# =============================================================================
# Benchmark 流程
# =============================================================================


def _create_openrouter_runner(spec: ModelSpec) -> ModelRunner:
    return OpenRouterRunner(spec)


def _create_qwen_stream_runner(spec: ModelSpec) -> ModelRunner:
    return QwenGuardStreamRunner(spec)


def _create_huggingface_runner(spec: ModelSpec) -> ModelRunner:
    profile = (spec.profile or "granite_guard_json").strip().lower()
    if profile in {"granite_guard_json", "granite", "granite_hf"}:
        return GraniteHuggingFaceRunner(spec)
    if profile in {"qwen_stream", "qwen_guard_stream"}:
        return QwenGuardStreamRunner(spec)
    raise ValueError(f"未知 huggingface profile: {spec.profile!r}")


RUNNER_FACTORIES: dict[str, Callable[[ModelSpec], ModelRunner]] = {
    "openrouter": _create_openrouter_runner,
    "huggingface": _create_huggingface_runner,
    "qwen_stream": _create_qwen_stream_runner,
}


def create_runner(spec: ModelSpec) -> ModelRunner:
    factory = RUNNER_FACTORIES.get(spec.provider)
    if factory is None:
        available = ", ".join(sorted(RUNNER_FACTORIES.keys()))
        raise ValueError(f"未知 provider: {spec.provider}（可用: {available}）")
    return factory(spec)


def _build_empty_content_inference() -> InferenceResult:
    return InferenceResult(
        output_json=json.dumps(
            {
                "contains_pii": None,
                "label": "無法解析",
                "confidence": None,
                "reason": "empty content",
                "raw_text": "",
            },
            ensure_ascii=False,
        ),
        contains_pii=None,
        cost_usd=0.0,
        prompt_tokens=0,
        completion_tokens=0,
    )


def _build_model_error_inference(error: Exception) -> InferenceResult:
    return InferenceResult(
        output_json=json.dumps(
            {
                "contains_pii": None,
                "label": "無法解析",
                "confidence": None,
                "reason": f"model_error: {error}",
                "raw_text": "",
            },
            ensure_ascii=False,
        ),
        contains_pii=None,
        cost_usd=0.0,
        prompt_tokens=0,
        completion_tokens=0,
    )


async def _predict_openrouter_async_batch(
    rows: list[tuple[int, str]],
    runner: Any,
    *,
    concurrency: int,
) -> tuple[list[InferenceResult], list[float]]:
    semaphore = asyncio.Semaphore(max(1, int(concurrency)))
    inference_results: list[InferenceResult] = [_build_empty_content_inference() for _ in rows]
    row_latencies_ms: list[float] = [0.0 for _ in rows]

    async def _run_one(slot_idx: int, content: str) -> None:
        async with semaphore:
            row_start = time.perf_counter()
            try:
                inference = await runner.predict_async(content)
            except Exception as e:
                inference = _build_model_error_inference(e)
            row_latencies_ms[slot_idx] = (time.perf_counter() - row_start) * 1000
            inference_results[slot_idx] = inference

    tasks = [_run_one(slot_idx, content) for slot_idx, (_, content) in enumerate(rows)]
    try:
        if tasks:
            await asyncio.gather(*tasks)
    finally:
        if hasattr(runner, "aclose_async_client"):
            try:
                await runner.aclose_async_client()
            except Exception:
                pass
    return inference_results, row_latencies_ms


def run_single_model(
    df: pd.DataFrame,
    spec: ModelSpec,
    *,
    content_column: str = CONTENT_COLUMN,
    ground_truth_column: str = GROUND_TRUTH_COLUMN,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    對單一模型執行 benchmark。

    Returns:
        (result_df, metrics_meta)
    """
    runner = create_runner(spec)

    outputs: list[str] = []
    predictions: list[bool | None] = []
    latencies_ms: list[float] = []
    costs_usd: list[float] = []
    prompt_tokens_total = 0
    completion_tokens_total = 0

    started = time.perf_counter()
    try:
        if spec.provider == "openrouter" and hasattr(runner, "predict_async"):
            prepared_rows: list[tuple[int, str]] = []
            if content_column in df.columns:
                raw_contents = df[content_column].tolist()
            else:
                raw_contents = ["" for _ in range(len(df))]
            for idx, raw_content in enumerate(raw_contents):
                content = raw_content
                content = "" if pd.isna(content) else str(content).strip()
                prepared_rows.append((idx, content))

            valid_slots: list[int] = []
            valid_rows: list[tuple[int, str]] = []
            inference_results: list[InferenceResult] = [_build_empty_content_inference() for _ in prepared_rows]
            row_latencies_ms: list[float] = [0.0 for _ in prepared_rows]

            for slot_idx, (_, content) in enumerate(prepared_rows):
                if content:
                    valid_slots.append(slot_idx)
                    valid_rows.append(prepared_rows[slot_idx])

            if valid_rows:
                concurrency = int(getattr(runner, "concurrency", 4) or 4)
                valid_inferences, valid_latencies = asyncio.run(
                    _predict_openrouter_async_batch(
                        valid_rows,
                        runner,
                        concurrency=concurrency,
                    )
                )
                for local_idx, slot_idx in enumerate(valid_slots):
                    inference_results[slot_idx] = valid_inferences[local_idx]
                    row_latencies_ms[slot_idx] = valid_latencies[local_idx]

            for slot_idx in range(len(prepared_rows)):
                inference = inference_results[slot_idx]
                latencies_ms.append(row_latencies_ms[slot_idx])
                costs_usd.append(inference.cost_usd)
                outputs.append(inference.output_json)
                predictions.append(inference.contains_pii)
                prompt_tokens_total += inference.prompt_tokens
                completion_tokens_total += inference.completion_tokens
        else:
            for _, row in df.iterrows():
                content = row.get(content_column, "")
                content = "" if pd.isna(content) else str(content).strip()

                if not content:
                    inference = _build_empty_content_inference()
                    latencies_ms.append(0.0)
                else:
                    row_start = time.perf_counter()
                    try:
                        inference = runner.predict(content)
                    except Exception as e:
                        inference = _build_model_error_inference(e)
                    latencies_ms.append((time.perf_counter() - row_start) * 1000)

                costs_usd.append(inference.cost_usd)
                outputs.append(inference.output_json)
                predictions.append(inference.contains_pii)
                prompt_tokens_total += inference.prompt_tokens
                completion_tokens_total += inference.completion_tokens
    finally:
        runner.close()
        _release_cuda_memory()

    elapsed = time.perf_counter() - started

    result_df, metrics = process_model_run(
        df,
        outputs,
        TASK_ID,
        content_column=content_column,
        ground_truth_column=ground_truth_column,
    )

    result_df["sample_type"] = df.get("sample_type", "unknown")
    result_df["latency_ms"] = [round(v, 2) for v in latencies_ms]
    result_df["cost_usd"] = [round(v, 8) for v in costs_usd]
    result_df["model_key"] = spec.key
    result_df["model_id"] = spec.model_id
    result_df["provider"] = spec.provider

    truths = [bool(v) for v in df[ground_truth_column].tolist()]
    guard_metrics = compute_guardrail_metrics(predictions, truths, latencies_ms, costs_usd)

    meta: dict[str, Any] = {
        "model_key": spec.key,
        "model_id": spec.model_id,
        "provider": spec.provider,
        "execution_time_sec": round(elapsed, 2),
        "model_params_b": load_model_params(spec.model_id),
        "prompt_tokens_total": prompt_tokens_total,
        "completion_tokens_total": completion_tokens_total,
        **guard_metrics,
    }

    if metrics is not None:
        meta.update(metrics.to_dict())

    return result_df, meta


def build_model_specs(
    model_ids: list[str] | None = None,
    *,
    models_config_path: Path = MODEL_SPECS_CONFIG_PATH,
) -> list[ModelSpec]:
    specs = load_model_specs_from_yaml(models_config_path)
    if not model_ids:
        return specs

    by_key = {spec.key: spec for spec in specs}
    by_model_id: dict[str, list[ModelSpec]] = {}
    for spec in specs:
        by_model_id.setdefault(spec.model_id, []).append(spec)

    selected: list[ModelSpec] = []
    seen_keys: set[str] = set()
    unknown: list[str] = []

    for selector in model_ids:
        if selector in by_key:
            spec = by_key[selector]
            if spec.key not in seen_keys:
                selected.append(spec)
                seen_keys.add(spec.key)
            continue

        matched = by_model_id.get(selector, [])
        if matched:
            for spec in matched:
                if spec.key not in seen_keys:
                    selected.append(spec)
                    seen_keys.add(spec.key)
            continue

        unknown.append(selector)

    if unknown:
        available_keys = ", ".join(sorted(by_key.keys()))
        available_model_ids = ", ".join(sorted(by_model_id.keys()))
        raise ValueError(
            "找不到指定模型: "
            f"{unknown}。可用 key: [{available_keys}]；可用 model_id: [{available_model_ids}]"
        )

    if not selected:
        raise ValueError("沒有任何模型被選取，請檢查 --models 參數")
    return selected


def main(
    *,
    sample_limit: int = 100,
    include_negative: bool = True,
    model_ids: list[str] | None = None,
    non_pii_dataset_path: Path = NON_PII_DATASET_PATH,
    models_config_path: Path = MODEL_SPECS_CONFIG_PATH,
) -> None:
    """主流程：載入資料、跑模型、輸出 benchmark 結果。"""
    load_dotenv()

    if sample_limit <= 0:
        raise ValueError("sample_limit 必須 > 0")

    df = pd.read_csv(DATASET_PATH)
    non_pii_df = (
        load_non_pii_dataset(non_pii_dataset_path, content_column=CONTENT_COLUMN)
        if include_negative
        else None
    )

    eval_df = build_eval_dataset(
        df,
        sample_limit=sample_limit,
        include_negative=include_negative,
        non_pii_df=non_pii_df,
        content_column=CONTENT_COLUMN,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    specs = build_model_specs(model_ids, models_config_path=models_config_path)
    if include_negative:
        if non_pii_df is not None:
            print(
                f"載入 {len(eval_df)} 筆評測資料（正樣本 {sample_limit} + non-PII 負樣本 {sample_limit}，來源: {non_pii_dataset_path}）"
            )
        else:
            print(f"載入 {len(eval_df)} 筆評測資料（正樣本 {sample_limit} + 合成負樣本 {sample_limit}）")
    else:
        print(f"載入 {len(eval_df)} 筆評測資料（正樣本 {sample_limit} 筆）")

    all_metrics: list[dict[str, Any]] = []
    for spec in specs:
        print(f"\\n執行模型: {spec.key} | {spec.model_id} ({spec.provider}) ...")
        try:
            result_df, meta = run_single_model(eval_df, spec)
            all_metrics.append(meta)

            safe_name = f"{spec.key}__{spec.model_id}".replace("/", "_").replace(":", "_")
            out_csv = RESULTS_DIR / f"{safe_name}_pii_benchmark_results.csv"
            save_processed_results(result_df, None, str(out_csv), extra_info=meta)
            print(
                f"  TPR={meta.get('tpr', 0):.4f} | FPR={meta.get('fpr', 0):.4f} | "
                f"Latency(avg)={meta.get('overhead_latency_ms_avg', 0):.2f}ms | "
                f"Cost=${meta.get('cost_usd_total', 0):.6f}"
            )
            print(f"  結果已存: {out_csv}")
        except Exception as e:
            error_meta = {
                "model_id": spec.model_id,
                "provider": spec.provider,
                "status": "failed",
                "error": str(e),
            }
            all_metrics.append(error_meta)
            print(f"  執行失敗: {e}")

    compare_df = pd.DataFrame(all_metrics)
    if not compare_df.empty:
        sort_cols = ["tpr", "fpr", "overhead_latency_ms_avg", "cost_usd_total"]
        if all(col in compare_df.columns for col in sort_cols):
            compare_df = compare_df.sort_values(
                by=sort_cols,
                ascending=[False, True, True, True],
            ).reset_index(drop=True)

    compare_path = RESULTS_DIR / "pii_benchmark_comparison.csv"
    compare_df.to_csv(compare_path, index=False, encoding="utf-8")

    print(f"\\n比較表已存: {compare_path}")
    if not compare_df.empty:
        display_cols = [
            "model_key",
            "model_id",
            "provider",
            "status",
            "tpr",
            "fpr",
            "overhead_latency_ms_avg",
            "cost_usd_total",
            "execution_time_sec",
            "error",
        ]
        print(compare_df[[c for c in display_cols if c in compare_df.columns]].to_string(index=False))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PII Guardrail Benchmark")
    parser.add_argument(
        "-n",
        "--sample-limit",
        type=int,
        default=100,
        help="正樣本數（預設 100）",
    )
    parser.add_argument(
        "--no-negative",
        action="store_true",
        help="只跑正樣本（不建立負樣本，FPR 將無意義）",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="",
        help="指定模型清單（可用 key 或 model_id），以逗號分隔；未指定則使用 YAML 中 enabled=true 的模型",
    )
    parser.add_argument(
        "--models-config",
        type=str,
        default=str(MODEL_SPECS_CONFIG_PATH),
        help="模型 YAML 設定路徑（預設 model-experiment/model_specs.yaml）",
    )
    parser.add_argument(
        "--non-pii-dataset",
        type=str,
        default=str(NON_PII_DATASET_PATH),
        help="non-PII CSV 路徑（預設 dataset/data_person_1000_non_pii_100.csv）",
    )

    args = parser.parse_args()
    model_ids = [m.strip() for m in args.models.split(",") if m.strip()] or None

    main(
        sample_limit=args.sample_limit,
        include_negative=not args.no_negative,
        model_ids=model_ids,
        non_pii_dataset_path=Path(args.non_pii_dataset),
        models_config_path=Path(args.models_config),
    )
