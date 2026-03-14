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
import time
import inspect
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
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional
    class _NoopTqdm:
        def __init__(self, iterable: Any = None, total: int | None = None, **_: Any) -> None:
            self._iterable = iterable
            self.total = total

        def __iter__(self):
            if self._iterable is None:
                return iter(())
            return iter(self._iterable)

        def update(self, _: int = 1) -> None:
            return None

        def close(self) -> None:
            return None

    def tqdm(iterable: Any = None, **kwargs: Any):  # type: ignore[override]
        return _NoopTqdm(iterable=iterable, total=kwargs.get("total"))

ROOT = Path(__file__).resolve().parent.parent
from benchmarking.env import get_env
from benchmarking.tasks import get_task_definition

# =============================================================================
# 設定
# =============================================================================

TASK_ID = "pii_binary"
CONTENT_COLUMN = "content"
GROUND_TRUTH_COLUMN = "ground_truth"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODELS_URL = f"{OPENROUTER_BASE_URL}/models"
OPENAI_COMPAT_DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
QWEN_DEBUG_LOG_PATH = ROOT / "results" / "qwen_stream_debug.log"
QWEN_SAFETY_PATTERN = r"Safety:\s*(Safe|Unsafe|Controversial)"
QWEN_CATEGORY_PATTERN = (
    r"(Violent|Non-violent Illegal Acts|Sexual Content or Sexual Acts|PII|"
    r"Suicide & Self-Harm|Unethical Acts|Politically Sensitive Topics|"
    r"Copyright Violation|Jailbreak|None)"
)
QWEN_REFUSAL_PATTERN = r"Refusal:\s*(Yes|No)"

_OPENROUTER_PRICING_CACHE: dict[str, tuple[float, float]] | None = None


def _qwen_debug_enabled() -> bool:
    return get_env("GUARDIAN_BENCHMARK_QWEN_DEBUG", "").strip().lower() in {"1", "true", "yes"}


def _append_qwen_debug(event: dict[str, Any]) -> None:
    """將 Qwen stream 偵錯事件追加寫入 JSONL。"""
    if not _qwen_debug_enabled():
        return
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


def _resolve_vram_torch_module(runner: Any) -> Any | None:
    tm = getattr(runner, "_torch", None)
    if tm is not None:
        return tm
    try:
        import torch as tm  # type: ignore

        return tm
    except Exception:
        return None


def _sample_total_vram_usage_mb(torch_module: Any | None = None) -> tuple[float | None, int]:
    try:
        tm = torch_module
        if tm is None:
            import torch as tm  # type: ignore

        if not tm.cuda.is_available():
            return None, 0

        device_count = max(int(tm.cuda.device_count()), 1)
        used_samples_mb: list[float] = []
        for device_idx in range(device_count):
            try:
                free_bytes, total_bytes = tm.cuda.mem_get_info(device_idx)
                used_samples_mb.append(float(total_bytes - free_bytes) / (1024 * 1024))
                continue
            except Exception:
                pass

            try:
                used_samples_mb.append(float(tm.cuda.memory_reserved(device_idx)) / (1024 * 1024))
            except Exception:
                continue

        if not used_samples_mb:
            return None, device_count
        return float(sum(used_samples_mb)), device_count
    except Exception:
        return None, 0


def _summarize_vram_usage_mb(samples_mb: list[float], *, device_count: int) -> dict[str, Any]:
    if not samples_mb:
        return {
            "vram_used_mb_min": None,
            "vram_used_mb_max": None,
            "vram_used_mb_avg": None,
            "vram_sample_count": 0,
            "vram_device_count": device_count,
        }

    return {
        "vram_used_mb_min": round(min(samples_mb), 2),
        "vram_used_mb_max": round(max(samples_mb), 2),
        "vram_used_mb_avg": round(sum(samples_mb) / len(samples_mb), 2),
        "vram_sample_count": len(samples_mb),
        "vram_device_count": device_count,
    }


def _build_chat_prompt_text(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    apply_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_template):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


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
    params_b: float | None = None


@dataclass
class InferenceResult:
    output_json: str
    contains_pii: bool | None
    cost_usd: float
    prompt_tokens: int
    completion_tokens: int
    detection_latency_ms: float | None = None
    stream_mode: str | None = None
    early_stopped: bool | None = None
    fallback_used: bool | None = None
    fallback_reason: str | None = None
    processed_tokens: int | None = None
    total_tokens: int | None = None
    setup_tokens: int | None = None
    moderated_tokens: int | None = None


class ModelRunner(Protocol):
    model_id: str
    provider: str

    def predict(self, content: str) -> InferenceResult:
        ...

    def close(self) -> None:
        ...


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


def _configure_hf_xet_env(settings: dict[str, Any]) -> dict[str, str]:
    """
    依模型 settings.hf_xet 設定 Hugging Face Xet 下載加速環境變數。

    範例：
    settings:
      hf_xet:
        enabled: true
        high_performance: true
        num_concurrent_range_gets: 32
        reconstruct_write_sequentially: false
        override_env: false
    """
    raw_cfg = settings.get("hf_xet")
    if raw_cfg is None:
        return {}
    if not isinstance(raw_cfg, dict):
        raise ValueError("settings.hf_xet 必須為 object")

    enabled = _coerce_bool(raw_cfg.get("enabled", True))
    if enabled is None:
        raise ValueError("settings.hf_xet.enabled 必須為布林值（true/false）")
    if not enabled:
        return {}

    override_env = _coerce_bool(raw_cfg.get("override_env", False))
    if override_env is None:
        raise ValueError("settings.hf_xet.override_env 必須為布林值（true/false）")

    applied: dict[str, str] = {}

    def _set_env(name: str, value: str) -> None:
        existing = get_env(name)
        if existing not in (None, "") and not override_env:
            return
        os.environ[name] = value
        applied[name] = value

    high_perf = _coerce_bool(raw_cfg.get("high_performance", True))
    if high_perf is None:
        raise ValueError("settings.hf_xet.high_performance 必須為布林值（true/false）")
    if high_perf:
        _set_env("HF_XET_HIGH_PERFORMANCE", "1")

    int_key_mapping = {
        "num_concurrent_range_gets": "HF_XET_NUM_CONCURRENT_RANGE_GETS",
        "chunk_cache_size_bytes": "HF_XET_CHUNK_CACHE_SIZE_BYTES",
        "shard_cache_size_limit": "HF_XET_SHARD_CACHE_SIZE_LIMIT",
    }
    for cfg_key, env_key in int_key_mapping.items():
        if cfg_key not in raw_cfg:
            continue
        raw_value = raw_cfg.get(cfg_key)
        try:
            parsed = int(raw_value)
        except (TypeError, ValueError) as e:
            raise ValueError(f"settings.hf_xet.{cfg_key} 必須為整數") from e
        if parsed <= 0:
            raise ValueError(f"settings.hf_xet.{cfg_key} 必須 > 0")
        _set_env(env_key, str(parsed))

    if "reconstruct_write_sequentially" in raw_cfg:
        sequential = _coerce_bool(raw_cfg.get("reconstruct_write_sequentially"))
        if sequential is None:
            raise ValueError(
                "settings.hf_xet.reconstruct_write_sequentially 必須為布林值（true/false）"
            )
        _set_env("HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY", "1" if sequential else "0")

    return applied


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
# 指標
# =============================================================================
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


def compute_latency_breakdown_metrics(
    predictions: list[bool | None],
    ground_truths: list[bool],
    latencies_ms: list[float],
    detection_latencies_ms: list[float | None],
) -> dict[str, Any]:
    """
    額外延遲指標：
    - TTD: 僅在 TP（truth=True 且 pred=True）計算
    - Full-pass latency: 僅在 TN（truth=False 且 pred=False）計算
    """
    if not (
        len(predictions)
        == len(ground_truths)
        == len(latencies_ms)
        == len(detection_latencies_ms)
    ):
        raise ValueError("latency breakdown 輸入長度不一致")

    ttd_values: list[float] = []
    full_pass_values: list[float] = []

    for pred, truth, latency_ms, detection_latency_ms in zip(
        predictions,
        ground_truths,
        latencies_ms,
        detection_latencies_ms,
    ):
        resolved_pred = bool(pred) if pred is not None else False
        if truth and resolved_pred:
            ttd_values.append(
                float(detection_latency_ms)
                if detection_latency_ms is not None
                else float(latency_ms)
            )
        if (not truth) and (not resolved_pred):
            full_pass_values.append(float(latency_ms))

    ttd_series = pd.Series(ttd_values, dtype=float)
    full_pass_series = pd.Series(full_pass_values, dtype=float)

    return {
        "ttd_ms_avg": round(float(ttd_series.mean()), 2) if not ttd_series.empty else 0.0,
        "ttd_ms_p95": round(float(ttd_series.quantile(0.95)), 2) if not ttd_series.empty else 0.0,
        "ttd_count": int(len(ttd_values)),
        "full_pass_latency_ms_avg": round(float(full_pass_series.mean()), 2)
        if not full_pass_series.empty
        else 0.0,
        "full_pass_latency_ms_p95": round(float(full_pass_series.quantile(0.95)), 2)
        if not full_pass_series.empty
        else 0.0,
        "full_pass_count": int(len(full_pass_values)),
    }


def compute_token_efficiency_metrics(
    processed_tokens: list[int | None],
    total_tokens: list[int | None],
    setup_tokens: list[int | None],
    moderated_tokens: list[int | None],
) -> dict[str, Any]:
    if not (
        len(processed_tokens)
        == len(total_tokens)
        == len(setup_tokens)
        == len(moderated_tokens)
    ):
        raise ValueError("token efficiency 輸入長度不一致")

    processed_values = [int(v) if v is not None else 0 for v in processed_tokens]
    total_values = [int(v) if v is not None else 0 for v in total_tokens]
    setup_values = [int(v) if v is not None else 0 for v in setup_tokens]
    moderated_values = [int(v) if v is not None else 0 for v in moderated_tokens]

    content_total_tokens = sum(total_values)
    content_processed_tokens = sum(processed_values)
    setup_total_tokens = sum(setup_values)
    moderated_total_tokens = sum(moderated_values)

    content_token_reduction_rate = (
        1.0 - (content_processed_tokens / content_total_tokens)
        if content_total_tokens
        else 0.0
    )

    return {
        "content_tokens_total": content_total_tokens,
        "content_processed_tokens_total": content_processed_tokens,
        "setup_tokens_total": setup_total_tokens,
        "moderated_tokens_total": moderated_total_tokens,
        "content_token_reduction_rate": round(content_token_reduction_rate, 4),
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

        api_key = get_env("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("找不到 OPENROUTER_API_KEY，請確認 .env 或環境變數")

        self._client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
        self._async_client = AsyncOpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)

        headers: dict[str, str] = {}
        referer = get_env("OPENROUTER_HTTP_REFERER") or get_env("HTTP_REFERER")
        title = get_env("OPENROUTER_TITLE") or get_env("X_OPENROUTER_TITLE")
        if referer:
            headers["HTTP-Referer"] = referer
        if title:
            headers["X-OpenRouter-Title"] = title
        self._extra_headers = headers
        self._concurrency = max(
            1,
            int(self._settings.get("concurrency") or get_env("OPENROUTER_CONCURRENCY", "5")),
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
        prompts = get_task_definition(TASK_ID).prompt_builder.build(
            model_id=self.model_id,
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


class OpenAICompatibleRunner:
    def __init__(self, spec: ModelSpec) -> None:
        try:
            from openai import OpenAI, AsyncOpenAI
        except ImportError as e:  # pragma: no cover - runtime dependency
            raise RuntimeError("缺少 openai 套件，請先安裝 requirements.txt") from e

        self.model_id = spec.model_id
        self.provider = "openai_compatible"
        self._settings = dict(spec.settings)

        base_url = str(
            self._settings.get("base_url")
            or get_env("OPENAI_COMPAT_BASE_URL")
            or OPENAI_COMPAT_DEFAULT_BASE_URL
        ).strip()
        if not base_url:
            base_url = OPENAI_COMPAT_DEFAULT_BASE_URL

        api_key = self._resolve_api_key(self._settings)
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._async_client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self._concurrency = max(1, int(self._settings.get("concurrency") or 8))

    @staticmethod
    def _resolve_api_key(settings: dict[str, Any]) -> str:
        direct_key = settings.get("api_key")
        if direct_key is not None and str(direct_key).strip():
            return str(direct_key).strip()

        env_name = str(settings.get("api_key_env") or "OPENAI_COMPAT_API_KEY").strip()
        if env_name:
            env_key = get_env(env_name, "")
            if env_key.strip():
                return env_key.strip()

        fallback = get_env("OPENAI_COMPAT_API_KEY", "")
        if fallback.strip():
            return fallback.strip()
        return "EMPTY"

    @staticmethod
    def _extract_usage(completion: Any) -> tuple[int, int]:
        usage = getattr(completion, "usage", None)
        if usage is None:
            return 0, 0
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        return prompt_tokens, completion_tokens

    def _build_completion_kwargs(self, content: str) -> dict[str, Any]:
        prompts = get_task_definition(TASK_ID).prompt_builder.build(
            model_id=self.model_id,
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
        return kwargs

    def _completion_to_result(self, completion: Any) -> InferenceResult:
        raw_text = ""
        if getattr(completion, "choices", None):
            message = completion.choices[0].message
            raw_text = message.content if message and message.content else ""

        normalized = normalize_pii_binary_json(raw_text)
        prompt_tokens, completion_tokens = self._extract_usage(completion)
        return InferenceResult(
            output_json=json.dumps(normalized, ensure_ascii=False),
            contains_pii=normalized.get("contains_pii"),
            cost_usd=0.0,
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


class VLLMOfflineRunner:
    def __init__(self, spec: ModelSpec) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as e:  # pragma: no cover - runtime dependency
            raise RuntimeError("缺少 transformers 套件，請先安裝 requirements.txt") from e

        try:
            from vllm import LLM, SamplingParams
        except ImportError as e:  # pragma: no cover - runtime dependency
            raise RuntimeError("缺少 vllm 套件，請先安裝對應版本") from e

        self.model_id = spec.model_id
        self.provider = "vllm_offline"
        self.profile = spec.profile
        self._settings = dict(spec.settings)
        self._SamplingParams = SamplingParams

        trust_remote_code = bool(self._settings.get("trust_remote_code", True))
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=trust_remote_code,
        )
        if getattr(self._tokenizer, "pad_token", None) is None:
            eos_token = getattr(self._tokenizer, "eos_token", None)
            if eos_token is not None:
                self._tokenizer.pad_token = eos_token

        llm_kwargs: dict[str, Any] = {
            "model": self.model_id,
            "trust_remote_code": trust_remote_code,
        }
        dtype = self._settings.get("dtype", self._settings.get("torch_dtype"))
        if dtype is not None:
            dtype_str = str(dtype).strip()
            if dtype_str and dtype_str.lower() != "auto":
                llm_kwargs["dtype"] = dtype_str

        for key in (
            "tensor_parallel_size",
            "gpu_memory_utilization",
            "max_model_len",
            "enforce_eager",
            "download_dir",
            "quantization",
            "swap_space",
            "seed",
            "max_num_batched_tokens",
        ):
            value = self._settings.get(key)
            if value is not None:
                llm_kwargs[key] = value

        self._llm = LLM(**llm_kwargs)

    def _build_prompt_text(self, messages: list[dict[str, str]]) -> str:
        return _build_chat_prompt_text(self._tokenizer, messages)

    def _build_sampling_params(self) -> Any:
        kwargs: dict[str, Any] = {
            "temperature": float(self._settings.get("temperature", 0)),
            "max_tokens": int(
                self._settings.get("max_tokens", self._settings.get("max_new_tokens", 128))
            ),
        }
        for key in (
            "top_p",
            "top_k",
            "min_p",
            "presence_penalty",
            "frequency_penalty",
            "repetition_penalty",
        ):
            value = self._settings.get(key)
            if value is not None:
                kwargs[key] = float(value)

        seed = self._settings.get("seed")
        if seed is not None:
            kwargs["seed"] = int(seed)

        stop = self._settings.get("stop")
        if isinstance(stop, str) and stop.strip():
            kwargs["stop"] = [stop]
        elif isinstance(stop, list):
            normalized_stop = [str(item) for item in stop if str(item).strip()]
            if normalized_stop:
                kwargs["stop"] = normalized_stop

        return self._SamplingParams(**kwargs)

    @staticmethod
    def _input_id_length(tokenized: Any) -> int:
        input_ids = tokenized.get("input_ids") if isinstance(tokenized, dict) else getattr(tokenized, "input_ids", None)
        if input_ids is None:
            return 0
        if hasattr(input_ids, "shape"):
            shape = input_ids.shape
            if len(shape) == 2:
                return int(shape[-1])
            if len(shape) == 1:
                return int(shape[0])
        if isinstance(input_ids, list):
            if input_ids and isinstance(input_ids[0], list):
                return len(input_ids[0])
            return len(input_ids)
        try:
            return len(input_ids)
        except TypeError:
            return 0

    @staticmethod
    def _completion_token_length(output_item: Any) -> int:
        token_ids = getattr(output_item, "token_ids", None)
        if token_ids is None:
            return 0
        try:
            return int(len(token_ids))
        except TypeError:
            return 0

    def predict(self, content: str) -> InferenceResult:
        return self.predict_batch([content])[0]

    def predict_batch(self, contents: list[str]) -> list[InferenceResult]:
        if not contents:
            return []

        prompts: list[str] = []
        prompt_token_lengths: list[int] = []
        for content in contents:
            task_prompts = get_task_definition(TASK_ID).prompt_builder.build(
                model_id=self.model_id,
                content=content,
            )
            messages = [
                {"role": "system", "content": task_prompts["system"]},
                {"role": "user", "content": task_prompts["user"]},
            ]
            prompt_text = self._build_prompt_text(messages)
            prompts.append(prompt_text)
            prompt_token_lengths.append(self._input_id_length(self._tokenizer(prompt_text)))

        outputs = self._llm.generate(prompts, self._build_sampling_params())
        if len(outputs) != len(contents):
            raise RuntimeError(
                "vllm generate 回傳筆數與輸入不一致: "
                f"{len(outputs)} != {len(contents)}"
            )
        results: list[InferenceResult] = []
        for idx, output in enumerate(outputs):
            output_items = getattr(output, "outputs", None) or []
            output_item = output_items[0] if output_items else None
            raw_text = str(getattr(output_item, "text", "") or "")
            normalized = normalize_pii_binary_json(raw_text)

            prompt_tokens = prompt_token_lengths[idx]
            prompt_token_ids = getattr(output, "prompt_token_ids", None)
            if prompt_token_ids is not None:
                try:
                    prompt_tokens = int(len(prompt_token_ids))
                except TypeError:
                    pass

            completion_tokens = self._completion_token_length(output_item) if output_item is not None else 0
            results.append(
                InferenceResult(
                    output_json=json.dumps(normalized, ensure_ascii=False),
                    contains_pii=normalized.get("contains_pii"),
                    cost_usd=0.0,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
            )
        return results

    def close(self) -> None:
        for attr in ("_llm", "_tokenizer"):
            if hasattr(self, attr):
                obj = getattr(self, attr)
                setattr(self, attr, None)
                try:
                    del obj
                except Exception:
                    pass
        _release_cuda_memory()
        return None


def _resolve_hf_runtime_settings(
    settings: dict[str, Any],
    *,
    cuda_available: bool,
    bf16_available: bool = False,
) -> dict[str, Any]:
    trust_remote_code = bool(settings.get("trust_remote_code", True))
    allow_bf16_fallback = _coerce_bool(settings.get("allow_bf16_fallback", True))
    if allow_bf16_fallback is None:
        allow_bf16_fallback = True

    dtype_cfg = str(settings.get("torch_dtype", "auto")).strip().lower()
    if dtype_cfg in {"bf16"}:
        dtype_cfg = "bfloat16"
    elif dtype_cfg in {"fp16", "half"}:
        dtype_cfg = "float16"
    elif dtype_cfg in {"fp32"}:
        dtype_cfg = "float32"

    if dtype_cfg == "auto":
        if cuda_available:
            # Colab/T4 常不支援原生 bf16，auto 應優先選 float16 避免額外顯存壓力。
            torch_dtype_name = "bfloat16" if bf16_available else "float16"
        else:
            torch_dtype_name = "float32"
    elif dtype_cfg in {"bfloat16", "float16", "float32"}:
        if dtype_cfg == "bfloat16" and cuda_available and not bf16_available:
            if allow_bf16_fallback:
                torch_dtype_name = "float16"
            else:
                raise ValueError(
                    "目前 GPU 不支援 bfloat16（需要 sm_80+）；"
                    "請改用 torch_dtype=float16 或設定 allow_bf16_fallback=true"
                )
        else:
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
        self._hf_xet_env_overrides = _configure_hf_xet_env(self._settings)

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
        bf16_supported = False
        if torch.cuda.is_available():
            try:
                capability = torch.cuda.get_device_capability()
                sm80_or_newer = isinstance(capability, tuple) and len(capability) >= 1 and capability[0] >= 8
            except Exception:
                sm80_or_newer = False
            try:
                torch_bf16 = bool(torch.cuda.is_bf16_supported())
            except Exception:
                torch_bf16 = False
            bf16_supported = bool(sm80_or_newer and torch_bf16)

        runtime = _resolve_hf_runtime_settings(
            self._settings,
            cuda_available=bool(torch.cuda.is_available()),
            bf16_available=bf16_supported,
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
        attn_impl = self._settings.get("attn_implementation")
        if attn_impl is not None and str(attn_impl).strip():
            kwargs["attn_implementation"] = str(attn_impl).strip()
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
        return _build_chat_prompt_text(self._tokenizer, messages)

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

    def _tokenize_prompt_texts(
        self,
        prompt_texts: list[str],
    ) -> tuple[dict[str, Any], list[int]]:
        max_input_tokens = self._settings.get("max_input_tokens")
        token_kwargs: dict[str, Any] = {
            "return_tensors": "pt",
            "padding": True,
        }
        if max_input_tokens is not None:
            token_kwargs["truncation"] = True
            token_kwargs["max_length"] = int(max_input_tokens)

        model_inputs = self._tokenizer(prompt_texts, **token_kwargs)
        if self._input_device is not None:
            model_inputs = {
                k: (v.to(self._input_device) if hasattr(v, "to") else v)
                for k, v in model_inputs.items()
            }

        input_lengths: list[int] = []
        attention_mask = model_inputs.get("attention_mask")
        if attention_mask is not None and hasattr(attention_mask, "sum"):
            if hasattr(attention_mask, "dim") and attention_mask.dim() == 2:
                input_lengths = [int(v) for v in attention_mask.sum(dim=1).tolist()]
            else:
                total_tensor = attention_mask.sum()
                total = int(total_tensor.item()) if hasattr(total_tensor, "item") else int(total_tensor)
                input_lengths = [total]
        elif "input_ids" in model_inputs and hasattr(model_inputs["input_ids"], "shape"):
            shape = model_inputs["input_ids"].shape
            if len(shape) == 2:
                input_lengths = [int(shape[-1])] * int(shape[0])
            elif len(shape) == 1:
                input_lengths = [int(shape[0])]
        return model_inputs, input_lengths

    def _build_inference_result_from_generated(
        self,
        output_row: Any,
        *,
        input_len: int,
        max_input_len: int,
    ) -> InferenceResult:
        if max_input_len and len(output_row) >= max_input_len:
            generated_part = output_row[max_input_len:]
        elif input_len and len(output_row) >= input_len:
            generated_part = output_row[input_len:]
        else:
            generated_part = output_row

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

    def predict(self, content: str) -> InferenceResult:
        return self.predict_batch([content])[0]

    def predict_batch(self, contents: list[str]) -> list[InferenceResult]:
        if not contents:
            return []

        prompt_texts: list[str] = []
        for content in contents:
            prompts = get_task_definition(TASK_ID).prompt_builder.build(
                model_id=self.model_id,
                content=content,
            )
            messages = [
                {"role": "system", "content": prompts["system"]},
                {"role": "user", "content": prompts["user"]},
            ]
            prompt_texts.append(self._build_prompt_text(messages))

        model_inputs, input_lengths = self._tokenize_prompt_texts(prompt_texts)
        with self._torch.no_grad():
            generated_ids = self._model.generate(
                **model_inputs,
                **self._build_generation_kwargs(),
            )

        max_input_len = int(model_inputs["input_ids"].shape[-1]) if "input_ids" in model_inputs else 0
        results: list[InferenceResult] = []
        for idx in range(len(prompt_texts)):
            output_row = generated_ids[idx]
            input_len = input_lengths[idx] if idx < len(input_lengths) else max_input_len
            results.append(
                self._build_inference_result_from_generated(
                    output_row,
                    input_len=input_len,
                    max_input_len=max_input_len,
                )
            )
        return results


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
        self._log_stream_api_diagnostics()

    def _log_stream_api_diagnostics(self) -> None:
        stream_fn = getattr(self._model, "stream_moderate_from_ids", None)
        stream_generate_fn = getattr(self._model, "stream_generate", None)
        close_stream_fn = getattr(self._model, "close_stream", None)
        event: dict[str, Any] = {
            "event": "model_stream_api_info",
            "model_id": self.model_id,
            "model_class": self._model.__class__.__name__,
            "model_module": self._model.__class__.__module__,
            "has_stream_moderate_from_ids": callable(stream_fn),
            "has_stream_generate": callable(stream_generate_fn),
            "has_close_stream": callable(close_stream_fn),
        }

        if callable(stream_fn):
            try:
                event["stream_moderate_signature"] = str(inspect.signature(stream_fn))
            except Exception as e:
                event["stream_moderate_signature_error"] = str(e)
        if callable(stream_generate_fn):
            try:
                event["stream_generate_signature"] = str(inspect.signature(stream_generate_fn))
            except Exception as e:
                event["stream_generate_signature_error"] = str(e)

        _append_qwen_debug(event)

    @staticmethod
    def _last_item(value: Any) -> str:
        if isinstance(value, list):
            return str(value[-1]) if value else ""
        return str(value) if value is not None else ""

    @staticmethod
    def _last_float(value: Any) -> float | None:
        if isinstance(value, list):
            if not value:
                return None
            value = value[-1]
        if value is None or isinstance(value, bool):
            return None
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return min(1.0, max(0.0, parsed))

    @staticmethod
    def _token_to_int(token: Any) -> int:
        if isinstance(token, int):
            return token
        item_fn = getattr(token, "item", None)
        if callable(item_fn):
            return int(item_fn())
        return int(token)

    def _stream_moderate_single_token(
        self,
        token_ids: Any,
        token_index: int,
        stream_state: Any,
    ) -> tuple[dict[str, Any], Any]:
        """
        透過 stream_generate 逐 token 推進，回傳「最新 token」的分類。

        為避開部分 remote-code 版本在 stream_moderate_from_ids(seq_len=1) 的形狀問題，
        這裡改走底層 generator 並自行解碼 logits。
        """
        if not callable(getattr(self._model, "stream_generate", None)):
            raise RuntimeError("stream_generate is unavailable on current model")

        if token_index < 0:
            raise ValueError(f"token_index 必須 >= 0，收到 {token_index}")

        token_count = int(token_ids.shape[0]) if hasattr(token_ids, "shape") else len(token_ids)
        if token_index >= token_count:
            raise ValueError(f"token_index 超出範圍: {token_index} >= {token_count}")

        token_0d = token_ids[token_index]
        token_ctx: dict[str, Any] = {
            "token_index": token_index,
            "stream_state_type": type(stream_state).__name__ if stream_state is not None else None,
            "token_type": type(token_0d).__name__,
        }
        if hasattr(token_0d, "dtype"):
            token_ctx["token_dtype"] = str(token_0d.dtype)
        if hasattr(token_0d, "device"):
            token_ctx["token_device"] = str(token_0d.device)

        if stream_state is None:
            first_token = token_ids[token_index : token_index + 1]
            if hasattr(first_token, "to"):
                first_token = first_token.to(self._model.device)
            stream_state = self._model.stream_generate(first_token)
            logits_tuple = next(stream_state)
        else:
            next_token_id = self._token_to_int(token_0d)
            logits_tuple = stream_state.send(next_token_id)

        try:
            # role=user 取 query heads（與原始 stream_moderate_from_ids 一致）
            risk_level_logits = logits_tuple[2]
            category_logits = logits_tuple[3]
            risk_map = getattr(self._model, "query_risk_level_map", {})
            category_map = getattr(self._model, "query_category_map", {})

            risk_label, risk_prob = self._decode_last_token_prediction(risk_level_logits, risk_map)
            category_label, category_prob = self._decode_last_token_prediction(category_logits, category_map)

            result = {
                "risk_level": [risk_label],
                "risk_prob": [risk_prob],
                "category": [category_label],
                "category_prob": [category_prob],
            }
            return result, stream_state
        except Exception as e:
            _append_qwen_debug(
                {
                    "event": "token_logits_decode_failed",
                    "model_id": self.model_id,
                    "error": str(e),
                    **token_ctx,
                }
            )
            raise

    def _decode_last_token_prediction(
        self,
        logits: Any,
        label_map: dict[int, str] | dict[str, str] | None = None,
    ) -> tuple[str, float]:
        """將 logits 轉為最後一個 token 的 (label, probability)。"""
        if logits is None:
            return "", 0.0

        if not hasattr(logits, "dim"):
            raise ValueError(f"logits 不是 tensor: {type(logits).__name__}")

        tensor = logits
        if tensor.dim() >= 3:
            # [batch, seq, classes] -> 取最後 token
            vector = tensor[0, -1, :]
        elif tensor.dim() == 2:
            vector = tensor[-1, :]
        elif tensor.dim() == 1:
            vector = tensor
        else:
            raise ValueError(f"不支援的 logits 維度: {tensor.dim()}")

        probs = self._torch.nn.functional.softmax(vector, dim=-1)
        pred_prob, pred_idx = self._torch.max(probs, dim=-1)
        idx_prob = self._torch.stack(
            (
                pred_idx.to(dtype=self._torch.float32),
                pred_prob.to(dtype=self._torch.float32),
            )
        ).detach().cpu()
        idx = int(idx_prob[0].item())
        confidence = round(float(idx_prob[1].item()), 2)

        label = None
        if isinstance(label_map, dict):
            label = label_map.get(idx)
            if label is None:
                label = label_map.get(str(idx))
        if label is None:
            label = str(idx)
        return str(label), confidence

    def _tokenize_user_only(self, content: str) -> Any:
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
        return model_inputs.input_ids[0]

    def _prepare_assistant_stream_tokens(self, assistant_content: str) -> tuple[Any, Any]:
        seed_user_message = str(
            self._settings.get("assistant_stream_user_message", "Please continue.")
        )
        messages = [
            {"role": "user", "content": seed_user_message},
            {"role": "assistant", "content": assistant_content},
        ]
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
        token_ids_list = token_ids.tolist()

        im_start_id = self._tokenizer.convert_tokens_to_ids("<|im_start|>")
        user_id = self._tokenizer.convert_tokens_to_ids("user")
        im_end_id = self._tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_start_id is None or user_id is None or im_end_id is None:
            raise RuntimeError("無法取得 chat template 關鍵 token id")

        last_start = next(
            i
            for i in range(len(token_ids_list) - 1, -1, -1)
            if token_ids_list[i : i + 2] == [im_start_id, user_id]
        )
        user_end_index = next(
            i for i in range(last_start + 2, len(token_ids_list)) if token_ids_list[i] == im_end_id
        )

        user_turn_ids = token_ids[: user_end_index + 1]
        assistant_turn_ids = token_ids[user_end_index + 1 :]
        assistant_turn_ids = self._trim_assistant_prefix_tokens(assistant_turn_ids)
        if int(assistant_turn_ids.shape[0]) <= 0:
            raise RuntimeError("assistant_stream_simulation 找不到 assistant token 序列")
        return user_turn_ids, assistant_turn_ids

    def _trim_assistant_prefix_tokens(self, assistant_turn_ids: Any) -> Any:
        """
        去除 assistant chat template 前綴，避免把 <|im_start|>assistant、<think> 等控制 token
        當作正文內容做 stream moderation，造成固定位置誤觸發。
        """
        if not hasattr(assistant_turn_ids, "tolist"):
            return assistant_turn_ids

        token_ids = assistant_turn_ids.tolist()
        if not token_ids:
            return assistant_turn_ids

        think_end_id = self._tokenizer.convert_tokens_to_ids("</think>")
        start_index = 0
        if think_end_id is not None and think_end_id in token_ids:
            start_index = token_ids.index(think_end_id) + 1

        while start_index < len(token_ids):
            token_id = token_ids[start_index]
            try:
                token_text = self._tokenizer.decode([token_id], skip_special_tokens=False)
            except Exception:
                break
            if token_text.strip():
                break
            start_index += 1

        if start_index <= 0:
            return assistant_turn_ids
        return assistant_turn_ids[start_index:]

    def predict(self, content: str) -> InferenceResult:
        predict_started = time.perf_counter()
        user_token_ids = self._tokenize_user_only(content)
        assistant_stream_simulation = _coerce_bool(
            self._settings.get("assistant_stream_simulation", True)
        )
        if assistant_stream_simulation is None:
            raise ValueError("settings.assistant_stream_simulation 必須為布林值（true/false）")
        stream_role_mode = (
            str(self._settings.get("stream_role_mode", "")).strip().lower()
            if self._settings.get("stream_role_mode") is not None
            else ""
        )
        use_assistant_stream = bool(assistant_stream_simulation) or (
            stream_role_mode in {"assistant", "assistant_stream", "assistant_realtime"}
        )
        token_ids = user_token_ids
        assistant_token_ids = None
        if use_assistant_stream:
            try:
                user_turn_ids, assistant_turn_ids = self._prepare_assistant_stream_tokens(content)
                token_ids = user_turn_ids
                assistant_token_ids = assistant_turn_ids
            except Exception as e:
                _append_qwen_debug(
                    {
                        "event": "assistant_stream_prepare_failed",
                        "model_id": self.model_id,
                        "reason": str(e),
                    }
                )
                use_assistant_stream = False

        stream_state = None
        result: dict[str, Any] = {}
        processed_tokens = 0
        setup_tokens = 0
        moderated_tokens = 0
        early_stopped = False
        trigger_idx: int | None = None
        inferred_pii = False
        inferred_trigger_idx: int | None = None
        detection_latency_ms: float | None = None
        fallback_reason: str | None = None
        stream_mode = "token_by_token"
        benchmark_mode = str(self._settings.get("benchmark_mode") or "").strip().lower()
        should_early_stop = benchmark_mode != "full_pass"
        saw_pii_category = False
        first_hit_category: str | None = None
        token_count = int(token_ids.shape[0]) if len(token_ids.shape) > 0 else 0
        if assistant_token_ids is not None:
            token_count = int(assistant_token_ids.shape[0]) if len(assistant_token_ids.shape) > 0 else 0
        _append_qwen_debug(
            {
                "event": "start_predict",
                "model_id": self.model_id,
                "token_count": token_count,
                "use_assistant_stream": use_assistant_stream,
                "content_preview": content[:120],
            }
        )
        try:
            if use_assistant_stream and assistant_token_ids is not None:
                # 官方 realtime 路徑：user turn 先 full pass，assistant 再逐 token 串流。
                setup_tokens = int(token_ids.shape[0]) if len(token_ids.shape) > 0 else 0
                user_started = time.perf_counter()
                result, stream_state = self._model.stream_moderate_from_ids(
                    token_ids,
                    role="user",
                    stream_state=None,
                )
                stream_mode = "assistant_token_stream_api"
                _append_qwen_debug(
                    {
                        "event": "assistant_stream_user_context_ready",
                        "model_id": self.model_id,
                        "user_tokens": int(token_ids.shape[0]),
                        "assistant_tokens": token_count,
                        "latency_ms": round((time.perf_counter() - user_started) * 1000, 2),
                    }
                )
                for i in range(token_count):
                    token_0d = assistant_token_ids[i]
                    try:
                        result, stream_state = self._model.stream_moderate_from_ids(
                            token_0d,
                            role="assistant",
                            stream_state=stream_state,
                        )
                    except Exception:
                        token_scalar = self._token_to_int(token_0d)
                        result, stream_state = self._model.stream_moderate_from_ids(
                            token_scalar,
                            role="assistant",
                            stream_state=stream_state,
                        )

                    processed_tokens = i + 1
                    moderated_tokens = setup_tokens + processed_tokens
                    step_category = self._last_item(result.get("category"))
                    if _contains_pii_category(step_category):
                        saw_pii_category = True
                        if first_hit_category is None:
                            first_hit_category = step_category
                            trigger_idx = i
                            detection_latency_ms = (time.perf_counter() - predict_started) * 1000
                        if should_early_stop:
                            early_stopped = True
                            _append_qwen_debug(
                                {
                                    "event": "assistant_stream_early_stop_triggered",
                                    "model_id": self.model_id,
                                    "token_index": i,
                                    "category": step_category,
                                    "detection_latency_ms": round(detection_latency_ms, 2),
                                }
                            )
                            break
            else:
                # 非 assistant 模式：優先走官方建議 user full pass。
                official_started = time.perf_counter()
                try:
                    result, stream_state = self._model.stream_moderate_from_ids(
                        token_ids,
                        role="user",
                        stream_state=None,
                    )
                    stream_mode = "user_full_pass_stream_api"
                    processed_tokens = token_count
                    moderated_tokens = token_count
                    _append_qwen_debug(
                        {
                            "event": "official_user_full_pass_succeeded",
                            "model_id": self.model_id,
                            "token_count": token_count,
                            "latency_ms": round((time.perf_counter() - official_started) * 1000, 2),
                        }
                    )
                except Exception as official_error:
                    _append_qwen_debug(
                        {
                            "event": "official_user_full_pass_failed",
                            "model_id": self.model_id,
                            "token_count": token_count,
                            "reason": str(official_error),
                        }
                    )
                    stream_mode = "token_by_token"

                # 官方路徑失敗時，退回逐 token moderation：命中 PII category 立即 early stop。
                for i in range(token_count if stream_mode == "token_by_token" else 0):
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
                            fallback_reason = str(e)
                            _append_qwen_debug(
                                {
                                    "event": "fallback_to_full_sequence",
                                    "model_id": self.model_id,
                                    "reason": fallback_reason,
                                    "token_count": token_count,
                                }
                            )
                            fallback_started = time.perf_counter()
                            try:
                                # 退回到整段 user moderation（完整 token ids）以保證可用性。
                                result, stream_state = self._model.stream_moderate_from_ids(
                                    token_ids,
                                    role="user",
                                    stream_state=stream_state,
                                )
                            except Exception as fallback_error:
                                _append_qwen_debug(
                                    {
                                        "event": "fallback_full_sequence_failed",
                                        "model_id": self.model_id,
                                        "reason": str(fallback_error),
                                        "token_count": token_count,
                                    }
                                )
                                raise RuntimeError(
                                    "fallback_full_sequence_failed: "
                                    f"{fallback_error}"
                                ) from fallback_error
                            _append_qwen_debug(
                                {
                                    "event": "fallback_full_sequence_succeeded",
                                    "model_id": self.model_id,
                                    "fallback_latency_ms": round(
                                        (time.perf_counter() - fallback_started) * 1000, 2
                                    ),
                                    "token_count": token_count,
                                }
                            )
                            processed_tokens = token_count
                            moderated_tokens = token_count
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
                    moderated_tokens = processed_tokens
                    step_category = self._last_item(result.get("category"))
                    if _contains_pii_category(step_category):
                        saw_pii_category = True
                        if first_hit_category is None:
                            first_hit_category = step_category
                            trigger_idx = i
                            detection_latency_ms = (time.perf_counter() - predict_started) * 1000
                        if should_early_stop:
                            early_stopped = True
                            _append_qwen_debug(
                                {
                                    "event": "early_stop_triggered_by_pii_category",
                                    "model_id": self.model_id,
                                    "token_index": i,
                                    "category": step_category,
                                    "detection_latency_ms": round(detection_latency_ms, 2),
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
                    if detection_latency_ms is None:
                        detection_latency_ms = (time.perf_counter() - predict_started) * 1000
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
            if moderated_tokens == 0:
                moderated_tokens = processed_tokens + setup_tokens
        finally:
            if stream_state is not None:
                try:
                    self._model.close_stream(stream_state)
                except Exception:
                    pass

        if not use_assistant_stream:
            detection_latency_ms = None

        risk_level = self._last_item(result.get("risk_level"))
        risk_prob = self._last_float(result.get("risk_prob"))
        category = self._last_item(result.get("category"))
        category_prob = self._last_float(result.get("category_prob"))
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
        elif saw_pii_category:
            contains_pii = True
        elif inferred_trigger_idx is not None:
            # 對 stream 輸出，優先採用序列級 category 判定，避免只看最後 token。
            contains_pii = True
        elif category:
            contains_pii = _contains_pii_category(category)

        reason_parts = []
        if risk_level:
            reason_parts.append(f"risk_level={risk_level}")
        if category:
            reason_parts.append(f"category={category}")
        if saw_pii_category and first_hit_category:
            reason_parts.append(f"stream_first_hit_category={first_hit_category}")
        if trigger_idx is not None:
            reason_parts.append(f"stream_first_hit_index={trigger_idx}")
        if parsed_refusal:
            reason_parts.append(f"refusal={parsed_refusal}")

        confidence = category_prob
        if confidence is None:
            confidence = risk_prob

        raw_text = json.dumps(
            {
                "risk_level": risk_level,
                "risk_prob": risk_prob,
                "category": category,
                "category_prob": category_prob,
                "refusal": parsed_refusal,
                "text_output": text_output,
                "stream_mode": stream_mode,
                "early_stopped": early_stopped,
                "trigger_index": trigger_idx,
                "stream_first_hit_category": first_hit_category,
                "stream_detected_pii": saw_pii_category,
                "inferred_pii": inferred_pii,
                "inferred_trigger_index": inferred_trigger_idx,
                "setup_tokens": setup_tokens,
                "processed_tokens": processed_tokens,
                "total_tokens": token_count,
                "moderated_tokens": moderated_tokens,
                "detection_latency_ms": round(detection_latency_ms, 2)
                if detection_latency_ms is not None
                else None,
                "fallback_reason": fallback_reason,
            },
            ensure_ascii=False,
        )

        normalized = {
            "contains_pii": contains_pii,
            "label": "是" if contains_pii is True else "否" if contains_pii is False else "無法解析",
            "confidence": confidence,
            "reason": "; ".join(reason_parts),
            "raw_text": raw_text,
        }

        return InferenceResult(
            output_json=json.dumps(normalized, ensure_ascii=False),
            contains_pii=contains_pii,
            cost_usd=0.0,
            prompt_tokens=moderated_tokens,
            completion_tokens=0,
            detection_latency_ms=detection_latency_ms,
            stream_mode=stream_mode,
            early_stopped=early_stopped,
            fallback_used=(stream_mode == "full_sequence_fallback"),
            fallback_reason=fallback_reason,
            processed_tokens=processed_tokens,
            total_tokens=token_count,
            setup_tokens=setup_tokens,
            moderated_tokens=moderated_tokens,
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


def _create_openai_compatible_runner(spec: ModelSpec) -> ModelRunner:
    return OpenAICompatibleRunner(spec)


def _create_vllm_offline_runner(spec: ModelSpec) -> ModelRunner:
    return VLLMOfflineRunner(spec)


def _create_huggingface_runner(spec: ModelSpec) -> ModelRunner:
    profile = (spec.profile or "granite_guard_json").strip().lower()
    if profile in {
        "granite_guard_json",
        "granite",
        "granite_hf",
        "causal_lm_json",
        "qwen_instruct_json",
        "qwen2_5_instruct_json",
    }:
        return GraniteHuggingFaceRunner(spec)
    if profile in {"qwen_stream", "qwen_guard_stream"}:
        return QwenGuardStreamRunner(spec)
    raise ValueError(f"未知 huggingface profile: {spec.profile!r}")


RUNNER_FACTORIES: dict[str, Callable[[ModelSpec], ModelRunner]] = {
    "openrouter": _create_openrouter_runner,
    "openai_compatible": _create_openai_compatible_runner,
    "vllm_offline": _create_vllm_offline_runner,
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
    progress_desc: str,
    show_progress: bool,
) -> tuple[list[InferenceResult], list[float]]:
    semaphore = asyncio.Semaphore(max(1, int(concurrency)))
    inference_results: list[InferenceResult] = [_build_empty_content_inference() for _ in rows]
    row_latencies_ms: list[float] = [0.0 for _ in rows]
    progress = tqdm(
        total=len(rows),
        desc=progress_desc,
        unit="row",
        disable=not show_progress,
        leave=False,
    )

    async def _run_one(slot_idx: int, content: str) -> None:
        async with semaphore:
            row_start = time.perf_counter()
            try:
                inference = await runner.predict_async(content)
            except Exception as e:
                inference = _build_model_error_inference(e)
            row_latencies_ms[slot_idx] = (time.perf_counter() - row_start) * 1000
            inference_results[slot_idx] = inference
            progress.update(1)

    tasks = [_run_one(slot_idx, content) for slot_idx, (_, content) in enumerate(rows)]
    try:
        if tasks:
            await asyncio.gather(*tasks)
    finally:
        try:
            progress.close()
        except Exception:
            pass
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
    show_progress: bool = True,
    task_id: str = TASK_ID,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    對單一模型執行 benchmark。

    Returns:
        (result_df, metrics_meta)
    """
    global TASK_ID
    TASK_ID = task_id
    runner = create_runner(spec)
    vram_torch_module = _resolve_vram_torch_module(runner)
    vram_samples_mb: list[float] = []
    vram_device_count = 0

    def capture_vram_sample() -> None:
        nonlocal vram_device_count
        used_mb, device_count = _sample_total_vram_usage_mb(vram_torch_module)
        if device_count:
            vram_device_count = max(vram_device_count, device_count)
        if used_mb is not None:
            vram_samples_mb.append(used_mb)

    outputs: list[str] = []
    predictions: list[bool | None] = []
    latencies_ms: list[float] = []
    costs_usd: list[float] = []
    detection_latencies_ms: list[float | None] = []
    stream_modes: list[str | None] = []
    early_stopped_flags: list[bool | None] = []
    fallback_used_flags: list[bool | None] = []
    fallback_reasons: list[str | None] = []
    processed_tokens_list: list[int | None] = []
    total_tokens_list: list[int | None] = []
    setup_tokens_list: list[int | None] = []
    moderated_tokens_list: list[int | None] = []
    prompt_tokens_total = 0
    completion_tokens_total = 0

    started = time.perf_counter()
    capture_vram_sample()
    try:
        if callable(getattr(runner, "predict_async", None)):
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
                        progress_desc=f"{spec.key} rows",
                        show_progress=show_progress,
                    )
                )
                for local_idx, slot_idx in enumerate(valid_slots):
                    inference_results[slot_idx] = valid_inferences[local_idx]
                    row_latencies_ms[slot_idx] = valid_latencies[local_idx]
                capture_vram_sample()

            for slot_idx in range(len(prepared_rows)):
                inference = inference_results[slot_idx]
                latencies_ms.append(row_latencies_ms[slot_idx])
                costs_usd.append(inference.cost_usd)
                outputs.append(inference.output_json)
                predictions.append(inference.contains_pii)
                detection_latencies_ms.append(inference.detection_latency_ms)
                stream_modes.append(inference.stream_mode)
                early_stopped_flags.append(inference.early_stopped)
                fallback_used_flags.append(inference.fallback_used)
                fallback_reasons.append(inference.fallback_reason)
                processed_tokens_list.append(inference.processed_tokens)
                total_tokens_list.append(inference.total_tokens)
                setup_tokens_list.append(inference.setup_tokens)
                moderated_tokens_list.append(inference.moderated_tokens)
                prompt_tokens_total += inference.prompt_tokens
                completion_tokens_total += inference.completion_tokens
        else:
            batch_predict = getattr(runner, "predict_batch", None)
            if callable(batch_predict):
                prepared_rows = []
                if content_column in df.columns:
                    raw_contents = df[content_column].tolist()
                else:
                    raw_contents = ["" for _ in range(len(df))]
                for idx, raw_content in enumerate(raw_contents):
                    content = "" if pd.isna(raw_content) else str(raw_content).strip()
                    prepared_rows.append((idx, content))

                inference_results = [_build_empty_content_inference() for _ in prepared_rows]
                row_latencies_ms: list[float] = [0.0 for _ in prepared_rows]
                valid_slots: list[int] = []
                batch_size = 8
                progress = tqdm(
                    total=len(prepared_rows),
                    desc=f"{spec.key} rows",
                    unit="row",
                    disable=not show_progress,
                    leave=False,
                )

                try:
                    for slot_idx, (_, content) in enumerate(prepared_rows):
                        if content:
                            valid_slots.append(slot_idx)
                        else:
                            progress.update(1)

                    for start in range(0, len(valid_slots), batch_size):
                        chunk_slots = valid_slots[start : start + batch_size]
                        chunk_contents = [prepared_rows[slot_idx][1] for slot_idx in chunk_slots]
                        batch_started = time.perf_counter()
                        try:
                            batch_inferences = batch_predict(chunk_contents)
                            if len(batch_inferences) != len(chunk_slots):
                                raise RuntimeError(
                                    "predict_batch 回傳筆數與輸入不一致: "
                                    f"{len(batch_inferences)} != {len(chunk_slots)}"
                                )
                            batch_latency_ms = (time.perf_counter() - batch_started) * 1000
                            for local_idx, slot_idx in enumerate(chunk_slots):
                                inference_results[slot_idx] = batch_inferences[local_idx]
                                row_latencies_ms[slot_idx] = batch_latency_ms
                            capture_vram_sample()
                        except Exception:
                            for slot_idx in chunk_slots:
                                content = prepared_rows[slot_idx][1]
                                row_started = time.perf_counter()
                                try:
                                    inference_results[slot_idx] = runner.predict(content)
                                except Exception as e:
                                    inference_results[slot_idx] = _build_model_error_inference(e)
                                row_latencies_ms[slot_idx] = (time.perf_counter() - row_started) * 1000
                                capture_vram_sample()
                        progress.update(len(chunk_slots))
                finally:
                    try:
                        progress.close()
                    except Exception:
                        pass

                for slot_idx in range(len(prepared_rows)):
                    inference = inference_results[slot_idx]
                    latencies_ms.append(row_latencies_ms[slot_idx])
                    costs_usd.append(inference.cost_usd)
                    outputs.append(inference.output_json)
                    predictions.append(inference.contains_pii)
                    detection_latencies_ms.append(inference.detection_latency_ms)
                    stream_modes.append(inference.stream_mode)
                    early_stopped_flags.append(inference.early_stopped)
                    fallback_used_flags.append(inference.fallback_used)
                    fallback_reasons.append(inference.fallback_reason)
                    processed_tokens_list.append(inference.processed_tokens)
                    total_tokens_list.append(inference.total_tokens)
                    setup_tokens_list.append(inference.setup_tokens)
                    moderated_tokens_list.append(inference.moderated_tokens)
                    prompt_tokens_total += inference.prompt_tokens
                    completion_tokens_total += inference.completion_tokens
            else:
                row_iterator = tqdm(
                    df.iterrows(),
                    total=len(df),
                    desc=f"{spec.key} rows",
                    unit="row",
                    disable=not show_progress,
                    leave=False,
                )
                for _, row in row_iterator:
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
                        capture_vram_sample()

                    costs_usd.append(inference.cost_usd)
                    outputs.append(inference.output_json)
                    predictions.append(inference.contains_pii)
                    detection_latencies_ms.append(inference.detection_latency_ms)
                    stream_modes.append(inference.stream_mode)
                    early_stopped_flags.append(inference.early_stopped)
                    fallback_used_flags.append(inference.fallback_used)
                    fallback_reasons.append(inference.fallback_reason)
                    processed_tokens_list.append(inference.processed_tokens)
                    total_tokens_list.append(inference.total_tokens)
                    setup_tokens_list.append(inference.setup_tokens)
                    moderated_tokens_list.append(inference.moderated_tokens)
                    prompt_tokens_total += inference.prompt_tokens
                    completion_tokens_total += inference.completion_tokens
    finally:
        capture_vram_sample()
        runner.close()
        _release_cuda_memory()

    elapsed = time.perf_counter() - started

    task = get_task_definition(task_id)
    parsed_outputs = [task.output_parser.parse(output) for output in outputs]
    result_df = pd.DataFrame(
        {
            "content": df[content_column].tolist() if content_column in df.columns else [""] * len(df),
            "ground_truth": [bool(value) for value in df[ground_truth_column].tolist()],
            "sample_type": df.get("sample_type", "unknown"),
            "raw_output": outputs,
            "parsed_prediction": [item.get("contains_pii") for item in parsed_outputs],
            "parsed_label": [
                item.get("label") if item.get("parse_status") == "parsed" else "無法解析"
                for item in parsed_outputs
            ],
            "parsed_confidence": [item.get("confidence") for item in parsed_outputs],
            "parse_status": [item.get("parse_status") for item in parsed_outputs],
            "parse_error": [item.get("error") for item in parsed_outputs],
            "parsed_reason": [item.get("reason") for item in parsed_outputs],
        }
    )

    result_df["latency_ms"] = [round(v, 2) for v in latencies_ms]
    result_df["cost_usd"] = [round(v, 8) for v in costs_usd]
    result_df["model_key"] = spec.key
    result_df["model_id"] = spec.model_id
    result_df["provider"] = spec.provider
    result_df["detection_latency_ms"] = [
        round(float(v), 2) if v is not None else np.nan for v in detection_latencies_ms
    ]
    result_df["stream_mode"] = [v if v is not None else "" for v in stream_modes]
    result_df["early_stopped"] = [
        bool(v) if v is not None else False for v in early_stopped_flags
    ]
    result_df["fallback_used"] = [
        bool(v) if v is not None else False for v in fallback_used_flags
    ]
    result_df["fallback_reason"] = [v if v is not None else "" for v in fallback_reasons]
    result_df["processed_tokens"] = [
        int(v) if v is not None else 0 for v in processed_tokens_list
    ]
    result_df["total_tokens"] = [
        int(v) if v is not None else 0 for v in total_tokens_list
    ]
    result_df["setup_tokens"] = [
        int(v) if v is not None else 0 for v in setup_tokens_list
    ]
    result_df["moderated_tokens"] = [
        int(v) if v is not None else 0 for v in moderated_tokens_list
    ]

    truths = [bool(v) for v in df[ground_truth_column].tolist()]
    resolved_preds = [bool(v) if v is not None else False for v in predictions]
    guard_metrics = task.metrics_calculator.compute(
        parsed_outputs,
        truths,
        latencies_ms=latencies_ms,
        costs_usd=costs_usd,
    )
    latency_breakdown = compute_latency_breakdown_metrics(
        predictions,
        truths,
        latencies_ms,
        detection_latencies_ms,
    )
    token_efficiency = compute_token_efficiency_metrics(
        processed_tokens_list,
        total_tokens_list,
        setup_tokens_list,
        moderated_tokens_list,
    )
    result_df["ttd_ms"] = [
        (
            round(float(det if det is not None else lat), 2)
            if truth and pred
            else np.nan
        )
        for truth, pred, lat, det in zip(truths, resolved_preds, latencies_ms, detection_latencies_ms)
    ]
    result_df["full_pass_latency_ms"] = [
        round(float(lat), 2) if (not truth and not pred) else np.nan
        for truth, pred, lat in zip(truths, resolved_preds, latencies_ms)
    ]

    stream_rows = sum(1 for mode in stream_modes if bool(mode))
    fallback_rows = sum(1 for flag in fallback_used_flags if flag)
    early_stop_rows = sum(1 for flag in early_stopped_flags if flag)
    stream_fallback_rate = (fallback_rows / stream_rows) if stream_rows else 0.0
    stream_early_stop_rate = (early_stop_rows / stream_rows) if stream_rows else 0.0

    meta: dict[str, Any] = {
        "model_key": spec.key,
        "model_id": spec.model_id,
        "provider": spec.provider,
        "execution_time_sec": round(elapsed, 2),
        "model_params_b": spec.params_b,
        "prompt_tokens_total": prompt_tokens_total,
        "completion_tokens_total": completion_tokens_total,
        **guard_metrics,
        **latency_breakdown,
        **token_efficiency,
        "stream_rows": stream_rows,
        "stream_fallback_count": fallback_rows,
        "stream_fallback_rate": round(stream_fallback_rate, 4),
        "stream_early_stop_count": early_stop_rows,
        "stream_early_stop_rate": round(stream_early_stop_rate, 4),
        **_summarize_vram_usage_mb(vram_samples_mb, device_count=vram_device_count),
    }

    return result_df, meta
