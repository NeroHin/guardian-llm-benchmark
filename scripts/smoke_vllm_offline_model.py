#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarking.runtime import ModelSpec, create_runner  # noqa: E402


DEFAULT_MODEL_ID = "nvidia/Nemotron-H-4B-Instruct-128K"
DEFAULT_PROMPT = "請判斷這段文字是否包含個資：王小明的手機是 0912-345-678。"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="最小 smoke test：直接走 repo 的 vllm_offline runner，驗證模型能否載入並生成。"
    )
    parser.add_argument("--key", default="smoke-vllm-offline")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--profile", default="qwen_instruct_json")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--trust-remote-code", dest="trust_remote_code", action="store_true")
    parser.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=131072)
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-generate", action="store_true")
    parser.set_defaults(trust_remote_code=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    spec = ModelSpec(
        key=args.key,
        model_id=args.model_id,
        provider="vllm_offline",
        profile=args.profile,
        settings={
            "trust_remote_code": bool(args.trust_remote_code),
            "torch_dtype": args.torch_dtype,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
        },
    )

    runner = None
    started = time.perf_counter()
    try:
        runner = create_runner(spec)
        load_elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        payload: dict[str, object] = {
            "status": "loaded",
            "model_id": spec.model_id,
            "provider": spec.provider,
            "profile": spec.profile,
            "load_elapsed_ms": load_elapsed_ms,
        }

        if args.skip_generate:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0

        infer_started = time.perf_counter()
        inference = runner.predict(args.prompt)
        infer_elapsed_ms = round((time.perf_counter() - infer_started) * 1000, 2)
        payload.update(
            {
                "status": "ok",
                "infer_elapsed_ms": infer_elapsed_ms,
                "contains_pii": inference.contains_pii,
                "prompt_tokens": inference.prompt_tokens,
                "completion_tokens": inference.completion_tokens,
                "output_json": inference.output_json,
            }
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        payload = {
            "status": "error",
            "model_id": spec.model_id,
            "provider": spec.provider,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 1
    finally:
        if runner is not None:
            try:
                runner.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
