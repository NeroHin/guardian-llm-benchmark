from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from benchmarking.config import (
    DEFAULT_DATASETS_DIR,
    DEFAULT_MODELS_DIR,
    load_benchmark_spec,
    load_dataset_registry,
    load_model_registry,
    resolve_model_selection,
    validate_benchmark_spec,
)
from benchmarking.datasets import build_eval_dataframe
from benchmarking.reporting import (
    write_leaderboard_csv,
    write_metrics_json,
    write_rows_csv,
    write_run_manifest,
)
from benchmarking.runtime import run_single_model
from benchmarking.tasks import get_task_definition


ROOT = Path(__file__).resolve().parents[1]


def _benchmark_mode(model_spec: Any) -> str:
    settings = getattr(model_spec, "settings", {}) or {}
    return str(settings.get("benchmark_mode") or "default")


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def run_benchmark(
    benchmark_path: Path,
    *,
    show_progress: bool = True,
) -> Path:
    benchmark = load_benchmark_spec(benchmark_path)
    model_registry = load_model_registry(DEFAULT_MODELS_DIR)
    dataset_registry = load_dataset_registry(DEFAULT_DATASETS_DIR)
    validate_benchmark_spec(benchmark, model_registry=model_registry, dataset_registry=dataset_registry)
    task = get_task_definition(benchmark.task)

    eval_df = build_eval_dataframe(benchmark, dataset_registry, root=ROOT)
    selected_models = resolve_model_selection(model_registry, benchmark.models)

    run_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_dir = ROOT / benchmark.outputs.dir / run_id
    rows_dir = run_dir / "rows"
    metrics_dir = run_dir / "metrics"

    all_metrics: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for model_spec in selected_models:
        started_at = datetime.now().isoformat(timespec="seconds")
        try:
            result_df, meta = run_single_model(
                eval_df,
                model_spec,
                content_column="content",
                ground_truth_column="ground_truth",
                show_progress=show_progress,
                task_id=task.task_id,
                local_batch_size=benchmark.runtime.batch_size,
            )
            finished_at = datetime.now().isoformat(timespec="seconds")
            metrics_payload = {
                **meta,
                "status": "ok",
                "task": task.task_id,
                "benchmark_key": benchmark.key,
                "benchmark_mode": _benchmark_mode(model_spec),
                "batch_size": benchmark.runtime.batch_size,
                "model_started_at": started_at,
                "model_finished_at": finished_at,
            }
            all_metrics.append(metrics_payload)
            if benchmark.outputs.save_rows:
                write_rows_csv(result_df, rows_dir / f"{model_spec.key}.csv")
            if benchmark.outputs.save_metrics_json:
                write_metrics_json(metrics_payload, metrics_dir / f"{model_spec.key}.json")
        except Exception as exc:
            failure = {
                "model_key": model_spec.key,
                "model_id": model_spec.model_id,
                "provider": model_spec.provider,
                "benchmark_mode": _benchmark_mode(model_spec),
                "batch_size": benchmark.runtime.batch_size,
                "status": "failed",
                "error": str(exc),
            }
            failures.append(failure)
            all_metrics.append(failure)
            if benchmark.runtime.fail_fast:
                raise

    leaderboard = write_leaderboard_csv(all_metrics, run_dir / "leaderboard.csv")
    write_run_manifest(
        {
            "benchmark_key": benchmark.key,
            "task": benchmark.task,
            "dataset": {
                "positive": list(benchmark.dataset.positive),
                "negative": list(benchmark.dataset.negative),
                "source": list(benchmark.dataset.source),
            },
            "models": [model.key for model in selected_models],
            "model_runs": [
                {
                    "model_key": model.key,
                    "model_id": model.model_id,
                    "provider": model.provider,
                    "benchmark_mode": _benchmark_mode(model),
                }
                for model in selected_models
            ],
            "config_snapshot": {
                "benchmark": str(benchmark.source_path),
                "models_dir": str(DEFAULT_MODELS_DIR),
                "datasets_dir": str(DEFAULT_DATASETS_DIR),
            },
            "start_end_note": "Per-model timestamps are stored in metrics files.",
            "git_commit": _git_commit(),
            "leaderboard_path": "leaderboard.csv",
            "failed_models": failures,
        },
        run_dir / "run_manifest.json",
    )
    return run_dir
