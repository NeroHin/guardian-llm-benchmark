from __future__ import annotations

import argparse
from pathlib import Path

from benchmarking.config import (
    DEFAULT_BENCHMARKS_DIR,
    DEFAULT_DATASETS_DIR,
    DEFAULT_MODELS_DIR,
    load_benchmark_spec,
    load_dataset_registry,
    load_model_registry,
    validate_benchmark_spec,
)
from benchmarking.core import run_benchmark


def _list_configs(target: str) -> None:
    mapping = {
        "models": DEFAULT_MODELS_DIR,
        "datasets": DEFAULT_DATASETS_DIR,
        "benchmarks": DEFAULT_BENCHMARKS_DIR,
    }
    base_dir = mapping[target]
    for path in sorted(base_dir.glob("*.yaml")):
        print(path.relative_to(base_dir.parent.parent))


def _validate(benchmark_path: Path) -> None:
    benchmark = load_benchmark_spec(benchmark_path)
    models = load_model_registry(DEFAULT_MODELS_DIR)
    datasets = load_dataset_registry(DEFAULT_DATASETS_DIR)
    validate_benchmark_spec(benchmark, model_registry=models, dataset_registry=datasets)
    print(f"benchmark validated: {benchmark_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="guardian-benchmark")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a benchmark suite")
    run_parser.add_argument("--benchmark", required=True, type=Path)
    run_parser.add_argument("--no-progress", action="store_true")

    validate_parser = subparsers.add_parser("validate", help="Validate benchmark config")
    validate_parser.add_argument("--benchmark", required=True, type=Path)

    list_parser = subparsers.add_parser("list", help="List available configs")
    list_parser.add_argument("target", choices=("models", "datasets", "benchmarks"))

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        run_dir = run_benchmark(args.benchmark, show_progress=not args.no_progress)
        print(run_dir)
        return 0
    if args.command == "validate":
        _validate(args.benchmark)
        return 0
    if args.command == "list":
        _list_configs(args.target)
        return 0
    parser.error(f"未知指令: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
