#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fnmatch
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK_DIR = ROOT / "configs" / "benchmarks"


@dataclass
class BenchmarkRunResult:
    benchmark: Path
    command: list[str]
    exit_code: int
    elapsed_sec: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="用 Python 逐一執行 benchmarking.cli，不依賴 uv。"
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=DEFAULT_BENCHMARK_DIR,
        help="benchmark YAML 所在目錄，預設為 configs/benchmarks",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="要拿來執行 CLI 的 Python，預設為目前的 sys.executable",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="只執行符合 glob pattern 的 benchmark，可重複指定",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="排除符合 glob pattern 的 benchmark，可重複指定",
    )
    parser.add_argument(
        "--validate-first",
        action="store_true",
        help="每個 benchmark 先執行 validate，再執行 run",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="只做 validate，不執行 run",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="run 時傳遞 --no-progress",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="任一 benchmark 失敗時立即停止",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只列出將執行的指令，不真的執行",
    )
    return parser


def _matches_any(path: Path, patterns: list[str]) -> bool:
    if not patterns:
        return False
    relative_name = path.relative_to(ROOT).as_posix()
    file_name = path.name
    stem_name = path.stem
    return any(
        fnmatch.fnmatch(relative_name, pattern)
        or fnmatch.fnmatch(file_name, pattern)
        or fnmatch.fnmatch(stem_name, pattern)
        for pattern in patterns
    )


def discover_benchmarks(
    benchmark_dir: Path,
    include_patterns: list[str],
    exclude_patterns: list[str],
) -> list[Path]:
    if not benchmark_dir.exists():
        raise FileNotFoundError(f"benchmark 目錄不存在: {benchmark_dir}")
    if not benchmark_dir.is_dir():
        raise NotADirectoryError(f"benchmark 路徑不是目錄: {benchmark_dir}")

    benchmarks = sorted(benchmark_dir.glob("*.yaml"))
    selected = [
        path
        for path in benchmarks
        if (not include_patterns or _matches_any(path, include_patterns))
        and not _matches_any(path, exclude_patterns)
    ]
    if not selected:
        raise FileNotFoundError("找不到符合條件的 benchmark YAML")
    return selected


def build_cli_command(
    python_bin: str,
    cli_command: str,
    benchmark_path: Path,
    *,
    no_progress: bool,
) -> list[str]:
    command = [
        python_bin,
        "-m",
        "benchmarking.cli",
        cli_command,
        "--benchmark",
        str(benchmark_path),
    ]
    if cli_command == "run" and no_progress:
        command.append("--no-progress")
    return command


def run_command(command: list[str]) -> tuple[int, float]:
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=ROOT, check=False)
    elapsed = time.perf_counter() - started
    return completed.returncode, elapsed


def print_command(command: list[str]) -> None:
    print("$", " ".join(command))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    benchmark_dir = args.benchmark_dir.resolve()
    benchmarks = discover_benchmarks(
        benchmark_dir,
        include_patterns=args.include,
        exclude_patterns=args.exclude,
    )

    print(f"python: {args.python}")
    print(f"benchmark_dir: {benchmark_dir}")
    print(f"benchmarks: {len(benchmarks)}")

    results: list[BenchmarkRunResult] = []

    for index, benchmark_path in enumerate(benchmarks, start=1):
        relative_path = benchmark_path.relative_to(ROOT)
        print(f"\n[{index}/{len(benchmarks)}] {relative_path}")

        commands: list[list[str]] = []
        if args.validate_first or args.validate_only:
            commands.append(
                build_cli_command(
                    args.python,
                    "validate",
                    benchmark_path,
                    no_progress=args.no_progress,
                )
            )
        if not args.validate_only:
            commands.append(
                build_cli_command(
                    args.python,
                    "run",
                    benchmark_path,
                    no_progress=args.no_progress,
                )
            )

        for command in commands:
            print_command(command)
            if args.dry_run:
                results.append(
                    BenchmarkRunResult(
                        benchmark=benchmark_path,
                        command=command,
                        exit_code=0,
                        elapsed_sec=0.0,
                    )
                )
                continue

            exit_code, elapsed_sec = run_command(command)
            results.append(
                BenchmarkRunResult(
                    benchmark=benchmark_path,
                    command=command,
                    exit_code=exit_code,
                    elapsed_sec=elapsed_sec,
                )
            )
            print(f"exit_code={exit_code} elapsed_sec={elapsed_sec:.2f}")
            if exit_code != 0 and args.fail_fast:
                print("\n中止: fail-fast 已啟用。")
                return exit_code

    failed = [item for item in results if item.exit_code != 0]
    print("\nSummary")
    print(f"total_commands={len(results)}")
    print(f"failed_commands={len(failed)}")
    for item in failed:
        print(
            f"- {item.benchmark.relative_to(ROOT)} "
            f"exit_code={item.exit_code} elapsed_sec={item.elapsed_sec:.2f}"
        )

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
