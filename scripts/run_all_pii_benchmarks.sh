#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -x "$ROOT/.venv/bin/guardian-benchmark" ]]; then
  RUNNER=("$ROOT/.venv/bin/guardian-benchmark")
elif command -v guardian-benchmark >/dev/null 2>&1; then
  RUNNER=("guardian-benchmark")
else
  RUNNER=("python" "-m" "benchmarking.cli")
fi

mapfile -t BENCHMARKS < <(find "$ROOT/configs/benchmarks" -maxdepth 1 -type f -name 'pii_*.yaml' | sort)

if [[ "${#BENCHMARKS[@]}" -eq 0 ]]; then
  echo "找不到 configs/benchmarks/pii_*.yaml" >&2
  exit 1
fi

echo "Using runner: ${RUNNER[*]}"
echo "Validating ${#BENCHMARKS[@]} benchmark files..."
for benchmark in "${BENCHMARKS[@]}"; do
  echo "VALIDATE $(basename "$benchmark")"
  "${RUNNER[@]}" validate --benchmark "$benchmark"
done

echo "Running ${#BENCHMARKS[@]} benchmark files..."
for benchmark in "${BENCHMARKS[@]}"; do
  echo "RUN $(basename "$benchmark")"
  "${RUNNER[@]}" run --benchmark "$benchmark" --no-progress
done

echo "All pii benchmarks completed."
