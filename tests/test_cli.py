from __future__ import annotations

from benchmarking.cli import main


def test_cli_list_benchmarks(capsys) -> None:
    exit_code = main(["list", "benchmarks"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "configs/benchmarks/pii_baseline.yaml" in captured.out


def test_cli_validate_default_benchmark(capsys) -> None:
    exit_code = main(["validate", "--benchmark", "configs/benchmarks/pii_baseline.yaml"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "benchmark validated" in captured.out
