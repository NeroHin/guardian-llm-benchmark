from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class PromptBuilder(Protocol):
    def build(self, model_id: str, content: str) -> dict[str, str]:
        ...


class OutputParser(Protocol):
    def parse(self, raw_text: str) -> dict[str, Any]:
        ...


class MetricsCalculator(Protocol):
    def compute(
        self,
        parsed_outputs: list[dict[str, Any]],
        ground_truths: list[bool],
        *,
        latencies_ms: list[float],
        costs_usd: list[float],
    ) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class TaskDefinition:
    task_id: str
    prompt_builder: PromptBuilder
    output_schema: dict[str, Any]
    output_parser: OutputParser
    metrics_calculator: MetricsCalculator
