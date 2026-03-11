from __future__ import annotations

from .base import TaskDefinition
from .pii_binary import (
    PIIBinaryMetricsCalculator,
    PIIBinaryPromptBuilder,
    PIIBinaryStrictJSONParser,
    PII_BINARY_OUTPUT_SCHEMA,
)


TASK_REGISTRY: dict[str, TaskDefinition] = {
    "pii_binary": TaskDefinition(
        task_id="pii_binary",
        prompt_builder=PIIBinaryPromptBuilder(),
        output_schema=PII_BINARY_OUTPUT_SCHEMA,
        output_parser=PIIBinaryStrictJSONParser(),
        metrics_calculator=PIIBinaryMetricsCalculator(),
    )
}


def get_task_definition(task_id: str) -> TaskDefinition:
    try:
        return TASK_REGISTRY[task_id]
    except KeyError as exc:
        raise ValueError(f"尚未支援的任務: {task_id}") from exc


__all__ = ["TASK_REGISTRY", "TaskDefinition", "get_task_definition"]
