from __future__ import annotations

from typing import Any

import pandas as pd


class PIIBinaryMetricsCalculator:
    def compute(
        self,
        parsed_outputs: list[dict[str, Any]],
        ground_truths: list[bool],
        *,
        latencies_ms: list[float],
        costs_usd: list[float],
    ) -> dict[str, Any]:
        if len(parsed_outputs) != len(ground_truths):
            raise ValueError("parsed_outputs 與 ground_truths 長度不一致")

        predictions = [item.get("contains_pii") for item in parsed_outputs]

        tp = tn = fp = fn = unparseable = 0
        for pred, truth in zip(predictions, ground_truths):
            if pred is None:
                unparseable += 1
                pred = False
            if truth and pred:
                tp += 1
            elif truth and not pred:
                fn += 1
            elif (not truth) and pred:
                fp += 1
            else:
                tn += 1

        total = len(ground_truths)
        accuracy = (tp + tn) / total if total else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        tpr = recall
        fpr = fp / (fp + tn) if (fp + tn) else 0.0

        lat_series = pd.Series(latencies_ms, dtype=float)
        total_cost_usd = float(sum(costs_usd))
        avg_cost_usd = total_cost_usd / len(costs_usd) if costs_usd else 0.0

        return {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "total": total,
            "unparseable": unparseable,
            "tpr": round(tpr, 4),
            "fpr": round(fpr, 4),
            "overhead_latency_ms_avg": round(float(lat_series.mean()), 2)
            if not lat_series.empty
            else 0.0,
            "overhead_latency_ms_p95": round(float(lat_series.quantile(0.95)), 2)
            if not lat_series.empty
            else 0.0,
            "cost_usd_total": round(total_cost_usd, 6),
            "cost_usd_avg": round(avg_cost_usd, 6),
        }
