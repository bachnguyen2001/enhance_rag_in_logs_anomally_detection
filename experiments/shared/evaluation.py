"""Metrics and result summaries used by all four methods."""

from collections import Counter
import numpy as np


def detection_metrics(labels, predictions):
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    valid = np.isin(predictions, ["NORMAL", "ANOMALY"])
    tp = int(((labels == "ANOMALY") & (predictions == "ANOMALY")).sum())
    tn = int(((labels == "NORMAL") & (predictions == "NORMAL")).sum())
    fp = int(((labels == "NORMAL") & (predictions == "ANOMALY")).sum())
    fn = int(((labels == "ANOMALY") & (predictions == "NORMAL")).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "queries": len(labels), "Normal": int((labels == "NORMAL").sum()),
        "Anomaly": int((labels == "ANOMALY").sum()), "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "invalid_normal": int(((~valid) & (labels == "NORMAL")).sum()),
        "invalid_anomaly": int(((~valid) & (labels == "ANOMALY")).sum()),
        "invalid_output_rate": float((~valid).mean()),
        "precision": precision if valid.all() else None,
        "recall": recall if valid.all() else None,
        "f1": f1 if valid.all() else None,
    }


def summarize(rows):
    def average(values):
        values = [value for value in values if value is not None]
        return float(np.mean(values)) if values else None
    result = detection_metrics([row["label"] for row in rows], [row["prediction"] for row in rows])
    observed = [row for row in rows if row.get("prompt_tokens") is not None and row.get("request_count", 0)]
    result.update({
        "average_contexts": average([len(row.get("contexts", [])) for row in rows]),
        "context_count_distribution": dict(Counter(str(len(row.get("contexts", []))) for row in rows)),
        "average_prompt_tokens": sum(row["prompt_tokens"] for row in observed) / sum(row["request_count"] for row in observed) if observed else 0,
        "average_latency_seconds_uncached": average([row.get("latency_seconds") for row in rows]),
        "cache_hits": sum(row.get("cache_hit", False) for row in rows),
        "new_api_calls": sum(row.get("new_api_calls", 0) for row in rows),
    })
    candidates = sum(row.get("candidate_count", 0) for row in rows)
    result["candidate_removal_rate"] = sum(row.get("filtered_out", 0) for row in rows) / candidates if candidates else None
    return result
