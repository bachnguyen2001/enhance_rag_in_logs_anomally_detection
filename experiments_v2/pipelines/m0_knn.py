"""M0: distance-weighted KNN on EventId occurrence vectors."""

import numpy as np

from experiments_v2.shared.evaluation import detection_metrics
from experiments_v2.shared.retrieval import knn_predict


def run(queries, resources, config):
    if config.get("selected_k") is not None:
        selected_k = config["selected_k"]
        predictions, latency = knn_predict(
            resources["train_vectors"], resources["train_labels"], resources["query_vectors"], selected_k
        )
        rows = [{
            "block_id": query["block_id"], "label": query["label"], "prediction": prediction,
            "method": "M0", "k": selected_k, "contexts": [], "latency_seconds": seconds,
            "prompt_tokens": 0, "request_count": 0, "cache_hit": False, "new_api_calls": 0,
        } for query, prediction, seconds in zip(queries, predictions, latency)]
        return rows, {"selected_k": selected_k}

    candidates = []
    for k in config.get("k_candidates", [1, 3, 5]):
        predictions, _ = knn_predict(resources["train_vectors"], resources["train_labels"], resources["query_vectors"], k)
        candidates.append({"k": k, **detection_metrics([row["label"] for row in queries], predictions)})
    best = max(candidates, key=lambda row: (row["f1"], row["recall"], -row["k"]))
    return run(queries, resources, {"selected_k": best["k"]})[0], {
        "candidates": candidates, "selected_k": best["k"]
    }
