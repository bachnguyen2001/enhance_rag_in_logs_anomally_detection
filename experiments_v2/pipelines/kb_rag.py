"""M2-M5: score the full KB, select references, then attach their labels."""

import time
import numpy as np

from experiments_v2.shared.llm import build_messages
from experiments_v2.shared.progress import Progress
from experiments_v2.shared.retrieval import top_indices


def select_contexts(scores, config):
    scores = np.asarray(scores)
    if config["selection"] == "fixed":
        return top_indices(scores, config["contexts"]), len(scores)
    pool = top_indices(scores, min(config["candidate_pool"], len(scores)))
    passing = [index for index in pool if scores[index] >= config["threshold"]]
    return passing, len(passing)


def predict_one(query, scores, resources, config):
    started = time.perf_counter()
    selected, passing_count = select_contexts(scores, config)
    kb = resources["knowledge_base"]
    contexts = [{"text": kb[i]["text"], "label": kb[i]["label"]} for i in selected]
    messages = build_messages(query["text"], contexts)
    selection_seconds = time.perf_counter() - started
    response = resources["llm"].classify(messages)
    return {
        "block_id": query["block_id"], "label": query["label"],
        "method": config["method_id"], "kb_variant": config["kb_variant"],
        "retrieval": config["retrieval"], "selection": config["selection"],
        "weights": config.get("weights"), "threshold": config.get("threshold"),
        "candidate_count": len(scores), "passing_count": passing_count,
        "filtered_out": len(scores) - passing_count,
        "context_count": len(selected), "selection_seconds": selection_seconds,
        "selected_evidence": [
            {"rank": rank, "trace_id": kb[i]["trace_id"], "label": kb[i]["label"],
             "score": float(scores[i])}
            for rank, i in enumerate(selected, 1)
        ],
        "contexts": [kb[i]["trace_id"] for i in selected],
        "messages": messages, **response,
    }


def run(queries, resources, config):
    progress = Progress(config["method_id"], len(queries), config.get("progress_every", 10))
    rows = []
    for index, query in enumerate(queries):
        progress.before(index + 1)
        started = time.perf_counter()
        scores = resources["scores"][index]
        retrieval_seconds = time.perf_counter() - started
        row = predict_one(query, scores, resources, config)
        row["retrieval_seconds"] = retrieval_seconds
        row["query_seconds_uncached"] = (
            retrieval_seconds + row["selection_seconds"] + row["latency_seconds"]
            if row.get("latency_seconds") is not None else None
        )
        rows.append(row)
        resources.get("record_prediction", lambda row: None)(row)
        progress.after(index + 1, row)
    return rows
