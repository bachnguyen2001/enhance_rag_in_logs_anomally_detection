"""Archived Normal-only pipeline; the active M2 uses pipelines/kb_rag.py."""

from experiments_v2.shared.llm import build_messages
from experiments_v2.shared.progress import Progress
from experiments_v2.shared.retrieval import top_indices


def predict_one(query, scores, resources, config):
    selected = top_indices(scores, 3)
    kb = resources["knowledge_base"]
    messages = build_messages(query["text"], [kb[index]["text"] for index in selected])
    response = resources["llm"].classify(messages)
    return {
        "block_id": query["block_id"], "label": query["label"], "method": "M2",
        "contexts": [kb[index]["trace_id"] for index in selected],
        "candidates": [{"trace_id": kb[index]["trace_id"], "similarity": float(scores[index])} for index in selected],
        "messages": messages, **response,
    }


def run(queries, resources, config):
    progress = Progress("M2", len(queries), config.get("progress_every", 10))
    rows = []
    for index, (query, scores) in enumerate(zip(queries, resources["similarities"]), 1):
        progress.before(index)
        rows.append(predict_one(query, scores, resources, config))
        resources.get("record_prediction", lambda row: None)(rows[-1])
        progress.after(index, rows[-1])
    return rows
