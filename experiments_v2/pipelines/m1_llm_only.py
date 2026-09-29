"""M1: classify the query with no retrieved context."""

from experiments_v2.shared.llm import build_messages
from experiments_v2.shared.progress import Progress


def predict_one(query, resources, config):
    messages = build_messages(query["text"], [])
    response = resources["llm"].classify(messages)
    return {"block_id": query["block_id"], "label": query["label"], "method": "M1",
            "contexts": [], "messages": messages, **response}


def run(queries, resources, config):
    progress = Progress("M1", len(queries), config.get("progress_every", 10))
    rows = []
    for index, query in enumerate(queries, 1):
        progress.before(index)
        rows.append(predict_one(query, resources, config))
        resources.get("record_prediction", lambda row: None)(rows[-1])
        progress.after(index, rows[-1])
    return rows
