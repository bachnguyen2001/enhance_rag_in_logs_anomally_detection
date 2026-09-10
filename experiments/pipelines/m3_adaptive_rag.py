"""M3: retrieve ten, filter by the validation threshold, keep at most three."""

from experiments.shared.llm import build_messages
from experiments.shared.progress import Progress
from experiments.shared.retrieval import top_indices


def select_contexts(scores, threshold):
    candidates = top_indices(scores, 10)
    passing = [index for index in candidates if scores[index] >= threshold]
    return candidates, passing[:3], len(candidates) - len(passing)


def predict_one(query, scores, resources, config):
    threshold = config["threshold"]
    candidates, selected, filtered_out = select_contexts(scores, threshold)
    kb = resources["knowledge_base"]
    messages = build_messages(query["text"], [kb[index]["text"] for index in selected])
    response = resources["llm"].classify(messages)
    selected_set = set(selected)
    evidence = [{
        "rank": rank,
        "trace_id": kb[index]["trace_id"],
        "similarity": float(scores[index]),
        "passes_relevance": bool(scores[index] >= threshold),
        "selected_as_context": index in selected_set,
    } for rank, index in enumerate(candidates, 1)]
    return {
        "block_id": query["block_id"], "label": query["label"], "method": "M3",
        "retrieval_policy": "cosine top-10 -> similarity >= threshold -> maximum 3 contexts",
        "threshold": threshold,
        "candidate_count": len(candidates),
        "context_count": len(selected),
        "filtered_out": filtered_out,
        "candidates": evidence,
        "contexts": [kb[index]["trace_id"] for index in selected],
        "messages": messages, **response,
    }


def run(queries, resources, config):
    progress = Progress("M3", len(queries), config.get("progress_every", 10))
    rows = []
    for index, (query, scores) in enumerate(zip(queries, resources["similarities"]), 1):
        progress.before(index)
        rows.append(predict_one(query, scores, resources, config))
        resources.get("record_prediction", lambda row: None)(rows[-1])
        progress.after(index, rows[-1])
    return rows
