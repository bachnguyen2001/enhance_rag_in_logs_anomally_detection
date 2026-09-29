"""Archived E5 pipeline; the active M5 uses pipelines/kb_rag.py."""

from experiments_v2.shared.llm import build_messages
from experiments_v2.shared.progress import Progress
from experiments_v2.shared.retrieval import top_indices


def _token_estimate(text):
    return len(text.split())


def select_contexts(scores, threshold, resources, config):
    kb = resources["knowledge_base"]
    candidates = top_indices(scores, 10)
    passing = [index for index in candidates if scores[index] >= threshold]
    selected = []
    budget = int(config.get("context_token_budget", 1200))
    used = 0
    redundancy = float(config.get("redundancy_threshold", 0.98))
    for index in passing:
        if len(selected) >= int(config.get("max_contexts", 3)):
            break
        if any(_sequence_similarity(kb[index]["event_ids"], kb[chosen]["event_ids"]) >= redundancy for chosen in selected):
            continue
        cost = _token_estimate(kb[index]["text"])
        if selected and used + cost > budget:
            continue
        selected.append(index)
        used += cost
    return candidates, selected, len(passing) - len(selected), used


def _sequence_similarity(left, right):
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    a, b = list(left), list(right)
    previous = [0] * (len(b) + 1)
    for event in a:
        current = [0]
        for index, other in enumerate(b, 1):
            current.append(previous[index - 1] + 1 if event == other else max(previous[index], current[-1]))
        previous = current
    return previous[-1] / max(len(a), len(b))


def predict_one(query, scores, resources, config):
    threshold = float(config["threshold"])
    candidates, selected, filtered_out, context_tokens = select_contexts(scores, threshold, resources, config)
    kb = resources["knowledge_base"]
    messages = build_messages(query["text"], [kb[index]["text"] for index in selected])
    response = resources["llm"].classify(messages)
    selected_set = set(selected)
    evidence = [{"rank": rank, "trace_id": kb[index]["trace_id"], "score": float(scores[index]),
                 "passes_relevance": bool(scores[index] >= threshold),
                 "selected_as_context": index in selected_set}
                for rank, index in enumerate(candidates, 1)]
    return {"block_id": query["block_id"], "label": query["label"], "method": "E5",
            "retrieval_policy": "hybrid top-10 -> threshold -> redundancy removal -> token budget -> max 3",
            "weights": list(config["weights"]), "threshold": threshold,
            "candidate_count": len(candidates), "filtered_out": filtered_out,
            "context_token_budget": int(config.get("context_token_budget", 1200)),
            "selected_context_tokens_estimate": context_tokens, "candidates": evidence,
            "contexts": [kb[index]["trace_id"] for index in selected],
            "messages": messages, **response}


def run(queries, resources, config):
    progress = Progress("E5", len(queries), config.get("progress_every", 10))
    rows = []
    for index, (query, scores) in enumerate(zip(queries, resources["structure_scores"]), 1):
        progress.before(index)
        rows.append(predict_one(query, scores, resources, config))
        resources.get("record_prediction", lambda row: None)(rows[-1])
        progress.after(index, rows[-1])
    return rows
