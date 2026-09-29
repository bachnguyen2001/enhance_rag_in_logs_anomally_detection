"""Deterministic structure-aware scoring for HDFS traces."""

import numpy as np


def template_jaccard(query, candidate):
    left, right = set(query), set(candidate)
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def normalized_lcs(query, candidate):
    if not query and not candidate:
        return 1.0
    if not query or not candidate:
        return 0.0
    previous = [0] * (len(candidate) + 1)
    for event in query:
        current = [0]
        for index, other in enumerate(candidate, 1):
            current.append(previous[index - 1] + 1 if event == other else max(previous[index], current[-1]))
        previous = current
    return previous[-1] / max(len(query), len(candidate))


def hybrid_score(semantic, query_events, candidate_events, weights):
    alpha, beta, gamma = weights
    semantic = float(np.clip((semantic + 1.0) / 2.0, 0.0, 1.0))
    return alpha * semantic + beta * template_jaccard(query_events, candidate_events) + gamma * normalized_lcs(query_events, candidate_events)


def _candidate_masks(events):
    masks = {}
    for index, event in enumerate(events):
        masks[event] = masks.get(event, 0) | (1 << index)
    return masks


def _lcs_length_bitset(query, candidate_masks):
    state = 0
    for event in query:
        matches = candidate_masks.get(event, 0)
        merged = state | matches
        state = merged & ~(merged - ((state << 1) | 1))
    return state.bit_count()


def build_structure_scores(queries, knowledge_base, semantic_scores, weights):
    alpha, beta, gamma = weights
    candidate_data = []
    for candidate in knowledge_base:
        events = candidate["event_ids"]
        candidate_data.append((set(events), _candidate_masks(events), len(events)))
    scores = np.empty_like(semantic_scores, dtype=np.float32)
    for row, query in enumerate(queries):
        query_events = query["event_ids"]
        query_set = set(query_events)
        for col, (candidate_set, masks, candidate_length) in enumerate(candidate_data):
            union = query_set | candidate_set
            template = len(query_set & candidate_set) / len(union) if union else 1.0
            sequence = _lcs_length_bitset(query_events, masks) / max(len(query_events), candidate_length)
            semantic = float(np.clip((semantic_scores[row, col] + 1.0) / 2.0, 0.0, 1.0))
            scores[row, col] = alpha * semantic + beta * template + gamma * sequence
    return scores


def valid_weights(weights):
    return len(weights) == 3 and all(value >= 0 for value in weights) and abs(sum(weights) - 1.0) < 1e-9
