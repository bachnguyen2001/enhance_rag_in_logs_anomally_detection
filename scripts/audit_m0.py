"""Audit why the M0 count-vector KNN baseline performs almost perfectly.

Run from the project root:
    .venv/bin/python -m scripts.audit_m0

This script uses only saved data/artifacts. It does not call an LLM.
"""

from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.shared.data import normalize, read_json, read_jsonl, save_json, sha256


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "experiments/artifacts"
OUTPUT = ROOT / "experiments/results/analysis"
EVENT_COLUMNS = [f"E{i}" for i in range(1, 30)]


def vector_key(vector):
    """KNN compares normalized directions; rounding absorbs harmless float noise."""
    return tuple(np.round(np.asarray(vector, dtype=np.float32), 7))


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    knn = ARTIFACTS / "knn"
    train_vectors = np.load(knn / "train_vectors.npy", mmap_mode="r")
    train_labels = np.load(knn / "train_labels.npy", allow_pickle=False)
    test_vectors = np.load(knn / "test_vectors.npy")
    train_ids = pd.read_csv(knn / "train_ids.csv")
    test_queries = read_jsonl(ARTIFACTS / "queries/test.jsonl")
    test_results = read_json(ROOT / "experiments/results/shared/test/m0_summary.json")
    predictions = {row["block_id"]: row for row in
                   read_jsonl(ROOT / "experiments/results/shared/test/m0_predictions.jsonl")}

    split_dir = ROOT / "processed_v2/group_trace"
    split_frames = {name: pd.read_csv(split_dir / f"{name}_block_ids.csv")
                    for name in ("train", "validation", "test")}
    train_blocks = set(split_frames["train"].block_id)
    test_blocks = set(split_frames["test"].block_id)
    train_signatures = set(split_frames["train"].trace_signature)
    test_signatures = set(split_frames["test"].trace_signature)

    if len(train_ids) != len(train_vectors) or len(train_labels) != len(train_vectors):
        raise AssertionError("KNN train IDs, labels and vectors are not aligned")
    if [row["block_id"] for row in test_queries] != list(predictions):
        raise AssertionError("Test query order and M0 prediction order differ")
    if train_ids.block_id.tolist() != sorted(train_ids.block_id):
        raise AssertionError("Train IDs are not in the documented stable tie-break order")
    if not np.array_equal(train_ids.label.str.upper().to_numpy(dtype="U7"), train_labels):
        raise AssertionError("Saved KNN labels differ from train manifest labels")

    groups = defaultdict(lambda: {"NORMAL": 0, "ANOMALY": 0, "first_index": None})
    for index, (vector, label) in enumerate(zip(train_vectors, train_labels)):
        group = groups[vector_key(vector)]
        group[str(label)] += 1
        if group["first_index"] is None:
            group["first_index"] = index

    unique_vectors = len(groups)
    ambiguous_vectors = sum(bool(group["NORMAL"] and group["ANOMALY"]) for group in groups.values())
    ambiguous_train_blocks = sum(group["NORMAL"] + group["ANOMALY"] for group in groups.values()
                                 if group["NORMAL"] and group["ANOMALY"])

    details = []
    unmatched_positions = []
    for position, (query, vector) in enumerate(zip(test_queries, test_vectors)):
        group = groups.get(vector_key(vector))
        if group is None:
            unmatched_positions.append(position)
            details.append(None)
            continue
        neighbor_index = group["first_index"]
        predicted = predictions[query["block_id"]]["prediction"]
        details.append({
            "block_id": query["block_id"], "label": query["label"], "prediction": predicted,
            "correct": predicted == query["label"], "exact_normalized_count_match": True,
            "nearest_train_block_id": train_ids.iloc[neighbor_index].block_id,
            "nearest_similarity": 1.0,
            "matching_train_normal": group["NORMAL"],
            "matching_train_anomaly": group["ANOMALY"],
            "matching_vector_is_label_ambiguous": bool(group["NORMAL"] and group["ANOMALY"]),
        })

    # Only unmatched directions require a full cosine search.
    for position in unmatched_positions:
        scores = np.clip(train_vectors @ test_vectors[position], -1, 1)
        neighbor_index = int(np.argmax(scores))
        query = test_queries[position]
        predicted = predictions[query["block_id"]]["prediction"]
        details[position] = {
            "block_id": query["block_id"], "label": query["label"], "prediction": predicted,
            "correct": predicted == query["label"], "exact_normalized_count_match": False,
            "nearest_train_block_id": train_ids.iloc[neighbor_index].block_id,
            "nearest_similarity": float(scores[neighbor_index]),
            "matching_train_normal": 0, "matching_train_anomaly": 0,
            "matching_vector_is_label_ambiguous": False,
        }

    detail_frame = pd.DataFrame(details)
    detail_frame.to_csv(OUTPUT / "m0_test_neighbor_audit.csv", index=False)
    exact = detail_frame.exact_normalized_count_match
    matching_label = ((detail_frame.label == "NORMAL") & detail_frame.matching_train_normal.gt(0)) | \
                     ((detail_frame.label == "ANOMALY") & detail_frame.matching_train_anomaly.gt(0))

    manifest = read_json(ARTIFACTS / "manifest.json")
    audit = {
        "conclusion": (
            "M0 is strong because EventId occurrence vectors are highly label-separable and many test "
            "queries have an identical normalized count-vector direction in training. This is not BlockId "
            "or exact ordered-trace leakage; it is a consequence of the lossy count representation."
        ),
        "feature_audit": {
            "representation": "29 EventId occurrence counts, independently L2-normalized per BlockId",
            "included_columns": EVENT_COLUMNS,
            "excluded_fields": ["BlockId", "label", "Type", "TimeInterval", "Latency", "EventTemplate"],
            "uses_tfidf_or_fitted_vocabulary": False,
            "uses_label_derived_features": False,
            "knn_k": test_results.get("selected_k", 1),
            "voting": "distance weighted; stable BlockId order resolves equal-score k=1 ties",
        },
        "split_audit": {
            "train_blocks": len(train_blocks), "test_blocks_full_split": len(test_blocks),
            "train_test_block_id_overlap": len(train_blocks & test_blocks),
            "train_test_exact_ordered_sequence_overlap": len(train_signatures & test_signatures),
            "test_subset_queries": len(test_queries),
            "test_subset_block_ids_all_belong_to_test_split": all(row["block_id"] in test_blocks for row in test_queries),
        },
        "artifact_audit": {
            "data_version": manifest.get("data_version"),
            "representation_version": manifest.get("representation"),
            "train_ids_sha256": sha256(knn / "train_ids.csv"),
            "train_vectors_sha256": sha256(knn / "train_vectors.npy"),
            "test_vectors_sha256": sha256(knn / "test_vectors.npy"),
            "saved_labels_match_train_ids": True,
            "query_order_matches_predictions": True,
        },
        "count_vector_overlap": {
            "train_blocks": len(train_vectors),
            "unique_normalized_train_vectors": unique_vectors,
            "train_blocks_per_unique_vector": float(len(train_vectors) / unique_vectors),
            "ambiguous_train_vector_groups": ambiguous_vectors,
            "train_blocks_in_ambiguous_vector_groups": ambiguous_train_blocks,
            "test_queries_with_exact_normalized_count_match_in_train": int(exact.sum()),
            "test_exact_match_fraction": float(exact.mean()),
            "test_queries_with_same_label_available_in_exact_match_group": int((exact & matching_label).sum()),
            "test_exact_matches_with_ambiguous_train_labels": int((exact & detail_frame.matching_vector_is_label_ambiguous).sum()),
            "test_queries_without_exact_count_match": int((~exact).sum()),
            "exact_match_label_counts": detail_frame.loc[exact, "label"].value_counts().to_dict(),
            "exact_match_correct": int(detail_frame.loc[exact, "correct"].sum()),
            "exact_match_accuracy": float(detail_frame.loc[exact, "correct"].mean()),
            "non_exact_label_counts": detail_frame.loc[~exact, "label"].value_counts().to_dict(),
            "non_exact_correct": int(detail_frame.loc[~exact, "correct"].sum()),
            "non_exact_accuracy": float(detail_frame.loc[~exact, "correct"].mean()),
            "nearest_similarity_for_non_exact_queries": (
                detail_frame.loc[~exact, "nearest_similarity"].describe().to_dict() if (~exact).any() else {}
            ),
        },
        "observed_result": test_results,
        "interpretation_limits": [
            "Group-by-trace prevents identical ordered EventId sequences across train and test.",
            "It does not prevent different ordered sequences from sharing the same event-count vector.",
            "Therefore a zero-distance M0 neighbor is representation overlap, not duplicate BlockId leakage.",
            "Only two test queries are unseen under M0's normalized count-vector representation, so the overall score mostly measures lookup of previously observed count patterns.",
            "Accuracy on the two unseen count directions is descriptive only and too small for a generalization claim.",
            "The result applies to this HDFS split and count representation; it is not evidence that KNN generalizes to other log datasets.",
        ],
        "detail_file": "experiments/results/analysis/m0_test_neighbor_audit.csv",
    }
    save_json(OUTPUT / "m0_audit.json", audit)
    print(json.dumps({
        "train_test_block_overlap": audit["split_audit"]["train_test_block_id_overlap"],
        "train_test_sequence_overlap": audit["split_audit"]["train_test_exact_ordered_sequence_overlap"],
        "unique_train_count_vectors": unique_vectors,
        "ambiguous_train_vector_groups": ambiguous_vectors,
        "test_exact_count_matches": int(exact.sum()),
        "test_exact_match_fraction": float(exact.mean()),
        "m0_test_f1": test_results["f1"],
    }, indent=2))


if __name__ == "__main__":
    main()
