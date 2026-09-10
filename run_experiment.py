"""Run the complete HDFS experiment with visible checkpoints.

Usage:
    .venv/bin/python run_experiment.py --models gemini
    .venv/bin/python run_experiment.py --models qwen-local
    .venv/bin/python run_experiment.py --models all

The script resumes from valid outputs, uses a separate cache per model, and
stops when a model's smoke test does not satisfy the JSON contract.
"""

import argparse

import numpy as np
import pandas as pd

from experiments.pipelines.m1_llm_only import predict_one as predict_m1
from experiments.run import (
    load_model_config, load_resources, model_artifacts, run_requested, runtime_signature,
)
from experiments.shared.data import ROOT, experiment_paths, read_json, read_jsonl, save_json


MODELS_PATH = ROOT / "experiments/models.json"


def heading(number, text):
    print(f"\n{'=' * 72}\nSTAGE {number}: {text}\n{'=' * 72}", flush=True)


def valid_summary(path, expected_queries):
    if not path.exists():
        return False
    summary = read_json(path)
    return summary.get("queries") == expected_queries and summary.get("f1") is not None


def run_if_needed(config, pipeline, split, expected_queries):
    summary = experiment_paths(config)["results"] / split / f"{pipeline}_summary.json"
    if valid_summary(summary, expected_queries):
        print(f"[SKIP] {pipeline.upper()} {split}: valid result already exists.", flush=True)
        return
    run_requested(config, pipeline, split)


def check_conditions(config):
    paths = experiment_paths(config)
    required = [
        paths["artifacts"] / "manifest.json",
        paths["artifacts"] / "queries/validation.jsonl",
        paths["artifacts"] / "queries/test.jsonl",
        paths["artifacts"] / "retrieval/normal_metadata.jsonl",
        paths["artifacts"] / "retrieval/normal_embeddings.npy",
        paths["artifacts"] / "retrieval/validation_embeddings.npy",
        paths["artifacts"] / "retrieval/test_embeddings.npy",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing experiment artifacts:\n" + "\n".join(missing))
    if not config["llm"].get("base_url") or not config["llm"].get("model"):
        raise ValueError("Set LLM_BASE_URL and LLM_MODEL in experiments/.env")
    runtime_signature(config, paths["artifacts"])
    print(f"Data: {config['data']['version']}")
    print(f"Retriever: {config['embedding']['model']} + cosine")
    print(f"LLM: {config['llm']['model']} | temperature={config['llm']['temperature']}")
    print("Validation: 160 Normal + 40 Anomaly | Test: 400 Normal + 100 Anomaly")


def smoke_test(config):
    paths = experiment_paths(config)
    destination = paths["results"] / "validation/smoke_test.json"
    if destination.exists() and read_json(destination).get("prediction") in {"NORMAL", "ANOMALY"}:
        print(f"[SKIP] {config['display_name']} smoke test already passed.")
        return
    queries, resources = load_resources(config, "validation", "m1")
    try:
        print(f"Sending one query to verify {config['display_name']} and the JSON contract...", flush=True)
        result = predict_m1(queries[0], resources, config)
    finally:
        resources["llm"].close()
    save_json(destination, result)
    if result["prediction"] not in {"NORMAL", "ANOMALY"}:
        content = result.get("attempts", [{}])[-1].get("content", "")
        raise RuntimeError(f"Smoke test failed: prediction={result['prediction']}, response={content!r}")
    print(f"Smoke test passed: {result['prediction']} | prompt_tokens={result.get('prompt_tokens')}")


def analyze_rag_help_harm(config):
    paths = experiment_paths(config)
    results = paths["results"] / "validation"
    destination = results / "rag_help_harm.csv"
    if destination.exists():
        print("[SKIP] RAG help/harm analysis already exists.")
        return
    m1 = {row["block_id"]: row for row in read_jsonl(results / "m1_predictions.jsonl")}
    m2 = {row["block_id"]: row for row in read_jsonl(results / "m2_predictions.jsonl")}
    query_rows = read_jsonl(paths["artifacts"] / "queries/validation.jsonl")
    retrieval = paths["artifacts"] / "retrieval"
    similarities = np.clip(
        np.load(retrieval / "validation_embeddings.npy") @ np.load(retrieval / "normal_embeddings.npy").T,
        -1, 1,
    )
    rows = []
    for query, scores in zip(query_rows, similarities):
        top3 = np.sort(scores)[-3:][::-1]
        first, second = m1[query["block_id"]], m2[query["block_id"]]
        m1_correct = first["prediction"] == query["label"]
        m2_correct = second["prediction"] == query["label"]
        category = ("RAG_HELP" if not m1_correct and m2_correct else
                    "RAG_HARM" if m1_correct and not m2_correct else
                    "BOTH_CORRECT" if m1_correct and m2_correct else "BOTH_WRONG")
        rows.append({
            "block_id": query["block_id"], "ground_truth": query["label"],
            "m1_prediction": first["prediction"], "m2_prediction": second["prediction"],
            "m1_correct": m1_correct, "m2_correct": m2_correct, "category": category,
            "top1_similarity": float(top3[0]), "mean_top3_similarity": float(top3.mean()),
            "minimum_top3_similarity": float(top3[-1]),
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(destination, index=False)
    group_stats = {}
    for category, part in frame.groupby("category"):
        group_stats[category] = {
            "queries": len(part),
            "median_top1_similarity": float(part.top1_similarity.median()),
            "median_mean_top3_similarity": float(part.mean_top3_similarity.median()),
            "median_minimum_top3_similarity": float(part.minimum_top3_similarity.median()),
        }
    save_json(results / "rag_help_harm_summary.json", group_stats)
    print(pd.DataFrame(group_stats).T.to_string())


def run_shared_m0():
    config = load_model_config("gemini")
    config["run_name"] = "shared"
    config["results_dir"] = "results/shared"
    heading(0, "SHARED M0 BASELINE (NO LLM)")
    run_if_needed(config, "m0", "validation", 200)
    run_if_needed(config, "m0", "test", 500)


def report_model(config):
    model_test = experiment_paths(config)["results"] / "test"
    shared_test = ROOT / "experiments/results/shared/test"
    rows = []
    m0_path = shared_test / "m0_summary.json"
    if m0_path.exists():
        rows.append({"method": "M0", **read_json(m0_path)})
    for path in sorted(model_test.glob("m[123]_summary.json")):
        rows.append({"method": path.stem.split("_")[0].upper(), **read_json(path)})
    if rows:
        frame = pd.DataFrame(rows)
        frame.to_csv(model_test / "comparison.csv", index=False)
        columns = ["method", "precision", "recall", "f1", "average_contexts", "invalid_output_rate"]
        print(frame[columns].to_string(index=False))


def run_model(profile_name):
    config = load_model_config(profile_name)

    print(f"\n{'#' * 72}\nMODEL: {config['display_name']} ({profile_name})\n{'#' * 72}", flush=True)

    heading(1, "LOCKED CONDITIONS AND ARTIFACT CHECK")
    check_conditions(config)

    heading(2, "MODEL SMOKE TEST")
    smoke_test(config)

    heading(3, "M1 AND M2 ON VALIDATION")
    for pipeline in ("m1", "m2"):
        run_if_needed(config, pipeline, "validation", 200)

    heading(4, "RAG_HELP / RAG_HARM ANALYSIS")
    analyze_rag_help_harm(config)

    heading(5, "M3 THRESHOLD SCREENING AND LOCK")
    paths = experiment_paths(config)
    threshold = model_artifacts(config, paths["artifacts"]) / "selected_threshold.json"
    if threshold.exists():
        print(f"[SKIP] Threshold already locked: {read_json(threshold)['threshold']}")
    else:
        run_requested(config, "m3", "validation")

    heading(6, "FINAL M1-M3 TEST")
    for pipeline in ("m1", "m2", "m3"):
        run_if_needed(config, pipeline, "test", 500)

    heading(7, "FINAL COMPARISON")
    report_model(config)
    print(f"\nCompleted model: {config['display_name']}", flush=True)


def combined_report(profile_names):
    rows = []
    shared_m0 = ROOT / "experiments/results/shared/test/m0_summary.json"
    if shared_m0.exists():
        rows.append({"model_profile": "shared", "model": "none", "method": "M0",
                     **read_json(shared_m0)})
    for profile_name in profile_names:
        config = load_model_config(profile_name)
        test_dir = experiment_paths(config)["results"] / "test"
        for path in sorted(test_dir.glob("m?_summary.json")):
            rows.append({
                "model_profile": profile_name,
                "model": config["llm"]["model"],
                "method": path.stem.split("_")[0].upper(),
                **read_json(path),
            })
    if rows:
        destination = ROOT / "experiments/results/model_comparison.csv"
        pd.DataFrame(rows).to_csv(destination, index=False)
        print(f"\nCombined comparison: {destination}")


def main():
    available = list(read_json(MODELS_PATH))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["all"],
                        choices=["all", *available], help="Run one profile or profiles sequentially")
    args = parser.parse_args()
    selected = available if "all" in args.models else args.models
    run_shared_m0()
    for profile_name in selected:
        run_model(profile_name)
    combined_report(available)
    print("\nAll requested model workflows completed.", flush=True)


if __name__ == "__main__":
    main()
