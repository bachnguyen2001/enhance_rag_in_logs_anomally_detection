"""Run the small HDFS M0–M3 experiment.

Examples:
  python -m experiments.run prepare
  python -m experiments.run embed
  python -m experiments.run run --pipeline m3 --split validation
  python -m experiments.run run --pipeline all --split test
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.pipelines import m0_knn, m1_llm_only, m2_fixed_rag, m3_adaptive_rag
from experiments.shared.data import (
    ROOT, experiment_paths, prepare_resources, read_json, read_jsonl, save_json, save_jsonl, sha256,
)
from experiments.shared.evaluation import summarize
from experiments.shared.llm import ChatLLM, SYSTEM_PROMPT, load_env
from experiments.shared.retrieval import build_embeddings


PIPELINES = {"m0": m0_knn, "m1": m1_llm_only, "m2": m2_fixed_rag, "m3": m3_adaptive_rag}


def load_config(path):
    config = read_json(path)
    load_env(ROOT / "experiments/.env")
    config["llm"]["base_url"] = os.environ.get("LLM_BASE_URL", "")
    config["llm"]["model"] = os.environ.get("LLM_MODEL", "")
    config["llm"]["api_key_env"] = "LLM_API_KEY"
    return config


def load_model_config(profile_name, config_path=None):
    config = load_config(config_path or ROOT / "experiments/config.json")
    profiles = read_json(ROOT / "experiments/models.json")
    profile = profiles[profile_name]
    if not profile.get("use_env"):
        config["llm"]["base_url"] = profile["base_url"]
        config["llm"]["model"] = profile["model"]
    for key, value in profile.items():
        if key not in {"display_name", "use_env", "base_url", "model"}:
            config["llm"][key] = value
    config["run_name"] = profile_name
    config["results_dir"] = f"results/{profile_name}"
    config["display_name"] = profile["display_name"]
    return config


def model_artifacts(config, artifacts):
    return artifacts / "model_runs" / config.get("run_name", "default")


def runtime_signature(config, artifacts):
    embedding_info = read_json(artifacts / "retrieval/embedding_config.json")
    if embedding_info.get("requested") != config["embedding"]:
        raise ValueError("Embedding configuration does not match the saved vectors")
    if embedding_info.get("manifest_sha256") != sha256(artifacts / "manifest.json"):
        raise ValueError("Embeddings do not belong to the current data manifest")
    llm_keys = ["provider", "base_url", "model", "temperature", "max_output_tokens",
                "output_token_parameter", "json_mode", "reasoning_effort"]
    experimental_config = {
        "data": config["data"],
        "embedding": config["embedding"],
        "representation_version": config["representation_version"],
        "llm": {key: config["llm"].get(key) for key in llm_keys},
    }
    payload = {"config": experimental_config, "system_prompt": SYSTEM_PROMPT,
               "manifest": sha256(artifacts / "manifest.json"),
               "embeddings": sha256(artifacts / "retrieval/embedding_config.json")}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def load_resources(config, split, pipeline):
    paths = experiment_paths(config)
    artifacts = paths["artifacts"]
    queries = read_jsonl(artifacts / f"queries/{split}.jsonl")
    resources = {}
    if pipeline == "m0":
        knn = artifacts / "knn"
        resources = {
            "train_vectors": np.load(knn / "train_vectors.npy", mmap_mode="r"),
            "train_labels": np.load(knn / "train_labels.npy", allow_pickle=False),
            "query_vectors": np.load(knn / f"{split}_vectors.npy"),
        }
    else:
        resources["llm"] = ChatLLM(config["llm"], model_artifacts(config, artifacts) / "llm_cache.jsonl")
        if pipeline in {"m2", "m3"}:
            retrieval = artifacts / "retrieval"
            resources["knowledge_base"] = read_jsonl(retrieval / "normal_metadata.jsonl")
            resources["similarities"] = np.clip(
                np.load(retrieval / f"{split}_embeddings.npy") @ np.load(retrieval / "normal_embeddings.npy").T,
                -1, 1,
            )
    return queries, resources


def save_run(rows, details, results_dir, split, pipeline, suffix=""):
    destination = results_dir / split
    tag = f"{pipeline}{suffix}"
    save_jsonl(destination / f"{tag}_predictions.jsonl", rows)
    save_json(destination / f"{tag}_summary.json", {**summarize(rows), **details})
    return read_json(destination / f"{tag}_summary.json")


def run_once(config, split, pipeline, pipeline_config=None, suffix=""):
    paths = experiment_paths(config)
    queries, resources = load_resources(config, split, pipeline)
    partial_path = paths["results"] / split / f"{pipeline}{suffix}_in_progress.jsonl"
    partial_stream = None
    if pipeline != "m0":
        partial_path.parent.mkdir(parents=True, exist_ok=True)
        partial_stream = partial_path.open("w", encoding="utf-8")
        def record_prediction(row):
            partial_stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            partial_stream.flush()
        resources["record_prediction"] = record_prediction
    try:
        effective_config = {"progress_every": config.get("progress_every", 10), **(pipeline_config or {})}
        output = PIPELINES[pipeline].run(queries, resources, effective_config)
        rows, details = output if pipeline == "m0" else (output, {})
        summary = save_run(rows, details, paths["results"], split, pipeline, suffix)
        print(
            f"Finished {pipeline.upper()} on {split}: F1={summary['f1']} | "
            f"predictions={paths['results'] / split / f'{pipeline}{suffix}_predictions.jsonl'}",
            flush=True,
        )
        if partial_stream is not None:
            partial_stream.close()
            partial_stream = None
            partial_path.unlink()
        return summary
    finally:
        if partial_stream is not None:
            partial_stream.close()
        if "llm" in resources:
            resources["llm"].close()


def tune_m3(config):
    paths = experiment_paths(config)
    lock = model_artifacts(config, paths["artifacts"]) / "selected_threshold.json"
    if lock.exists():
        raise FileExistsError("Threshold is already locked. Remove it only when starting a new experiment version.")
    retrieval = paths["artifacts"] / "retrieval"
    scores = np.clip(np.load(retrieval / "validation_embeddings.npy") @ np.load(retrieval / "normal_embeddings.npy").T, -1, 1)
    top10 = np.sort(scores, axis=1)[:, -10:]
    thresholds = np.quantile(top10.ravel(), [.25, .5, .75], method="linear")
    records = []
    for quantile, threshold in zip((25, 50, 75), thresholds):
        stats = run_once(config, "validation", "m3", {"threshold": float(threshold)}, f"_q{quantile}")
        records.append({"quantile": quantile, "threshold": float(threshold), **stats})
    if any(record["f1"] is None for record in records):
        raise ValueError("Validation contains INVALID_OUTPUT; inspect responses before locking a threshold")
    best = max(records, key=lambda row: (round(row["f1"], 6), -row["average_contexts"], -row["threshold"]))
    save_json(lock, {
        "threshold": best["threshold"], "source": f"validation_q{best['quantile']}",
        "validation_f1": best["f1"], "selected_on": "validation",
        "runtime_signature": runtime_signature(config, paths["artifacts"]),
    })
    save_json(paths["results"] / "validation/threshold_comparison.json", records)
    print(f"Locked M3 threshold={best['threshold']:.8f} from validation.")


def run_requested(config, pipeline, split):
    paths = experiment_paths(config)
    selected = ["m0", "m1", "m2", "m3"] if pipeline == "all" else [pipeline]
    if split == "validation" and pipeline in {"m3", "all"}:
        if pipeline == "all":
            for method in ("m0", "m1", "m2"):
                run_requested(config, method, split)
        tune_m3(config)
        return

    for method in selected:
        method_config = config.get(method, {}).copy()
        if method == "m0" and split == "test":
            selection = paths["results"] / "validation/m0_summary.json"
            if not selection.exists():
                run_once(config, "validation", "m0", config["m0"])
            method_config["selected_k"] = read_json(selection)["selected_k"]
        if method == "m3":
            lock = model_artifacts(config, paths["artifacts"]) / "selected_threshold.json"
            if not lock.exists():
                raise FileNotFoundError("Run M3 on validation to select and lock the threshold before test")
            selected_threshold = read_json(lock)
            if selected_threshold["runtime_signature"] != runtime_signature(config, paths["artifacts"]):
                raise ValueError("Threshold configuration does not match the current data/model/prompt")
            method_config["threshold"] = selected_threshold["threshold"]
        print(f"Running {method.upper()} on {split}...")
        run_once(config, split, method, method_config)


def report(config):
    results = experiment_paths(config)["results"] / "test"
    rows = []
    for path in sorted(results.glob("m?_summary.json")):
        rows.append({"method": path.stem.split("_")[0].upper(), **read_json(path)})
    if not rows:
        print("No test results yet.")
        return
    frame = pd.DataFrame(rows)
    frame.to_csv(results / "comparison.csv", index=False)
    columns = ["method", "precision", "recall", "f1", "average_contexts", "invalid_output_rate"]
    print(frame[columns].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "embed", "run", "report"])
    parser.add_argument("--pipeline", choices=["m0", "m1", "m2", "m3", "all"], default="all")
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--config", type=Path, default=ROOT / "experiments/config.json")
    parser.add_argument("--model-profile", choices=list(read_json(ROOT / "experiments/models.json")),
                        default="gemini")
    args = parser.parse_args()
    config = load_model_config(args.model_profile, args.config)
    paths = experiment_paths(config)
    if args.command == "prepare":
        prepare_resources(config)
    elif args.command == "embed":
        build_embeddings(config, paths["artifacts"])
    elif args.command == "run":
        run_requested(config, args.pipeline, args.split)
    else:
        report(config)


if __name__ == "__main__":
    main()
