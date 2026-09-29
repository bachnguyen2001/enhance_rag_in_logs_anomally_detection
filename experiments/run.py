"""Run the HDFS BlockId anomaly-detection experiment protocol.

This module is the orchestration layer only. The actual method logic lives in
`pipelines/`, while shared data, retrieval, structure, LLM, and evaluation
utilities live in `shared/`.

Main lifecycle:

1. prepare: build shared query manifests, KB metadata, and KNN artifacts.
2. embed: build semantic embedding files for KB/query splits.
3. run: execute M0-M5 on one split.
4. report: aggregate summaries into CSVs and confusion matrices.

Important design constraint: every method reads the same query manifest for a
split, so M0-M5 are compared on identical BlockId queries.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.pipelines import kb_rag, m0_knn, m1_llm_only
from experiments.shared.config import BASE, KB_VARIANTS, METHODS, load_config, model_config, shared_config
from experiments.shared.data import (
    experiment_paths, load_queries, prepare_resources, read_json, read_jsonl,
    save_json, save_jsonl, sha256, validate_prepared,
)
from experiments.shared.evaluation import summarize
from experiments.shared.llm import ChatLLM, SYSTEM_PROMPT
from experiments.shared.retrieval import build_embeddings, top_indices
from experiments.shared.structure import build_structure_scores

PROTOCOL = "full-kb-threshold-labeled-v2"
EXPERIMENT_CONFIG_KEYS = ("models", "methods", "sampling", "knn", "retrieval", "selection")


def digest(value):
    """Stable hash for configs/signatures used to protect experiment reproducibility."""
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def tag(method, kb=None):
    return f"{method}_{kb}" if kb else method


def versioned_comparison_paths(output, split="test"):
    """Return the next non-existing comparison CSV/config pair for a split."""
    version = 1
    prefix = "comparison" if split == "test" else f"comparison_{split}"
    while True:
        suffix = "" if version == 1 else f"_v{version}"
        csv_path = output / f"{prefix}{suffix}.csv"
        config_path = output / f"{prefix}{suffix}_config.json"
        if not csv_path.exists():
            return csv_path, config_path, version
        version += 1


def experiment_config_snapshot(config):
    """Keep only research-design settings in the report sidecar config."""
    return {key: config[key] for key in EXPERIMENT_CONFIG_KEYS if key in config}


def experiment_id(row):
    parts = [str(row["model_profile"]), str(row["method"])]
    if row.get("kb") is not None and pd.notna(row.get("kb")):
        parts.append(str(row["kb"]))
    if row.get("context_setting") is not None and pd.notna(row.get("context_setting")):
        parts.append(str(row["context_setting"]))
    return "_".join(parts).replace("/", "-")


def save_confusion_matrices(frame, output, split):
    """Export one confusion matrix CSV/PNG per reported experiment row."""
    folder = output / "confusion_matrices" / split
    folder.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        plt = None
    for row in frame.to_dict("records"):
        name = experiment_id(row)
        matrix = pd.DataFrame(
            [[row.get("TN", 0), row.get("FP", 0)], [row.get("FN", 0), row.get("TP", 0)]],
            index=["True NORMAL", "True ANOMALY"],
            columns=["Pred NORMAL", "Pred ANOMALY"],
        )
        matrix.to_csv(folder / f"{name}.csv")
        if plt is None:
            continue
        fig, ax = plt.subplots(figsize=(4, 3.4))
        image = ax.imshow(matrix.values, cmap="Blues")
        ax.set_xticks([0, 1], labels=matrix.columns)
        ax.set_yticks([0, 1], labels=matrix.index)
        ax.set_title(name)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(int(matrix.values[i, j])), ha="center", va="center", color="black")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(folder / f"{name}.png", dpi=160)
        plt.close(fig)


def parameters(config, method, kb=None, threshold=None):
    """Build the effective runtime parameters for one method/KB execution."""
    result = {"progress_every": config["progress_every"]}
    if method == "m0":
        return {**result, **config["knn"]}
    if method == "m1":
        return result
    result.update(method_id=method, kb_variant=kb, **METHODS[method])
    result.update(config["selection"][result["selection"]])
    if result["selection"] == "adaptive":
        result["context_setting"] = "adaptive"
    if result["retrieval"] == "structure":
        structure = config["retrieval"]["structure_aware"]
        result["weights"] = [structure[p]["weight"] for p in ("semantic", "template", "sequence")]
    if threshold is not None:
        result["threshold"] = threshold
    return result


def signature(config, method, kb=None):
    """Hash every input that should invalidate/recreate a result file."""
    artifacts = experiment_paths(config)["artifacts"]
    payload = {"protocol": PROTOCOL, "manifest": sha256(artifacts / "manifest.json"),
               "method": method, "kb": kb}
    if method == "m0":
        payload["knn"] = config["knn"]
    else:
        keys = ("provider", "base_url", "chat_path", "payload_format", "model", "temperature", "max_output_tokens",
                "output_token_parameter", "json_mode", "reasoning_effort")
        payload["llm"] = {key: config["llm"].get(key) for key in keys}
        payload["prompt"] = SYSTEM_PROMPT
        if method in METHODS:
            payload["retrieval"] = config["retrieval"]
            payload["selection"] = config["selection"]
            payload["embedding"] = sha256(artifacts / "retrieval/embedding_config.json")
    return digest(payload)


class ScoreRows:
    """Score the complete KB one query at a time, including for full-size test."""

    def __init__(self, queries, kb, query_vectors, kb_vectors, weights=None):
        self.queries, self.kb = queries, kb
        self.query_vectors, self.kb_vectors, self.weights = query_vectors, kb_vectors, weights

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, index):
        cosine = np.clip(self.kb_vectors @ self.query_vectors[index], -1, 1)
        if self.weights is None:
            return cosine
        return build_structure_scores(
            [self.queries[index]], self.kb, cosine[None, :], self.weights,
        )[0]


def load_scores(config, split, method, kb):
    """Load query/KB embeddings and expose lazy per-query retrieval scores."""
    artifacts = experiment_paths(config)["artifacts"]
    queries = load_queries(config, split)
    folder = artifacts / "retrieval"
    info = read_json(folder / "embedding_config.json")
    if info["manifest_sha256"] != sha256(artifacts / "manifest.json") or info["requested"] != config["embedding"]:
        raise ValueError("Embeddings do not match data/config; run embed with matching artifacts")
    for name in (kb, split):
        if sha256(folder / f"{name}_embeddings.npy") != info["vector_hashes"][name]:
            raise ValueError(f"Embedding changed: {name}")
    manifest = validate_prepared(config)
    metadata_path = f"retrieval/{kb}_metadata.jsonl"
    if sha256(artifacts / metadata_path) != manifest["files"][metadata_path]:
        raise ValueError(f"KB metadata changed: {kb}")
    metadata = read_jsonl(artifacts / metadata_path)
    query_vectors = np.load(folder / f"{split}_embeddings.npy", mmap_mode="r")
    kb_vectors = np.load(folder / f"{kb}_embeddings.npy", mmap_mode="r")
    if len(query_vectors) != len(queries) or len(kb_vectors) != len(metadata):
        raise ValueError("Embedding rows do not match query/KB manifest")
    if not metadata:
        raise ValueError(f"Empty KB: {kb}")
    fixed_sweep = config["selection"]["fixed"].get("contexts_sweep", [config["selection"]["fixed"].get("contexts")])
    if method in ("m2", "m4") and len(metadata) < max(fixed_sweep):
        raise ValueError("Fixed context count exceeds KB size")
    weights = parameters(config, method, kb).get("weights")
    return queries, metadata, ScoreRows(queries, metadata, query_vectors, kb_vectors, weights)


def load_resources(config, split, method, kb=None):
    """Load only the resources needed by one method on one split."""
    artifacts = experiment_paths(config)["artifacts"]
    queries = load_queries(config, split)
    resources = {}
    if method == "m0":
        manifest = validate_prepared(config)
        for name in ("train_vectors", "train_labels", f"{split}_vectors"):
            relative = f"knn/{name}.npy"
            if sha256(artifacts / relative) != manifest["files"][relative]:
                raise ValueError(f"KNN artifact changed: {name}")
        relative = "knn/feature_config.json"
        if sha256(artifacts / relative) != manifest["files"][relative]:
            raise ValueError("KNN artifact changed: feature_config")
        resources = {
            "train_vectors": np.load(artifacts / "knn/train_vectors.npy", mmap_mode="r"),
            "train_labels": np.load(artifacts / "knn/train_labels.npy"),
            "query_vectors": np.load(artifacts / f"knn/{split}_vectors.npy"),
        }
        if max(config["knn"]["k_candidates"]) > len(resources["train_labels"]):
            raise ValueError("KNN k exceeds training set size")
        return queries, resources
    if method in METHODS:
        queries, metadata, scores = load_scores(config, split, method, kb)
        resources.update(knowledge_base=metadata, scores=scores)
    cache = BASE / config["llm_cache_dir"] / config["run_name"] / "llm_cache.jsonl"
    resources["llm"] = ChatLLM(config["llm"], cache)
    return queries, resources


def check_prediction_identity(rows, queries):
    """Guardrail: predictions must preserve the shared query order and labels."""
    if [(r["block_id"], r["label"]) for r in rows] != [(r["block_id"], r["label"]) for r in queries]:
        raise ValueError("Predictions do not match the shared query manifest (IDs/order/labels)")


def run_once(config, split, method, kb=None, extra=None, suffix=""):
    """Run one concrete experiment and cache/skip it by configuration signature."""
    paths = experiment_paths(config)
    queries = load_queries(config, split)
    effective = {**parameters(config, method, kb), **(extra or {})}
    run_signature = digest({
        "signature": signature(config, method, kb),
        "parameters": {key: value for key, value in effective.items() if key != "progress_every"},
    })
    destination = paths["results"] / split
    signature_id = run_signature[:12]
    name = f"{tag(method, kb)}_{signature_id}{suffix}"
    summary_path = destination / f"{name}_summary.json"
    prediction_path = destination / f"{name}_predictions.jsonl"
    query_hash = sha256(paths["artifacts"] / f"queries/{split}.jsonl")
    # Existing results are reused only when both data manifest and effective
    # method parameters match the current run signature.
    if summary_path.exists():
        previous = read_json(summary_path)
        if previous.get("run_signature") != run_signature:
            raise ValueError(f"Configuration changed for {name}; choose a new results_dir")
        if previous.get("f1") is not None and prediction_path.exists():
            check_prediction_identity(read_jsonl(prediction_path), queries)
            print(f"[SKIP] {name} {split}: same manifest and configuration", flush=True)
            return previous
    queries, resources = load_resources(config, split, method, kb)
    destination.mkdir(parents=True, exist_ok=True)
    partial = destination / f"{name}_in_progress.jsonl"
    try:
        with partial.open("w", encoding="utf-8") as stream:
            def record(row):
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
            resources["record_prediction"] = record
            if method == "m0":
                rows, details = m0_knn.run(queries, resources, effective)
            elif method == "m1":
                rows, details = m1_llm_only.run(queries, resources, effective), {}
            else:
                rows, details = kb_rag.run(queries, resources, effective), {}
        check_prediction_identity(rows, queries)
        summary = {**summarize(rows), **details, "method": method, "kb": kb,
                   "model_profile": config["run_name"], "run_signature": run_signature,
                   "query_manifest_sha256": query_hash, "parameters": effective}
        save_jsonl(prediction_path, rows)
        save_json(summary_path, summary)
        partial.unlink()
        print(f"Finished {name} {split}: F1={summary['f1']}, queries={len(rows)}", flush=True)
        return summary
    finally:
        if "llm" in resources:
            resources["llm"].close()


def threshold_path(config, method, kb):
    """Location of the validation-locked adaptive threshold for method/KB."""
    signature_id = signature(config, method, kb)[:12]
    return (experiment_paths(config)["artifacts"] / "model_runs" / config["run_name"]
            / f"selected_threshold_{tag(method, kb)}_{signature_id}.json")


def read_threshold(config, method, kb):
    """Read a previously selected adaptive threshold and verify its signature."""
    path = threshold_path(config, method, kb)
    if not path.exists():
        raise FileNotFoundError(f"Run validation first: missing {path.name}")
    lock = read_json(path)
    if lock.get("signature") != signature(config, method, kb):
        raise ValueError(f"Threshold configuration changed for {tag(method, kb)}; use a new artifacts_dir")
    return lock["threshold"]


def tune_threshold(config, method, kb):
    """Select adaptive threshold on validation top-candidate scores only."""
    if threshold_path(config, method, kb).exists():
        return read_threshold(config, method, kb)
    queries, metadata, scores = load_scores(config, "validation", method, kb)
    if {q["label"] for q in queries} != {"NORMAL", "ANOMALY"}:
        raise ValueError("Random validation has only one class; increase validation_size and use a new artifacts_dir")
    # Adaptive selection is tuned from the same top-N candidate pool later used
    # at inference time. We try several validation quantiles and lock the best.
    pool_size = min(config["selection"]["adaptive"]["candidate_pool"], len(metadata))
    pooled = np.empty((len(queries), pool_size), dtype=np.float32)
    for index in range(len(queries)):
        row = scores[index]
        pooled[index] = row[top_indices(row, pool_size)]
    quantiles = config["selection"]["adaptive"]["threshold_quantiles"]
    thresholds = np.quantile(pooled, quantiles, method="linear")
    records = []
    for index, (quantile, threshold) in enumerate(zip(quantiles, thresholds)):
        stats = run_once(config, "validation", method, kb,
                         {"threshold": float(threshold)}, suffix=f"_q{index}")
        records.append({**stats, "quantile": quantile, "threshold": float(threshold)})
    if any(row["f1"] is None for row in records):
        raise ValueError("INVALID_OUTPUT on validation; threshold was not locked")
    best = max(records, key=lambda row: (round(row["f1"], 6), -row["average_contexts"], -row["threshold"]))
    save_json(threshold_path(config, method, kb), {
        "threshold": best["threshold"], "signature": signature(config, method, kb),
        "selected_on": "validation", "score_population": "top-" + str(pool_size) + " query-candidate pairs",
        "validation_f1": best["f1"], "quantile": best["quantile"],
    })
    save_json(experiment_paths(config)["results"] / "validation" / f"{tag(method, kb)}_threshold_comparison.json", records)
    return best["threshold"]


def run_requested(config, split):
    """Execute all configured methods for a split under the shared protocol."""
    validate_prepared(config, verify_all=True)
    methods = config["methods"]
    if "m0" in methods:
        shared = shared_config(config)
        # k is always selected on the shared validation queries.
        validation = run_once(shared, "validation", "m0")
        if split == "test":
            run_once(shared, "test", "m0", extra={"selected_k": validation["selected_k"]})
    for name in config["models"]:
        current = model_config(config, name)
        for method in methods:
            if method == "m0":
                continue
            if method == "m1":
                run_once(current, split, method)
                continue
            for kb in KB_VARIANTS:
                if METHODS[method]["selection"] == "fixed":
                    fixed_sweep = config["selection"]["fixed"].get("contexts_sweep", [config["selection"]["fixed"].get("contexts")])
                    for contexts in fixed_sweep:
                        run_once(current, split, method, kb, {
                            "contexts": contexts,
                            "context_setting": f"fixed_k{contexts}",
                        })
                    continue
                extra = {}
                if METHODS[method]["selection"] == "adaptive":
                    threshold = (tune_threshold(current, method, kb) if split == "validation"
                                 else read_threshold(current, method, kb))
                    extra["threshold"] = threshold
                run_once(current, split, method, kb, extra)


def report(config, split="test"):
    """Aggregate completed runs into a comparison table and confusion matrices."""
    load_queries(config, split)
    expected_hash = sha256(experiment_paths(config)["artifacts"] / f"queries/{split}.jsonl")
    rows = []
    jobs = [(shared_config(config), "m0", None)] if "m0" in config["methods"] else []
    for name in config["models"]:
        current = model_config(config, name)
        for method in config["methods"]:
            if method == "m0":
                continue
            jobs.extend((current, method, kb) for kb in (KB_VARIANTS if method in METHODS else (None,)))
    for current, method, kb in jobs:
        base_signature = signature(current, method, kb)
        for path in sorted((experiment_paths(current)["results"] / split).glob(f"{tag(method, kb)}_*_summary.json")):
            if path.stem.endswith(("_q0_summary", "_q1_summary", "_q2_summary")):
                continue
            row = read_json(path)
            parameters_without_progress = {
                key: value for key, value in row.get("parameters", {}).items()
                if key != "progress_every"
            }
            expected_run_signature = digest({"signature": base_signature, "parameters": parameters_without_progress})
            if row.get("run_signature") != expected_run_signature:
                continue
            if row["query_manifest_sha256"] != expected_hash:
                raise ValueError("Cannot compare results from different test query manifests")
            rows.append({key: row.get(key) for key in (
                "model_profile", "method", "kb", "queries", "Normal", "Anomaly",
                "TP", "TN", "FP", "FN", "precision", "recall", "f1",
                "average_contexts", "average_prompt_tokens", "average_latency_seconds_uncached",
                "average_query_seconds_uncached",
                "query_manifest_sha256",
            )} | {
                "retrieval": METHODS[method]["retrieval"] if method in METHODS else None,
                "selection": METHODS[method]["selection"] if method in METHODS else None,
                "context_setting": row.get("parameters", {}).get("context_setting"),
                "contexts": row.get("parameters", {}).get("contexts"),
            })
    if rows:
        output = experiment_paths(config)["results"]
        output.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(rows)
        # Compare adaptive methods against both the largest fixed-k and the best
        # observed fixed-k for the same model/retrieval/KB setting.
        fixed_sweep = config["selection"]["fixed"].get("contexts_sweep", [config["selection"]["fixed"].get("contexts")])
        reference_contexts = max(fixed_sweep)
        frame["context_reduction_vs_fixed_contexts"] = frame.apply(
            lambda row: (1 - row.average_contexts / reference_contexts)
            if row.selection == "adaptive" and pd.notna(row.average_contexts) else 0.0,
            axis=1,
        )
        fixed_f1 = {
            (row.model_profile, row.retrieval, row.kb): row.f1
            for row in frame.itertuples()
            if row.selection == "fixed" and row.contexts == reference_contexts
        }
        best_fixed = {}
        for row in frame.itertuples():
            if row.selection != "fixed" or pd.isna(row.f1):
                continue
            key = (row.model_profile, row.retrieval, row.kb)
            current = best_fixed.get(key)
            if current is None or (row.f1, -row.contexts) > (current["f1"], -current["contexts"]):
                best_fixed[key] = {"f1": row.f1, "contexts": row.contexts, "context_setting": row.context_setting}
        frame["fixed_kmax_f1"] = frame.apply(
            lambda row: fixed_f1.get((row.model_profile, row.retrieval, row.kb))
            if row.selection == "adaptive" else None,
            axis=1,
        )
        frame["delta_f1_vs_fixed_kmax"] = frame.apply(
            lambda row: row.f1 - row.fixed_kmax_f1
            if row.selection == "adaptive" and pd.notna(row.fixed_kmax_f1) and pd.notna(row.f1) else None,
            axis=1,
        )
        frame["best_fixed_f1"] = frame.apply(
            lambda row: best_fixed.get((row.model_profile, row.retrieval, row.kb), {}).get("f1")
            if row.selection == "adaptive" else None,
            axis=1,
        )
        frame["best_fixed_context_setting"] = frame.apply(
            lambda row: best_fixed.get((row.model_profile, row.retrieval, row.kb), {}).get("context_setting")
            if row.selection == "adaptive" else None,
            axis=1,
        )
        frame["delta_f1_vs_best_fixed"] = frame.apply(
            lambda row: row.f1 - row.best_fixed_f1
            if row.selection == "adaptive" and pd.notna(row.best_fixed_f1) and pd.notna(row.f1) else None,
            axis=1,
        )
        frame["confusion_matrix"] = frame.apply(
            lambda row: f"TN={int(row.TN)} FP={int(row.FP)} FN={int(row.FN)} TP={int(row.TP)}",
            axis=1,
        )
        frame["method_order"] = frame["method"].map({"m0": 0, "m1": 1, "m2": 2, "m3": 3, "m4": 4, "m5": 5}).fillna(99)
        frame["kb_order"] = frame["kb"].map({"normal": 0, "anomaly": 1, "mixed": 2}).fillna(-1)
        frame["context_order"] = frame["contexts"].fillna(-1)
        frame = frame.sort_values(["method_order", "kb_order", "context_order", "context_setting"]).reset_index(drop=True)
        frame["best_in_method_kb"] = frame.groupby(["model_profile", "method", "kb"], dropna=False)["f1"].transform(lambda col: col == col.max())
        frame["best_overall"] = frame["f1"] == frame["f1"].max()
        report_columns = [
            "model_profile", "method", "retrieval", "selection", "kb", "context_setting", "contexts",
            "queries", "Normal", "Anomaly", "precision", "recall", "f1",
            "TP", "TN", "FP", "FN", "confusion_matrix",
            "average_contexts", "context_reduction_vs_fixed_contexts",
            "fixed_kmax_f1", "delta_f1_vs_fixed_kmax",
            "best_fixed_context_setting", "best_fixed_f1", "delta_f1_vs_best_fixed",
            "best_in_method_kb", "best_overall",
            "average_prompt_tokens", "average_latency_seconds_uncached", "average_query_seconds_uncached",
            "query_manifest_sha256",
        ]
        report_columns = [column for column in report_columns if column in frame.columns]
        save_confusion_matrices(frame, output, split)
        csv_path, config_path, version = versioned_comparison_paths(output, split)
        experiment_config = experiment_config_snapshot(config)
        frame[report_columns].to_csv(csv_path, index=False)
        save_json(config_path, {
            "comparison_file": csv_path.name,
            "split": split,
            "version": version,
            "experiment_config_sha256": digest(experiment_config),
            "experiment_config": experiment_config,
        })
        print(f"Saved {csv_path.name} with {config_path.name}", flush=True)
        print(f"Saved confusion matrices to {(output / 'confusion_matrices' / split).as_posix()}", flush=True)
        print(frame[["model_profile", "method", "kb", "context_setting", "queries", "f1", "average_contexts", "best_in_method_kb"]].to_string(index=False))
    else:
        print("No test results yet.")


def main():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["all", "prepare", "embed", "run", "report"], nargs="?", default="all")
    parser.add_argument("--system-config", type=Path, default=BASE / "configs" / "config_system.json")
    parser.add_argument("--experiment-config", type=Path, default=BASE / "configs" / "config_experiment.json")
    parser.add_argument("--model-profile", help="Override models with one profile")
    parser.add_argument("--pipeline", choices=["all", "m0", "m1", *METHODS], default="all")
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    args = parser.parse_args()
    config = load_config(args.system_config, args.experiment_config)
    if args.model_profile:
        if args.model_profile not in config["model_profiles"]:
            parser.error("Unknown model profile")
        config["models"] = [args.model_profile]
    if args.pipeline != "all":
        config["methods"] = [args.pipeline]
    if args.command in {"all", "prepare"}:
        prepare_resources(config)
    if args.command in {"all", "embed"}:
        build_embeddings(config, experiment_paths(config)["artifacts"])
    if args.command == "all":
        run_requested(config, "validation")
        run_requested(config, "test")
        report(config)
    elif args.command == "run":
        run_requested(config, args.split)
    elif args.command == "report":
        report(config, args.split)


if __name__ == "__main__":
    main()
