"""System/experiment configuration and the fixed M0-M5 method definitions."""

from copy import deepcopy
import math
import os
from pathlib import Path

from experiments_v2.shared.data import ROOT, read_json
from experiments_v2.shared.llm import load_env

BASE = ROOT / "experiments_v2"
KB_VARIANTS = ("normal", "anomaly", "mixed")
METHODS = {
    "m2": {"retrieval": "semantic", "selection": "fixed"},
    "m3": {"retrieval": "semantic", "selection": "adaptive"},
    "m4": {"retrieval": "structure", "selection": "fixed"},
    "m5": {"retrieval": "structure", "selection": "adaptive"},
}


def positive_integer(value):
    return type(value) is int and value > 0


def valid_sample_spec(value):
    if value == "all" or positive_integer(value):
        return True
    if type(value) is dict:
        return (positive_integer(value.get("normal")) and
                positive_integer(value.get("anomaly")))
    return False


def load_config(system_path=BASE / "config_system.json", experiment_path=BASE / "config_experiment.json"):
    system, experiment = read_json(system_path), read_json(experiment_path)
    if set(system) & set(experiment):
        raise ValueError("System and experiment config must not contain overlapping keys")
    config = {**system, **experiment}
    validate_config(config)
    return config


def validate_config(config):
    for key, allowed in (("methods", {"m0", "m1", *METHODS}),
                         ("models", set(config["model_profiles"]))):
        values = config[key]
        if not values or len(values) != len(set(values)) or not set(values) <= allowed:
            raise ValueError(f"Invalid {key}: {values}")
    sampling = config["sampling"]
    if type(sampling["seed"]) is not int:
        raise ValueError("sampling.seed must be an integer")
    for split in ("validation", "test"):
        value = sampling.get(split, sampling.get(f"{split}_size"))
        if not valid_sample_spec(value):
            raise ValueError(f"sampling.{split} must be a positive integer, all, or normal/anomaly counts")
    knn = config["knn"]
    ks = knn["k_candidates"]
    if not ks or not all(positive_integer(k) for k in ks):
        raise ValueError("knn.k_candidates must contain positive integers")
    if knn.get("feature") != "sequence_ngram_hash":
        raise ValueError("knn.feature must be sequence_ngram_hash")
    ngram_range = knn.get("ngram_range")
    if (type(ngram_range) is not list or len(ngram_range) != 2 or
            not all(positive_integer(n) for n in ngram_range) or ngram_range[1] < ngram_range[0]):
        raise ValueError("knn.ngram_range must be [min_n, max_n]")
    if not positive_integer(knn.get("hash_dim")):
        raise ValueError("knn.hash_dim must be a positive integer")
    if type(knn.get("binary", False)) is not bool:
        raise ValueError("knn.binary must be a boolean")
    retrieval = config["retrieval"]
    structure = retrieval["structure_aware"]
    if retrieval["semantic"] != {"similarity": "cosine"}:
        raise ValueError("Semantic retrieval currently supports cosine only")
    for part, metric in (("semantic", "cosine"), ("template", "jaccard"), ("sequence", "normalized_lcs")):
        if structure[part]["similarity"] != metric:
            raise ValueError(f"Unsupported structure metric for {part}")
    if structure["semantic"]["normalization"] != "cosine_to_01":
        raise ValueError("Structure semantic normalization must be cosine_to_01")
    weights = [structure[p]["weight"] for p in ("semantic", "template", "sequence")]
    if not all(math.isfinite(w) and w >= 0 for w in weights) or not math.isclose(sum(weights), 1):
        raise ValueError("Structure weights must be non-negative and sum to 1")
    selection = config["selection"]
    fixed_sweep = selection["fixed"].get("contexts_sweep", [selection["fixed"].get("contexts")])
    if (not fixed_sweep or len(fixed_sweep) != len(set(fixed_sweep)) or
            not all(positive_integer(value) for value in fixed_sweep)):
        raise ValueError("selection.fixed.contexts_sweep must contain unique positive integers")
    if not positive_integer(selection["adaptive"].get("candidate_pool")):
        raise ValueError("selection.adaptive.candidate_pool must be a positive integer")
    quantiles = selection["adaptive"]["threshold_quantiles"]
    if not quantiles or not all(math.isfinite(q) and 0 <= q <= 1 for q in quantiles):
        raise ValueError("threshold_quantiles must be within [0, 1]")


def model_config(config, name):
    result = deepcopy(config)
    profile = result["model_profiles"][name]
    load_env(BASE / result["env_file"])
    result["llm"].update({k: v for k, v in profile.items() if k not in {"display_name", "use_env"}})
    if profile.get("use_env"):
        result["llm"]["base_url"] = os.environ.get("LLM_BASE_URL", "")
        result["llm"]["model"] = os.environ.get("LLM_MODEL", "")
    result["run_name"] = name
    result["display_name"] = profile["display_name"]
    result["results_dir"] = str(Path(config["results_dir"]) / name)
    return result


def shared_config(config):
    result = deepcopy(config)
    result["run_name"] = "shared"
    result["results_dir"] = str(Path(config["results_dir"]) / "shared")
    return result
