"""Read/write experiment data and build the fixed query resources."""

from itertools import groupby
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def save_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize(vectors):
    vectors = np.asarray(vectors, dtype=np.float32)
    return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)




def ngram_hash_vectors(sequences, config):
    """Dense hashed n-gram vectors for ordered EventId sequences."""
    knn = config["knn"]
    ngram_range = knn.get("ngram_range", [2, 3])
    if len(ngram_range) != 2:
        raise ValueError("knn.ngram_range must be [min_n, max_n]")
    min_n, max_n = ngram_range
    hash_dim = int(knn.get("hash_dim", 1024))
    if min_n <= 0 or max_n < min_n or hash_dim <= 0:
        raise ValueError("Invalid KNN n-gram configuration")
    vectors = np.zeros((len(sequences), hash_dim), dtype=np.float32)
    binary = bool(knn.get("binary", False))
    for row, events in enumerate(sequences):
        used_fallback = True
        for n in range(min_n, max_n + 1):
            if len(events) < n:
                continue
            used_fallback = False
            for start in range(0, len(events) - n + 1):
                gram = "\x1f".join(events[start:start + n]).encode()
                column = int.from_bytes(hashlib.blake2b(gram, digest_size=8).digest(), "big") % hash_dim
                vectors[row, column] = 1.0 if binary else vectors[row, column] + 1.0
        if used_fallback:
            for event in events:
                column = int.from_bytes(hashlib.blake2b(event.encode(), digest_size=8).digest(), "big") % hash_dim
                vectors[row, column] = 1.0 if binary else vectors[row, column] + 1.0
    return normalize(vectors)

def render_trace(events, templates):
    """Keep order and consecutive repetition; never include labels or Type."""
    definitions = "\n".join(f"{event}: {templates[event]}" for event in dict.fromkeys(events))
    runs = [(event, len(list(items))) for event, items in groupby(events)]
    ordered = " → ".join(event if count == 1 else f"{event} × {count}" for event, count in runs)
    return f"EVENT DEFINITIONS:\n{definitions}\n\nORDERED TRACE:\n{ordered}"


def experiment_paths(config):
    base = ROOT / "experiments_v2"
    return {
        "base": base,
        "artifacts": base / config.get("artifacts_dir", "artifacts"),
        "results": base / config.get("results_dir", "results"),
    }


def _sample(source, size, seed):
    """Sample BlockIds from stable source ordering."""
    source = source.sort_values("block_id").reset_index(drop=True)
    if source.block_id.duplicated().any():
        raise ValueError("Duplicate BlockId in source split")
    if size == "all":
        return source
    if type(size) is dict:
        frames = []
        for offset, (label_key, label_value) in enumerate((("normal", "Normal"), ("anomaly", "Anomaly"))):
            count = size.get(label_key)
            subset = source.loc[source.label == label_value].reset_index(drop=True)
            if type(count) is not int or not 0 < count <= len(subset):
                raise ValueError(f"Invalid {label_key} sample size {count} for {len(subset)} available queries")
            indices = random.Random(seed + offset).sample(range(len(subset)), count)
            frames.append(subset.iloc[indices])
        return pd.concat(frames).sort_values("block_id").reset_index(drop=True)
    if type(size) is not int or not 0 < size <= len(source):
        raise ValueError(f"Invalid sample size {size} for {len(source)} available queries")
    indices = random.Random(seed).sample(range(len(source)), size)
    return source.iloc[indices].reset_index(drop=True)


def preparation_spec(config):
    return {key: config[key] for key in ("data", "representation_version", "sampling", "knn")}


def validate_prepared(config, verify_all=False):
    artifacts = experiment_paths(config)["artifacts"]
    manifest = read_json(artifacts / "manifest.json")
    if manifest.get("preparation") != preparation_spec(config):
        raise ValueError("Data/sampling changed. Choose a new artifacts_dir and run prepare; existing queries are locked.")
    if verify_all:
        for relative, expected in manifest["files"].items():
            if sha256(artifacts / relative) != expected:
                raise ValueError(f"Prepared artifact modified: {relative}")
    return manifest


def load_queries(config, split):
    """Every method/model loads this same immutable manifest; never samples here."""
    artifacts = experiment_paths(config)["artifacts"]
    manifest = validate_prepared(config)
    relative = f"queries/{split}.jsonl"
    if sha256(artifacts / relative) != manifest["files"][relative]:
        raise ValueError(f"Query manifest modified: {relative}")
    rows = read_jsonl(artifacts / relative)
    if len(rows) != manifest["query_counts"][split] or len({r["block_id"] for r in rows}) != len(rows):
        raise ValueError("Query manifest count/identity mismatch")
    return rows


def prepare_resources(config):
    """Create query resources and three label-aware train KBs without loading all traces."""
    paths = experiment_paths(config)
    artifacts = paths["artifacts"]
    manifest_path = artifacts / "manifest.json"
    if manifest_path.exists():
        validate_prepared(config, verify_all=True)
        print("Experiment artifacts already prepared; manifest was not changed.")
        return

    split_dir = ROOT / config["data"]["split_dir"]
    source_paths = {
        **{name: ROOT / config["data"][name] for name in ("traces", "templates", "occurrence_matrix")},
        **{split: split_dir / f"{split}_block_ids.csv" for split in ("train", "validation", "test")},
    }
    sampling = config["sampling"]
    full_splits = {
        split: pd.read_csv(split_dir / f"{split}_block_ids.csv")
        for split in ("train", "validation", "test")
    }
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        for column in ("block_id", "trace_signature"):
            if set(full_splits[left][column]) & set(full_splits[right][column]):
                raise ValueError(f"{column} overlap between {left} and {right}")
    frames = {
        split: _sample(
            full_splits[split], sampling.get(split, sampling.get(f"{split}_size")), sampling["seed"],
        )
        for split in ("validation", "test")
    }
    if set(frames["validation"].block_id) & set(frames["test"].block_id):
        raise ValueError("Validation and test BlockIds overlap")

    templates = pd.read_csv(ROOT / config["data"]["templates"]).set_index("EventId").EventTemplate.to_dict()
    train = full_splits["train"].sort_values("block_id")
    if train.block_id.duplicated().any() or not set(train.label) <= {"Normal", "Anomaly"}:
        raise ValueError("Invalid train IDs or labels")
    train_labels = dict(zip(train.block_id, train.label.str.upper()))
    train_ids = set(train_labels)
    query_targets = set().union(*(set(frame.block_id) for frame in frames.values()))
    query_sequences = {}
    grouped = {}
    seen_traces = set()
    traces_path = ROOT / config["data"]["traces"]
    for chunk in pd.read_csv(traces_path, usecols=["BlockId", "Features"], chunksize=20000):
        relevant = chunk.loc[chunk.BlockId.isin(train_ids | query_targets)]
        for row in relevant.itertuples(index=False):
            if row.BlockId in seen_traces:
                raise ValueError(f"Duplicate trace BlockId: {row.BlockId}")
            seen_traces.add(row.BlockId)
            events = tuple(event.strip() for event in row.Features[1:-1].split(","))
            if not events or any(event not in templates for event in events):
                raise ValueError(f"Invalid events for {row.BlockId}")
            if row.BlockId in query_targets:
                query_sequences[row.BlockId] = list(events)
            label = train_labels.get(row.BlockId)
            if label:
                key = (events, label)
                if key not in grouped:
                    grouped[key] = {"count": 0, "representative": row.BlockId}
                grouped[key]["count"] += 1
    if seen_traces != train_ids | query_targets:
        raise ValueError("Missing train/query sequences")

    query_dir = artifacts / "queries"
    for split, frame in frames.items():
        rows = [{
            "block_id": row.block_id,
            "label": row.label.upper(),
            "trace_signature": row.trace_signature,
            "event_ids": query_sequences[row.block_id],
            "text": render_trace(query_sequences[row.block_id], templates),
        } for row in frame.itertuples(index=False)]
        save_jsonl(query_dir / f"{split}.jsonl", rows)
        frame.to_csv(query_dir / f"{split}_queries.csv", index=False)

    retrieval_dir = artifacts / "retrieval"
    retrieval_dir.mkdir(parents=True, exist_ok=True)
    kb_rows = {"normal": [], "anomaly": []}
    labels_by_events = {}
    for events, label in grouped:
        labels_by_events.setdefault(events, set()).add(label)
    for (events, label), details in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0])):
        prefix = "normal" if label == "NORMAL" else "anomaly"
        trace_id = f"{prefix}_{hashlib.sha1(json.dumps(events).encode()).hexdigest()[:16]}"
        kb_rows[prefix].append({
            "trace_id": trace_id, "event_ids": list(events),
            "text": render_trace(events, templates), "label": label,
            "occurrence_count": details["count"],
            "representative_block_id": details["representative"],
            "conflicting_sequence": len(labels_by_events[events]) > 1,
        })
    save_jsonl(retrieval_dir / "normal_metadata.jsonl", kb_rows["normal"])
    save_jsonl(retrieval_dir / "anomaly_metadata.jsonl", kb_rows["anomaly"])
    save_jsonl(retrieval_dir / "mixed_metadata.jsonl", kb_rows["normal"] + kb_rows["anomaly"])

    knn_dir = artifacts / "knn"
    knn_dir.mkdir(parents=True, exist_ok=True)
    train_items = sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0]))
    train_sequences = [list(events) for (events, _label), _details in train_items]
    train_knn_labels = np.array([label for (_events, label), _details in train_items], dtype="U7")
    np.save(knn_dir / "train_vectors.npy", ngram_hash_vectors(train_sequences, config))
    np.save(knn_dir / "train_labels.npy", train_knn_labels)
    pd.DataFrame([{
        "trace_id": f"{'normal' if label == 'NORMAL' else 'anomaly'}_"
                    f"{hashlib.sha1(json.dumps(events).encode()).hexdigest()[:16]}",
        "label": label,
        "occurrence_count": details["count"],
        "representative_block_id": details["representative"],
    } for (events, label), details in train_items]).to_csv(knn_dir / "train_ids.csv", index=False)
    for split, frame in frames.items():
        sequences = [query_sequences[block_id] for block_id in frame.block_id]
        np.save(knn_dir / f"{split}_vectors.npy", ngram_hash_vectors(sequences, config))
    save_json(knn_dir / "feature_config.json", config["knn"])

    files = sorted(path for path in artifacts.rglob("*") if path.is_file())
    save_json(manifest_path, {
        "preparation": preparation_spec(config),
        "source_hashes": {name: sha256(path) for name, path in source_paths.items()},
        "query_counts": {split: len(frame) for split, frame in frames.items()},
        "query_class_counts": {split: frame.label.value_counts().to_dict() for split, frame in frames.items()},
        "data_version": config["data"]["version"],
        "representation": config["representation_version"],
        "kb_variants": {name: len(rows) for name, rows in kb_rows.items()},
        "files": {str(path.relative_to(artifacts)): sha256(path) for path in files},
    })
    print(
        f"Prepared {len(train_items):,} canonical KNN n-gram records, "
        f"{len(kb_rows['normal']):,} Normal and {len(kb_rows['anomaly']):,} Anomaly canonical traces, "
        f"{len(frames['validation']):,} validation and {len(frames['test']):,} test queries."
    )
