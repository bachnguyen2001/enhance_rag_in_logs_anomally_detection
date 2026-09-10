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


def render_trace(events, templates):
    """Keep order and consecutive repetition; never include labels or Type."""
    definitions = "\n".join(f"{event}: {templates[event]}" for event in dict.fromkeys(events))
    runs = [(event, len(list(items))) for event, items in groupby(events)]
    ordered = " → ".join(event if count == 1 else f"{event} × {count}" for event, count in runs)
    return f"EVENT DEFINITIONS:\n{definitions}\n\nORDERED TRACE:\n{ordered}"


def experiment_paths(config):
    base = ROOT / "experiments"
    return {
        "base": base,
        "artifacts": base / config.get("artifacts_dir", "artifacts"),
        "results": base / config.get("results_dir", "results"),
    }


def _sample(source, amounts, seed):
    rng = random.Random(seed)
    chosen = []
    for label, amount in amounts.items():
        pool = source.loc[source.label.eq(label)].sort_values("block_id").to_dict("records")
        chosen.extend(rng.sample(pool, amount))
    rng.shuffle(chosen)
    return pd.DataFrame(chosen)


def prepare_resources(config):
    """Create fixed subsets and representations. Refuse to overwrite a manifest."""
    paths = experiment_paths(config)
    artifacts = paths["artifacts"]
    manifest_path = artifacts / "manifest.json"
    if manifest_path.exists():
        print("Experiment artifacts already prepared; manifest was not changed.")
        return

    split_dir = ROOT / config["data"]["split_dir"]
    subsets = config["subsets"]
    frames = {
        split: _sample(
            pd.read_csv(split_dir / f"{split}_block_ids.csv"),
            {"Normal": spec["normal"], "Anomaly": spec["anomaly"]},
            spec["seed"],
        )
        for split, spec in subsets.items()
    }
    if set(frames["validation"].block_id) & set(frames["test"].block_id):
        raise ValueError("Validation and test BlockIds overlap")

    templates = pd.read_csv(ROOT / config["data"]["templates"]).set_index("EventId").EventTemplate.to_dict()
    targets = set().union(*(set(frame.block_id) for frame in frames.values()))
    sequences = {}
    for chunk in pd.read_csv(ROOT / config["data"]["traces"], usecols=["BlockId", "Features"], chunksize=20000):
        for row in chunk.loc[chunk.BlockId.isin(targets)].itertuples(index=False):
            sequences[row.BlockId] = [event.strip() for event in row.Features[1:-1].split(",")]

    query_dir = artifacts / "queries"
    for split, frame in frames.items():
        rows = [{
            "block_id": row.block_id,
            "label": row.label.upper(),
            "trace_signature": row.trace_signature,
            "event_ids": sequences[row.block_id],
            "text": render_trace(sequences[row.block_id], templates),
        } for row in frame.itertuples(index=False)]
        save_jsonl(query_dir / f"{split}.jsonl", rows)
        frame.to_csv(query_dir / f"{split}_queries.csv", index=False)

    retrieval_dir = artifacts / "retrieval"
    kb = pd.read_csv(split_dir / "normal_knowledge_base.csv").sort_values("trace_id")
    save_jsonl(retrieval_dir / "normal_metadata.jsonl", [{
        "trace_id": row.trace_id,
        "event_ids": json.loads(row.event_ids),
        "text": render_trace(json.loads(row.event_ids), templates),
        "occurrence_count": int(row.occurrence_count),
    } for row in kb.itertuples(index=False)])

    train = pd.read_csv(split_dir / "train_block_ids.csv").sort_values("block_id")
    matrix = pd.read_csv(ROOT / config["data"]["occurrence_matrix"]).set_index("BlockId")
    columns = [f"E{i}" for i in range(1, 30)]
    knn_dir = artifacts / "knn"
    knn_dir.mkdir(parents=True, exist_ok=True)
    np.save(knn_dir / "train_vectors.npy", normalize(matrix.loc[train.block_id, columns].to_numpy()))
    np.save(knn_dir / "train_labels.npy", train.label.str.upper().to_numpy(dtype="U7"))
    train[["block_id", "label"]].to_csv(knn_dir / "train_ids.csv", index=False)
    for split, frame in frames.items():
        np.save(knn_dir / f"{split}_vectors.npy", normalize(matrix.loc[frame.block_id, columns].to_numpy()))

    files = sorted(path for path in artifacts.rglob("*") if path.is_file())
    save_json(manifest_path, {
        "data_version": config["data"]["version"],
        "representation": config["representation_version"],
        "files": {str(path.relative_to(artifacts)): sha256(path) for path in files},
    })
    print(f"Prepared {len(train):,} KNN records, {len(kb):,} Normal traces, 200 validation and 500 test queries.")
