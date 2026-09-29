"""Embedding and cosine retrieval functions."""

import time
from pathlib import Path

import numpy as np

from experiments.shared.data import ROOT, normalize, read_json, read_jsonl, save_json, sha256, validate_prepared


class BGEEncoder:
    def __init__(self, config, cache_dir):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        torch.set_num_threads(config.get("cpu_threads", 4))
        self.config = config
        self.device = config.get("device", "cpu")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA unavailable. Set embedding.device to cpu in config_system.json.")
        kwargs = {"revision": config.get("revision", "main"), "cache_dir": str(cache_dir)}
        self.tokenizer = AutoTokenizer.from_pretrained(config["model"], **kwargs)
        self.model = AutoModel.from_pretrained(config["model"], **kwargs).to(self.device).eval()
        self.max_tokens = min(int(self.tokenizer.model_max_length), int(self.model.config.max_position_embeddings))

    def encode(self, texts):
        torch = self.torch
        capacity = self.max_tokens - self.tokenizer.num_special_tokens_to_add(pair=False)
        vectors = np.zeros((len(texts), self.model.config.hidden_size), dtype=np.float32)
        lengths = []
        batch_size = self.config.get("batch_size", 16)
        for text_start in range(0, len(texts), batch_size):
            text_batch = texts[text_start:text_start + batch_size]
            chunks, owners, weights = [], [], []
            for owner, text in enumerate(text_batch):
                token_ids = self.tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
                if not token_ids:
                    raise ValueError("Cannot embed empty text")
                lengths.append(len(token_ids))
                for offset in range(0, len(token_ids), capacity):
                    chunk = token_ids[offset:offset + capacity]
                    chunks.append(self.tokenizer.prepare_for_model(chunk, add_special_tokens=True, truncation=False))
                    owners.append(owner)
                    weights.append(len(chunk))
            with torch.inference_mode():
                for start in range(0, len(chunks), batch_size):
                    batch = self.tokenizer.pad(chunks[start:start + batch_size], padding=True, return_tensors="pt")
                    batch = {key: value.to(self.device) for key, value in batch.items()}
                    cls = torch.nn.functional.normalize(self.model(**batch).last_hidden_state[:, 0], p=2, dim=1).cpu().numpy()
                    for position, vector in enumerate(cls, start):
                        vectors[text_start + owners[position]] += vector * weights[position]
            print(f"  encoded {min(text_start + len(text_batch), len(texts))}/{len(texts)}", flush=True)
        vectors = normalize(vectors)
        return vectors, {
            "model": self.config["model"], "resolved_revision": self.model.config._commit_hash,
            "dimension": vectors.shape[1], "normalize": True, "max_tokens": self.max_tokens,
            "texts": len(texts), "max_content_tokens": max(lengths),
            "long_text_policy": "weighted mean of normalized non-overlapping chunk CLS vectors",
        }


def build_embeddings(config, artifacts):
    validate_prepared(config, verify_all=True)
    retrieval_dir = artifacts / "retrieval"
    info_path = retrieval_dir / "embedding_config.json"
    if info_path.exists():
        info = read_json(info_path)
        if info["requested"] != config["embedding"] or info["manifest_sha256"] != sha256(artifacts / "manifest.json"):
            raise ValueError("Embedding settings/data changed; choose a new artifacts_dir")
        for name, checksum in info.get("vector_hashes", {}).items():
            if sha256(retrieval_dir / f"{name}_embeddings.npy") != checksum:
                raise ValueError(f"Embedding file changed: {name}")
        print("Embeddings already exist; configuration was not changed.")
        return
    cache_dir = Path(config.get("model_cache_dir", "artifacts/model_cache"))
    if not cache_dir.is_absolute():
        cache_dir = ROOT / "experiments" / cache_dir
    encoder = BGEEncoder(config["embedding"], cache_dir)
    info = {}
    sources = {
        "normal": retrieval_dir / "normal_metadata.jsonl",
        "anomaly": retrieval_dir / "anomaly_metadata.jsonl",
        "validation": artifacts / "queries/validation.jsonl",
        "test": artifacts / "queries/test.jsonl",
    }
    for name, source in sources.items():
        rows = read_jsonl(source)
        print(f"Embedding {name}: {len(rows)} traces")
        vectors, info[name] = encoder.encode([row["text"] for row in rows])
        np.save(retrieval_dir / f"{name}_embeddings.npy", vectors)
    # Mixed metadata has exactly this order. Reuse vectors, never re-embed Mixed.
    mixed = read_jsonl(retrieval_dir / "mixed_metadata.jsonl")
    combined = (read_jsonl(retrieval_dir / "normal_metadata.jsonl") +
                read_jsonl(retrieval_dir / "anomaly_metadata.jsonl"))
    if mixed != combined:
        raise ValueError("Mixed KB order differs from Normal + Anomaly")
    np.save(retrieval_dir / "mixed_embeddings.npy", np.concatenate([
        np.load(retrieval_dir / "normal_embeddings.npy"),
        np.load(retrieval_dir / "anomaly_embeddings.npy"),
    ]))
    info["mixed"] = {"texts": len(mixed), "source": "concatenated normal + anomaly"}
    save_json(info_path, {
        "requested": config["embedding"], "actual": info,
        "manifest_sha256": sha256(artifacts / "manifest.json"),
        "vector_hashes": {name: sha256(retrieval_dir / f"{name}_embeddings.npy")
                          for name in (*sources, "mixed")},
    })


def top_indices(scores, k):
    return np.argsort(-np.asarray(scores), kind="stable")[:k].tolist()


def knn_predict(train_vectors, train_labels, query_vectors, k):
    predictions, latency = [], []
    for query in query_vectors:
        start = time.perf_counter()
        scores = np.clip(train_vectors @ query, -1, 1)
        neighbors = top_indices(scores, k)
        distances = 1 - scores[neighbors]
        zero = distances <= 1e-7
        weights = zero.astype(float) if zero.any() else 1 / np.maximum(distances, 1e-7)
        anomaly = weights[train_labels[neighbors] == "ANOMALY"].sum()
        normal = weights[train_labels[neighbors] == "NORMAL"].sum()
        predictions.append("ANOMALY" if anomaly > normal else "NORMAL")
        latency.append(time.perf_counter() - start)
    return predictions, latency
