"""Embedding and cosine retrieval functions."""

import time

import numpy as np

from experiments.shared.data import normalize, read_jsonl, save_json, sha256


class BGEEncoder:
    def __init__(self, config, cache_dir):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        torch.set_num_threads(config.get("cpu_threads", 4))
        self.config = config
        self.device = config.get("device", "cpu")
        kwargs = {"revision": config.get("revision", "main"), "cache_dir": str(cache_dir)}
        self.tokenizer = AutoTokenizer.from_pretrained(config["model"], **kwargs)
        self.model = AutoModel.from_pretrained(config["model"], **kwargs).to(self.device).eval()
        self.max_tokens = min(int(self.tokenizer.model_max_length), int(self.model.config.max_position_embeddings))

    def encode(self, texts):
        torch = self.torch
        capacity = self.max_tokens - self.tokenizer.num_special_tokens_to_add(pair=False)
        chunks, owners, weights, lengths = [], [], [], []
        for owner, text in enumerate(texts):
            token_ids = self.tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
            if not token_ids:
                raise ValueError("Cannot embed empty text")
            lengths.append(len(token_ids))
            for offset in range(0, len(token_ids), capacity):
                chunk = token_ids[offset:offset + capacity]
                chunks.append(self.tokenizer.prepare_for_model(chunk, add_special_tokens=True, truncation=False))
                owners.append(owner)
                weights.append(len(chunk))
        vectors = np.zeros((len(texts), self.model.config.hidden_size), dtype=np.float32)
        batch_size = self.config.get("batch_size", 16)
        with torch.inference_mode():
            for start in range(0, len(chunks), batch_size):
                batch = self.tokenizer.pad(chunks[start:start + batch_size], padding=True, return_tensors="pt")
                batch = {key: value.to(self.device) for key, value in batch.items()}
                cls = torch.nn.functional.normalize(self.model(**batch).last_hidden_state[:, 0], p=2, dim=1).cpu().numpy()
                for position, vector in enumerate(cls, start):
                    vectors[owners[position]] += vector * weights[position]
        vectors = normalize(vectors)
        return vectors, {
            "model": self.config["model"], "resolved_revision": self.model.config._commit_hash,
            "dimension": vectors.shape[1], "normalize": True, "max_tokens": self.max_tokens,
            "texts": len(texts), "max_content_tokens": max(lengths),
            "long_text_policy": "weighted mean of normalized non-overlapping chunk CLS vectors",
        }


def build_embeddings(config, artifacts):
    retrieval_dir = artifacts / "retrieval"
    info_path = retrieval_dir / "embedding_config.json"
    if info_path.exists():
        print("Embeddings already exist; configuration was not changed.")
        return
    encoder = BGEEncoder(config["embedding"], artifacts / "model_cache")
    info = {}
    sources = {
        "normal": retrieval_dir / "normal_metadata.jsonl",
        "validation": artifacts / "queries/validation.jsonl",
        "test": artifacts / "queries/test.jsonl",
    }
    for name, source in sources.items():
        rows = read_jsonl(source)
        print(f"Embedding {name}: {len(rows)} traces")
        vectors, info[name] = encoder.encode([row["text"] for row in rows])
        np.save(retrieval_dir / f"{name}_embeddings.npy", vectors)
    save_json(info_path, {
        "requested": config["embedding"], "actual": info,
        "manifest_sha256": sha256(artifacts / "manifest.json"),
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
