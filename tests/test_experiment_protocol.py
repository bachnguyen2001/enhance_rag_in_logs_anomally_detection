"""Offline integration tests for shared sampling and the M0-M5 research protocol."""

from collections import Counter
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import io
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from experiments import run
from experiments.pipelines.kb_rag import select_contexts
from experiments.shared.config import load_config, model_config
from experiments.shared.data import (
    _sample, load_queries, normalize, prepare_resources, read_json, read_jsonl, sha256,
)
from experiments.shared.retrieval import build_embeddings
from experiments.shared.structure import normalized_lcs, _candidate_masks, _lcs_length_bitset


class FakeEncoder:
    calls = []

    def __init__(self, *args):
        pass

    def encode(self, texts):
        self.calls.append(list(texts))
        vectors = np.array([list(hashlib.sha256(t.encode()).digest()[:8]) for t in texts], dtype=np.float32)
        return normalize(vectors), {"texts": len(texts), "dimension": 8}


class FakeLLM:
    def __init__(self, *args):
        pass

    def classify(self, messages):
        assert "verified NORMAL training" not in messages[0]["content"]
        assert "QUERY BLOCK SEQUENCE:" in messages[1]["content"]
        query = messages[1]["content"].split("QUERY BLOCK SEQUENCE:")[1]
        assert "LABEL:" not in query
        return {"prediction": "NORMAL", "reason": "offline fixture", "prompt_tokens": 12,
                "request_count": 1, "cache_hit": False, "new_api_calls": 1, "latency_seconds": .01}

    def close(self):
        pass


class ExperimentV2Tests(unittest.TestCase):
    def fixture(self, root):
        config = load_config()
        config["artifacts_dir"] = str(root / "artifacts")
        config["results_dir"] = str(root / "results")
        config["llm_cache_dir"] = str(root / "cache")
        config["models"] = ["qwen-local", "second-local"]
        config["model_profiles"]["second-local"] = {
            **config["model_profiles"]["qwen-local"], "model": "second-model",
        }
        config["sampling"] = {"seed": 42, "validation_size": 20, "test_size": 500}
        config["selection"]["fixed"]["contexts_sweep"] = [1, 2]
        config["progress_every"] = 1000
        sources = root / "source"
        sources.mkdir()
        config["data"] = {
            "version": "synthetic", "split_dir": str(sources),
            "traces": str(sources / "traces.csv"), "templates": str(sources / "templates.csv"),
            "occurrence_matrix": str(sources / "counts.csv"),
        }
        pd.DataFrame([{"EventId": f"E{i}", "EventTemplate": f"event {i}"} for i in range(1, 30)]).to_csv(
            config["data"]["templates"], index=False)
        traces, counts = [], []
        offset = 0
        for split, size in (("train", 40), ("validation", 50), ("test", 550)):
            manifest = []
            for index in range(offset, offset + size):
                events = ["E1"] + [("E2" if bit == "0" else "E3") for bit in bin(index + 1)[2:]]
                block = f"block_{index:04d}"
                label = "Normal" if index % 2 else "Anomaly"
                signature = hashlib.sha256(",".join(events).encode()).hexdigest()
                manifest.append({"block_id": block, "label": label, "trace_signature": signature})
                traces.append({"BlockId": block, "Features": "[" + ",".join(events) + "]"})
                counter = Counter(events)
                counts.append({"BlockId": block, **{f"E{i}": counter[f"E{i}"] for i in range(1, 30)}})
            pd.DataFrame(manifest).to_csv(sources / f"{split}_block_ids.csv", index=False)
            offset += size
        pd.DataFrame(traces).to_csv(config["data"]["traces"], index=False)
        pd.DataFrame(counts).to_csv(config["data"]["occurrence_matrix"], index=False)
        return config

    def test_uniform_sampling_ignores_labels_and_source_order(self):
        source = pd.DataFrame({"block_id": [str(i).zfill(3) for i in range(50)], "label": ["Normal"] * 50})
        selected = _sample(source, 20, 42).block_id.tolist()
        changed = source.sample(frac=1, random_state=9).copy()
        changed["label"] = "Anomaly"
        self.assertEqual(selected, _sample(changed, 20, 42).block_id.tolist())
        self.assertEqual(len(_sample(source, "all", 42)), 50)

    def test_stratified_sampling_uses_requested_label_counts(self):
        source = pd.DataFrame({
            "block_id": [str(i).zfill(3) for i in range(30)],
            "label": ["Normal"] * 20 + ["Anomaly"] * 10,
        })
        sampled = _sample(source, {"normal": 7, "anomaly": 3}, 42)
        self.assertEqual(sampled.label.value_counts().to_dict(), {"Normal": 7, "Anomaly": 3})
        self.assertEqual(sampled.block_id.tolist(), _sample(source.sample(frac=1, random_state=9), {"normal": 7, "anomaly": 3}, 42).block_id.tolist())

    def test_adaptive_selection_uses_candidate_pool_threshold_and_zero_context(self):
        scores = np.arange(30, dtype=float) / 30
        selected, passing = select_contexts(scores, {"selection": "adaptive", "threshold": .5, "candidate_pool": 10})
        self.assertEqual(passing, 10)
        self.assertEqual(selected, list(range(29, 19, -1)))
        self.assertEqual(select_contexts(scores, {"selection": "adaptive", "threshold": 1, "candidate_pool": 10}), ([], 0))
        self.assertEqual(select_contexts(scores, {"selection": "fixed", "contexts": 3})[0], [29, 28, 27])

    def test_lcs_optimization_matches_reference(self):
        rng = random.Random(0)
        for _ in range(50):
            a = [rng.choice("ABC") for _ in range(20)]
            b = [rng.choice("ABC") for _ in range(12)]
            self.assertAlmostEqual(normalized_lcs(a, b), _lcs_length_bitset(a, _candidate_masks(b)) / 20)

    def test_all_methods_two_models_share_exact_500_queries_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            config = self.fixture(root)
            prepare_resources(config)
            queries = load_queries(config, "test")
            self.assertEqual(len(queries), 500)
            expected = [q["block_id"] for q in queries]
            query_path = root / "artifacts/queries/test.jsonl"
            query_hash = sha256(query_path)
            prepare_resources(config)
            self.assertEqual(sha256(query_path), query_hash)
            changed = deepcopy(config)
            changed["sampling"]["seed"] = 99
            with self.assertRaisesRegex(ValueError, "sampling changed"):
                prepare_resources(changed)
            with self.assertRaisesRegex(ValueError, "sampling changed"):
                load_queries(changed, "test")
            FakeEncoder.calls = []
            with patch("experiments.shared.retrieval.BGEEncoder", FakeEncoder):
                build_embeddings(config, root / "artifacts")
                build_embeddings(config, root / "artifacts")
            self.assertEqual(len(FakeEncoder.calls), 4)  # Normal, Anomaly, validation, test; no Mixed call.
            folder = root / "artifacts/retrieval"
            np.testing.assert_array_equal(np.load(folder / "mixed_embeddings.npy"), np.concatenate([
                np.load(folder / "normal_embeddings.npy"), np.load(folder / "anomaly_embeddings.npy")]))
            with patch("experiments.run.ChatLLM", FakeLLM):
                run.run_requested(config, "validation")
                run.run_requested(config, "test")
            summaries = list((root / "results").glob("*/test/*_summary.json"))
            self.assertEqual(len(summaries), 39)  # shared M0 + 2*(M1 + 12 fixed sweep + 6 adaptive)
            for summary_path in summaries:
                summary = read_json(summary_path)
                rows = read_jsonl(summary_path.with_name(summary_path.name.replace("_summary.json", "_predictions.jsonl")))
                self.assertEqual([r["block_id"] for r in rows], expected)
                self.assertEqual(summary["query_manifest_sha256"], query_hash)
            mixed_path = next((root / "results/qwen-local/test").glob("m2_mixed_*_predictions.jsonl"))
            mixed = read_jsonl(mixed_path)
            self.assertTrue(any("LABEL: ANOMALY" in r["messages"][1]["content"] for r in mixed))
            with patch("experiments.run.ChatLLM", side_effect=AssertionError("Resume called LLM")):
                run.run_requested(config, "validation")
                run.run_requested(config, "test")
            run.report(config)
            self.assertEqual(len(pd.read_csv(root / "results/comparison.csv")), 39)
            first_config = read_json(root / "results/comparison_config.json")
            self.assertEqual(first_config["version"], 1)
            self.assertEqual(first_config["experiment_config"]["selection"], config["selection"])
            run.report(config)
            self.assertEqual(len(pd.read_csv(root / "results/comparison_v2.csv")), 39)
            self.assertEqual(read_json(root / "results/comparison_v2_config.json")["version"], 2)
            (root / "results/comparison.csv").unlink()
            run.report(config)
            self.assertTrue((root / "results/comparison.csv").exists())
            altered = model_config(config, "qwen-local")
            altered["llm"]["temperature"] = .5
            with self.assertRaisesRegex(FileNotFoundError, "Run validation first"):
                run.read_threshold(altered, "m3", "normal")
            query_path.write_text(query_path.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "Query manifest modified"):
                load_queries(config, "test")


if __name__ == "__main__":
    unittest.main()
