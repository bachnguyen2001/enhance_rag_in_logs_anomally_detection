import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx
import numpy as np

from experiments.pipelines import m2_fixed_rag, m3_adaptive_rag
from experiments.shared.data import normalize, render_trace
from experiments.shared.evaluation import detection_metrics
from experiments.shared.llm import ChatLLM, build_messages, parse_prediction
from experiments.shared.retrieval import knn_predict


class ExperimentTests(unittest.TestCase):
    def test_trace_representation_preserves_order_and_runs(self):
        text = render_trace(["E2", "E2", "E1", "E2"], {"E1": "first", "E2": "second"})
        self.assertIn("E2 × 2 → E1 → E2", text)
        self.assertEqual(text.count("E2: second"), 1)
        self.assertIn("HISTORICAL NORMAL REFERENCES:\nNONE", build_messages(text, [])[1]["content"])

    def test_adaptive_selection(self):
        scores = np.asarray([.9, .8, .7, .6, .5, .4, .3, .2, .1, 0, -.1])
        candidates, selected, removed = m3_adaptive_rag.select_contexts(scores, .8)
        self.assertEqual(len(candidates), 10)
        self.assertEqual(selected, [0, 1])
        self.assertEqual(removed, 8)
        self.assertEqual(m3_adaptive_rag.select_contexts(scores, 1)[1], [])
        self.assertEqual(m3_adaptive_rag.select_contexts(scores, -1)[2], 0)

    def test_m3_saves_rank_relevance_and_actual_context_count(self):
        class FakeLLM:
            def classify(self, messages):
                return {"prediction": "NORMAL", "reason": "fixture", "request_count": 1,
                        "prompt_tokens": 42, "latency_seconds": .1, "cache_hit": False,
                        "new_api_calls": 1, "rate_limit_retries": 0}
        kb = [{"trace_id": f"t{i}", "text": f"trace {i}"} for i in range(10)]
        scores = np.asarray([.95, .85, .75, .65, .55, .45, .35, .25, .15, .05])
        row = m3_adaptive_rag.predict_one(
            {"block_id": "q", "label": "NORMAL", "text": "query"}, scores,
            {"llm": FakeLLM(), "knowledge_base": kb}, {"threshold": .75},
        )
        self.assertEqual(row["context_count"], 3)
        self.assertEqual(row["prompt_tokens"], 42)
        self.assertEqual([item["rank"] for item in row["candidates"]], list(range(1, 11)))
        self.assertTrue(row["candidates"][2]["passes_relevance"])
        self.assertFalse(row["candidates"][3]["passes_relevance"])
        self.assertTrue(row["candidates"][2]["selected_as_context"])

    def test_m2_always_passes_three_contexts(self):
        class FakeLLM:
            def classify(self, messages):
                return {"prediction": "NORMAL", "reason": "fixture", "request_count": 1,
                        "prompt_tokens": 10, "latency_seconds": .1, "cache_hit": False, "new_api_calls": 1}
        kb = [{"trace_id": f"t{i}", "text": f"trace {i}"} for i in range(4)]
        row = m2_fixed_rag.predict_one(
            {"block_id": "q", "label": "NORMAL", "text": "query"},
            np.asarray([.1, .9, .8, .7]), {"llm": FakeLLM(), "knowledge_base": kb}, {},
        )
        self.assertEqual(row["contexts"], ["t1", "t2", "t3"])

    def test_knn_zero_distance_uses_exact_neighbors(self):
        train = normalize([[1, 0], [.8, .2], [.7, .3]])
        labels = np.array(["ANOMALY", "NORMAL", "NORMAL"])
        predictions, _ = knn_predict(train, labels, normalize([[1, 0]]), 3)
        self.assertEqual(predictions, ["ANOMALY"])

    def test_invalid_output_is_visible_in_metrics(self):
        self.assertEqual(parse_prediction("ANOMALY")[0], "INVALID_OUTPUT")
        self.assertEqual(parse_prediction('{"label":"normal","reason":"ok"}')[0], "NORMAL")
        summary = detection_metrics(["NORMAL", "ANOMALY"], ["NORMAL", "INVALID_OUTPUT"])
        self.assertIsNone(summary["f1"])
        self.assertEqual(summary["invalid_anomaly"], 1)

    def test_format_retry_and_jsonl_cache(self):
        calls = []
        def handler(request):
            calls.append(json.loads(request.content))
            content = "wrong" if len(calls) == 1 else '{"label":"NORMAL","reason":"ok"}'
            return httpx.Response(200, json={"choices": [{"message": {"content": content}}],
                                             "model": "fake", "usage": {"prompt_tokens": 12}})
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"TEST_LLM_KEY": "dummy"}):
            config = {"model": "fake", "base_url": "https://example.invalid/v1", "api_key_env": "TEST_LLM_KEY"}
            llm = ChatLLM(config, Path(folder) / "cache.jsonl", httpx.Client(transport=httpx.MockTransport(handler)))
            messages = build_messages("E1", [])
            first = llm.classify(messages)
            second = llm.classify(messages)
            self.assertEqual(first["request_count"], 2)
            self.assertEqual(first["prompt_tokens"], 24)
            self.assertTrue(second["cache_hit"])
            self.assertEqual(len(calls), 2)
            llm.close()

    def test_rate_limit_is_retried_and_recorded(self):
        calls = []
        def handler(request):
            calls.append(request)
            if len(calls) == 1:
                return httpx.Response(429, headers={"Retry-After": "0"})
            return httpx.Response(200, json={"choices": [{"message": {
                "content": '{"label":"NORMAL","reason":"ok"}'}}], "usage": {"prompt_tokens": 8}})
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"TEST_LLM_KEY": "dummy"}):
            config = {"model": "fake", "base_url": "https://example.invalid/v1",
                      "api_key_env": "TEST_LLM_KEY", "rate_limit_backoff_seconds": 0}
            llm = ChatLLM(config, Path(folder) / "cache.jsonl",
                          httpx.Client(transport=httpx.MockTransport(handler)))
            result = llm.classify(build_messages("E1", []))
            self.assertEqual(result["prediction"], "NORMAL")
            self.assertEqual(result["rate_limit_retries"], 1)
            self.assertEqual(len(calls), 2)
            llm.close()

    def test_read_timeout_is_retried_and_recorded(self):
        calls = []
        def handler(request):
            calls.append(request)
            if len(calls) == 1:
                raise httpx.ReadTimeout("fixture timeout", request=request)
            return httpx.Response(200, json={"choices": [{"message": {
                "content": '{"label":"NORMAL","reason":"ok"}'}}], "usage": {"prompt_tokens": 8}})
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"TEST_LLM_KEY": "dummy"}):
            config = {"model": "fake", "base_url": "https://example.invalid/v1",
                      "api_key_env": "TEST_LLM_KEY", "transient_backoff_seconds": 0}
            llm = ChatLLM(config, Path(folder) / "cache.jsonl",
                          httpx.Client(transport=httpx.MockTransport(handler)))
            result = llm.classify(build_messages("E1", []))
            self.assertEqual(result["prediction"], "NORMAL")
            self.assertEqual(result["transient_retries"], 1)
            self.assertEqual(len(calls), 2)
            llm.close()


if __name__ == "__main__":
    unittest.main()
