"""Negative tests: verifier phải phát hiện artifact bị sửa, không chỉ đọc report PASS."""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from scripts.verify_prepared import verify_prepared


class VerificationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "source"
        self.prepared = self.root / "prepared"
        self.data.mkdir()
        self.prepared.mkdir()
        rows = []
        for i in range(6):
            events = ["E1"] * (i + 1)
            digest = hashlib.sha256(json.dumps(events, separators=(",", ":")).encode()).hexdigest()
            for label in ("Normal", "Anomaly"):
                rows.append({"block_id": f"b{i}_{label}", "trace_signature": digest,
                             "event_ids": json.dumps(events), "template_sequence": json.dumps(["event"] * len(events)),
                             "sequence_length": len(events), "label": label, "conflicting_trace": True,
                             "split": ("train", "validation", "test")[i // 2]})
        frame = pd.DataFrame(rows)
        frame.drop(columns="split").to_csv(self.prepared / "block_sequences.csv", index=False)
        frame[["block_id", "label"]].rename(columns={"block_id": "BlockId", "label": "Label"}).to_csv(self.data / "anomaly_label.csv", index=False)
        pd.DataFrame({"EventId": ["E1"], "EventTemplate": ["event"]}).to_csv(self.data / "HDFS.log_templates.csv", index=False)
        pd.DataFrame({"BlockId": frame.block_id,
                      "Features": frame.event_ids.map(lambda s: "[" + ",".join(json.loads(s)) + "]")}).to_csv(self.data / "Event_traces.csv", index=False)
        frame.groupby(["trace_signature", "label"]).size().unstack().to_csv(self.prepared / "conflicting_sequences.csv")
        for protocol in ("random_block", "group_trace"):
            folder = self.prepared / protocol
            folder.mkdir()
            for split in ("train", "validation", "test"):
                frame.loc[frame.split == split, ["block_id", "trace_signature", "label", "split"]].to_csv(folder / f"{split}_block_ids.csv", index=False)
            normal_train = frame.loc[frame.split.eq("train") & frame.label.eq("Normal")]
            kb = pd.DataFrame([{"trace_id": row.trace_signature, "event_ids": row.event_ids,
                                "template_text": "\n".join(["E1: event"] * row.sequence_length),
                                "occurrence_count": 1, "representative_block_id": row.block_id}
                               for row in normal_train.itertuples()])
            kb.to_csv(folder / "normal_knowledge_base.csv", index=False)
            annotations = frame.loc[frame.split.ne("train"), ["block_id", "trace_signature", "label", "split", "conflicting_trace"]].copy()
            annotations["seen_in_train"] = False
            annotations["exact_match_in_normal_kb"] = False
            annotations.to_csv(folder / "evaluation_annotations.csv", index=False)

    def verify(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return verify_prepared(self.prepared, self.data, chunksize=3)

    def test_valid_artifacts_without_any_producer_report(self):
        result = self.verify()
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["blocks_checked"], 12)

    def test_rejects_wrong_kb_occurrence_count(self):
        path = self.prepared / "random_block/normal_knowledge_base.csv"
        kb = pd.read_csv(path)
        kb.loc[0, "occurrence_count"] = 999
        kb.to_csv(path, index=False)
        with self.assertRaisesRegex(AssertionError, "counts"):
            self.verify()

    def test_rejects_normal_train_representative_of_wrong_sequence(self):
        path = self.prepared / "random_block/normal_knowledge_base.csv"
        kb = pd.read_csv(path)
        kb.loc[0, "representative_block_id"] = kb.loc[1, "representative_block_id"]
        kb.to_csv(path, index=False)
        with self.assertRaisesRegex(AssertionError, "different pattern"):
            self.verify()

    def test_rejects_cross_split_block_overlap(self):
        folder = self.prepared / "group_trace"
        train = pd.read_csv(folder / "train_block_ids.csv")
        test = pd.read_csv(folder / "test_block_ids.csv")
        duplicated = train.iloc[[0]].copy()
        duplicated["split"] = "test"
        pd.concat([test, duplicated]).to_csv(folder / "test_block_ids.csv", index=False)
        with self.assertRaisesRegex(AssertionError, "cross-split BlockId overlap"):
            self.verify()


if __name__ == "__main__":
    unittest.main()
