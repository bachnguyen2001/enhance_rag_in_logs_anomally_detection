"""Kiểm tra ranh giới split và Normal-only KB, không cần dataset đầy đủ."""

import unittest

import pandas as pd

from scripts.prepare_data import build_kb, make_split, overlap_report, signature


class PreparationTests(unittest.TestCase):
    def setUp(self):
        # Mỗi nhóm có hai nhãn: nếu chia đúng group thì cả hai phải đi cùng nhau.
        self.meta = pd.DataFrame([
            (f"blk_{i}_{label}", signature(["E1", f"E{i + 2}"]), label)
            for i in range(30) for label in ("Normal", "Anomaly")
        ], columns=["block_id", "trace_signature", "label"])

    def test_group_split_preserves_conflicting_groups_and_is_reproducible(self):
        first = make_split(self.meta, "group_trace", 42, .2, .2)
        reordered = make_split(self.meta.iloc[::-1], "group_trace", 42, .2, .2)
        pd.testing.assert_frame_equal(first, reordered)
        self.assertEqual(set(first.block_id), set(self.meta.block_id))
        self.assertTrue(first.groupby("trace_signature").split.nunique().eq(1).all())
        report = overlap_report(first)
        for pair in report["pairwise_overlap"].values():
            self.assertEqual(pair["shared_exact_sequences"], 0)
            self.assertEqual(pair["shared_block_ids"], 0)

    def test_random_split_stratifies_and_is_reproducible(self):
        first = make_split(self.meta, "random_block", 42, .2, .2)
        second = make_split(self.meta.iloc[::-1], "random_block", 42, .2, .2)
        pd.testing.assert_frame_equal(first, second)
        distribution = first.groupby(["split", "label"]).size()
        for label in ("Normal", "Anomaly"):
            self.assertEqual(distribution["train", label], 18)
            self.assertEqual(distribution["validation", label], 6)
            self.assertEqual(distribution["test", label], 6)

    def test_kb_excludes_non_normal_and_non_train_even_with_exact_matches(self):
        manifest = pd.DataFrame([
            ("a", "same", "Normal", "train"),
            ("b", "same", "Normal", "train"),
            ("c", "same", "Anomaly", "train"),
            ("d", "same", "Normal", "test"),
            ("e", "other", "Normal", "validation"),
            ("f", "anomaly_only", "Anomaly", "train"),
        ], columns=["block_id", "trace_signature", "label", "split"])
        kb = build_kb(manifest, {"same": (["E1", "E1"], ["template", "template"])})
        self.assertEqual(len(kb), 1)
        self.assertEqual(kb.iloc[0].occurrence_count, 2)
        self.assertEqual(kb.iloc[0].representative_block_id, "a")
        self.assertEqual(kb.iloc[0].event_ids, '["E1", "E1"]')

    def test_order_and_repetition_are_part_of_signature(self):
        self.assertNotEqual(signature(["E1", "E2"]), signature(["E2", "E1"]))
        self.assertNotEqual(signature(["E1"]), signature(["E1", "E1"]))

    def test_group_allocation_targets_blocks_with_unequal_group_sizes(self):
        rows = []
        for i, size in enumerate([500, 200, 100, 50] + [10] * 100):
            for j in range(size):
                rows.append((f"b{i}_{j}", f"g{i}", "Anomaly" if j % 10 == 0 else "Normal"))
        meta = pd.DataFrame(rows, columns=["block_id", "trace_signature", "label"])
        split = make_split(meta, "group_trace", 42, .15, .15)
        fractions = split.split.value_counts() / len(split)
        for part, target in [("train", .7), ("validation", .15), ("test", .15)]:
            self.assertLess(abs(fractions[part] - target), .01)
        self.assertTrue(split.groupby("trace_signature").split.nunique().eq(1).all())

    def test_overlap_reports_anomaly_exact_matches_separately(self):
        rows = [("a", "shared", "Normal", "train"),
                ("b", "shared", "Anomaly", "test"),
                ("c", "new", "Normal", "test"),
                ("d", "validation", "Normal", "validation")]
        manifest = pd.DataFrame(rows, columns=["block_id", "trace_signature", "label", "split"])
        stats = overlap_report(manifest)["relative_to_train"]["test"]["by_query_label"]
        self.assertEqual(stats["Anomaly"]["exact_match_in_normal_kb"], 1)
        self.assertEqual(stats["Anomaly"]["exact_match_rate"], 1)
        self.assertEqual(stats["Normal"]["seen_trace_rate"], 0)


if __name__ == "__main__":
    unittest.main()
