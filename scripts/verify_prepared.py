"""Kiểm chứng độc lập CSV xuất ra; không import prepare_data hoặc tin report của nó.

Chạy: .venv/bin/python -m scripts.verify_prepared --prepared-dir processed_v2
Kiểm tra toàn bộ records, raw CSV trace, manifests, Normal KB và annotations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def verify_prepared(prepared_dir: Path, data_dir: Path, chunksize: int = 20000) -> dict:
    checks = []

    def check(condition, description):
        if not condition:
            raise AssertionError(description)
        checks.append(description)

    truth = pd.read_csv(data_dir / "anomaly_label.csv").set_index("BlockId").Label
    templates = pd.read_csv(data_dir / "HDFS.log_templates.csv").set_index("EventId").EventTemplate
    check(truth.index.is_unique and truth.index.notna().all(), "unique source labels")
    check(templates.index.is_unique and templates.notna().all(), "unique valid source templates")
    check(set(truth) == {"Normal", "Anomaly"}, "source label vocabulary")
    payloads, parts = {}, []
    print("Independent verification: đọc toàn bộ block_sequences.csv...", flush=True)
    for chunk in pd.read_csv(prepared_dir / "block_sequences.csv", chunksize=chunksize):
        if not chunk.label.eq(chunk.block_id.map(truth)).all():
            raise AssertionError("Record labels differ from source")
        for row in chunk.itertuples(index=False):
            cached = payloads.get(row.trace_signature)
            if cached is None:
                sequence = json.loads(row.event_ids)
                texts = json.loads(row.template_sequence)
                if not isinstance(sequence, list) or not sequence or not all(isinstance(e, str) for e in sequence):
                    raise AssertionError("Invalid event_ids JSON array")
                digest = hashlib.sha256(json.dumps(sequence, separators=(",", ":")).encode()).hexdigest()
                if digest != row.trace_signature or any(e not in templates for e in sequence):
                    raise AssertionError("Signature or template mapping differs")
                if texts != [templates[e] for e in sequence]:
                    raise AssertionError("Template text or order differs")
                cached = (row.event_ids, row.template_sequence, len(sequence), sequence, texts)
                payloads[digest] = cached
            if cached[:3] != (row.event_ids, row.template_sequence, row.sequence_length):
                raise AssertionError("Same signature contains different sequence or length")
        parts.append(chunk[["block_id", "trace_signature", "label", "conflicting_trace"]])
    records = pd.concat(parts, ignore_index=True).set_index("block_id")
    check(records.index.is_unique and records.index.notna().all(), "unique exported BlockIds")
    check(set(records.index) == set(truth.index), "export covers every source BlockId")
    check(True, "all JSON sequences, signatures, lengths, labels and template order validated")
    # Xác minh sự kiện từng BlockId với Event_traces.csv, không dùng parser của producer.
    raw_cache, raw_seen = {}, set()
    for chunk in pd.read_csv(data_dir / "Event_traces.csv", usecols=["BlockId", "Features"], chunksize=chunksize):
        raw_ids = []
        for value in chunk.Features:
            if value not in raw_cache:
                if not isinstance(value, str) or not value.startswith("[") or not value.endswith("]"):
                    raise AssertionError("Invalid source feature syntax")
                events = [e.strip() for e in value[1:-1].split(",")]
                raw_cache[value] = hashlib.sha256(json.dumps(events, separators=(",", ":")).encode()).hexdigest()
            raw_ids.append(raw_cache[value])
        if chunk.BlockId.duplicated().any() or raw_seen.intersection(chunk.BlockId):
            raise AssertionError("Duplicate source trace BlockId")
        raw_seen.update(chunk.BlockId)
        if chunk.BlockId.map(records.trace_signature).tolist() != raw_ids:
            raise AssertionError("Exported sequence differs from source trace")
    check(raw_seen == set(records.index), "every exported sequence equals its source BlockId trace")
    conflicts = records.groupby("trace_signature").label.nunique().gt(1)
    check(records.conflicting_trace.eq(records.trace_signature.map(conflicts)).all(),
          "conflicting_trace recomputed from labels")
    conflict_file = pd.read_csv(prepared_dir / "conflicting_sequences.csv").set_index("trace_signature")
    histogram = records.groupby(["trace_signature", "label"]).size().unstack(fill_value=0)
    expected_conflicts = histogram.loc[conflicts[conflicts].index].sort_index()
    pd.testing.assert_frame_equal(conflict_file.sort_index()[["Normal", "Anomaly"]],
                                  expected_conflicts[["Normal", "Anomaly"]], check_names=False)
    check(True, "conflicting_sequences.csv matches exported labels")

    summaries, overlap_rows, compression = [], [], []
    for protocol in ("random_block", "group_trace"):
        folder = prepared_dir / protocol
        split_frames = {}
        for split in ("train", "validation", "test"):
            part = pd.read_csv(folder / f"{split}_block_ids.csv").set_index("block_id")
            check(part.index.is_unique and part.index.notna().all(), f"{protocol}/{split}: unique BlockIds")
            check(part.split.eq(split).all(), f"{protocol}/{split}: correct split tags")
            check(set(part.index).issubset(records.index), f"{protocol}/{split}: IDs in records")
            for column in ("label", "trace_signature"):
                check(part[column].eq(records.loc[part.index, column]).all(), f"{protocol}/{split}: {column} matches records")
            check(set(part.label) == {"Normal", "Anomaly"}, f"{protocol}/{split}: both classes")
            split_frames[split] = part
            summaries.append({"protocol": protocol, "split": split, "blocks": len(part),
                              "Normal": int(part.label.eq("Normal").sum()),
                              "Anomaly": int(part.label.eq("Anomaly").sum()),
                              "block_fraction": len(part) / len(records),
                              "anomaly_prevalence": float(part.label.eq("Anomaly").mean())})
        combined = pd.concat(split_frames.values())
        check(combined.index.is_unique, f"{protocol}: no cross-split BlockId overlap")
        check(set(combined.index) == set(records.index), f"{protocol}: union equals all records")
        if protocol == "group_trace":
            check(combined.groupby("trace_signature").split.nunique().eq(1).all(), "group_trace: no cross-split sequence overlap")
        normal_train = split_frames["train"].loc[split_frames["train"].label.eq("Normal")]
        expected_counts = normal_train.groupby("trace_signature").size().sort_index()
        kb = pd.read_csv(folder / "normal_knowledge_base.csv").set_index("trace_id")
        check(kb.index.is_unique and kb.index.notna().all(), f"{protocol}: unique canonical KB traces")
        check(set(kb.index) == set(expected_counts.index), f"{protocol}: KB exactly covers Normal train patterns")
        check(kb.occurrence_count.sort_index().eq(expected_counts).all(), f"{protocol}: KB counts equal Normal train counts")
        check(set(kb.representative_block_id).issubset(normal_train.index), f"{protocol}: all representatives in Normal train")
        for trace_id, row in kb.iterrows():
            sequence, texts = payloads[trace_id][3:]
            if json.loads(row.event_ids) != sequence or row.template_text != "\n".join(f"{e}: {t}" for e, t in zip(sequence, texts)):
                raise AssertionError("KB payload differs from canonical source sequence")
            if records.loc[row.representative_block_id, "trace_signature"] != trace_id:
                raise AssertionError("KB representative belongs to a different pattern")
        check(True, f"{protocol}: KB payload and representative sequence checked")
        compression.append({"protocol": protocol, "normal_train_blocks": len(normal_train),
                            "canonical_normal_traces": len(kb), "retained_record_fraction": len(kb) / len(normal_train)})
        train_signatures = set(split_frames["train"].trace_signature)
        for split in ("validation", "test"):
            for label in ("Normal", "Anomaly"):
                queries = split_frames[split].loc[split_frames[split].label.eq(label)]
                seen = queries.trace_signature.isin(train_signatures)
                exact = queries.trace_signature.isin(kb.index)
                overlap_rows.append({"protocol": protocol, "split": split, "query_label": label,
                                     "query_blocks": len(queries), "seen_blocks": int(seen.sum()),
                                     "unseen_blocks": int((~seen).sum()), "seen_trace_rate": float(seen.mean()),
                                     "unseen_trace_rate": float((~seen).mean()),
                                     "exact_match_in_normal_kb": int(exact.sum()),
                                     "exact_match_rate": float(exact.mean())})
        annotations = pd.read_csv(folder / "evaluation_annotations.csv").set_index("block_id")
        evaluation = pd.concat([split_frames["validation"], split_frames["test"]])
        check(annotations.index.is_unique and set(annotations.index) == set(evaluation.index), f"{protocol}: annotation coverage")
        for column in ("trace_signature", "label", "split"):
            check(annotations[column].eq(evaluation.loc[annotations.index, column]).all(), f"{protocol}: annotation {column}")
        for column, expected in {
            "seen_in_train": annotations.trace_signature.isin(train_signatures),
            "exact_match_in_normal_kb": annotations.trace_signature.isin(kb.index),
            "conflicting_trace": annotations.trace_signature.map(conflicts),
        }.items():
            check(annotations[column].eq(expected).all(), f"{protocol}: annotation {column}")
    return {"status": "passed", "verification_schema_version": 1,
            "prepared_dir": str(prepared_dir.resolve()), "blocks_checked": len(records),
            "check_count": len(checks), "checks": checks,
            "split_composition": summaries, "overlap_by_query_label": overlap_rows,
            "kb_compression": compression,
            "independence": "Recomputed from all records, source CSVs, manifests and KB; no producer function or JSON report used"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, default=Path("processed_v2"))
    parser.add_argument("--data-dir", type=Path, default=Path("preprocessed"))
    parser.add_argument("--output", type=Path, default=Path("outputs/data_understanding/prepared_verification.json"))
    args = parser.parse_args()
    result = verify_prepared(args.prepared_dir, args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"PASS: {result['check_count']} checks; {result['blocks_checked']:,} records")
