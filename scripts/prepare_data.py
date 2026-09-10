"""Tạo experimental dataset HDFS từ CSV; không tạo embedding hoặc chạy model.

Chạy: .venv/bin/python -m scripts.prepare_data
Phụ thuộc: pandas, numpy. Output mặc định: processed_v2/ (phải chưa tồn tại).
Hai protocol dùng chung records, nhưng có manifest và Normal-only KB riêng.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import platform
import shutil
import tempfile

import numpy as np
import pandas as pd

from scripts.data_understanding import LABELS, parse_list, require_unique


SOURCE_FILES = ["anomaly_label.csv", "Event_traces.csv",
                "HDFS.log_templates.csv", "Event_occurrence_matrix.csv"]
SPLITS = ("train", "validation", "test")


def signature(events: list[str]) -> str:
    """Hash của danh sách có thứ tự, không phải set hay vector counts."""
    return hashlib.sha256(json.dumps(events, separators=(",", ":")).encode()).hexdigest()


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def source_hash(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def assign_parts(items: np.ndarray, rng: np.random.Generator,
                 validation_fraction: float, test_fraction: float) -> dict[str, str]:
    items = rng.permutation(items)
    nv = int(len(items) * validation_fraction)
    nt = int(len(items) * test_fraction)
    if min(nv, nt, len(items) - nv - nt) < 1:
        raise ValueError("Không đủ mẫu/nhóm cho tỷ lệ split đã chọn.")
    return {str(item): split for split, part in zip(
        SPLITS, [items[nv + nt:], items[:nv], items[nv:nv + nt]]) for item in part}


def make_split(meta: pd.DataFrame, protocol: str, seed: int,
               validation_fraction: float, test_fraction: float) -> pd.DataFrame:
    """A stratify theo nhãn; B giữ group và tối ưu sai lệch số block/anomaly."""
    rng = np.random.default_rng(seed)
    result = meta[["block_id", "trace_signature", "label"]].copy()
    if protocol == "random_block":
        mapping = {}
        for label in ("Normal", "Anomaly"):
            ids = np.sort(result.loc[result.label == label, "block_id"].to_numpy())
            mapping.update(assign_parts(ids, rng, validation_fraction, test_fraction))
        result["split"] = result.block_id.map(mapping)
    elif protocol == "group_trace":
        mapping = allocate_groups(result, rng, validation_fraction, test_fraction)
        result["split"] = result.trace_signature.map(mapping)
    else:
        raise ValueError(f"Protocol không hỗ trợ: {protocol}")
    if result.block_id.duplicated().any() or result.split.isna().any():
        raise ValueError("Manifest trùng BlockId hoặc có mẫu chưa được gán split.")
    for split in SPLITS:
        if set(result.loc[result.split == split, "label"]) != {"Normal", "Anomaly"}:
            raise ValueError(f"{protocol}/{split} không có đủ hai lớp; xem lại thiết kế chia nhóm.")
    if protocol == "group_trace" and result.groupby("trace_signature").split.nunique().max() != 1:
        raise AssertionError("Trace đã bị tách qua nhiều split.")
    return result.sort_values("block_id").reset_index(drop=True)


def allocate_groups(meta: pd.DataFrame, rng: np.random.Generator,
                    validation_fraction: float, test_fraction: float) -> dict[str, str]:
    """Greedy largest-first, seeded tie order, rồi tối đa 5 pass cải thiện đơn nhóm.

    Objective: tổng bình phương sai lệch block/total_blocks và
    anomaly/total_anomalies so với target của ba split. Không dùng model score.
    Không đảm bảo optimum toàn cục hoặc tỷ lệ chính xác với group bất khả phân.
    """
    table = meta.groupby("trace_signature").label.agg(
        blocks="size", anomalies=lambda values: int(values.eq("Anomaly").sum()))
    table = table.iloc[rng.permutation(len(table))]
    scale = np.array([len(meta), max(int(meta.label.eq("Anomaly").sum()), 1)], dtype=float)
    weights = table[["blocks", "anomalies"]].to_numpy(dtype=float) / scale
    order = np.argsort(-weights.max(axis=1), kind="stable")
    table = table.iloc[order]
    weights = weights[order]
    fractions = np.array([1 - validation_fraction - test_fraction, validation_fraction, test_fraction])
    target = fractions[:, None] * weights.sum(axis=0)
    current = np.zeros((3, 2))
    assigned = np.empty(len(table), dtype=int)
    for i, weight in enumerate(weights):
        delta = ((current + weight - target) ** 2 - (current - target) ** 2).sum(axis=1)
        split = int(np.argmin(delta))
        assigned[i] = split
        current[split] += weight
    for _ in range(5):
        changed = False
        for i, weight in enumerate(weights):
            old = assigned[i]
            remaining = (current[old] - weight) * scale
            if remaining[1] < .5 or remaining[0] - remaining[1] < .5:
                continue
            remove_delta = ((current[old] - weight - target[old]) ** 2 -
                            (current[old] - target[old]) ** 2).sum()
            deltas = ((current + weight - target) ** 2 - (current - target) ** 2).sum(axis=1) + remove_delta
            deltas[old] = 0
            new = int(np.argmin(deltas))
            if deltas[new] < -1e-15:
                current[old] -= weight
                current[new] += weight
                assigned[i] = new
                changed = True
        if not changed:
            break
    return {trace_id: SPLITS[part] for trace_id, part in zip(table.index, assigned)}


def overlap_report(manifest: pd.DataFrame) -> dict:
    groups = {split: set(manifest.loc[manifest.split == split, "trace_signature"])
              for split in SPLITS}
    normal_kb = set(manifest.loc[(manifest.split == "train") &
                                (manifest.label == "Normal"), "trace_signature"])
    report = {"splits": {}, "pairwise_overlap": {}, "relative_to_train": {}}
    for split in SPLITS:
        part = manifest.loc[manifest.split == split]
        report["splits"][split] = {
            "blocks": len(part), "unique_sequences": len(groups[split]),
            "Normal": int((part.label == "Normal").sum()),
            "Anomaly": int((part.label == "Anomaly").sum()),
            "anomaly_prevalence": float((part.label == "Anomaly").mean()),
            "fraction_of_all_blocks": len(part) / len(manifest)}
    for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        report["pairwise_overlap"][f"{left}__{right}"] = {
            "shared_exact_sequences": len(groups[left] & groups[right]),
            "shared_block_ids": len(set(manifest.loc[manifest.split == left, "block_id"]) &
                                    set(manifest.loc[manifest.split == right, "block_id"]))}
    for split in ("validation", "test"):
        part = manifest.loc[manifest.split == split]
        seen = part.trace_signature.isin(groups["train"])
        report["relative_to_train"][split] = {
            "seen_unique_sequences": len(groups[split] & groups["train"]),
            "seen_unique_sequence_fraction": len(groups[split] & groups["train"]) / len(groups[split]),
            "seen_blocks": int(seen.sum()), "seen_block_fraction": float(seen.mean()),
            "unseen_blocks": int((~seen).sum()),
            "exact_match_in_normal_kb_blocks": int(part.trace_signature.isin(normal_kb).sum()),
            "seen_label_counts": part.loc[seen, "label"].value_counts().to_dict(),
            "unseen_label_counts": part.loc[~seen, "label"].value_counts().to_dict()}
        exact = part.trace_signature.isin(normal_kb)
        by_label = {}
        for label in ("Normal", "Anomaly"):
            mask = part.label.eq(label)
            denominator = int(mask.sum())
            by_label[label] = {
                "query_blocks": denominator,
                "exact_match_in_normal_kb": int((exact & mask).sum()),
                "exact_match_rate": float(exact[mask].mean()) if denominator else None,
                "seen_blocks": int((seen & mask).sum()),
                "seen_trace_rate": float(seen[mask].mean()) if denominator else None,
                "unseen_blocks": int((~seen & mask).sum()),
                "unseen_trace_rate": float((~seen[mask]).mean()) if denominator else None}
        report["relative_to_train"][split]["by_query_label"] = by_label
    return report


def build_kb(manifest: pd.DataFrame, payloads: dict) -> pd.DataFrame:
    """Canonicalize chỉ Normal train; count không được lấy từ toàn dataset."""
    normal_train = manifest.loc[(manifest.split == "train") & (manifest.label == "Normal")]
    rows = []
    for trace_id, group in normal_train.groupby("trace_signature", sort=True):
        events, templates = payloads[trace_id]
        rows.append({"trace_id": trace_id, "event_ids": json.dumps(events),
                     "template_text": "\n".join(f"{e}: {t}" for e, t in zip(events, templates)),
                     "occurrence_count": len(group),
                     "representative_block_id": group.block_id.min()})
    kb = pd.DataFrame(rows)
    if int(kb.occurrence_count.sum()) != len(normal_train) or kb.trace_id.duplicated().any():
        raise AssertionError("KB count hoặc canonicalization không nhất quán.")
    if not set(kb.representative_block_id).issubset(set(normal_train.block_id)):
        raise AssertionError("KB chứa representative ngoài Normal train.")
    return kb


def prepare(data_dir: Path, destination: Path, seed: int = 42,
            validation_fraction: float = .15, test_fraction: float = .15,
            chunksize: int = 20000) -> None:
    if destination.exists():
        raise FileExistsError(f"{destination} đã tồn tại. Chọn --output-dir mới để giữ dataset đã khóa.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".prepare-", dir=destination.parent))
    try:
        print("[1/4] Xác minh nhãn, template và ma trận...", flush=True)
        labels = pd.read_csv(data_dir / "anomaly_label.csv")
        templates = pd.read_csv(data_dir / "HDFS.log_templates.csv")
        matrix = pd.read_csv(data_dir / "Event_occurrence_matrix.csv")
        for frame, key, name in [(labels, "BlockId", "labels"),
                                 (templates, "EventId", "templates"),
                                 (matrix, "BlockId", "matrix")]:
            require_unique(frame, key, name)
        if set(labels.Label) != {"Normal", "Anomaly"}:
            raise ValueError("Nhãn nguồn không đúng Normal/Anomaly.")
        if templates.EventTemplate.isna().any() or templates.EventTemplate.str.strip().eq("").any():
            raise ValueError("Template rỗng hoặc thiếu.")
        truth = labels.set_index("BlockId").Label
        matrix = matrix.set_index("BlockId")
        if set(matrix.index) != set(truth.index):
            raise ValueError("BlockId giữa matrix và labels không khớp.")
        if not matrix.Label.map(LABELS).eq(truth.reindex(matrix.index)).all():
            raise ValueError("Nhãn matrix không khớp ground truth.")
        template_map = templates.set_index("EventId").EventTemplate.to_dict()
        events = sorted(template_map, key=lambda e: int(e[1:]))
        matrix_events = {c for c in matrix if c.startswith("E") and c[1:].isdigit()}
        if matrix_events != set(events):
            raise ValueError("Tập EventId giữa matrix và templates không khớp.")
        counts = matrix[events].apply(pd.to_numeric, errors="raise")
        values = counts.to_numpy()
        if not np.isfinite(values).all() or (values < 0).any() or (values % 1 != 0).any():
            raise ValueError("Ma trận có count thiếu/âm/không nguyên.")
        payloads, records, seen = {}, [], set()
        print("[2/4] Parse và đối chiếu từng BlockId trace...", flush=True)
        for chunk in pd.read_csv(data_dir / "Event_traces.csv", chunksize=chunksize):
            if chunk.BlockId.isna().any() or not chunk.BlockId.isin(truth.index).all():
                raise ValueError("Trace thiếu BlockId hoặc không có nhãn tham chiếu.")
            expected = counts.reindex(chunk.BlockId).to_numpy()
            for row, vector in zip(chunk.itertuples(index=False), expected):
                if row.BlockId in seen:
                    raise ValueError(f"BlockId trace bị trùng: {row.BlockId}")
                seen.add(row.BlockId)
                label = truth[row.BlockId]
                if LABELS.get(row.Label) != label:
                    raise ValueError(f"Nhãn trace không khớp: {row.BlockId}")
                sequence = parse_list(row.Features)
                if not sequence or not set(sequence).issubset(template_map):
                    raise ValueError(f"Trace rỗng hoặc EventId không có template: {row.BlockId}")
                frequency = Counter(sequence)
                if not np.array_equal([frequency[e] for e in events], vector):
                    raise ValueError(f"Số đếm trace/matrix không khớp: {row.BlockId}")
                trace_id = signature(sequence)
                if trace_id in payloads and payloads[trace_id][0] != sequence:
                    raise ValueError("Hash collision; không được gộp hai chuỗi khác nhau.")
                payloads.setdefault(trace_id, (sequence, [template_map[e] for e in sequence]))
                records.append((row.BlockId, trace_id, len(sequence), label))
            print(f"  {len(records):,} BlockIds đã kiểm tra", flush=True)
        if seen != set(truth.index):
            raise ValueError("Không phải tất cả BlockId có nhãn đều có trace.")
        meta = pd.DataFrame(records, columns=["block_id", "trace_signature", "sequence_length", "label"])
        meta = meta.sort_values("block_id").reset_index(drop=True)
        hist = meta.groupby(["trace_signature", "label"]).size().unstack(fill_value=0)
        conflicting = hist.loc[(hist.Normal > 0) & (hist.Anomaly > 0)].copy()
        conflicting.to_csv(staging / "conflicting_sequences.csv")
        meta["conflicting_trace"] = meta.trace_signature.isin(conflicting.index)
        print("[3/4] Xuất records theo BlockId và hai protocol...", flush=True)
        with (staging / "block_sequences.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["block_id", "trace_signature", "event_ids", "template_sequence",
                             "sequence_length", "label", "conflicting_trace"])
            # Serialize một lần mỗi pattern; CSV vẫn chứa một record cho mỗi BlockId.
            serialized = {key: (json.dumps(seq), json.dumps(text, ensure_ascii=False))
                          for key, (seq, text) in payloads.items()}
            for row in meta.itertuples(index=False):
                seq, text = serialized[row.trace_signature]
                writer.writerow([row.block_id, row.trace_signature, seq, text,
                                 row.sequence_length, row.label, row.conflicting_trace])
        protocol_reports = {}
        for protocol in ("random_block", "group_trace"):
            folder = staging / protocol
            folder.mkdir()
            manifest = make_split(meta, protocol, seed, validation_fraction, test_fraction)
            for split in SPLITS:
                manifest.loc[manifest.split == split].to_csv(folder / f"{split}_block_ids.csv", index=False)
            kb = build_kb(manifest, payloads)
            kb.to_csv(folder / "normal_knowledge_base.csv", index=False)
            report = overlap_report(manifest)
            report["knowledge_base"] = {"canonical_normal_traces": len(kb),
                                        "normal_train_blocks": int(kb.occurrence_count.sum())}
            save_json(folder / "split_overlap_report.json", report)
            # Cờ phục vụ đánh giá, không phải feature của query.
            train_groups = set(manifest.loc[manifest.split == "train", "trace_signature"])
            evaluation = manifest.loc[manifest.split != "train"].copy()
            evaluation["seen_in_train"] = evaluation.trace_signature.isin(train_groups)
            evaluation["exact_match_in_normal_kb"] = evaluation.trace_signature.isin(set(kb.trace_id))
            evaluation["conflicting_trace"] = evaluation.trace_signature.isin(conflicting.index)
            evaluation.to_csv(folder / "evaluation_annotations.csv", index=False)
            protocol_reports[protocol] = report
        print("[4/4] Lưu policy, provenance và quality report...", flush=True)
        config = {
            "policy_version": 2, "sample_unit": "BlockId", "preserve_event_order": True,
            "label_values": ["Normal", "Anomaly"], "label_source": "anomaly_label.csv",
            "knowledge_base_labels": ["Normal"], "canonicalize_exact_traces": True,
            "knowledge_base_split": "train", "occurrence_count_scope": "Normal train only",
            "use_type_feature": False, "use_time_features": False, "use_npz": False,
            "query_fields": ["event_ids", "template_sequence"],
            "not_query_features": ["block_id", "label", "conflicting_trace", "trace_signature"],
            "conflicting_trace_scope": "full dataset labels; post-hoc analysis only",
            "template_normalization": "identity: retain original template text and wildcards",
            "signature": "SHA256 of compact JSON ordered EventId list",
            "random_seed": seed, "validation_fraction": validation_fraction,
            "test_fraction": test_fraction, "train_fraction": 1 - validation_fraction - test_fraction,
            "random_block_algorithm": "sorted BlockIds, per-label seeded permutation; floor validation/test sizes",
            "group_trace_algorithm": "seeded tie order; largest normalized groups first; greedy squared-error minimization; up to 5 strictly improving whole-group move passes",
            "group_trace_objective": "sum over splits of ((blocks-target_blocks)/total_blocks)^2 + ((anomalies-target_anomalies)/total_anomalies)^2",
            "group_trace_stratified": "approximate block/anomaly balance; never break a group",
            "group_trace_fractions_apply_to": "BlockIds and Anomaly BlockIds",
            "seed_search_performed": False,
            "type_exclusion_reason": "Type missingness encodes label in this dataset; excluded along with derived missingness indicators",
            "comparison_policy": "M0-M3 share manifests within each protocol; M2-M3 share the same Normal-only KB",
            "metrics": {"primary": "anomaly-class F1", "also_report": ["Precision", "Recall", "confusion matrix", "class counts", "seen/unseen metrics"]},
            "threshold_selection": "validation only; test is not used to choose preprocessing",
            "source_files": {name: {"sha256": source_hash(data_dir / name),
                                    "bytes": (data_dir / name).stat().st_size} for name in SOURCE_FILES},
            "software": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__}}
        save_json(staging / "preprocessing_config.json", config)
        save_json(staging / "data_quality_report.json", {
            "status": "passed", "blocks": len(meta), "templates": len(templates),
            "unique_sequences": len(payloads), "label_counts": meta.label.value_counts().to_dict(),
            "conflicting_sequences": len(conflicting), "conflicting_blocks": int(meta.conflicting_trace.sum()),
            "checks_passed": ["unique nonmissing BlockIds", "exact BlockId coverage across CSVs",
                              "valid matching labels", "nonempty ordered sequences", "complete template mapping",
                              "trace counts equal matrix", "valid nonnegative integer matrix",
                              "disjoint BlockIds across splits", "complete manifest coverage",
                              "both classes in each split", "group protocol has no exact-sequence overlap",
                              "KB representatives and counts come only from Normal train"],
            "not_verified": ["NPZ contents vs CSV", "timing units", "Type semantics", "raw HDFS.log vs parsed CSV"],
            "raw_log_audit": "Optional follow-up investigation; not performed by this script"})
        readme = """# Experimental dataset HDFS

block_sequences.csv chứa một record mỗi BlockId. event_ids và template_sequence
là JSON arrays trong CSV: dùng json.loads để đọc, giữ nguyên thứ tự.
Nhãn giữ cách viết Normal/Anomaly từ CSV gốc.

random_block/ và group_trace/ là hai protocol độc lập, mỗi thư mục chứa ba
manifest, normal_knowledge_base.csv, split_overlap_report.json và evaluation_annotations.csv.
M0–M3 dùng chung manifest trong từng protocol; M2–M3 dùng cùng KB của protocol đó.
KB chỉ chứa Normal train, canonicalize exact sequence, chưa tạo embedding/index.

Random BlockId split stratify theo nhãn. Group split phân bổ nguyên nhóm theo
objective sai lệch số BlockId và Anomaly, mục tiêu 70/15/15 theo BlockId;
không bảo đảm optimum toàn cục. Xem phân bố thực tế trong split_overlap_report.json. Không tìm seed
theo kết quả test; báo cáo riêng hai protocol.

Chỉ event_ids và template_sequence là nội dung query. label, BlockId và các cờ
phân tích không được đưa vào embedding/prompt. conflicting_trace được suy ra
từ nhãn toàn dataset: chỉ dùng phân tích sau đánh giá, không dùng để xây KB,
lọc mẫu, chọn split, route query hoặc điều chỉnh mô hình. Cờ seen_in_train lấy
train gồm cả hai lớp; exact_match_in_normal_kb chỉ xét Normal train.

Normal-only KB không có nhãn đối lập để tính label-consensus: nếu nghiên cứu
adaptive routing, dùng tín hiệu tương đồng/cấu trúc đã chọn trên validation.
Không tự suy ra anomaly chỉ vì query chưa xuất hiện trong Normal KB.

Giữ nguyên các mẫu xung đột. Không dùng Type, timing hoặc NPZ làm feature.
Type và các cờ missingness của Type bị loại vì mã hóa nhãn của dataset.
Raw HDFS.log chưa được đối chiếu trong bước này. Không khẳng định đã kiểm tra
tất cả leakage: experiment còn phải kiểm soát prompt, feature và index.

preprocessing_config.json lưu policy, seed, source hashes và software versions.
Script từ chối ghi đè output đã tồn tại; dùng --output-dir mới để tái lập và so sánh.
Không lựa chọn preprocessing theo F1 test. Mọi thay đổi policy phải ghi phiên bản.
"""
        (staging / "README.md").write_text(readme, encoding="utf-8")
        staging.rename(destination)
        print(f"Hoàn tất: {destination.resolve()}", flush=True)
        for protocol, report in protocol_reports.items():
            print(protocol, json.dumps(report["splits"], ensure_ascii=False), flush=True)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=root / "preprocessed")
    parser.add_argument("--output-dir", type=Path, default=root / "processed_v2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=.15)
    parser.add_argument("--test-fraction", type=float, default=.15)
    parser.add_argument("--chunksize", type=int, default=20000)
    args = parser.parse_args()
    if args.chunksize <= 0 or args.seed < 0 or not (0 < args.validation_fraction < 1 and
            0 < args.test_fraction < 1 and args.validation_fraction + args.test_fraction < 1):
        parser.error("Cần chunksize > 0, seed >= 0, tỷ lệ validation/test > 0 và tổng < 1.")
    prepare(args.data_dir, args.output_dir, args.seed, args.validation_fraction, args.test_fraction, args.chunksize)


if __name__ == "__main__":
    main()
