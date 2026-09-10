"""Khám phá HDFS trước khi xây dựng Enhanced RAG cho log anomaly detection.

Chạy: python -m scripts.data_understanding
Phụ thuộc: pandas, numpy, matplotlib (requirements-data-understanding.txt).
Không sửa dữ liệu nguồn, không huấn luyện mô hình, không xây retrieval index.
Các thống kê nhãn chỉ phục vụ EDA; không đưa nhãn/Type vào query của mô hình.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import zipfile

import numpy as np
import pandas as pd


LABELS = {"Success": "Normal", "Fail": "Anomaly",
          "Normal": "Normal", "Anomaly": "Anomaly"}

# Mỗi key dưới đây luôn được ghi, kể cả khi số vi phạm bằng 0.
QUALITY_KEYS = (
    "matrix_events_without_template", "templates_without_matrix_column",
    "labels_without_matrix", "matrix_without_labels", "unknown_ground_truth_labels",
    "unknown_matrix_labels", "unknown_trace_labels", "matrix_label_disagreement",
    "invalid_matrix_cells", "duplicate_trace_block_ids", "missing_trace_block_ids",
    "trace_label_disagreement", "malformed_features", "empty_traces",
    "unknown_event_tokens", "trace_matrix_count_disagreement", "interval_length_disagreement",
    "invalid_intervals", "invalid_latency", "latency_vs_interval_sum_disagreement",
    "malformed_time_fields", "labels_without_trace", "traces_without_labels", "matrix_without_trace",
)
TIME_RTOL = 0
TIME_ATOL = 1e-6


def parse_list(value: str) -> list[str]:
    """Features dùng [E5,E22,...], không phải JSON; không sử dụng eval."""
    value = str(value).strip()
    if not (value.startswith("[") and value.endswith("]")):
        raise ValueError(f"Danh sách sai định dạng: {value[:80]}")
    if not value[1:-1].strip():
        return []
    items = [v.strip() for v in value[1:-1].split(",")]
    if any(not item for item in items):
        raise ValueError("Danh sách có phần tử rỗng hoặc dấu phẩy dư")
    return items


def require_unique(frame: pd.DataFrame, key: str, name: str) -> None:
    if frame[key].isna().any() or frame[key].duplicated().any():
        raise ValueError(f"{name}: {key} thiếu hoặc trùng; cần xử lý trước khi join.")


def run(data_dir: Path, output: Path, chunksize: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    quality = Counter({key: 0 for key in QUALITY_KEYS})
    inventory = []
    for path in sorted(data_dir.iterdir()):
        if path.is_file():
            inventory.append({"file": path.name, "bytes": path.stat().st_size})
    pd.DataFrame(inventory).to_csv(output / "file_inventory.csv", index=False)

    print("[1/5] Đọc templates, nhãn và ma trận sự kiện...", flush=True)
    templates = pd.read_csv(data_dir / "HDFS.log_templates.csv")
    labels = pd.read_csv(data_dir / "anomaly_label.csv")
    matrix = pd.read_csv(data_dir / "Event_occurrence_matrix.csv")
    for frame, key, name in [(templates, "EventId", "templates"),
                             (labels, "BlockId", "labels"),
                             (matrix, "BlockId", "matrix")]:
        require_unique(frame, key, name)
    events = [c for c in matrix if re.fullmatch(r"E\d+", c)]
    template_map = templates.set_index("EventId")["EventTemplate"].to_dict()
    quality["matrix_events_without_template"] = len(set(events) - set(template_map))
    quality["templates_without_matrix_column"] = len(set(template_map) - set(events))
    truth = labels.set_index("BlockId")["Label"]
    matrix = matrix.set_index("BlockId")
    quality["labels_without_matrix"] = len(set(truth.index) - set(matrix.index))
    quality["matrix_without_labels"] = len(set(matrix.index) - set(truth.index))
    quality["unknown_ground_truth_labels"] = int((~truth.isin(["Normal", "Anomaly"])).sum())
    quality["unknown_matrix_labels"] = int((~matrix.Label.isin(LABELS)).sum())
    aligned = truth.reindex(matrix.index)
    quality["matrix_label_disagreement"] = int((matrix.Label.map(LABELS) != aligned).sum())
    counts = matrix[events].apply(pd.to_numeric, errors="coerce")
    invalid = counts.isna() | (counts < 0) | (counts % 1 != 0)
    quality["invalid_matrix_cells"] = int(invalid.sum().sum())
    if quality["invalid_matrix_cells"]:
        raise ValueError("Ma trận chứa số đếm thiếu, âm hoặc không nguyên.")
    missing = matrix.isna().sum().rename("missing").to_frame()
    missing["fraction"] = missing["missing"] / len(matrix)
    missing.to_csv(output / "matrix_missing_values.csv")
    stats = templates.set_index("EventId").reindex(events)
    for label in ["Normal", "Anomaly"]:
        subset = counts.loc[aligned == label]
        stats[f"{label}_occurrences"] = subset.sum()
        stats[f"{label}_blocks"] = (subset > 0).sum()
        stats[f"{label}_prevalence"] = (subset > 0).mean()
    stats["prevalence_difference"] = stats.Anomaly_prevalence - stats.Normal_prevalence
    stats.to_csv(output / "event_statistics.csv")

    print("[2/5] Phân tích toàn bộ trace theo từng chunk...", flush=True)
    patterns = defaultdict(Counter)
    transitions = {label: Counter() for label in ["Normal", "Anomaly", "Unknown"]}
    transition_blocks = {label: Counter() for label in transitions}
    seen = set()
    features = []
    examples = defaultdict(list)
    trace_missing = Counter()
    for chunk in pd.read_csv(data_dir / "Event_traces.csv", chunksize=chunksize):
        trace_missing.update(chunk.isna().sum().to_dict())
        expected_rows = counts.reindex(chunk.BlockId).to_numpy()
        for row, expected in zip(chunk.itertuples(index=False), expected_rows):
            block = row.BlockId
            quality["missing_trace_block_ids"] += int(pd.isna(block))
            quality["unknown_trace_labels"] += int(row.Label not in LABELS)
            quality["duplicate_trace_block_ids"] += int(block in seen)
            seen.add(block)
            label = truth.get(block, "Unknown")
            if label not in transitions:
                label = "Unknown"
            quality["trace_label_disagreement"] += int(LABELS.get(row.Label) != label)
            try:
                sequence = parse_list(row.Features)
            except ValueError:
                quality["malformed_features"] += 1
                continue
            event_counts = Counter(sequence)
            quality["empty_traces"] += int(not sequence)
            quality["unknown_event_tokens"] += sum(v for k, v in event_counts.items() if k not in template_map)
            quality["trace_matrix_count_disagreement"] += int(
                not np.array_equal([event_counts[e] for e in events], expected)
                or bool(set(sequence) - set(events)))
            try:
                intervals = np.asarray([float(v) for v in parse_list(row.TimeInterval)])
                latency = float(row.Latency)
                # N sự kiện có N-1 khoảng thời gian giữa các sự kiện liên tiếp.
                quality["interval_length_disagreement"] += int(len(intervals) != max(len(sequence) - 1, 0))
                quality["invalid_intervals"] += int((~np.isfinite(intervals) | (intervals < 0)).sum())
                quality["invalid_latency"] += int(not np.isfinite(latency) or latency < 0)
                quality["latency_vs_interval_sum_disagreement"] += int(not np.isclose(
                    intervals.sum(), latency, rtol=TIME_RTOL, atol=TIME_ATOL))
            except (ValueError, TypeError):
                quality["malformed_time_fields"] += 1
                latency = float("nan")
            patterns[tuple(sequence)][label] += 1
            transitions[label].update(zip(sequence, sequence[1:]))
            transition_blocks[label].update(set(zip(sequence, sequence[1:])))
            features.append((block, label, row.Type, len(sequence), len(event_counts), latency))
            if len(examples[label]) < 2:
                examples[label].append({"BlockId": block, "label_for_analysis_only": label,
                                        "events": sequence,
                                        "template_sequence": [template_map.get(e, e) for e in sequence]})
    quality["labels_without_trace"] = len(set(truth.index) - seen)
    quality["traces_without_labels"] = len(seen - set(truth.index))
    quality["matrix_without_trace"] = len(set(matrix.index) - seen)
    frame = pd.DataFrame(features, columns=["BlockId", "Label", "Type", "length", "unique_events", "latency"])
    frame.to_csv(output / "trace_features.csv", index=False)
    summary = frame.groupby("Label")[["length", "unique_events", "latency"]].describe(percentiles=[.5, .9, .95, .99])
    summary.to_csv(output / "trace_summary_by_label.csv")
    pd.Series(trace_missing, name="missing").to_csv(output / "trace_missing_values.csv")
    frame.groupby(["Label", "Type"], dropna=False).size().rename("blocks").to_csv(output / "type_distribution.csv")
    type_rows = []
    for source, values, source_labels in [("matrix", matrix.Type, aligned),
                                          ("traces", frame.Type, frame.Label)]:
        for label in ("Normal", "Anomaly"):
            selected = values.loc[source_labels == label]
            type_rows.append({"source": source, "label": label, "blocks": len(selected),
                              "Type_missing": int(selected.isna().sum()),
                              "Type_present": int(selected.notna().sum())})
    pd.DataFrame(type_rows).to_csv(output / "type_missingness_by_label.csv", index=False)

    print("[3/5] Kiểm tra chuỗi trùng, nhãn xung đột và transitions...", flush=True)
    pattern_rows = [{"sequence": " ".join(seq), "blocks": sum(c.values()),
                     "Normal": c["Normal"], "Anomaly": c["Anomaly"],
                     "Unknown": c["Unknown"]} for seq, c in patterns.items()]
    pattern_df = pd.DataFrame(pattern_rows).sort_values("blocks", ascending=False)
    pattern_df.to_csv(output / "sequence_patterns.csv", index=False)
    conflicts = pattern_df[(pattern_df.Normal > 0) & (pattern_df.Anomaly > 0)]
    conflicts.to_csv(output / "conflicting_sequences.csv", index=False)
    transition_rows = [{"label": label, "from": a, "to": b, "count": count,
                        "fraction_within_class": count / sum(counter.values())}
                       for label, counter in transitions.items() for (a, b), count in counter.most_common()]
    pd.DataFrame(transition_rows).to_csv(output / "transitions.csv", index=False)
    all_pairs = sorted(set().union(*(set(counter) for counter in transitions.values())))
    transition_prevalence = []
    class_sizes = frame.Label.value_counts()
    for a, b in all_pairs:
        record = {"from": a, "to": b}
        for label in ("Normal", "Anomaly"):
            containing = transition_blocks[label][a, b]
            record[f"{label}_blocks_containing_transition"] = containing
            record[f"{label}_transition_prevalence"] = containing / class_sizes[label] if class_sizes.get(label, 0) else float("nan")
        transition_prevalence.append(record)
    pd.DataFrame(transition_prevalence).to_csv(output / "transition_block_prevalence.csv", index=False)
    (output / "trace_examples.json").write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8")
    # Đọc header NPY trong NPZ, không unpickle hoặc tải các mảng object.
    npz_metadata = []
    npz_path = data_dir / "HDFS.npz"
    if npz_path.exists():
        with zipfile.ZipFile(npz_path) as archive:
            for name in archive.namelist():
                if not name.endswith(".npy"):
                    continue
                with archive.open(name) as stream:
                    version = np.lib.format.read_magic(stream)
                    if version == (1, 0):
                        shape, order, dtype = np.lib.format.read_array_header_1_0(stream)
                    elif version == (2, 0):
                        shape, order, dtype = np.lib.format.read_array_header_2_0(stream)
                    else:
                        npz_metadata.append({"name": name, "unsupported_header_version": version})
                        continue
                    npz_metadata.append({"name": name, "shape": shape, "dtype": str(dtype), "fortran_order": order})
    (output / "npz_metadata.json").write_text(json.dumps(npz_metadata, indent=2), encoding="utf-8")
    (output / "quality_checks.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    (output / "quality_check_schema.json").write_text(json.dumps({
        "schema_version": 2, "checked_keys": QUALITY_KEYS,
        "zero_means": "checked, no violation found", "time_rtol": TIME_RTOL,
        "time_atol": TIME_ATOL, "time_units": "unverified",
        "key_and_schema_checks": "fail-fast; no successful report if required columns or unique keys are invalid",
    }, indent=2), encoding="utf-8")

    print("[4/5] Xuất 6 biểu đồ EDA...", flush=True)
    from scripts.eda_visuals import make_eda_plots
    figure_paths = make_eda_plots(output)
    plot_status = f"Đã tạo {len(figure_paths)} biểu đồ trong figures/."

    print("[5/5] Viết báo cáo nghiên cứu...", flush=True)
    distribution = truth.value_counts()
    n = len(truth)
    duplicated = int(pattern_df.loc[pattern_df.blocks > 1, "blocks"].sum())
    report = f"""# Data Understanding — HDFS và Enhanced RAG

## 1. Phạm vi và đơn vị phân tích
Phân tích toàn bộ CSV tại `{data_dir.resolve()}`; không lấy mẫu.
Theo docs/dataset.md của dataset, các log được nhóm theo BlockId và gán nhãn ở cấp block.
Một trace là chuỗi sự kiện của một block; nhãn Anomaly không có nghĩa mọi dòng trong trace đều bất thường.
Script chỉ kiểm tra metadata của HDFS.npz; chưa xác minh nội dung NPZ khớp CSV.

## 2. Các file và ý nghĩa
- HDFS.log_templates.csv: EventId → mẫu thông điệp; [*] đại diện phần biến đổi đã bị loại khỏi template.
- anomaly_label.csv: nhãn Normal/Anomaly theo BlockId, dùng làm nhãn tham chiếu khi đối chiếu.
- Event_traces.csv: Features giữ thứ tự sự kiện; TimeInterval và Latency là đặc trưng thời gian.
- Kiểm tra TimeInterval có N-1 phần tử cho trace N sự kiện và tổng khoảng thời gian khớp Latency.
- Event_occurrence_matrix.csv: số lần xuất hiện từng EventId, mất thông tin thứ tự.
- Success được ánh xạ sang Normal, Fail sang Anomaly và kiểm tra lại bằng BlockId.
- Type bị loại cùng mọi indicator missingness: trạng thái có/thiếu của nó mã hóa nhãn trong dataset này.
  Ý nghĩa mã chưa xác minh; chỉ phân tích hậu nghiệm sau khi có tài liệu giải nghĩa.
- Chưa xác minh đơn vị thời gian từ mã tiền xử lý; không mặc định giây hoặc mili giây.

## 3. Thống kê đo được
- Số block có nhãn: {n:,}.
- Normal: {distribution.get('Normal', 0):,}; Anomaly: {distribution.get('Anomaly', 0):,} ({distribution.get('Anomaly', 0) / n:.2%}).
- Số template: {len(templates)}; số cột sự kiện: {len(events)}.
- Số trace phân tích được: {len(frame):,}; số chuỗi sự kiện phân biệt: {len(pattern_df):,}.
- Số trace thuộc nhóm chuỗi xuất hiện nhiều lần: {duplicated:,}.
- Số chuỗi giống hệt nhau nhưng có cả hai nhãn: {len(conflicts):,}, bao phủ {int(conflicts.blocks.sum()):,} trace.

Thống kê độ dài, số loại sự kiện và latency (gồm p50/p90/p95/p99): trace_summary_by_label.csv.
event_statistics.csv ghi cả tần suất tuyệt đối và tỷ lệ block chứa sự kiện trong từng lớp;
chênh lệch prevalence chỉ là liên hệ thống kê trên dataset, không chứng minh nguyên nhân.
transitions.csv đếm cặp sự kiện liền kề trong từng trace, không nối giữa các block.
transition_block_prevalence.csv đếm mỗi cặp tối đa một lần mỗi BlockId và chia cho số block từng lớp;
khác với fraction_within_class có trọng số theo số lần transition xuất hiện.

## 4. Kiểm tra chất lượng
Giá trị dưới đây là số vi phạm/khác biệt, trừ các trường tên rõ là số template hoặc token.
Schema kiểm tra cố định nằm trong quality_check_schema.json; mọi key đều xuất hiện, kể cả 0.
So sánh tổng TimeInterval với Latency dùng rtol=0, atol=1e-6; đây là kiểm tra giả thuyết về cách tiền xử lý,
khác biệt chưa đủ để kết luận dữ liệu sai. Type thiếu có thể là có chủ đích.

```json
{json.dumps(quality, indent=2)}
```

## Phạm vi experiment và các mục chưa xác minh

Ba mục sau vẫn **chưa được xác minh**; các kiểm tra chất lượng đã chạy không bao phủ chúng:

| Mục | Trạng thái hiện tại | Khi nào cần xác minh thêm? |
| --- | --- | --- |
| Nội dung `HDFS.npz` có khớp CSV | Mới đọc metadata (shape, dtype); chưa đối chiếu các phần tử và nhãn | Trước khi dùng NPZ thay cho CSV, cần xác minh cả thứ tự mẫu và ánh xạ nhãn |
| Đơn vị `TimeInterval` và `Latency` | Đã kiểm tra số khoảng thời gian và tổng khoảng thời gian khớp Latency; điều này không xác định được đơn vị | Trước khi dùng đặc trưng thời gian, đặt ngưỡng theo thời gian hoặc diễn giải kết quả thời gian |
| Ý nghĩa `Type` | Mới thống kê mã và giá trị thiếu; chưa có ánh xạ mã sang loại bất thường được xác minh | Trước khi đánh giá theo loại lỗi hoặc dùng Type để diễn giải nguyên nhân |

**Ba mục này không ngăn cản experiment chỉ sử dụng EventId và template từ CSV.**
Phạm vi đầu vào của experiment này là chuỗi `Features` trong `Event_traces.csv` và ánh xạ
`EventId → EventTemplate` trong `HDFS.log_templates.csv`. `BlockId` dùng để nối và quản lý mẫu;
nhãn trong `anomaly_label.csv` phục vụ huấn luyện/đánh giá theo thiết kế thí nghiệm,
không được đưa nhãn của query đánh giá vào prompt hoặc embedding.
Không sử dụng `HDFS.npz`, `TimeInterval`, `Latency` hoặc `Type` làm đầu vào trong phạm vi này.

Vẫn cần tách train/validation/test trước khi xây retrieval index, kiểm soát chuỗi trùng giữa các tập
và giữ nhãn test độc lập. Kết quả chỉ phản ánh biểu diễn EventId/template;
chưa đủ để kết luận về bất thường thời gian hoặc nguyên nhân/loại lỗi cụ thể.

## 5. Implications for Data Preparation and Experimental Design
- Một BlockId là một sample; giữ nguyên ordered EventId sequence và mapping template.
- Occurrence matrix mất thứ tự; event prevalence khác nhau giữa lớp là động lực để đánh giá baseline đơn giản,
  chưa chứng minh baseline hoặc LLM/RAG có hiệu quả đến mức nào.
- Tách train/validation/test trước khi xây KB; KB chỉ canonicalize Normal train, giữ occurrence_count.
- Group-by-trace v2 là protocol experiment chính; Random BlockId chỉ giữ làm đối chiếu EDA.
- Giữ mẫu xung đột; cờ suy ra từ nhãn toàn dataset chỉ dùng phân tích hậu nghiệm.
- Type và các indicator missingness của Type bị loại khỏi model input vì mã hóa nhãn trong dữ liệu này.
  Xem type_missingness_by_label.csv để kiểm chứng cả hai nguồn CSV; ý nghĩa mã vẫn chưa xác minh.
- Primary metric: anomaly-class F1; thêm Precision, Recall, confusion matrix và phân bố test.
  Experiment dùng subset cố định 400 Normal/100 Anomaly; Precision/F1 phản ánh subset 20% Anomaly. Không chọn preprocessing hoặc ngưỡng bằng test score.
- Các representation có thể được đánh giá trong experiment; EDA không kết luận representation nào tốt hơn.
  Thiết kế tối thiểu M0–M3 và giới hạn phạm vi nằm trong docs/research_design.md.

## 6. Sản phẩm và cách đọc
Bắt đầu bằng quality_checks.json, sau đó trace_summary_by_label.csv, event_statistics.csv,
sequence_patterns.csv và conflicting_sequences.csv. trace_examples.json chứa hai ví dụ đầu mỗi lớp,
chỉ để minh họa biểu diễn, không phải mẫu đại diện ngẫu nhiên. {plot_status}
Đây là EDA toàn bộ dataset; trong thí nghiệm đánh giá chặt chẽ, các quyết định điều chỉnh mô hình
phải dựa trên train/validation và giữ test độc lập.
"""
    (output / "Data_Understanding.md").write_text(report, encoding="utf-8")
    print(f"Hoàn tất: {output.resolve() / 'Data_Understanding.md'}")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=root / "preprocessed")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs" / "data_understanding")
    parser.add_argument("--chunksize", type=int, default=20000)
    args = parser.parse_args()
    if args.chunksize <= 0:
        parser.error("--chunksize phải lớn hơn 0")
    if args.output_dir.resolve() == args.data_dir.resolve():
        parser.error("Thư mục output phải khác thư mục dữ liệu nguồn")
    run(args.data_dir, args.output_dir, args.chunksize)


if __name__ == "__main__":
    main()
