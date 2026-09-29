# Experiment Guide

Tài liệu này mô tả pipeline hiện tại sau khi tái cấu trúc repo.

## Protocol hiện tại

- Package chạy experiment: `experiments/`
- Config hệ thống: `experiments/configs/config_system.json`
- Config thiết kế thí nghiệm: `experiments/configs/config_experiment.json`
- Dataset chính: `data/processed_v2/group_trace`
- Output local: `experiments/_generated/`

Validation dùng để khóa cấu hình. Test chỉ dùng sau khi validation đã khóa.

## Methods

| Method | Ý nghĩa |
|---|---|
| M0 | Sequence n-gram KNN baseline, không dùng LLM |
| M1 | LLM-only baseline |
| M2 | Semantic retrieval + fixed top-k sweep |
| M3 | Semantic retrieval + adaptive threshold |
| M4 | Structure-aware retrieval + fixed top-k sweep |
| M5 | Structure-aware retrieval + adaptive threshold |

M2-M5 chạy trên 3 KB variants: `normal`, `anomaly`, `mixed`.

Chi tiết phương pháp nằm trong `docs/METHODS.md`.

## Lệnh chạy chính

```bash
.venv/bin/python -m experiments.run prepare --model-profile qwen-local
.venv/bin/python -m experiments.run embed --model-profile qwen-local
.venv/bin/python -m experiments.run run --model-profile qwen-local --split validation
.venv/bin/python -m experiments.run report --model-profile qwen-local --split validation
```

Sau khi validation đã khóa:

```bash
.venv/bin/python -m experiments.run run --model-profile qwen-local --split test
.venv/bin/python -m experiments.run report --model-profile qwen-local --split test
```

Không nên dùng `all` trong giai đoạn debug vì nó chạy validation rồi full test.

## Output quan trọng

```text
experiments/_generated/artifacts_protocol_v4/
experiments/_generated/results_protocol_v4/
experiments/_generated/cache_protocol_v4/
```

Report tạo:

```text
comparison_validation*.csv
comparison*.csv
confusion_matrices/<split>/
```

Notebook dashboard:

```text
experiments/notebooks/report_dashboard.ipynb
```

## Lưu ý reproducibility

- Mọi method dùng cùng query manifest cho cùng split.
- Label không đi vào embedding text.
- Adaptive threshold được chọn trên validation, không chọn trên test.
- Fixed top-k lưu toàn bộ sweep để so sánh, không chỉ một k duy nhất.
