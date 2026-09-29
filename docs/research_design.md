# Research Design

Mục tiêu nghiên cứu là phát hiện bất thường log HDFS ở mức **BlockId ordered EventId sequence**.

Nghiên cứu kiểm tra hai câu hỏi chính:

1. Retrieval chỉ dựa trên semantic similarity có đủ tốt cho log anomaly detection không?
2. Adaptive context selection còn hữu ích không khi retrieval được cải thiện bằng thông tin cấu trúc log?

## Trục thí nghiệm

Pipeline tách ba yếu tố:

| Yếu tố | Giá trị |
|---|---|
| Retrieval | semantic, structure-aware |
| Context selection | fixed top-k sweep, adaptive threshold |
| KB composition | normal-only, anomaly-only, mixed |

## Methods

| Method | Retrieval | Selection | KB |
|---|---|---|---|
| M0 | sequence n-gram KNN | validation-selected k | labeled train references |
| M1 | none | none | none |
| M2 | semantic | fixed top-k sweep | normal/anomaly/mixed |
| M3 | semantic | adaptive threshold | normal/anomaly/mixed |
| M4 | structure-aware | fixed top-k sweep | normal/anomaly/mixed |
| M5 | structure-aware | adaptive threshold | normal/anomaly/mixed |

M0 và M1 là baseline. M2-M5 là các biến thể RAG để tách contribution của retrieval và context selection.

## Data protocol

- Dataset chính: `data/processed_v2/group_trace`
- Validation: dùng để chọn k/threshold và quan sát fixed-k sweep
- Test: chỉ dùng sau khi cấu hình đã khóa
- Cùng một query manifest được dùng cho mọi M0-M5 và mọi model profile
- Label không được đưa vào embedding; label chỉ được gắn lại sau retrieval khi tạo context cho LLM

Chi tiết nằm trong `docs/DATA_PROTOCOL.md`.

## Metrics

Metric chính là anomaly-class F1. Báo cáo thêm:

- Precision, Recall
- TP/TN/FP/FN và confusion matrix
- average contexts
- prompt tokens / latency khi có LLM usage
- adaptive-vs-fixed deltas

## Scope control

Không mở rộng sang nhiều dataset, nhiều vector database, router, learned classifier hoặc timing feature trong protocol hiện tại. Các output EDA nằm ở `reports/data_understanding/` và không phải model source code.
