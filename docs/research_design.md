# Experiment tối thiểu — HDFS Enhanced RAG

Thiết kế này thay thế các đề xuất mở rộng trước đây. Data Understanding đã hoàn tất;
câu hỏi nghiên cứu là: số context thích ứng theo similarity có cải thiện anomaly F1
hoặc giảm prompt tokens so với luôn dùng ba context hay không?
Chưa chạy embedding, LLM hoặc đo kết quả model.

## Dữ liệu và phạm vi

Chỉ dùng **Group-by-trace split v2** cho experiment chính:

| Split | BlockIds | Normal | Anomaly |
| --- | ---: | ---: | ---: |
| Train | 402.543 | 390.757 | 11.786 |
| Validation | 86.259 | 83.733 | 2.526 |
| Test | 86.259 | 83.733 | 2.526 |

Không chồng BlockId hoặc exact ordered sequence giữa các tập. Các file nằm trong
`processed_v2/group_trace/`. Random split được giữ làm bằng chứng EDA: 97,68% random
test BlockIds có pattern đã xuất hiện trong train; không chạy bộ LLM experiment trên protocol này.

Test experiment là **500 BlockIds cố định: 400 Normal, 100 Anomaly**, lấy ngẫu nhiên
không hoàn lại trong từng lớp từ group_trace test, seed 42. M0–M3 dùng cùng manifest
`experiments/artifacts/queries/test_queries.csv`. Lấy mẫu theo BlockId, không cân bằng theo unique pattern,
không lựa chọn mẫu theo retrieval score hoặc kết quả model. Nhãn trong manifest chỉ dùng đánh giá.

> Test subset được lấy mẫu phân tầng với tỷ lệ 80% Normal/20% Anomaly để có đủ anomaly
> phục vụ so sánh. Precision và F1 phản ánh subset này, không đại diện trực tiếp cho
> prevalence 2,93% của toàn HDFS_v1 hoặc 2,928% của group-trace test đầy đủ.

## Representation và knowledge base

Query text giữ ordered EventIds và template text gốc, cùng một định dạng cho mọi phương pháp.
Không đưa label, Type, indicator missingness, timing, BlockId hoặc cờ conflict vào model input.
M2/M3 dùng cùng BGE embedding model và cosine similarity; M0 dùng 29-dimensional occurrence vectors, L2 normalization và cosine distance.

M2 và M3 dùng đúng **một Normal KB** đã tạo:
390.757 Normal train BlockIds → **4.759 canonical ordered sequences**.
`occurrence_count` và representative_block_id chỉ là metadata thống kê/truy nguyên;
không đưa vào embedding, prompt, similarity weighting hoặc voting của RAG.
Canonicalization đã hoàn tất ở Data Preparation, không thực hiện runtime dedup và không coi là đóng góp M3.

M0 cần **labeled train neighbors** của cả hai lớp, nên không dùng riêng Normal KB của M2/M3.
M0 truy vấn count vectors của tất cả train BlockIds có nhãn gốc, giữ cả các vectors trùng nhau.

## Bốn phương pháp duy nhất

| Phương pháp | Chính sách |
| --- | --- |
| M0 — KNN | Occurrence vector → cosine KNN với k chọn trên validation → distance-weighted vote → NORMAL/ANOMALY; không gọi LLM |
| M1 — LLM-only | Query → LLM → NORMAL/ANOMALY; 0 historical contexts |
| M2 — Fixed top-3 RAG | Query → retrieve 3 Normal contexts gần nhất → luôn đưa cả 3 vào LLM |
| M3 — Adaptive RAG | Query → retrieve top-10 Normal candidates → giữ score ≥ threshold → lấy tối đa 3 → LLM |

KNN chọn k trong {1,3,5} trên 200 validation queries (160 Normal/40 Anomaly): F1 cao nhất, rồi Recall cao nhất, rồi k nhỏ nhất.
Sắp candidate theo similarity giảm dần, dùng ID làm tie-break để tái lập.
M1–M3 dùng cùng API LLM, instruction phân loại, cấu hình sinh và định dạng đầu ra;
sự khác biệt là phần historical contexts. Không khẳng định label chỉ vì exact match hoặc thiếu exact match.

M3 có thể đưa 0, 1, 2 hoặc 3 contexts vào prompt. Nếu không có candidate đạt threshold,
phần historical context để trống; vẫn gọi LLM. **M3 không giảm số lần gọi LLM trong thiết kế này**.
Adaptive chỉ điều chỉnh số contexts, không có detector/router/fallback module riêng.

Threshold candidates là Q25/Q50/Q75 của pooled top-10 scores trên 200 validation queries. Chọn F1 (6 chữ số) cao nhất, rồi ít contexts hơn, rồi threshold thấp hơn; khóa trước test.

## Metrics và số lần gọi

Báo cáo Precision, Recall, anomaly-class F1, confusion matrix (TP/TN/FP/FN),
average contexts, average prompt tokens và average latency.
Không cần PR-AUC vì LLM trả label, không có anomaly score đáng tin cậy.

Contexts là số historical contexts thực sự đưa vào LLM, không phải số candidates retrieve.
M0/M1 có 0 LLM contexts; M0 không có LLM prompt tokens. Prompt tokens lấy từ usage thực tế
của API khi có, gồm instruction, query và contexts; nếu chỉ ước lượng phải ghi rõ.
Latency đo xử lý query (embedding/retrieval/LLM theo phương pháp), không tính xây index offline;
ghi rõ chính sách cache để kết quả so sánh có ý nghĩa. Không bỏ âm thầm output lỗi hoặc không parse được.

Với 500 queries và một API LLM:

- M0: 0 LLM calls.
- M1/M2/M3: 500 calls mỗi phương pháp.
- Tổng danh nghĩa **1.500 test calls**, chưa tính validation để chọn threshold hoặc retry khi lỗi.

Một local model nhỏ chỉ là demo/bổ sung nếu còn thời gian, không phải điều kiện hoàn thành experiment chính.
API tương thích OpenAI được cấu hình trong `experiments/.env`. Embedding dùng BAAI/bge-small-en-v1.5; tuning theo quantile trên 200 validation queries. Chưa phát sinh LLM API calls.

## Giới hạn phạm vi

Không bổ sung structural reranking, BM25/hybrid, token-budget packing, runtime dedup,
stage-1 detector, learned router, nhiều vector database, baseline hoặc split mới.
Không mở rộng EDA, raw-log audit, NPZ verification, giải nghĩa Type hoặc nghiên cứu timing.
Notebook/CSV là tài liệu tái lập; phần Data Understanding trong luận văn chỉ cần khoảng 1–2 trang.

Code và hướng dẫn chạy: docs/experiment.md. Primary P/R/F1 không được báo cáo như kết quả đầy đủ nếu còn INVALID_OUTPUT.
