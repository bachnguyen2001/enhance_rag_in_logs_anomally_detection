# HDFS experiments M0–M5

Chạy toàn bộ từ chuẩn bị dữ liệu đến báo cáo bằng một lệnh:

```bash
.venv/bin/python -m experiments_v2.run all
```

Mặc định chỉ dùng **qwen-local**, endpoint trong `config_system.json`.
Lệnh thực hiện prepare → embed → validation → test → report.
Chạy lại cùng lệnh sẽ kiểm tra và sử dụng artifact, threshold và kết quả đã hoàn tất;
các request đã có trong cache không gọi LLM lại.

## Hai config

- `config_system.json`: đường dẫn input/artifact/cache/output; model profiles,
  temperature/token limit/timeout/retry; embedding model, device, batch size.
  Input paths tính từ repo root; output/cache/env paths tính từ `experiments_v2/`.
  API key nằm trong `.env`. `device: "cuda"` yêu cầu PyTorch nhận GPU;
  đổi thành `"cpu"` nếu chạy CPU.
- `config_experiment.json`: models, methods, sampling, KNN k, thuật toán và
  trọng số retrieval, số context Fixed và tham số Adaptive.

Muốn chạy cả hai model, đặt `"models": ["qwen-local", "gemini"]`.
Có thể chỉ định profile cho một lần chạy:

```bash
.venv/bin/python -m experiments_v2.run all --model-profile qwen-local
```

## Pipeline và tham số

| Method | Retrieval | Selection | Dữ liệu |
|---|---|---|---|
| M0 | KNN sequence n-gram hash vectors | k chọn trên validation | Canonical labeled train sequences; chạy một lần dùng chung |
| M1 | Không | Không context | Mỗi model một lần |
| M2 | Semantic cosine | Fixed | Tự chạy Normal, Anomaly, Mixed |
| M3 | Semantic cosine | Adaptive | Tự chạy Normal, Anomaly, Mixed |
| M4 | Structure-aware | Fixed | Tự chạy Normal, Anomaly, Mixed |
| M5 | Structure-aware | Adaptive | Tự chạy Normal, Anomaly, Mixed |

Một model có 12 RAG experiments + M1; M0 dùng chung. Hai model có 24 RAG
experiments + 2 M1 + 1 M0. Validation screening phát sinh thêm lượt chạy.

Retrieval chấm điểm **toàn KB**. Selection dùng chung một candidate pool kích
thước `selection.adaptive.candidate_pool` cho Adaptive:

- Fixed lấy `selection.fixed.contexts` reference đứng đầu.
- Adaptive lấy top candidate pool, lọc `score >= threshold`, rồi giữ các
  reference vượt ngưỡng theo thứ tự score giảm dần. Có thể lấy 0 context và vẫn
  gọi LLM. Không dedup runtime hay token-budget packing.
- Threshold thử tại các `threshold_quantiles` của **top candidate pool trên
  validation**, chọn F1 cao nhất, rồi ít context hơn, rồi threshold thấp hơn.
  Khóa riêng theo model, KB và retrieval; không tune trên test.

Structure-aware = tổng có trọng số của:

- Semantic: `(cosine + 1) / 2`.
- Template: Jaccard tập EventId, không xét số lần xuất hiện.
- Sequence: LCS / max(độ dài hai sequence), xét thứ tự và lặp.

Các tên thuật toán được kiểm tra khi đọc config. Trọng số không âm, tổng bằng 1;
0.5/0.2/0.3 là cấu hình cố định ban đầu, không tự tìm weights tốt nhất.
`knn.k_candidates` chỉ áp dụng M0.

## Query dùng chung, sampling đúng một lần

`sampling.validation_size: 200` và `test_size: 500` chọn ngẫu nhiên không hoàn
lại từ **hai split riêng biệt**, không ép số Normal/Anomaly. `seed` giúp tái lập.
`"all"` lấy toàn bộ split. Validation ngẫu nhiên quá nhỏ có thể thiếu một lớp;
runner báo rõ lỗi nếu không đủ hai lớp để tune Adaptive.

Prepare tạo manifest trước khi chạy model:

```text
artifacts_protocol_v2/
  manifest.json
  queries/validation_queries.csv
  queries/test_queries.csv
  queries/validation.jsonl
  queries/test.jsonl
```

Mọi M0–M5, ba KB và mọi model đọc **cùng BlockIds, cùng thứ tự** từ những file này.
Runner không random lại. Nó kiểm tra hash và đối chiếu ID/nhãn của prediction;
mỗi summary lưu `query_manifest_sha256`, report từ chối trộn hai bộ query.

Đổi seed/số mẫu/input mà vẫn dùng artifact cũ sẽ báo lỗi. Muốn mở một protocol
mới, chọn `artifacts_dir` và `results_dir` mới trong system config rồi chạy `all`.
Không copy threshold cũ sang protocol mới vì signature phải khớp.

Train xây ba KB; canonicalize theo (exact ordered sequence, label).
Nếu cùng sequence có hai nhãn, giữ hai reference có nhãn gốc và đánh dấu
`conflicting_sequence`; không tự chọn một nhãn. Label chỉ được gắn vào reference
sau retrieval, không nằm trong text embedding; nhãn query chỉ dùng đánh giá.
Mixed vectors ghép từ Normal và Anomaly, không embedding Mixed lần nữa.

## Kết quả và các bước riêng

```text
results_protocol_v4/
  shared/             # M0
  qwen-local/         # M1, m2_normal … m5_mixed
  gemini/
  comparison.csv
cache_protocol_v4/<model>/llm_cache.jsonl
artifacts_protocol_v4/model_runs/<model>/selected_threshold_m3_normal_<hash>.json
```

Nếu cần chạy riêng:

```bash
.venv/bin/python -m experiments_v2.run run --split validation
.venv/bin/python -m experiments_v2.run run --split test
.venv/bin/python -m experiments_v2.run run --pipeline m4 --split test
.venv/bin/python -m experiments_v2.run report
```

Override config bằng `--system-config PATH --experiment-config PATH`.
Config cũ được lưu tại `legacy/`; artifact/kết quả cũ không bị ghi đè.
Kết quả cache có thể tái dùng để phân loại, nhưng thời gian cache hit không phải
latency suy luận thực. BGE embedding là bước offline, không nằm trong latency LLM.
