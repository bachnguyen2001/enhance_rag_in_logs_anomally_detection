# Hướng dẫn experiment M0–M3

## Dữ liệu cố định

- Protocol chính: `processed_v2/group_trace`.
- Validation: 160 Normal + 40 Anomaly.
- Test: 400 Normal + 100 Anomaly.
- M2/M3 dùng chung 4.759 canonical Normal train traces.
- M0 dùng toàn bộ labeled train BlockIds và 29 EventId counts.

Precision và F1 phản ánh subset có 20% Anomaly, không đại diện trực tiếp prevalence 2,93% của HDFS.

## Code

```text
experiments/run.py                 điều phối lần chạy
experiments/shared/data.py         đọc và chuẩn bị dữ liệu
experiments/shared/retrieval.py    embedding, cosine top-k và KNN
experiments/shared/llm.py          prompt, API, JSON parser và cache
experiments/shared/evaluation.py   metrics và summary
experiments/pipelines/             logic riêng của M0, M1, M2, M3
```

Pipeline chỉ dự đoán. `run.py` nạp cấu hình và shared resources, gọi pipeline, lưu prediction và metrics.
Thông tin API nằm trong `experiments/.env`; tham số không bí mật nằm trong `experiments/config.json`.

## Trình tự chạy

Artifact dữ liệu và BGE embeddings hiện đã tồn tại. Chỉ chạy lại `prepare` và `embed` khi tạo một
experiment version mới.

Để chạy toàn bộ quy trình theo đúng thứ tự, dùng:

```bash
.venv/bin/python run_experiment.py --models gemini
.venv/bin/python run_experiment.py --models qwen-local
.venv/bin/python run_experiment.py --models all
```

File này chạy M0 đúng một lần, kiểm tra artifact và cấu hình, chạy smoke test cho từng model,
chạy M1/M2 validation, tạo phân tích RAG_HELP/RAG_HARM, screening và khóa threshold M3,
chạy final M1–M3 test rồi tạo
bảng comparison. Kết quả hợp lệ đã tồn tại được bỏ qua; cache giúp tiếp tục sau khi process dừng.

M0 nằm tại `experiments/results/shared/`. Bảng của Gemini và Qwen đều tham chiếu cùng baseline này;
không chạy lại KNN khi chuyển model.

Audit M0 có thể tái lập bằng `.venv/bin/python -m scripts.audit_m0`. Kết quả được lưu tại
`experiments/results/analysis/m0_audit.json` và `m0_test_neighbor_audit.csv`. Audit phân biệt rõ
ordered-trace overlap của split với count-vector overlap của representation M0.

Hai profile nằm trong `experiments/models.json`: `gemini` đọc URL/model/key từ `.env`, còn
`qwen-local` gọi `http://192.168.56.1:1234/v1` mà không gửi Authorization header. Mỗi profile có
cache và `selected_threshold.json` riêng dưới `experiments/artifacts/model_runs/<profile>/`.

```bash
# Không gọi LLM API
.venv/bin/python -m experiments.run run --pipeline m0 --split validation
.venv/bin/python -m experiments.run run --pipeline m0 --split test

# Có gọi LLM API
.venv/bin/python -m experiments.run run --pipeline m1 --split test
.venv/bin/python -m experiments.run run --pipeline m2 --split test
.venv/bin/python -m experiments.run run --pipeline m3 --split validation
.venv/bin/python -m experiments.run run --pipeline m3 --split test

# Sau khi đã khóa threshold
.venv/bin/python -m experiments.run run --pipeline all --split test
.venv/bin/python -m experiments.run report
```

M3 validation thử Q25, Q50 và Q75 của pooled top-10 similarity scores. Nó chọn anomaly F1 cao nhất,
ưu tiên ít context hơn khi hòa, rồi lưu `experiments/artifacts/selected_threshold.json`.

Mỗi prediction M3 lưu đầy đủ bằng chứng retrieval: top-10 candidate theo đúng rank, trace ID,
cosine similarity, candidate có đạt threshold hay không và có được đưa vào prompt hay không.
Record cũng lưu threshold, `context_count`, danh sách context, exact messages, prediction, reason,
prompt tokens, latency, cache status và số rate-limit retries. Vì vậy có thể kiểm tra lại bốn bước:
xếp hạng, relevance filtering, số context thích nghi và chi phí prompt thực tế cho từng query.
Trong khi đang chạy, từng record được flush ngay vào file `*_in_progress.jsonl`. Nếu API hoặc process
dừng giữa chừng, các bằng chứng đã hoàn thành vẫn còn trên đĩa; khi hoàn tất, runner tạo file
`*_predictions.jsonl` chính thức và xóa file tạm.

## Artifact và kết quả

```text
experiments/artifacts/
  manifest.json
  queries/
  knn/
  retrieval/
  selected_threshold.json
  llm_cache.jsonl
experiments/results/
  gemini/validation/
  gemini/test/
  qwen-local/validation/
  qwen-local/test/
  model_comparison.csv
  previous_run/
```

`previous_run/` chỉ dùng đối chiếu với code trước khi tái cấu trúc. Runner mới ghi kết quả chính vào
`validation/` và `test/`. Mỗi phương pháp tạo predictions JSONL và summary JSON.

Metrics gồm anomaly Precision, Recall, F1, TP/TN/FP/FN, số context trung bình, prompt tokens và latency.
Output không đúng JSON được ghi `INVALID_OUTPUT`, không tự đổi thành một nhãn hợp lệ.

Cache dùng SHA256 của provider, URL, model, prompt, temperature và giới hạn output token.
Thay prompt hoặc model tạo cache key mới; cache không chứa API key.

Trong lúc chạy M1–M3, terminal hiển thị tiến độ mỗi 10 query, prediction gần nhất, số context,
số API call mới, cache hit và thời gian đã chạy. Có thể đổi tần suất bằng `progress_every` trong
`experiments/config.json`. Nếu tiến trình bị dừng, chạy lại cùng lệnh; các prompt đã có trong cache
sẽ không gọi API lần nữa.

`llm.request_delay_seconds` đặt khoảng nghỉ giữa hai request thật; mặc định 4 giây. Khi Gemini trả
HTTP 429, code ưu tiên header `Retry-After`, nếu không có thì chờ tăng dần 10, 20, 40, 80, 120 giây,
tối đa 5 lần. Cache hit không bị áp dụng khoảng nghỉ. Có thể tăng delay nếu quota tài khoản thấp hơn.
Lỗi timeout/kết nối và HTTP 500/502/503/504 được thử lại tối đa 3 lần, với khoảng chờ mặc định
5, 10 và 20 giây. Các lần retry này được ghi trong prediction dưới trường `transient_retries`.
