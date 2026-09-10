# HDFS Enhanced RAG for Log Anomaly Detection

Experiment so sánh M0 KNN, M1 LLM-only, M2 fixed top-3 RAG và M3 adaptive RAG
trên cùng group-by-trace split v2.

## Cấu trúc

```text
notebooks/     Data Understanding và giải thích preprocessing
scripts/       preprocessing có thể tái lập và kiểm thử
experiments/   runner, bốn pipeline, artifact và kết quả
docs/          thiết kế nghiên cứu và hướng dẫn
preprocessed/  CSV nguồn
processed_v2/  processed dataset và split chính
tests/         automated tests
```

Notebook [Data Understanding and Preprocessing](notebooks/01_data_understanding_and_preprocessing.ipynb)
trình bày cả hai nội dung. Final dataset vẫn được tạo bởi `scripts/prepare_data.py`, nhờ vậy
preprocessing không phụ thuộc vào trạng thái hoặc thứ tự chạy cell.

## Chạy experiment

Điền API tương thích OpenAI trong `experiments/.env` theo mẫu `experiments/.env.example`:

```dotenv
LLM_API_KEY=
LLM_BASE_URL=
LLM_MODEL=
```

```bash
# Chạy riêng từng model
.venv/bin/python run_experiment.py --models gemini
.venv/bin/python run_experiment.py --models qwen-local

# Hoặc chạy Gemini xong rồi tự động chạy Qwen local
.venv/bin/python run_experiment.py --models all

# Chỉ dùng khi tạo artifact mới
.venv/bin/python -m experiments.run prepare
.venv/bin/python -m experiments.run embed

# Validation chọn hyperparameter cho một model
.venv/bin/python -m experiments.run run --model-profile gemini --pipeline m3 --split validation

# Đánh giá test và tạo bảng so sánh
.venv/bin/python -m experiments.run run --model-profile gemini --pipeline all --split test
.venv/bin/python -m experiments.run report
```

M0 không dùng LLM nên `run_experiment.py` chỉ chạy một lần và lưu tại `experiments/results/shared/`.
Mỗi model chạy riêng M1–M3 và tham chiếu cùng M0 trong bảng comparison. Threshold và cache của
từng model nằm dưới `experiments/artifacts/model_runs/<profile>/`. Vectors là file NumPy cục bộ;
không cần database service.

Xem [hướng dẫn experiment](docs/experiment.md) và [thiết kế nghiên cứu](docs/research_design.md).

## Kiểm tra

```bash
.venv/bin/python -m unittest discover -s tests -v
```
