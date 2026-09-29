# HDFS Log Anomaly Detection

Repository này chứa pipeline nghiên cứu phát hiện bất thường log HDFS ở mức **BlockId event sequence**.

Pipeline hiện tại nằm trong package `experiments/` và đánh giá các phương pháp M0-M5:

- M0: sequence n-gram KNN baseline
- M1: LLM-only baseline
- M2: semantic retrieval + fixed top-k
- M3: semantic retrieval + adaptive context
- M4: structure-aware retrieval + fixed top-k
- M5: structure-aware retrieval + adaptive context

## Repository layout

```text
data/
  raw/              # HDFS.log, HDFS_v1.zip
  preprocessed/     # parsed CSV files: traces, templates, labels, occurrence matrix
  processed/        # older prepared dataset version kept for audit
  processed_v2/     # current prepared dataset and train/validation/test splits

experiments/
  configs/          # system/runtime config and experiment design config
  pipelines/        # method implementations: M0, M1, RAG M2-M5
  shared/           # data, retrieval, structure, LLM, evaluation utilities
  notebooks/        # experiment report dashboard
  _generated/       # local artifacts/results/cache, ignored by Git
  _archive/         # old experimental code, ignored by Git

docs/               # research notes, method protocol, data protocol
notebooks/          # data understanding / preprocessing notebook
scripts/            # data preparation and audit scripts
reports/            # generated EDA/report outputs; not model source code
tests/              # automated tests
```

`reports/` replaces the old ambiguous `outputs/` folder. It contains generated data-understanding reports, figures, and CSV summaries. It is not part of the model implementation.

## Install

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Experiment commands

Local Qwen validation run:

```bash
.venv/bin/python -m experiments.run prepare --model-profile qwen-local
.venv/bin/python -m experiments.run embed --model-profile qwen-local
.venv/bin/python -m experiments.run run --model-profile qwen-local --split validation
.venv/bin/python -m experiments.run report --model-profile qwen-local --split validation
```

Final test should be run only after validation choices are locked:

```bash
.venv/bin/python -m experiments.run run --model-profile qwen-local --split test
.venv/bin/python -m experiments.run report --model-profile qwen-local --split test
```

Avoid `python -m experiments.run all` during development because it runs validation and then full test.

## Important docs

- `docs/METHODS.md`: M0-M5 definitions
- `docs/DATA_PROTOCOL.md`: train/validation/test, KB construction, label handling
- `docs/PROJECT_STRUCTURE.md`: file-by-file map
- `docs/research_design.md`: research design notes

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```
