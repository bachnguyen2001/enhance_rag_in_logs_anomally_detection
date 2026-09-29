# HDFS BlockId Anomaly Detection Experiments v2

This folder contains the current experiment pipeline for HDFS BlockId sequence anomaly detection with classical KNN, LLM-only, and retrieval-augmented LLM methods.

The current protocol is `protocol_v4`.

## Research Goal

Classify one complete HDFS BlockId event sequence as:

```text
NORMAL / ANOMALY
```

The experiment separates three factors:

- retrieval strategy: semantic vs structure-aware
- context selection: fixed top-k sweep vs adaptive threshold
- knowledge base composition: normal-only, anomaly-only, mixed

## Current Methods

| Method | Description | Uses LLM | Uses KB |
|---|---|---:|---:|
| M0 | Sequence n-gram KNN baseline | No | No |
| M1 | LLM-only baseline | Yes | No |
| M2 | Semantic retrieval + fixed top-k sweep | Yes | Yes |
| M3 | Semantic retrieval + adaptive threshold | Yes | Yes |
| M4 | Structure-aware retrieval + fixed top-k sweep | Yes | Yes |
| M5 | Structure-aware retrieval + adaptive threshold | Yes | Yes |

RAG methods M2-M5 run on three KB variants:

```text
normal-only
anomaly-only
mixed
```

## Data Protocol

The pipeline uses pre-split HDFS data from `data/processed_v2/group_trace`.

Current sampling config:

```json
"validation": {"normal": 400, "anomaly": 100}
"test": "all"
```

Validation is used for tuning:

- M0 selects KNN `k`
- M2/M4 evaluate fixed top-k from 1 to 10
- M3/M5 select adaptive threshold

Test is reserved for the final locked evaluation. Since `test = "all"`, running test with the local LLM is expensive.

## Method Details

### M0: Sequence n-gram KNN

M0 converts each ordered EventId sequence into hashed bigram/trigram features.

Example:

```text
E1 E2 E3 E4
```

Bigrams:

```text
E1->E2, E2->E3, E3->E4
```

Trigrams:

```text
E1->E2->E3, E2->E3->E4
```

M0 uses canonical labeled train sequences, not all duplicate BlockIds.

### M1: LLM-only

M1 sends the query sequence directly to the LLM without retrieved context.

### M2/M4: Fixed Selection

Fixed selection runs a sweep:

```text
fixed_k = 1..10
```

The report marks the best fixed-k per method/KB.

### M3/M5: Adaptive Selection

Adaptive first retrieves a top-10 candidate pool, then keeps candidates whose score passes a threshold selected on validation.

```text
top-10 candidates -> threshold -> selected context subset
```

Thresholds are tuned only on validation, never on test.

### Structure-aware Retrieval

M4/M5 use a weighted retrieval score:

```text
0.5 * semantic cosine
+ 0.2 * EventId-set Jaccard
+ 0.3 * normalized LCS sequence similarity
```

## Main Commands

Run only data preparation:

```bash
.venv/bin/python -m experiments.run prepare --model-profile qwen-local
```

Build embeddings:

```bash
.venv/bin/python -m experiments.run embed --model-profile qwen-local
```

Run validation experiments:

```bash
.venv/bin/python -m experiments.run run --model-profile qwen-local --split validation
```

Create validation report:

```bash
.venv/bin/python -m experiments.run report --model-profile qwen-local --split validation
```

Run final test only after the validation protocol is locked:

```bash
.venv/bin/python -m experiments.run run --model-profile qwen-local --split test
.venv/bin/python -m experiments.run report --model-profile qwen-local --split test
```

Avoid `all` during development because it runs validation and then full test.

## Outputs

Generated files are ignored by Git.

Important generated output locations:

```text
experiments/_generated/artifacts_protocol_v4/     # prepared data, query manifests, KB, embeddings, KNN vectors
experiments/_generated/results_protocol_v4/       # summaries, predictions, reports
experiments/_generated/cache_protocol_v4/         # LLM cache
```

Report files:

```text
comparison_validation*.csv
comparison_validation*_config.json
confusion_matrices/validation/*.csv
confusion_matrices/validation/*.png
```

The report CSV includes:

- method, retrieval, selection, KB
- context setting (`fixed_k1` ... `fixed_k10`, `adaptive`)
- precision, recall, F1
- TP/TN/FP/FN and confusion matrix text
- average contexts and prompt/latency metrics
- best fixed-k and adaptive-vs-fixed deltas

## Notebook Dashboard

Open:

```text
experiments/notebooks/report_dashboard.ipynb
```

It visualizes:

- fixed top-k curves
- adaptive vs best fixed
- heatmaps by method/KB
- confusion matrices
- F1 vs context usage

## Files to Read First

```text
configs/config_experiment.json   # experiment design knobs
configs/config_system.json       # paths, model profile, runtime settings
run.py                   # orchestration: prepare/embed/run/report
shared/data.py           # data preparation and KNN n-gram features
shared/retrieval.py      # embedding and KNN prediction helpers
shared/structure.py      # structure-aware similarity
shared/llm.py            # prompt, local/Gemini-compatible LLM client
pipelines/               # M0, M1, and KB-RAG pipeline logic
```

For more detail, read:

- `../docs/METHODS.md` for M0-M5 method definitions
- `../docs/DATA_PROTOCOL.md` for split, KB, and label-handling rules
- `../docs/PROJECT_STRUCTURE.md` for a file-by-file map
