# Project Structure

This repository has one active experiment system: `experiments/`.

## Root layout

| Path | Purpose |
|---|---|
| `data/raw/` | Original raw files such as `HDFS.log` and `HDFS_v1.zip` |
| `data/preprocessed/` | Parsed HDFS CSV files used by preprocessing and EDA |
| `data/processed/` | Older prepared dataset version kept for audit |
| `data/processed_v2/` | Current prepared dataset and train/validation/test splits |
| `experiments/` | Current M0-M5 experiment pipeline |
| `docs/` | Research notes, method protocol, data protocol, project map |
| `notebooks/` | Data understanding / preprocessing notebook |
| `scripts/` | Data preparation, verification, and audit scripts |
| `reports/` | Generated data-understanding reports; formerly `outputs/` |
| `tests/` | Automated tests |

`reports/` is generated analysis output, not model source code. The current experiment outputs are under `experiments/_generated/`.

## `experiments/` layout

| Path | Purpose |
|---|---|
| `experiments/README.md` | Experiment-specific overview and commands |
| `experiments/configs/config_system.json` | Runtime paths, model profiles, embedding device, LLM settings |
| `experiments/configs/config_experiment.json` | Research design: methods, sampling, KNN, retrieval, selection |
| `experiments/run.py` | CLI entrypoint and orchestration for prepare/embed/run/report |
| `experiments/notebooks/report_dashboard.ipynb` | Dashboard for comparison CSVs and confusion matrices |
| `experiments/pipelines/m0_knn.py` | M0 sequence n-gram KNN baseline |
| `experiments/pipelines/m1_llm_only.py` | M1 LLM-only baseline |
| `experiments/pipelines/kb_rag.py` | Shared RAG pipeline for M2-M5 |
| `experiments/shared/config.py` | Loads and validates configs |
| `experiments/shared/data.py` | Builds query manifests, KB metadata, KNN vectors, manifest hashes |
| `experiments/shared/retrieval.py` | BGE embedding, cosine retrieval, KNN helpers |
| `experiments/shared/structure.py` | Jaccard, LCS, and structure-aware retrieval scores |
| `experiments/shared/llm.py` | Prompt construction, LLM HTTP client, cache, response parsing |
| `experiments/shared/evaluation.py` | Precision/recall/F1 and confusion counts |
| `experiments/shared/progress.py` | Progress logging during LLM runs |

## Main flow

```text
data/processed_v2/group_trace
        |
        v
experiments.run prepare
        |-- shared validation/test query manifests
        |-- normal/anomaly/mixed KB metadata
        |-- M0 sequence n-gram KNN vectors
        |
        v
experiments.run embed
        |-- semantic embeddings for KB/query splits
        |
        v
experiments.run run --split validation
        |-- M0 selects k
        |-- M2/M4 sweep fixed_k=1..10
        |-- M3/M5 tune adaptive threshold
        |
        v
experiments.run report --split validation
        |-- comparison_validation*.csv
        |-- confusion matrices
        |
        v
locked final test
```

## What should not be committed

The following are local/generated and ignored by Git:

```text
data/raw/
data/preprocessed/
data/processed/
data/processed_v2/
reports/
experiments/_generated/
experiments/_archive/
.env
__pycache__/
```

If the advisor/hội đồng needs to reproduce locally, provide the dataset separately or remove the data ignores intentionally.
