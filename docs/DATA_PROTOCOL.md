# Data Protocol

This document explains how data flows through `experiments`.

## Input data

The pipeline reads preprocessed HDFS BlockId traces from:

```text
data/processed_v2/group_trace
```

Each row represents one BlockId sequence with its ground-truth label:

```text
NORMAL / ANOMALY
```

## Splits

The experiment uses three conceptual sets:

| Split | Purpose |
|---|---|
| train | source of KB entries and KNN references |
| validation | choose configuration such as KNN k and adaptive threshold |
| test | final evaluation after configuration is locked |

The same validation/test query manifests are shared by all M0-M5 methods and all model profiles. This prevents different methods from being evaluated on different random queries.

## Current protocol

Current protocol folder names use `protocol_v4`:

```text
_generated/artifacts_protocol_v4/
_generated/results_protocol_v4/
_generated/cache_protocol_v4/
```

Current sampling design:

```json
"validation": {"normal": 400, "anomaly": 100}
"test": "all"
```

Validation is intentionally smaller so experiments can be tuned quickly. Test can be the full held-out set when running the final protocol.

## Knowledge base construction

Training sequences are used to build KB variants:

| KB | Included train sequences |
|---|---|
| normal | normal only |
| anomaly | anomaly only |
| mixed | normal + anomaly |

Exact duplicate ordered sequences are canonicalized so repeated identical traces do not dominate retrieval.

## Label handling

Labels are not included in embedding text.

The reason is methodological: if labels are embedded, retrieval can become artificially easy and contaminate similarity search.

Instead, the pipeline does this:

```text
embed sequence text without label
retrieve nearest KB entries
attach each retrieved entry's label when building the LLM context
```

So the LLM can use labels in context, but the retrieval model cannot search directly by label text.

## Validation locking

Validation is used to choose:

- M0 KNN `k`
- adaptive threshold for M3/M5
- observed best fixed-k setting for M2/M4 analysis

Once validation is locked, test should be run without changing these settings.

## Generated artifacts

`prepare` creates:

```text
queries/validation.jsonl
queries/test.jsonl
retrieval/normal_metadata.jsonl
retrieval/anomaly_metadata.jsonl
retrieval/mixed_metadata.jsonl
knn/train_vectors.npy
knn/train_labels.npy
knn/validation_vectors.npy
knn/test_vectors.npy
manifest.json
```

`embed` creates semantic embedding files for KB variants and query splits.

`run` creates predictions and summaries.

`report` creates comparison CSVs and confusion matrices.
