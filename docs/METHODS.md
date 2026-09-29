# Method Design: M0-M5

This document maps each experiment ID to the actual algorithm used in code.

## Summary

| ID | Role | Retrieval | Selection | LLM | KB variants |
|---|---|---|---|---:|---|
| M0 | Non-LLM baseline | Sequence n-gram KNN | Validation-selected k | No | None |
| M1 | LLM-only baseline | None | None | Yes | None |
| M2 | Semantic RAG | Semantic cosine | Fixed top-k sweep | Yes | normal, anomaly, mixed |
| M3 | Semantic RAG + adaptive context | Semantic cosine | Validation-selected threshold | Yes | normal, anomaly, mixed |
| M4 | Structure-aware RAG | Semantic + structure score | Fixed top-k sweep | Yes | normal, anomaly, mixed |
| M5 | Structure-aware RAG + adaptive context | Semantic + structure score | Validation-selected threshold | Yes | normal, anomaly, mixed |

## M0: sequence n-gram KNN

M0 is a classical non-LLM baseline. It represents each ordered EventId sequence using hashed n-gram features.

Example sequence:

```text
E1 E2 E3 E4
```

Bigram features:

```text
E1->E2, E2->E3, E3->E4
```

Trigram features:

```text
E1->E2->E3, E2->E3->E4
```

Why this matters: simple EventId count vectors ignore order, while n-grams preserve local order patterns.

The KNN value is selected on validation only. Test uses the selected value without retuning.

## M1: LLM-only

M1 sends only the query BlockId sequence to the LLM. It does not retrieve similar cases from the knowledge base.

This checks whether the LLM can classify the sequence without examples.

## M2 and M3: semantic retrieval

Semantic retrieval embeds the text representation of each sequence and ranks KB entries by cosine similarity.

Labels are not embedded. The label is attached only after retrieval, when building the context shown to the LLM.

- M2 uses fixed top-k context selection.
- M3 uses adaptive threshold selection.

## M4 and M5: structure-aware retrieval

Structure-aware retrieval combines semantic similarity with two structural signals:

```text
weighted_score = semantic_weight * semantic_cosine
               + template_weight * EventId-set similarity
               + sequence_weight * ordered sequence similarity
```

In the current protocol, ordered sequence similarity uses normalized LCS-style matching.

- M4 uses fixed top-k context selection.
- M5 uses adaptive threshold selection.

## Fixed top-k selection

Fixed selection tests a range of context sizes, for example:

```text
fixed_k1, fixed_k2, ..., fixed_k10
```

This is useful because the best number of retrieved examples is not known in advance.

The report keeps every fixed-k result and marks the best fixed-k per method/KB.

## Adaptive threshold selection

Adaptive selection first retrieves a candidate pool, currently top-10. It then keeps only candidates whose retrieval score is above a threshold.

Threshold candidates are derived from validation score quantiles. Each candidate threshold is evaluated on validation, then the best threshold is locked and reused for test.

Adaptive is therefore not allowed to learn from test results.

## Knowledge base variants

For M2-M5, each retrieval method is tested against three KB compositions:

| KB | Meaning |
|---|---|
| `normal` | only normal train sequences |
| `anomaly` | only anomalous train sequences |
| `mixed` | both normal and anomalous train sequences |

The sequence text stored for retrieval does not contain the label. The label is attached after retrieval so the LLM can see whether a retrieved case was normal or anomalous.
