# Knowledge Distillation

## Why Distill?

The teacher model (`e5-large-v2`, 1.3 GB) is too large for AWS Lambda cold-start constraints. A distilled student (`e5-base-v2`, 418 MB) is produced via knowledge distillation and passes all quality gates, enabling Lambda deployment with a ~3.2s cold-start.

Direct fine-tuning of the student is not used. Knowledge distillation only.

---

## Approach

Pair-level `CosineSimilarityLoss` with hard negatives.

For each (alias, label) pair, the student minimizes:
```
MSE(cosine(student(alias), student(label)), cosine(teacher(alias), teacher(label)))
```

Hard negatives are mined from the teacher's top-k non-gold labels per alias via full similarity matrix multiply.

---

## Running Distillation

Requires teacher model to exist first.

```bash
conda run -n alias-label-retriever python scripts/distill_model.py
```

Outputs:
- `models/trained/alias_label_e5_distilled/` -- student weights (418 MB, e5-base-v2)
- `faiss/labels_distilled.index` -- FAISS label index at 768-dim
- `faiss/aliases_distilled.index` -- FAISS alias index at 768-dim

Teacher embedding caches (`data/cache/`) are reused across runs. Delete them to force re-encoding from scratch.

---

## Evaluation

```bash
conda run -n alias-label-retriever python scripts/evaluate_student.py --distilled --sample 2000
```

### Quality Gates

| Gate | Threshold | Result |
|------|-----------|--------|
| Accuracy gap | student >= teacher - 5pp (>=51.05%) | 51.80% PASS |
| Agreement | >=75% | 75.90% PASS |

---

## Cold-Start Profile (CPU / Lambda)

| Component | Time |
|-----------|------|
| Model load | 0.06s |
| FAISS index load | 0.02s |
| First query | ~3.10s |
| Total | ~3.2s |

---

## Experiment History

| Attempt | Approach | Student Accuracy |
|---------|----------|-----------------|
| 2/3 | Sentence-level MSELoss + PCA | 30.25% |
| 4 | Pair-level, random negatives, 5 epochs | 46.50% |
| 5 | Pair-level, hard negatives, 10 epochs, e5-small-v2 | 46.55% (capacity ceiling) |
| 6 | e5-base-v2 student (larger capacity) | 51.80% PASS |

Attempts 4 and 5 hit a capacity ceiling on `e5-small-v2` (~46-47%). Switching to `e5-base-v2` (109M params, 768-dim) broke through the ceiling.

Full experiment history and retrospective: [GitHub Issue #4](https://github.com/craigtrim/edu-alias-mapper-e5/issues/4)

---

## Reproducibility

```bash
# Full distillation pipeline from scratch
conda run -n alias-label-retriever python scripts/distill_model.py
conda run -n alias-label-retriever python scripts/evaluate_student.py --distilled --sample 2000
```

To force teacher re-encoding (normally cached):
```bash
rm data/cache/teacher_alias_embs.npy data/cache/teacher_label_embs.npy
conda run -n alias-label-retriever python scripts/distill_model.py
```
