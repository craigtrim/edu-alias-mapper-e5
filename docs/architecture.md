# Architecture

## Project Structure

```
/alias-label-retriever
├── data/
│   ├── raw/                         # Source datasets (.parquet, .csv)
│   ├── processed/                   # Cleaned, merged, or chunked data
│   └── cache/                       # Cached teacher embeddings (reused across distillation runs)
│       ├── teacher_alias_embs.npy
│       └── teacher_label_embs.npy
│
├── models/
│   ├── base/                        # Pretrained weights (e.g., intfloat/e5-large-v2)
│   └── trained/
│       ├── alias_label_e5/          # Fine-tuned teacher (e5-large-v2, 1024-dim)
│       └── alias_label_e5_distilled/ # Distilled student (e5-base-v2, 768-dim)
│
├── faiss/                           # Vector indexes for alias/label retrieval
│   ├── labels.index                 # Teacher label index (1024-dim)
│   ├── aliases.index                # Teacher alias index (1024-dim)
│   ├── labels_distilled.index       # Student label index (768-dim)
│   └── aliases_distilled.index      # Student alias index (768-dim)
│
├── logs/                            # Training and lookup logs
│   └── metrics/                     # Epoch-level performance data
│
├── scripts/                         # Executable Python modules
│   ├── gpu_check.py
│   ├── train_alias_label.py
│   ├── distill_model.py
│   ├── evaluate_student.py
│   └── test_lookup.py
│
└── notebooks/                       # Jupyter inference notebooks
    ├── alias-label-inference-teacher.ipynb
    └── alias-label-inference-distilled.ipynb
```

---

## Model and Embedding Configuration

### Teacher (`intfloat/e5-large-v2`)
- **Embedding Dimension:** 1024
- **Fine-tuning Objective:** `MultipleNegativesRankingLoss`
- **Batch Size:** 64
- **Framework:** PyTorch + CUDA, mixed-precision (AMP)
- **Device:** NVIDIA GB10 (119 GB VRAM)
- **Top-1 Accuracy:** 56.05% on 20,272-way retrieval

The teacher uses bi-encoder embeddings -- alias and label strings are independently encoded into the same vector space, allowing cosine similarity to rank candidate matches. `MultipleNegativesRankingLoss` brings correct alias-label pairs closer together and pushes unrelated pairs apart.

### Distilled Student (`intfloat/e5-base-v2`)
- **Embedding Dimension:** 768
- **Distillation Objective:** Pair-level `CosineSimilarityLoss` with hard negatives
- **Training Signal:** For each (alias, label) pair, the student minimizes `MSE(cosine(student(alias), student(label)), cosine(teacher(alias), teacher(label)))`. Hard negatives are the teacher's top-k non-gold labels per alias, mined via full similarity matrix multiply.
- **Epochs:** 10
- **Top-1 Accuracy:** 51.80% | Teacher-student agreement: 75.90%

---

## FAISS Indexing

Each model has its own pair of FAISS indexes built from its respective embeddings:

| Index | Model | Dim | Purpose |
|-------|-------|-----|---------|
| `faiss/labels.index` | Teacher | 1024 | Alias → label lookup |
| `faiss/aliases.index` | Teacher | 1024 | Label → alias lookup |
| `faiss/labels_distilled.index` | Student | 768 | Alias → label lookup (Lambda) |
| `faiss/aliases_distilled.index` | Student | 768 | Label → alias lookup (Lambda) |

All indexes use `IndexFlatIP` (inner-product similarity), equivalent to cosine similarity for normalized embeddings. Fully memory-resident for fast retrieval on GPU or CPU.

---

## Training Flow

### Teacher (`train_alias_label.py`)
1. Load dataset from Parquet (`data/raw/dbpedia_schools.parquet`)
2. Construct alias-label pairs
3. Initialize pretrained `e5-large-v2`
4. Fine-tune using `MultipleNegativesRankingLoss`
5. Save model → `models/trained/alias_label_e5/`
6. Encode all aliases + labels → 1024-dim embeddings
7. Write FAISS indexes (`labels.index`, `aliases.index`)

### Distilled Student (`distill_model.py`)
1. Load teacher and encode all aliases + labels → cached embeddings (`data/cache/`)
2. Compute full alias-to-label similarity matrix via matrix multiply
3. Mine hard negatives: teacher's top-k non-gold labels per alias
4. Build pair-level training examples with teacher cosine similarities as scalar targets
5. Train `e5-base-v2` student using `CosineSimilarityLoss` for 10 epochs
6. Save student → `models/trained/alias_label_e5_distilled/`
7. Encode all aliases + labels with student → 768-dim embeddings
8. Write FAISS indexes (`labels_distilled.index`, `aliases_distilled.index`)

---

## Query Flow

1. User enters query (alias or label).
2. Model encodes query → embedding (1024-dim teacher, 768-dim student).
3. FAISS performs nearest-neighbor search in the relevant index.
4. Returns ranked candidates with cosine similarity scores.

```
Query: "UdeG"
Top-1 Match: "University of Guadalajara" (0.9761)
```

---

## Dataset Schema

| Column | Type | Description |
|---------|------|-------------|
| `qid` | str | Wikidata or DBpedia ID |
| `label` | str | Canonical institution name |
| `alias` | str | Known alternative name or abbreviation |
| `description` | str | Optional metadata |
| `website` | str | Official site |

---

## Libraries and Versions

| Library | Version | Role |
|----------|----------|------|
| `torch` | ≥ 2.3.0 | GPU compute |
| `sentence-transformers` | ≥ 3.0 | Model fine-tuning |
| `faiss-gpu` | ≥ 1.8.0 | Vector similarity search |
| `datasets` | ≥ 4.2.0 | Data streaming backend |
| `accelerate` | ≥ 0.26.0 | Trainer orchestration |
| `pandas` / `pyarrow` | latest | Data I/O |

---

## Performance Considerations

- **Encoding Speed:** ~30k sentences/minute on GB10 GPU
- **Index Size:** ~400 MB per 50k unique entries (float32)
- **Scalability:** Extendable to 10M+ records via FAISS IVF or HNSW variants
- **Retraining:** Fine-tuning can resume from any checkpoint in `models/trained/`
