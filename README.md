# 🧭 edu-alias-mapper-e5
**Semantic Mapping of Institution Aliases → Canonical Names Using Sentence Transformers + FAISS**

![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![CUDA](https://img.shields.io/badge/cuda-Enabled-brightgreen.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Framework](https://img.shields.io/badge/framework-PyTorch-red.svg)

---

## 🚀 Overview
**Alias–Label Retriever** is a GPU-accelerated pipeline that learns semantic relationships between institution aliases and canonical names. It fine-tunes a **SentenceTransformer** model on alias↔label pairs (e.g., *"UdeG" → "University of Guadalajara"*) and builds **FAISS** indexes for instant bi-directional retrieval.  
Built and trained on a **DGX (Sparx)** node, the system delivers lightning-fast semantic lookup across tens of thousands of entries.

---

## 🧩 Architecture
```
/alias-label-retriever
├── data/
│ ├── raw/ # Source datasets (.parquet, .csv)
│ ├── processed/ # Cleaned, merged, or chunked data
│ └── cache/ # Temporary embeddings, staging data
│
├── models/
│ ├── base/ # Pretrained weights (e.g., intfloat/e5-large-v2)
│ └── trained/
│     ├── alias_label_e5/           # Fine-tuned teacher (e5-large-v2, 1024-dim)
│     └── alias_label_e5_distilled/ # Distilled student (e5-base-v2, 768-dim)
│
├── faiss/ # Vector indexes for alias/label retrieval
│ ├── labels.index             # Teacher label index (1024-dim)
│ ├── aliases.index            # Teacher alias index (1024-dim)
│ ├── labels_distilled.index   # Student label index (768-dim)
│ └── aliases_distilled.index  # Student alias index (768-dim)
│
├── data/
│ └── cache/                   # Cached teacher embeddings (reused across distillation runs)
│     ├── teacher_alias_embs.npy
│     └── teacher_label_embs.npy
│
├── logs/ # Training and lookup logs
│ └── metrics/ # Epoch-level performance data
│
├── scripts/ # Executable Python modules
│ ├── gpu_check.py
│ ├── train_alias_label.py
│ ├── distill_model.py
│ ├── evaluate_student.py
│ └── test_lookup.py
│
└── notebooks/ # Optional Jupyter exploration
```

---

## ⚙️ Setup
### 1️⃣ Clone the repo
```bash
git clone https://github.com/craigtrim/edu-alias-mapper-e5
cd edu-alias-mapper-e5
```

### 2️⃣ Create and activate environment
```bash
conda create -n sparx python=3.11 -y
conda activate sparx
```

### 3️⃣ Install dependencies
```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
pip install sentence-transformers faiss-cpu datasets accelerate pandas pyarrow
```
_(If your GPU supports CUDA, replace faiss-cpu with faiss-gpu)_

### 🧠 Training

Train the semantic model on your dataset:
```python
python scripts/train_alias_label.py
```

Outputs:
- Fine-tuned model → models/trained/alias_label_e5/
- FAISS indexes → faiss/aliases.index and faiss/labels.index
- Logs → logs/training.log

### 🔍 Inference / Lookup
Alias → Canonical Label
```bash
python scripts/test_lookup.py --query "UdeG" --direction alias-to-label
```

Canonical Label → Likely Aliases
```bash
python scripts/test_lookup.py --query "University of Guadalajara" --direction label-to-alias
```

Example output:
```
🔎 Query: UdeG
📈 Top matches:
  1. University of Guadalajara                         (0.9761)
  2. Universidad de Guadalajara                        (0.9518)
  3. UdG                                              (0.9423)
```


### 🗜️ Distilled Model (Lambda Deployment)

The teacher model (`e5-large-v2`, 1.3 GB) is too large for AWS Lambda cold-start. A distilled student (`e5-base-v2`, 418 MB) is produced via knowledge distillation and passes all quality gates.

See [GitHub Issue #4](https://github.com/craigtrim/edu-alias-mapper-e5/issues/4) for full experiment history, metrics, and retrospective.

**Run distillation** (requires teacher model to exist first):
```bash
conda run -n alias-label-retriever python scripts/distill_model.py
```

Outputs:
- `models/trained/alias_label_e5_distilled/` -- student weights (418 MB, e5-base-v2)
- `faiss/labels_distilled.index` -- FAISS label index at 768-dim (60 MB)
- `faiss/aliases_distilled.index` -- FAISS alias index at 768-dim

**Evaluate distilled model:**
```bash
conda run -n alias-label-retriever python scripts/evaluate_student.py --distilled --sample 2000
```

Quality gates (Attempt 6 results):

| Gate | Threshold | Result |
|------|-----------|--------|
| Accuracy gap | student >= teacher - 5pp (>=51.05%) | 51.80% PASS |
| Agreement | >=75% | 75.90% PASS |

**Cold-start profile (CPU / Lambda):**

| Component | Time |
|-----------|------|
| Model load | 0.06s |
| FAISS index load | 0.02s |
| First query | ~3.10s |
| Total | ~3.2s |

---

### 🧮 GPU Verification
Before training, confirm CUDA readiness:
```bash
python scripts/gpu_check.py
```

Example log:
```bash
🧮 Detected 1 CUDA device(s)
✅ GPU[0] NVIDIA GB10 — 79.1 GB free / 119.7 GB total
🎯 GPU environment verified successfully
```

## 🧱 Core Stack
| Component | Purpose |
|------------|----------|
| **PyTorch** | Model training and GPU compute |
| **SentenceTransformers** | Fine-tuning & embedding generation |
| **FAISS** | Fast vector similarity search |
| **Hugging Face Datasets** | Efficient data streaming |
| **Pandas + Parquet** | Data I/O and preprocessing |
| **Accelerate** | GPU orchestration backend |

---

## 🧾 Example Dataset Schema
| Column | Type | Description |
|---------|------|-------------|
| `qid` | str | Wikidata or DBpedia ID |
| `label` | str | Canonical institution name |
| `alias` | str | Known alternative name or abbreviation |
| `description` | str | Optional metadata |
| `website` | str | Official site |

---

## 📊 Performance Snapshot

### Teacher (e5-large-v2)
| Metric | Value |
|--------|--------|
| Model size | 1.3 GB |
| Embedding dim | 1024 |
| Top-1 accuracy | 56.05% (20,272-way retrieval) |
| GPU | NVIDIA GB10 (119 GB) |

### Distilled Student (e5-base-v2)
| Metric | Value |
|--------|--------|
| Model size | 418 MB |
| Embedding dim | 768 |
| Top-1 accuracy | 51.80% |
| Teacher-student agreement | 75.90% |
| CPU cold-start (Lambda) | ~3.2s total |
| Gate A (accuracy gap <=5pp) | PASS |
| Gate B (agreement >=75%) | PASS |

---

## 🧠 Future Extensions
- ⚡ Batch embedding + streaming loader for 10M+ entries  
- 🧩 ONNX export for low-latency inference  
- 🌎 Multilingual alias handling (e.g., English ↔ Spanish)  
- 🔁 Continuous fine-tuning from new institutional data  

---

## 👨‍💻 Author
**Craig Trim**  
⚙️ *AI / Data Engineering – Maryville University*  
📍 Built and trained on local DGX (**Sparx**)  

---

## 🪪 License
MIT License — feel free to fork, modify, and extend.

---

## 🧭 Quick Start Summary
| Step | Command |
|------|----------|
| 🧩 Verify GPU | `python scripts/gpu_check.py` |
| 🧠 Train teacher | `python scripts/train_alias_label.py` |
| 🗜️ Distill student | `conda run -n alias-label-retriever python scripts/distill_model.py` |
| ✅ Evaluate student | `conda run -n alias-label-retriever python scripts/evaluate_student.py --distilled --sample 2000` |
| 🔍 Query index | `python scripts/test_lookup.py --query "UdeG"` |
| 🧮 Monitor GPU | `watch -n 1 nvidia-smi` |

## 🤔 Which Model to Use

| Scenario | Model | Why |
|----------|-------|-----|
| Local / GPU inference | Teacher (`alias_label_e5`) | Highest accuracy (56.05%), no size constraint |
| AWS Lambda / CPU | Distilled student (`alias_label_e5_distilled`) | 418 MB, 3.2s cold-start, passes quality gates |

## ⚙️ Technical Deep Dive

### 🧠 Model and Embedding Configuration

**Teacher (`intfloat/e5-large-v2`)**
- **Embedding Dimension:** 1024
- **Fine-tuning Objective:** `MultipleNegativesRankingLoss`
- **Batch Size:** 64
- **Framework:** PyTorch + CUDA, mixed-precision (AMP)
- **Device:** NVIDIA GB10 (119 GB VRAM)
- **Top-1 Accuracy:** 56.05% on 20,272-way retrieval

The teacher uses bi-encoder embeddings -- alias and label strings are independently encoded into the same vector space, allowing cosine similarity to rank candidate matches. `MultipleNegativesRankingLoss` brings correct alias-label pairs closer together and pushes unrelated pairs apart.

**Distilled Student (`intfloat/e5-base-v2`)**
- **Embedding Dimension:** 768
- **Distillation Objective:** Pair-level `CosineSimilarityLoss` with hard negatives
- **Training Signal:** For each (alias, label) pair, the student minimizes `MSE(cosine(student(alias), student(label)), cosine(teacher(alias), teacher(label)))`. Hard negatives are the teacher's top-k non-gold labels per alias, mined via full similarity matrix multiply.
- **Epochs:** 10
- **Top-1 Accuracy:** 51.80% | Teacher-student agreement: 75.90%

See [GitHub Issue #4](https://github.com/craigtrim/edu-alias-mapper-e5/issues/4) for full distillation experiment history.

---

### ⚙️ FAISS Indexing
Each model has its own pair of FAISS indexes built from its respective embeddings:

| Index | Model | Dim | Purpose |
|-------|-------|-----|---------|
| `faiss/labels.index` | Teacher | 1024 | Alias → label lookup |
| `faiss/aliases.index` | Teacher | 1024 | Label → alias lookup |
| `faiss/labels_distilled.index` | Student | 768 | Alias → label lookup (Lambda) |
| `faiss/aliases_distilled.index` | Student | 768 | Label → alias lookup (Lambda) |

All indexes use `IndexFlatIP` (inner-product similarity), equivalent to cosine similarity for normalized embeddings. Fully memory-resident for fast retrieval on GPU or CPU.

---

### 🔩 Training Flow Overview

**Teacher (`train_alias_label.py`)**
1. Load dataset from Parquet (`data/raw/dbpedia_schools.parquet`)
2. Construct alias-label pairs
3. Initialize pretrained `e5-large-v2`
4. Fine-tune using `MultipleNegativesRankingLoss`
5. Save model → `models/trained/alias_label_e5/`
6. Encode all aliases + labels → 1024-dim embeddings
7. Write FAISS indexes (`labels.index`, `aliases.index`)

**Distilled Student (`distill_model.py`)**
1. Load teacher and encode all aliases + labels → cached embeddings (`data/cache/`)
2. Compute full alias-to-label similarity matrix via matrix multiply
3. Mine hard negatives: teacher's top-k non-gold labels per alias
4. Build pair-level training examples with teacher cosine similarities as scalar targets
5. Train `e5-base-v2` student using `CosineSimilarityLoss` for 10 epochs
6. Save student → `models/trained/alias_label_e5_distilled/`
7. Encode all aliases + labels with student → 768-dim embeddings
8. Write FAISS indexes (`labels_distilled.index`, `aliases_distilled.index`)

---

### 🧮 Query Flow
1. User enters query (alias or label).
2. Model encodes query → embedding (1024-dim teacher, 768-dim student).
3. FAISS performs nearest-neighbor search in the relevant index.
4. Returns ranked candidates with cosine similarity scores.

Example:
```
Query: "UdeG"
Top-1 Match: "University of Guadalajara" (0.9761)
```


---

### 🧰 Libraries and Versions
| Library | Version | Role |
|----------|----------|------|
| `torch` | ≥ 2.3.0 | GPU compute |
| `sentence-transformers` | ≥ 3.0 | Model fine-tuning |
| `faiss-gpu` | ≥ 1.8.0 | Vector similarity search |
| `datasets` | ≥ 4.2.0 | Data streaming backend |
| `accelerate` | ≥ 0.26.0 | Trainer orchestration |
| `pandas` / `pyarrow` | latest | Data I/O |

---

### 🧩 Performance Considerations
- ⚡ **Encoding Speed:** ~30 k sentences/minute on GB10 GPU  
- 💾 **Index Size:** ~400 MB per 50 k unique entries (float32)  
- 🧠 **Scalability:** Extendable to 10 M+ records via FAISS IVF or HNSW variants  
- 🔁 **Retraining:** Fine-tuning can resume from any checkpoint in `/models/trained/`

---

### 🛡️ Reproducibility
To reproduce teacher training:
```bash
python scripts/gpu_check.py
python scripts/train_alias_label.py
python scripts/test_lookup.py --query "UdeG"
```

To reproduce distillation:
```bash
conda run -n alias-label-retriever python scripts/distill_model.py
conda run -n alias-label-retriever python scripts/evaluate_student.py --distilled --sample 2000
```

Teacher embedding caches (`data/cache/`) are reused across distillation runs. Delete them to force re-encoding from scratch.
