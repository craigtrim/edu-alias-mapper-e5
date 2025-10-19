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
│ └── trained/ # Fine-tuned SentenceTransformer checkpoints
│
├── faiss/ # Vector indexes for alias/label retrieval
│ ├── labels.index
│ └── aliases.index
│
├── logs/ # Training and lookup logs
│ └── metrics/ # Epoch-level performance data
│
├── scripts/ # Executable Python modules
│ ├── gpu_check.py
│ ├── train_alias_label.py
│ └── test_lookup.py
│
└── notebooks/ # Optional Jupyter exploration
```

---

## ⚙️ Setup
### 1️⃣ Clone the repo
```bash
git clone https://github.com/<your-username>/alias-label-retriever.git
cd alias-label-retriever
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
| Metric | Value |
|--------|--------|
| Epochs | 3 |
| Runtime | ~6.8 min |
| Throughput | 160 samples/sec |
| Final Loss | **0.388** |
| GPU | NVIDIA GB10 (119 GB) |

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
| 🧠 Train Model | `python scripts/train_alias_label.py` |
| 🔍 Query Index | `python scripts/test_lookup.py --query "UdeG"` |
| 🧮 Monitor GPU | `watch -n 1 nvidia-smi` |

## ⚙️ Technical Deep Dive

### 🧠 Model and Embedding Configuration
- **Base Model:** `intfloat/e5-large-v2` (Sentence Transformers)
- **Embedding Dimension:** 1024  
- **Fine-tuning Objective:** `MultipleNegativesRankingLoss`
- **Training Duration:** ~6.8 minutes (3 epochs)
- **Final Loss:** 0.3884  
- **Batch Size:** 64  
- **Framework:** PyTorch 2.3.0 + CUDA  
- **Precision:** Mixed-precision (AMP enabled)  
- **Device:** NVIDIA GB10 (119 GB VRAM)

This setup uses bi-encoder embeddings—alias and label strings are independently encoded into the same vector space, allowing cosine similarity to rank candidate matches. The `MultipleNegativesRankingLoss` encourages the model to bring correct alias–label pairs closer together and push unrelated pairs apart.

---

### ⚙️ FAISS Indexing
Two independent FAISS indexes are generated post-training:

| Index | Purpose | File Path |
|--------|----------|-----------|
| `labels.index` | Enables alias → label lookup | `/faiss/labels.index` |
| `aliases.index` | Enables label → alias lookup | `/faiss/aliases.index` |

Both use `IndexFlatIP` (inner-product similarity), which is equivalent to cosine similarity for normalized embeddings.  
Each index is fully memory-resident for sub-millisecond retrieval on GPU or CPU.

---

### 🔩 Training Flow Overview
1. **Load dataset** from Parquet → Pandas (`data/raw/dbpedia_schools.parquet`)  
2. **Construct alias↔label pairs**  
3. **Initialize** pretrained E5 model  
4. **Fine-tune** for 3 epochs using `MultipleNegativesRankingLoss`  
5. **Persist** fine-tuned model → `models/trained/alias_label_e5/`  
6. **Encode** all aliases + labels → dense embeddings  
7. **Write** FAISS indexes for instant semantic search  

---

### 🧮 Query Flow
1. User enters query (alias or label).  
2. Model encodes query → 1024-D embedding.  
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
To reproduce identical results:
```bash
python scripts/gpu_check.py
python scripts/train_alias_label.py
python scripts/test_lookup.py --query "UdeG"
```
Model artifacts, logs, and indexes are version-controlled under their respective folders to ensure deterministic behavior across reruns.
