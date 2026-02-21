# 🧭 edu-alias-mapper-e5
**Semantic Mapping of Institution Aliases → Canonical Names Using Sentence Transformers + FAISS**

![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![CUDA](https://img.shields.io/badge/cuda-Enabled-brightgreen.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Framework](https://img.shields.io/badge/framework-PyTorch-red.svg)
![SentenceTransformers](https://img.shields.io/badge/sentence--transformers-%E2%89%A53.0-orange.svg)

---

**Alias–Label Retriever** is a GPU-accelerated semantic retrieval pipeline that maps messy institution aliases to canonical names with high accuracy. Given *"UdeG"*, it returns *"University of Guadalajara"* in milliseconds -- across 20,000+ institutions, bi-directionally.

Built and trained on a DGX node (NVIDIA GB10, 119 GB VRAM). A distilled student model passes all quality gates for AWS Lambda cold-start deployment.

---

## Performance

| Model | Size | Accuracy | Use Case |
|-------|------|----------|----------|
| Teacher `e5-large-v2` | 1.3 GB | 56.05% | Local / GPU |
| Student `e5-base-v2` | 418 MB | 51.80% | Lambda / CPU (~3.2s cold-start) |

Accuracy = Top-1 on 20,272-way retrieval. Teacher-student agreement: 75.90%.

---

## Quick Start

```bash
git clone https://github.com/craigtrim/edu-alias-mapper-e5
cd edu-alias-mapper-e5
conda create -n sparx python=3.11 -y && conda activate sparx
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
pip install sentence-transformers faiss-gpu datasets accelerate pandas pyarrow
```

| Step | Command |
|------|---------|
| Verify GPU | `python scripts/gpu_check.py` |
| Train teacher | `python scripts/train_alias_label.py` |
| Distill student | `conda run -n alias-label-retriever python scripts/distill_model.py` |
| Evaluate student | `conda run -n alias-label-retriever python scripts/evaluate_student.py --distilled --sample 2000` |
| Query | `python scripts/test_lookup.py --query "UdeG"` |

Example output:
```
🔎 Query: UdeG
  1. University of Guadalajara   (0.9761)
  2. Universidad de Guadalajara  (0.9518)
  3. UdG                         (0.9423)
```

---

## Core Stack

| Component | Purpose |
|-----------|---------|
| PyTorch | Model training and GPU compute |
| SentenceTransformers | Fine-tuning and embedding generation |
| FAISS | Fast vector similarity search |
| Hugging Face Datasets | Efficient data streaming |
| Pandas + Parquet | Data I/O and preprocessing |

---

## Documentation

- [docs/architecture.md](docs/architecture.md) -- project structure, model configs, FAISS indexing, training flow, query flow, dataset schema
- [docs/distillation.md](docs/distillation.md) -- distillation approach, experiment history, quality gates, Lambda cold-start profile

---

## Author

**Craig Trim**
AI / Data Engineering -- Maryville University
Built and trained on local DGX (**Sparx**)

---

## License

MIT License -- feel free to fork, modify, and extend.
