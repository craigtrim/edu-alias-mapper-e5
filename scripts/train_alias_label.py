#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
train_alias_label.py
Fine-tunes a SentenceTransformer model on alias↔label pairs using a local GPU.
Builds FAISS indexes for retrieval in both directions.
"""

import os
import faiss
import torch
import pandas as pd
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
import logging

from gpu_check import verify_gpu

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # project root
LOG_DIR = BASE_DIR / "logs"
RAW_PARQUET = f"{BASE_DIR}/data/raw/dbpedia_schools.parquet"
MODEL_NAME = "intfloat/e5-large-v2"
MODEL_SAVE_PATH = f"{BASE_DIR}/models/trained/alias_label_e5"
FAISS_LABELS_PATH = f"{BASE_DIR}/faiss/labels.index"
FAISS_ALIASES_PATH = f"{BASE_DIR}/faiss/aliases.index"
LOG_FILE = f"{BASE_DIR}/logs/training.log"

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
        logging.StreamHandler()
    ],
)
logger = logging.getLogger("train_alias_label")

# ---------------------------------------------------------------------
# GPU Verification
# ---------------------------------------------------------------------
logger.info("🔍 Checking GPU readiness...")
device = verify_gpu(min_vram_gb=10)
logger.info(f"🚀 GPU verified: {torch.cuda.get_device_name(0)}\n")

# ---------------------------------------------------------------------
# Load Dataset
# ---------------------------------------------------------------------
logger.info(f"📂 Loading dataset: {RAW_PARQUET}")
df = pd.read_parquet(RAW_PARQUET)
logger.info(f"✅ Dataset loaded with {len(df):,} rows")

pairs = [
    (row["alias"], row["label"])
    for _, row in df.iterrows()
    if pd.notnull(row.get("alias")) and pd.notnull(row.get("label"))
]

logger.info(f"🧩 Constructed {len(pairs):,} alias–label training pairs")

# ---------------------------------------------------------------------
# Initialize Model
# ---------------------------------------------------------------------
logger.info(f"🧠 Loading model weights: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME, device=device)
logger.info(f"✅ Model loaded ({model.get_sentence_embedding_dimension()}-dim embeddings)")

# ---------------------------------------------------------------------
# Prepare Training Data
# ---------------------------------------------------------------------
train_examples = [InputExample(texts=[a, b]) for a, b in pairs]
train_loader = DataLoader(train_examples, batch_size=64, shuffle=True)
train_loss = losses.MultipleNegativesRankingLoss(model)
logger.info(f"🧮 DataLoader ready: {len(train_loader)} batches")

# ---------------------------------------------------------------------
# Train Model
# ---------------------------------------------------------------------
logger.info("🔥 Starting fine-tuning...")
model.fit(
    [(train_loader, train_loss)],
    epochs=3,
    warmup_steps=int(len(train_loader) * 0.1),
    show_progress_bar=True,
    use_amp=True
)
logger.info("✅ Training complete")

# ---------------------------------------------------------------------
# Save Model
# ---------------------------------------------------------------------
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
model.save(MODEL_SAVE_PATH)
logger.info(f"💾 Fine-tuned model saved to {MODEL_SAVE_PATH}")

# ---------------------------------------------------------------------
# Build FAISS Indexes
# ---------------------------------------------------------------------
logger.info("⚙️ Encoding aliases and labels for FAISS indexes...")
labels = df["label"].dropna().unique().tolist()
aliases = df["alias"].dropna().unique().tolist()

label_emb = model.encode(labels, convert_to_numpy=True, batch_size=512, show_progress_bar=True)
alias_emb = model.encode(aliases, convert_to_numpy=True, batch_size=512, show_progress_bar=True)

index_labels = faiss.IndexFlatIP(label_emb.shape[1])
index_aliases = faiss.IndexFlatIP(alias_emb.shape[1])
index_labels.add(label_emb)
index_aliases.add(alias_emb)

os.makedirs(os.path.dirname(FAISS_LABELS_PATH), exist_ok=True)
faiss.write_index(index_labels, FAISS_LABELS_PATH)
faiss.write_index(index_aliases, FAISS_ALIASES_PATH)
logger.info(f"📦 FAISS indexes written to:")
logger.info(f"   • {FAISS_LABELS_PATH}")
logger.info(f"   • {FAISS_ALIASES_PATH}")

logger.info("🎯 Training and indexing pipeline completed successfully ✅")
