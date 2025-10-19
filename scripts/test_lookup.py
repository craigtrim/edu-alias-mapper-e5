#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
test_lookup.py
Interactive or CLI-based FAISS retriever for alias↔label lookups.

Usage examples:
  python test_lookup.py --query "UdeG" --direction alias-to-label
  python test_lookup.py --query "University of Guadalajara" --direction label-to-alias
"""

import os
import faiss
import argparse
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import logging
from gpu_check import verify_gpu

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
BASE_DIR = "/home/craigtrim/projects/alias-label-retriever"
MODEL_PATH = f"{BASE_DIR}/models/trained/alias_label_e5"
FAISS_LABELS_PATH = f"{BASE_DIR}/faiss/labels.index"
FAISS_ALIASES_PATH = f"{BASE_DIR}/faiss/aliases.index"
RAW_PARQUET = f"{BASE_DIR}/data/raw/dbpedia_schools.parquet"
LOG_FILE = f"{BASE_DIR}/logs/test_lookup.log"

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
logger = logging.getLogger("test_lookup")

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def load_faiss_index(path: str) -> faiss.Index:
    """Load a FAISS index from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"FAISS index not found: {path}")
    index = faiss.read_index(path)
    logger.info(f"📦 Loaded FAISS index → {path}")
    return index


def search(model, index, items: list[str], query: str, top_k: int = 5) -> list[tuple[str, float]]:
    """Encode query, perform FAISS search, return top results."""
    qv = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    D, I = index.search(qv, top_k)
    results = [(items[i], float(D[0][pos])) for pos, i in enumerate(I[0])]
    return results


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True, help="Text to search")
    parser.add_argument(
        "--direction",
        type=str,
        choices=["alias-to-label", "label-to-alias"],
        default="alias-to-label",
        help="Search direction",
    )
    parser.add_argument("--top", type=int, default=5, help="Number of results to show")
    args = parser.parse_args()

    # -------------------------------------------------------------
    # Verify GPU
    # -------------------------------------------------------------
    device = verify_gpu()
    logger.info(f"🚀 Using device: {device}")

    # -------------------------------------------------------------
    # Load model and dataset
    # -------------------------------------------------------------
    logger.info(f"🧠 Loading model from {MODEL_PATH}")
    model = SentenceTransformer(MODEL_PATH, device=device)

    df = pd.read_parquet(RAW_PARQUET)
    labels = df["label"].dropna().unique().tolist()
    aliases = df["alias"].dropna().unique().tolist()

    # -------------------------------------------------------------
    # Load indexes
    # -------------------------------------------------------------
    if args.direction == "alias-to-label":
        index = load_faiss_index(FAISS_LABELS_PATH)
        items = labels
    else:
        index = load_faiss_index(FAISS_ALIASES_PATH)
        items = aliases

    # -------------------------------------------------------------
    # Query and display
    # -------------------------------------------------------------
    results = search(model, index, items, args.query, top_k=args.top)

    print("\n🔎 Query:", args.query)
    print("📈 Top matches:")
    for rank, (item, score) in enumerate(results, start=1):
        print(f"  {rank:>2}. {item:<60} ({score:.4f})")

    logger.info(f"✅ Lookup completed for query: {args.query}")


if __name__ == "__main__":
    main()
