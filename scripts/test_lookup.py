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
from pathlib import Path
from sentence_transformers import SentenceTransformer
import logging
from gpu_check import verify_gpu

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # project root
RAW_PARQUET = f"{BASE_DIR}/data/raw/dbpedia_schools.parquet"
LOG_FILE = f"{BASE_DIR}/logs/test_lookup.log"

# Large model paths (e5-large-v2, 1024-dim) — default
MODEL_PATH_LARGE = f"{BASE_DIR}/models/trained/alias_label_e5"
FAISS_LABELS_PATH_LARGE = f"{BASE_DIR}/faiss/labels.index"
FAISS_ALIASES_PATH_LARGE = f"{BASE_DIR}/faiss/aliases.index"

# Small model paths (e5-small-v2, 384-dim) — selected via --small flag
# Produced by scripts/train_small_model.py
MODEL_PATH_SMALL = f"{BASE_DIR}/models/trained/alias_label_e5_small"
FAISS_LABELS_PATH_SMALL = f"{BASE_DIR}/faiss/labels_small.index"
FAISS_ALIASES_PATH_SMALL = f"{BASE_DIR}/faiss/aliases_small.index"

# Distilled model paths (e5-small-v2, 384-dim) — selected via --distilled flag
# Produced by scripts/distill_model.py
MODEL_PATH_DISTILLED = f"{BASE_DIR}/models/trained/alias_label_e5_distilled"
FAISS_LABELS_PATH_DISTILLED = f"{BASE_DIR}/faiss/labels_distilled.index"
FAISS_ALIASES_PATH_DISTILLED = f"{BASE_DIR}/faiss/aliases_distilled.index"

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
    parser.add_argument(
        "--small",
        action="store_true",
        default=False,
        help="Use the directly fine-tuned e5-small-v2 model (384-dim). "
             "Requires train_small_model.py to have been run first.",
    )
    parser.add_argument(
        "--distilled",
        action="store_true",
        default=False,
        help="Use the distilled e5-small-v2 model (384-dim). "
             "Requires distill_model.py to have been run first.",
    )
    args = parser.parse_args()

    # Select model and index paths
    if args.distilled:
        MODEL_PATH = MODEL_PATH_DISTILLED
        FAISS_LABELS_PATH = FAISS_LABELS_PATH_DISTILLED
        FAISS_ALIASES_PATH = FAISS_ALIASES_PATH_DISTILLED
        logger.info("Using distilled model (e5-small-v2, 384-dim)")
    elif args.small:
        MODEL_PATH = MODEL_PATH_SMALL
        FAISS_LABELS_PATH = FAISS_LABELS_PATH_SMALL
        FAISS_ALIASES_PATH = FAISS_ALIASES_PATH_SMALL
        logger.info("Using small model (e5-small-v2, 384-dim)")
    else:
        MODEL_PATH = MODEL_PATH_LARGE
        FAISS_LABELS_PATH = FAISS_LABELS_PATH_LARGE
        FAISS_ALIASES_PATH = FAISS_ALIASES_PATH_LARGE
        logger.info("Using large model (e5-large-v2, 1024-dim)")

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
