#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
distill_model.py
================
Distils the fine-tuned e5-large-v2 teacher into e5-base-v2 using
pair-level soft-label MSELoss.

APPROACH (Attempt 6)
--------------------
Pair-level CosineSimilarityLoss with hard negatives and 10 epochs.
Student upgraded from e5-small-v2 (384-dim, 33M params) to e5-base-v2
(768-dim, 109M params) after e5-small-v2 plateaued at ~46-47% accuracy
(capacity ceiling).

For each (alias, gold_label) pair, the teacher's cosine similarity is the
training target.  Hard negatives are the teacher's top-k ranked non-gold
labels for each alias -- the labels the teacher itself nearly confuses with
the correct answer.  These are far more informative than random negatives:
they force the student to learn exactly the fine-grained margins the teacher
relies on to rank the correct answer first.

Hard negatives are mined in bulk via matrix multiplication of all alias
embeddings against all label embeddings, then argsorted.  No per-alias
teacher inference at training time -- all signal comes from the cached
teacher embeddings.

    minimize: MSE( cosine(student(alias), student(label)),
                   cosine(teacher(alias), teacher(label)) )

Attempt 5 used e5-small-v2 with hard negatives and 10 epochs: 46.55%
accuracy (capacity ceiling).  Attempt 6 switches to e5-base-v2 (3x more
capacity) to close the remaining 4.5pp gap to Gate A (threshold 51.05%).

INPUTS
------
  models/trained/alias_label_e5/           -- fine-tuned teacher (frozen)
  data/raw/dbpedia_schools.parquet         -- source of training pairs

OUTPUTS
-------
  data/cache/teacher_alias_embs.npy        -- cached teacher alias embeddings
  data/cache/teacher_alias_list.txt        -- alias list matching embedding rows
  data/cache/teacher_label_embs.npy        -- cached teacher label embeddings
  data/cache/teacher_label_list.txt        -- label list matching embedding rows
  models/trained/alias_label_e5_distilled/ -- distilled student weights
  faiss/labels_distilled.index             -- FAISS label index at 768-dim
  faiss/aliases_distilled.index            -- FAISS alias index at 768-dim
  logs/distill_model.log                   -- append-mode log

USAGE
-----
    conda run -n alias-label-retriever python scripts/distill_model.py
"""

import sys
import numpy as np
import faiss
import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
import logging

# ---------------------------------------------------------------------------
# Ensure scripts/ is on sys.path so gpu_check can be imported regardless of
# where the script is invoked from.
# ---------------------------------------------------------------------------
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from gpu_check import verify_gpu  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

RAW_PARQUET = BASE_DIR / "data" / "raw" / "dbpedia_schools.parquet"

# Teacher -- fine-tuned e5-large-v2, frozen during distillation
TEACHER_MODEL_PATH = BASE_DIR / "models" / "trained" / "alias_label_e5"

# Student base -- pretrained e5-base-v2, pulled from HuggingFace Hub
# Upgraded from e5-small-v2 (384-dim) after capacity ceiling at ~46-47% accuracy.
STUDENT_BASE_MODEL = "intfloat/e5-base-v2"

# Cache: teacher embeddings computed once per unique alias/label list
CACHE_DIR = BASE_DIR / "data" / "cache"
TEACHER_ALIAS_EMB_CACHE  = CACHE_DIR / "teacher_alias_embs.npy"
TEACHER_ALIAS_LIST_CACHE = CACHE_DIR / "teacher_alias_list.txt"
TEACHER_LABEL_EMB_CACHE  = CACHE_DIR / "teacher_label_embs.npy"
TEACHER_LABEL_LIST_CACHE = CACHE_DIR / "teacher_label_list.txt"

# Distilled student output
MODEL_SAVE_PATH    = BASE_DIR / "models" / "trained" / "alias_label_e5_distilled"
FAISS_LABELS_PATH  = BASE_DIR / "faiss" / "labels_distilled.index"
FAISS_ALIASES_PATH = BASE_DIR / "faiss" / "aliases_distilled.index"

LOG_FILE = BASE_DIR / "logs" / "distill_model.log"

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
TRAIN_EPOCHS      = 10
TRAIN_BATCH_SIZE  = 64
ENCODE_BATCH_SIZE = 512
K_NEGATIVES       = 5     # hard negative labels mined per positive pair

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("distill_model")

# ---------------------------------------------------------------------------
# Step 0 -- GPU
# ---------------------------------------------------------------------------
logger.info("=" * 70)
logger.info("  alias-label-retriever  |  distill_model.py  (Attempt 6)")
logger.info("  Approach: pair-level CosineSimilarityLoss, hard negatives, e5-base-v2")
logger.info("  GitHub issue: #4")
logger.info("=" * 70)
logger.info("Step 0 -- Verifying GPU...")
device = verify_gpu(min_vram_gb=4)
logger.info(f"GPU ready: {torch.cuda.get_device_name(0)}")

# ---------------------------------------------------------------------------
# Step 1 -- Load dataset and build alias/label lists
# ---------------------------------------------------------------------------
logger.info(f"Step 1 -- Loading dataset from {RAW_PARQUET}")

if not RAW_PARQUET.exists():
    raise FileNotFoundError(f"Dataset not found: {RAW_PARQUET}")

df = pd.read_parquet(RAW_PARQUET)

unique_aliases = df["alias"].dropna().str.strip().unique().tolist()
unique_labels  = df["label"].dropna().str.strip().unique().tolist()
unique_aliases = [s for s in unique_aliases if s]
unique_labels  = [s for s in unique_labels  if s]

logger.info(f"  Unique aliases : {len(unique_aliases):,}")
logger.info(f"  Unique labels  : {len(unique_labels):,}")

# Index lookups used when computing per-pair cosine similarities from cache.
alias_to_idx = {a: i for i, a in enumerate(unique_aliases)}
label_to_idx = {l: i for i, l in enumerate(unique_labels)}

# ---------------------------------------------------------------------------
# Step 2 -- Precompute and cache teacher embeddings (aliases and labels separately)
#
# Separate caches for aliases and labels allow per-pair cosine similarities
# to be recovered by index lookup at training-example build time.  The teacher
# is discarded from GPU memory once both caches are populated.
# ---------------------------------------------------------------------------
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _load_cache(emb_path: Path, list_path: Path, expected: list):
    """
    Return cached embeddings if the stored sentence list matches expected.
    Returns None if the cache is absent or stale (triggers regeneration).
    """
    if not emb_path.exists() or not list_path.exists():
        return None
    with open(list_path, "r", encoding="utf-8") as f:
        cached = [line.rstrip("\n") for line in f]
    if cached != expected:
        logger.warning(f"  Cache stale ({list_path.name}); regenerating.")
        emb_path.unlink(missing_ok=True)
        list_path.unlink(missing_ok=True)
        return None
    return np.load(str(emb_path))


logger.info("Step 2 -- Loading or computing teacher embeddings...")

teacher_alias_embs = _load_cache(
    TEACHER_ALIAS_EMB_CACHE, TEACHER_ALIAS_LIST_CACHE, unique_aliases
)
teacher_label_embs = _load_cache(
    TEACHER_LABEL_EMB_CACHE, TEACHER_LABEL_LIST_CACHE, unique_labels
)

if teacher_alias_embs is None or teacher_label_embs is None:
    if not TEACHER_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Teacher model not found at {TEACHER_MODEL_PATH}.\n"
            "Run scripts/train_alias_label.py first."
        )
    logger.info(f"  Loading teacher from {TEACHER_MODEL_PATH}...")
    teacher = SentenceTransformer(str(TEACHER_MODEL_PATH), device=device)
    logger.info(f"  Teacher embedding dim: {teacher.get_sentence_embedding_dimension()}")

    if teacher_alias_embs is None:
        logger.info(f"  Encoding {len(unique_aliases):,} aliases with teacher...")
        teacher_alias_embs = teacher.encode(
            unique_aliases,
            batch_size=ENCODE_BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        np.save(str(TEACHER_ALIAS_EMB_CACHE), teacher_alias_embs)
        with open(TEACHER_ALIAS_LIST_CACHE, "w", encoding="utf-8") as f:
            f.write("\n".join(unique_aliases))
        logger.info(f"  Alias embeddings cached: {TEACHER_ALIAS_EMB_CACHE}")

    if teacher_label_embs is None:
        logger.info(f"  Encoding {len(unique_labels):,} labels with teacher...")
        teacher_label_embs = teacher.encode(
            unique_labels,
            batch_size=ENCODE_BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        np.save(str(TEACHER_LABEL_EMB_CACHE), teacher_label_embs)
        with open(TEACHER_LABEL_LIST_CACHE, "w", encoding="utf-8") as f:
            f.write("\n".join(unique_labels))
        logger.info(f"  Label embeddings cached: {TEACHER_LABEL_EMB_CACHE}")

    del teacher
    torch.cuda.empty_cache()
    logger.info("  Teacher unloaded from GPU.")

logger.info(f"  Alias embeddings shape : {teacher_alias_embs.shape}")
logger.info(f"  Label embeddings shape : {teacher_label_embs.shape}")

# ---------------------------------------------------------------------------
# Step 2b -- Mine hard negatives via teacher similarity matrix
#
# Compute the full alias-to-label similarity matrix in one matrix multiply.
# Both embedding sets are unit-norm so dot product equals cosine similarity.
# Argsort descending gives ranked label indices per alias.  The gold label
# is excluded; the next K_NEGATIVES are the hard negatives -- labels the
# teacher ranks just below the correct answer and most likely to be confused.
# ---------------------------------------------------------------------------
logger.info("Step 2b -- Mining hard negatives via teacher similarity matrix...")

all_sims = teacher_alias_embs @ teacher_label_embs.T  # (n_aliases, n_labels)
ranked_label_indices = np.argsort(all_sims, axis=1)[:, ::-1][:, : K_NEGATIVES + 2]

logger.info(f"  Similarity matrix shape    : {all_sims.shape}")
logger.info(f"  Hard negatives per alias   : {K_NEGATIVES}")

# ---------------------------------------------------------------------------
# Step 3 -- Build pair-level training examples
#
# InputExample(texts=[alias, label], label=float) tells CosineSimilarityLoss
# to compute cosine_similarity(student(alias), student(label)) and minimise
# MSE against the scalar teacher similarity.  Positive pairs have high target
# similarity; hard negatives have lower target similarity calibrated to the
# teacher's exact scores.  This trains the student to reproduce the teacher's
# ranking margins directly.
# ---------------------------------------------------------------------------
logger.info("Step 3 -- Building pair-level training examples...")

pos_pairs = (
    df[["alias", "label"]]
    .dropna()
    .assign(
        alias=lambda d: d["alias"].str.strip(),
        label=lambda d: d["label"].str.strip(),
    )
    .drop_duplicates()
    .values.tolist()
)
pos_pairs = [
    (a, l) for a, l in pos_pairs
    if a and l and a in alias_to_idx and l in label_to_idx
]

logger.info(f"  Positive pairs : {len(pos_pairs):,}")

train_examples = []

for alias, gold_label in pos_pairs:
    a_idx = alias_to_idx[alias]
    l_idx = label_to_idx[gold_label]

    # Positive pair
    pos_sim = float(all_sims[a_idx, l_idx])
    train_examples.append(InputExample(texts=[alias, gold_label], label=pos_sim))

    # Hard negatives: teacher's top-ranked non-gold labels for this alias
    hard_neg_indices = [
        i for i in ranked_label_indices[a_idx] if i != l_idx
    ][:K_NEGATIVES]

    for neg_idx in hard_neg_indices:
        neg_sim = float(all_sims[a_idx, neg_idx])
        train_examples.append(
            InputExample(texts=[alias, unique_labels[neg_idx]], label=neg_sim)
        )

n_pos = len(pos_pairs)
n_neg = len(train_examples) - n_pos
logger.info(f"  Total training examples : {len(train_examples):,}")
logger.info(f"    Positive             : {n_pos:,}")
logger.info(f"    Negative (hard)      : {n_neg:,}")

train_loader = DataLoader(
    train_examples, batch_size=TRAIN_BATCH_SIZE, shuffle=True
)
logger.info(f"  Batches per epoch : {len(train_loader):,}")
logger.info(f"  Epochs            : {TRAIN_EPOCHS}")

# ---------------------------------------------------------------------------
# Step 4 -- Load student and define MSELoss
#
# With two-text InputExamples CosineSimilarityLoss computes:
#   output = cosine_similarity(student(texts[0]), student(texts[1]))
#   loss   = MSE(output, label)
# No dimension mismatch -- both teacher and student sides produce a scalar.
# ---------------------------------------------------------------------------
logger.info(f"Step 4 -- Loading student base model: {STUDENT_BASE_MODEL}")
student = SentenceTransformer(STUDENT_BASE_MODEL, device=device)
logger.info(f"  Student embedding dim: {student.get_sentence_embedding_dimension()}")

train_loss = losses.CosineSimilarityLoss(student)

# ---------------------------------------------------------------------------
# Step 5 -- Distil
# ---------------------------------------------------------------------------
logger.info("Step 5 -- Starting distillation...")
student.fit(
    [(train_loader, train_loss)],
    epochs=TRAIN_EPOCHS,
    warmup_steps=int(len(train_loader) * 0.1),
    show_progress_bar=True,
    use_amp=True,
)
logger.info("  Distillation complete.")

# ---------------------------------------------------------------------------
# Step 6 -- Save distilled student
# ---------------------------------------------------------------------------
logger.info(f"Step 6 -- Saving distilled student to {MODEL_SAVE_PATH}...")
MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
student.save(str(MODEL_SAVE_PATH))
logger.info("  Student saved.")

# ---------------------------------------------------------------------------
# Step 7 -- Rebuild FAISS indexes at 384-dim
# ---------------------------------------------------------------------------
logger.info("Step 7 -- Encoding corpus and building FAISS indexes...")

logger.info("  Encoding labels...")
label_emb = student.encode(
    unique_labels,
    batch_size=ENCODE_BATCH_SIZE,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True,
)

logger.info("  Encoding aliases...")
alias_emb = student.encode(
    unique_aliases,
    batch_size=ENCODE_BATCH_SIZE,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True,
)

dim = label_emb.shape[1]
assert dim == 768, f"Expected 768-dim embeddings from e5-base-v2 student, got {dim}"

index_labels  = faiss.IndexFlatIP(dim)
index_aliases = faiss.IndexFlatIP(dim)
index_labels.add(label_emb)
index_aliases.add(alias_emb)

FAISS_LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
faiss.write_index(index_labels,  str(FAISS_LABELS_PATH))
faiss.write_index(index_aliases, str(FAISS_ALIASES_PATH))

logger.info(
    f"  Labels index  : {FAISS_LABELS_PATH}  "
    f"({index_labels.ntotal:,} vectors, {dim}-dim)"
)
logger.info(
    f"  Aliases index : {FAISS_ALIASES_PATH}  "
    f"({index_aliases.ntotal:,} vectors, {dim}-dim)"
)

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
logger.info("=" * 70)
logger.info("  distill_model.py completed successfully.")
logger.info(f"  Distilled model : {MODEL_SAVE_PATH}")
logger.info(f"  Label index     : {FAISS_LABELS_PATH}")
logger.info(f"  Alias index     : {FAISS_ALIASES_PATH}")
logger.info("")
logger.info("  Next step: python scripts/evaluate_student.py --distilled")
logger.info("=" * 70)
