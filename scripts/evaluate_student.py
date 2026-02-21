#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
evaluate_student.py
===================
Evaluates the distilled e5-base-v2 student model against the fine-tuned
e5-large-v2 teacher model using two quality gates.

PURPOSE
-------
Before deploying the distilled student to AWS Lambda we need confidence that
knowledge distillation (3x size reduction: 1.3 GB to 418 MB) has not
meaningfully degraded alias-to-label retrieval quality. This script answers
that question with concrete, reproducible metrics.

HOW IT WORKS
------------
A stratified random sample of alias-label pairs is drawn from the dataset (the
same parquet used for training). For every alias in the sample, both the
teacher model and the student model independently retrieve the top-1 label from
the full label corpus via their respective FAISS indexes. The retrieved labels
are compared against the ground-truth label from the dataset.

Two gates are evaluated:

  Gate A -- Absolute accuracy gap
      Student top-1 accuracy must be within 5 percentage points of the teacher.
      Threshold: student_accuracy >= teacher_accuracy - 0.05

  Gate B -- Retrieval agreement
      The rate at which student and teacher return the same top-1 result,
      regardless of whether either is correct. A high agreement rate means the
      student has learned a similar retrieval geometry to the teacher.
      Threshold: agreement_rate >= 0.75

Both gates must pass for the student to be considered deployment-ready.

IMPORTANT: SEPARATE FAISS INDEXES
----------------------------------
The teacher model uses 1024-dimensional embeddings encoded in:
  faiss/labels.index    (1024-dim)

The distilled student uses 768-dimensional embeddings encoded in:
  faiss/labels_distilled.index    (768-dim)

Run distill_model.py first to generate the student index.

USAGE
-----
    conda run -n alias-label-retriever python scripts/evaluate_student.py
    conda run -n alias-label-retriever python scripts/evaluate_student.py --sample 2000
    conda run -n alias-label-retriever python scripts/evaluate_student.py --seed 99

OUTPUTS
-------
  logs/evaluate_student.log   -- append-mode evaluation log
  Printed gate report in the terminal at the end of the run.
"""

import sys
import argparse
import random
import logging
import numpy as np
import pandas as pd
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Resolve paths and ensure shared utilities are importable.
# ---------------------------------------------------------------------------
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

BASE_DIR = Path(__file__).resolve().parent.parent  # project root

# ---------------------------------------------------------------------------
# Paths — teacher artifacts (produced by train_alias_label.py)
# ---------------------------------------------------------------------------
TEACHER_MODEL_PATH = BASE_DIR / "models" / "trained" / "alias_label_e5"
TEACHER_LABELS_INDEX = BASE_DIR / "faiss" / "labels.index"

# ---------------------------------------------------------------------------
# Paths — distilled student (produced by distill_model.py)
# ---------------------------------------------------------------------------
STUDENT_MODEL_PATH_DISTILLED = BASE_DIR / "models" / "trained" / "alias_label_e5_distilled"
STUDENT_LABELS_INDEX_DISTILLED = BASE_DIR / "faiss" / "labels_distilled.index"

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
RAW_PARQUET = BASE_DIR / "data" / "raw" / "dbpedia_schools.parquet"

# ---------------------------------------------------------------------------
# Quality gate thresholds
# ---------------------------------------------------------------------------
GATE_A_MAX_GAP = 0.05   # student accuracy must be >= teacher accuracy - 5 pp
GATE_B_AGREEMENT = 0.75  # teacher-student top-1 agreement rate

# ---------------------------------------------------------------------------
# Log file
# ---------------------------------------------------------------------------
LOG_FILE = BASE_DIR / "logs" / "evaluate_student.log"
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
logger = logging.getLogger("evaluate_student")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_device() -> str:
    """
    Return 'cuda' if a CUDA GPU is available, otherwise 'cpu'.

    Evaluation does not require GPU -- loading both models on CPU is slow but
    perfectly valid for a one-time quality check. Running on GPU is
    significantly faster if available.
    """
    try:
        import torch
        if torch.cuda.is_available():
            import torch.cuda as cuda
            name = cuda.get_device_name(0)
            logger.info(f"GPU detected: {name} — running evaluation on CUDA.")
            return "cuda"
    except ImportError:
        pass
    logger.warning(
        "No GPU detected or torch not available.  "
        "Evaluation will run on CPU (slower but correct)."
    )
    return "cpu"


def load_model(path: Path, device: str) -> SentenceTransformer:
    """
    Load a SentenceTransformer model from a local checkpoint directory.

    Parameters
    ----------
    path : Path
        Directory containing the fine-tuned model weights (produced by
        model.save() in the training scripts).
    device : str
        'cuda' or 'cpu'.

    Raises
    ------
    FileNotFoundError
        If the directory does not exist, with a helpful message pointing to
        whichever training script should be run first.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}.\n"
            f"  • Teacher missing? Run:  python scripts/train_alias_label.py\n"
            f"  • Student missing? Run:  conda run -n alias-label-retriever python scripts/distill_model.py"
        )
    logger.info(f"  Loading model from {path}  (device={device})")
    model = SentenceTransformer(str(path), device=device)
    logger.info(f"  Loaded — embedding dim: {model.get_sentence_embedding_dimension()}")
    return model


def load_faiss_index(path: Path) -> faiss.Index:
    """
    Load a FAISS index from disk with a clear error if missing.

    Parameters
    ----------
    path : Path
        Absolute path to the .index file.

    Raises
    ------
    FileNotFoundError
        If the index file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {path}.\n"
            "Make sure the corresponding training script has completed."
        )
    index = faiss.read_index(str(path))
    logger.info(
        f"  Loaded FAISS index: {path.name}  "
        f"({index.ntotal:,} vectors, d={index.d})"
    )
    return index


def top1_label(
    model: SentenceTransformer,
    index: faiss.Index,
    label_list: list[str],
    query: str,
) -> str:
    """
    Retrieve the single top-scoring label for a given alias query.

    The query is encoded with ``normalize_embeddings=True`` so that the inner-
    product score (IndexFlatIP) equals cosine similarity when the query vector
    has unit norm.

    Parameters
    ----------
    model : SentenceTransformer
        The model to use for query encoding.
    index : faiss.Index
        The FAISS label index to search against.
    label_list : list[str]
        The ordered list of canonical labels whose embeddings were added to the
        index.  Position i in label_list corresponds to index vector i.
    query : str
        The alias string to look up.

    Returns
    -------
    str
        The canonical label string ranked first by the model.
    """
    query_vec = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    # search() returns shape (n_queries, top_k); we search top-1 only.
    _scores, indices = index.search(query_vec, 1)
    best_idx = int(indices[0][0])
    return label_list[best_idx]


def sample_eval_pairs(
    df: pd.DataFrame,
    n: int,
    seed: int,
) -> list[tuple[str, str]]:
    """
    Draw a stratified random sample of (alias, label) pairs from the dataset.

    Pairs with null or empty alias/label are excluded before sampling.
    If the dataset is smaller than ``n``, all available pairs are used and a
    warning is logged.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset with 'alias' and 'label' columns.
    n : int
        Desired sample size.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list of (alias, label) tuples
    """
    valid = df[
        df["alias"].notnull()
        & df["label"].notnull()
        & (df["alias"].str.strip() != "")
        & (df["label"].str.strip() != "")
    ][["alias", "label"]].drop_duplicates()

    if len(valid) <= n:
        logger.warning(
            f"Dataset has only {len(valid):,} valid unique pairs "
            f"(requested sample size: {n}).  Using all available pairs."
        )
        pairs = list(zip(valid["alias"], valid["label"]))
    else:
        sampled = valid.sample(n=n, random_state=seed)
        pairs = list(zip(sampled["alias"], sampled["label"]))

    logger.info(f"  Evaluation pairs drawn: {len(pairs):,}")
    return pairs


def run_gate_evaluation(
    teacher_model: SentenceTransformer,
    teacher_index: faiss.Index,
    student_model: SentenceTransformer,
    student_index: faiss.Index,
    label_list: list[str],
    eval_pairs: list[tuple[str, str]],
) -> dict:
    """
    Run retrieval for every evaluation pair and compute per-gate metrics.

    For each (alias, ground_truth_label) pair:
      • Teacher retrieves its top-1 label.
      • Student retrieves its top-1 label.
      • Both predictions are compared against the ground truth.
      • Teacher and student predictions are compared against each other.

    Parameters
    ----------
    teacher_model, student_model : SentenceTransformer
        Fine-tuned models to compare.
    teacher_index, student_index : faiss.Index
        FAISS label indexes built from each model's embeddings.
    label_list : list[str]
        Ordered list of canonical labels for index lookup.
    eval_pairs : list of (alias, label) tuples
        Ground-truth evaluation set.

    Returns
    -------
    dict with keys:
        teacher_accuracy  : float
        student_accuracy  : float
        agreement_rate    : float
        n_pairs           : int
    """
    teacher_correct = 0
    student_correct = 0
    agreements = 0
    n = len(eval_pairs)

    logger.info(f"  Evaluating {n:,} alias queries...")

    for i, (alias, ground_truth) in enumerate(eval_pairs, start=1):
        if i % 100 == 0 or i == n:
            logger.info(f"    Progress: {i:,} / {n:,}")

        teacher_pred = top1_label(teacher_model, teacher_index, label_list, alias)
        student_pred = top1_label(student_model, student_index, label_list, alias)

        # Exact string match against ground truth (case-sensitive, consistent
        # with how labels are stored in the dataset).
        if teacher_pred == ground_truth:
            teacher_correct += 1
        if student_pred == ground_truth:
            student_correct += 1
        if teacher_pred == student_pred:
            agreements += 1

    return {
        "teacher_accuracy": teacher_correct / n,
        "student_accuracy": student_correct / n,
        "agreement_rate": agreements / n,
        "n_pairs": n,
    }


def print_gate_report(metrics: dict) -> bool:
    """
    Print a formatted pass/fail gate report and return True if all gates pass.

    Parameters
    ----------
    metrics : dict
        Output of run_gate_evaluation().

    Returns
    -------
    bool
        True if all three gates pass, False otherwise.
    """
    teacher_acc = metrics["teacher_accuracy"]
    student_acc = metrics["student_accuracy"]
    agreement = metrics["agreement_rate"]
    n = metrics["n_pairs"]

    # Gate evaluations
    gate_a_threshold = teacher_acc - GATE_A_MAX_GAP
    gate_a_pass = student_acc >= gate_a_threshold
    gate_b_pass = agreement >= GATE_B_AGREEMENT

    all_pass = gate_a_pass and gate_b_pass

    sep = "=" * 70
    print()
    print(sep)
    print("  STUDENT MODEL QUALITY GATE REPORT")
    print(sep)
    print(f"  Evaluation pairs : {n:,}")
    print()
    print(f"  Teacher top-1 accuracy : {teacher_acc:.4f}  ({teacher_acc*100:.2f} %)")
    print(f"  Student top-1 accuracy : {student_acc:.4f}  ({student_acc*100:.2f} %)")
    print(f"  Teacher-student agreement : {agreement:.4f}  ({agreement*100:.2f} %)")
    print()
    print("  Gate Results")
    print("  " + "-" * 60)
    print(
        f"  Gate A — Accuracy gap <= 5 pp     : "
        f"{'PASS' if gate_a_pass else 'FAIL'}"
        f"  (student={student_acc*100:.2f}%  threshold={gate_a_threshold*100:.2f}%)"
    )
    print(
        f"  Gate B — Agreement >= {GATE_B_AGREEMENT*100:.0f} %         : "
        f"{'PASS' if gate_b_pass else 'FAIL'}"
        f"  ({agreement*100:.2f}%)"
    )
    print()
    verdict = "ALL GATES PASSED — student model is ready for deployment." if all_pass else (
        "ONE OR MORE GATES FAILED."
    )
    print(f"  Verdict: {verdict}")
    print(sep)
    print()

    # Mirror the report to the log file as well.
    logger.info("Gate A (accuracy gap <=5pp): %s  student=%.4f  threshold=%.4f",
                "PASS" if gate_a_pass else "FAIL", student_acc, gate_a_threshold)
    logger.info("Gate B (agreement >=%.2f): %s  agreement=%.4f",
                GATE_B_AGREEMENT, "PASS" if gate_b_pass else "FAIL", agreement)
    logger.info("Overall verdict: %s", "PASS" if all_pass else "FAIL")

    return all_pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate student model quality against the teacher model."
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=1000,
        help="Number of alias-label pairs to evaluate (default: 1000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42).",
    )
    args = parser.parse_args()

    STUDENT_MODEL_PATH = STUDENT_MODEL_PATH_DISTILLED
    STUDENT_LABELS_INDEX = STUDENT_LABELS_INDEX_DISTILLED
    student_label = "distilled (distill_model.py)"

    logger.info("=" * 70)
    logger.info("  alias-label-retriever  |  evaluate_student.py")
    logger.info(f"  Student variant: {student_label}")
    logger.info("=" * 70)

    # -----------------------------------------------------------------------
    # Step 0 — Detect compute device
    # -----------------------------------------------------------------------
    device = get_device()

    # -----------------------------------------------------------------------
    # Step 1 — Load models
    # -----------------------------------------------------------------------
    logger.info("Step 1 — Loading teacher and student models...")
    teacher_model = load_model(TEACHER_MODEL_PATH, device)
    student_model = load_model(STUDENT_MODEL_PATH, device)

    # -----------------------------------------------------------------------
    # Step 2 — Load FAISS indexes
    # -----------------------------------------------------------------------
    logger.info("Step 2 — Loading FAISS label indexes...")
    teacher_index = load_faiss_index(TEACHER_LABELS_INDEX)
    student_index = load_faiss_index(STUDENT_LABELS_INDEX)

    assert teacher_index.d == teacher_model.get_sentence_embedding_dimension(), (
        f"Teacher index dimension ({teacher_index.d}) does not match "
        f"teacher model output ({teacher_model.get_sentence_embedding_dimension()})."
    )
    assert student_index.d == student_model.get_sentence_embedding_dimension(), (
        f"Student index dimension ({student_index.d}) does not match "
        f"student model output ({student_model.get_sentence_embedding_dimension()})."
    )

    # -----------------------------------------------------------------------
    # Step 3 — Load dataset and build the shared label list
    # -----------------------------------------------------------------------
    # Both models search the same set of canonical labels.  The label list
    # ordering must match the order in which embeddings were added to each
    # FAISS index — which is the order returned by df["label"].unique().
    # train_alias_label.py and distill_model.py both use the same pattern,
    # so the ordering is consistent between teacher and student indexes as long
    # as neither the parquet file nor the training scripts have changed.
    # -----------------------------------------------------------------------
    logger.info(f"Step 3 — Loading dataset: {RAW_PARQUET}")
    if not RAW_PARQUET.exists():
        raise FileNotFoundError(f"Dataset not found: {RAW_PARQUET}")
    df = pd.read_parquet(RAW_PARQUET)
    label_list = df["label"].dropna().unique().tolist()
    logger.info(f"  Canonical label pool size: {len(label_list):,}")

    # -----------------------------------------------------------------------
    # Step 4 — Sample evaluation pairs
    # -----------------------------------------------------------------------
    logger.info(f"Step 4 — Sampling {args.sample:,} evaluation pairs (seed={args.seed})...")
    eval_pairs = sample_eval_pairs(df, n=args.sample, seed=args.seed)

    # -----------------------------------------------------------------------
    # Step 5 — Run retrieval and compute metrics
    # -----------------------------------------------------------------------
    logger.info("Step 5 — Running retrieval evaluation...")
    metrics = run_gate_evaluation(
        teacher_model=teacher_model,
        teacher_index=teacher_index,
        student_model=student_model,
        student_index=student_index,
        label_list=label_list,
        eval_pairs=eval_pairs,
    )

    # -----------------------------------------------------------------------
    # Step 6 — Print gate report
    # -----------------------------------------------------------------------
    all_pass = print_gate_report(metrics)

    # Exit with code 1 if any gate fails so CI pipelines can detect failure.
    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
