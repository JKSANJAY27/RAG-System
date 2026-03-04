"""
evals/metrics.py — Evaluation Scoring Functions (Phase 3)

WHY EVALUATION MATTERS:
    A RAG system without evaluation is a black box.
    You need to know:
        - Is the system finding the right chunks?
        - Is the generated answer actually correct?
        - Is the citation rate acceptable?
        - When we change something, did quality improve or regress?

    These functions implement the core metrics. They are intentionally
    simple — complexity in metrics makes debugging harder.

THE THREE METRICS:

    1. contains_check (precision for key facts)
       The answer must contain all expected keywords/phrases.
       Binary: 1.0 if all present, fraction otherwise.
       WHY: Catches hallucinations and missing facts.

    2. token_f1 (lexical similarity to reference)
       SQuAD-style token overlap F1 between generated and reference answer.
       Range: 0.0 (no overlap) to 1.0 (perfect match).
       WHY: Measures answer quality when we have a reference answer.

    3. citation_check (grounding check)
       Does the answer cite at least one source?
       WHY: Citation = the system retrieved something relevant.
            No citation = either declined OR hallucinated.

    4. faithfulness_check (no hallucination check)
       What fraction of answer tokens appear in the retrieved chunks?
       WHY: High faithfulness = answer is grounded in retrieved text.

HOW TO RUN:
    These are imported and called by evals/run_evals.py
"""

import re
from typing import List


def contains_check(answer: str, expected_keywords: List[str]) -> float:
    """
    Check whether the answer contains all expected keywords/phrases.

    Case-insensitive. Each keyword counts equally toward the score.

    Args:
        answer: The generated answer string.
        expected_keywords: List of strings that should appear in the answer.

    Returns:
        Float in [0, 1]. 1.0 = all keywords found, 0.0 = none found.

    Example:
        >>> contains_check("Self-attention uses Query, Key, and Value vectors.",
        ...                ["query", "key", "value"])
        1.0
        >>> contains_check("Transformers are good.", ["query", "key", "value"])
        0.0
    """
    if not expected_keywords:
        return 1.0

    answer_lower = answer.lower()
    found = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return round(found / len(expected_keywords), 4)


def token_f1(prediction: str, reference: str) -> float:
    """
    Compute token-level F1 between prediction and reference (SQuAD-style).

    Steps:
        1. Tokenize both strings (lowercase, split on whitespace/punctuation)
        2. Find the set of shared tokens
        3. Precision = shared / prediction_tokens
        4. Recall    = shared / reference_tokens
        5. F1        = 2 × (precision × recall) / (precision + recall)

    This is the same metric used to evaluate SQuAD reading comprehension
    models — it's robust to paraphrasing while still measuring correctness.

    Args:
        prediction: The generated answer.
        reference: The expected/gold answer.

    Returns:
        F1 score in [0, 1]. 0.0 = no overlap, 1.0 = identical.
    """
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)

    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_set = set(pred_tokens)
    ref_set = set(ref_tokens)
    common = pred_set & ref_set

    if not common:
        return 0.0

    precision = len(common) / len(pred_set)
    recall = len(common) / len(ref_set)
    f1 = 2 * precision * recall / (precision + recall)
    return round(f1, 4)


def citation_check(answer: str, sources: List[str]) -> bool:
    """
    Check whether the answer contains at least one source citation.

    We check for [Source: ...] patterns in the answer text, OR if the
    sources list is non-empty (meaning the pipeline returned evidence).

    Args:
        answer: The generated answer.
        sources: List of source paths/URLs returned by the pipeline.

    Returns:
        True if the answer appears to have cited a source.
    """
    # Non-empty sources list = retrieval found something
    if sources:
        return True

    # Also check for inline citation pattern in the answer text
    return bool(re.search(r"\[source:", answer.lower()))


def faithfulness_score(answer: str, chunk_texts: List[str]) -> float:
    """
    Estimate what fraction of the answer is grounded in retrieved chunks.

    METHOD:
        For each token in the answer, check if it appears in ANY of the
        retrieved chunks. Faithfulness = grounded_tokens / total_tokens.

    This is a simplified version of RAGAS faithfulness. A production system
    would use an LLM-as-judge approach, but this is fast and free.

    Args:
        answer: The generated answer.
        chunk_texts: List of retrieved chunk texts used for generation.

    Returns:
        Float in [0, 1]. 1.0 = every answer token found in chunks.
    """
    answer_tokens = set(_tokenize(answer))
    if not answer_tokens:
        return 0.0

    if not chunk_texts:
        return 0.0

    # Combine all chunk text into one large searchable corpus
    corpus_tokens = set()
    for chunk in chunk_texts:
        corpus_tokens.update(_tokenize(chunk))

    grounded = answer_tokens & corpus_tokens
    return round(len(grounded) / len(answer_tokens), 4)


def score_response(
    answer: str,
    sources: List[str],
    chunk_texts: List[str],
    expected_keywords: List[str],
    reference_answer: str,
) -> dict:
    """
    Compute all metrics for one question-answer pair. Returns a dict.

    This is the single function called by the evaluation runner for each
    entry in the golden dataset.

    Returns:
        {
            "contains_score": float,   # fraction of expected keywords found
            "token_f1": float,         # F1 vs reference answer
            "citations_present": bool, # did the answer cite sources?
            "faithfulness": float,     # fraction of answer grounded in chunks
            "declined": bool,          # True if system declined to answer
        }
    """
    declined = (len(sources) == 0 and len(chunk_texts) == 0)

    return {
        "contains_score": contains_check(answer, expected_keywords),
        "token_f1": token_f1(answer, reference_answer) if reference_answer else None,
        "citations_present": citation_check(answer, sources),
        "faithfulness": faithfulness_score(answer, chunk_texts),
        "declined": declined,
    }


# ─── Internal Helper ──────────────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    """Lowercase tokenizer: split on non-word chars, drop empties."""
    return [t for t in re.split(r"\W+", text.lower()) if t]
