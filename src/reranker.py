"""
src/reranker.py — Cross-Encoder Re-Ranking Module

WHAT IS RE-RANKING?
    After hybrid retrieval gives us a candidate set of chunks, re-ranking
    applies a much more powerful (but slower) model to score each chunk
    individually with respect to the query.

    TWO-STAGE RETRIEVAL PATTERN (industry standard):
        Stage 1 — RETRIEVAL:  Fast, approximate.
                               Returns top 10-20 candidate chunks.
                               Tools: BM25 + vector search (what Phase 2 adds)
        Stage 2 — RE-RANKING: Slow, precise.
                               Reorders the candidates by true relevance.
                               Tools: cross-encoder model

BI-ENCODER vs CROSS-ENCODER (why this matters):
    Our embedding model (all-MiniLM-L6-v2) is a BI-ENCODER:
      - Encodes query → vector, document → vector SEPARATELY
      - Compares them with cosine similarity
      - FAST: O(1) to compare once vectors are computed

    The re-ranker (cross-encoder/ms-marco-MiniLM-L-6-v2) is a CROSS-ENCODER:
      - Takes query + chunk TOGETHER as one input
      - Attends to both simultaneously (full cross-attention)
      - Can capture subtle relevance signals a bi-encoder misses
      - SLOW: must run inference for every (query, chunk) pair
      - But since we only run it on 10-20 candidates (not millions),
        it's fast enough in practice.

CITATION ENFORCEMENT:
    This is the "don't hallucinate" feature. If the best re-ranker score
    is below a threshold (configurable in .env), we refuse to answer
    instead of generating something unsupported. This is what separates
    a production system from a demo.

MODEL: cross-encoder/ms-marco-MiniLM-L-6-v2
    - Trained on MS MARCO (a massive question-answering dataset)
    - ~23MB download, runs fast on CPU
    - Scores are raw logits (can be any real number)
    - We apply sigmoid() to normalize to [0, 1] for interpretability
"""

import math
from dataclasses import dataclass
from typing import List, Optional

from sentence_transformers import CrossEncoder

from src.vector_store import RetrievedChunk
from config import settings


# ─── Data Models ──────────────────────────────────────────────────────────────

@dataclass
class RankedChunk:
    """A re-ranked chunk with both its original retrieval score and re-ranker score."""
    text: str
    source: str
    chunk_index: int
    retrieval_score: float    # Score from hybrid retrieval (before re-ranking)
    rerank_score: float       # Raw cross-encoder logit
    rerank_score_normalized: float  # sigmoid(rerank_score) → [0, 1]
    metadata: dict


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _sigmoid(x: float) -> float:
    """Maps any real number to [0, 1]. Used to normalize cross-encoder scores."""
    return 1.0 / (1.0 + math.exp(-x))


# ─── Re-Ranker ────────────────────────────────────────────────────────────────

class CrossEncoderReranker:
    """
    Uses a cross-encoder model to re-score retrieved chunks by relevance.

    WORKFLOW:
        1. Takes: query (str) + list of candidate chunks
        2. Builds pairs: [(query, chunk1_text), (query, chunk2_text), ...]
        3. Cross-encoder scores each pair
        4. Sorts by score descending
        5. Applies citation enforcement: if best score < threshold → decline
        6. Returns top-k re-ranked chunks
    """

    def __init__(self):
        print(f"  ⟳ Loading cross-encoder '{settings.reranker_model}'...")
        # CrossEncoder automatically downloads the model on first use (~23MB)
        self._model = CrossEncoder(
            settings.reranker_model,
            max_length=512,  # Truncate context if too long
        )
        print(f"  ✓ Re-ranker ready.")

    def rerank(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        top_k: Optional[int] = None,
    ) -> List[RankedChunk]:
        """
        Re-rank a list of retrieved chunks by cross-attention relevance.

        Args:
            query: The user's question.
            chunks: Candidate chunks from hybrid retrieval.
            top_k: How many to keep (uses settings.reranker_top_k if None).

        Returns:
            List of RankedChunk, sorted by re-rank score (best first).
            May be empty if citation enforcement threshold is not met.
        """
        k = top_k or settings.reranker_top_k

        if not chunks:
            return []

        # Build pairs for the cross-encoder
        pairs = [(query, chunk.text) for chunk in chunks]

        print(f"  ⟳ Re-ranking {len(pairs)} chunks with cross-encoder...")

        # Predict returns a numpy array of raw logit scores
        raw_scores = self._model.predict(pairs)

        # Build RankedChunk objects with both scores
        ranked = []
        for chunk, raw_score in zip(chunks, raw_scores):
            normalized = _sigmoid(float(raw_score))
            ranked.append(
                RankedChunk(
                    text=chunk.text,
                    source=chunk.source,
                    chunk_index=chunk.chunk_index,
                    retrieval_score=chunk.score,
                    rerank_score=round(float(raw_score), 4),
                    rerank_score_normalized=round(normalized, 4),
                    metadata=chunk.metadata,
                )
            )

        # Sort by re-rank score, best first
        ranked.sort(key=lambda x: x.rerank_score, reverse=True)

        # ── Citation Enforcement ───────────────────────────────────────────
        # If the BEST chunk (after re-ranking) is still below the threshold,
        # return empty list — the generator will decline to answer.
        if ranked and ranked[0].rerank_score_normalized < settings.citation_score_threshold:
            print(
                f"  ⚠ Citation enforcement triggered! "
                f"Best score {ranked[0].rerank_score_normalized:.3f} < "
                f"threshold {settings.citation_score_threshold}. "
                f"System will decline to answer."
            )
            return []

        kept = ranked[:k]
        print(
            f"  ✓ Re-ranking complete. Kept {len(kept)} chunks. "
            f"Scores: {[c.rerank_score_normalized for c in kept]}"
        )
        return kept

    def to_retrieved_chunks(self, ranked: List[RankedChunk]) -> List[RetrievedChunk]:
        """
        Convert RankedChunks back to RetrievedChunks for the generator.
        Uses the normalized re-rank score as the display score.
        """
        return [
            RetrievedChunk(
                text=c.text,
                source=c.source,
                chunk_index=c.chunk_index,
                score=c.rerank_score_normalized,
                metadata={**c.metadata, "rerank_score": c.rerank_score},
            )
            for c in ranked
        ]
