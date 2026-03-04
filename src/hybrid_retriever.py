"""
src/hybrid_retriever.py — Hybrid Retrieval with Reciprocal Rank Fusion (Phase 4)

WHAT CHANGED FROM PHASE 3?
    Added optional `trace_ctx` parameter to .retrieve().
    Now records five spans for every query:

        "bm25"       — keyword candidates, BM25 scores for top-5
        "vector"     — semantic candidates, cosine scores for top-5
        "rrf_fusion" — merged candidate count, top-5 RRF scores
        "rerank"     — before/after comparison: shows how re-ranking
                       changes the order, plus citation enforcement result

    This is the "glass box" upgrade. You can now answer:
        "Why was chunk 4 the top result even though BM25 didn't rank it first?"
        "Did citation enforcement fire? What was the highest score it saw?"
        "How long did the cross-encoder take vs. vector retrieval?"

WHY THE BEFORE/AFTER RERANK LOG IS VALUABLE:
    Before rerank (RRF order):
        chunk_5: rrf=0.03247 | transformer_architecture.md
        chunk_2: rrf=0.03101 | transformer_architecture.md
        chunk_8: rrf=0.01600 | transformer_architecture.md

    After rerank (cross-encoder order):
        chunk_2: ce_score=0.93 ← moved UP (RRF undervalued it)
        chunk_5: ce_score=0.71 ← stayed high
        chunk_8: ce_score=0.12 ← dropped (almost below citation threshold)

    This is the verification that re-ranking actually helps — without this
    log you can't tell if the cross-encoder is doing useful work.
"""

import time
from typing import Dict, List, Optional, Tuple

from src.bm25_store import BM25Store
from src.embedder import Embedder
from src.reranker import CrossEncoderReranker
from src.trace_context import TraceContext, SpanTimer
from src.vector_store import RetrievedChunk, VectorStore
from config import settings


# ─── RRF Implementation ───────────────────────────────────────────────────────

def _reciprocal_rank_fusion(
    rankings: List[List[RetrievedChunk]],
    k: int = 60,
) -> List[RetrievedChunk]:
    """
    Merge multiple ranked lists using Reciprocal Rank Fusion.

    Args:
        rankings: List of ranked lists (each is one retriever's results).
        k: RRF constant (60 from the original 2009 paper).

    Returns:
        Single merged list sorted by RRF score descending.
    """
    rrf_scores: Dict[str, float] = {}
    chunk_map: Dict[str, RetrievedChunk] = {}

    for ranked_list in rankings:
        for rank, chunk in enumerate(ranked_list):
            chunk_id = f"{chunk.source}::chunk_{chunk.chunk_index}"
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
            if chunk_id not in chunk_map:
                chunk_map[chunk_id] = chunk

    merged = []
    for chunk_id, rrf_score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
        chunk = chunk_map[chunk_id]
        merged.append(
            RetrievedChunk(
                text=chunk.text,
                source=chunk.source,
                chunk_index=chunk.chunk_index,
                score=round(rrf_score, 6),
                metadata=chunk.metadata,
            )
        )

    return merged


# ─── Hybrid Retriever ─────────────────────────────────────────────────────────

class HybridRetriever:
    """
    Production retriever: BM25 + Vector Search + RRF + Cross-Encoder Re-Ranking.

    Phase 4: All stages timed individually and recorded to trace_ctx when provided.
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        bm25_store: BM25Store,
        reranker: CrossEncoderReranker,
    ):
        self._embedder = embedder
        self._vector_store = vector_store
        self._bm25_store = bm25_store
        self._reranker = reranker

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        candidate_k: Optional[int] = None,
        trace_ctx: Optional[TraceContext] = None,
    ) -> List[RetrievedChunk]:
        """
        Run the full hybrid retrieval pipeline.

        Args:
            query: User's question.
            top_k: Final chunks to keep after re-ranking.
            candidate_k: Candidates to fetch per retriever before fusion.
            trace_ctx: Optional Phase 4 trace context. When provided, each
                       retrieval stage records its own span with timing and scores.

        Returns:
            Re-ranked RetrievedChunk list (empty if citation enforcement fires).
        """
        if self._vector_store.count() == 0:
            raise ValueError(
                "The vector store is empty! Please ingest documents first:\n"
                "  python ingest.py --source <path_or_url> --type <pdf|markdown|web>"
            )

        k = top_k or settings.reranker_top_k
        ck = candidate_k or (settings.top_k * 2)

        print(f"\n  ⟳ HYBRID RETRIEVAL for: '{query[:60]}...'")
        print(f"     Candidate pool: {ck} per retriever → RRF → re-rank to {k}")

        # ── Stage 1a: BM25 keyword retrieval ──────────────────────────────────
        with SpanTimer() as bm25_timer:
            bm25_results = self._bm25_store.search(query, top_k=ck)

        print(f"  ✓ BM25: {len(bm25_results)} candidates in {bm25_timer.elapsed_ms:.0f}ms")

        if trace_ctx is not None:
            trace_ctx.record(
                name="bm25",
                latency_ms=bm25_timer.elapsed_ms,
                input={"query": query, "top_k": ck},
                output={
                    "candidates": len(bm25_results),
                    "top_scores": [round(c.score, 4) for c in bm25_results[:5]],
                    "top_sources": [c.source.split("/")[-1] for c in bm25_results[:5]],
                },
            )

        # ── Stage 1b: Vector semantic retrieval ───────────────────────────────
        with SpanTimer() as vec_timer:
            query_embedding = self._embedder.embed_single(query)
            vector_results = self._vector_store.query(query_embedding, top_k=ck)

        print(f"  ✓ Vector: {len(vector_results)} candidates in {vec_timer.elapsed_ms:.0f}ms")

        if trace_ctx is not None:
            trace_ctx.record(
                name="vector",
                latency_ms=vec_timer.elapsed_ms,
                input={"query": query, "top_k": ck},
                output={
                    "candidates": len(vector_results),
                    "top_scores": [round(c.score, 4) for c in vector_results[:5]],
                    "top_sources": [c.source.split("/")[-1] for c in vector_results[:5]],
                },
            )

        if not bm25_results and not vector_results:
            return []

        # ── Stage 2: Reciprocal Rank Fusion ───────────────────────────────────
        with SpanTimer() as rrf_timer:
            rankings_to_fuse = [r for r in [bm25_results, vector_results] if r]
            fused = _reciprocal_rank_fusion(rankings_to_fuse)

        print(f"  ✓ RRF: {len(fused)} unique candidates in {rrf_timer.elapsed_ms:.0f}ms")

        if trace_ctx is not None:
            trace_ctx.record(
                name="rrf_fusion",
                latency_ms=rrf_timer.elapsed_ms,
                input={
                    "bm25_count": len(bm25_results),
                    "vector_count": len(vector_results),
                    "k_constant": 60,
                },
                output={
                    "fused_count": len(fused),
                    "top_rrf_scores": [round(c.score, 6) for c in fused[:5]],
                    "top_sources": [c.source.split("/")[-1] for c in fused[:5]],
                },
            )

        # Print top-5 pre-rerank for transparency
        for i, chunk in enumerate(fused[:5], 1):
            print(
                f"    [{i}] rrf={chunk.score:.5f} | "
                f"{chunk.source.split('/')[-1]} | chunk_{chunk.chunk_index}"
            )

        # ── Stage 3: Cross-Encoder Re-Ranking + Citation Enforcement ──────────
        # Save the pre-rerank order so we can log the before/after comparison
        pre_rerank_snapshot = [
            {"source": c.source.split("/")[-1], "chunk_idx": c.chunk_index,
             "rrf_score": round(c.score, 6)}
            for c in fused[:10]
        ]

        with SpanTimer() as rerank_timer:
            ranked = self._reranker.rerank(query, fused, top_k=k)

        citation_fired = len(ranked) == 0

        if trace_ctx is not None:
            post_rerank_snapshot = [
                {"source": r.source.split("/")[-1],
                 "chunk_idx": r.chunk_index,
                 "ce_score": round(r.rerank_score, 4)}
                for r in ranked
            ] if ranked else []

            trace_ctx.record(
                name="rerank",
                latency_ms=rerank_timer.elapsed_ms,
                input={
                    "input_count": len(fused),
                    "target_top_k": k,
                    "citation_threshold": settings.citation_score_threshold,
                    "before_rerank": pre_rerank_snapshot,
                },
                output={
                    "kept_count": len(ranked),
                    "citation_enforced": citation_fired,
                    "top_score": round(ranked[0].rerank_score, 4) if ranked else 0.0,
                    "after_rerank": post_rerank_snapshot,
                },
            )

        if citation_fired:
            return []

        final = self._reranker.to_retrieved_chunks(ranked)
        print(f"  ✓ Final: {len(final)} chunks after re-ranking "
              f"in {rerank_timer.elapsed_ms:.0f}ms")

        return final

    def refresh_bm25(self) -> None:
        self._bm25_store.refresh()
