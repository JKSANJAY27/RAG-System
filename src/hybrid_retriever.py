"""
src/hybrid_retriever.py — Hybrid Retrieval with Reciprocal Rank Fusion

WHAT IS HYBRID RETRIEVAL?
    Combining TWO search strategies:
        1. BM25 (keyword search)  — finds chunks with matching WORDS
        2. Vector search          — finds chunks with matching MEANING

    Then fusing their ranked lists into a single ranked list using
    Reciprocal Rank Fusion (RRF).

    Final step: Re-ranking with a cross-encoder for precision.

WHY HYBRID IS BETTER THAN EITHER ALONE:
    ┌─────────────────────┬──────────────────────────────────────────────┐
    │ Scenario            │ Which search wins                            │
    ├─────────────────────┼──────────────────────────────────────────────┤
    │ "What is attention?"│ Vector (semantic, finds explanations)        │
    │ "HTTP 404 error"    │ BM25 (finds exact code/term matches)         │
    │ "Vaswani et al."    │ BM25 (author name = exact keyword match)     │
    │ "Main idea of paper"│ Vector (conceptual query)                    │
    └─────────────────────┴──────────────────────────────────────────────┘

RECIPROCAL RANK FUSION (RRF):
    For each document d, across ranking systems s1, s2:
        RRF_score(d) = Σ  1 / (k + rank_s(d))
                       s∈{bm25, vector}

    where k=60 is a constant (standard choice, validated in literature).

    Example:
        Doc A: BM25 rank 1, Vector rank 3  → 1/(60+1) + 1/(60+3) = 0.032
        Doc B: BM25 rank 5, Vector rank 1  → 1/(60+5) + 1/(60+1) = 0.032
        Doc C: BM25 rank 2, Vector rank 2  → 1/(60+2) + 1/(60+2) = 0.032

    RRF promotes documents that rank well in BOTH systems over documents
    that rank first in only one. This is why it's powerful.

PIPELINE:
    query
      │
      ├── BM25 search  → top 10 [keyword candidates]
      ├── Vector search → top 10 [semantic candidates]
      │
      ├── RRF fusion   → merged top 10 [best of both worlds]
      │
      └── Cross-encoder re-ranking → top 3 [precision-filtered]
                                         ↓
                                    Citation enforcement
                                    (decline if below threshold)
"""

from typing import Dict, List, Optional, Tuple

from src.bm25_store import BM25Store
from src.embedder import Embedder
from src.reranker import CrossEncoderReranker
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
        rankings: List of ranked lists (each list is one retriever's results).
        k: RRF constant (60 is the standard, from the original 2009 paper).

    Returns:
        Single merged list, sorted by RRF score descending.
        Scores are stored in the .score field of RetrievedChunk.
    """
    # Map from unique chunk identity → accumulated RRF score + original chunk
    rrf_scores: Dict[str, float] = {}
    chunk_map: Dict[str, RetrievedChunk] = {}

    for ranked_list in rankings:
        for rank, chunk in enumerate(ranked_list):
            # Use source + chunk_index as the unique key for this chunk
            chunk_id = f"{chunk.source}::chunk_{chunk.chunk_index}"

            # Accumulate the RRF contribution from this ranking list
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)

            # Keep the chunk object (from whichever list we saw it first)
            if chunk_id not in chunk_map:
                chunk_map[chunk_id] = chunk

    # Build result list sorted by accumulated RRF score
    merged = []
    for chunk_id, rrf_score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
        chunk = chunk_map[chunk_id]
        merged.append(
            RetrievedChunk(
                text=chunk.text,
                source=chunk.source,
                chunk_index=chunk.chunk_index,
                score=round(rrf_score, 6),   # RRF score replaces original score
                metadata=chunk.metadata,
            )
        )

    return merged


# ─── Hybrid Retriever ─────────────────────────────────────────────────────────

class HybridRetriever:
    """
    Production retriever: BM25 + Vector Search + RRF + Cross-Encoder Re-Ranking.

    This is the core of Phase 2 and represents how enterprise RAG systems
    actually work in production. The three stages are:
        1. Recall:    BM25 + vector search get a broad candidate set
        2. Fusion:    RRF merges them fairly
        3. Precision: cross-encoder re-ranks for accuracy + citation enforcement
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
    ) -> List[RetrievedChunk]:
        """
        Run the full hybrid retrieval pipeline.

        Args:
            query: User's question in plain English.
            top_k: Final number of chunks after re-ranking.
                   Defaults to settings.reranker_top_k.
            candidate_k: How many candidates each retriever fetches before fusion.
                         Defaults to settings.top_k * 2 (cast a wider net).

        Returns:
            Re-ranked list of RetrievedChunk objects, most relevant first.
            Returns empty list if citation enforcement threshold is not met.

        Raises:
            ValueError: If the vector store is empty.
        """
        if self._vector_store.count() == 0:
            raise ValueError(
                "The vector store is empty! Please ingest documents first:\n"
                "  python ingest.py --source <path_or_url> --type <pdf|markdown|web>"
            )

        k = top_k or settings.reranker_top_k
        # Fetch more candidates than we'll keep — gives RRF more material to work with
        ck = candidate_k or (settings.top_k * 2)

        print(f"\n  ⟳ HYBRID RETRIEVAL for: '{query[:60]}...'")
        print(f"     Candidate pool: {ck} per retriever → RRF → re-rank to {k}")

        # ── Stage 1a: BM25 keyword retrieval ──────────────────────────────────
        bm25_results = self._bm25_store.search(query, top_k=ck)
        print(f"  ✓ BM25: {len(bm25_results)} candidates")

        # ── Stage 1b: Vector semantic retrieval ───────────────────────────────
        query_embedding = self._embedder.embed_single(query)
        vector_results = self._vector_store.query(query_embedding, top_k=ck)
        print(f"  ✓ Vector: {len(vector_results)} candidates")

        if not bm25_results and not vector_results:
            return []

        # ── Stage 2: Reciprocal Rank Fusion ───────────────────────────────────
        # Only pass non-empty rankings to avoid RRF treating missing as rank 0
        rankings_to_fuse = [r for r in [bm25_results, vector_results] if r]
        fused = _reciprocal_rank_fusion(rankings_to_fuse)
        print(f"  ✓ RRF: merged into {len(fused)} unique candidates")

        # Show the top candidates before re-ranking (for transparency)
        for i, chunk in enumerate(fused[:5], 1):
            print(
                f"    [{i}] rrf={chunk.score:.5f} | "
                f"{chunk.source.split('/')[-1]} | chunk_{chunk.chunk_index}"
            )

        # ── Stage 3: Cross-Encoder Re-Ranking + Citation Enforcement ──────────
        ranked = self._reranker.rerank(query, fused, top_k=k)

        if not ranked:
            # Citation enforcement triggered: return empty (generator will decline)
            return []

        # Convert back to RetrievedChunks for the generator
        final = self._reranker.to_retrieved_chunks(ranked)
        print(f"  ✓ Final: {len(final)} chunks after re-ranking")

        return final

    def refresh_bm25(self) -> None:
        """Refresh the BM25 index after new documents are ingested."""
        self._bm25_store.refresh()
