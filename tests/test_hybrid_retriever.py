"""
tests/test_hybrid_retriever.py — Unit/Integration Tests for Hybrid Retrieval

WHAT DO THESE TESTS VERIFY?
    1. RRF correctly merges and scores two ranked lists
    2. Documents appearing in both lists rank higher than those in only one
    3. BM25Store builds from an existing vector store correctly
    4. HybridRetriever raises on empty store
    5. After ingestion, BM25Store refreshes correctly

HOW TO RUN:
    pytest tests/test_hybrid_retriever.py -v
"""

import pytest
from unittest.mock import MagicMock, patch

from src.bm25_store import BM25Store, _tokenize
from src.hybrid_retriever import HybridRetriever, _reciprocal_rank_fusion
from src.vector_store import RetrievedChunk


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_chunk(text: str, source: str, chunk_index: int = 0, score: float = 0.9):
    return RetrievedChunk(
        text=text,
        source=source,
        chunk_index=chunk_index,
        score=score,
        metadata={"token_count": 10},
    )


# ─── Tests: Tokenizer ─────────────────────────────────────────────────────────

class TestTokenizer:
    def test_lowercases(self):
        tokens = _tokenize("Hello WORLD")
        assert all(t == t.lower() for t in tokens)

    def test_splits_on_punctuation(self):
        tokens = _tokenize("self-attention: a mechanism.")
        assert "self" in tokens
        assert "attention" in tokens
        assert "mechanism" in tokens

    def test_empty_string_returns_empty(self):
        assert _tokenize("") == []

    def test_filters_empty_tokens(self):
        tokens = _tokenize("  hello   world  ")
        assert "" not in tokens


# ─── Tests: Reciprocal Rank Fusion ────────────────────────────────────────────

class TestRRF:
    def test_document_in_both_lists_ranks_higher(self):
        """
        Doc A is #1 in BM25 but not in vector.
        Doc B is #1 in vector but not in BM25.
        Doc C is #2 in both — should rank higher than either A or B.
        """
        doc_a = make_chunk("only in bm25", "a.pdf", 0)
        doc_b = make_chunk("only in vector", "b.pdf", 0)
        doc_c_bm25 = make_chunk("in both searches", "c.pdf", 0)
        doc_c_vector = make_chunk("in both searches", "c.pdf", 0)  # Same chunk

        bm25_ranking = [doc_a, doc_c_bm25]
        vector_ranking = [doc_b, doc_c_vector]

        fused = _reciprocal_rank_fusion([bm25_ranking, vector_ranking])

        # Find positions
        sources_in_order = [c.source for c in fused]
        assert "c.pdf" in sources_in_order
        # c.pdf should beat a.pdf (only in 1 list) on RRF score
        c_idx = sources_in_order.index("c.pdf")
        a_idx = sources_in_order.index("a.pdf")
        assert c_idx < a_idx, "Doc in both lists should rank ahead of doc in one list"

    def test_single_ranking_preserved(self):
        """With only one ranking system, order should be preserved by RRF."""
        chunks = [make_chunk(f"text {i}", f"{i}.pdf", 0, score=1.0-i*0.1) for i in range(3)]
        fused = _reciprocal_rank_fusion([chunks])
        # RRF rank 1 should be first
        assert fused[0].source == "0.pdf"

    def test_empty_rankings_returns_empty(self):
        assert _reciprocal_rank_fusion([]) == []

    def test_deduplication(self):
        """Same chunk appearing in both lists should appear ONCE in output."""
        chunk = make_chunk("shared", "doc.pdf", 0)
        fused = _reciprocal_rank_fusion([[chunk], [chunk]])
        doc_sources = [c.source for c in fused]
        assert doc_sources.count("doc.pdf") == 1


# ─── Tests: BM25Store ─────────────────────────────────────────────────────────

class TestBM25Store:
    def test_builds_from_empty_store(self):
        """BM25Store should handle an empty vector store gracefully."""
        mock_vs = MagicMock()
        mock_vs.get_all_chunks.return_value = []
        mock_vs.count.return_value = 0

        store = BM25Store(mock_vs)
        results = store.search("attention mechanism", top_k=5)
        assert results == []

    def test_search_returns_relevant_chunks(self):
        """BM25 should return chunks that contain query keywords."""
        mock_vs = MagicMock()
        mock_vs.get_all_chunks.return_value = [
            make_chunk("self-attention is a key mechanism in transformers", "paper.pdf", 0),
            make_chunk("the stock market crashed yesterday", "news.pdf", 1),
            make_chunk("attention allows the model to focus on relevant words", "paper.pdf", 2),
        ]
        store = BM25Store(mock_vs)
        results = store.search("attention mechanism", top_k=2)

        # The attention chunks should rank above the stock market chunk
        assert len(results) == 2
        sources = [r.source for r in results]
        assert "paper.pdf" in sources

    def test_refresh_rebuilds_index(self):
        """After refresh, doc_count should reflect new corpus."""
        mock_vs = MagicMock()
        mock_vs.get_all_chunks.return_value = []

        store = BM25Store(mock_vs)
        assert store.doc_count == 0

        # Simulate new documents added
        mock_vs.get_all_chunks.return_value = [
            make_chunk("new document text", "new.pdf", 0)
        ]
        store.refresh()
        assert store.doc_count == 1


# ─── Tests: HybridRetriever ───────────────────────────────────────────────────

class TestHybridRetriever:
    def test_raises_on_empty_vector_store(self):
        """Should raise ValueError with helpful message when store is empty."""
        mock_embedder = MagicMock()
        mock_vs = MagicMock()
        mock_vs.count.return_value = 0
        mock_bm25 = MagicMock()
        mock_reranker = MagicMock()

        retriever = HybridRetriever(
            embedder=mock_embedder,
            vector_store=mock_vs,
            bm25_store=mock_bm25,
            reranker=mock_reranker,
        )

        with pytest.raises(ValueError, match="empty"):
            retriever.retrieve("What is X?")
