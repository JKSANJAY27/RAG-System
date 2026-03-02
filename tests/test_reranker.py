"""
tests/test_reranker.py — Unit Tests for the Cross-Encoder Re-Ranker

WHAT DO THESE TESTS VERIFY?
    1. Re-ranker correctly sorts chunks by relevance score
    2. Citation enforcement fires when best score is below threshold
    3. Citation enforcement passes when score is above threshold
    4. to_retrieved_chunks() correctly converts RankedChunks back

NOTE: We mock the CrossEncoder model to avoid requiring model downloads in CI.
      This tests the LOGIC (sorting, enforcement, conversion) not the model.

HOW TO RUN:
    pytest tests/test_reranker.py -v
"""

from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from src.reranker import CrossEncoderReranker, RankedChunk, _sigmoid
from src.vector_store import RetrievedChunk
from config import settings


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_chunk(text: str, source: str, chunk_index: int = 0, score: float = 0.8):
    return RetrievedChunk(
        text=text, source=source, chunk_index=chunk_index, score=score,
        metadata={"token_count": 20},
    )


# ─── Tests: Sigmoid ───────────────────────────────────────────────────────────

class TestSigmoid:
    def test_zero_maps_to_half(self):
        assert abs(_sigmoid(0.0) - 0.5) < 1e-6

    def test_large_positive_maps_near_one(self):
        assert _sigmoid(100.0) > 0.99

    def test_large_negative_maps_near_zero(self):
        assert _sigmoid(-100.0) < 0.01

    def test_output_in_range(self):
        for x in [-10, -1, 0, 1, 10]:
            s = _sigmoid(x)
            assert 0.0 <= s <= 1.0


# ─── Tests: CrossEncoderReranker ──────────────────────────────────────────────

class TestCrossEncoderReranker:
    """All tests mock CrossEncoder to avoid downloading the model."""

    @patch("src.reranker.CrossEncoder")
    def test_reranks_in_correct_order(self, MockCE):
        """Higher raw scores should appear first after re-ranking."""
        mock_model = MagicMock()
        # Returns scores: chunk 0 has low score (-3), chunk 1 has high score (5)
        mock_model.predict.return_value = np.array([-3.0, 5.0])
        MockCE.return_value = mock_model

        reranker = CrossEncoderReranker()
        chunks = [
            make_chunk("low relevance text", "docs/low.pdf", 0),
            make_chunk("high relevance text", "docs/high.pdf", 1),
        ]

        # Set threshold low so no enforcement triggers
        original_threshold = settings.citation_score_threshold
        settings.citation_score_threshold = 0.0

        try:
            result = reranker.rerank("some query", chunks, top_k=2)
            assert len(result) == 2
            # high.pdf should be first (score=5.0 > -3.0)
            assert result[0].source == "docs/high.pdf"
            assert result[1].source == "docs/low.pdf"
        finally:
            settings.citation_score_threshold = original_threshold

    @patch("src.reranker.CrossEncoder")
    def test_citation_enforcement_fires_when_below_threshold(self, MockCE):
        """
        If the best score is below the threshold, rerank() should return []
        so the generator knows to decline.
        """
        mock_model = MagicMock()
        # Very negative scores → sigmoid ≈ 0.0 → below any reasonable threshold
        mock_model.predict.return_value = np.array([-10.0, -12.0])
        MockCE.return_value = mock_model

        reranker = CrossEncoderReranker()
        # Force threshold to something that will definitely trigger
        original_threshold = settings.citation_score_threshold
        settings.citation_score_threshold = 0.5  # 50% — very negative scores won't pass

        try:
            chunks = [
                make_chunk("irrelevant content", "doc.pdf", 0),
                make_chunk("also irrelevant", "doc.pdf", 1),
            ]
            result = reranker.rerank("very specific query", chunks, top_k=2)
            assert result == [], "Expected empty list when citation enforcement fires"
        finally:
            settings.citation_score_threshold = original_threshold

    @patch("src.reranker.CrossEncoder")
    def test_citation_enforcement_passes_when_above_threshold(self, MockCE):
        """High scores should NOT trigger citation enforcement."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([8.0])  # sigmoid(8) ≈ 0.9997
        MockCE.return_value = mock_model

        reranker = CrossEncoderReranker()
        original_threshold = settings.citation_score_threshold
        settings.citation_score_threshold = 0.1

        try:
            chunks = [make_chunk("very relevant text here", "doc.pdf", 0)]
            result = reranker.rerank("relevant query", chunks, top_k=1)
            assert len(result) == 1
        finally:
            settings.citation_score_threshold = original_threshold

    @patch("src.reranker.CrossEncoder")
    def test_empty_chunks_returns_empty(self, MockCE):
        """Re-ranking an empty list should return empty without errors."""
        MockCE.return_value = MagicMock()
        reranker = CrossEncoderReranker()
        result = reranker.rerank("query", chunks=[], top_k=3)
        assert result == []

    @patch("src.reranker.CrossEncoder")
    def test_to_retrieved_chunks_conversion(self, MockCE):
        """
        to_retrieved_chunks should use the normalized re-rank score
        as the display score for the user.
        """
        MockCE.return_value = MagicMock()
        reranker = CrossEncoderReranker()

        ranked = [
            RankedChunk(
                text="Some content",
                source="doc.pdf",
                chunk_index=0,
                retrieval_score=0.7,
                rerank_score=3.5,
                rerank_score_normalized=_sigmoid(3.5),
                metadata={"token_count": 15},
            )
        ]
        converted = reranker.to_retrieved_chunks(ranked)

        assert len(converted) == 1
        assert converted[0].source == "doc.pdf"
        # Score should be the normalized re-rank score
        assert abs(converted[0].score - _sigmoid(3.5)) < 1e-4
        # Original rerank_score preserved in metadata
        assert "rerank_score" in converted[0].metadata
