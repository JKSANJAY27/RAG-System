"""
tests/test_chunker.py — Unit Tests for the Chunker Module

WHAT DO THESE TESTS VERIFY?
    1. Chunks are within the configured token limit
    2. Overlap exists between consecutive chunks (shared tokens)
    3. Source metadata is inherited from the parent document
    4. Empty documents raise an appropriate error

HOW TO RUN:
    pytest tests/test_chunker.py -v
"""

import pytest

from src.chunker import Chunk, Chunker, _count_tokens
from src.ingestor import Document


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_document():
    """A Document long enough to produce multiple chunks."""
    # Repeat a paragraph many times to force splitting
    paragraph = (
        "Retrieval Augmented Generation (RAG) is a technique in natural language "
        "processing that combines retrieval-based methods with generative language "
        "models. The system first retrieves relevant documents from a knowledge base "
        "and then uses a language model to generate a coherent answer grounded in "
        "those retrieved documents. This approach significantly reduces hallucination "
        "and improves factual accuracy in large language model outputs. "
    )
    long_content = paragraph * 30  # ~30 * ~100 tokens = ~3000 tokens total
    return Document(
        content=long_content,
        source="test_doc.md",
        metadata={"type": "markdown"},
    )


@pytest.fixture
def chunker():
    """A Chunker with small settings so tests run fast."""
    return Chunker(chunk_size=200, chunk_overlap=40)


# ─── Tests ────────────────────────────────────────────────────────────────────

class TestChunker:
    def test_produces_multiple_chunks(self, chunker, sample_document):
        """A long document should be split into multiple chunks."""
        chunks = chunker.chunk(sample_document)
        assert len(chunks) > 1, "Expected more than 1 chunk for a long document"

    def test_chunk_token_count_within_limit(self, chunker, sample_document):
        """No chunk should exceed the configured chunk_size in tokens."""
        chunks = chunker.chunk(sample_document)
        for chunk in chunks:
            token_count = _count_tokens(chunk.text)
            # Allow 10% tolerance (LangChain's splitter is approximate)
            assert token_count <= chunker.chunk_size * 1.10, (
                f"Chunk {chunk.chunk_index} has {token_count} tokens, "
                f"exceeding limit of {chunker.chunk_size}"
            )

    def test_source_metadata_preserved(self, chunker, sample_document):
        """Every chunk must inherit the source from the parent document."""
        chunks = chunker.chunk(sample_document)
        for chunk in chunks:
            assert chunk.source == "test_doc.md", (
                f"Chunk {chunk.chunk_index} lost its source metadata!"
            )

    def test_chunk_indices_are_sequential(self, chunker, sample_document):
        """Chunks should have sequential indices starting at 0."""
        chunks = chunker.chunk(sample_document)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_overlap_exists_between_consecutive_chunks(self, chunker, sample_document):
        """
        Consecutive chunks should share text at their boundaries.
        We verify overlap by checking that the end of chunk N appears
        in the beginning of chunk N+1.
        """
        chunks = chunker.chunk(sample_document)
        if len(chunks) < 2:
            pytest.skip("Need at least 2 chunks to test overlap")

        # Check first two chunks
        chunk_a = chunks[0].text
        chunk_b = chunks[1].text

        # The last 20 characters of chunk A should appear in chunk B
        chunk_a_tail = chunk_a[-50:].strip()
        assert chunk_a_tail in chunk_b, (
            "No overlap detected between chunk 0 and chunk 1. "
            "Check your chunk_overlap setting."
        )

    def test_empty_document_raises_error(self, chunker):
        """An empty document should raise a ValueError, not silently produce 0 chunks."""
        empty_doc = Document(content="   ", source="empty.txt", metadata={})
        with pytest.raises(ValueError, match="no content"):
            chunker.chunk(empty_doc)

    def test_metadata_token_count_stored(self, chunker, sample_document):
        """Each chunk's metadata should store its token_count for observability."""
        chunks = chunker.chunk(sample_document)
        for chunk in chunks:
            assert "token_count" in chunk.metadata
            assert chunk.metadata["token_count"] > 0
