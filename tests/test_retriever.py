"""
tests/test_retriever.py — Integration Tests for the Retriever

WHAT DO THESE TESTS VERIFY?
    1. Retrieving from an empty store raises a clear error
    2. After adding a known document, querying for its content returns it
    3. The retrieved chunk includes a non-zero similarity score
    4. The source path is preserved in retrieved results

NOTE: These are INTEGRATION tests — they actually use ChromaDB and the
      real embedding model (all-MiniLM-L6-v2). They're slower than unit
      tests but verify the full retrieval stack works end-to-end.

HOW TO RUN:
    pytest tests/test_retriever.py -v
"""

import pytest

from src.chunker import Chunk
from src.embedder import Embedder
from src.retriever import Retriever
from src.vector_store import VectorStore


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def embedder():
    """Load the embedding model once for all tests in this module."""
    return Embedder()


@pytest.fixture
def temp_vector_store(tmp_path):
    """
    A fresh in-memory-like ChromaDB, stored in a temp directory per test.
    tmp_path is a pytest built-in that gives a unique temp directory.
    Using a unique collection name ensures no test bleeds into another.
    """
    return VectorStore(
        persist_dir=str(tmp_path / "chroma"),
        collection_name="test_collection",
    )


@pytest.fixture
def retriever(embedder, temp_vector_store):
    """A Retriever wired to a fresh, empty vector store."""
    return Retriever(embedder=embedder, vector_store=temp_vector_store)


@pytest.fixture
def populated_store(embedder, temp_vector_store):
    """A vector store pre-populated with a known document."""
    known_text = (
        "The transformer architecture introduced the self-attention mechanism "
        "which allows the model to weigh the importance of different words "
        "in a sequence when computing a representation for any position."
    )
    chunks = [
        Chunk(
            text=known_text,
            source="attention_paper.pdf",
            chunk_index=0,
            metadata={"type": "pdf", "token_count": 50},
        )
    ]
    embeddings = embedder.embed([known_text])
    temp_vector_store.add_chunks(chunks, embeddings)
    return temp_vector_store


# ─── Tests ────────────────────────────────────────────────────────────────────

class TestRetriever:
    def test_empty_store_raises_value_error(self, retriever):
        """Querying an empty vector store should raise a helpful ValueError."""
        with pytest.raises(ValueError, match="empty"):
            retriever.retrieve("What is attention?")

    def test_retrieves_known_content(self, embedder, populated_store):
        """After adding a known chunk, querying for its content should return it."""
        retriever = Retriever(embedder=embedder, vector_store=populated_store)
        results = retriever.retrieve("What is the self-attention mechanism?", top_k=1)

        assert len(results) == 1
        assert "self-attention" in results[0].text.lower()

    def test_retrieved_chunk_has_positive_score(self, embedder, populated_store):
        """The similarity score of the top result should be > 0."""
        retriever = Retriever(embedder=embedder, vector_store=populated_store)
        results = retriever.retrieve("transformer self-attention", top_k=1)

        assert results[0].score > 0.0

    def test_source_is_preserved_in_results(self, embedder, populated_store):
        """The source file path should be preserved through ingestion→retrieval."""
        retriever = Retriever(embedder=embedder, vector_store=populated_store)
        results = retriever.retrieve("self-attention mechanism", top_k=1)

        assert results[0].source == "attention_paper.pdf"

    def test_top_k_limits_results(self, embedder, tmp_path):
        """top_k should limit the number of returned chunks."""
        store = VectorStore(str(tmp_path / "chroma_topk"), "topk_test")
        # Add 5 distinct chunks
        texts = [f"Unique content number {i} about topic {i}" for i in range(5)]
        chunks = [
            Chunk(text=t, source="test.pdf", chunk_index=i, metadata={"token_count": 10})
            for i, t in enumerate(texts)
        ]
        embeddings = embedder.embed(texts)
        store.add_chunks(chunks, embeddings)

        r = Retriever(embedder=embedder, vector_store=store)
        results = r.retrieve("unique content", top_k=3)
        assert len(results) == 3
