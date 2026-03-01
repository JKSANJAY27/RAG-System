"""
src/retriever.py — Retrieval Module

WHAT DOES THE RETRIEVER DO?
    It's the "search engine" of your RAG system. Given a user's question,
    it:
      1. Converts the question into an embedding vector
      2. Queries the vector store to find the most similar chunks
      3. Returns those chunks with their scores for inspection

WHY IS RETRIEVAL THE HEART OF RAG?
    "Retrieval Augmented Generation" — the whole point is that the LLM
    generates its answer from RETRIEVED evidence, not from memory.
    Better retrieval = better answers. (Phase 2 will make this even better
    with hybrid search + re-ranking.)

WHAT IS "TOP-K"?
    K is how many chunks you retrieve. Typical values are 3-10.
    - Too low (K=1): might miss important context
    - Too high (K=20): floods the LLM with irrelevant text, increases costs
    - Sweet spot (K=5): usually enough context without noise
"""

from typing import List

from src.embedder import Embedder
from src.vector_store import RetrievedChunk, VectorStore
from config import settings


class Retriever:
    """
    Combines the Embedder and VectorStore to perform semantic search.

    This is a "facade" class — it hides the complexity of embeddings and
    vector math behind a simple .retrieve(question) interface.
    """

    def __init__(self, embedder: Embedder, vector_store: VectorStore):
        self._embedder = embedder
        self._vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = None,
    ) -> List[RetrievedChunk]:
        """
        Find the most relevant document chunks for a given query.

        Args:
            query: The user's question in plain English.
            top_k: How many chunks to return (uses settings.top_k if None).

        Returns:
            List of RetrievedChunk objects, sorted by relevance descending.

        Raises:
            ValueError: If the vector store is empty (nothing ingested yet).
        """
        k = top_k or settings.top_k

        if self._vector_store.count() == 0:
            raise ValueError(
                "The vector store is empty! Please ingest documents first:\n"
                "  python ingest.py --source <path_or_url> --type <pdf|markdown|web>"
            )

        print(f"\n  ⟳ Embedding query: '{query[:60]}...'")
        query_embedding = self._embedder.embed_single(query)

        print(f"  ⟳ Searching vector store for top-{k} chunks...")
        chunks = self._vector_store.query(query_embedding, top_k=k)

        print(f"  ✓ Retrieved {len(chunks)} chunks:")
        for i, chunk in enumerate(chunks, 1):
            print(
                f"    [{i}] score={chunk.score:.4f} | "
                f"source={chunk.source} | "
                f"chunk_idx={chunk.chunk_index}"
            )

        return chunks
