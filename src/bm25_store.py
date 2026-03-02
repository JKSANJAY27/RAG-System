"""
src/bm25_store.py — BM25 Keyword Search Index

WHAT IS BM25?
    BM25 (Best Match 25) is a probabilistic ranking algorithm used by search
    engines like Elasticsearch and Lucene. It is the standard for keyword-based
    document retrieval.

    Given a query, BM25 scores every document by counting how often each query
    word appears in that document — but with smart saturation and normalization:
      - Term Frequency (TF): how many times the word appears in the chunk
      - Inverse Document Frequency (IDF): rare words count more than common ones
      - Length normalization: shorter chunks aren't unfairly penalized

HOW IS BM25 DIFFERENT FROM VECTOR SEARCH?
    Vector search: finds chunks with similar MEANING (semantic)
      Query "automobile crash" → finds chunks about "car accident" ✓
    BM25 search: finds chunks with matching KEYWORDS (lexical)
      Query "HTTP 404 error code" → finds chunks containing exactly "404" ✓

    Neither is universally better. They catch different things. This is
    exactly why hybrid search (both combined) outperforms either alone.

IMPLEMENTATION:
    BM25 needs the full corpus in memory to score documents. We load all
    stored chunks from ChromaDB at startup and rebuild the index. After
    ingesting new documents, the index is refreshed automatically.
"""

import re
from typing import List, Tuple

from rank_bm25 import BM25Okapi

from src.vector_store import RetrievedChunk, VectorStore


def _tokenize(text: str) -> List[str]:
    """
    Simple but effective tokenizer for BM25.

    Steps:
        1. Lowercase (so "Transformer" and "transformer" match)
        2. Split on non-word characters (spaces, punctuation, hyphens)
        3. Filter out empty tokens

    We keep it simple here. Phase 2+ could use NLTK stemming or stopwords
    for even better results, but this already works very well.
    """
    text = text.lower()
    tokens = re.split(r"\W+", text)
    return [t for t in tokens if t]


class BM25Store:
    """
    Wraps rank-bm25's BM25Okapi to provide keyword-based chunk retrieval.

    LIFECYCLE:
        1. At startup: loads all chunks from ChromaDB, builds BM25 index
        2. After new ingestion: call .refresh() to rebuild the index
        3. At query time: .search(query, top_k) returns ranked chunks

    NOTE: The BM25 index is in memory only. It is rebuilt from ChromaDB
          on demand. For very large corpora (millions of chunks), you would
          want to persist the index, but this is sufficient for our use case.
    """

    def __init__(self, vector_store: VectorStore):
        self._vector_store = vector_store
        self._chunks: List[RetrievedChunk] = []
        self._bm25: BM25Okapi = None
        self._build_index()

    def _build_index(self) -> None:
        """
        Load all chunks from ChromaDB and build the BM25 index.

        TOKENIZATION: Each chunk text is tokenized into word list.
        BM25Okapi expects a list of tokenized documents (list of word lists).
        """
        self._chunks = self._vector_store.get_all_chunks()

        if not self._chunks:
            self._bm25 = None
            print("  ⓘ BM25: No documents yet — index will build after ingestion.")
            return

        tokenized_corpus = [_tokenize(chunk.text) for chunk in self._chunks]
        self._bm25 = BM25Okapi(tokenized_corpus)
        print(f"  ✓ BM25 index built from {len(self._chunks)} chunks.")

    def refresh(self) -> None:
        """
        Rebuild the BM25 index after new documents are ingested.

        Call this after every .ingest() so the BM25 index stays in sync
        with the ChromaDB vector store.
        """
        print("  ⟳ Refreshing BM25 index...")
        self._build_index()

    def search(self, query: str, top_k: int = 10) -> List[RetrievedChunk]:
        """
        Find the top-k chunks by BM25 keyword relevance.

        Args:
            query: The user's question or search phrase.
            top_k: How many chunks to return.

        Returns:
            List of RetrievedChunk with BM25 scores (higher = more relevant).
            Returns empty list if the index is empty.
        """
        if self._bm25 is None or not self._chunks:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        # BM25 scores all documents; higher = better keyword match
        scores = self._bm25.get_scores(query_tokens)

        # Pair each chunk with its BM25 score and sort descending
        scored_chunks: List[Tuple[float, RetrievedChunk]] = sorted(
            zip(scores, self._chunks),
            key=lambda x: x[0],
            reverse=True,
        )

        # Return the top-k, using the BM25 score as the "score" field
        results = []
        for score, chunk in scored_chunks[:top_k]:
            results.append(
                RetrievedChunk(
                    text=chunk.text,
                    source=chunk.source,
                    chunk_index=chunk.chunk_index,
                    score=round(float(score), 4),
                    metadata=chunk.metadata,
                )
            )

        return results

    @property
    def doc_count(self) -> int:
        """Number of chunks currently indexed."""
        return len(self._chunks)
