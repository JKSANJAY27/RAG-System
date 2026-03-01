"""
src/vector_store.py — ChromaDB Vector Store Wrapper

WHAT IS A VECTOR STORE?
    A vector store is a specialized database that stores embeddings (vectors)
    and allows you to find the most similar ones to a query vector very quickly.

    Traditional DB query:  "WHERE content CONTAINS 'machine learning'"
    Vector store query:    "Find chunks whose MEANING is closest to 'machine learning'"

    The second query finds chunks about "deep learning", "neural networks",
    "AI models" — even if they never use the exact phrase "machine learning".
    This is why vector search is so powerful for RAG.

WHY CHROMADB?
    - Runs 100% locally (no API, no cloud)
    - Persists data to disk (your index survives restarts)
    - Simple Python API
    - Free and open source
    - Production-grade (used by many companies)

PERSISTENCE:
    When you configure CHROMA_PERSIST_DIR=./chroma_db in .env, ChromaDB
    saves all your embeddings to that folder. Next time you start the app,
    they're still there — no need to re-ingest!
"""

import uuid
from typing import List

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.chunker import Chunk


# ─── Data Model ───────────────────────────────────────────────────────────────

from dataclasses import dataclass


@dataclass
class RetrievedChunk:
    """A chunk that was returned by a similarity search, with its relevance score."""
    text: str
    source: str
    chunk_index: int
    score: float       # Similarity score (0-1, higher = more relevant)
    metadata: dict


# ─── Vector Store ─────────────────────────────────────────────────────────────

class VectorStore:
    """
    Wraps ChromaDB to provide document storage and semantic search.

    ARCHITECTURE:
        ChromaDB organizes data into "collections" (like tables in SQL).
        Each collection stores:
          - Documents (the chunk text)
          - Embeddings (the vectors for each chunk)
          - Metadata (source file, chunk index, etc.)
          - IDs (unique identifier for each chunk)
    """

    def __init__(self, persist_dir: str, collection_name: str):
        """
        Initialize ChromaDB with persistent storage.

        Args:
            persist_dir: Directory where ChromaDB stores its data on disk.
            collection_name: Name for the collection (like a table name).
        """
        print(f"  ⟳ Connecting to ChromaDB at '{persist_dir}'...")

        # PersistentClient saves data to disk automatically
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),  # No phone-home
        )

        # get_or_create: if the collection exists, load it; if not, create it
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # Use cosine similarity (standard for text)
        )

        print(
            f"  ✓ ChromaDB ready. Collection '{collection_name}' "
            f"has {self._collection.count()} existing chunks."
        )

    def add_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        """
        Add a list of chunks and their embeddings to the vector store.

        IDs: We generate a deterministic ID from source + chunk_index so that
        re-ingesting the same document doesn't create duplicates (it upserts).

        Args:
            chunks: The text chunks from the chunker.
            embeddings: The corresponding embedding vectors from the embedder.
        """
        if not chunks:
            return

        ids = [
            f"{chunk.source}::chunk_{chunk.chunk_index}"
            for chunk in chunks
        ]
        documents = [chunk.text for chunk in chunks]
        metadatas = [
            {
                "source": chunk.source,
                "chunk_index": chunk.chunk_index,
                "token_count": chunk.metadata.get("token_count", 0),
                "doc_type": chunk.metadata.get("type", "unknown"),
            }
            for chunk in chunks
        ]

        # upsert = insert + update (idempotent — safe to run multiple times)
        self._collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        print(f"  ✓ Stored {len(chunks)} chunks in the vector store.")

    def query(
        self,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> List[RetrievedChunk]:
        """
        Find the top-k most semantically similar chunks to the query.

        HOW SIMILARITY WORKS:
            Cosine similarity measures the angle between two vectors.
            Score = 1.0 → identical meaning
            Score = 0.5 → moderately related
            Score = 0.0 → unrelated

        Args:
            query_embedding: The embedding vector for the user's question.
            top_k: How many chunks to return.

        Returns:
            List of RetrievedChunk objects sorted by relevance (most relevant first).
        """
        if self._collection.count() == 0:
            return []

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self._collection.count()),  # Can't retrieve more than we have
            include=["documents", "metadatas", "distances"],
        )

        retrieved = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # ChromaDB with cosine space returns distances (lower=closer).
            # Convert to a similarity score (higher=better) for intuitive display.
            similarity_score = 1 - dist

            retrieved.append(
                RetrievedChunk(
                    text=doc,
                    source=meta.get("source", "unknown"),
                    chunk_index=meta.get("chunk_index", 0),
                    score=round(similarity_score, 4),
                    metadata=meta,
                )
            )

        return retrieved

    def count(self) -> int:
        """Return the number of chunks currently in the store."""
        return self._collection.count()

    def delete_collection(self) -> None:
        """Delete all data (useful for testing or starting fresh)."""
        self._client.delete_collection(self._collection.name)
        print("  ⚠ Collection deleted.")
