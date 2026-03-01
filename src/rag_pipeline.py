"""
src/rag_pipeline.py — Top-Level RAG Orchestrator

WHAT IS THIS FILE?
    This is the "conductor" of your RAG orchestra. It wires together all
    the individual modules:
        Ingestor → Chunker → Embedder → VectorStore
                              ↑                        (ingest path)
        Retriever ← Embedder ← VectorStore
            ↓
        Generator → GeneratedAnswer                    (query path)

    You only need to interact with RAGPipeline — it handles all the rest.

TWO MAIN OPERATIONS:
    1. .ingest(source, doc_type)  — Add a document to your knowledge base
    2. .query(question)           — Ask a question, get a cited answer

SINGLETON PATTERN:
    The heavy objects (Embedder, VectorStore, Generator) are created once
    when you instantiate RAGPipeline and reused for every subsequent call.
    This avoids reloading the 90MB embedding model on every query.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from src.chunker import Chunker
from src.embedder import Embedder
from src.generator import GeneratedAnswer, Generator
from src.ingestor import Document, get_ingestor
from src.retriever import Retriever
from src.vector_store import RetrievedChunk, VectorStore
from config import settings


# ─── Response Model ───────────────────────────────────────────────────────────

@dataclass
class RAGResponse:
    """Everything your system produces for a single query."""
    question: str
    answer: str
    sources: List[str]           # Paths/URLs of documents consulted
    retrieved_chunks: List[RetrievedChunk]  # Raw chunks (for debugging)
    model: str
    prompt_version: str


# ─── Pipeline ─────────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    The main entry point for the RAG system.

    Usage:
        pipeline = RAGPipeline()

        # Ingest a document
        pipeline.ingest("docs/ml_paper.pdf", "pdf")

        # Ask a question
        response = pipeline.query("What is the main contribution of this paper?")
        print(response.answer)
        print("Sources:", response.sources)
    """

    def __init__(self):
        print("\n🚀 Initializing RAG Pipeline...")
        print("=" * 60)

        # Initialize in dependency order
        self._chunker = Chunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        self._embedder = Embedder()
        self._vector_store = VectorStore(
            persist_dir=settings.chroma_persist_dir,
            collection_name=settings.chroma_collection_name,
        )
        self._retriever = Retriever(
            embedder=self._embedder,
            vector_store=self._vector_store,
        )
        self._generator = Generator()

        print("=" * 60)
        print(f"✅ RAG Pipeline ready! Vector store has {self._vector_store.count()} chunks.")
        print()

    # ── INGEST PATH ───────────────────────────────────────────────────────────

    def ingest(self, source: str, doc_type: str) -> int:
        """
        Ingest a document into the knowledge base.

        Steps:
            1. Load the document (ingestor)
            2. Split into chunks (chunker)
            3. Embed each chunk (embedder)
            4. Store chunks + embeddings (vector store)

        Args:
            source: File path or URL to the document.
            doc_type: One of 'pdf', 'markdown', 'md', 'web', 'url'.

        Returns:
            Number of chunks added to the vector store.
        """
        print(f"\n📥 Ingesting: '{source}' (type={doc_type})")
        print("-" * 50)

        # Step 1: Load
        ingestor = get_ingestor(doc_type)
        document: Document = ingestor.ingest(source)
        print(f"  ✓ Loaded document ({len(document.content):,} characters)")

        # Step 2: Chunk
        chunks = self._chunker.chunk(document)

        # Step 3: Embed
        print(f"  ⟳ Embedding {len(chunks)} chunks...")
        texts = [chunk.text for chunk in chunks]
        embeddings = self._embedder.embed(texts)
        print(f"  ✓ Generated {len(embeddings)} embeddings")

        # Step 4: Store
        self._vector_store.add_chunks(chunks, embeddings)

        print(f"\n✅ Ingestion complete! Added {len(chunks)} chunks.")
        print(f"   Total chunks in store: {self._vector_store.count()}")
        return len(chunks)

    # ── QUERY PATH ────────────────────────────────────────────────────────────

    def query(self, question: str, top_k: Optional[int] = None) -> RAGResponse:
        """
        Answer a question using the RAG pipeline.

        Steps:
            1. Retrieve relevant chunks (retriever)
            2. Generate a cited answer (generator)

        Args:
            question: The user's question in plain English.
            top_k: Override the default top-k setting from .env.

        Returns:
            RAGResponse with answer, sources, and raw chunks.
        """
        print(f"\n❓ Question: '{question}'")
        print("-" * 50)

        # Step 1: Retrieve
        retrieved_chunks = self._retriever.retrieve(question, top_k=top_k)

        # Step 2: Generate
        generated: GeneratedAnswer = self._generator.generate(question, retrieved_chunks)

        return RAGResponse(
            question=question,
            answer=generated.answer,
            sources=generated.sources,
            retrieved_chunks=generated.retrieved_chunks,
            model=generated.model,
            prompt_version=generated.prompt_version,
        )

    @property
    def chunk_count(self) -> int:
        """Number of chunks currently in the vector store."""
        return self._vector_store.count()
