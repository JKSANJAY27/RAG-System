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

    Phase 1: Retriever used vector search only (bi-encoder similarity)
    Phase 2: HybridRetriever uses BM25 + vector + RRF + cross-encoder re-ranking

    The rest of the pipeline (ingest, generate, cite) is the same.
    The orchestrator just swaps the retrieval stage — this is the power of
    good module design.

TWO MAIN OPERATIONS (unchanged API):
    1. .ingest(source, doc_type)  — Add documents (now also refreshes BM25 index)
    2. .query(question)           — Hybrid search → re-rank → cite → answer
"""

from dataclasses import dataclass
from typing import List, Optional

from src.bm25_store import BM25Store
from src.chunker import Chunker
from src.embedder import Embedder
from src.generator import GeneratedAnswer, Generator
from src.hybrid_retriever import HybridRetriever
from src.ingestor import Document, get_ingestor
from src.reranker import CrossEncoderReranker
from src.vector_store import RetrievedChunk, VectorStore
from config import settings


# ─── Response Model ───────────────────────────────────────────────────────────

@dataclass
class RAGResponse:
    """Everything your system produces for a single query."""
    question: str
    answer: str
    sources: List[str]                       # Paths/URLs of documents consulted
    retrieved_chunks: List[RetrievedChunk]   # Re-ranked chunks (for debugging)
    model: str
    prompt_version: str
    citation_enforced: bool                  # True if the system declined to answer


# ─── Pipeline ─────────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    The main entry point for the RAG system (Phase 2: Hybrid).

    Usage:
        pipeline = RAGPipeline()

        # Ingest documents
        pipeline.ingest("docs/paper.pdf", "pdf")

        # Query with hybrid retrieval + re-ranking
        response = pipeline.query("What is the main contribution?")
        print(response.answer)
        if response.citation_enforced:
            print("(System declined — no reliable evidence found)")
    """

    def __init__(self):
        print("\n🚀 Initializing RAG Pipeline (Phase 2 — Hybrid)...")
        print("=" * 65)

        self._chunker = Chunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        self._embedder = Embedder()
        self._vector_store = VectorStore(
            persist_dir=settings.chroma_persist_dir,
            collection_name=settings.chroma_collection_name,
        )
        self._bm25_store = BM25Store(self._vector_store)
        self._reranker = CrossEncoderReranker()
        self._hybrid_retriever = HybridRetriever(
            embedder=self._embedder,
            vector_store=self._vector_store,
            bm25_store=self._bm25_store,
            reranker=self._reranker,
        )
        self._generator = Generator()

        print("=" * 65)
        print(
            f"✅ Pipeline ready! "
            f"Chunks: {self._vector_store.count()} | "
            f"BM25 docs: {self._bm25_store.doc_count} | "
            f"Model: {settings.ollama_model}"
        )
        print()

    # ── INGEST PATH ───────────────────────────────────────────────────────────

    def ingest(self, source: str, doc_type: str) -> int:
        """
        Ingest a document into the knowledge base.

        Phase 2 addition: After storing chunks in ChromaDB, refreshes the
        BM25 index so keyword search stays in sync.

        Args:
            source: File path or URL.
            doc_type: 'pdf', 'markdown', 'md', 'web', 'url'.

        Returns:
            Number of chunks added.
        """
        print(f"\n📥 Ingesting: '{source}' (type={doc_type})")
        print("-" * 55)

        # Step 1: Load
        ingestor = get_ingestor(doc_type)
        document: Document = ingestor.ingest(source)
        print(f"  ✓ Loaded ({len(document.content):,} chars)")

        # Step 2: Chunk
        chunks = self._chunker.chunk(document)

        # Step 3: Embed
        print(f"  ⟳ Embedding {len(chunks)} chunks...")
        embeddings = self._embedder.embed([c.text for c in chunks])
        print(f"  ✓ {len(embeddings)} embeddings generated")

        # Step 4: Store in ChromaDB
        self._vector_store.add_chunks(chunks, embeddings)

        # Step 5 (Phase 2 NEW): Refresh BM25 index ─────────────────────────
        self._bm25_store.refresh()

        print(f"\n✅ Done! {len(chunks)} chunks added. "
              f"Total: {self._vector_store.count()}")
        return len(chunks)

    # ── QUERY PATH ────────────────────────────────────────────────────────────

    def query(self, question: str, top_k: Optional[int] = None) -> RAGResponse:
        """
        Answer a question using the full hybrid RAG pipeline.

        Steps:
            1. Hybrid retrieve: BM25 + vector → RRF → cross-encoder re-rank
            2. Citation enforcement: empty result → decline answer
            3. Generate: send re-ranked chunks to Ollama → cited answer

        Args:
            question: Plain English question.
            top_k: Override for number of chunks after re-ranking.

        Returns:
            RAGResponse (answer is a decline message if citation_enforced=True)
        """
        print(f"\n❓ Question: '{question}'")
        print("-" * 55)

        # Step 1: Hybrid retrieval (may return [] if citation enforcement fires)
        retrieved_chunks = self._hybrid_retriever.retrieve(question, top_k=top_k)

        # Step 2: Generate (passes [] if citation enforcement fired)
        generated: GeneratedAnswer = self._generator.generate(question, retrieved_chunks)

        # Detect if citation enforcement triggered (no chunks were kept)
        citation_enforced = len(retrieved_chunks) == 0

        return RAGResponse(
            question=question,
            answer=generated.answer,
            sources=generated.sources,
            retrieved_chunks=generated.retrieved_chunks,
            model=generated.model,
            prompt_version=generated.prompt_version,
            citation_enforced=citation_enforced,
        )

    @property
    def chunk_count(self) -> int:
        return self._vector_store.count()
