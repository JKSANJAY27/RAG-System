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
    Phase 3: Added tracing (Langfuse / local JSONL) for observability

TWO MAIN OPERATIONS (unchanged API):
    1. .ingest(source, doc_type)  — Add documents (also refreshes BM25 index)
    2. .query(question)           — Hybrid search → re-rank → cite → answer
"""

import time
from dataclasses import dataclass
from typing import List, Optional

from src.bm25_store import BM25Store
from src.chunker import Chunker
from src.embedder import Embedder
from src.generator import GeneratedAnswer, Generator
from src.hybrid_retriever import HybridRetriever
from src.ingestor import Document, get_ingestor
from src.reranker import CrossEncoderReranker
from src.tracer import RAGTracer
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
    # Phase 3: Timing
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    total_latency_ms: float = 0.0


# ─── Pipeline ─────────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    The main entry point for the RAG system (Phase 3: Hybrid + Tracing).

    Usage:
        pipeline = RAGPipeline()

        # Ingest documents
        pipeline.ingest("docs/paper.pdf", "pdf")

        # Query with hybrid retrieval + re-ranking + tracing
        response = pipeline.query("What is the main contribution?")
        print(response.answer)
        print(f"  Took {response.total_latency_ms:.0f}ms")
    """

    def __init__(self):
        print("\n🚀 Initializing RAG Pipeline (Phase 3 — Hybrid + Tracing)...")
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
        self._tracer = RAGTracer()   # Phase 3: observability

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

        # Step 5: Refresh BM25 index
        self._bm25_store.refresh()

        print(f"\n✅ Done! {len(chunks)} chunks added. "
              f"Total: {self._vector_store.count()}")
        return len(chunks)

    # ── QUERY PATH ────────────────────────────────────────────────────────────

    def query(self, question: str, top_k: Optional[int] = None) -> RAGResponse:
        """
        Answer a question using the full hybrid RAG pipeline.

        Phase 3 addition: Starts a trace, records retrieval + generation
        latency, and flushes to Langfuse / local JSONL on completion.

        Steps:
            1. Hybrid retrieve: BM25 + vector → RRF → cross-encoder re-rank
            2. Citation enforcement: empty result → decline answer
            3. Generate: send re-ranked chunks to Ollama → cited answer
            4. Trace: record all telemetry and flush

        Args:
            question: Plain English question.
            top_k: Override for number of chunks after re-ranking.

        Returns:
            RAGResponse with answer, citations, and timing info.
        """
        print(f"\n❓ Question: '{question}'")
        print("-" * 55)

        # ── Phase 3: Start trace ───────────────────────────────────────────────
        trace = self._tracer.start(question)

        # ── Step 1: Hybrid retrieval ───────────────────────────────────────────
        t_retrieval_start = time.time()
        retrieved_chunks = self._hybrid_retriever.retrieve(question, top_k=top_k)
        retrieval_ms = (time.time() - t_retrieval_start) * 1000

        citation_enforced = len(retrieved_chunks) == 0

        # Log retrieval to trace
        top_score = retrieved_chunks[0].score if retrieved_chunks else 0.0
        trace.log_retrieval(
            bm25_count=self._bm25_store.doc_count,
            vector_count=self._vector_store.count(),
            fused_count=len(retrieved_chunks),
            final_count=len(retrieved_chunks),
            top_rerank_score=top_score,
            citation_enforced=citation_enforced,
        )

        # ── Step 2: Generate ───────────────────────────────────────────────────
        t_gen_start = time.time()
        generated: GeneratedAnswer = self._generator.generate(question, retrieved_chunks)
        generation_ms = (time.time() - t_gen_start) * 1000

        # Log generation to trace
        trace.log_generation(
            answer=generated.answer,
            sources=generated.sources,
            prompt_version=generated.prompt_version,
        )

        # ── Flush trace ────────────────────────────────────────────────────────
        self._tracer.finish(trace)

        total_ms = retrieval_ms + generation_ms
        print(f"\n  ⏱ Latency: retrieval={retrieval_ms:.0f}ms  "
              f"generation={generation_ms:.0f}ms  total={total_ms:.0f}ms")

        return RAGResponse(
            question=question,
            answer=generated.answer,
            sources=generated.sources,
            retrieved_chunks=generated.retrieved_chunks,
            model=generated.model,
            prompt_version=generated.prompt_version,
            citation_enforced=citation_enforced,
            retrieval_latency_ms=round(retrieval_ms, 1),
            generation_latency_ms=round(generation_ms, 1),
            total_latency_ms=round(total_ms, 1),
        )

    @property
    def chunk_count(self) -> int:
        return self._vector_store.count()
