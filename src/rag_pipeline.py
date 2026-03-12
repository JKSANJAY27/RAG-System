"""
src/rag_pipeline.py — Top-Level RAG Orchestrator (Phase 4: Full Tracing)

WHAT CHANGED FROM PHASE 3?
    Phase 3: Simple timer logging around retrieval + generation blocks.
    Phase 4: Creates a TraceContext at start of each query, passes it down
             through HybridRetriever and Generator so EACH sub-component
             records its own span, then calls tracer.flush(ctx) at the end.

    The result: one rich JSON record per query with the complete span tree:
        bm25 → vector → rrf_fusion → rerank → generation
    And a matching nested trace in Langfuse showing the visual timeline.

WHAT YOU SEE IN LANGFUSE AFTER PHASE 4:
    Traces tab → click any trace → span timeline:
        rag-query         (7.4s)
          retrieval       (512ms)
            bm25          (12ms)  ← candidates=10, top BM25 scores
            vector        (38ms)  ← candidates=10, top cosine scores
            rrf_fusion     (2ms)  ← fused=12, top RRF scores
            rerank        (340ms) ← kept=3, before/after comparison
          llm-generation  (6.9s)  ← 847 in / 203 out tokens
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
from src.trace_context import TraceContext
from src.tracer import RAGTracer
from src.vector_store import RetrievedChunk, VectorStore
from config import settings


# ─── Response Model ───────────────────────────────────────────────────────────

@dataclass
class RAGResponse:
    """Everything your system produces for a single query."""
    question: str
    answer: str
    sources: List[str]
    retrieved_chunks: List[RetrievedChunk]
    model: str
    prompt_version: str
    citation_enforced: bool
    # Phase 4: timing and token details
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    trace_id: str = ""
    # Exposed trace context for UI
    trace_ctx: Optional[TraceContext] = None



# ─── Pipeline ─────────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    The main entry point for the RAG system (Phase 4: Full span-based tracing).

    Every call to .query() now:
        1. Creates a TraceContext at the top
        2. Passes it into HybridRetriever.retrieve() and Generator.generate()
        3. Each sub-component records its own span (bm25, vector, rrf, rerank, gen)
        4. tracer.flush() writes the full span tree to Langfuse + local JSONL
    """

    def __init__(self):
        print("\n🚀 Initializing RAG Pipeline (Phase 4 — Full Tracing)...")
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
        self._tracer = RAGTracer()

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
        (Unchanged from Phase 3 — ingest is not traced per-span.)
        """
        print(f"\n📥 Ingesting: '{source}' (type={doc_type})")
        print("-" * 55)

        ingestor = get_ingestor(doc_type)
        document: Document = ingestor.ingest(source)
        print(f"  ✓ Loaded ({len(document.content):,} chars)")

        chunks = self._chunker.chunk(document)

        print(f"  ⟳ Embedding {len(chunks)} chunks...")
        embeddings = self._embedder.embed([c.text for c in chunks])
        print(f"  ✓ {len(embeddings)} embeddings generated")

        self._vector_store.add_chunks(chunks, embeddings)
        self._bm25_store.refresh()

        print(f"\n✅ Done! {len(chunks)} chunks added. "
              f"Total: {self._vector_store.count()}")
        return len(chunks)

    # ── QUERY PATH (fully traced) ─────────────────────────────────────────────

    def query(self, question: str, top_k: Optional[int] = None) -> RAGResponse:
        """
        Answer a question using the full hybrid RAG pipeline.

        Phase 4 upgrade:
            - Creates a TraceContext and passes it into every sub-component
            - Measures wall-clock time for retrieval and generation separately
            - Flushes the full span tree to Langfuse + local JSONL after generation

        Returned RAGResponse now includes trace_id (so you can look it up
        in Langfuse), and prompt/completion/total token counts.
        """
        print(f"\n❓ Question: '{question}'")
        print("-" * 55)

        # ── Create the trace context for this query ────────────────────────────
        # One TraceContext per query — lives only for the duration of this call
        ctx = TraceContext(question)

        # ── Step 1: Hybrid retrieval (BM25 + vector + RRF + rerank) ───────────
        t_retrieval_start = time.monotonic()
        retrieved_chunks = self._hybrid_retriever.retrieve(
            question,
            top_k=top_k,
            trace_ctx=ctx,          # ← all sub-spans recorded inside retrieve()
        )
        retrieval_ms = (time.monotonic() - t_retrieval_start) * 1000

        citation_enforced = len(retrieved_chunks) == 0

        # ── Step 2: Generate ───────────────────────────────────────────────────
        t_gen_start = time.monotonic()
        generated: GeneratedAnswer = self._generator.generate(
            question,
            retrieved_chunks,
            trace_ctx=ctx,          # ← "generation" span recorded inside generate()
        )
        generation_ms = (time.monotonic() - t_gen_start) * 1000
        total_ms = retrieval_ms + generation_ms

        # ── Step 3: Flush the complete span tree ───────────────────────────────
        self._tracer.flush(
            ctx,
            total_latency_ms=total_ms,
            citation_enforced=citation_enforced,
            sources=generated.sources,
        )

        print(f"\n  📊 Trace: {ctx.trace_id[:8]}... | "
              f"retrieval={retrieval_ms:.0f}ms | "
              f"generation={generation_ms:.0f}ms | "
              f"tokens={generated.total_tokens}")

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
            prompt_tokens=generated.prompt_tokens,
            completion_tokens=generated.completion_tokens,
            total_tokens=generated.total_tokens,
            trace_id=ctx.trace_id,
            trace_ctx=ctx,
        )

    @property
    def chunk_count(self) -> int:
        return self._vector_store.count()
