"""
src/tracer.py — Observability & Tracing (Phase 3)

WHAT IS TRACING IN RAG SYSTEMS?
    Every query your system processes generates rich signal:
        - How long did retrieval take?
        - How many chunks were returned? Declined?
        - What was the top re-ranker score?
        - Which prompt version was used?
        - When did citation enforcement fire?

    Without tracing, you're flying blind. You can't answer:
        "Did last week's prompt change hurt quality?"
        "Are there queries that consistently fail to retrieve?"
        "How has latency changed since we added re-ranking?"

TWO BACKENDS (whichever you have configured):
    1. Langfuse (preferred): Open-source LLM observability platform.
       - Sign up free at https://cloud.langfuse.com OR self-host
       - Set LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY in .env
       - You get a beautiful dashboard with latency charts, trace explorer,
         cost tracking, and much more
    2. Local JSONL file (always available as fallback):
       - Written to traces/traces.jsonl
       - Machine-readable — can be analyzed with pandas or any JSON tool
       - No external service required

USAGE IN PIPELINE:
    tracer = RAGTracer()
    trace = tracer.start(question)
    trace.log_retrieval(chunks, latency_ms)
    trace.log_generation(answer, sources, latency_ms)
    tracer.finish(trace)
"""

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

from config import settings

# ─── Optional Langfuse import ─────────────────────────────────────────────────
# Langfuse is optional — the system works without it.
# Requires langfuse>=2.0.0,<3.0.0 (v2 API: .trace() / .span())
try:
    from langfuse import Langfuse
    _LANGFUSE_SDK_AVAILABLE = True
except ImportError:
    _LANGFUSE_SDK_AVAILABLE = False


# ─── Trace Data Model ─────────────────────────────────────────────────────────

@dataclass
class QueryTrace:
    """
    All telemetry for a single RAG query.

    Why log everything together?
        Having one record per query means you can join on trace_id,
        slice by prompt_version to compare before/after prompt changes,
        and filter by citation_enforced to find hard queries.
    """
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp_utc: str = ""

    # ── Input ──────────────────────────────────────────────────────────────────
    question: str = ""
    model: str = ""
    prompt_version: str = ""

    # ── Retrieval ─────────────────────────────────────────────────────────────
    bm25_candidates: int = 0
    vector_candidates: int = 0
    fused_candidates: int = 0
    final_chunk_count: int = 0        # After re-ranking
    top_rerank_score: float = 0.0     # Highest normalized cross-encoder score
    citation_enforced: bool = False   # True if system declined to answer

    # ── Output ────────────────────────────────────────────────────────────────
    answer_length_chars: int = 0
    sources: List[str] = field(default_factory=list)

    # ── Latency (ms) ──────────────────────────────────────────────────────────
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    total_latency_ms: float = 0.0


# ─── Active Trace Context ─────────────────────────────────────────────────────

class ActiveTrace:
    """
    Mutable trace context built up during a query.
    Call log_retrieval() then log_generation(), then RAGTracer.finish().
    """

    def __init__(self, question: str):
        self._data = QueryTrace(
            timestamp_utc=_now_iso(),
            question=question,
            model=settings.ollama_model,
        )
        self._start_ms = _now_ms()
        self._retrieval_end_ms: Optional[float] = None

    def log_retrieval(
        self,
        bm25_count: int,
        vector_count: int,
        fused_count: int,
        final_count: int,
        top_rerank_score: float,
        citation_enforced: bool,
    ) -> None:
        """Record retrieval stage telemetry."""
        now = _now_ms()
        self._data.bm25_candidates = bm25_count
        self._data.vector_candidates = vector_count
        self._data.fused_candidates = fused_count
        self._data.final_chunk_count = final_count
        self._data.top_rerank_score = top_rerank_score
        self._data.citation_enforced = citation_enforced
        self._data.retrieval_latency_ms = round(now - self._start_ms, 1)
        self._retrieval_end_ms = now

    def log_generation(
        self,
        answer: str,
        sources: List[str],
        prompt_version: str,
    ) -> None:
        """Record generation stage telemetry."""
        now = _now_ms()
        self._data.answer_length_chars = len(answer)
        self._data.sources = sources
        self._data.prompt_version = prompt_version
        gen_start = self._retrieval_end_ms or self._start_ms
        self._data.generation_latency_ms = round(now - gen_start, 1)
        self._data.total_latency_ms = round(now - self._start_ms, 1)

    @property
    def data(self) -> QueryTrace:
        return self._data


# ─── RAG Tracer ───────────────────────────────────────────────────────────────

class RAGTracer:
    """
    The main tracer. Call .start() at the top of a query, build up the trace,
    then call .finish() to persist it.

    Backends (auto-selected based on config):
        - Langfuse if LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY are set
        - Local JSONL file otherwise (traces/traces.jsonl)
    """

    def __init__(self):
        self._langfuse = _try_init_langfuse()
        self._local_path = Path("traces") / "traces.jsonl"
        self._local_path.parent.mkdir(exist_ok=True)

        if self._langfuse:
            print("  ✓ Tracer: Langfuse backend active")
        else:
            print(f"  ✓ Tracer: Local backend → {self._local_path}")

    def start(self, question: str) -> ActiveTrace:
        """Create and return a new trace context for a query."""
        return ActiveTrace(question)

    def finish(self, trace: ActiveTrace) -> None:
        """Persist a completed trace to all configured backends."""
        import threading
        data = trace.data

        # ── Local JSONL (always, synchronous — fast) ──────────────────────────
        _write_local(self._local_path, data)

        # ── Langfuse (async daemon thread — never blocks the pipeline) ────────
        if self._langfuse:
            t = threading.Thread(
                target=_send_to_langfuse,
                args=(self._langfuse, data),
                daemon=True,
            )
            t.start()


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _now_ms() -> float:
    """Current time in milliseconds."""
    return time.time() * 1000


def _now_iso() -> str:
    """Helper for ISO-8601 timestamps."""
    import datetime
    return datetime.datetime.now(datetime.UTC).isoformat().replace("+00:00", "Z")


def _write_local(path: Path, data: QueryTrace) -> None:
    """Append trace as a JSON line to the local trace file."""
    record = asdict(data)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _try_init_langfuse():
    """
    Create a Langfuse v2 client if:
        1. langfuse>=2,<3 is installed
        2. LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY are set in .env

    Returns None if either condition is not met (graceful degradation).
    """
    if not _LANGFUSE_SDK_AVAILABLE:
        return None

    pk = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    sk = os.getenv("LANGFUSE_SECRET_KEY", "")

    if not pk or not sk:
        return None

    try:
        return Langfuse(
            public_key=pk,
            secret_key=sk,
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
    except Exception as e:
        print(f"  ⚠ Langfuse init failed: {e}. Falling back to local tracing.")
        return None


def _send_to_langfuse(langfuse, data: QueryTrace) -> None:
    """
    Send a completed trace to Langfuse using the v2 SDK.

    Langfuse v2 trace structure:
        trace (top-level)
          └── span: hybrid-retrieval
          └── span: llm-generation
    """
    try:
        trace = langfuse.trace(
            id=data.trace_id,
            name="rag-query",
            input={"question": data.question},
            output={"answer_length_chars": data.answer_length_chars, "sources": data.sources},
            metadata={
                "model": data.model,
                "prompt_version": data.prompt_version,
                "citation_enforced": data.citation_enforced,
                "total_latency_ms": data.total_latency_ms,
            },
        )

        trace.span(
            name="hybrid-retrieval",
            input={"question": data.question},
            output={
                "bm25_candidates": data.bm25_candidates,
                "vector_candidates": data.vector_candidates,
                "fused_candidates": data.fused_candidates,
                "final_chunk_count": data.final_chunk_count,
                "top_rerank_score": data.top_rerank_score,
                "citation_enforced": data.citation_enforced,
            },
            metadata={"latency_ms": data.retrieval_latency_ms},
        )

        trace.span(
            name="llm-generation",
            input={"chunk_count": data.final_chunk_count},
            output={"answer_length_chars": data.answer_length_chars},
            metadata={"latency_ms": data.generation_latency_ms},
        )

        langfuse.flush()

    except Exception as e:
        print(f"  ⚠ Langfuse upload failed: {e}")

