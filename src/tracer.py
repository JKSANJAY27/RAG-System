"""
src/tracer.py — Observability Backend
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
    
WHAT CHANGED FROM PHASE 3?
    Phase 3: One log record per query (total latency, chunk count, top score)
    Phase 4: A full span tree per query — every sub-step is individually timed
             and logged, with inputs/outputs at each stage.

BACKENDS:
    1. Langfuse — visual dashboard with nested span timeline, token usage,
                  latency histograms. Keys detected from .env automatically.
    2. Local JSONL — always-on fallback, written to traces/traces.jsonl.
                     Readable with any JSON tool, or the Phase 5 dashboard.

HOW TO READ THE LANGFUSE DASHBOARD:
    Go to cloud.langfuse.com → "Traces" tab.
    Each query appears as one row. Click it to expand the span timeline:
      ┌─────────────────────────────────────────────────────────┐
      │ rag-query                               7412ms          │
      │   retrieval                             512ms           │
      │     bm25                     12ms                       │
      │     vector                   38ms                       │
      │     rrf_fusion                2ms                       │
      │     rerank                  340ms                       │
      │   generation                           6900ms           │
      │     llm-call (847 in / 203 out tokens)                  │
      └─────────────────────────────────────────────────────────┘
    This is the "glass box" — you can see exactly where time is spent.

LOCAL JSONL FORMAT:
    Each line is one query. The span tree is nested under "spans":
    {
      "trace_id": "abc-123",
      "question": "What is self-attention?",
      "timestamp_utc": "2024-01-01T12:00:00Z",
      "total_latency_ms": 7412,
      "citation_enforced": false,
      "sources": ["docs/transformer_architecture.md"],
      "tokens": {"prompt_tokens": 847, "completion_tokens": 203, "total_tokens": 1050},
      "spans": {
        "bm25":       {"latency_ms": 12,   "input": {...}, "output": {...}},
        "vector":     {"latency_ms": 38,   "input": {...}, "output": {...}},
        "rrf_fusion": {"latency_ms": 2,    "input": {...}, "output": {...}},
        "rerank":     {"latency_ms": 340,  "input": {...}, "output": {...}},
        "generation": {"latency_ms": 6900, "input": {...}, "output": {...}}
      }
    }
"""

import json
import os
from pathlib import Path
from typing import List, Optional

from src.trace_context import TraceContext
from config import settings

# ─── Optional Langfuse ────────────────────────────────────────────────────────
try:
    from langfuse import Langfuse
    _LANGFUSE_SDK_AVAILABLE = True
except ImportError:
    _LANGFUSE_SDK_AVAILABLE = False


# ─── RAG Tracer ───────────────────────────────────────────────────────────────

class RAGTracer:
    """
    Consumes a completed TraceContext and writes to all configured backends.

    Usage (in rag_pipeline.py):
        tracer = RAGTracer()                     # init once
        ctx = TraceContext(question)             # per-query
        # ... pipeline runs, ctx.record() called by each component ...
        tracer.flush(ctx, total_latency_ms=..., citation_enforced=..., sources=...)
    """

    def __init__(self):
        self._langfuse = _try_init_langfuse()
        self._local_path = Path("traces") / "traces.jsonl"
        self._local_path.parent.mkdir(exist_ok=True)

        if self._langfuse:
            print("  ✓ Tracer: Langfuse backend active → cloud.langfuse.com")
        else:
            print(f"  ✓ Tracer: Local backend → {self._local_path}")

    def flush(
        self,
        ctx: TraceContext,
        total_latency_ms: float,
        citation_enforced: bool,
        sources: List[str],
    ) -> None:
        """
        Write the completed trace to all backends.

        Args:
            ctx: The TraceContext accumulated during the query.
            total_latency_ms: Wall-clock time for the whole query.
            citation_enforced: True if the system declined to answer.
            sources: Document sources cited in the answer.
        """
        tokens = ctx.token_summary()
        spans = ctx.all_spans()

        record = {
            "trace_id": ctx.trace_id,
            "question": ctx.question,
            "timestamp_utc": ctx.timestamp_utc,
            "total_latency_ms": round(total_latency_ms, 1),
            "citation_enforced": citation_enforced,
            "sources": sources,
            "tokens": tokens,
            "model": settings.ollama_model,
            "prompt_version": _get_prompt_version(),
            "spans": spans,
        }

        # ── Local JSONL (always) ───────────────────────────────────────────────
        _write_local(self._local_path, record)

        # ── Langfuse (if configured) ───────────────────────────────────────────
        if self._langfuse:
            _send_to_langfuse(self._langfuse, ctx, record)


# ─── Local Writer ─────────────────────────────────────────────────────────────

def _write_local(path: Path, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ─── Langfuse Integration (Nested Spans) ──────────────────────────────────────

def _send_to_langfuse(langfuse, ctx: TraceContext, record: dict) -> None:
    """
    Send the full span tree to Langfuse using their nested span API.

    Langfuse shows these as a visual timeline:
        trace (top-level row)
          └── span: retrieval
          │     └── sub-spans: bm25, vector, rrf_fusion, rerank
          └── generation: llm-call (special node with token badge)
    """
    try:
        # ── Top-level trace ────────────────────────────────────────────────────
        trace = langfuse.trace(
            id=ctx.trace_id,
            name="rag-query",
            input={"question": ctx.question},
            output={
                "citation_enforced": record["citation_enforced"],
                "sources": record["sources"],
                "total_tokens": record["tokens"].get("total_tokens", 0),
            },
            metadata={
                "model": settings.ollama_model,
                "prompt_version": record["prompt_version"],
            },
        )

        # ── Retrieval parent span ──────────────────────────────────────────────
        retrieval_latency = sum(
            record["spans"].get(name, {}).get("latency_ms", 0)
            for name in ["bm25", "vector", "rrf_fusion", "rerank"]
        )
        retrieval_span = trace.span(
            name="retrieval",
            input={"query": ctx.question},
            output={
                "final_chunks": record["spans"].get("rerank", {})
                                               .get("output", {})
                                               .get("kept_count", 0),
                "citation_enforced": record["citation_enforced"],
            },
            metadata={"latency_ms": retrieval_latency},
        )

        # ── BM25 sub-span ──────────────────────────────────────────────────────
        _maybe_child_span(retrieval_span, "bm25", record["spans"])

        # ── Vector sub-span ────────────────────────────────────────────────────
        _maybe_child_span(retrieval_span, "vector", record["spans"])

        # ── RRF sub-span ───────────────────────────────────────────────────────
        _maybe_child_span(retrieval_span, "rrf_fusion", record["spans"])

        # ── Rerank sub-span ────────────────────────────────────────────────────
        _maybe_child_span(retrieval_span, "rerank", record["spans"])

        # ── Generation span (special Langfuse "generation" type) ──────────────
        # This shows as a dedicated "LLM call" node with input/output tokens
        gen = record["spans"].get("generation", {})
        if gen:
            gen_input = gen.get("input", {})
            gen_output = gen.get("output", {})
            trace.generation(
                name="llm-generation",
                model=settings.ollama_model,
                input=gen_input.get("prompt_preview", ""),
                output=gen_output.get("answer_preview", ""),
                usage={
                    "input": gen_input.get("prompt_tokens", 0),
                    "output": gen_output.get("completion_tokens", 0),
                    "total": gen_output.get("total_tokens", 0),
                },
                metadata={"latency_ms": gen.get("latency_ms", 0)},
            )

        langfuse.flush()

    except Exception as e:
        print(f"  ⚠ Langfuse upload failed: {e}")


def _maybe_child_span(parent_span, name: str, spans: dict) -> None:
    """Add a child span to parent_span if the named span was recorded."""
    data = spans.get(name)
    if data:
        parent_span.span(
            name=name,
            input=data.get("input", {}),
            output=data.get("output", {}),
            metadata={"latency_ms": data.get("latency_ms", 0)},
        )


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _try_init_langfuse():
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


def _get_prompt_version() -> str:
    """Read version from prompts.yaml without importing the whole generator."""
    try:
        import yaml
        p = Path(__file__).parent.parent / "prompts" / "prompts.yaml"
        return yaml.safe_load(p.read_text(encoding="utf-8")).get("version", "unknown")
    except Exception:
        return "unknown"
