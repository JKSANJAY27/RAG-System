"""
src/trace_context.py — Lightweight Span Collector (Phase 4)

WHAT IS THIS FILE?
    A "trace context" is a small object that travels through the entire
    request chain — from the pipeline down into the retriever, and into
    the generator. Each sub-component records its own telemetry into it.

    At the end of the request, the tracer reads it all in one go and
    writes to Langfuse + local JSONL.

WHY THIS DESIGN (instead of global state)?
    We pass the context explicitly parameter-by-parameter. This means:
      1. Every component that accepts it is testable WITHOUT tracing
         (just pass None — the instrumentation is opt-in, zero overhead)
      2. No thread-safety issues — each request gets its own context object
      3. Easy to understand: the data flows where the code flows

SPAN NAMES (convention):
    "bm25"         — BM25 keyword retrieval
    "vector"       — sentence-transformer semantic retrieval
    "rrf_fusion"   — Reciprocal Rank Fusion merge step
    "rerank"       — cross-encoder re-ranking + citation enforcement
    "generation"   — LLM prompt → response (with token counts)

READING THE DATA:
    >>> ctx = TraceContext("What is attention?")
    >>> ctx.record("bm25", latency_ms=12.4, input={"top_k": 10},
    ...            output={"candidates": 10, "top_scores": [3.2, 2.8]})
    >>> ctx.get_span("bm25")
    {"latency_ms": 12.4, "input": {...}, "output": {...}}
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def _utc_now() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _ms() -> float:
    # perf_counter gives sub-millisecond precision on all platforms (including Windows)
    return time.perf_counter() * 1000


# ─── Timing Helper ────────────────────────────────────────────────────────────

class SpanTimer:
    """
    Context manager that measures a block of code's execution time.

    Usage:
        timer = SpanTimer()
        with timer:
            result = do_something()
        latency_ms = timer.elapsed_ms
    """
    def __init__(self):
        self._start: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self):
        self._start = _ms()
        return self

    def __exit__(self, *_):
        self.elapsed_ms = round(_ms() - self._start, 1)


# ─── Trace Context ────────────────────────────────────────────────────────────

class TraceContext:
    """
    A mutable container that accumulates telemetry for one RAG query.

    Lifecycle:
        1. Created by RAGPipeline at the start of .query()
        2. Passed down to HybridRetriever and Generator via optional params
        3. Each component calls .record() with its span data
        4. At the end of .query(), RAGTracer.flush(ctx) writes everything

    Key property: if trace_ctx is None, calling code skips tracing entirely.
    This means components stay fast and testable without a tracer.
    """

    def __init__(self, question: str):
        self.trace_id: str = str(uuid.uuid4())
        self.question: str = question
        self.timestamp_utc: str = _utc_now()
        self._spans: Dict[str, dict] = {}

    def record(
        self,
        name: str,
        latency_ms: float,
        input: dict,
        output: dict,
    ) -> None:
        """
        Record a completed span.

        Args:
            name: Span name (e.g. "bm25", "rerank", "generation").
            latency_ms: How long this stage took in milliseconds.
            input: What went IN to this step (query, parameters).
            output: What came OUT (candidates, scores, token counts, etc.).
        """
        self._spans[name] = {
            "latency_ms": round(latency_ms, 1),
            "input": _safe_dict(input),
            "output": _safe_dict(output),
        }

    def get_span(self, name: str) -> Optional[dict]:
        """Return a recorded span by name, or None if not recorded yet."""
        return self._spans.get(name)

    def all_spans(self) -> Dict[str, dict]:
        """Return a copy of all recorded spans."""
        return dict(self._spans)

    def total_latency_ms(self) -> float:
        """Sum of all recorded span latencies."""
        return round(sum(s["latency_ms"] for s in self._spans.values()), 1)

    def token_summary(self) -> dict:
        """Extract token counts from the generation span (if present)."""
        gen = self._spans.get("generation", {}).get("output", {})
        return {
            "prompt_tokens": gen.get("prompt_tokens", 0),
            "completion_tokens": gen.get("completion_tokens", 0),
            "total_tokens": gen.get("total_tokens", 0),
        }


# ─── Utility ──────────────────────────────────────────────────────────────────

def _safe_dict(d: dict) -> dict:
    """
    Make a dict JSON-serializable by converting non-serializable values to strings.
    Prevents tracing from crashing the pipeline due to type issues.
    """
    result = {}
    for k, v in d.items():
        try:
            import json
            json.dumps(v)
            result[k] = v
        except (TypeError, ValueError):
            result[k] = str(v)
    return result
