"""
tests/test_tracer.py — Unit Tests for Phase 4 Tracing (Part 1)

WHAT IS COVERED:
    - TraceContext span recording and retrieval
    - SpanTimer measures elapsed time correctly
    - token_summary() extracts generation tokens
    - total_latency_ms() sums all spans
    - RAGTracer writes valid JSONL to local file
    - JSONL record contains all expected Phase 4 fields and span tree

NO EXTERNAL DEPENDENCIES:
    These tests do NOT require Langfuse or Ollama.
    RAGTracer is tested with the local JSONL backend only.
"""

import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from src.trace_context import SpanTimer, TraceContext


# ─── SpanTimer Tests ─────────────────────────────────────────────────────────

class TestSpanTimer:
    def test_measures_positive_elapsed_time(self):
        timer = SpanTimer()
        with timer:
            time.sleep(0.01)  # 10ms
        assert timer.elapsed_ms >= 5.0   # allow slack for slow CI runners

    def test_zero_block_has_near_zero_elapsed(self):
        timer = SpanTimer()
        with timer:
            pass
        assert timer.elapsed_ms >= 0.0
        assert timer.elapsed_ms < 100.0  # should be essentially instant

    def test_elapsed_available_after_exit(self):
        timer = SpanTimer()
        with timer:
            x = 1 + 1
        # Should be accessible outside the with-block
        assert isinstance(timer.elapsed_ms, float)


# ─── TraceContext Tests ───────────────────────────────────────────────────────

class TestTraceContext:
    def test_has_unique_trace_id(self):
        ctx1 = TraceContext("question one")
        ctx2 = TraceContext("question two")
        assert ctx1.trace_id != ctx2.trace_id

    def test_trace_id_is_uuid_format(self):
        import uuid
        ctx = TraceContext("test")
        # Should not raise:
        uuid.UUID(ctx.trace_id)

    def test_records_span(self):
        ctx = TraceContext("What is attention?")
        ctx.record("bm25", latency_ms=12.4,
                   input={"query": "attention", "top_k": 10},
                   output={"candidates": 10, "top_scores": [3.2, 2.8]})
        span = ctx.get_span("bm25")
        assert span is not None
        assert span["latency_ms"] == 12.4

    def test_span_input_preserved(self):
        ctx = TraceContext("Q")
        ctx.record("vector", 38.0,
                   input={"query": "Q", "top_k": 5},
                   output={"candidates": 5})
        assert ctx.get_span("vector")["input"]["top_k"] == 5

    def test_span_output_preserved(self):
        ctx = TraceContext("Q")
        ctx.record("rrf_fusion", 2.0, input={}, output={"fused_count": 12})
        assert ctx.get_span("rrf_fusion")["output"]["fused_count"] == 12

    def test_get_span_returns_none_if_not_recorded(self):
        ctx = TraceContext("Q")
        assert ctx.get_span("nonexistent") is None

    def test_all_spans_returns_copy(self):
        ctx = TraceContext("Q")
        ctx.record("bm25", 10.0, input={}, output={})
        ctx.record("vector", 20.0, input={}, output={})
        spans = ctx.all_spans()
        assert "bm25" in spans
        assert "vector" in spans
        assert len(spans) == 2

    def test_total_latency_sums_all_spans(self):
        ctx = TraceContext("Q")
        ctx.record("bm25",    12.5, input={}, output={})
        ctx.record("vector",  38.0, input={}, output={})
        ctx.record("rrf_fusion", 2.0, input={}, output={})
        assert ctx.total_latency_ms() == pytest.approx(52.5)

    def test_token_summary_extracts_from_generation_span(self):
        ctx = TraceContext("Q")
        ctx.record("generation", 6900.0,
                   input={"prompt_tokens": 847},
                   output={"completion_tokens": 203, "total_tokens": 1050,
                           "prompt_tokens": 847})
        tokens = ctx.token_summary()
        assert tokens["prompt_tokens"] == 847
        assert tokens["completion_tokens"] == 203
        assert tokens["total_tokens"] == 1050

    def test_token_summary_returns_zeros_if_no_generation_span(self):
        ctx = TraceContext("Q")
        tokens = ctx.token_summary()
        assert tokens == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def test_overwriting_same_span_name_replaces_it(self):
        ctx = TraceContext("Q")
        ctx.record("bm25", 10.0, input={}, output={"first": True})
        ctx.record("bm25", 15.0, input={}, output={"second": True})
        assert ctx.get_span("bm25")["output"]["second"] is True
        assert ctx.get_span("bm25").get("output", {}).get("first") is None


# ─── RAGTracer (Local JSONL) Tests ───────────────────────────────────────────

class TestRAGTracerLocal:
    """
    Test the tracer with the local JSONL backend only.
    We redirect the output to a temp file so tests don't pollute traces/.
    """

    def _make_ctx(self) -> TraceContext:
        """Build a fully-populated TraceContext for testing."""
        ctx = TraceContext("What is self-attention?")
        ctx.record("bm25", 12.4,
                   input={"query": "attention", "top_k": 10},
                   output={"candidates": 10, "top_scores": [3.2, 2.8, 1.4]})
        ctx.record("vector", 38.1,
                   input={"query": "attention", "top_k": 10},
                   output={"candidates": 10, "top_scores": [0.91, 0.85, 0.77]})
        ctx.record("rrf_fusion", 2.0,
                   input={"bm25_count": 10, "vector_count": 10},
                   output={"fused_count": 12, "top_rrf_scores": [0.032, 0.031]})
        ctx.record("rerank", 340.5,
                   input={"input_count": 12, "target_top_k": 3},
                   output={"kept_count": 3, "top_score": 0.93,
                           "citation_enforced": False,
                           "after_rerank": [{"source": "doc.md", "ce_score": 0.93}]})
        ctx.record("generation", 6900.0,
                   input={"prompt_preview": "You are...", "prompt_tokens": 847},
                   output={"answer_preview": "Self-attention...",
                           "completion_tokens": 203, "total_tokens": 1050,
                           "prompt_tokens": 847})
        return ctx

    def test_writes_jsonl_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "traces.jsonl"

            from src.tracer import RAGTracer
            # Patch the local path inside the tracer
            tracer = RAGTracer.__new__(RAGTracer)
            tracer._langfuse = None
            tracer._local_path = trace_path
            trace_path.parent.mkdir(parents=True, exist_ok=True)

            ctx = self._make_ctx()
            tracer.flush(ctx, total_latency_ms=7412.0,
                         citation_enforced=False, sources=["doc.md"])

            assert trace_path.exists()

    def test_jsonl_is_valid_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "traces.jsonl"

            from src.tracer import RAGTracer
            tracer = RAGTracer.__new__(RAGTracer)
            tracer._langfuse = None
            tracer._local_path = trace_path

            ctx = self._make_ctx()
            tracer.flush(ctx, total_latency_ms=7412.0,
                         citation_enforced=False, sources=["doc.md"])

            line = trace_path.read_text(encoding="utf-8").strip()
            record = json.loads(line)   # raises if invalid
            assert isinstance(record, dict)

    def test_record_has_required_top_level_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "traces.jsonl"

            from src.tracer import RAGTracer
            tracer = RAGTracer.__new__(RAGTracer)
            tracer._langfuse = None
            tracer._local_path = trace_path

            ctx = self._make_ctx()
            tracer.flush(ctx, total_latency_ms=7412.0,
                         citation_enforced=False, sources=["doc.md"])

            record = json.loads(trace_path.read_text(encoding="utf-8").strip())

            for field in ["trace_id", "question", "timestamp_utc",
                          "total_latency_ms", "citation_enforced",
                          "sources", "tokens", "spans"]:
                assert field in record, f"Missing field: {field}"

    def test_record_contains_all_five_spans(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "traces.jsonl"

            from src.tracer import RAGTracer
            tracer = RAGTracer.__new__(RAGTracer)
            tracer._langfuse = None
            tracer._local_path = trace_path

            ctx = self._make_ctx()
            tracer.flush(ctx, total_latency_ms=7412.0,
                         citation_enforced=False, sources=["doc.md"])

            record = json.loads(trace_path.read_text(encoding="utf-8").strip())
            spans = record["spans"]

            for span_name in ["bm25", "vector", "rrf_fusion", "rerank", "generation"]:
                assert span_name in spans, f"Missing span: {span_name}"

    def test_token_counts_in_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "traces.jsonl"

            from src.tracer import RAGTracer
            tracer = RAGTracer.__new__(RAGTracer)
            tracer._langfuse = None
            tracer._local_path = trace_path

            ctx = self._make_ctx()
            tracer.flush(ctx, total_latency_ms=7412.0,
                         citation_enforced=False, sources=["doc.md"])

            record = json.loads(trace_path.read_text(encoding="utf-8").strip())
            assert record["tokens"]["prompt_tokens"] == 847
            assert record["tokens"]["completion_tokens"] == 203
            assert record["tokens"]["total_tokens"] == 1050

    def test_each_query_appends_new_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "traces.jsonl"

            from src.tracer import RAGTracer
            tracer = RAGTracer.__new__(RAGTracer)
            tracer._langfuse = None
            tracer._local_path = trace_path

            for _ in range(3):
                ctx = self._make_ctx()
                tracer.flush(ctx, total_latency_ms=1000.0,
                             citation_enforced=False, sources=[])

            lines = [l for l in trace_path.read_text(encoding="utf-8").splitlines() if l.strip()]
            assert len(lines) == 3
