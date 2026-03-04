"""
tests/test_metrics_dashboard.py — Unit Tests for Phase 5 (Metrics Dashboard)

WHAT IS COVERED:
    - load_traces(): loads valid JSONL, handles missing file, skips malformed lines
    - percentile(): known mathematical results for P50/P95/P99
    - compute_metrics(): citation rate, declination rate, token aggregation,
                         stage latency extraction, date range detection

All tests are fully offline — no Ollama, no network required.
"""

import json
import tempfile
from pathlib import Path

import pytest

from metrics_dashboard import (
    compute_metrics,
    load_traces,
    percentile,
    safe_mean,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def _make_trace(
    question="What is attention?",
    total_latency_ms=3000.0,
    citation_enforced=False,
    sources=("docs/paper.md",),
    prompt_tokens=500,
    completion_tokens=150,
    total_tokens=650,
    bm25_ms=12.0,
    vector_ms=38.0,
    rrf_ms=2.0,
    rerank_ms=300.0,
    gen_ms=2500.0,
    timestamp_utc="2024-01-10T10:00:00Z",
) -> dict:
    """Create a realistic mock trace record matching Phase 4 JSONL format."""
    return {
        "trace_id": "abc-123",
        "question": question,
        "timestamp_utc": timestamp_utc,
        "total_latency_ms": total_latency_ms,
        "citation_enforced": citation_enforced,
        "sources": list(sources),
        "tokens": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
        "spans": {
            "bm25":       {"latency_ms": bm25_ms,   "input": {}, "output": {}},
            "vector":     {"latency_ms": vector_ms,  "input": {}, "output": {}},
            "rrf_fusion": {"latency_ms": rrf_ms,     "input": {}, "output": {}},
            "rerank":     {"latency_ms": rerank_ms,  "input": {}, "output": {}},
            "generation": {"latency_ms": gen_ms,     "input": {}, "output": {}},
        },
    }


# ─── load_traces ─────────────────────────────────────────────────────────────

class TestLoadTraces:
    def test_returns_empty_list_for_missing_file(self):
        result = load_traces("/path/that/does/not/exist.jsonl")
        assert result == []

    def test_loads_valid_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "traces.jsonl"
            p.write_text(
                json.dumps({"question": "q1"}) + "\n" +
                json.dumps({"question": "q2"}) + "\n",
                encoding="utf-8",
            )
            result = load_traces(str(p))
            assert len(result) == 2
            assert result[0]["question"] == "q1"

    def test_skips_empty_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "traces.jsonl"
            p.write_text(
                json.dumps({"question": "q1"}) + "\n\n" +
                json.dumps({"question": "q2"}) + "\n",
                encoding="utf-8",
            )
            result = load_traces(str(p))
            assert len(result) == 2

    def test_skips_malformed_lines_gracefully(self, capsys):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "traces.jsonl"
            p.write_text(
                json.dumps({"question": "good"}) + "\n"
                "THIS IS NOT JSON\n" +
                json.dumps({"question": "also good"}) + "\n",
                encoding="utf-8",
            )
            result = load_traces(str(p))
            assert len(result) == 2  # malformed line skipped

    def test_since_filter_excludes_old_traces(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "traces.jsonl"
            p.write_text(
                json.dumps({"question": "old", "timestamp_utc": "2024-01-01T00:00:00Z"}) + "\n" +
                json.dumps({"question": "new", "timestamp_utc": "2024-02-01T00:00:00Z"}) + "\n",
                encoding="utf-8",
            )
            result = load_traces(str(p), since="2024-01-15")
            assert len(result) == 1
            assert result[0]["question"] == "new"

    def test_since_filter_includes_traces_on_cutoff_date(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "traces.jsonl"
            p.write_text(
                json.dumps({"question": "exact", "timestamp_utc": "2024-01-15T00:00:00Z"}) + "\n",
                encoding="utf-8",
            )
            result = load_traces(str(p), since="2024-01-15")
            assert len(result) == 1


# ─── percentile ──────────────────────────────────────────────────────────────

class TestPercentile:
    def test_p50_of_sorted_list(self):
        values = list(range(1, 11))   # 1..10
        result = percentile(values, 50)
        # nearest-rank: idx = int(10 * 50 / 100) = 5 → values[5] = 6
        assert result == 6

    def test_p100_returns_max(self):
        values = [1.0, 2.0, 3.0, 10.0]
        assert percentile(values, 100) == 10.0

    def test_p0_returns_min_or_close(self):
        values = [5.0, 10.0, 15.0]
        result = percentile(values, 0)
        assert result == 5.0

    def test_empty_list_returns_zero(self):
        assert percentile([], 50) == 0.0

    def test_single_element(self):
        assert percentile([42.0], 50) == 42.0
        assert percentile([42.0], 95) == 42.0

    def test_p95_higher_than_p50(self):
        values = [100.0] * 95 + [10000.0] * 5
        assert percentile(values, 95) > percentile(values, 50)

    def test_unsorted_input_handled(self):
        values = [10, 1, 5, 3, 8]
        p50 = percentile(values, 50)
        assert p50 == percentile(sorted(values), 50)


# ─── compute_metrics ─────────────────────────────────────────────────────────

class TestComputeMetrics:
    def test_empty_traces_returns_empty_dict(self):
        assert compute_metrics([]) == {}

    def test_n_total_matches_input(self):
        traces = [_make_trace() for _ in range(5)]
        m = compute_metrics(traces)
        assert m["n_total"] == 5

    def test_citation_rate_all_answered(self):
        traces = [_make_trace(sources=("doc.md",)) for _ in range(4)]
        m = compute_metrics(traces)
        assert m["quality"]["citation_rate"] == 1.0

    def test_citation_rate_none_answered(self):
        traces = [_make_trace(sources=()) for _ in range(3)]
        m = compute_metrics(traces)
        assert m["quality"]["citation_rate"] == 0.0

    def test_citation_rate_partial(self):
        traces = (
            [_make_trace(sources=("doc.md",)) for _ in range(3)] +
            [_make_trace(sources=()) for _ in range(1)]
        )
        m = compute_metrics(traces)
        assert m["quality"]["citation_rate"] == pytest.approx(0.75)

    def test_declination_rate_calculation(self):
        traces = (
            [_make_trace(citation_enforced=True) for _ in range(2)] +
            [_make_trace(citation_enforced=False) for _ in range(8)]
        )
        m = compute_metrics(traces)
        assert m["quality"]["declination_rate"] == pytest.approx(0.2)

    def test_error_rate_zero_when_no_errors(self):
        traces = [_make_trace() for _ in range(5)]
        m = compute_metrics(traces)
        assert m["quality"]["error_rate"] == 0.0

    def test_token_totals(self):
        traces = [_make_trace(prompt_tokens=100, completion_tokens=50, total_tokens=150)
                  for _ in range(4)]
        m = compute_metrics(traces)
        assert m["tokens"]["total_consumed"] == 600
        assert m["tokens"]["mean_prompt"] == pytest.approx(100.0)
        assert m["tokens"]["mean_completion"] == pytest.approx(50.0)

    def test_latency_p50_single_trace(self):
        traces = [_make_trace(total_latency_ms=5000.0)]
        m = compute_metrics(traces)
        assert m["latency"]["p50_ms"] == 5000.0

    def test_stage_latency_extracted(self):
        traces = [_make_trace(bm25_ms=20.0, vector_ms=40.0,
                              rerank_ms=350.0, gen_ms=4000.0)]
        m = compute_metrics(traces)
        assert m["stage_latency_p50"]["bm25"]   == 20.0
        assert m["stage_latency_p50"]["vector"] == 40.0
        assert m["stage_latency_p50"]["generation"] == 4000.0

    def test_date_range_extracted(self):
        traces = [
            _make_trace(timestamp_utc="2024-01-05T10:00:00Z"),
            _make_trace(timestamp_utc="2024-01-20T10:00:00Z"),
        ]
        m = compute_metrics(traces)
        assert m["date_range"]["first"] == "2024-01-05"
        assert m["date_range"]["last"]  == "2024-01-20"

    def test_n_answered_excludes_declined_and_errors(self):
        traces = (
            [_make_trace(citation_enforced=False) for _ in range(7)] +
            [_make_trace(citation_enforced=True)  for _ in range(2)] +
            [{**_make_trace(), "error": "Connection failed"}  for _ in range(1)]
        )
        m = compute_metrics(traces)
        assert m["n_answered"] == 7
        assert m["n_declined"] == 2
        assert m["n_errors"]   == 1
