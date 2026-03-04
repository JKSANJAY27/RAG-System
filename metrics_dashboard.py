"""
metrics_dashboard.py — Quality Metrics CLI Dashboard (Phase 5)

WHAT IS THIS?
    A command-line tool that reads traces/traces.jsonl (written by Phase 4)
    and produces a rich quality report covering:
        - P50 / P95 / P99 total query latency
        - Median latency BROKEN DOWN BY STAGE (bm25, vector, rerank, generation)
        - Token usage (prompt + completion, estimated cost)
        - Citation coverage %  — are answers grounded in evidence?
        - Declination rate %   — how often does citation enforcement trigger?
        - Error rate %         — queries that raised exceptions

WHY P50/P95 INSTEAD OF AVERAGE?
    The AVERAGE latency hides your worst-case performance.

    Example: 9 queries take 2s, one takes 20s.
        Average = 3.8s  ← looks acceptable
        P95     = 20s   ← reveals the problem

    P50 (median) = what a typical user experiences
    P95          = what your slowest 1-in-20 users experience
    P99          = your absolute worst case (important for SLAs)

    If P95 is 3× higher than P50, you have a bimodal latency distribution
    — some queries are systematically slower. That's a signal to investigate
    (usually: the cross-encoder is slow on long chunks, or Ollama is cold).

HOW TO USE:
    # After running some queries (they generate traces automatically)
    python metrics_dashboard.py

    # Point at a specific trace file
    python metrics_dashboard.py --traces path/to/traces.jsonl

    # Export an HTML report for your portfolio
    python metrics_dashboard.py --format html

    # Filter to recent queries
    python metrics_dashboard.py --since 2024-01-15
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# ─── Cost Constants (GPT-4 equivalent, for educational illustration) ──────────
# Our system uses free local Ollama, so cost is $0.
# These constants let us show "what this would cost on a paid API" —
# a useful comparison when pitching the value of local models.
GPT4_PROMPT_COST_PER_1K  = 0.03   # USD / 1K prompt tokens
GPT4_COMPLETION_COST_PER_1K = 0.06 # USD / 1K completion tokens


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_traces(path: str, since: Optional[str] = None) -> List[dict]:
    """
    Load trace records from a JSONL file.

    Args:
        path: Path to traces.jsonl
        since: ISO date string (YYYY-MM-DD). If provided, only load traces
               from this date onwards.

    Returns:
        List of trace dicts (empty list if file doesn't exist).
    """
    p = Path(path)
    if not p.exists():
        return []

    traces = []
    with open(p, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                # Date filter
                if since:
                    ts = record.get("timestamp_utc", "")
                    if ts and ts[:10] < since:
                        continue
                traces.append(record)
            except json.JSONDecodeError:
                print(f"  ⚠ Skipping malformed line {line_num} in {path}")

    return traces


# ─── Statistical Helpers ──────────────────────────────────────────────────────

def percentile(values: List[float], p: float) -> float:
    """
    Compute the p-th percentile of a list of values. No numpy required.

    Args:
        values: List of numeric values.
        p: Percentile (0-100). E.g. 50 for median, 95 for P95.

    Returns:
        The p-th percentile value, or 0.0 for empty input.
    """
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    # Nearest-rank method
    rank = int(len(sorted_vals) * p / 100)
    rank = min(rank, len(sorted_vals) - 1)
    return sorted_vals[rank]


def safe_mean(values: List[float]) -> float:
    return round(sum(values) / len(values), 1) if values else 0.0


# ─── Metrics Computation ──────────────────────────────────────────────────────

def compute_metrics(traces: List[dict]) -> dict:
    """
    Compute all quality metrics from a list of trace records.

    This is the core aggregation function. Separate from printing so it
    can be unit-tested independently.

    Returns:
        Dict with keys: n_total, latency, stage_latency_p50,
                        tokens, quality, date_range
    """
    if not traces:
        return {}

    n_total = len(traces)

    # ── Latency ───────────────────────────────────────────────────────────────
    total_latencies = [t.get("total_latency_ms", 0.0) for t in traces]

    # ── Per-stage latency (from span tree) ────────────────────────────────────
    stage_names = ["bm25", "vector", "rrf_fusion", "rerank", "generation"]
    stage_latencies: Dict[str, List[float]] = {s: [] for s in stage_names}

    for trace in traces:
        spans = trace.get("spans", {})
        for stage in stage_names:
            if stage in spans:
                ms = spans[stage].get("latency_ms", 0.0)
                if ms > 0:
                    stage_latencies[stage].append(ms)

    # ── Token usage ───────────────────────────────────────────────────────────
    tokens_data = [t.get("tokens", {}) for t in traces]
    prompt_tokens   = [td.get("prompt_tokens", 0)     for td in tokens_data]
    completion_tokens = [td.get("completion_tokens", 0) for td in tokens_data]
    total_tokens    = [td.get("total_tokens", 0)      for td in tokens_data]

    total_prompt_k     = sum(prompt_tokens) / 1000
    total_completion_k = sum(completion_tokens) / 1000
    estimated_cost_usd = (
        total_prompt_k * GPT4_PROMPT_COST_PER_1K
        + total_completion_k * GPT4_COMPLETION_COST_PER_1K
    )

    # ── Quality ───────────────────────────────────────────────────────────────
    n_declined = sum(1 for t in traces if t.get("citation_enforced", False))
    n_with_sources = sum(1 for t in traces if t.get("sources"))
    n_errors = sum(1 for t in traces if t.get("error"))
    n_answered = n_total - n_declined - n_errors

    citation_rate   = n_with_sources / n_total if n_total else 0.0
    declination_rate = n_declined / n_total if n_total else 0.0
    error_rate       = n_errors / n_total if n_total else 0.0

    # ── Date range ────────────────────────────────────────────────────────────
    timestamps = sorted(t.get("timestamp_utc", "") for t in traces if "timestamp_utc" in t)
    date_range = {
        "first": timestamps[0][:10] if timestamps else "N/A",
        "last":  timestamps[-1][:10] if timestamps else "N/A",
    }

    return {
        "n_total":    n_total,
        "n_answered": n_answered,
        "n_declined": n_declined,
        "n_errors":   n_errors,
        "date_range": date_range,

        "latency": {
            "p50_ms": round(percentile(total_latencies, 50), 1),
            "p95_ms": round(percentile(total_latencies, 95), 1),
            "p99_ms": round(percentile(total_latencies, 99), 1),
            "mean_ms": round(safe_mean(total_latencies), 1),
        },

        "stage_latency_p50": {
            stage: round(percentile(stage_latencies[stage], 50), 1)
            for stage in stage_names
        },

        "tokens": {
            "mean_prompt":     round(safe_mean(prompt_tokens), 0),
            "mean_completion": round(safe_mean(completion_tokens), 0),
            "total_consumed":  sum(total_tokens),
            "estimated_cost_usd": round(estimated_cost_usd, 6),
        },

        "quality": {
            "citation_rate":    round(citation_rate, 4),
            "declination_rate": round(declination_rate, 4),
            "error_rate":       round(error_rate, 4),
        },
    }


# ─── Terminal Dashboard ───────────────────────────────────────────────────────

def _bar(value: float, max_val: float, width: int = 30) -> str:
    """ASCII progress bar: ████████░░░░░░░░"""
    if max_val <= 0:
        return "░" * width
    filled = min(int(value / max_val * width), width)
    return "█" * filled + "░" * (width - filled)


def _pct_bar(rate: float, width: int = 30) -> str:
    return _bar(rate, 1.0, width)


def print_dashboard(m: dict) -> None:
    """
    Print the full metrics dashboard to stdout.
    """
    W = 66  # terminal width

    def divider(char="═"):
        print(char * W)

    def section(title):
        print()
        print(f"  {title}")
        print("  " + "─" * (W - 4))

    divider()
    print(f"  📊 RAG METRICS DASHBOARD  (Phase 5)")
    print(f"  {m['n_total']} queries analyzed | "
          f"{m['date_range']['first']} → {m['date_range']['last']}")
    print(f"  Answered: {m['n_answered']}  |  "
          f"Declined: {m['n_declined']}  |  "
          f"Errors: {m['n_errors']}")
    divider()

    # ── Latency ───────────────────────────────────────────────────────────────
    lat = m["latency"]
    p99 = lat["p99_ms"] or 1
    section("⏱  TOTAL QUERY LATENCY")
    print(f"  {'Metric':<8}  {'Value':>10}   Bar (relative to P99)")
    for label, key in [("P50 ", "p50_ms"), ("P95 ", "p95_ms"), ("P99 ", "p99_ms"), ("Mean", "mean_ms")]:
        val = lat[key]
        print(f"  {label}    {val:>8,.0f}ms   {_bar(val, p99)}")

    # ── Stage Breakdown ───────────────────────────────────────────────────────
    stages = m["stage_latency_p50"]
    max_stage = max(stages.values()) if stages else 1
    section("⏱  STAGE LATENCY (P50 median)")
    stage_labels = {
        "bm25": "BM25 search  ", "vector": "Vector search",
        "rrf_fusion": "RRF fusion   ", "rerank": "Re-ranking   ",
        "generation": "LLM generation",
    }
    for key, label in stage_labels.items():
        val = stages.get(key, 0.0)
        note = "← often slowest" if key == "generation" else ""
        print(f"  {label}: {val:>8,.0f}ms   {_bar(val, max_stage)} {note}")

    # ── Token Usage ───────────────────────────────────────────────────────────
    tok = m["tokens"]
    section("📦  TOKEN USAGE")
    print(f"  Mean prompt tokens     : {tok['mean_prompt']:>8,.0f}")
    print(f"  Mean completion tokens : {tok['mean_completion']:>8,.0f}")
    print(f"  Total tokens consumed  : {tok['total_consumed']:>8,}")
    print(f"  Estimated cost (GPT-4) : ${tok['estimated_cost_usd']:>8.4f}  "
          f"← actual cost with local Ollama: $0.00")

    # ── Quality ───────────────────────────────────────────────────────────────
    q = m["quality"]
    section("✅  QUALITY METRICS")
    def q_row(label, rate):
        pct = f"{rate * 100:.1f}%"
        return f"  {label:<25}: {pct:>6}   {_pct_bar(rate)}"

    print(q_row("Citation coverage",  q["citation_rate"]))
    print(q_row("Declination rate",   q["declination_rate"]))
    print(q_row("Error rate",         q["error_rate"]))

    # ── SRE Interpretation ────────────────────────────────────────────────────
    print()
    divider("─")
    _print_alerts(lat, q)
    divider()
    print()


def _print_alerts(lat: dict, q: dict) -> None:
    """
    Print SRE-style alerts if metrics cross thresholds.
    This is what a Site Reliability Engineer would look at first.
    """
    alerts = []

    p95 = lat.get("p95_ms", 0)
    p50 = lat.get("p50_ms", 1)

    if p95 > 10000:
        alerts.append(f"  ⚠  P95 latency is {p95:.0f}ms. Users in the 95th percentile wait >10s.")
    if p50 > 0 and p95 / p50 > 3:
        alerts.append(f"  ⚠  P95 is {p95/p50:.1f}× P50 — bimodal latency. Some queries are much slower.")
    if q["citation_rate"] < 0.7:
        alerts.append(f"  ⚠  Citation coverage {q['citation_rate']:.0%} < 70%. Answers may be ungrounded.")
    if q["declination_rate"] > 0.3:
        alerts.append(f"  ⚠  Declination rate {q['declination_rate']:.0%} > 30%. "
                      f"Citation threshold may be too strict.")
    if q["error_rate"] > 0:
        alerts.append(f"  🔴 Error rate {q['error_rate']:.0%} > 0%. Check Ollama connectivity.")

    if alerts:
        print("  📍 SRE ALERTS")
        for a in alerts:
            print(a)
    else:
        print("  ✅  All metrics within acceptable thresholds.")


# ─── HTML Report ──────────────────────────────────────────────────────────────

def export_html(m: dict, output_path: str) -> None:
    """
    Export a clean HTML quality report. Useful for portfolio / team sharing.
    """
    lat = m["latency"]
    tok = m["tokens"]
    q   = m["quality"]
    stages = m["stage_latency_p50"]

    def pct(rate: float) -> str:
        return f"{rate * 100:.1f}%"

    def html_bar(rate: float, color: str = "#4ade80") -> str:
        w = min(int(rate * 100), 100)
        return (f'<div style="background:#1e293b;border-radius:4px;height:12px;width:200px;display:inline-block">'
                f'<div style="background:{color};width:{w}%;height:100%;border-radius:4px"></div></div>')

    rows_latency = "".join(
        f"<tr><td>{label}</td><td>{lat[key]:,.0f}ms</td></tr>"
        for label, key in [("P50 (median)", "p50_ms"), ("P95", "p95_ms"),
                            ("P99", "p99_ms"), ("Mean", "mean_ms")]
    )
    max_stage = max(stages.values()) if stages else 1
    rows_stages = "".join(
        f"<tr><td>{name}</td><td>{stages.get(k,0):,.0f}ms</td></tr>"
        for k, name in [("bm25","BM25"), ("vector","Vector"),
                        ("rrf_fusion","RRF Fusion"), ("rerank","Re-rank"),
                        ("generation","LLM Generation")]
    )
    quality_rows = "".join([
        f"<tr><td>Citation coverage</td><td>{pct(q['citation_rate'])}</td>"
        f"<td>{html_bar(q['citation_rate'])}</td></tr>",
        f"<tr><td>Declination rate</td><td>{pct(q['declination_rate'])}</td>"
        f"<td>{html_bar(q['declination_rate'], '#f59e0b')}</td></tr>",
        f"<tr><td>Error rate</td><td>{pct(q['error_rate'])}</td>"
        f"<td>{html_bar(q['error_rate'], '#ef4444')}</td></tr>",
    ])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RAG Metrics Report</title>
<style>
  body  {{ font-family: 'Segoe UI', sans-serif; background: #0f172a; color: #e2e8f0; margin: 0; padding: 2rem; }}
  h1   {{ color: #38bdf8; font-size: 1.5rem; }}
  h2   {{ color: #94a3b8; font-size: 1rem; margin-top: 2rem; border-bottom: 1px solid #334155; padding-bottom: .5rem; }}
  .stat-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 1rem; margin: 1rem 0; }}
  .stat {{ background: #1e293b; border-radius: 8px; padding: 1rem; }}
  .stat-val {{ font-size: 1.8rem; font-weight: 700; color: #38bdf8; }}
  .stat-lbl {{ font-size: .75rem; color: #64748b; margin-top: .25rem; }}
  table {{ border-collapse: collapse; width: 100%; max-width: 600px; }}
  td, th {{ text-align: left; padding: .5rem .75rem; border-bottom: 1px solid #1e293b; font-size: .9rem; }}
  th {{ color: #64748b; font-weight: 500; }}
</style>
</head>
<body>
<h1>📊 RAG System — Quality Metrics Report (Phase 5)</h1>
<p style="color:#64748b">{m["date_range"]["first"]} → {m["date_range"]["last"]} &nbsp;|&nbsp; {m["n_total"]} queries</p>

<div class="stat-grid">
  <div class="stat"><div class="stat-val">{lat["p50_ms"]:,.0f}ms</div><div class="stat-lbl">P50 Latency</div></div>
  <div class="stat"><div class="stat-val">{lat["p95_ms"]:,.0f}ms</div><div class="stat-lbl">P95 Latency</div></div>
  <div class="stat"><div class="stat-val">{pct(q["citation_rate"])}</div><div class="stat-lbl">Citation Coverage</div></div>
  <div class="stat"><div class="stat-val">{m["n_answered"]}</div><div class="stat-lbl">Queries Answered</div></div>
  <div class="stat"><div class="stat-val">{tok["total_consumed"]:,}</div><div class="stat-lbl">Total Tokens</div></div>
  <div class="stat"><div class="stat-val">${tok["estimated_cost_usd"]:.4f}</div><div class="stat-lbl">GPT-4 Equiv. Cost</div></div>
</div>

<h2>⏱ Latency Percentiles</h2>
<table><tr><th>Metric</th><th>Value</th></tr>{rows_latency}</table>

<h2>⏱ Stage Breakdown (P50)</h2>
<table><tr><th>Stage</th><th>P50 Latency</th></tr>{rows_stages}</table>

<h2>✅ Quality Metrics</h2>
<table><tr><th>Metric</th><th>Value</th><th>Bar</th></tr>{quality_rows}</table>

<p style="color:#334155;font-size:.8rem;margin-top:3rem">
  Generated by RAG Metrics Dashboard (Phase 5) · Local Ollama (actual cost: $0.00)
</p>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n  💾 HTML report saved to: {output_path}")
    print(f"     Open in browser: file://{Path(output_path).absolute()}")


# ─── CLI Entry Point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="RAG Metrics Dashboard — reads traces/traces.jsonl",
    )
    parser.add_argument(
        "--traces",
        default="traces/traces.jsonl",
        help="Path to traces JSONL (default: traces/traces.jsonl)",
    )
    parser.add_argument(
        "--since",
        default=None,
        metavar="YYYY-MM-DD",
        help="Only include traces from this date onwards",
    )
    parser.add_argument(
        "--format",
        choices=["terminal", "html"],
        default="terminal",
        help="Output format (default: terminal)",
    )
    parser.add_argument(
        "--output",
        default="metrics_report.html",
        help="HTML output file path (only used with --format html)",
    )
    args = parser.parse_args()

    traces = load_traces(args.traces, since=args.since)

    if not traces:
        print(f"\n  ⚠  No traces found in '{args.traces}'")
        print("     Run a query first:")
        print("       python ask.py --question \"What is self-attention?\"")
        print("     Then re-run this dashboard.\n")
        sys.exit(0)

    metrics = compute_metrics(traces)

    if args.format == "html":
        export_html(metrics, args.output)
    else:
        print_dashboard(metrics)
