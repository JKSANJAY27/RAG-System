#!/usr/bin/env python
"""
evals/run_evals.py — Golden Dataset Evaluation Runner (Phase 3)

WHAT IS THIS?
    This script runs your RAG system against every entry in the golden
    dataset and reports quality metrics. Think of it as a "test suite"
    for answer quality, not just code correctness.

WHY DOES THIS MATTER?
    Unit tests verify your code is correct.
    Evaluations verify your system is GOOD.

    Example: You change the prompt. pytest still passes (code is correct)
    but evaluation reveals token_f1 dropped 15% (quality regressed).
    Without this script, you'd ship that regression to production.

HOW TO RUN:
    # Run against the default golden dataset
    python evals/run_evals.py

    # Run against a custom dataset
    python evals/run_evals.py --dataset evals/golden_dataset.jsonl

    # Save results to a specific file
    python evals/run_evals.py --output evals/results/my_run.json

RESULTS FORMAT:
    Results are saved to evals/results/eval_YYYYMMDD_HHMMSS.json
    Each result contains per-question scores plus aggregate statistics.

REQUIREMENTS:
    - Documents must be ingested first: python ingest.py ...
    - Ollama must be running: ollama serve (and model pulled)
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

# Make sure repo root is on path (when running from any directory)
sys.path.insert(0, str(Path(__file__).parent.parent))

from evals.metrics import score_response
from src.rag_pipeline import RAGPipeline


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_dataset(path: str) -> list:
    """Load JSONL golden dataset. Each line is one test case."""
    cases = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    cases.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"  ⚠ Skipping malformed line {i}: {e}")
    return cases


def _bar(score: float, width: int = 20) -> str:
    """ASCII progress bar for a 0-1 score."""
    filled = int(score * width)
    return "█" * filled + "░" * (width - filled)


def _fmt(score) -> str:
    """Format a score for display."""
    if score is None:
        return " N/A "
    return f"{score:.3f}"


# ─── Main Evaluation Loop ─────────────────────────────────────────────────────

def run_evaluation(dataset_path: str, output_path: str = None) -> dict:
    """
    Run the full evaluation and return results dict.

    Args:
        dataset_path: Path to .jsonl golden dataset.
        output_path: Path to save JSON results (auto-generated if None).

    Returns:
        Dict with 'results' (per-question) and 'aggregate' (summary stats).
    """
    print("\n" + "═" * 65)
    print("  🧪 RAG EVALUATION RUNNER (Phase 3)")
    print("═" * 65)

    # ── Load dataset ──────────────────────────────────────────────────────────
    cases = load_dataset(dataset_path)
    print(f"\n📋 Loaded {len(cases)} test cases from '{dataset_path}'")

    # ── Initialize pipeline ───────────────────────────────────────────────────
    print("\n⚙  Initializing pipeline...")
    try:
        pipeline = RAGPipeline()
    except Exception as e:
        print(f"\n❌ Pipeline initialization failed: {e}")
        print("   Make sure Ollama is running: ollama serve")
        sys.exit(1)

    if pipeline.chunk_count == 0:
        print("\n❌ The vector store is empty — please ingest documents first:")
        print("   python ingest.py --source docs/transformer_architecture.md --type markdown")
        sys.exit(1)

    print(f"\n✅ Ready. {pipeline.chunk_count} chunks in knowledge base.\n")
    print("─" * 65)

    # ── Run each test case ────────────────────────────────────────────────────
    results = []

    for case in cases:
        qid = case.get("id", "?")
        question = case["question"]
        expected_keywords = case.get("expected_keywords", [])
        reference_answer = case.get("reference_answer", "")

        print(f"\n[{qid}] {question[:70]}")

        try:
            response = pipeline.query(question)
        except Exception as e:
            print(f"  ❌ Query failed: {e}")
            results.append({
                "id": qid,
                "question": question,
                "error": str(e),
                "contains_score": 0.0,
                "token_f1": 0.0,
                "citations_present": False,
                "faithfulness": 0.0,
                "declined": True,
            })
            continue

        chunk_texts = [c.text for c in response.retrieved_chunks]

        scores = score_response(
            answer=response.answer,
            sources=response.sources,
            chunk_texts=chunk_texts,
            expected_keywords=expected_keywords,
            reference_answer=reference_answer,
        )

        result = {
            "id": qid,
            "question": question,
            "answer": response.answer[:300] + "..." if len(response.answer) > 300 else response.answer,
            "sources": response.sources,
            "citation_enforced": response.citation_enforced,
            **scores,
        }
        results.append(result)

        # Pretty print score row
        status = "🔴 DECLINED" if scores["declined"] else "🟢 ANSWERED"
        print(f"  {status}")
        print(f"  contains_score : {_fmt(scores['contains_score'])} {_bar(scores['contains_score'])}")
        print(f"  token_f1       : {_fmt(scores['token_f1'])}       {_bar(scores['token_f1'] or 0)}")
        print(f"  citations      : {'✓' if scores['citations_present'] else '✗'}")
        print(f"  faithfulness   : {_fmt(scores['faithfulness'])}   {_bar(scores['faithfulness'])}")

    # ── Aggregate stats ───────────────────────────────────────────────────────
    answered = [r for r in results if not r.get("declined")]
    n_total = len(results)
    n_answered = len(answered)
    n_declined = n_total - n_answered

    def safe_mean(values):
        v = [x for x in values if x is not None]
        return round(sum(v) / len(v), 4) if v else 0.0

    aggregate = {
        "n_total": n_total,
        "n_answered": n_answered,
        "n_declined": n_declined,
        "answer_rate": round(n_answered / n_total, 4) if n_total else 0.0,
        "mean_contains_score": safe_mean([r.get("contains_score") for r in answered]),
        "mean_token_f1": safe_mean([r.get("token_f1") for r in answered]),
        "citation_rate": safe_mean([float(r.get("citations_present", False)) for r in answered]),
        "mean_faithfulness": safe_mean([r.get("faithfulness") for r in answered]),
    }

    # ── Summary report ────────────────────────────────────────────────────────
    print("\n" + "═" * 65)
    print("  📊 EVALUATION SUMMARY")
    print("═" * 65)
    print(f"  Answered   : {n_answered}/{n_total}  ({aggregate['answer_rate']:.0%})")
    print(f"  Declined   : {n_declined}/{n_total}")
    print(f"  Contains   : {aggregate['mean_contains_score']:.3f}  {_bar(aggregate['mean_contains_score'])}")
    print(f"  Token F1   : {aggregate['mean_token_f1']:.3f}      {_bar(aggregate['mean_token_f1'])}")
    print(f"  Citation   : {aggregate['citation_rate']:.3f}      {_bar(aggregate['citation_rate'])}")
    print(f"  Faithfulns : {aggregate['mean_faithfulness']:.3f}  {_bar(aggregate['mean_faithfulness'])}")
    print("═" * 65)

    # ── Save results ──────────────────────────────────────────────────────────
    if output_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("evals") / "results"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / f"eval_{ts}.json")

    full_results = {
        "timestamp": datetime.now().isoformat(),
        "dataset": dataset_path,
        "aggregate": aggregate,
        "results": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Results saved to: {output_path}")

    return full_results


# ─── CLI Entry Point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the RAG golden dataset evaluation.",
    )
    parser.add_argument(
        "--dataset",
        default="evals/golden_dataset.jsonl",
        help="Path to the JSONL golden dataset (default: evals/golden_dataset.jsonl)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save results JSON (default: evals/results/eval_<timestamp>.json)",
    )
    args = parser.parse_args()
    run_evaluation(args.dataset, args.output)
