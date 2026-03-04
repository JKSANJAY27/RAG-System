"""
tests/test_evaluation.py — Quality Gate Tests (Phase 3)

WHAT MAKES THESE DIFFERENT FROM UNIT TESTS?
    Unit tests verify code correctness (does the function behave as expected?).
    Evaluation tests verify system quality (are answers actually good?).

    These tests are marked with @pytest.mark.eval — they require a running
    Ollama server and an ingested document. They auto-skip if Ollama is down,
    so they don't break fast CI.

QUALITY THRESHOLDS (defined as constants):
    These encode our expectations as executable contracts.
    When the system degrades below a threshold, the test FAILS, alerting us
    BEFORE the regression ships to users.

    Thresholds are intentionally conservative — we want them to catch real
    problems, not produce noisy false alarms.

HOW TO RUN:
    # Run eval tests (requires Ollama + ingested doc)
    pytest tests/test_evaluation.py -v

    # Skip eval tests (fast CI — no Ollama needed)
    pytest tests/ -m "not eval" -v

PREREQUISITE:
    Before running these tests, ingest the sample document:
        python ingest.py --source docs/transformer_architecture.md --type markdown
"""

import json
import pytest

from evals.metrics import (
    citation_check,
    contains_check,
    faithfulness_score,
    score_response,
    token_f1,
)


# ─── Quality Gate Thresholds ──────────────────────────────────────────────────
# These constants are the "spec" for our RAG system's minimum acceptable quality.
# Treat them like SLAs — if the system falls below these, something broke.

MIN_ANSWER_RATE = 0.6         # At least 60% of golden questions should be answered
MIN_MEAN_CONTAINS = 0.5       # Average contains_score >= 0.5 across answered Qs
MIN_CITATION_RATE = 0.7       # At least 70% of answered questions cite a source
MIN_MEAN_FAITHFULNESS = 0.4   # Average faithfulness >= 40% (answer grounded in chunks)


# ─── Unit Tests: Metrics Logic (Fast, No Ollama) ──────────────────────────────

class TestMetrics:
    """
    These tests verify the metric functions themselves — no RAG pipeline needed.
    They run in milliseconds and are always included in fast CI.
    """

    def test_contains_all_present(self):
        answer = "Self-attention uses a Query, Key, and Value mechanism."
        assert contains_check(answer, ["query", "key", "value"]) == 1.0

    def test_contains_partial(self):
        answer = "Self-attention uses a Query vector."
        score = contains_check(answer, ["query", "key", "value"])
        assert 0.0 < score < 1.0

    def test_contains_none_present(self):
        answer = "The Transformer is a neural network."
        assert contains_check(answer, ["query", "key", "value"]) == 0.0

    def test_contains_empty_keywords(self):
        """Empty keyword list should return 1.0 (no requirements = always satisfied)."""
        assert contains_check("anything", []) == 1.0

    def test_token_f1_perfect_match(self):
        text = "self attention mechanism"
        assert token_f1(text, text) == 1.0

    def test_token_f1_no_overlap(self):
        assert token_f1("machine learning algorithms", "apple orange banana") == 0.0

    def test_token_f1_partial_overlap(self):
        f1 = token_f1("self attention model", "self attention transformer model")
        assert 0.0 < f1 < 1.0

    def test_token_f1_empty_strings(self):
        assert token_f1("", "reference") == 0.0
        assert token_f1("prediction", "") == 0.0

    def test_citation_check_with_sources(self):
        assert citation_check("Answer text.", ["docs/paper.pdf"]) is True

    def test_citation_check_no_sources(self):
        assert citation_check("Answer text.", []) is False

    def test_citation_check_inline_citation(self):
        answer = "The model uses attention [Source: paper.pdf]."
        assert citation_check(answer, []) is True

    def test_faithfulness_high(self):
        answer = "Self attention uses query vectors."
        chunks = ["Self-attention computes query, key, and value vectors for each token."]
        score = faithfulness_score(answer, chunks)
        assert score > 0.5

    def test_faithfulness_zero_no_chunks(self):
        assert faithfulness_score("some answer text", []) == 0.0

    def test_score_response_declined(self):
        """When sources and chunks are both empty, declined should be True."""
        result = score_response(
            answer="I cannot find a reliable answer.",
            sources=[],
            chunk_texts=[],
            expected_keywords=["query"],
            reference_answer="Self-attention uses query vectors.",
        )
        assert result["declined"] is True

    def test_score_response_answered(self):
        """When sources are present, declined should be False."""
        result = score_response(
            answer="The model computes query, key, and value vectors.",
            sources=["docs/paper.pdf"],
            chunk_texts=["Self-attention computes query, key, and value vectors."],
            expected_keywords=["query", "key", "value"],
            reference_answer="Self-attention uses query, key, value vectors.",
        )
        assert result["declined"] is False
        assert result["contains_score"] == 1.0
        assert result["faithfulness"] > 0.0
        assert result["citations_present"] is True


# ─── Quality Gate Tests (Require Ollama) ─────────────────────────────────────

@pytest.mark.eval
class TestQualityGates:
    """
    End-to-end quality gate tests.

    These run the full RAG pipeline against the golden dataset and assert
    that quality metrics stay above minimum thresholds.

    Auto-skipped if Ollama is not running (see conftest.py).
    """

    @pytest.fixture(autouse=True)
    def require_documents(self, live_pipeline):
        """Skip if the knowledge base is empty."""
        if live_pipeline.chunk_count == 0:
            pytest.skip(
                "Knowledge base is empty. Ingest a document first:\n"
                "  python ingest.py --source docs/transformer_architecture.md --type markdown"
            )

    # Cache to avoid re-running the slow LLM pipeline 4 separate times for the 4 tests
    _cached_results = None

    def _run_golden_questions(self, live_pipeline) -> list:
        """Run all golden questions and collect scored results (cached)."""
        if TestQualityGates._cached_results is not None:
            return TestQualityGates._cached_results

        import json
        from pathlib import Path
        from evals.metrics import score_response

        dataset_path = Path("evals/golden_dataset.jsonl")
        if not dataset_path.exists():
            pytest.skip("Golden dataset not found at evals/golden_dataset.jsonl")

        results = []
        with open(dataset_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                case = json.loads(line)
                response = live_pipeline.query(case["question"])
                chunk_texts = [c.text for c in response.retrieved_chunks]
                scores = score_response(
                    answer=response.answer,
                    sources=response.sources,
                    chunk_texts=chunk_texts,
                    expected_keywords=case.get("expected_keywords", []),
                    reference_answer=case.get("reference_answer", ""),
                )
                results.append(scores)

        TestQualityGates._cached_results = results
        return results

    def test_answer_rate_above_threshold(self, live_pipeline):
        """At least MIN_ANSWER_RATE of questions must be answered (not declined)."""
        results = self._run_golden_questions(live_pipeline)
        answered = [r for r in results if not r["declined"]]
        answer_rate = len(answered) / len(results)
        assert answer_rate >= MIN_ANSWER_RATE, (
            f"Answer rate {answer_rate:.2%} is below threshold {MIN_ANSWER_RATE:.2%}. "
            f"Citation enforcement may be too aggressive — try lowering CITATION_SCORE_THRESHOLD."
        )

    def test_mean_contains_score_above_threshold(self, live_pipeline):
        """Mean contains_score for answered questions must meet minimum."""
        results = self._run_golden_questions(live_pipeline)
        answered = [r for r in results if not r["declined"]]
        if not answered:
            pytest.skip("No questions were answered — increase citation threshold or re-ingest.")
        mean_contains = sum(r["contains_score"] for r in answered) / len(answered)
        assert mean_contains >= MIN_MEAN_CONTAINS, (
            f"Mean contains_score {mean_contains:.3f} < threshold {MIN_MEAN_CONTAINS}. "
            f"Answers are missing expected facts — check retrieval quality."
        )

    def test_citation_rate_above_threshold(self, live_pipeline):
        """At least MIN_CITATION_RATE of answered questions must cite a source."""
        results = self._run_golden_questions(live_pipeline)
        answered = [r for r in results if not r["declined"]]
        if not answered:
            pytest.skip("No questions were answered.")
        citation_rate = sum(r["citations_present"] for r in answered) / len(answered)
        assert citation_rate >= MIN_CITATION_RATE, (
            f"Citation rate {citation_rate:.2%} < threshold {MIN_CITATION_RATE:.2%}. "
            f"The generator may not be including [Source: ...] in answers."
        )

    def test_mean_faithfulness_above_threshold(self, live_pipeline):
        """Mean faithfulness (grounding) for answered questions must meet minimum."""
        results = self._run_golden_questions(live_pipeline)
        answered = [r for r in results if not r["declined"]]
        if not answered:
            pytest.skip("No questions were answered.")
        mean_faith = sum(r["faithfulness"] for r in answered) / len(answered)
        assert mean_faith >= MIN_MEAN_FAITHFULNESS, (
            f"Mean faithfulness {mean_faith:.3f} < threshold {MIN_MEAN_FAITHFULNESS}. "
            f"Answers may contain information not present in retrieved chunks."
        )
