"""
tests/conftest.py — Shared pytest fixtures and configuration (Phase 3)

WHAT IS conftest.py?
    pytest automatically loads this file before any tests run.
    Use it to define:
        - Shared fixtures (reusable test helpers)
        - Custom markers (e.g., @pytest.mark.slow)
        - Global skip conditions (e.g., skip if Ollama not running)

MARKERS DEFINED HERE:
    @pytest.mark.integration
        Tests that touch real models (slow, first run downloads weights).
        Run with: pytest -m integration
        Skip with: pytest -m "not integration"  (for fast CI)

    @pytest.mark.eval
        Tests that require a running Ollama server.
        Auto-skipped if Ollama is not reachable.

HOW FAST CI WORKS:
    Fast CI (no Ollama, no GPU): pytest tests/ -m "not eval"
    Full eval (with Ollama):     pytest tests/ -v
"""

import pytest
import requests


# ─── Custom Marker Registration ───────────────────────────────────────────────

def pytest_configure(config):
    """Register custom markers so pytest doesn't warn about unknown marks."""
    config.addinivalue_line(
        "markers",
        "integration: marks tests needing real model downloads (slow on first run)",
    )
    config.addinivalue_line(
        "markers",
        "eval: marks tests requiring a live Ollama server (auto-skipped if unavailable)",
    )


# ─── Ollama Availability Check ────────────────────────────────────────────────

def is_ollama_running() -> bool:
    """
    Check if the Ollama server is reachable at localhost:11434.

    Used by the 'eval' marker to auto-skip tests that need a live LLM.
    Without this, eval tests would fail with unhelpful connection errors.
    """
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=2)
        return resp.status_code == 200
    except Exception:
        return False


# ─── Shared Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def ollama_available() -> bool:
    """
    Session-scoped fixture: check Ollama once for the entire test run.
    Reused by all eval tests without repeated network calls.
    """
    return is_ollama_running()


@pytest.fixture(scope="session")
def live_pipeline(ollama_available):
    """
    Session-scoped fixture: create a real RAGPipeline (requires Ollama).

    session-scoped = created once and shared across all tests that use it.
    This is important because RAGPipeline downloads the embedding model
    and cross-encoder on first run (~200MB total), which is slow.

    Auto-skips the test if Ollama is not running.
    """
    if not ollama_available:
        pytest.skip("Ollama not running — skipping live pipeline tests")

    from src.rag_pipeline import RAGPipeline
    return RAGPipeline()


@pytest.fixture
def sample_chunks():
    """
    A list of RetrievedChunk objects for use in unit tests.
    These are synthetic — no real embedding is computed.
    """
    from src.vector_store import RetrievedChunk

    return [
        RetrievedChunk(
            text="Self-attention computes query, key, and value vectors for each token.",
            source="docs/transformer_architecture.md",
            chunk_index=0,
            score=0.95,
            metadata={"token_count": 15},
        ),
        RetrievedChunk(
            text="Positional encoding adds order information to embeddings.",
            source="docs/transformer_architecture.md",
            chunk_index=1,
            score=0.88,
            metadata={"token_count": 10},
        ),
        RetrievedChunk(
            text="Multi-head attention runs multiple attention computations in parallel.",
            source="docs/transformer_architecture.md",
            chunk_index=2,
            score=0.82,
            metadata={"token_count": 12},
        ),
    ]
