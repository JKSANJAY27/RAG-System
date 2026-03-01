"""
tests/test_generator.py — Unit Tests for the Generator

WHAT DO THESE TESTS VERIFY?
    1. With no chunks, the fallback message is returned
    2. The prompt is constructed with context and question
    3. Sources are correctly extracted from retrieved chunks
    4. The generator handles multiple chunks from multiple sources

KEY PATTERN — MOCKING THE LLM:
    We mock the OllamaLLM so these tests don't require Ollama to be running.
    This makes the tests fast, deterministic, and CI-friendly.
    The mock returns a fixed string, so we test the LOGIC around the LLM,
    not the LLM's actual output.

HOW TO RUN:
    pytest tests/test_generator.py -v
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.generator import Generator
from src.vector_store import RetrievedChunk


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_chunk(text: str, source: str, chunk_index: int = 0, score: float = 0.9):
    """Helper to build a RetrievedChunk for tests."""
    return RetrievedChunk(
        text=text,
        source=source,
        chunk_index=chunk_index,
        score=score,
        metadata={"token_count": len(text.split())},
    )


# ─── Tests ────────────────────────────────────────────────────────────────────

class TestGenerator:
    """
    We patch 'langchain_ollama.OllamaLLM' so no real Ollama server is needed.
    The 'patch' decorator replaces the class with a MagicMock for the test.
    """

    @patch("src.generator.OllamaLLM")
    def test_no_chunks_returns_fallback(self, mock_llm_class):
        """When no chunks are retrieved, the fallback message should be returned."""
        generator = Generator()
        result = generator.generate("What is X?", retrieved_chunks=[])

        # LLM should NOT be called (no context to generate from)
        mock_llm_class.return_value.invoke.assert_not_called()
        assert result.answer  # Has some text
        assert result.sources == []

    @patch("src.generator.OllamaLLM")
    def test_generates_answer_from_chunks(self, mock_llm_class):
        """Generator should call the LLM with a prompt containing the context."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "The answer is 42. [Source: test.pdf]"
        mock_llm_class.return_value = mock_llm

        generator = Generator()
        chunk = make_chunk(
            text="The answer to everything is 42.",
            source="docs/test.pdf",
        )
        result = generator.generate("What is the answer?", retrieved_chunks=[chunk])

        # LLM was called once
        mock_llm.invoke.assert_called_once()

        # The prompt should contain the question and the chunk text
        called_prompt = mock_llm.invoke.call_args[0][0]
        assert "What is the answer?" in called_prompt
        assert "The answer to everything is 42." in called_prompt

        # Answer should be the mock's return value
        assert "42" in result.answer

    @patch("src.generator.OllamaLLM")
    def test_sources_deduplicated(self, mock_llm_class):
        """Multiple chunks from the same source should appear only once in sources."""
        mock_llm_class.return_value.invoke.return_value = "Answer text."

        generator = Generator()
        chunks = [
            make_chunk("Text A", "docs/paper.pdf", chunk_index=0),
            make_chunk("Text B", "docs/paper.pdf", chunk_index=1),
            make_chunk("Text C", "docs/notes.md", chunk_index=0),
        ]
        result = generator.generate("Question?", retrieved_chunks=chunks)

        # 2 unique sources, not 3
        assert len(result.sources) == 2
        assert "docs/paper.pdf" in result.sources
        assert "docs/notes.md" in result.sources

    @patch("src.generator.OllamaLLM")
    def test_prompt_contains_source_annotation(self, mock_llm_class):
        """Each chunk's source should be annotated in the context block."""
        mock_llm_class.return_value.invoke.return_value = "Answer."

        generator = Generator()
        chunk = make_chunk("Some content.", source="docs/important.pdf")
        generator.generate("A question?", retrieved_chunks=[chunk])

        prompt = mock_llm_class.return_value.invoke.call_args[0][0]
        # The source filename should appear in the prompt context
        assert "important.pdf" in prompt

    @patch("src.generator.OllamaLLM")
    def test_model_name_in_result(self, mock_llm_class):
        """The result should record which model produced the answer."""
        mock_llm_class.return_value.invoke.return_value = "Answer."

        generator = Generator()
        chunk = make_chunk("Content.", source="docs/doc.pdf")
        result = generator.generate("Q?", retrieved_chunks=[chunk])

        assert result.model  # Not empty
        # Should match the settings model name
        from config import settings
        assert result.model == settings.ollama_model
