"""
src/generator.py — Answer Generation Module

WHAT DOES THE GENERATOR DO?
    Takes the retrieved chunks + user's question and sends them to the
    local Ollama LLM to produce a cited, grounded answer.

THE RAG GENERATION PATTERN:
    Instead of asking the LLM: "What is X?"
    We ask: "Given THESE SPECIFIC TEXTS from documents, what is X?"

    This is the key insight of RAG:
      - Without RAG: LLM answers from its training data (may be outdated/
        hallucinated)
      - With RAG: LLM answers ONLY from retrieved evidence (grounded,
        citable, trustworthy)

CITATION MECHANISM:
    Each chunk carries its source file path. We inject this into the context
    as "Source: filename.pdf | Chunk 3" so the model can reference it
    in its answer. The output includes a deduplicated list of all sources
    that were used to generate the answer.

PROMPT VERSIONING:
    The actual prompt template lives in prompts/prompts.yaml — not hardcoded
    here. This is the "engineering maturity" that makes prompts auditable.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml
from langchain_ollama import OllamaLLM

from src.vector_store import RetrievedChunk
from config import settings


# ─── Data Models ──────────────────────────────────────────────────────────────

@dataclass
class GeneratedAnswer:
    """The complete output from the RAG generator."""
    answer: str                         # The LLM's answer text
    sources: List[str]                  # Deduplicated list of source files cited
    retrieved_chunks: List[RetrievedChunk]  # The raw chunks that were used
    model: str                          # Which model generated this answer
    prompt_version: str                 # Which prompt version was used


# ─── Prompt Loader ────────────────────────────────────────────────────────────

def _load_prompts() -> dict:
    """Load prompt templates from prompts/prompts.yaml."""
    prompts_path = Path(__file__).parent.parent / "prompts" / "prompts.yaml"
    with open(prompts_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ─── Generator ────────────────────────────────────────────────────────────────

class Generator:
    """
    Uses a local Ollama LLM to generate cited answers from retrieved chunks.

    OLLAMA INTEGRATION:
        Ollama runs a local server (http://localhost:11434) that exposes
        your locally-pulled models via a REST API. LangChain's OllamaLLM
        talks to that server automatically.

        You never need an internet connection or API key once the model is
        pulled locally with `ollama pull llama3.2:3b`.
    """

    def __init__(self):
        prompts_data = _load_prompts()
        self._prompt_template: str = prompts_data["rag_answer"]["template"]
        self._no_context_template: str = prompts_data["no_context_fallback"]["template"]
        self._prompt_version: str = prompts_data.get("version", "unknown")

        print(f"  ⟳ Connecting to Ollama model '{settings.ollama_model}'...")
        self._llm = OllamaLLM(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0.1,   # Low temperature = more factual, less creative
        )
        print(f"  ✓ Generator ready (model={settings.ollama_model}, temp=0.1)")

    def generate(
        self,
        question: str,
        retrieved_chunks: List[RetrievedChunk],
    ) -> GeneratedAnswer:
        """
        Generate a cited answer from the retrieved chunks.

        Args:
            question: The user's question in plain English.
            retrieved_chunks: Chunks from the retriever.

        Returns:
            GeneratedAnswer with the answer text and source citations.
        """
        if not retrieved_chunks:
            # No chunks means the vector store is empty or query found nothing
            fallback = self._no_context_template
            return GeneratedAnswer(
                answer=fallback,
                sources=[],
                retrieved_chunks=[],
                model=settings.ollama_model,
                prompt_version=self._prompt_version,
            )

        # ── Build the context string from retrieved chunks ─────────────────
        # Format each chunk with a header showing where it came from.
        # This is what gets injected into {context} in the prompt.
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            source_name = Path(chunk.source).name  # Just the filename, not full path
            header = f"[Chunk {i} | Source: {source_name} | Score: {chunk.score:.3f}]"
            context_parts.append(f"{header}\n{chunk.text}")

        context_block = "\n\n" + "\n\n---\n\n".join(context_parts) + "\n"

        # ── Fill in the prompt template ────────────────────────────────────
        prompt = self._prompt_template.format(
            context=context_block,
            question=question,
        )

        # ── Call the local LLM ─────────────────────────────────────────────
        print(f"\n  ⟳ Sending prompt to '{settings.ollama_model}'...")
        print(f"     Context: {len(retrieved_chunks)} chunks | "
              f"~{sum(c.metadata.get('token_count', 0) for c in retrieved_chunks)} tokens")

        answer_text = self._llm.invoke(prompt)

        # ── Extract deduplicated sources ───────────────────────────────────
        # Show unique sources so the user knows which documents were consulted
        seen = set()
        unique_sources = []
        for chunk in retrieved_chunks:
            src = chunk.source
            if src not in seen:
                seen.add(src)
                unique_sources.append(src)

        print(f"  ✓ Answer generated. Sources: {[Path(s).name for s in unique_sources]}")

        return GeneratedAnswer(
            answer=answer_text.strip(),
            sources=unique_sources,
            retrieved_chunks=retrieved_chunks,
            model=settings.ollama_model,
            prompt_version=self._prompt_version,
        )
