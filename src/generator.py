"""
src/generator.py — Answer Generation Module (Phase 4: Instrumented)

WHAT CHANGED FROM PHASE 3?
    Added optional `trace_ctx` parameter to .generate().
    When provided, it records to the "generation" span:
        - The exact rendered prompt (first 500 chars as preview)
        - Prompt token count  (via tiktoken cl100k_base)
        - Completion token count
        - Total token count
        - The answer preview (first 500 chars)
        - Generation latency

WHY TOKEN COUNTING MATTERS:
    Even with a free local model, token count is a proxy for:
        - Latency correlation: more tokens → slower generation
        - Quality sanity check: too few tokens → truncated answer?
        - Phase 5 cost estimation placeholder (swap in an API cost
          per-token later if you move to GPT-4 or Claude)

    We use tiktoken's cl100k_base encoding because it closely matches
    how most modern LLMs (including llama series) tokenize text.
    Exact token counts differ by model, but estimates are close enough.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import tiktoken
import yaml
from langchain_ollama import OllamaLLM

from src.trace_context import TraceContext, SpanTimer
from src.vector_store import RetrievedChunk
from config import settings


# ─── Token Counter (module-level, cached once) ────────────────────────────────
# cl100k_base is the tokenizer for GPT-4 and a reasonable proxy for any
# modern LLM. Load once at import time.
_TOKENIZER = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    """Count approximate tokens in a string using the cl100k_base tokenizer."""
    return len(_TOKENIZER.encode(text))


# ─── Data Models ──────────────────────────────────────────────────────────────

@dataclass
class GeneratedAnswer:
    """The complete output from the RAG generator."""
    answer: str
    sources: List[str]
    retrieved_chunks: List[RetrievedChunk]
    model: str
    prompt_version: str
    # Phase 4 additions:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


# ─── Prompt Loader ────────────────────────────────────────────────────────────

def _load_prompts() -> dict:
    prompts_path = Path(__file__).parent.parent / "prompts" / "prompts.yaml"
    with open(prompts_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ─── Generator ────────────────────────────────────────────────────────────────

class Generator:
    """
    Uses a local Ollama LLM to generate cited answers from retrieved chunks.

    Phase 4 addition:
        - Accepts optional trace_ctx; records a "generation" span with:
            prompt token count, response token count, prompt preview,
            answer preview.
        - Token counts are also returned in GeneratedAnswer for Phase 5
          cost/usage tracking.
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
            temperature=0.1,
        )
        print(f"  ✓ Generator ready (model={settings.ollama_model}, temp=0.1)")

    def generate(
        self,
        question: str,
        retrieved_chunks: List[RetrievedChunk],
        trace_ctx: Optional[TraceContext] = None,
    ) -> GeneratedAnswer:
        """
        Generate a cited answer from the retrieved chunks.

        Args:
            question: The user's question.
            retrieved_chunks: Re-ranked chunks from hybrid retriever.
            trace_ctx: Optional trace context (Phase 4). When provided,
                       records a "generation" span with prompt + token data.

        Returns:
            GeneratedAnswer with answer, sources, and token counts.
        """
        if not retrieved_chunks:
            fallback = self._no_context_template
            return GeneratedAnswer(
                answer=fallback,
                sources=[],
                retrieved_chunks=[],
                model=settings.ollama_model,
                prompt_version=self._prompt_version,
            )

        # ── Build the context block ────────────────────────────────────────────
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            source_name = Path(chunk.source).name
            header = f"[Chunk {i} | Source: {source_name} | Score: {chunk.score:.3f}]"
            context_parts.append(f"{header}\n{chunk.text}")

        context_block = "\n\n" + "\n\n---\n\n".join(context_parts) + "\n"

        # ── Render prompt ──────────────────────────────────────────────────────
        prompt = self._prompt_template.format(
            context=context_block,
            question=question,
        )

        # ── Count prompt tokens BEFORE the LLM call ───────────────────────────
        prompt_tokens = _count_tokens(prompt)

        print(f"\n  ⟳ Sending prompt to '{settings.ollama_model}'...")
        print(f"     Context: {len(retrieved_chunks)} chunks | "
              f"~{prompt_tokens} prompt tokens")

        # ── Call the LLM (timed) ───────────────────────────────────────────────
        with SpanTimer() as gen_timer:
            answer_text = self._llm.invoke(prompt)

        answer_text = answer_text.strip()

        # ── Count completion tokens AFTER ──────────────────────────────────────
        completion_tokens = _count_tokens(answer_text)
        total_tokens = prompt_tokens + completion_tokens

        print(f"  ✓ Answer: {completion_tokens} tokens in "
              f"{gen_timer.elapsed_ms:.0f}ms "
              f"(prompt={prompt_tokens}, total={total_tokens})")

        # ── Record "generation" span ───────────────────────────────────────────
        if trace_ctx is not None:
            trace_ctx.record(
                name="generation",
                latency_ms=gen_timer.elapsed_ms,
                input={
                    "prompt_preview": prompt[:500],       # first 500 chars
                    "prompt_tokens": prompt_tokens,
                    "chunk_count": len(retrieved_chunks),
                },
                output={
                    "answer_preview": answer_text[:500],  # first 500 chars
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "prompt_tokens": prompt_tokens,
                },
            )

        # ── Extract deduplicated sources ───────────────────────────────────────
        seen = set()
        unique_sources = []
        for chunk in retrieved_chunks:
            if chunk.source not in seen:
                seen.add(chunk.source)
                unique_sources.append(chunk.source)

        print(f"  ✓ Sources: {[Path(s).name for s in unique_sources]}")

        return GeneratedAnswer(
            answer=answer_text,
            sources=unique_sources,
            retrieved_chunks=retrieved_chunks,
            model=settings.ollama_model,
            prompt_version=self._prompt_version,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
