"""
config.py — Central configuration for the RAG system.

All settings are loaded from the .env file (or environment variables).
Every module imports from here, so there's one single source of truth.

PHASE HISTORY:
    Phase 1: ollama, chroma, retrieval, chunking
    Phase 2: hybrid_alpha, reranker_model, reranker_top_k, citation_score_threshold
    Phase 6: quality gate thresholds used by CI regression gating
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load the .env file from the project root
load_dotenv()


@dataclass
class Settings:
    # ── Ollama ──────────────────────────────────────────────────────────────
    ollama_base_url: str
    ollama_model: str

    # ── ChromaDB ────────────────────────────────────────────────────────────
    chroma_persist_dir: str
    chroma_collection_name: str

    # ── Retrieval ───────────────────────────────────────────────────────────
    top_k: int

    # ── Chunking ────────────────────────────────────────────────────────────
    chunk_size: int
    chunk_overlap: int

    # ── Phase 2: Hybrid Retrieval ────────────────────────────────────────────
    # Weight for semantic search in RRF fusion (0=BM25 only, 1=vector only)
    hybrid_alpha: float

    # ── Phase 2: Re-Ranking ──────────────────────────────────────────────────
    reranker_model: str            # cross-encoder model name
    reranker_top_k: int            # keep only this many chunks after re-ranking
    citation_score_threshold: float  # min sigmoid score; below = decline to answer

    # ── Phase 6: CI Quality Gate Thresholds ──────────────────────────────────
    # These define the MINIMUM acceptable quality for the system to pass CI.
    # The quality gate (python evals/run_evals.py --ci) checks these after
    # every change to the prompt, retrieval config, or document corpus.
    # Fail = build blocked. Pass = safe to merge.
    min_answer_rate: float         # min fraction of questions that get answered
    min_mean_contains: float       # min avg keyword match score across answers
    min_citation_rate: float       # min fraction of answers citing a source
    min_mean_faithfulness: float   # min avg faithfulness (grounding) score


def load_settings() -> Settings:
    """Read environment variables and return a validated Settings object."""
    return Settings(
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
        chroma_persist_dir=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"),
        chroma_collection_name=os.getenv("CHROMA_COLLECTION_NAME", "rag_documents"),
        top_k=int(os.getenv("TOP_K", "5")),
        chunk_size=int(os.getenv("CHUNK_SIZE", "800")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "100")),
        # Phase 2
        hybrid_alpha=float(os.getenv("HYBRID_ALPHA", "0.5")),
        reranker_model=os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        reranker_top_k=int(os.getenv("RERANKER_TOP_K", "3")),
        citation_score_threshold=float(os.getenv("CITATION_SCORE_THRESHOLD", "0.1")),
        # Phase 6: quality gate thresholds (match the values in .env)
        min_answer_rate=float(os.getenv("MIN_ANSWER_RATE", "0.6")),
        min_mean_contains=float(os.getenv("MIN_MEAN_CONTAINS", "0.5")),
        min_citation_rate=float(os.getenv("MIN_CITATION_RATE", "0.7")),
        min_mean_faithfulness=float(os.getenv("MIN_MEAN_FAITHFULNESS", "0.4")),
    )


# Module-level singleton — import this anywhere in the project
settings = load_settings()
