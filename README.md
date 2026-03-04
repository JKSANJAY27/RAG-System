# 🔍 Production RAG System

A production-grade **Retrieval Augmented Generation (RAG)** system built with Python, ChromaDB, and local Ollama LLMs. Supports PDF, Markdown, and web pages. Provides **cited answers** grounded in your documents, features an advanced **Hybrid Retrieval** pipeline with BM25, Reciprocal Rank Fusion, and Cross-Encoder Re-Ranking, and includes **full observability** via Langfuse tracing and a golden dataset quality gate.

> **Portfolio Project** — Built phase by phase to demonstrate full AI engineering lifecycle: ingestion → chunking → embedding → retrieval → generation → testing → monitoring.

---

## 📐 Architecture

### Phase 1: Core Fundamentals
```
                        INGEST PATH
┌──────────┐    ┌─────────┐    ┌──────────┐    ┌───────────────┐
│ Document │───▶│Ingestor │───▶│ Chunker  │───▶│   Embedder    │
│(PDF/MD/  │    │         │    │800 tok / │    │all-MiniLM-L6  │
│  Web)    │    └─────────┘    │100 ovrlp │    │   → vectors   │
└──────────┘                   └──────────┘    └───────┬───────┘
                                                        │
                                                        ▼
                                               ┌────────────────┐
                                               │   ChromaDB     │
                                               │ (persistent on │
                                               │    disk)       │
                                               └───────┬────────┘
                        QUERY PATH                     │
┌──────────┐    ┌─────────┐    ┌──────────┐    ┌──────▼───────┐
│  User    │    │ Answer  │◀───│Generator │◀───│  Retriever   │
│ Question │───▶│ + Cites │    │ Ollama   │    │ top-k chunks │
└──────────┘    └─────────┘    │llama3.2  │    └──────────────┘
                                └──────────┘
```

### Phase 2: Hybrid Retrieval & Re-Ranking
*Added lexical search and precision re-filtering.*
```
                             query
                               │
                ┌──────────────┴──────────────┐
                ▼                             ▼
         ┌────────────┐                ┌────────────┐
         │ BM25 Store │                │Chroma Store│
         │ (Keywords) │                │ (Semantic) │
         └──────┬─────┘                └──────┬─────┘
                │                             │
                └──────────────┬──────────────┘
                               ▼
                    ┌────────────────────┐
                    │Reciprocal Rank     │  ← RRF Merges disparate
                    │Fusion (RRF)        │    score scales
                    └──────────┬─────────┘
                               ▼
                    ┌────────────────────┐
                    │Cross-Encoder       │  ← ms-marco-MiniLM-L-6-v2
                    │Re-Ranker           │    (re-scores top chunks)
                    └──────────┬─────────┘
                               ▼
                    ┌────────────────────┐
                    │Citation Enforcement│  ← If best score < threshold,
                    │Threshold           │    decline to answer
                    └──────────┬─────────┘
                               ▼
                          (Generator)
```

---

## 🗂️ Project Structure

```
rag_system/
├── src/
│   ├── ingestor.py         # PDF, Markdown, Web page loading
│   ├── chunker.py          # Token-aware text splitting (800t / 100t overlap)
│   ├── embedder.py         # sentence-transformers: all-MiniLM-L6-v2
│   ├── vector_store.py     # ChromaDB wrapper (cosine similarity)
│   ├── bm25_store.py       # 🆕 Lexical keyword search index (Phase 2)
│   ├── retriever.py        # Basic semantic search (Phase 1)
│   ├── hybrid_retriever.py # 🆕 BM25 + Vector + RRF fusion (Phase 2)
│   ├── reranker.py         # 🆕 cross-encoder precision re-ranking (Phase 2)
│   ├── tracer.py           # 🆕 Langfuse / local JSONL observability (Phase 3)
│   ├── generator.py        # Ollama LLM answer generation with citations
│   └── rag_pipeline.py     # Top-level orchestrator
├── prompts/
│   └── prompts.yaml        # ⭐ Version-controlled prompt templates
├── evals/                    # 🆕 Phase 3: Evaluation framework
│   ├── golden_dataset.jsonl  # 8 grounded Q&A pairs for regression testing
│   ├── metrics.py            # Scoring: contains_check, token_f1, faithfulness
│   └── run_evals.py          # CLI runner — scores all golden questions
├── traces/                   # (gitignored) Per-query telemetry JSONL
├── tests/
│   ├── conftest.py              # Shared fixtures + Ollama availability check
│   ├── test_chunker.py
│   ├── test_retriever.py
│   ├── test_generator.py
│   ├── test_hybrid_retriever.py # RRF and BM25 tests
│   ├── test_reranker.py         # Citation enforcement tests
│   └── test_evaluation.py       # 🆕 Quality gate tests

├── docs/
│   └── transformer_architecture.md  # Sample document for testing
├── config.py             # Centralized settings from .env
├── ingest.py             # CLI: add a document to knowledge base
├── ask.py                # CLI: one-shot question answering
├── main.py               # Interactive REPL (multi-turn Q&A)
├── requirements.txt
├── .env.example          # Template — copy to .env and configure
└── README.md
```

---

## ⚡ Quick Start

### 1. Prerequisites

| Tool | Install | Version |
|------|---------|---------|
| Python | [python.org](https://python.org) | 3.11+ |
| Ollama | [ollama.com/download](https://ollama.com/download) | Latest |

After installing Ollama, pull the model:
```bash
ollama pull llama3.2:3b
```

### 2. Install Dependencies

```bash
# Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate      # Mac/Linux

# Install all packages
pip install -r requirements.txt
```

> ⏳ First install downloads ~90MB of model weights for `all-MiniLM-L6-v2`. This is cached locally after the first run.

### 3. Configure Environment

```bash
# .env is already pre-configured with defaults — no changes needed!
# To customize, edit .env:
copy .env.example .env
```

### 4. Ingest a Document

```bash
# Ingest the included sample document
python ingest.py --source docs/transformer_architecture.md --type markdown

# Or ingest a PDF
python ingest.py --source path/to/your/paper.pdf --type pdf

# Or ingest a web page
python ingest.py --source https://en.wikipedia.org/wiki/Transformer_(deep_learning) --type web
```

### 5. Ask Questions

**One-shot query:**
```bash
python ask.py --question "What is the self-attention mechanism?"

# Show the raw retrieved chunks too
python ask.py --question "How do Transformers beat RNNs?" --show-chunks
```

**Interactive REPL (recommended for exploration):**
```bash
python main.py
# Then just type your questions!
```

---

## 📖 Example Output

```
❓ Question: 'What is self-attention and why does it matter?'
──────────────────────────────────────────────────────────────────────

═══════════════════════════════════════════════════════════════════════
  ANSWER
═══════════════════════════════════════════════════════════════════════

Self-attention is a mechanism that allows every word in a sequence to
directly attend to every other word simultaneously [Source: transformer_
architecture.md]. This solved the critical "vanishing gradient" problem
of RNNs, where important information faded away over long distances
[Source: transformer_architecture.md]. 

It works by computing three vectors for each word — a Query, Key, and
Value — and using the dot product of Query and Key to determine how
much each word should attend to every other word [Source: transformer_
architecture.md].

──────────────────────────────────────────────────────────────────────
  SOURCES CONSULTED
──────────────────────────────────────────────────────────────────────
  [1] docs/transformer_architecture.md

Model: llama3.2:3b | Prompt v1.0.0 | Time: 8.3s
═══════════════════════════════════════════════════════════════════════
```

---

## 🧪 Running Tests

```bash
# ─── Fast CI: no Ollama needed (~60s)
venv\Scripts\activate
pytest tests/ -m "not eval" -v

# ─── Full evaluation suite (requires Ollama + ingested doc)
pytest tests/test_evaluation.py -v
# Quality gate tests auto-skip if Ollama is not running

# ─── Run the golden dataset CLI evaluation
python evals/run_evals.py
# Saves timestamped results to: evals/results/eval_YYYYMMDD_HHMMSS.json
```

Expected output:
```
tests/test_chunker.py::TestChunker::test_produces_multiple_chunks PASSED
...
tests/test_evaluation.py::TestMetrics::test_contains_all_present PASSED
tests/test_evaluation.py::TestMetrics::test_token_f1_perfect_match PASSED
...
tests/test_hybrid_retriever.py::TestRRF::test_document_in_both_lists_ranks_higher PASSED
tests/test_reranker.py::TestCrossEncoderReranker::test_citation_enforcement_fires_when_below_threshold PASSED
...
====== 53 passed, 4 deselected (eval skipped — Ollama not running) ======
```

---

## ⚙️ Configuration Reference

All settings live in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.2:3b` | Model for answer generation |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | Where ChromaDB stores data |
| `CHROMA_COLLECTION_NAME` | `rag_documents` | Collection name (like a table) |
| `TOP_K` | `5` | Candidate chunks per retriever |
| `CHUNK_SIZE` | `800` | Max tokens per chunk |
| `CHUNK_OVERLAP` | `100` | Overlapping tokens between chunks |
| `HYBRID_ALPHA` | `0.5` | Weight for fusion (Phase 2) |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Model for precision ranking |
| `RERANKER_TOP_K` | `3` | Final chunks sent to LLM |
| `CITATION_SCORE_THRESHOLD` | `0.1` | Safety cutoff to prevent hallucination |
| `LANGFUSE_PUBLIC_KEY` | *(blank)* | Langfuse public key (Phase 3) |
| `LANGFUSE_SECRET_KEY` | *(blank)* | Langfuse secret key (Phase 3) |
| `MIN_ANSWER_RATE` | `0.6` | Quality gate: min answer rate (Phase 3) |

---

## 🔬 Key Engineering Concepts

### Why Token-Based Chunking?
Character count is imprecise — `"ChatGPT"` is 1 word but could be 1-3 tokens. We use `tiktoken` for exact token counts that match what the LLM actually sees.

### Why 100 Token Overlap?
If chunk N ends with sentence A and chunk N+1 starts with sentence B, the idea connecting them lives at the boundary. Overlap ensures that boundary context is captured in both chunks — so it's never lost regardless of which chunk is retrieved.

### Why Cosine Similarity?
For semantic text search, we care about the *direction* (meaning) of vectors, not their magnitude. Cosine similarity measures the angle between vectors — identical meaning = angle of 0° = similarity of 1.0.

### Why Local Ollama?
- **Privacy**: Your documents never leave your machine
- **Cost**: No per-token API fees
- **Speed**: No network latency after model is loaded
- **Learning**: You understand the full stack you're running

### Why Prompt Versioning? (Phase 1 & 2)
A prompt change can affect system quality as dramatically as a code change. `prompts/prompts.yaml` is tracked in Git — so when behavior changes, you can `git diff` your prompts just like your code. In Phase 2, we incremented to v1.1.0 to add re-ranker awareness.

### Why Hybrid Search? (Phase 2)
Vector search finds *meaning*, but BM25 finds *exact terms*. A query like `"HTTP 404"` needs BM25; a query like `"what causes page not found errors"` needs vector. Hybrid search combined with **Reciprocal Rank Fusion (RRF)** gets the best of both worlds.

### Why Cross-Encoder Re-Ranking? (Phase 2)
Bi-encoders (like our vector embeddings) are fast but imprecise because they compute document and query vectors separately. Cross-encoders look at the query and document *together*, providing much higher accuracy at the cost of speed. We use it only on the top 10 candidates.

### What is Citation Enforcement? (Phase 2)
If the highest-scoring chunk from the cross-encoder falls below a threshold, the system **refuses to answer**. Without this, the LLM will try to answer using weak evidence and hallucinate. Returning a polite decline is what production systems actually do.

### Why a Golden Dataset? (Phase 3)
A golden dataset is a curated set of question-answer pairs that encode your quality expectations as executable contracts. Every time you change a prompt, chunk size, or model, run `evals/run_evals.py` to check you haven't regressed. It's the difference between **hoping** your change helped and **knowing** it did.

### What is LLM Observability? (Phase 3)
Every query your system handles generates rich telemetry: retrieval latency, re-ranker scores, number of chunks retrieved, citation enforcement rate. Without capturing this, you're operating blind. Langfuse (or the local JSONL fallback) makes every query inspectable, enabling you to catch regressions, optimize slow queries, and track quality over time.

### Two Types of Tests in Production AI
- **Unit/integration tests** (`pytest tests/ -m "not eval"`) — verify code correctness, run in CI in seconds, no GPU needed
- **Evaluation tests** (`pytest tests/ -m eval`) — verify *answer quality*, require Ollama, run before releases

---

## 🗺️ Roadmap

| Phase | Status | Features |
|-------|--------|---------|
| Phase 1 | ✅ **Complete** | Core RAG pipeline, PDF/MD/Web ingestion, citations, tests |
| Phase 2 | ✅ **Complete** | Hybrid BM25 + vector search, RRF fusion, cross-encoder re-ranking, citation enforcement |
| Phase 3 | ✅ **Complete** | Langfuse tracing, golden eval dataset, quality gate tests, per-query latency tracking |

---

## 📚 Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Orchestration | LangChain | Industry standard, excellent ecosystem |
| Vector DB | ChromaDB | Local, persistent, production-grade |
| Embeddings | sentence-transformers | Fast, local, no API needed |
| Lexical Search | rank-bm25 | Standard keyword matching |
| Re-Ranker | cross-encoder/ms-marco | Precision relevance scoring |
| Observability | Langfuse | LLM-native tracing dashboard |
| LLM | Ollama (llama3.2:3b) | Runs locally, no cost |
| Token counting | tiktoken | Accurate (matches real model tokenizer) |
| Config | python-dotenv + YAML | Clean separation of code and config |
| Testing | pytest | Industry standard Python testing |

---

*Built as a portfolio project to demonstrate production AI engineering practices.*
