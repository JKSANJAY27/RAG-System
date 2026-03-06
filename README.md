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

### Phase 3 & 4: Evaluation & Full-Span Observability
*Added automated quality gates and deep nested trace contexts.*
- Every query generates a `TraceContext`.
- Individual spans are timed: `bm25`, `vector`, `rrf_fusion`, `rerank`, `generation`.
- The full span tree (with input/output token metrics) is flushed to **Langfuse** (for visual timelines) or a local JSONL fallback.
- **Evaluation** is done via `evals/run_evals.py` on a Golden Dataset to catch regressions.```

---

## 🗂️ Project Structure

```
rag_system/
├── src/
│   ├── ingestor.py         # PDF, Markdown, Web page loading
│   ├── chunker.py          # Token-aware text splitting (800t / 100t overlap)
│   ├── embedder.py         # sentence-transformers: all-MiniLM-L6-v2
│   ├── vector_store.py     # ChromaDB wrapper (cosine similarity)
│   ├── bm25_store.py       # Lexical keyword search (Phase 2)
│   ├── retriever.py        # Basic semantic search (Phase 1)
│   ├── hybrid_retriever.py # BM25 + Vector + RRF fusion + spans (Phase 2 & 4)
│   ├── reranker.py         # Cross-encoder re-ranking (Phase 2)
│   ├── trace_context.py    # Span collector threaded through call chain (Phase 4)
│   ├── tracer.py           # Langfuse / local JSONL backend (Phase 3 & 4)
│   ├── generator.py        # Ollama LLM + tiktoken token counting (Phase 4)
│   └── rag_pipeline.py     # Top-level orchestrator
├── prompts/
│   └── prompts.yaml        # ⭐ Version-controlled prompt templates (CI-verified)
├── evals/
│   ├── golden_dataset.jsonl  # 8 grounded Q&A pairs
│   ├── metrics.py            # contains_check, token_f1, faithfulness
│   ├── results/              # (gitignored) Per-run eval JSON
│   └── run_evals.py          # CLI: python evals/run_evals.py [--ci]
├── .github/workflows/
│   └── ci.yml                # 🆕 GitHub Actions: fast tests + quality gate (Phase 6)
├── traces/                   # (gitignored) Per-query span tree JSONL
├── tests/
│   ├── conftest.py
│   ├── test_chunker.py
│   ├── test_retriever.py
│   ├── test_generator.py
│   ├── test_hybrid_retriever.py
│   ├── test_reranker.py
│   ├── test_evaluation.py    # Quality gate tests
│   ├── test_tracer.py        # SpanTimer + TraceContext + JSONL (Phase 4)
│   └── test_metrics_dashboard.py  # Percentile math + metric aggregation (Phase 5)
├── docs/
│   └── transformer_architecture.md
├── config.py             # Centralized settings with Phase 6 quality gate thresholds
├── metrics_dashboard.py  # 🆕 CLI: P50/P95 latency, citation rate, SRE alerts (Phase 5)
├── pytest.ini            # 🆕 Marker config, warning filters (Phase 6)
├── ingest.py
├── ask.py
├── main.py
├── requirements.txt
├── .env.example          # Complete config template (schema-checked in CI)
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
tests/test_chunker.py::TestChunker::... PASSED
tests/test_evaluation.py::TestMetrics::... PASSED
tests/test_tracer.py::TestSpanTimer::... PASSED
tests/test_tracer.py::TestTraceContext::... PASSED
tests/test_tracer.py::TestRAGTracerLocal::... PASSED
tests/test_hybrid_retriever.py::TestRRF::... PASSED
tests/test_reranker.py::TestCrossEncoderReranker::... PASSED
...
====== 73 passed, 4 deselected (eval tests auto-skip without Ollama) ======
```

1. Ingesting Documents (Knowledge Base Building)
These commands process documents (chunking, embedding, vector DB storage, and BM25 index building).

# Ingest local markdown or text files
python ingest.py --source docs/transformer_architecture.md --type markdown
# Ingest local PDF files
python ingest.py --source path/to/your/document.pdf --type pdf
# Ingest web pages directly via URL
python ingest.py --source https://en.wikipedia.org/wiki/Transformer_(deep_learning) --type web
2. Single-Shot Querying (ask.py)
Best for quick tests. This triggers the full Phase 2 & 4 pipeline: BM25 + Vector → Reciprocal Rank Fusion → Cross-Encoder Re-Ranking → Citation Enforcement → LLM Generation → Langfuse Tracing.

# Ask a standard question (will cite sources)
python ask.py --question "What is the self-attention mechanism?"
# Ask a question and force the system to print the raw chunks it retrieved and re-ranked
python ask.py --question "How do Transformers beat RNNs?" --show-chunks
# Force citation enforcement (ask something NOT in the document to see it decline gracefully)
python ask.py --question "What is the capital of France?"
3. Interactive REPL (main.py)
Best for an engaging user experience. This drops you into a chat-like terminal interface where you can ask multiple questions in a row without reloading the models each time.

# Start the interactive query loop
python main.py
4. Running the Evaluation Suite (Quality Assurance)
This runs the system against the 8 "Golden Questions" curated in evals/golden_dataset.jsonl
 to ensure answer quality hasn't regressed. It calculates Metrics like Contains, Token F1, and Faithfulness.

# Run the full evaluation and generate a detailed summary report
python evals/run_evals.py
# Run in CI mode (less verbose output, pure JSON logs)
python evals/run_evals.py --ci
5. Running the Unit Tests (pytest)
The project is hardened with a robust test suite covering chunkers, retrievers, metrics, and tracers.

# Run all fast unit/integration tests (skips the slow LLM generation tests)
pytest tests/ -m "not eval" -v
# Run the FULL test suite, including the LLM automated Quality Gates
pytest tests/ -v
6. Viewing Traces & Observability
After running any of the ask.py, main.py, or run_evals.py commands, your telemetry is captured:

Option A: Cloud Dashboard (The "Glass Box")

Go to https://cloud.langfuse.com
Log in and go to your Project → Traces
Click any row to see the beautiful nested span tree showing exactly how many milliseconds bm25, vector, and rerank took, plus exact token counts.

Option B: Local Fallback If you aren't using the cloud dashboard, all the nested trace data is perfectly persisted locally:

# View the raw JSONL traces (contains the full span tree for every query)
cat traces/traces.jsonl

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
Every query your system handles generates rich telemetry: retrieval latency, re-ranker scores, number of chunks retrieved, citation enforcement rate. Without capturing this, you're operating blind. Phase 3 introduced basic query-level logging to catch regressions and track quality over time.

### What is Span-Based Tracing? (Phase 4)
Instead of just logging the total time a query took (Phase 3), span-based tracing creates a nested tree of every sub-operation. When a query is slow, you don't just know it was slow—you can see exactly that `bm25` took 12ms, `vector` took 38ms, `rerank` took 340ms, and `generation` took 6.9s. We pass a `TraceContext` down the entire call chain to build this "glass box", which is crucial for optimizing operations and token usage in production.

### What is CI Regression Gating? (Phase 6)
A **regression** is when a change makes the system worse without anyone noticing. CI gating prevents this automatically:
1. You change the prompt in `prompts.yaml` (or the chunk size, or the model)
2. Push to GitHub → workflow triggers
3. Fast tests run (code still correct ✓)
4. On PR to `main`: `python evals/run_evals.py --ci` runs all 8 golden questions
5. If `mean_faithfulness` drops from 0.72 → 0.35, the gate catches it → PR blocked with clear message
6. You fix the change → gate passes → merge

This loop makes quality a **measurable, enforceable contract**, not a hope.

### Why Version Prompts Like Code? (Phase 6)
A prompt is a configuration file that directly controls system behavior. `prompts.yaml` has a `version` field (e.g. `1.2.0`). CI fails if someone edits the template without bumping the version. This gives you:
- `git blame` for every quality change
- A paper trail: "quality dropped after a prompt change on Jan 15"
- The ability to rollback a bad prompt change the same way you rollback code

---
Averages hide your worst-case performance. 9 queries at 2s + 1 query at 20s = average 3.8s, but P95 = 20s. P50 (median) is what a *typical* user experiences. P95 is what your slowest 1-in-20 users experience. That's the number you optimize and put in your SLA.

### What is Citation Coverage? (Phase 5)
The percentage of answered queries that included at least one source citation. A high citation rate means the system is retrieving relevant documents and grounding its answers. A low rate means the retrieval is weak — time to review your chunking strategy or ingested documents.

---

## 🗺️ Roadmap

| Phase | Status | Features |
|-------|--------|---------|
| Phase 1 | ✅ **Complete** | Core RAG pipeline, PDF/MD/Web ingestion, citations, tests |
| Phase 2 | ✅ **Complete** | Hybrid BM25 + vector search, RRF fusion, cross-encoder re-ranking, citation enforcement |
| Phase 3 | ✅ **Complete** | Langfuse integration, golden eval dataset, quality gate tests |
| Phase 4 | ✅ **Complete** | Full span-based tracing (`TraceContext`, `SpanTimer`), detailed nested Langfuse timelines, token counting |
| Phase 5 | ✅ **Complete** | P50/P95/P99 latency metrics, per-stage breakdown, citation coverage, SRE alerts, HTML report |
| Phase 6 | ✅ **Complete** | GitHub Actions CI, prompt versioning guard, `.env.example` schema check, `--ci` regression gate |

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
