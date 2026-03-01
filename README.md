# 🔍 Production RAG System

A production-grade **Retrieval Augmented Generation (RAG)** system built with Python, ChromaDB, and local Ollama LLMs. Supports PDF, Markdown, and web pages. Provides **cited answers** grounded in your documents — no hallucinations.

> **Portfolio Project** — Built phase by phase to demonstrate full AI engineering lifecycle: ingestion → chunking → embedding → retrieval → generation → testing → monitoring.

---

## 📐 Architecture (Phase 1)

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

---

## 🗂️ Project Structure

```
rag_system/
├── src/
│   ├── ingestor.py       # PDF, Markdown, Web page loading
│   ├── chunker.py        # Token-aware text splitting (800t / 100t overlap)
│   ├── embedder.py       # sentence-transformers: all-MiniLM-L6-v2
│   ├── vector_store.py   # ChromaDB wrapper (cosine similarity)
│   ├── retriever.py      # Semantic search (query → top-k chunks)
│   ├── generator.py      # Ollama LLM answer generation with citations
│   └── rag_pipeline.py   # Top-level orchestrator
├── prompts/
│   └── prompts.yaml      # ⭐ Version-controlled prompt templates
├── tests/
│   ├── test_chunker.py   # Unit tests: chunks, overlap, metadata
│   ├── test_retriever.py # Integration tests: search quality
│   └── test_generator.py # Unit tests: prompt construction (mocked LLM)
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
# Run all tests
pytest tests/ -v

# Run specific test files
pytest tests/test_chunker.py -v
pytest tests/test_retriever.py -v    # Requires embedding model (slow first run)
pytest tests/test_generator.py -v    # Uses mocked LLM (fast, no Ollama needed)
```

Expected output:
```
tests/test_chunker.py::TestChunker::test_produces_multiple_chunks PASSED
tests/test_chunker.py::TestChunker::test_chunk_token_count_within_limit PASSED
tests/test_chunker.py::TestChunker::test_source_metadata_preserved PASSED
...
tests/test_generator.py::TestGenerator::test_no_chunks_returns_fallback PASSED
...
========================= 15 passed in 12.3s ==========================
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
| `TOP_K` | `5` | Chunks retrieved per query |
| `CHUNK_SIZE` | `800` | Max tokens per chunk |
| `CHUNK_OVERLAP` | `100` | Overlapping tokens between chunks |

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

### Why Prompt Versioning?
A prompt change can affect system quality as dramatically as a code change. `prompts/prompts.yaml` is tracked in Git — so when behavior changes, you can `git diff` your prompts just like your code.

---

## 🗺️ Roadmap

| Phase | Status | Features |
|-------|--------|---------|
| Phase 1 | ✅ **Complete** | Core RAG pipeline, PDF/MD/Web ingestion, citations, tests |
| Phase 2 | 🔜 Upcoming | Hybrid BM25 + vector search, cross-encoder re-ranker, citation enforcement |
| Phase 3 | 🔜 Upcoming | Langfuse tracing, golden eval dataset, CI regression gating, cost tracking |

---

## 📚 Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Orchestration | LangChain | Industry standard, excellent ecosystem |
| Vector DB | ChromaDB | Local, persistent, production-grade |
| Embeddings | sentence-transformers | Fast, local, no API needed |
| LLM | Ollama (llama3.2:3b) | Runs locally, no cost |
| Token counting | tiktoken | Accurate (matches real model tokenizer) |
| Config | python-dotenv + YAML | Clean separation of code and config |
| Testing | pytest | Industry standard Python testing |

---

*Built as a portfolio project to demonstrate production AI engineering practices.*
