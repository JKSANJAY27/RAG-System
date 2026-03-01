"""
ask.py — CLI Script to Query the RAG System

USAGE:
    python ask.py --question "Your question here"
    python ask.py --question "What is attention mechanism?" --top-k 3  (retrieve only 3 chunks)
    python ask.py --question "What is X?" --show-chunks  (also print the raw retrieved chunks)

EXAMPLES:
    python ask.py --question "What are the main advantages of transformer architecture?"
    python ask.py --question "How does self-attention work?" --show-chunks
"""

import argparse
import sys
import time
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

from src.rag_pipeline import RAGPipeline


def print_separator(char="─", width=70):
    print(char * width)


def main():
    parser = argparse.ArgumentParser(
        description="Query the RAG knowledge base with a question.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ask.py --question "What is the main idea of this document?"
  python ask.py --question "Explain the key findings" --show-chunks
  python ask.py --question "What methods were used?" --top-k 3
        """,
    )
    parser.add_argument(
        "--question",
        required=True,
        help="The question to ask the RAG system.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of chunks to retrieve (overrides .env setting).",
    )
    parser.add_argument(
        "--show-chunks",
        action="store_true",
        help="Also print the raw retrieved chunks that the answer was based on.",
    )
    args = parser.parse_args()

    try:
        start = time.time()
        pipeline = RAGPipeline()
        response = pipeline.query(args.question, top_k=args.top_k)
        elapsed = time.time() - start

        # ── Print the Answer ──────────────────────────────────────────────
        print("\n")
        print_separator("═")
        print("  ANSWER")
        print_separator("═")
        print(f"\n{response.answer}\n")

        # ── Print Citations ───────────────────────────────────────────────
        print_separator()
        print("  SOURCES CONSULTED")
        print_separator()
        if response.sources:
            for i, source in enumerate(response.sources, 1):
                print(f"  [{i}] {source}")
        else:
            print("  (No sources — vector store may be empty)")

        # ── Optionally print raw chunks ────────────────────────────────────
        if args.show_chunks and response.retrieved_chunks:
            print()
            print_separator()
            print("  RETRIEVED CHUNKS (raw evidence)")
            print_separator()
            for i, chunk in enumerate(response.retrieved_chunks, 1):
                print(f"\n  ┌─ Chunk {i} | {Path(chunk.source).name} | score={chunk.score:.4f}")
                # Indent and truncate for readability
                preview = chunk.text[:400].replace("\n", "\n  │ ")
                print(f"  │ {preview}")
                if len(chunk.text) > 400:
                    print(f"  │ ... [{len(chunk.text) - 400} more chars]")
                print(f"  └────")

        print()
        print_separator()
        print(f"  Model: {response.model} | Prompt v{response.prompt_version} | Time: {elapsed:.1f}s")
        print_separator("═")
        print()

    except ValueError as e:
        print(f"\n❌ {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
