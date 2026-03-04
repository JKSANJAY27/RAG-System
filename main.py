"""
main.py — Interactive REPL for the RAG System

WHAT IS A REPL?
    REPL stands for Read-Eval-Print Loop. This is an interactive session
    where you type a question, the system answers it, and you type another
    question — all without restarting the program each time.

    This is the most natural way to explore your knowledge base and
    understand how the system behaves.

USAGE:
    python main.py
    python main.py --show-chunks   (also print the raw retrieved evidence)
"""

import argparse
import sys
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

from src.rag_pipeline import RAGPipeline


BANNER = """
╔══════════════════════════════════════════════════════════════╗
║          🔍  RAG System — Interactive Query Mode             ║
║                                                              ║
║  Ask questions about your ingested documents.                ║
║  Type  'help'  for tips   |   'quit'  to exit               ║
╚══════════════════════════════════════════════════════════════╝
"""

HELP_TEXT = """
Tips for better results:
  • Ask specific questions: "What method was used for X?"
  • Instead of: "Tell me about this document"
  • Ask: "What are the main contributions of the paper?"
  
Commands:
  help   — Show this help text
  chunks — Toggle showing raw retrieved chunks
  quit   — Exit the program
"""


def format_response(response, show_chunks: bool = False):
    """Pretty-print a RAGResponse to the terminal."""
    print("\n" + "═" * 70)
    print("ANSWER")
    print("═" * 70)
    print(f"\n{response.answer}\n")

    print("─" * 70)
    print("SOURCES")
    print("─" * 70)
    if response.sources:
        for i, source in enumerate(response.sources, 1):
            print(f"  [{i}] {source}")
    else:
        print("  (No sources found — try ingesting documents first)")

    if show_chunks and response.retrieved_chunks:
        print("\n" + "─" * 70)
        print("RETRIEVED EVIDENCE")
        print("─" * 70)
        for i, chunk in enumerate(response.retrieved_chunks, 1):
            print(f"\n  Chunk {i} | {Path(chunk.source).name} | score={chunk.score:.4f}")
            print("  " + "·" * 60)
            text_preview = chunk.text[:300].replace("\n", "\n  ")
            print(f"  {text_preview}")
            if len(chunk.text) > 300:
                print(f"  ... [{len(chunk.text) - 300} more characters]")
    
    print("\n" + "─" * 70)
    print(f"Model: {response.model} | Prompt v{response.prompt_version}")
    print("═" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Interactive RAG query REPL.")
    parser.add_argument(
        "--show-chunks",
        action="store_true",
        help="Show the raw retrieved chunks alongside answers.",
    )
    args = parser.parse_args()

    print(BANNER)

    # Initialize the pipeline once — the heavy models load here
    try:
        pipeline = RAGPipeline()
    except Exception as e:
        print(f"\n❌ Failed to initialize RAG pipeline: {e}")
        print("   Make sure Ollama is running: ollama serve")
        sys.exit(1)

    if pipeline.chunk_count == 0:
        print("\n⚠  No documents ingested yet!")
        print("   Run: python ingest.py --source <path> --type <pdf|markdown|web>\n")

    show_chunks = args.show_chunks

    # ── REPL Loop ─────────────────────────────────────────────────────────────
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("\n👋 Goodbye!")
            break

        if user_input.lower() == "help":
            print(HELP_TEXT)
            continue

        if user_input.lower() == "chunks":
            show_chunks = not show_chunks
            status = "ON" if show_chunks else "OFF"
            print(f"\n  ✓ Show chunks: {status}\n")
            continue

        # Process the query
        try:
            response = pipeline.query(user_input)
            format_response(response, show_chunks=show_chunks)
        except ValueError as e:
            print(f"\n⚠  {e}\n")
        except Exception as e:
            print(f"\n❌ Error processing query: {e}\n")


if __name__ == "__main__":
    main()
