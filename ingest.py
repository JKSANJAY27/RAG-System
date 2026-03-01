"""
ingest.py — CLI Script to Ingest Documents

USAGE:
    python ingest.py --source <path_or_url> --type <pdf|markdown|web>

EXAMPLES:
    python ingest.py --source docs/attention_paper.pdf --type pdf
    python ingest.py --source docs/README.md --type markdown
    python ingest.py --source https://en.wikipedia.org/wiki/Transformer_(deep_learning) --type web
"""

import argparse
import sys
import time

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

from src.rag_pipeline import RAGPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Ingest a document into the RAG knowledge base.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ingest.py --source docs/paper.pdf --type pdf
  python ingest.py --source docs/notes.md --type markdown
  python ingest.py --source https://docs.python.org/3/ --type web
        """,
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Path to the document file or URL of the web page to ingest.",
    )
    parser.add_argument(
        "--type",
        required=True,
        choices=["pdf", "markdown", "md", "web", "url"],
        help="Type of the document.",
    )
    args = parser.parse_args()

    try:
        start = time.time()
        pipeline = RAGPipeline()
        num_chunks = pipeline.ingest(args.source, args.type)
        elapsed = time.time() - start

        print(f"\n🎉 Done! Ingested {num_chunks} chunks in {elapsed:.1f}s")
        print(f"   You can now query the system with:")
        print(f"   python ask.py --question \"Your question here\"")

    except FileNotFoundError as e:
        print(f"\n❌ File not found: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"\n❌ Invalid input: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()
