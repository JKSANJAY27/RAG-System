"""
src/chunker.py — Text Chunking Module

WHY CHUNKING MATTERS:
    Language models have a context window limit — they can only process a
    certain number of tokens at once. Entire documents are usually too long.
    So we split them into smaller, overlapping "chunks" that:
      1. Fit within the model's context
      2. Can be individually embedded and searched
      3. Preserve meaning at boundaries via overlap

CHUNK SIZE (800 tokens):
    Large enough to contain complete thoughts / paragraphs.
    Small enough to embed precisely and retrieve accurately.

CHUNK OVERLAP (100 tokens):
    The last 100 tokens of chunk N are also the first 100 tokens of chunk N+1.
    This ensures that a sentence at the boundary of two chunks is captured
    in BOTH chunks — so it won't be lost no matter which one is retrieved.
"""

from dataclasses import dataclass
from typing import List

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ingestor import Document


# ─── Data Model ───────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    """A single text chunk produced by splitting a Document."""
    text: str          # The text content of this chunk
    source: str        # Inherited from the parent Document (for citations!)
    chunk_index: int   # Position of this chunk within its parent document
    metadata: dict     # Any extra info (e.g., page number, doc type)


# ─── Token Counter ────────────────────────────────────────────────────────────

def _count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """
    Count how many tokens a text has using tiktoken.

    WHY TIKTOKEN?
        Word count is imprecise. "ChatGPT" is 1 word but 1 token.
        "Supercalifragilistic" is 1 word but 4 tokens. Tiktoken gives
        accurate token counts that match what the model actually sees.

    cl100k_base is the encoding used by GPT-4 / text-embedding-ada-002.
    We use it here as a standard measure even when using a different model.
    """
    enc = tiktoken.get_encoding(encoding_name)
    return len(enc.encode(text))


# ─── Chunker ──────────────────────────────────────────────────────────────────

class Chunker:
    """
    Splits Documents into overlapping Chunks using LangChain's
    RecursiveCharacterTextSplitter.

    HOW RecursiveCharacterTextSplitter WORKS:
        It tries to split text at the most natural boundaries first:
          1. Double newlines (\\n\\n) — paragraph breaks
          2. Single newlines (\\n)   — line breaks
          3. Spaces                  — word breaks
          4. Characters              — last resort
        This means chunks end at paragraph/sentence/word boundaries whenever
        possible, making them more semantically coherent.
    """

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # LangChain splitter that uses token count (not characters) as the measure
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=_count_tokens,  # ← Token-aware, not character-aware
            separators=["\n\n", "\n", " ", ""],
        )

    def chunk(self, document: Document) -> List[Chunk]:
        """
        Split a Document into a list of Chunks.

        Args:
            document: A loaded Document from the ingestor.

        Returns:
            List of Chunk objects, each carrying the source info for citations.
        """
        if not document.content.strip():
            raise ValueError(f"Document from '{document.source}' has no content to chunk.")

        # LangChain returns plain strings; we wrap them in our Chunk dataclass
        raw_chunks: List[str] = self._splitter.split_text(document.content)

        chunks = []
        for i, text in enumerate(raw_chunks):
            chunks.append(
                Chunk(
                    text=text,
                    source=document.source,
                    chunk_index=i,
                    metadata={
                        **document.metadata,  # Inherit parent doc metadata
                        "chunk_index": i,
                        "total_chunks": len(raw_chunks),
                        "token_count": _count_tokens(text),
                    },
                )
            )

        print(
            f"  ✓ Chunked '{document.source}' into {len(chunks)} chunks "
            f"(size={self.chunk_size}, overlap={self.chunk_overlap})"
        )
        return chunks
