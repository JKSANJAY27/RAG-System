"""
src/ingestor.py — Document Ingestion Module

WHY THIS EXISTS:
    Before you can search through documents, you need to load them into memory
    as plain text. This module handles three common document formats:
      • PDF files         (research papers, reports)
      • Markdown files    (documentation, notes)
      • Web pages         (online articles, docs websites)

    All three return the same `Document` object so the rest of the pipeline
    doesn't care what format the source was in. This is the "adapter pattern".

WHAT IS A `Document`?
    A simple container with:
      - content  : the raw text extracted from the source
      - source   : the file path or URL (used later for citations!)
      - metadata : any extra info (page count, title, etc.)
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader


# ─── Data Model ───────────────────────────────────────────────────────────────

@dataclass
class Document:
    """A single loaded document, ready to be chunked."""
    content: str               # The raw text
    source: str                # Original path or URL (used for citations)
    metadata: dict = field(default_factory=dict)  # Extra info


# ─── Base Class ───────────────────────────────────────────────────────────────

class BaseIngestor:
    """All ingestors share this interface: call .ingest(source) → Document."""

    def ingest(self, source: str) -> Document:
        raise NotImplementedError("Subclasses must implement .ingest()")


# ─── PDF Ingestor ─────────────────────────────────────────────────────────────

class PDFIngestor(BaseIngestor):
    """
    Loads a PDF file and extracts all text from every page.

    HOW IT WORKS:
        pypdf reads each page and extracts the text layer. Note: scanned PDFs
        (images of text) won't work — those need OCR, which is Phase 2+ territory.
    """

    def ingest(self, source: str) -> Document:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {source}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a .pdf file, got: {path.suffix}")

        reader = PdfReader(str(path))
        pages_text = []

        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text:  # Some pages may be blank or image-only
                pages_text.append(text)

        full_text = "\n\n".join(pages_text)

        if not full_text.strip():
            raise ValueError(
                f"No text could be extracted from {source}. "
                "This may be a scanned PDF (image-based). "
                "Please use a text-based PDF."
            )

        return Document(
            content=full_text,
            source=str(path),
            metadata={
                "type": "pdf",
                "page_count": len(reader.pages),
                "filename": path.name,
            },
        )


# ─── Markdown Ingestor ────────────────────────────────────────────────────────

class MarkdownIngestor(BaseIngestor):
    """
    Loads a Markdown (.md) file.

    WHY MARKDOWN?
        Most technical documentation — README files, wikis, research notes —
        is written in Markdown. This ingestor reads it as plain text,
        stripping Markdown syntax for cleaner chunking.
    """

    def ingest(self, source: str) -> Document:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Markdown file not found: {source}")

        raw_text = path.read_text(encoding="utf-8")

        # Strip Markdown syntax so the LLM sees clean prose
        clean_text = self._strip_markdown(raw_text)

        return Document(
            content=clean_text,
            source=str(path),
            metadata={
                "type": "markdown",
                "filename": path.name,
                "char_count": len(clean_text),
            },
        )

    def _strip_markdown(self, text: str) -> str:
        """Remove common Markdown syntax to produce cleaner text."""
        # Remove code blocks (```...```)
        text = re.sub(r"```[\s\S]*?```", "", text)
        # Remove inline code (`...`)
        text = re.sub(r"`[^`]+`", "", text)
        # Remove headers (## Title → Title)
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
        # Remove bold/italic markers (**text** → text)
        text = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", text)
        # Remove links ([text](url) → text)
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
        # Remove images (![alt](url))
        text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
        # Remove HTML tags
        text = re.sub(r"<[^>]+>", "", text)
        # Collapse excessive blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


# ─── Web Ingestor ─────────────────────────────────────────────────────────────

class WebIngestor(BaseIngestor):
    """
    Fetches a web page and extracts its main text content.

    HOW IT WORKS:
        1. Downloads the HTML with requests
        2. Parses it with BeautifulSoup
        3. Extracts visible text (removes nav, footer, ads etc.)

    NOTE: This works best on documentation sites and articles.
          It won't handle JavaScript-rendered pages (e.g., SPAs).
    """

    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; RAGBot/1.0; "
            "+https://github.com/JKSANJAY27/RAG-System)"
        )
    }

    def ingest(self, source: str) -> Document:
        if not source.startswith(("http://", "https://")):
            raise ValueError(f"Expected a URL starting with http/https, got: {source}")

        response = requests.get(source, headers=self.HEADERS, timeout=30)
        response.raise_for_status()  # Raises an error for 4xx/5xx responses

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove noise: navigation, footers, scripts, styles
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        # Try to find main content area (common in documentation sites)
        main_content = (
            soup.find("main")
            or soup.find("article")
            or soup.find(id="content")
            or soup.find(class_="content")
            or soup.body
        )

        text = main_content.get_text(separator="\n", strip=True) if main_content else ""
        # Collapse excessive blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)

        if not text.strip():
            raise ValueError(f"Could not extract text from: {source}")

        # Get the page title for nicer citations
        title = soup.title.string.strip() if soup.title else source

        return Document(
            content=text,
            source=source,
            metadata={
                "type": "web",
                "url": source,
                "title": title,
                "char_count": len(text),
            },
        )


# ─── Factory Function ─────────────────────────────────────────────────────────

def get_ingestor(doc_type: str) -> BaseIngestor:
    """
    Returns the right ingestor based on document type string.

    Usage:
        ingestor = get_ingestor("pdf")
        doc = ingestor.ingest("path/to/file.pdf")
    """
    ingestors = {
        "pdf": PDFIngestor,
        "markdown": MarkdownIngestor,
        "md": MarkdownIngestor,
        "web": WebIngestor,
        "url": WebIngestor,
    }
    doc_type_lower = doc_type.lower()
    if doc_type_lower not in ingestors:
        raise ValueError(
            f"Unknown document type: '{doc_type}'. "
            f"Choose from: {list(ingestors.keys())}"
        )
    return ingestors[doc_type_lower]()
