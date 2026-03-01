"""
src/embedder.py — Text Embedding Module

WHAT IS AN EMBEDDING?
    An embedding is a list of numbers (a "vector") that represents the
    *meaning* of a piece of text in mathematical space.

    Example:
        "The cat sat on the mat"  → [0.12, -0.34, 0.89, ...]  (384 numbers)
        "A feline rested on rug"  → [0.11, -0.33, 0.87, ...]  (very similar!)
        "Stock market crashes"    → [-0.45, 0.67, -0.12, ...]  (very different)

    Texts with similar *meanings* have vectors that are *close together*
    in this 384-dimensional space. This is how semantic search works.

MODEL: all-MiniLM-L6-v2
    - 384-dimensional vectors (small and fast)
    - ~90MB model size (downloads once, cached locally)
    - Great balance of speed vs quality for RAG applications
    - Runs entirely locally — no API calls needed!
"""

from typing import List

from sentence_transformers import SentenceTransformer


class Embedder:
    """
    Converts text strings into numerical embedding vectors.

    The model is loaded once when you create an Embedder instance,
    then reused for every subsequent call (very efficient).
    """

    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self):
        print(f"  ⟳ Loading embedding model '{self.MODEL_NAME}'...")
        # This downloads the model on first use (~90MB), then caches it locally
        self._model = SentenceTransformer(self.MODEL_NAME)
        print(f"  ✓ Embedding model loaded. Vector size: {self._model.get_sentence_embedding_dimension()}")

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Convert a list of text strings into a list of embedding vectors.

        Args:
            texts: List of strings to embed

        Returns:
            List of embedding vectors (each is a list of floats)

        Example:
            embedder = Embedder()
            vectors = embedder.embed(["Hello world", "Goodbye world"])
            # vectors[0] and vectors[1] will be different but close in meaning
        """
        if not texts:
            return []

        # encode() batches the texts for efficiency
        # convert_to_numpy=False keeps them as Python lists (needed by ChromaDB)
        vectors = self._model.encode(
            texts,
            show_progress_bar=len(texts) > 100,  # Show progress for large batches
            convert_to_numpy=True,
        )
        return vectors.tolist()

    def embed_single(self, text: str) -> List[float]:
        """Convenience method to embed a single string."""
        return self.embed([text])[0]
