"""
embeddings.py
=============
Step 2 of the RAG pipeline: convert text into numerical vectors (embeddings).

Why embeddings?
- Language models and vector databases work with numbers, not raw text.
- A good embedding model places semantically similar texts close together in
  vector space, enabling semantic (meaning-based) search rather than only
  keyword matching.

This module supports two backends:
  1. sentence-transformers (free, runs locally, no API key required) ← default
  2. OpenAI embeddings (higher quality, requires OPENAI_API_KEY)
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np


# ---------------------------------------------------------------------------
# Backend: sentence-transformers (local, free)
# ---------------------------------------------------------------------------

class LocalEmbeddingModel:
    """
    Wraps a sentence-transformers model for local embedding generation.

    Usage
    -----
    >>> model = LocalEmbeddingModel()               # loads all-MiniLM-L6-v2
    >>> vectors = model.embed(["Hello!", "World!"])
    >>> vectors.shape
    (2, 384)
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Parameters
        ----------
        model_name: Any sentence-transformers model name.
                    'all-MiniLM-L6-v2' is small (80 MB) and fast while still
                    producing high-quality 384-dimensional embeddings.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            ) from exc

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Encode a list of strings into a 2-D numpy array of shape (N, D).

        Parameters
        ----------
        texts: List of strings to embed.

        Returns
        -------
        numpy array of shape (len(texts), embedding_dim).
        """
        if not texts:
            return np.array([])
        return self._model.encode(texts, convert_to_numpy=True)

    def embed_query(self, query: str) -> np.ndarray:
        """Convenience wrapper for embedding a single query string."""
        return self.embed([query])[0]


# ---------------------------------------------------------------------------
# Backend: OpenAI embeddings (requires OPENAI_API_KEY)
# ---------------------------------------------------------------------------

class OpenAIEmbeddingModel:
    """
    Wraps the OpenAI embeddings API.

    Set the environment variable OPENAI_API_KEY before using this class.

    Usage
    -----
    >>> model = OpenAIEmbeddingModel()
    >>> vectors = model.embed(["Hello!", "World!"])
    """

    def __init__(self, model_name: str = "text-embedding-3-small"):
        """
        Parameters
        ----------
        model_name: OpenAI embedding model to use.
                    'text-embedding-3-small' offers a good speed/quality trade-off.
        """
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai package is not installed. Run: pip install openai"
            ) from exc

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY environment variable is not set. "
                "Export it before using OpenAIEmbeddingModel."
            )

        self.model_name = model_name
        from openai import OpenAI  # noqa: PLC0415
        self._client = OpenAI(api_key=api_key)

    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Encode a list of strings using the OpenAI API.

        Parameters
        ----------
        texts: List of strings to embed.

        Returns
        -------
        numpy array of shape (len(texts), embedding_dim).
        """
        if not texts:
            return np.array([])
        response = self._client.embeddings.create(input=texts, model=self.model_name)
        vectors = [item.embedding for item in response.data]
        return np.array(vectors, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Convenience wrapper for embedding a single query string."""
        return self.embed([query])[0]


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def get_embedding_model(backend: str = "local", **kwargs):
    """
    Return an embedding model instance.

    Parameters
    ----------
    backend: "local"  → LocalEmbeddingModel (sentence-transformers)
             "openai" → OpenAIEmbeddingModel
    kwargs:  Forwarded to the model constructor (e.g., model_name="...").
    """
    if backend == "local":
        return LocalEmbeddingModel(**kwargs)
    if backend == "openai":
        return OpenAIEmbeddingModel(**kwargs)
    raise ValueError(f"Unknown embedding backend: {backend!r}. Choose 'local' or 'openai'.")
