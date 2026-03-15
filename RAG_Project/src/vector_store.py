"""
vector_store.py
===============
Step 3 of the RAG pipeline: store embeddings and retrieve the most relevant
chunks for a given query.

Why a vector database?
- Standard databases search by exact match or range comparisons.
- Vector databases support *approximate nearest-neighbour* (ANN) search:
  given a query vector, return the k stored vectors that are most similar
  according to a distance metric such as cosine similarity.
- This enables semantic search — finding documents that *mean* the same thing
  as the query even if they don't share keywords.

This module uses ChromaDB, an open-source, in-process vector database that
needs no external server and can optionally persist data to disk.
"""

from __future__ import annotations

import uuid
from typing import Any

import numpy as np


class VectorStore:
    """
    A thin wrapper around a ChromaDB collection.

    Basic usage
    -----------
    >>> store = VectorStore(collection_name="my_docs")
    >>> store.add_documents(chunks)          # list[dict] from document_loader
    >>> results = store.query("What is RAG?", top_k=3)
    >>> for r in results:
    ...     print(r["text"])
    """

    def __init__(
        self,
        collection_name: str = "rag_collection",
        persist_directory: str | None = None,
    ):
        """
        Parameters
        ----------
        collection_name:    Name of the ChromaDB collection to create / reuse.
        persist_directory:  Path to a directory for on-disk persistence.
                            If None the collection is stored in memory only
                            and will be lost when the process exits.
        """
        try:
            import chromadb
        except ImportError as exc:
            raise ImportError(
                "chromadb is not installed. Run: pip install chromadb"
            ) from exc

        import chromadb  # noqa: PLC0415

        if persist_directory:
            self._client = chromadb.PersistentClient(path=persist_directory)
        else:
            self._client = chromadb.Client()

        # get_or_create_collection is idempotent — safe to call multiple times
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # use cosine similarity
        )

    # ------------------------------------------------------------------
    # Adding documents
    # ------------------------------------------------------------------

    def add_documents(
        self,
        chunks: list[dict],
        embedding_model=None,
    ) -> None:
        """
        Embed and store a list of document chunks.

        Parameters
        ----------
        chunks:          Output of ``document_loader.load_and_chunk_documents``.
                         Each item must have "id", "text", and "metadata" keys.
        embedding_model: An object with an ``embed(texts) -> np.ndarray`` method
                         (e.g., LocalEmbeddingModel or OpenAIEmbeddingModel).
                         If None, ChromaDB will use its built-in default embedder.
        """
        if not chunks:
            return

        ids = [c["id"] for c in chunks]
        texts = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]

        if embedding_model is not None:
            embeddings: list[list[float]] = embedding_model.embed(texts).tolist()
            self._collection.add(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )
        else:
            # Let ChromaDB handle embeddings with its default model
            self._collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas,
            )

    def add_texts(
        self,
        texts: list[str],
        metadatas: list[dict] | None = None,
        embedding_model=None,
    ) -> list[str]:
        """
        Convenience method: embed and store raw strings (no pre-chunking needed).

        Returns the auto-generated IDs.
        """
        ids = [str(uuid.uuid4()) for _ in texts]
        metas = metadatas or [None for _ in texts]

        chunks = [
            {"id": doc_id, "text": text, "metadata": meta}
            for doc_id, text, meta in zip(ids, texts, metas)
        ]
        self.add_documents(chunks, embedding_model=embedding_model)
        return ids

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        embedding_model=None,
        where: dict | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve the *top_k* most relevant chunks for *query_text*.

        Parameters
        ----------
        query_text:      Natural-language question or search phrase.
        top_k:           Number of results to return.
        embedding_model: Must be the *same* model used when adding documents.
                         If None, ChromaDB's default embedder is used.
        where:           Optional ChromaDB metadata filter
                         (e.g., {"doc_index": {"$eq": 0}}).

        Returns
        -------
        List of dicts, each containing:
            "id"       : document chunk ID
            "text"     : chunk text
            "metadata" : metadata dict
            "distance" : cosine distance (lower = more similar)
        """
        kwargs: dict[str, Any] = {
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }

        if where:
            kwargs["where"] = where

        if embedding_model is not None:
            query_vector: list[float] = embedding_model.embed_query(query_text).tolist()
            kwargs["query_embeddings"] = [query_vector]
        else:
            kwargs["query_texts"] = [query_text]

        raw = self._collection.query(**kwargs)

        results = []
        for i, doc_id in enumerate(raw["ids"][0]):
            results.append(
                {
                    "id": doc_id,
                    "text": raw["documents"][0][i],
                    "metadata": raw["metadatas"][0][i],
                    "distance": raw["distances"][0][i],
                }
            )
        return results

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Return the total number of chunks stored in the collection."""
        return self._collection.count()

    def reset(self) -> None:
        """Delete all documents from the collection (useful in tests)."""
        self._client.delete_collection(self._collection.name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection.name,
            metadata={"hnsw:space": "cosine"},
        )
