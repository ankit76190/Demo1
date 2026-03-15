"""
rag_pipeline.py
===============
Step 4: the full RAG pipeline.

This module ties together document loading, embedding, retrieval, and
(optionally) language-model generation into a single easy-to-use class.

Architecture recap:
    User question
        │
        ▼ embed_query()
    Query vector ──────▶ VectorStore.query() ──▶ Retrieved chunks
                                                        │
                                                        ▼ build_prompt()
                                                    Augmented prompt
                                                        │
                                                        ▼ generate()
                                                      Answer
"""

from __future__ import annotations

import os
from typing import Any

from .document_loader import load_and_chunk_documents
from .embeddings import get_embedding_model
from .vector_store import VectorStore


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_prompt(question: str, context_chunks: list[str]) -> str:
    """
    Construct an LLM prompt that includes retrieved context.

    The system instruction tells the model to:
    - Answer only from the provided context.
    - Admit when it does not know, rather than hallucinating.

    Parameters
    ----------
    question:       The user's original question.
    context_chunks: List of retrieved text passages.

    Returns
    -------
    A formatted multi-line prompt string.
    """
    separator = "\n\n---\n\n"
    context_text = separator.join(context_chunks)

    prompt = (
        "You are a helpful assistant. Use ONLY the context below to answer "
        "the question. If the context does not contain enough information to "
        "answer, say \"I don't know based on the provided context.\"\n\n"
        f"CONTEXT:\n{context_text}\n\n"
        f"QUESTION: {question}\n\n"
        "ANSWER:"
    )
    return prompt


# ---------------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------------

def _answer_with_openai(prompt: str, model: str = "gpt-4o-mini") -> str:
    """Call OpenAI chat completions and return the assistant's reply."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "openai package is not installed. Run: pip install openai"
        ) from exc

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY environment variable is not set."
        )

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def _answer_no_llm(prompt: str) -> str:
    """
    Fallback when no LLM is available.

    Simply returns the prompt so users can paste it into a chat UI manually.
    """
    divider = "=" * 60
    return (
        f"{divider}\n"
        "No LLM configured. Copy the prompt below into ChatGPT or another\n"
        "chat interface to get an answer:\n"
        f"{divider}\n\n"
        f"{prompt}"
    )


# ---------------------------------------------------------------------------
# Main RAG Pipeline class
# ---------------------------------------------------------------------------

class RAGPipeline:
    """
    End-to-end Retrieval-Augmented Generation pipeline.

    Quick start
    -----------
    >>> pipeline = RAGPipeline()
    >>> pipeline.index_file("data/sample_documents.txt")
    >>> answer = pipeline.ask("What is a transformer in machine learning?")
    >>> print(answer)
    """

    def __init__(
        self,
        embedding_backend: str = "local",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        llm_backend: str | None = None,
        llm_model: str = "gpt-4o-mini",
        top_k: int = 3,
        chunk_size: int = 300,
        chunk_overlap: int = 50,
        collection_name: str = "rag_collection",
        persist_directory: str | None = None,
    ):
        """
        Parameters
        ----------
        embedding_backend:    "local" or "openai"
        embedding_model_name: Model name passed to the embedding backend.
        llm_backend:          None  → print the prompt (no LLM call)
                              "openai" → call OpenAI chat API
        llm_model:            OpenAI model to use (if llm_backend="openai").
        top_k:                Number of context chunks to retrieve per query.
        chunk_size:           Characters per chunk when indexing documents.
        chunk_overlap:        Character overlap between consecutive chunks.
        collection_name:      ChromaDB collection name.
        persist_directory:    If set, embeddings are saved to this directory.
        """
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.llm_backend = llm_backend
        self.llm_model = llm_model

        self.embedding_model = get_embedding_model(
            backend=embedding_backend,
            model_name=embedding_model_name,
        )
        self.vector_store = VectorStore(
            collection_name=collection_name,
            persist_directory=persist_directory,
        )

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_file(self, file_path: str) -> int:
        """
        Load, chunk, and embed a plain-text file into the vector store.

        Parameters
        ----------
        file_path: Path to the text file to index.

        Returns
        -------
        Number of chunks added to the vector store.
        """
        chunks = load_and_chunk_documents(
            file_path,
            chunk_size=self.chunk_size,
            overlap=self.chunk_overlap,
        )
        self.vector_store.add_documents(chunks, embedding_model=self.embedding_model)
        return len(chunks)

    def index_texts(self, texts: list[str], metadatas: list[dict] | None = None) -> list[str]:
        """
        Directly index a list of text strings.

        Returns the list of auto-generated document IDs.
        """
        return self.vector_store.add_texts(
            texts,
            metadatas=metadatas,
            embedding_model=self.embedding_model,
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, question: str) -> list[dict[str, Any]]:
        """
        Return the top-k most relevant chunks for *question*.

        Each result dict has keys: "id", "text", "metadata", "distance".
        """
        return self.vector_store.query(
            question,
            top_k=self.top_k,
            embedding_model=self.embedding_model,
        )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def ask(self, question: str) -> str:
        """
        Full RAG query: retrieve relevant context then generate an answer.

        Parameters
        ----------
        question: The user's natural-language question.

        Returns
        -------
        The generated answer string (or the formatted prompt if no LLM is
        configured).
        """
        results = self.retrieve(question)
        context_chunks = [r["text"] for r in results]
        prompt = build_prompt(question, context_chunks)

        if self.llm_backend == "openai":
            return _answer_with_openai(prompt, model=self.llm_model)

        return _answer_no_llm(prompt)
