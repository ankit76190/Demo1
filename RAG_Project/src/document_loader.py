"""
document_loader.py
==================
Step 1 of the RAG pipeline: load raw text and split it into manageable chunks.

Why do we chunk?
- Embedding models have a maximum token limit (e.g., 512 tokens).
- Smaller, focused chunks improve retrieval precision.
- We want the retrieved context to be relevant rather than diluted.
"""

import os
import re


def load_text_file(file_path: str) -> str:
    """Read the entire contents of a plain-text file."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Document not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as fh:
        return fh.read()


def parse_documents(raw_text: str) -> list[str]:
    """
    Split a raw text string into individual documents.

    The sample_documents.txt file uses '---' on its own line as a separator.
    Each resulting document has leading/trailing whitespace stripped.
    """
    # Split on lines that contain only '---'
    parts = re.split(r"^\s*---\s*$", raw_text, flags=re.MULTILINE)
    # Drop empty strings and the comment header at the top
    documents = [part.strip() for part in parts if part.strip()]
    return documents


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """
    Split *text* into overlapping windows of *chunk_size* characters.

    Parameters
    ----------
    text:       The full text to split.
    chunk_size: Maximum number of characters per chunk.
    overlap:    Number of characters shared between consecutive chunks.
                Overlap helps retain context across chunk boundaries.

    Returns a list of non-empty chunk strings.

    Example
    -------
    >>> chunks = chunk_text("Hello world, this is a test.", chunk_size=15, overlap=5)
    >>> chunks[0]
    'Hello world, th'
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if overlap < 0:
        raise ValueError("overlap must be a non-negative integer")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap  # advance by (chunk_size - overlap)
    return chunks


def load_and_chunk_documents(
    file_path: str,
    chunk_size: int = 300,
    overlap: int = 50,
) -> list[dict]:
    """
    High-level helper: load a file, parse it into documents, and chunk each one.

    Returns a list of dicts with keys:
        - "id"       : unique string identifier  (e.g. "doc_0_chunk_2")
        - "text"     : the chunk text
        - "metadata" : {"source": file_path, "doc_index": int, "chunk_index": int}
    """
    raw_text = load_text_file(file_path)
    documents = parse_documents(raw_text)

    all_chunks = []
    for doc_idx, doc_text in enumerate(documents):
        chunks = chunk_text(doc_text, chunk_size=chunk_size, overlap=overlap)
        for chunk_idx, chunk in enumerate(chunks):
            all_chunks.append(
                {
                    "id": f"doc_{doc_idx}_chunk_{chunk_idx}",
                    "text": chunk,
                    "metadata": {
                        "source": os.path.basename(file_path),
                        "doc_index": doc_idx,
                        "chunk_index": chunk_idx,
                    },
                }
            )
    return all_chunks
