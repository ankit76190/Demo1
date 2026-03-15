"""
basic_rag.py — Beginner Example
================================
This script demonstrates the **retrieval** half of RAG without requiring any
paid LLM API. It:

1. Loads and chunks the sample knowledge base.
2. Embeds all chunks with a free local model (sentence-transformers).
3. Stores embeddings in an in-memory ChromaDB collection.
4. Asks several questions and prints the top matching passages.

Run with:
    python examples/basic_rag.py
"""

import sys
import os

# Make the src package importable when running from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.document_loader import load_and_chunk_documents
from src.embeddings import LocalEmbeddingModel
from src.vector_store import VectorStore


DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "sample_documents.txt")


def main():
    # ------------------------------------------------------------------
    # Step 1: Load and chunk the knowledge base
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 1 — Loading and chunking documents")
    print("=" * 60)

    chunks = load_and_chunk_documents(DATA_FILE, chunk_size=400, overlap=60)
    print(f"  Total chunks created: {len(chunks)}")
    print(f"  First chunk preview : {chunks[0]['text'][:80]}...\n")

    # ------------------------------------------------------------------
    # Step 2: Create embeddings
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 2 — Loading embedding model (all-MiniLM-L6-v2)")
    print("         (This may download ~80 MB the first time)")
    print("=" * 60)

    embedding_model = LocalEmbeddingModel()
    print("  Embedding model ready.\n")

    # ------------------------------------------------------------------
    # Step 3: Store embeddings in ChromaDB
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 3 — Embedding and storing chunks in ChromaDB")
    print("=" * 60)

    store = VectorStore(collection_name="basic_demo")
    store.add_documents(chunks, embedding_model=embedding_model)
    print(f"  Chunks stored in vector store: {store.count()}\n")

    # ------------------------------------------------------------------
    # Step 4: Retrieve relevant chunks for sample questions
    # ------------------------------------------------------------------
    questions = [
        "What is Retrieval-Augmented Generation?",
        "How do neural networks work?",
        "What embedding models are available?",
        "Which Python library is used for HTTP requests?",
    ]

    print("=" * 60)
    print("Step 4 — Semantic search (retrieval demo)")
    print("=" * 60)

    for question in questions:
        print(f"\nQuestion: {question}")
        print("-" * 50)
        results = store.query(question, top_k=2, embedding_model=embedding_model)
        for rank, result in enumerate(results, start=1):
            print(f"  [{rank}] (distance={result['distance']:.4f}) {result['text'][:120]}...")

    print("\n" + "=" * 60)
    print("Done! To get LLM-generated answers, run examples/advanced_rag.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
