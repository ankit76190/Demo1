"""
advanced_rag.py — Intermediate Example
=======================================
Full RAG pipeline: retrieve context, build a prompt, and generate an answer.

Supports two LLM modes:
  1. No LLM (default) — prints the augmented prompt so you can paste it into
     any chat interface manually.
  2. OpenAI GPT       — set OPENAI_API_KEY and pass --llm openai.

Usage:
    # No LLM (prints the prompt):
    python examples/advanced_rag.py

    # With OpenAI:
    OPENAI_API_KEY=sk-... python examples/advanced_rag.py --llm openai
"""

import argparse
import sys
import os

# Make the src package importable when running from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.rag_pipeline import RAGPipeline


DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "sample_documents.txt")

SAMPLE_QUESTIONS = [
    "What is Retrieval-Augmented Generation and how does it reduce hallucinations?",
    "What are the key differences between supervised and unsupervised learning?",
    "What is ChromaDB and why is it useful for prototyping?",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Advanced RAG demo")
    parser.add_argument(
        "--llm",
        choices=["none", "openai"],
        default="none",
        help="LLM backend to use for answer generation (default: none → print prompt)",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model name when --llm openai is used",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of context chunks to retrieve per question (default: 3)",
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Ask a custom question instead of running the built-in samples",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    llm_backend = None if args.llm == "none" else args.llm

    # ------------------------------------------------------------------
    # Build the pipeline
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Building RAG pipeline …")
    print(f"  Embedding backend : local (all-MiniLM-L6-v2)")
    print(f"  LLM backend       : {args.llm}")
    print(f"  top_k             : {args.top_k}")
    print("=" * 60 + "\n")

    pipeline = RAGPipeline(
        embedding_backend="local",
        llm_backend=llm_backend,
        llm_model=args.model,
        top_k=args.top_k,
    )

    # ------------------------------------------------------------------
    # Index the knowledge base
    # ------------------------------------------------------------------
    print("Indexing knowledge base …")
    n_chunks = pipeline.index_file(DATA_FILE)
    print(f"  {n_chunks} chunks indexed.\n")

    # ------------------------------------------------------------------
    # Answer questions
    # ------------------------------------------------------------------
    questions = [args.question] if args.question else SAMPLE_QUESTIONS

    for i, question in enumerate(questions, start=1):
        print(f"{'=' * 60}")
        print(f"Question {i}: {question}")
        print(f"{'=' * 60}")

        # Show retrieved context
        results = pipeline.retrieve(question)
        print(f"\nTop {len(results)} retrieved chunks:")
        for rank, result in enumerate(results, start=1):
            print(
                f"  [{rank}] distance={result['distance']:.4f} | "
                f"{result['text'][:100]}…"
            )

        # Generate answer
        print(f"\nAnswer:")
        answer = pipeline.ask(question)
        print(answer)
        print()


if __name__ == "__main__":
    main()
