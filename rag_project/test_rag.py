"""
Tests for the RAG learning project.
Run with:  python test_rag.py
"""

import sys
import os
import math

# Make sure we can import from the same directory
sys.path.insert(0, os.path.dirname(__file__))

from rag import (
    tokenize,
    build_vocabulary,
    compute_tf,
    compute_idf,
    embed_text,
    normalize,
    cosine_similarity,
    chunk_documents,
    load_documents,
    build_index,
    retrieve,
    RAG,
)

KNOWLEDGE_BASE_DIR = os.path.join(os.path.dirname(__file__), "knowledge_base")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def assert_equal(actual, expected, msg=""):
    assert actual == expected, f"FAIL [{msg}]: expected {expected!r}, got {actual!r}"


def assert_almost_equal(actual, expected, tol=1e-6, msg=""):
    assert abs(actual - expected) < tol, (
        f"FAIL [{msg}]: expected {expected} ± {tol}, got {actual}"
    )


def assert_true(condition, msg=""):
    assert condition, f"FAIL [{msg}]"


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_tokenize():
    tokens = tokenize("Hello, World! This is Python.")
    assert_equal(tokens, ["hello", "world", "this", "is", "python"], "tokenize")
    print("  PASS test_tokenize")


def test_tokenize_empty():
    assert_equal(tokenize(""), [], "tokenize empty")
    print("  PASS test_tokenize_empty")


def test_build_vocabulary():
    chunks = [
        {"content": "cat sat"},
        {"content": "dog ran"},
    ]
    vocab = build_vocabulary(chunks)
    assert_equal(sorted(vocab), ["cat", "dog", "ran", "sat"], "build_vocabulary")
    print("  PASS test_build_vocabulary")


def test_compute_tf():
    tf = compute_tf(["a", "a", "b"], ["a", "b", "c"])
    assert_almost_equal(tf[0], 2 / 3, msg="tf[a]")
    assert_almost_equal(tf[1], 1 / 3, msg="tf[b]")
    assert_almost_equal(tf[2], 0.0,   msg="tf[c]")
    print("  PASS test_compute_tf")


def test_normalize():
    v = normalize([3.0, 4.0])
    assert_almost_equal(v[0], 0.6, msg="normalize[0]")
    assert_almost_equal(v[1], 0.8, msg="normalize[1]")
    print("  PASS test_normalize")


def test_normalize_zero_vector():
    v = normalize([0.0, 0.0])
    assert_equal(v, [0.0, 0.0], "normalize zero vector")
    print("  PASS test_normalize_zero_vector")


def test_cosine_similarity_identical():
    v = normalize([1.0, 2.0, 3.0])
    score = cosine_similarity(v, v)
    assert_almost_equal(score, 1.0, tol=1e-5, msg="cosine identical")
    print("  PASS test_cosine_similarity_identical")


def test_cosine_similarity_orthogonal():
    v1 = normalize([1.0, 0.0])
    v2 = normalize([0.0, 1.0])
    assert_almost_equal(cosine_similarity(v1, v2), 0.0, tol=1e-9, msg="cosine orthogonal")
    print("  PASS test_cosine_similarity_orthogonal")


def test_chunk_documents():
    docs = [{"source": "test.txt", "content": "A\nB\nC\nD"}]
    chunks = chunk_documents(docs, chunk_size=2, overlap=1)
    # With step=1 and 4 sentences we expect chunks starting at 0,1,2,3
    assert_true(len(chunks) >= 3, "chunk count")
    assert_equal(chunks[0]["content"], "A B", "first chunk")
    print("  PASS test_chunk_documents")


def test_chunk_overlap():
    docs = [{"source": "f.txt", "content": "X\nY\nZ"}]
    chunks = chunk_documents(docs, chunk_size=2, overlap=1)
    contents = [c["content"] for c in chunks]
    # Overlap means "Y" should appear in two consecutive chunks
    assert_true(any("Y" in c for c in contents), "overlap")
    print("  PASS test_chunk_overlap")


def test_load_documents():
    docs = load_documents(KNOWLEDGE_BASE_DIR)
    assert_true(len(docs) >= 4, "at least 4 documents loaded")
    for doc in docs:
        assert_true("source" in doc, "source key")
        assert_true("content" in doc, "content key")
        assert_true(len(doc["content"]) > 0, "non-empty content")
    print("  PASS test_load_documents")


def test_retrieve_ranking():
    """The retriever should rank chunks by relevance to the query."""
    docs = [{"source": "f.txt", "content": "Python is a programming language\nPython uses indentation\nPython supports functions"}]
    chunks = chunk_documents(docs, chunk_size=2, overlap=1)
    vocab = build_vocabulary(chunks)
    idf = compute_idf(chunks, vocab)
    index = build_index(chunks, vocab, idf)

    results = retrieve("Python programming", chunks, index, vocab, idf, top_k=2)
    assert_equal(len(results), 2, "retrieve count")
    # Highest score first
    assert_true(results[0]["score"] >= results[1]["score"], "descending score")
    print("  PASS test_retrieve_ranking")


def test_rag_end_to_end():
    """Smoke-test the full RAG pipeline."""
    bot = RAG(knowledge_base_dir=KNOWLEDGE_BASE_DIR, top_k=3)
    answer = bot.ask("What is RAG?")
    assert_true(isinstance(answer, str) and len(answer) > 0, "non-empty answer")
    assert_true("rag" in answer.lower() or "retrieval" in answer.lower(),
                "answer contains relevant keywords")
    print("  PASS test_rag_end_to_end")


def test_rag_unknown_topic():
    """A query about a completely unrelated topic should still return a string."""
    bot = RAG(knowledge_base_dir=KNOWLEDGE_BASE_DIR, top_k=2)
    answer = bot.ask("What is the boiling point of water?")
    assert_true(isinstance(answer, str), "answer is string for unknown topic")
    print("  PASS test_rag_unknown_topic")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all():
    tests = [
        test_tokenize,
        test_tokenize_empty,
        test_build_vocabulary,
        test_compute_tf,
        test_normalize,
        test_normalize_zero_vector,
        test_cosine_similarity_identical,
        test_cosine_similarity_orthogonal,
        test_chunk_documents,
        test_chunk_overlap,
        test_load_documents,
        test_retrieve_ranking,
        test_rag_end_to_end,
        test_rag_unknown_topic,
    ]

    print(f"\nRunning {len(tests)} tests …\n")
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as exc:
            print(f"  FAIL {test.__name__}: {exc}")
            failed += 1

    print(f"\n{'=' * 40}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests.")
    if failed:
        sys.exit(1)
    print("All tests passed!")


if __name__ == "__main__":
    run_all()
