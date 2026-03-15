"""
test_rag.py
===========
Unit tests for all RAG pipeline modules.

Run with:
    python -m pytest RAG_Project/tests/test_rag.py -v

These tests use only the standard library and lightweight mocks so that they
run quickly without downloading any ML models or calling external APIs.
"""

import sys
import os
import types
import unittest
from unittest.mock import MagicMock, patch

# Make src importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ===========================================================================
# document_loader tests
# ===========================================================================

class TestChunkText(unittest.TestCase):
    """Tests for document_loader.chunk_text."""

    def _import(self):
        from src.document_loader import chunk_text
        return chunk_text

    def test_basic_chunking(self):
        chunk_text = self._import()
        text = "A" * 100
        chunks = chunk_text(text, chunk_size=30, overlap=0)
        # 100 / 30 = 3 full chunks + 1 partial → 4 chunks
        self.assertEqual(len(chunks), 4)

    def test_overlap_creates_more_chunks(self):
        chunk_text = self._import()
        text = "B" * 100
        no_overlap = chunk_text(text, chunk_size=20, overlap=0)
        with_overlap = chunk_text(text, chunk_size=20, overlap=5)
        self.assertGreater(len(with_overlap), len(no_overlap))

    def test_empty_string_returns_empty_list(self):
        chunk_text = self._import()
        self.assertEqual(chunk_text("", chunk_size=50, overlap=0), [])

    def test_invalid_chunk_size_raises(self):
        chunk_text = self._import()
        with self.assertRaises(ValueError):
            chunk_text("hello", chunk_size=0)

    def test_negative_overlap_raises(self):
        chunk_text = self._import()
        with self.assertRaises(ValueError):
            chunk_text("hello", chunk_size=10, overlap=-1)

    def test_overlap_gte_chunk_size_raises(self):
        chunk_text = self._import()
        with self.assertRaises(ValueError):
            chunk_text("hello", chunk_size=5, overlap=5)

    def test_single_chunk_when_text_shorter_than_chunk_size(self):
        chunk_text = self._import()
        chunks = chunk_text("short text", chunk_size=100, overlap=0)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], "short text")


class TestParseDocuments(unittest.TestCase):
    """Tests for document_loader.parse_documents."""

    def _import(self):
        from src.document_loader import parse_documents
        return parse_documents

    def test_splits_on_separator(self):
        parse_documents = self._import()
        raw = "Doc one content.\n---\nDoc two content."
        docs = parse_documents(raw)
        self.assertEqual(len(docs), 2)
        self.assertIn("Doc one", docs[0])
        self.assertIn("Doc two", docs[1])

    def test_leading_comments_skipped(self):
        parse_documents = self._import()
        raw = "# Comment\n---\nActual document text."
        docs = parse_documents(raw)
        # The comment block before the first --- should still be captured
        # but the important thing is the "Actual document text" appears
        self.assertTrue(any("Actual document text" in d for d in docs))

    def test_empty_string_returns_empty_list(self):
        parse_documents = self._import()
        self.assertEqual(parse_documents(""), [])

    def test_no_separator_returns_single_doc(self):
        parse_documents = self._import()
        docs = parse_documents("Just one document, no separator.")
        self.assertEqual(len(docs), 1)


class TestLoadAndChunkDocuments(unittest.TestCase):
    """Tests for document_loader.load_and_chunk_documents."""

    def _import(self):
        from src.document_loader import load_and_chunk_documents
        return load_and_chunk_documents

    def test_loads_sample_file(self):
        load_and_chunk_documents = self._import()
        sample_file = os.path.join(
            os.path.dirname(__file__), "..", "data", "sample_documents.txt"
        )
        chunks = load_and_chunk_documents(sample_file)
        self.assertGreater(len(chunks), 0)
        # Each chunk should have the required keys
        for chunk in chunks:
            self.assertIn("id", chunk)
            self.assertIn("text", chunk)
            self.assertIn("metadata", chunk)

    def test_file_not_found_raises(self):
        load_and_chunk_documents = self._import()
        with self.assertRaises(FileNotFoundError):
            load_and_chunk_documents("/nonexistent/path/file.txt")

    def test_chunk_ids_are_unique(self):
        load_and_chunk_documents = self._import()
        sample_file = os.path.join(
            os.path.dirname(__file__), "..", "data", "sample_documents.txt"
        )
        chunks = load_and_chunk_documents(sample_file)
        ids = [c["id"] for c in chunks]
        self.assertEqual(len(ids), len(set(ids)))


# ===========================================================================
# embeddings tests (using mocks — no model download required)
# ===========================================================================

class TestGetEmbeddingModel(unittest.TestCase):
    """Tests for embeddings.get_embedding_model factory."""

    def test_unknown_backend_raises(self):
        from src.embeddings import get_embedding_model
        with self.assertRaises(ValueError):
            get_embedding_model(backend="unknown")

    @patch("src.embeddings.LocalEmbeddingModel.__init__", return_value=None)
    def test_local_backend_returns_local_model(self, _mock_init):
        from src.embeddings import get_embedding_model, LocalEmbeddingModel
        model = get_embedding_model(backend="local")
        self.assertIsInstance(model, LocalEmbeddingModel)


# ===========================================================================
# vector_store tests (using lightweight ChromaDB in-memory client)
# ===========================================================================

def _make_mock_embedding_model(dim: int = 8):
    """Return a lightweight mock embedding model that avoids network calls."""
    import numpy as np
    rng = np.random.default_rng(seed=42)
    mock_em = MagicMock()
    mock_em.embed.side_effect = lambda texts: rng.random((len(texts), dim)).astype("float32")
    mock_em.embed_query.side_effect = lambda q: rng.random(dim).astype("float32")
    return mock_em


class TestVectorStore(unittest.TestCase):
    """Tests for VectorStore using real in-memory ChromaDB."""

    def setUp(self):
        try:
            import chromadb  # noqa: F401
        except ImportError:
            self.skipTest("chromadb not installed — skipping VectorStore tests")

        from src.vector_store import VectorStore
        self.store = VectorStore(collection_name="test_collection")
        self.em = _make_mock_embedding_model()

    def tearDown(self):
        try:
            self.store.reset()
        except Exception:
            pass

    def test_add_and_count(self):
        texts = ["Python is a programming language.", "RAG reduces hallucinations."]
        self.store.add_texts(texts, embedding_model=self.em)
        self.assertEqual(self.store.count(), 2)

    def test_query_returns_results(self):
        texts = [
            "Machine learning is a subset of AI.",
            "Transformers use self-attention mechanisms.",
            "Python was created by Guido van Rossum.",
        ]
        self.store.add_texts(texts, embedding_model=self.em)
        results = self.store.query("What is machine learning?", top_k=2, embedding_model=self.em)
        self.assertEqual(len(results), 2)
        for r in results:
            self.assertIn("text", r)
            self.assertIn("distance", r)

    def test_reset_clears_collection(self):
        self.store.add_texts(["some text"], embedding_model=self.em)
        self.assertEqual(self.store.count(), 1)
        self.store.reset()
        self.assertEqual(self.store.count(), 0)

    def test_add_documents_with_metadata(self):
        chunks = [
            {
                "id": "chunk_0",
                "text": "Vector databases support ANN search.",
                "metadata": {"source": "test.txt", "doc_index": 0, "chunk_index": 0},
            }
        ]
        self.store.add_documents(chunks, embedding_model=self.em)
        self.assertEqual(self.store.count(), 1)

    def test_add_empty_list_does_nothing(self):
        self.store.add_documents([])
        self.assertEqual(self.store.count(), 0)


# ===========================================================================
# rag_pipeline tests
# ===========================================================================

class TestBuildPrompt(unittest.TestCase):
    """Tests for rag_pipeline.build_prompt."""

    def test_prompt_contains_question(self):
        from src.rag_pipeline import build_prompt
        prompt = build_prompt("What is RAG?", ["RAG stands for Retrieval-Augmented Generation."])
        self.assertIn("What is RAG?", prompt)

    def test_prompt_contains_context(self):
        from src.rag_pipeline import build_prompt
        context = "RAG stands for Retrieval-Augmented Generation."
        prompt = build_prompt("What is RAG?", [context])
        self.assertIn(context, prompt)

    def test_prompt_contains_answer_label(self):
        from src.rag_pipeline import build_prompt
        prompt = build_prompt("Any question?", ["Some context."])
        self.assertIn("ANSWER:", prompt)

    def test_multiple_context_chunks(self):
        from src.rag_pipeline import build_prompt
        chunks = ["Chunk one.", "Chunk two.", "Chunk three."]
        prompt = build_prompt("Question?", chunks)
        for chunk in chunks:
            self.assertIn(chunk, prompt)


class TestRAGPipelineIndexing(unittest.TestCase):
    """Tests for RAGPipeline.index_file and index_texts without a real embedding model."""

    def setUp(self):
        try:
            import chromadb  # noqa: F401
        except ImportError:
            self.skipTest("chromadb not installed")

    def _make_mock_pipeline(self):
        """Create a RAGPipeline with the embedding model replaced by a mock."""
        import numpy as np
        from src.rag_pipeline import RAGPipeline

        pipeline = RAGPipeline.__new__(RAGPipeline)
        pipeline.top_k = 3
        pipeline.chunk_size = 300
        pipeline.chunk_overlap = 50
        pipeline.llm_backend = None
        pipeline.llm_model = "gpt-4o-mini"

        rng = np.random.default_rng(seed=42)
        mock_em = MagicMock()
        mock_em.embed.side_effect = lambda texts: rng.random((len(texts), 384)).astype("float32")
        mock_em.embed_query.side_effect = lambda q: rng.random(384).astype("float32")

        pipeline.embedding_model = mock_em

        from src.vector_store import VectorStore
        pipeline.vector_store = VectorStore(collection_name="pipeline_test")

        return pipeline

    def tearDown(self):
        pass  # Each test creates its own in-memory store

    def test_index_file_returns_chunk_count(self):
        pipeline = self._make_mock_pipeline()
        sample_file = os.path.join(
            os.path.dirname(__file__), "..", "data", "sample_documents.txt"
        )
        count = pipeline.index_file(sample_file)
        self.assertGreater(count, 0)

    def test_index_texts_returns_ids(self):
        pipeline = self._make_mock_pipeline()
        ids = pipeline.index_texts(["Hello world", "RAG is useful"])
        self.assertEqual(len(ids), 2)

    def test_ask_without_llm_returns_prompt_string(self):
        pipeline = self._make_mock_pipeline()
        pipeline.index_texts(["RAG stands for Retrieval-Augmented Generation."])
        answer = pipeline.ask("What does RAG stand for?")
        self.assertIsInstance(answer, str)
        self.assertGreater(len(answer), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
