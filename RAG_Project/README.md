# RAG Learning Project 🚀

A hands-on project to learn **Retrieval-Augmented Generation (RAG)** from scratch using Python.

---

## 📖 What is RAG?

**Retrieval-Augmented Generation (RAG)** is an AI architecture that enhances a Large Language Model (LLM) by giving it access to an external knowledge base at query time.

Instead of relying solely on what an LLM "memorized" during training, RAG:
1. **Retrieves** relevant documents from a knowledge base
2. **Augments** the prompt with that retrieved context
3. **Generates** an answer grounded in real, up-to-date information

```
User Question
      │
      ▼
┌─────────────┐      ┌───────────────────┐      ┌──────────────┐
│  Embedding  │─────▶│   Vector Store    │─────▶│  Retrieved   │
│   Model     │      │  (ChromaDB/FAISS) │      │  Documents   │
└─────────────┘      └───────────────────┘      └──────┬───────┘
                                                        │
                                                        ▼
                                               ┌──────────────────┐
                                               │  Prompt =        │
                                               │  Context + Query │
                                               └────────┬─────────┘
                                                        │
                                                        ▼
                                               ┌──────────────────┐
                                               │       LLM        │
                                               │  (GPT / local)   │
                                               └────────┬─────────┘
                                                        │
                                                        ▼
                                                    Answer
```

---

## 🗂️ Project Structure

```
RAG_Project/
├── README.md                  ← You are here
├── requirements.txt           ← Python dependencies
│
├── data/
│   └── sample_documents.txt   ← Knowledge base (edit with your own docs!)
│
├── src/
│   ├── __init__.py
│   ├── document_loader.py     ← Step 1: Load & chunk documents
│   ├── embeddings.py          ← Step 2: Convert text to vectors
│   ├── vector_store.py        ← Step 3: Store & search vectors
│   └── rag_pipeline.py        ← Step 4: Full RAG pipeline
│
├── examples/
│   ├── basic_rag.py           ← Beginner: run RAG without an LLM
│   └── advanced_rag.py        ← Intermediate: RAG with OpenAI / local LLM
│
└── tests/
    └── test_rag.py            ← Unit tests for every module
```

---

## 🛠️ Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. (Optional) Set your OpenAI API key

Only needed for the `advanced_rag.py` example. Skip if you want the free,
local-model path.

```bash
export OPENAI_API_KEY="sk-..."
```

---

## 🏃 Quick Start

### Beginner — run RAG without a paid LLM

```bash
python examples/basic_rag.py
```

This demonstrates the **retrieval** half of RAG: embed documents, store them
in ChromaDB, ask a question, and see the top matching passages.

### Intermediate — full pipeline with an LLM

```bash
python examples/advanced_rag.py
```

Feeds the retrieved context into an LLM (OpenAI GPT or a free HuggingFace
model) and prints a generated answer.

---

## 📚 Learning Path

| Step | File | Concept |
|------|------|---------|
| 1 | `src/document_loader.py` | Chunking — why and how to split large documents |
| 2 | `src/embeddings.py` | Embeddings — turning text into numbers |
| 3 | `src/vector_store.py` | Vector databases — similarity search at scale |
| 4 | `src/rag_pipeline.py` | Prompt engineering — combining context with a query |
| 5 | `examples/basic_rag.py` | End-to-end retrieval demo |
| 6 | `examples/advanced_rag.py` | End-to-end generation demo |

---

## 🔑 Key Concepts

### Chunking
Large documents must be split into smaller pieces so that each chunk fits
within the embedding model's token limit and so that retrieved context stays
focused.

### Embeddings
A numerical vector that captures the *semantic meaning* of a piece of text.
Sentences with similar meanings have vectors that are close together in
high-dimensional space (measured by cosine similarity).

### Vector Store
A database optimised for **approximate nearest-neighbour (ANN)** search.
Given a query vector it returns the *k* stored vectors that are most similar.

### Retrieval
Convert the user's question to an embedding, search the vector store for the
most similar document chunks, and return them as context.

### Augmented Generation
Prepend the retrieved chunks to the user's question in the LLM prompt so the
model has concrete facts to draw on when it generates its answer.

---

## 🌱 Ideas for Further Learning

1. **Swap the embedding model** — try `text-embedding-3-small` (OpenAI) vs
   `all-MiniLM-L6-v2` (HuggingFace sentence-transformers) and compare quality.
2. **Try different chunking strategies** — fixed-size, sentence-aware,
   semantic, recursive character splitting.
3. **Add a re-ranker** — after initial retrieval, re-score the top-k chunks
   with a cross-encoder for higher precision.
4. **Persistent vector store** — save the ChromaDB collection to disk so you
   don't need to re-embed on every run.
5. **Evaluate your RAG pipeline** — use `ragas` or `TruLens` to measure
   faithfulness, answer relevancy, and context precision.
6. **Build a UI** — wrap the pipeline in a `Streamlit` or `Gradio` app.
