# RAG Learning Project 🚀

A beginner-friendly, **zero-dependency** implementation of the
**Retrieval-Augmented Generation (RAG)** pattern, written entirely in Python's
standard library.  The goal is to help you understand *how* RAG works by
reading, running, and extending the code.

---

## What is RAG?

**RAG (Retrieval-Augmented Generation)** is a technique that makes a language
model smarter by giving it relevant facts *at query time*, rather than
expecting it to memorise everything during training.

```
User query
    │
    ▼
┌──────────┐     similarity search     ┌──────────────┐
│  Embed   │ ────────────────────────► │  Vector      │
│  query   │                           │  Index       │
└──────────┘                           └──────┬───────┘
                                              │ top-k chunks
                                              ▼
                                       ┌──────────────┐
                                       │  Prompt      │
                                       │  = context   │
                                       │  + question  │
                                       └──────┬───────┘
                                              │
                                              ▼
                                       ┌──────────────┐
                                       │  LLM         │
                                       │  generates   │
                                       │  answer      │
                                       └──────────────┘
```

### Why RAG?

| Problem without RAG | How RAG helps |
|---|---|
| LLM knowledge has a training cut-off | Retrieve from an up-to-date knowledge base |
| LLM hallucinates facts | Ground the answer in retrieved documents |
| Fine-tuning is expensive | Just update the knowledge base |

---

## Project Structure

```
rag_project/
├── rag.py               ← Core RAG pipeline (read this first!)
├── test_rag.py          ← Unit tests for every pipeline step
├── requirements.txt     ← No mandatory dependencies
└── knowledge_base/      ← Plain-text documents the RAG searches
    ├── python_basics.txt
    ├── machine_learning.txt
    ├── rag_concepts.txt
    └── large_language_models.txt
```

---

## Six Steps of the Pipeline

`rag.py` implements each step as a clearly named function:

| # | Step | Function(s) | What happens |
|---|---|---|---|
| 1 | **Load** | `load_documents()` | Read `.txt` files from the knowledge base |
| 2 | **Chunk** | `chunk_documents()` | Split long docs into small, overlapping passages |
| 3 | **Embed** | `tokenize()`, `compute_tf()`, `compute_idf()`, `embed_text()` | Turn each chunk into a TF-IDF vector |
| 4 | **Index** | `build_index()` | Store all vectors in memory |
| 5 | **Retrieve** | `retrieve()`, `cosine_similarity()` | Find the top-k most relevant chunks |
| 6 | **Generate** | `generate_answer()` | Build a grounded prompt and produce an answer |

---

## Quick Start

### 1 – Run the interactive demo

No installation required.

```bash
cd rag_project
python rag.py
```

Example session:

```
You: What is RAG?
...
Answer:
[Mock answer – wire in a real LLM to get a proper response]
Based on 'rag_concepts.txt': RAG stands for Retrieval-Augmented Generation ...
```

### 2 – Run the tests

```bash
python test_rag.py
```

Expected output:

```
Running 14 tests …

  PASS test_tokenize
  PASS test_tokenize_empty
  ...
  PASS test_rag_end_to_end
  PASS test_rag_unknown_topic

Results: 14 passed, 0 failed out of 14 tests.
All tests passed!
```

### 3 – Use the `RAG` class in your own script

```python
from rag import RAG

bot = RAG()                            # builds index on first call
answer = bot.ask("What is cosine similarity?")
print(answer)
```

---

## Extending the Project

### Add your own documents

Drop any `.txt` file into `knowledge_base/` and restart the pipeline.  Each
line should be one sentence (the chunker splits on newlines).

### Wire in a real LLM (OpenAI example)

Replace the body of `generate_answer()` in `rag.py` with:

```python
import openai

def generate_answer(query: str, retrieved_chunks: list[dict]) -> str:
    context = "\n\n".join(c["content"] for c in retrieved_chunks)
    prompt = (
        "Use ONLY the context below to answer the question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\nAnswer:"
    )
    client = openai.OpenAI()           # reads OPENAI_API_KEY from env
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content
```

Install the dependency first:

```bash
pip install openai
```

### Swap TF-IDF embeddings for a neural embedding model

Replace `embed_text()` calls with a sentence-transformer model:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
vector = model.encode("your text here").tolist()
```

Install the dependency first:

```bash
pip install sentence-transformers
```

### Use a production vector store (FAISS example)

```python
import faiss, numpy as np

dim = len(vocab)
index = faiss.IndexFlatIP(dim)         # inner-product (cosine on normalised vecs)
matrix = np.array(chunk_vectors, dtype="float32")
index.add(matrix)

# Query
q_vec = np.array([query_vector], dtype="float32")
scores, indices = index.search(q_vec, k=3)
```

Install the dependency first:

```bash
pip install faiss-cpu
```

---

## Key Concepts Glossary

| Term | Definition |
|---|---|
| **Chunk** | A short passage (typically 100–500 tokens) extracted from a document |
| **Embedding** | A dense numeric vector that captures the meaning of a piece of text |
| **TF-IDF** | A statistical method to quantify how important a word is to a document |
| **Cosine similarity** | A measure of similarity between two vectors based on the angle between them |
| **Vector store** | A database optimised for storing and searching embedding vectors |
| **Context window** | The maximum number of tokens an LLM can process in one call |
| **Hallucination** | When an LLM generates confident but factually incorrect information |
| **Grounding** | Tying an LLM's answer to retrieved source documents to reduce hallucination |

---

## Learning Path

1. **Read `rag.py`** top-to-bottom – every function has explanatory comments.
2. **Run the demo** and ask a few questions to see the pipeline in action.
3. **Run the tests** to understand the expected behaviour of each component.
4. **Add a document** to `knowledge_base/` and verify the new content is retrieved.
5. **Wire in a real LLM** following the OpenAI example above.
6. **Replace TF-IDF** with a neural embedding model for much better retrieval quality.
7. **Add a vector store** like FAISS or Chroma to scale to millions of documents.
