"""
RAG (Retrieval-Augmented Generation) Learning Project
======================================================
This module implements a complete RAG pipeline from scratch using only
Python's standard library.  It is intentionally written for
learning, so every step is broken out into a clearly named function with
comments that explain *why* things are done, not just *what* is done.

RAG pipeline overview
---------------------
1. LOAD      – read documents from the knowledge base
2. CHUNK     – split long documents into smaller, overlapping passages
3. EMBED     – turn each chunk into a vector (using TF-IDF here)
4. INDEX     – store the vectors so we can search them
5. RETRIEVE  – given a user query, find the most relevant chunks
6. GENERATE  – build a prompt from the retrieved context and answer

Run this file directly to try an interactive Q&A session:
    python rag.py

Or import the RAG class and use it programmatically:
    from rag import RAG
    bot = RAG()
    answer = bot.ask("What is cosine similarity?")
    print(answer)
"""

import os
import math
import string
import textwrap
from collections import Counter


# ---------------------------------------------------------------------------
# Step 1 – LOAD: read every .txt file in the knowledge base
# ---------------------------------------------------------------------------

def load_documents(knowledge_base_dir: str) -> list[dict]:
    """Load all .txt documents from *knowledge_base_dir*.

    Returns a list of dicts, each with keys:
        - 'source'  : the file name (for citation)
        - 'content' : the full text of the file
    """
    documents = []
    for filename in sorted(os.listdir(knowledge_base_dir)):
        if filename.endswith(".txt"):
            filepath = os.path.join(knowledge_base_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read().strip()
            documents.append({"source": filename, "content": content})
            print(f"  Loaded: {filename} ({len(content)} chars)")
    return documents


# ---------------------------------------------------------------------------
# Step 2 – CHUNK: split documents into small, overlapping passages
# ---------------------------------------------------------------------------

def chunk_documents(documents: list[dict],
                    chunk_size: int = 2,
                    overlap: int = 1) -> list[dict]:
    """Split each document into chunks of *chunk_size* sentences.

    Why chunking?
    ------------
    A language model has a limited context window.  Passing a whole book to
    it is impossible.  Instead we index small passages so the retriever can
    pinpoint the exact sentences that answer a question.

    Why overlap?
    ------------
    Answers sometimes span sentence boundaries.  Overlapping chunks make it
    less likely that a relevant passage is cut in half.

    Returns a list of chunk dicts, each with keys:
        - 'source'  : originating file name
        - 'content' : the chunk text
    """
    chunks = []
    for doc in documents:
        # Split on newlines – each line in our knowledge base is one sentence.
        sentences = [s.strip() for s in doc["content"].splitlines() if s.strip()]

        step = max(1, chunk_size - overlap)
        for start in range(0, len(sentences), step):
            chunk_sentences = sentences[start: start + chunk_size]
            chunk_text = " ".join(chunk_sentences)
            chunks.append({"source": doc["source"], "content": chunk_text})

    print(f"\n  Created {len(chunks)} chunks from {len(documents)} documents.")
    return chunks


# ---------------------------------------------------------------------------
# Step 3 – EMBED: convert text to vectors with TF-IDF
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    """Lowercase, remove punctuation, and split into words."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.split()


def build_vocabulary(chunks: list[dict]) -> list[str]:
    """Collect the set of all unique tokens across all chunks."""
    vocab = set()
    for chunk in chunks:
        vocab.update(tokenize(chunk["content"]))
    return sorted(vocab)


def compute_tf(tokens: list[str], vocab: list[str]) -> list[float]:
    """Term-frequency vector: fraction of chunk tokens that equal each vocab word."""
    counts = Counter(tokens)
    total = len(tokens) if tokens else 1
    return [counts.get(word, 0) / total for word in vocab]


def compute_idf(chunks: list[dict], vocab: list[str]) -> list[float]:
    """Inverse-document-frequency: down-weight common words across all chunks.

    IDF(word) = log( (1 + N) / (1 + df) ) + 1
    where N = total chunks, df = number of chunks containing the word.
    """
    n = len(chunks)
    idf_values = []
    for word in vocab:
        df = sum(1 for chunk in chunks if word in tokenize(chunk["content"]))
        idf_values.append(math.log((1 + n) / (1 + df)) + 1)
    return idf_values


def embed_text(tokens: list[str],
               vocab: list[str],
               idf: list[float]) -> list[float]:
    """Compute the TF-IDF embedding for a list of *tokens*."""
    tf = compute_tf(tokens, vocab)
    return [t * i for t, i in zip(tf, idf)]


def normalize(vector: list[float]) -> list[float]:
    """L2-normalize *vector* so cosine similarity is just a dot product."""
    magnitude = math.sqrt(sum(x * x for x in vector))
    if magnitude == 0:
        return vector
    return [x / magnitude for x in vector]


# ---------------------------------------------------------------------------
# Step 4 – INDEX: store the embedded chunks
# ---------------------------------------------------------------------------

def build_index(chunks: list[dict],
                vocab: list[str],
                idf: list[float]) -> list[list[float]]:
    """Embed every chunk and return a list of normalized TF-IDF vectors.

    The index is a simple list-of-lists here (a plain 'flat' index).
    In production you would use a dedicated vector store such as FAISS,
    Chroma, or Pinecone for fast approximate nearest-neighbor search.
    """
    index = []
    for chunk in chunks:
        tokens = tokenize(chunk["content"])
        vector = embed_text(tokens, vocab, idf)
        index.append(normalize(vector))
    return index


# ---------------------------------------------------------------------------
# Step 5 – RETRIEVE: find the top-k most relevant chunks
# ---------------------------------------------------------------------------

def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Dot product of two pre-normalized vectors = cosine similarity."""
    return sum(a * b for a, b in zip(vec_a, vec_b))


def retrieve(query: str,
             chunks: list[dict],
             index: list[list[float]],
             vocab: list[str],
             idf: list[float],
             top_k: int = 3) -> list[dict]:
    """Return the *top_k* chunks most similar to *query*.

    Why cosine similarity?
    ----------------------
    We care about the *direction* of the vector (which topics are present),
    not its length (how long the text is).  Cosine similarity measures the
    angle between two vectors, making it length-independent.
    """
    query_tokens = tokenize(query)
    query_vector = normalize(embed_text(query_tokens, vocab, idf))

    scored = []
    for i, chunk_vector in enumerate(index):
        score = cosine_similarity(query_vector, chunk_vector)
        scored.append((score, i))

    scored.sort(reverse=True)

    results = []
    for score, idx in scored[:top_k]:
        results.append({
            "score": round(score, 4),
            "source": chunks[idx]["source"],
            "content": chunks[idx]["content"],
        })
    return results


# ---------------------------------------------------------------------------
# Step 6 – GENERATE: build a grounded answer from the retrieved context
# ---------------------------------------------------------------------------

def generate_answer(query: str, retrieved_chunks: list[dict]) -> str:
    """Produce an answer grounded in the retrieved context.

    In a real RAG system this function would call an LLM API (e.g. OpenAI,
    Anthropic, or a local Ollama model) and pass the context in the prompt.

    Here we use a *mock generator* that:
      1. Shows the prompt that would be sent to an LLM.
      2. Returns the most relevant retrieved passage as the answer.

    This lets you run and study the full RAG pipeline without needing an
    API key.  See the README for instructions on wiring in a real LLM.
    """
    context = "\n\n".join(
        f"[Source: {c['source']} | Score: {c['score']}]\n{c['content']}"
        for c in retrieved_chunks
    )

    # --- This is the prompt you would send to an LLM ---
    prompt = (
        "You are a helpful assistant.  Use ONLY the context below to answer "
        "the question.  If the context does not contain enough information, "
        "say so.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )

    separator = "-" * 60
    print(f"\n{separator}")
    print("PROMPT SENT TO LLM (mock mode – showing prompt only):")
    print(separator)
    print(textwrap.indent(prompt, "  "))
    print(separator)

    # Mock answer: use the highest-scoring chunk as the response
    if retrieved_chunks:
        best = retrieved_chunks[0]
        answer = (
            f"[Mock answer – wire in a real LLM to get a proper response]\n"
            f"Based on '{best['source']}': {best['content']}"
        )
    else:
        answer = "No relevant context found for your question."

    return answer


# ---------------------------------------------------------------------------
# High-level RAG class that ties everything together
# ---------------------------------------------------------------------------

class RAG:
    """A self-contained RAG pipeline.

    Usage
    -----
    >>> bot = RAG()
    >>> print(bot.ask("What is cosine similarity?"))
    """

    def __init__(self,
                 knowledge_base_dir: str | None = None,
                 chunk_size: int = 2,
                 overlap: int = 1,
                 top_k: int = 3):
        if knowledge_base_dir is None:
            knowledge_base_dir = os.path.join(
                os.path.dirname(__file__), "knowledge_base"
            )

        print("=" * 60)
        print("RAG PIPELINE – INITIALIZING")
        print("=" * 60)

        print("\n[1/4] Loading documents …")
        documents = load_documents(knowledge_base_dir)

        print("\n[2/4] Chunking documents …")
        self.chunks = chunk_documents(documents, chunk_size, overlap)

        print("\n[3/4] Building vocabulary and IDF table …")
        self.vocab = build_vocabulary(self.chunks)
        self.idf = compute_idf(self.chunks, self.vocab)
        print(f"  Vocabulary size: {len(self.vocab)} unique tokens")

        print("\n[4/4] Embedding chunks and building index …")
        self.index = build_index(self.chunks, self.vocab, self.idf)
        print(f"  Index contains {len(self.index)} vectors "
              f"of dimension {len(self.vocab)}.")

        self.top_k = top_k
        print("\nRAG pipeline ready!  Call bot.ask(<question>) to query.\n")

    def ask(self, query: str) -> str:
        """Run the full RAG pipeline for *query* and return an answer."""
        print(f"\n{'=' * 60}")
        print(f"QUERY: {query}")
        print("=" * 60)

        print("\n[Retrieve] Searching for relevant chunks …")
        results = retrieve(query, self.chunks, self.index,
                           self.vocab, self.idf, self.top_k)

        print(f"  Top-{self.top_k} results:")
        for i, r in enumerate(results, 1):
            print(f"  {i}. score={r['score']}  source={r['source']}")
            print(f"     {r['content'][:80]}…")

        print("\n[Generate] Composing answer …")
        return generate_answer(query, results)


# ---------------------------------------------------------------------------
# Interactive demo
# ---------------------------------------------------------------------------

def interactive_demo():
    """Run an interactive Q&A session in the terminal."""
    bot = RAG()

    print("Type your question and press Enter.  Type 'quit' to exit.\n")
    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        answer = bot.ask(query)
        print(f"\nAnswer:\n{answer}\n")


if __name__ == "__main__":
    interactive_demo()
