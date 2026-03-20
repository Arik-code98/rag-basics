# RAG Pipeline — Retrieval-Augmented Generation

A minimal, end-to-end Retrieval-Augmented Generation (RAG) pipeline built from scratch. It reads a text document, stores it as vector embeddings in ChromaDB, and uses semantic search to retrieve relevant context before passing it to LLaMA 3.3 70B (via Groq API) to generate a grounded, accurate answer.

---

## How It Works

```
Document (.txt)
      |
      v
  Text Chunking
      |
      v
Sentence Embeddings  <-- SentenceTransformer (all-MiniLM-L6-v2)
      |
      v
  ChromaDB Storage
      |
      v
  User Question  -->  Question Embedding  -->  Semantic Search
                                                      |
                                                      v
                                            Top-K Relevant Chunks
                                                      |
                                                      v
                                          Prompt = Context + Question
                                                      |
                                                      v
                                       LLaMA 3.3 70B (Groq API)
                                                      |
                                                      v
                                            Grounded Answer
```

---

## Tech Stack

- **SentenceTransformers** - Generates semantic vector embeddings (`all-MiniLM-L6-v2`)
- **ChromaDB** - In-memory vector database for storing and querying embeddings
- **Groq API** - LLM inference backend
- **LLaMA 3.3 70B Versatile** - The underlying large language model
- **python-dotenv** - Secure API key management

---

## Project Structure

```
project/
├── main.py          # RAG pipeline script
├── example.txt      # Source document to query against
├── .env             # Environment variables (not committed)
└── requirements.txt # Project dependencies
```

---

## Setup and Installation

**1. Clone the repository**

```bash
git clone https://github.com/Arik-code98/rag-basics
cd rag-basics
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Configure environment variables**

Create a `.env` file in the root directory:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free API key from [https://console.groq.com](https://console.groq.com).

**4. Add your document**

Place your source text in `example.txt`. The file is split into chunks by double newlines (`\n\n`), so structure your text with paragraph breaks between sections.

**5. Run the pipeline**

```bash
python main.py
```

---

## Pipeline Breakdown

### 1. Document Loading and Chunking

```python
with open("example.txt", "r") as f:
    text = f.read()
    chunks = text.split("\n\n")
```

The document is split on double newlines, treating each paragraph as an independent chunk. This is a simple but effective chunking strategy for structured text.

### 2. Embedding Generation

```python
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunks)
```

Each chunk is converted into a dense vector representation using the `all-MiniLM-L6-v2` model — a lightweight, fast model well-suited for semantic similarity tasks.

### 3. Vector Storage

```python
collection.add(documents=chunks, embeddings=embeddings.tolist(), ids=[...])
```

Chunks and their embeddings are stored in a ChromaDB collection in memory, ready for semantic search.

### 4. Semantic Retrieval

```python
results = collection.query(query_embeddings=question_embedding, n_results=2)
```

The user's question is embedded using the same model, and ChromaDB retrieves the top-2 most semantically similar chunks.

### 5. Grounded Answer Generation

```python
prompt = f"Answer the question based on the context below.\n\nContext:\n{context}\n\nQuestion:\n{question}"
```

The retrieved chunks are injected into the prompt as context, and the LLM generates an answer grounded in the document — rather than relying solely on its training data.

---

## Key Concepts Explored

- Text chunking and preprocessing strategies
- Semantic embeddings with Sentence Transformers
- Vector database storage and similarity search with ChromaDB
- Context-aware prompt engineering
- Grounded LLM responses to reduce hallucination
- The full RAG retrieval loop

---

## Limitations and Improvements

- **Fixed chunk IDs**: The current implementation uses hardcoded IDs (`chunk1`, `chunk2`, `chunk3`). For documents with dynamic chunk counts, generate IDs programmatically.
- **In-memory storage**: ChromaDB runs in memory and does not persist between runs. For production use, configure a persistent ChromaDB client or use a hosted vector database.
- **Basic chunking**: Splitting on `\n\n` works for structured documents but may not be ideal for all formats. Consider using a sliding window or token-based chunking for better coverage.
- **No reranking**: Retrieved chunks are returned by similarity score only. Adding a reranker (e.g., a cross-encoder) can improve answer quality significantly.

---

## Notes

- The model used is `llama-3.3-70b-versatile`. This can be swapped for any model available on the Groq API.
- The embedding model `all-MiniLM-L6-v2` runs locally and requires no API key.
- This project is intentionally minimal — designed for learning the fundamentals of RAG before moving to frameworks like LangChain or LlamaIndex.