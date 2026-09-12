# RAG Pipeline — Document Q&A System

A production-style Retrieval-Augmented Generation (RAG) pipeline that ingests documents, builds a searchable knowledge base, and answers natural language questions with cited sources.

## Architecture

```
┌──────────────┐    ┌──────────┐    ┌────────────┐    ┌──────────────┐
│  Documents   │───▶│  Chunk   │───▶│   Embed    │───▶│ Vector Store │
│  (PDF/text)  │    │          │    │            │    │  (ChromaDB)  │
└──────────────┘    └──────────┘    └────────────┘    └──────┬───────┘
                                                            │
┌──────────────┐    ┌──────────┐    ┌────────────┐          │
│   Answer +   │◀───│   LLM    │◀───│  Retrieve  │◀─────────┘
│  Citations   │    │(Generate)│    │  Top-K     │◀── User Query
└──────────────┘    └──────────┘    └────────────┘
```

## Features

- **Document ingestion** — Load PDFs and text files, split into semantically meaningful chunks
- **Embedding & indexing** — Convert chunks to vectors using sentence-transformers, store in ChromaDB
- **Semantic retrieval** — Find relevant context via cosine similarity search
- **Grounded generation** — Generate answers with source citations using LLM APIs
- **Evaluation** — Measure retrieval recall and answer quality on a benchmark test set

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector Store | ChromaDB |
| Framework | LangChain |
| LLM | OpenAI GPT-4 / Anthropic Claude (configurable) |
| Document Loading | PyPDF, LangChain document loaders |

## Project Structure

```
├── src/rag_pipeline/       # Python package — the RAG pipeline
│   ├── embeddings.py       # Embedding model utilities
│   ├── chunking.py         # Document loading & chunking strategies
│   ├── vectorstore.py      # ChromaDB vector store operations
│   ├── retriever.py        # Retrieval logic (semantic + hybrid search)
│   ├── generator.py        # LLM-based answer generation
│   └── pipeline.py         # End-to-end pipeline orchestration
├── notebooks/              # Exploration & demo notebooks
│   └── 01_embeddings_exploration.ipynb
├── data/sample/            # Sample documents for testing
├── tests/                  # Evaluation & unit tests
├── docs/                   # Architecture docs & design decisions
└── pyproject.toml          # Project config & dependencies
```

## Quick Start

```bash
# Clone the repo
git clone https://github.com/nischal-acharya27/rag-pipeline.git
cd rag-pipeline

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Copy and configure environment variables
cp .env.example .env
# Edit .env with your API key (OpenAI or Anthropic)

# Run the demo notebook
jupyter notebook notebooks/01_embeddings_exploration.ipynb
```

## Learning Journey

This project was built incrementally as a learning exercise in RAG pipelines, progressing from basic embeddings to a production-quality system:

1. **Embeddings & semantic search** — Understanding vector representations of text
2. **Document chunking** — Splitting documents for optimal retrieval
3. **Vector databases** — Persistent storage and efficient similarity search
4. **End-to-end pipeline** — Connecting retrieval to LLM generation
5. **Evaluation** — Measuring and improving retrieval and answer quality
6. **Production patterns** — Hybrid search, reranking, and deployment

## License

MIT
