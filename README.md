# Naive RAG: AI/ML Knowledge Assistant

A Retrieval-Augmented Generation (RAG) application that answers AI/ML questions using content extracted from a reference PDF. It combines semantic retrieval, cross-encoder reranking, and LLM-based answer generation.

## Overview

The application retrieves relevant passages from a local knowledge base and supplies them as context to a large language model (LLM). The model uses the retrieved passages to generate answers and can indicate when the available context does not contain sufficient information.

## Architecture

```text
Source PDF
    |
    v
Document Ingestion (Docling)
    |
    v
Text Chunking
    |
    v
Embedding Generation (all-MiniLM-L6-v2)
    |
    v
Vector Storage (ChromaDB)
    |
    v
Question Embedding
    |
    v
Similarity Retrieval (Top 10)
    |
    v
Cross-Encoder Reranking (Top 5)
    |
    v
Answer Generation (gpt-oss-20b via Groq)
```

## Tech Stack

- **Language:** Python
- **Document extraction:** Docling
- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Vector database:** ChromaDB
- **Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **LLM:** `openai/gpt-oss-20b` via Groq API
- **Configuration:** python-dotenv

## Features

- PDF document extraction
- Text chunking and embedding generation
- Persistent vector storage
- Similarity-based document retrieval
- Cross-encoder reranking
- Context-based answer generation
- Fallback behavior when retrieved context is insufficient

## Project Structure

```text
naive_rag/
├── data/
│   ├── documents/       # Source PDFs (local)
│   ├── extracted/       # Extracted text and chunks
│   ├── embedded/        # Generated embeddings
│   └── chroma_db/       # Persistent vector database
├── scripts/
│   ├── audit_ingestion.py
│   ├── inspect_chunks.py
│   ├── validate_chunks.py
│   └── validate_extraction.py
├── src/
│   ├── ingest.py
│   ├── chunk.py
│   ├── embed.py
│   ├── store.py
│   ├── retrieve.py
│   ├── rerank.py
│   └── generate.py
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Prerequisites

- Python installed on your system
- A Groq API key
- The source PDF available locally

## Setup

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd naive_rag
```

Replace `<your-repository-url>` with your actual GitHub repository URL.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows Command Prompt:

```cmd
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure the API key

Copy `.env.example` to `.env` and set your key:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Never commit the `.env` file or expose your API key.

### 5. Add the source document

Place the reference PDF in:

```text
data/documents/
```

The PDF is excluded from Git by default. Ensure the document is available locally and that you have permission to use it.

## Build the Knowledge Base

Run these commands from the project root, in order:

```bash
python src/ingest.py
python src/chunk.py
python src/embed.py
python src/store.py
```

These steps extract the document, create chunks, generate embeddings, and store them in ChromaDB.

The exact input and output paths must match those configured in the scripts.

## Run the Application

After the knowledge base has been built, start the interactive RAG application:

```bash
python src/generate.py
```

Enter a question in the terminal. Type `exit` to end the session.

The application retrieves candidate passages, reranks them, and sends the selected context to the Groq-hosted model for answer generation.

## Limitations

- Answer quality depends on document extraction, chunking, and retrieval quality.
- Relevant passages may not always appear among the retrieved candidates.
- Reranking can reorder retrieved passages but cannot recover passages that retrieval missed.
- The LLM may produce claims that are not fully supported by the retrieved context.
- The current application uses a terminal interface.
- API availability, rate limits, and usage costs depend on the hosting provider and account.

## Security

- Keep `.env` out of version control.
- Use `.env.example` to document required environment variables without storing secrets.
- Avoid committing generated embeddings, vector database files, or source documents unnecessarily.

## Project Status

A working Naive RAG prototype with document ingestion, chunking, embeddings, vector retrieval, reranking, and LLM-based answer generation.
