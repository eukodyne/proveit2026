# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Manufacturing RAG (Retrieval-Augmented Generation) system - a FastAPI-based application for ingesting manufacturing SOPs (PDF, TXT, HTML) and answering questions using vector search (Milvus) and LLM generation (vLLM).

## Architecture

- **app/main.py**: FastAPI application with `/ingest` and `/query` endpoints
- **Milvus**: Vector database for storing document embeddings (port 19530)
- **vLLM**: LLM server for text generation (port 8000)
- **Embedding Model**: `all-MiniLM-L6-v2` (384 dimensions)

## Technology Stack

- **Language**: Python 3.12
- **Framework**: FastAPI + Uvicorn
- **Vector DB**: Milvus (GPU-accelerated, ARM64)
- **LLM Server**: vLLM with OpenAI-compatible API
- **Embedding**: sentence-transformers

## Development Setup (Windows - Anaconda)

```bash
# Create and activate conda environment
conda create -n manufacturing-rag python=3.12
conda activate manufacturing-rag

# Install dependencies
pip install -r app/requirements.txt

# Run the API locally (requires Milvus and vLLM running)
cd app
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

## Environment Variables

| Variable | Local Dev (Windows/Linux) | Docker Container |
|----------|---------------------------|------------------|
| `MILVUS_URL` | `http://localhost:19530` (default) | `http://milvus:19530` |
| `VLLM_URL` | `http://localhost:8000/v1` (default) | `http://llm-server:8000/v1` |

## Docker Deployment (Dell Pro Max GB10)

```bash
# Start all services
docker compose up -d

# View logs
docker compose logs -f rag-api
```

## Commands

```bash
# Run API locally
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload

# Test ingestion
curl -X POST http://localhost:8080/ingest \
  -F "machine_id=CNC-001" \
  -F "file=@document.pdf"

# Test query
curl -X POST http://localhost:8080/query \
  -F "user_query=How do I calibrate the spindle?" \
  -F "machine_id=CNC-001"
```

## Supported File Types

- PDF (`.pdf`)
- Plain text (`.txt`)
- HTML (`.html`, `.htm`)
