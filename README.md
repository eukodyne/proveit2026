# Manufacturing RAG System

A FastAPI-based Retrieval-Augmented Generation system for manufacturing SOPs using Milvus vector database and LLM inference.

## API Documentation (Swagger UI)

Interactive API documentation is available via Swagger UI:

- **Swagger UI**: http://localhost:8080/docs
- **ReDoc**: http://localhost:8080/redoc

Use Swagger UI to explore endpoints, view request/response schemas, and test API calls directly in the browser.

## API Endpoints

### RAG API (Port 8080)

**List All Documents**
```bash
curl http://localhost:8080/documents
# Returns: document_id, filename, machine_id, chunk_count for each document
```

**List Documents by Machine ID**
```bash
curl http://localhost:8080/documents?machine_id=CNC-001
```

**Delete All Documents for a Machine**
```bash
curl -X DELETE http://localhost:8080/documents/CNC-001
# Deletes all documents and their chunks for the specified machine_id
```

**Delete a Specific Document by Document ID**
```bash
curl -X DELETE http://localhost:8080/documents/CNC-001/7d33bf69-b055-45c7-93e1-c74ecc091aae
# Deletes all chunks belonging to the specified document (by document_id UUID)
```

**Ingest a Document**
```bash
curl -X POST http://localhost:8080/ingest \
  -F "machine_id=CNC-001" \
  -F "file=@document.pdf"
# Returns: document_id (UUID), filename, chunks_ingested, machine_id
```

**Query Documents**
```bash
curl -X POST http://localhost:8080/query \
  -F "user_query=How do I calibrate the spindle?" \
  -F "machine_id=CNC-001"
```

**OpenAI-Compatible RAG Chat Endpoint**
```bash
# Non-streaming
curl -X POST http://localhost:8080/rag/no-stream/CNC-001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-oss-20b",
    "messages": [{"role": "user", "content": "How do I calibrate the spindle?"}]
  }'

# Streaming
curl -X POST http://localhost:8080/rag/stream/CNC-001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-oss-20b",
    "messages": [{"role": "user", "content": "How do I calibrate the spindle?"}]
  }'
```

### LLM Server (Port 8000)

**Chat Completion**
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-oss-20b",
    "messages": [{"role": "user", "content": "Hello, how are you?"}],
    "max_tokens": 100
  }'
```

**List Models**
```bash
curl http://localhost:8000/v1/models
```

### n8n Workflow Automation (Port 5678)

**Web Interface**
```
http://localhost:5678
```

**Webhook Endpoint (example)**
```bash
curl -X POST http://localhost:5678/webhook/your-webhook-id \
  -H "Content-Type: application/json" \
  -d '{"message": "trigger workflow"}'
```

### Milvus Vector Database (Port 19530)

**Health Check**
```bash
curl http://localhost:9091/healthz
```

## Deployment Setup

```bash
docker compose up -d
```

This will start:
- **Milvus** - Vector database for document embeddings
- **LLM Server** - GPT-OSS-20B via vLLM with MXFP4 quantization
- **RAG API** - FastAPI application for document ingestion and querying
- **n8n** - Workflow automation platform

### Verify Services Are Running

```bash
docker compose ps
docker compose logs -f
```

### Test Endpoints

```bash
# Test RAG API
curl http://localhost:8080/docs

# Test LLM Server
curl http://localhost:8000/v1/models

# Test n8n
curl http://localhost:5678
```

## Verifying Docker Auto-Start After Reboot

To ensure the Docker service and containers start automatically after a server reboot, run these commands:

### Check Docker Service Status

```bash
# Check if Docker service is enabled to start on boot
sudo systemctl is-enabled docker

# Check current Docker service status
sudo systemctl status docker
```

### Enable Docker Service (if not enabled)

```bash
sudo systemctl enable docker
sudo systemctl enable containerd
```

### Verify Container Restart Policy

All services in this configuration have `restart: always` set, which means they will automatically restart when the Docker daemon starts. Verify with:

```bash
# Check restart policy for all containers
docker inspect --format '{{.Name}}: {{.HostConfig.RestartPolicy.Name}}' $(docker ps -aq)
```

### Test Auto-Start After Reboot

```bash
# Reboot the server
sudo reboot

# After reboot, verify containers are running
docker compose ps
```

### Troubleshooting Auto-Start

If containers don't start after reboot:

```bash
# Check Docker daemon logs
sudo journalctl -u docker --since "boot"

# Manually start containers if needed
cd /home/devpartner/manufacturing-rag
docker compose up -d
```

## Exposed Ports

| Service | Port | Description |
|---------|------|-------------|
| RAG API | 8080 | FastAPI application with /ingest and /query endpoints |
| LLM Server | 8000 | OpenAI-compatible chat completions API |
| n8n | 5678 | Workflow automation web interface |
| Milvus | 19530 | Vector database |
| Milvus Metrics | 9091 | Prometheus metrics and health checks |

## Architecture

```
                    ┌─────────────────┐
                    │   n8n (5678)    │
                    │   Workflows     │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐    ┌─────────────────┐
                    │   RAG API       │───▶│   LLM Server    │
                    │   (8080)        │    │   (8000)         │
                    │   FastAPI       │    │   vLLM           │
                    └────────┬────────┘    └─────────────────┘
                             │
                    ┌────────▼────────┐
                    │   Milvus        │
                    │   (19530)       │
                    │   Vector DB     │
                    └─────────────────┘
```
