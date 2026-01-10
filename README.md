# Manufacturing RAG System

A FastAPI-based Retrieval-Augmented Generation system for manufacturing SOPs using Milvus vector database and LLM inference.

## Deployment Setup

This system uses Docker Compose with override files to support multiple LLM backends.

### Step 1: Choose Your LLM Backend

Copy one of the LLM configuration files to `docker-compose.override.yml`:

**Option A: Nemotron (llama.cpp)**
```bash
cp docker-compose.nemotron.yml docker-compose.override.yml
```

**Option B: GPT-OSS-20B (vLLM with MXFP4 quantization)**
```bash
cp docker-compose.gptoss20b.yml docker-compose.override.yml
```

### Step 2: Start the Services

```bash
docker compose up -d
```

### Step 3: Verify Services Are Running

```bash
docker compose ps
docker compose logs -f
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
cd /home/devmaster/manufacturing-rag/manufacturing-rag
docker compose up -d
```

## Exposed Ports

| Service | Port | Description |
|---------|------|-------------|
| RAG API | 8080 | FastAPI application |
| LLM Server | 8000 | OpenAI-compatible API |
| Milvus | 19530 | Vector database |
| Milvus Metrics | 9091 | Prometheus metrics |
