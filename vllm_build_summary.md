# vLLM Build Summary for Dell Pro Max GB10 (NVIDIA Spark)

## Problem

vLLM was failing to start with:
```
ImportError: libtorch_cuda.so: cannot open shared object file: No such file or directory
```

**Root cause**: PyTorch wheels from standard PyPI indexes (`cu128`, `cu130`) don't include ARM64 CUDA builds. On ARM64 systems, pip silently falls back to CPU-only wheels, resulting in missing `libtorch_cuda.so`.

## Hardware

- **Device**: Dell Pro Max GB10 / NVIDIA DGX Spark
- **Architecture**: ARM64 (aarch64)
- **GPU**: NVIDIA GB10 (Blackwell, SM 12.1 / cuda capability 12.1)
- **OS**: Ubuntu 24.04 (Linux 6.14.0-1015-nvidia)

## Solution

Based on the community project [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker), we used:

1. **CUDA 13.1 base image** (not 12.8 or 13.0)
2. **`uv` package manager** with `--torch-backend=auto` flag
3. **vLLM nightly wheels** from `wheels.vllm.ai/nightly/cu130`
4. **System CUDA libraries** via apt (`libcudnn9-cuda-13`, `libnccl2`)

## Working Dockerfile

```dockerfile
# vLLM 0.13+ for NVIDIA Spark / Dell Pro Max GB10 (ARM64 + Blackwell SM120)
# Based on eugr/spark-vllm-docker approach

FROM nvidia/cuda:13.1.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# Build parallelism settings (prevent OOM)
ARG BUILD_JOBS=8
ENV MAX_JOBS=${BUILD_JOBS}
ENV CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS}

# PyTorch/CUDA settings for Blackwell SM120
ENV TORCH_CUDA_ARCH_LIST=12.1a
ENV TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas

# uv package manager settings
ENV UV_SYSTEM_PYTHON=1
ENV UV_BREAK_SYSTEM_PACKAGES=1

# Install runtime dependencies including CUDA libs
RUN apt update && apt upgrade -y \
    && apt install -y --allow-change-held-packages --no-install-recommends \
        python3 python3-pip python3-dev \
        git wget curl jq \
        libcudnn9-cuda-13 \
        libnccl-dev libnccl2 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install uv

# Install vLLM with auto PyTorch backend (handles CUDA deps automatically)
RUN uv pip install -U vllm \
    --torch-backend=auto \
    --extra-index-url https://wheels.vllm.ai/nightly/cu130

# Install FlashInfer for attention optimization
RUN uv pip install flashinfer-python -U --no-deps --index-url https://flashinfer.ai/whl && \
    uv pip install flashinfer-cubin --index-url https://flashinfer.ai/whl && \
    uv pip install flashinfer-jit-cache --index-url https://flashinfer.ai/whl/cu130

# Version verification
RUN python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.version.cuda}')" && \
    python3 -c "import vllm; print(f'vLLM {vllm.__version__}')"

ENTRYPOINT ["vllm"]
```

## Key Insights

### Why Standard PyTorch Wheels Don't Work on ARM64

| Index URL | ARM64 CUDA Support |
|-----------|-------------------|
| `https://download.pytorch.org/whl/cu128` | No (x86_64 only) |
| `https://download.pytorch.org/whl/cu130` | No (x86_64 only) |
| `uv pip --torch-backend=auto` | Yes (resolves correctly) |

### Critical Components

1. **`uv` package manager**: The `--torch-backend=auto` flag intelligently resolves PyTorch with CUDA for ARM64
2. **CUDA 13.1 base**: Required for Blackwell (SM 12.0/12.1) support
3. **System CUDA libs**: `libcudnn9-cuda-13` and `libnccl2` from apt provide native ARM64 CUDA libraries
4. **Nightly wheels**: `wheels.vllm.ai/nightly/cu130` has ARM64 builds

### Resulting Versions

| Component | Version |
|-----------|---------|
| PyTorch | 2.9.1+cu130 |
| CUDA | 13.0 |
| vLLM | 0.14.0rc1 (nightly) |
| FlashInfer | 0.5.3+cu130 |
| Triton | 3.5.1 |

## docker-compose.yml Configuration

```yaml
services:
  llm-server:
    image: vllm-blackwell-native:0.13.0
    container_name: gpt-server
    runtime: nvidia
    restart: always
    ipc: host
    ulimits:
      memlock: -1
      stack: 67108864
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - TORCH_CUDA_ARCH_LIST="12.0"
    volumes:
      - /home/devmaster/.cache/huggingface:/root/.cache/huggingface
    ports:
      - "8000:8000"
    command: >
      serve YOUR_MODEL_NAME
      --max-model-len 131072
      --gpu-memory-utilization 0.70
      --trust-remote-code
```

**Note**: Remove explicit `--quantization` flag - vLLM auto-detects from model config.

## Build Commands

```bash
# Build the image
docker build -f Dockerfile.vllm_upgrade -t vllm-blackwell-native:0.13.0 .

# Start the service
docker compose up -d llm-server

# Monitor logs
docker logs -f gpt-server
```

## References

- [vLLM GitHub Issue #31128](https://github.com/vllm-project/vllm/issues/31128) - vLLM on NVIDIA Spark
- [NVIDIA Forum: Run vLLM in Spark](https://forums.developer.nvidia.com/t/run-vllm-in-spark/348862/9)
- [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) - Community Docker setup

## Troubleshooting

### "Unknown quantization method: nvfp4"
The nightly vLLM renamed quantization methods. Remove `--quantization` flag and let vLLM auto-detect from model config.

### PyTorch CUDA version is None
You're using CPU-only wheels. Ensure you're using `uv pip` with `--torch-backend=auto`.

### Triton errors with gpt-oss models
Build Triton from main branch if you encounter `cutlass_moe_mm_sm100` errors:
```dockerfile
RUN pip uninstall triton -y && \
    git clone --depth 1 https://github.com/triton-lang/triton.git /opt/triton && \
    cd /opt/triton && pip install .
```

---
*Last updated: 2026-01-06*
