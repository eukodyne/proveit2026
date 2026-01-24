#!/bin/bash
set -e

CHECKPOINT_DIR="/workspace/whisper_checkpoint"
ENGINE_DIR="/workspace/whisper_engine"
WHISPER_EXAMPLES="/app/tensorrt_llm/examples/models/core/whisper"

# Always ensure tokenizer files exist (needed for inference)
ASSETS_DIR="$WHISPER_EXAMPLES/assets"
if [ ! -f "$ASSETS_DIR/multilingual.tiktoken" ]; then
    echo "Downloading Whisper tokenizer files..."
    mkdir -p "$ASSETS_DIR"
    curl -sL -o "$ASSETS_DIR/multilingual.tiktoken" "https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/multilingual.tiktoken"
    curl -sL -o "$ASSETS_DIR/gpt2.tiktoken" "https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/gpt2.tiktoken"
    curl -sL -o "$ASSETS_DIR/mel_filters.npz" "https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/mel_filters.npz"
    echo "Tokenizer files downloaded."
fi

# Check if engine already exists (persistent volume)
if [ -d "$ENGINE_DIR/encoder" ] && [ -d "$ENGINE_DIR/decoder" ]; then
    echo "TensorRT engines already exist, skipping conversion..."
else
    echo "Converting Whisper model to TensorRT-LLM format..."

    cd "$WHISPER_EXAMPLES"

    # Download model and tokenizer files if not exists
    if [ ! -f "assets/large-v3-turbo.pt" ]; then
        echo "Downloading Whisper large-v3-turbo model..."
        mkdir -p assets
        python3 -c "import whisper; whisper.load_model('large-v3-turbo', download_root='assets')"
    fi

    # Download tokenizer files if not exists
    if [ ! -f "assets/multilingual.tiktoken" ]; then
        echo "Downloading Whisper tokenizer files..."
        python3 -c "
import os
import urllib.request

assets_dir = 'assets'
os.makedirs(assets_dir, exist_ok=True)

# Download tokenizer files from OpenAI
base_url = 'https://raw.githubusercontent.com/openai/whisper/main/whisper/assets'
files = ['multilingual.tiktoken', 'gpt2.tiktoken', 'mel_filters.npz']

for f in files:
    url = f'{base_url}/{f}'
    dest = os.path.join(assets_dir, f)
    if not os.path.exists(dest):
        print(f'Downloading {f}...')
        urllib.request.urlretrieve(url, dest)
        print(f'Downloaded {f}')
"
    fi

    # Convert checkpoint (FP16 - no quantization for SM120/Blackwell compatibility)
    python3 convert_checkpoint.py \
        --model_name large-v3-turbo \
        --output_dir "$CHECKPOINT_DIR"

    echo "Building encoder engine..."
    trtllm-build \
        --checkpoint_dir "$CHECKPOINT_DIR/encoder" \
        --output_dir "$ENGINE_DIR/encoder" \
        --max_batch_size 8 \
        --gemm_plugin float16 \
        --bert_attention_plugin float16

    echo "Building decoder engine..."
    trtllm-build \
        --checkpoint_dir "$CHECKPOINT_DIR/decoder" \
        --output_dir "$ENGINE_DIR/decoder" \
        --max_batch_size 8 \
        --max_beam_width 4 \
        --max_input_len 14 \
        --max_seq_len 114 \
        --gemm_plugin float16 \
        --gpt_attention_plugin float16

    echo "TensorRT engines built successfully!"
fi

cd /workspace
exec "$@"
