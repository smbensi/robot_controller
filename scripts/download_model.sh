#!/usr/bin/env bash
# =============================================================================
# Download the recommended GGUF model for Jetson Orin NX 16GB
# =============================================================================
set -euo pipefail

MODEL_DIR="${HOME}/models"
mkdir -p "$MODEL_DIR"

# Primary recommendation: Qwen2.5-7B-Instruct Q4_K_M (~4.7 GB)
# Best function-calling accuracy at this size tier
MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf"
MODEL_FILE="qwen2.5-7b-instruct-q4_k_m.gguf"

# Alternative: Phi-4-mini 3.8B Q4_K_M (~2.4 GB) — faster, less accurate
# MODEL_URL="https://huggingface.co/microsoft/Phi-4-mini-instruct-GGUF/resolve/main/phi-4-mini-instruct-q4_k_m.gguf"
# MODEL_FILE="phi-4-mini-instruct-q4_k_m.gguf"

if [ -f "${MODEL_DIR}/${MODEL_FILE}" ]; then
    echo "Model already downloaded: ${MODEL_DIR}/${MODEL_FILE}"
    exit 0
fi

echo "Downloading ${MODEL_FILE} to ${MODEL_DIR}..."
echo "This may take a while (~4.7 GB)..."

wget -c -O "${MODEL_DIR}/${MODEL_FILE}" "$MODEL_URL"

echo "Download complete: ${MODEL_DIR}/${MODEL_FILE}"
echo "Verify with: ls -lh ${MODEL_DIR}/${MODEL_FILE}"
