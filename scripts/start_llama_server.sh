#!/usr/bin/env bash
# =============================================================================
# Start llama-server optimized for Jetson Orin NX 16GB + JetPack 5.1
# =============================================================================
set -euo pipefail

LLAMA_BIN="/opt/llama.cpp/build/bin/llama-server"
MODEL_FOLDER="${HOME}/models"
MODEL="${MODEL_FOLDER}/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
# MODEL="qwen3-0.6b-q4_k_m.gguf"
GRAMMAR="$(dirname "$0")/../grammars/commands.gbnf"
HOST="0.0.0.0"
PORT="8080"

# Verify binary exists
if [ ! -f "$LLAMA_BIN" ]; then
    echo "ERROR: llama-server not found at $LLAMA_BIN"
    echo "Run scripts/setup_jetson_jp5.sh first."
    exit 1
fi

# Verify model exists — auto-download if missing
if [ ! -f "$MODEL" ]; then
    echo "Model not found at $MODEL — downloading Qwen3-0.6B GGUF..."
    huggingface-cli download Qwen/Qwen3-0.6B-GGUF \
        qwen3-0.6b-q4_k_m.gguf \
        --local-dir "${HOME}/models"
fi

# Ensure MAXN power mode and clocks locked
# sudo nvpmodel -m 0 2>/dev/null || true
# sudo jetson_clocks 2>/dev/null || true

echo "Starting llama-server..."
echo "  Model:   $MODEL"
echo "  Host:    $HOST:$PORT"
echo "  Context: 8192 tokens"
echo ""

# JetPack 5.1 specific flags:
#   --ngl 99          : Offload all layers to GPU
#   -fa               : Flash attention (supported on compute_87)
#   -c 8192           : Larger context — Qwen3-0.6B is small enough to allow it
#   --jinja           : Enable Jinja template for Qwen native function calling
#   -t 6              : More CPU threads — 0.6B needs less GPU babysitting
#   --no-mmap         : Disable mmap — use direct read (more predictable on ARM)
#
# NOTE: Do NOT use --mlock on Jetson — unified memory means mlock
#       prevents the GPU from accessing the model weights.

exec "$LLAMA_BIN" \
    --model "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --jinja \
    --flash-attn on \
    -ngl 99 \
    -c 8192 \
    -t 6 \
    --no-mmap \
    --log-disable
