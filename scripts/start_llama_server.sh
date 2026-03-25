#!/usr/bin/env bash
# =============================================================================
# Start llama-server optimized for Jetson Orin NX 16GB + JetPack 5.1
# =============================================================================
set -euo pipefail

LLAMA_BIN="${HOME}/llama.cpp/build/bin/llama-server"
MODEL="${HOME}/models/qwen2.5-7b-instruct-q4_k_m.gguf"
GRAMMAR="$(dirname "$0")/../grammars/commands.gbnf"
HOST="0.0.0.0"
PORT="8080"

# Verify binary exists
if [ ! -f "$LLAMA_BIN" ]; then
    echo "ERROR: llama-server not found at $LLAMA_BIN"
    echo "Run scripts/setup_jetson_jp5.sh first."
    exit 1
fi

# Verify model exists
if [ ! -f "$MODEL" ]; then
    echo "ERROR: Model not found at $MODEL"
    echo "Run scripts/download_model.sh first."
    exit 1
fi

# Ensure MAXN power mode and clocks locked
sudo nvpmodel -m 0 2>/dev/null || true
sudo jetson_clocks 2>/dev/null || true

echo "Starting llama-server..."
echo "  Model:   $MODEL"
echo "  Host:    $HOST:$PORT"
echo "  Context: 4096 tokens"
echo ""

# JetPack 5.1 specific flags:
#   --ngl 99          : Offload all layers to GPU
#   -fa               : Flash attention (supported on compute_87)
#   -c 4096           : Context length (keep small for speed + RAM)
#   --jinja           : Enable Jinja template for Qwen native function calling
#   -t 4              : 4 CPU threads for prompt processing (save CPU for app)
#   --no-mmap         : Disable mmap — use direct read (more predictable on ARM)
#
# NOTE: Do NOT use --mlock on Jetson — unified memory means mlock
#       prevents the GPU from accessing the model weights.

exec "$LLAMA_BIN" \
    --model "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --jinja \
    -fa \
    --ngl 99 \
    -c 4096 \
    -t 4 \
    --no-mmap \
    --log-disable
