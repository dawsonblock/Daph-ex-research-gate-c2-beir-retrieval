#!/usr/bin/env bash
set -euo pipefail

# Fast llama-server startup for DAPH experiments
# Keeps Liquid model resident in GPU memory for the entire experiment.
#
# Usage:
#   ./tools/colab/start_llama_fast.sh <model_path> [reasoning_budget] [parallel] [port]
#
# Defaults:
#   reasoning_budget = 0
#   parallel = 8
#   port = 8080

MODEL_PATH="${1:?Usage: $0 <model_path> [reasoning_budget] [parallel] [port]}"
REASONING_BUDGET="${2:-0}"
PARALLEL="${3:-8}"
PORT="${4:-8080}"

LLAMA_SERVER="/content/llama.cpp/build/bin/llama-server"

if [ ! -f "$LLAMA_SERVER" ]; then
    echo "ERROR: $LLAMA_SERVER not found. Build llama.cpp first."
    exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    exit 1
fi

echo "[start_llama_fast] Starting llama-server"
echo "  model:           $MODEL_PATH"
echo "  reasoning_budget: $REASONING_BUDGET"
echo "  parallel slots:  $PARALLEL"
echo "  ctx-size:        4096"
echo "  batch-size:      2048"
echo "  ubatch-size:     512"
echo "  gpu layers:      99 (all)"
echo "  flash attention: on"
echo "  port:            $PORT"

exec "$LLAMA_SERVER" \
    -m "$MODEL_PATH" \
    -ngl 99 \
    -fa on \
    --reasoning-budget "$REASONING_BUDGET" \
    --parallel "$PARALLEL" \
    --cont-batching \
    --ctx-size 4096 \
    --batch-size 2048 \
    --ubatch-size 512 \
    --temp 0.0 \
    --seed 42 \
    --threads 4 \
    --no-mmap \
    --host 127.0.0.1 \
    --port "$PORT"
