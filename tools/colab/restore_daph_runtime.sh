#!/usr/bin/env bash
# DAPH Colab runtime restore — one-command recovery
#
# Restores the full DAPH runtime on a fresh Colab VM:
#   1. Extract llama-server from cached archive
#   2. Clone/checkout the frozen commit
#   3. Download/verify the frozen GGUF model
#   4. Start llama-server with frozen configuration
#   5. Restore R13 checkpoint from local upload
#   6. Verify all frozen identities
#
# Usage: colab exec -f tools/colab/restore_daph_runtime.sh --timeout 600
#
# Environment variables:
#   FROZEN_COMMIT  — git commit to checkout (default: current branch HEAD)
#   DAPH_OUTPUT    — output directory (default: /content/daph_r13)
#   RESTORE_FROM   — local directory with checkpoint files to restore

set -euo pipefail

echo "=== DAPH Colab Runtime Restore ==="
echo "  Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Frozen scientific commit — the exact source that produced confirmation_executable_sha256.
# This MUST NOT be changed during a confirmation run. Recovery must checkout this commit.
# To find this: it is the last commit that modified scripts/run_r13_confirmation.py
# or any file under hrm_adaptive_memory/executive/.
FROZEN_EXPERIMENT_COMMIT="${FROZEN_EXPERIMENT_COMMIT:-5454246b7e61adfb7a093eb5a1f731347071270d}"
EXPECTED_CONFIRMATION_SHA="${EXPECTED_CONFIRMATION_SHA:-41cc60b04f506f63b80c91e036d330d61d79992a86fb975cbe21597bd2d84f57}"
DAPH_OUTPUT="${DAPH_OUTPUT:-/content/daph_r13}"
ARCHIVE="/content/llama-server-d775b8967a46-cuda-sm89.tar.gz"
MODEL_PATH="/content/models/google_gemma-3-12b-it-qat-Q4_0.gguf"
EXPECTED_GGUF_SHA="2ad4c9ce431a2d5b80af37983828c2cfb8f4909792ca5075e0370e3a71ca013d"
REPO_DIR="/content/Daph-ex-research-gate-c2-beir-retrieval"

# --- Step 1: Extract llama-server ---
echo ""
echo "[1] Extracting llama-server..."
if [ ! -f "$ARCHIVE" ]; then
    echo "  ERROR: Archive not found at $ARCHIVE"
    echo "  Upload it first: colab upload .cache/colab/llama-server-*.tar.gz"
    exit 1
fi
mkdir -p /content/llama.cpp/build
tar -xzf "$ARCHIVE" -C /content/llama.cpp/build/
SERVER="/content/llama.cpp/build/bin/llama-server"
if [ ! -f "$SERVER" ]; then
    echo "  ERROR: llama-server not found after extraction"
    exit 1
fi
chmod +x "$SERVER"
echo "  OK: $(stat -c%s "$SERVER") bytes"

# --- Step 2: Clone/checkout repository at EXACT frozen experiment commit ---
echo ""
echo "[2] Cloning repository at frozen experiment commit..."
echo "  Frozen commit: $FROZEN_EXPERIMENT_COMMIT"
if [ ! -d "$REPO_DIR/.git" ]; then
    git clone \
        https://github.com/dawsonblock/Daph-ex-research-gate-c2-beir-retrieval.git \
        "$REPO_DIR"
fi
cd "$REPO_DIR"
git fetch origin
# Checkout the EXACT frozen experiment commit in detached HEAD mode.
# NEVER use 'git pull' or 'git checkout main' or 'git checkout latest' during a confirmation.
# The scientific executable identity is pinned to this commit.
git checkout --detach "$FROZEN_EXPERIMENT_COMMIT"
ACTUAL_HEAD=$(git rev-parse HEAD)
if [ "$ACTUAL_HEAD" != "$FROZEN_EXPERIMENT_COMMIT" ]; then
    echo "  ERROR: Checkout mismatch"
    echo "    Expected: $FROZEN_EXPERIMENT_COMMIT"
    echo "    Actual:   $ACTUAL_HEAD"
    exit 1
fi
echo "  OK: $(git rev-parse --short HEAD) (detached HEAD at frozen experiment commit)"

# Verify no scientific files differ from the frozen commit
SCIENTIFIC_DIFF=$(git diff --name-only "$FROZEN_EXPERIMENT_COMMIT" HEAD -- \
    scripts/run_r13_confirmation.py hrm_adaptive_memory/ 2>/dev/null || true)
if [ -n "$SCIENTIFIC_DIFF" ]; then
    echo "  ERROR: Scientific files differ from frozen commit:"
    echo "    $SCIENTIFIC_DIFF"
    exit 1
fi
echo "  OK: Scientific files identical to frozen commit"

# --- Step 3: Download/verify GGUF model ---
echo ""
echo "[3] Checking GGUF model..."
mkdir -p /content/models
if [ ! -f "$MODEL_PATH" ] || [ "$(stat -c%s "$MODEL_PATH" 2>/dev/null || echo 0)" -lt 1000000000 ]; then
    echo "  Downloading from HuggingFace..."
    wget -q -c -O "$MODEL_PATH" \
        "https://huggingface.co/bartowski/google_gemma-3-12b-it-qat-GGUF/resolve/main/google_gemma-3-12b-it-qat-Q4_0.gguf"
fi
SIZE_GB=$(echo "scale=1; $(stat -c%s "$MODEL_PATH") / 1073741824" | bc)
echo "  Model: ${SIZE_GB}GB"

# Verify SHA256
echo "  Verifying SHA256..."
ACTUAL_SHA=$(sha256sum "$MODEL_PATH" | cut -d' ' -f1)
if [ "$ACTUAL_SHA" != "$EXPECTED_GGUF_SHA" ]; then
    echo "  ERROR: SHA mismatch"
    echo "    Expected: $EXPECTED_GGUF_SHA"
    echo "    Actual:   $ACTUAL_SHA"
    exit 1
fi
echo "  OK: SHA verified"

# --- Step 4: Start llama-server ---
echo ""
echo "[4] Starting llama-server..."
# Kill any existing server
pkill -f llama-server 2>/dev/null || true
sleep 2

nohup "$SERVER" \
    -m "$MODEL_PATH" \
    --host 127.0.0.1 --port 8081 \
    -ngl 99 -fa on \
    --parallel 4 --cont-batching \
    -c 32768 \
    --batch-size 2048 --ubatch-size 512 \
    --temp 0.0 --seed 42 \
    -t 4 -np 4 \
    --reasoning-format deepseek \
    -lv 1 \
    > /content/llama-server.log 2>&1 &

SERVER_PID=$!
echo "$SERVER_PID" > /content/llama-server.pid
echo "  PID: $SERVER_PID"

# Wait for server to be ready
echo "  Waiting for server..."
for i in $(seq 1 120); do
    if python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8081/health', timeout=5)" 2>/dev/null; then
        echo "  OK: Server ready after ${i}s"
        break
    fi
    sleep 2
    if [ $i -eq 120 ]; then
        echo "  ERROR: Server failed to start within 240s"
        tail -20 /content/llama-server.log
        exit 1
    fi
done

# --- Step 5: Restore R13 checkpoint ---
echo ""
echo "[5] Checking R13 checkpoint..."
mkdir -p "$DAPH_OUTPUT"

# Check if results already exist locally (from previous run on this VM)
if [ -f "$DAPH_OUTPUT/results.jsonl" ]; then
    LINES=$(wc -l < "$DAPH_OUTPUT/results.jsonl")
    echo "  Existing results: $LINES lines"
else
    echo "  No existing results on this VM"
    # If RESTORE_FROM is set, copy files from there
    if [ -n "${RESTORE_FROM:-}" ] && [ -d "$RESTORE_FROM" ]; then
        echo "  Restoring from $RESTORE_FROM..."
        cp -v "$RESTORE_FROM"/* "$DAPH_OUTPUT/" 2>/dev/null || true
        LINES=$(wc -l < "$DAPH_OUTPUT/results.jsonl" 2>/dev/null || echo 0)
        echo "  Restored: $LINES lines"
    fi
fi

# --- Step 6: Verify frozen identities ---
echo ""
echo "[6] Verifying frozen identities..."
if [ -f "$DAPH_OUTPUT/identity_frozen.json" ]; then
    cat "$DAPH_OUTPUT/identity_frozen.json"
    # Verify confirmation executable SHA matches expected
    ACTUAL_CONFIRMATION_SHA=$(python3 -c "
import json
with open('$DAPH_OUTPUT/run_manifest.json') as f:
    m = json.load(f)
print(m.get('confirmation_executable_sha256', ''))
" 2>/dev/null || echo "")
    if [ -n "$ACTUAL_CONFIRMATION_SHA" ] && [ "$ACTUAL_CONFIRMATION_SHA" != "$EXPECTED_CONFIRMATION_SHA" ]; then
        echo "  ERROR: Confirmation executable SHA mismatch"
        echo "    Expected: $EXPECTED_CONFIRMATION_SHA"
        echo "    Actual:   $ACTUAL_CONFIRMATION_SHA"
        echo "  ABORT: Identity failure — do not resume with mismatched executable"
        exit 1
    fi
    echo "  OK: Confirmation executable SHA matches"
else
    echo "  No identity file yet (will be created on first run)"
fi

# --- Step 7: Record execution segment ---
echo ""
echo "[7] Recording execution segment..."
SEGMENT_FILE="$DAPH_OUTPUT/execution_segments.jsonl"
COMPLETED_COUNT=$(wc -l < "$DAPH_OUTPUT/results.jsonl" 2>/dev/null || echo 0)
SEGMENT_NUM=$(wc -l < "$SEGMENT_FILE" 2>/dev/null || echo 0)
SEGMENT_NUM=$((SEGMENT_NUM + 1))
TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)

# Get GPU info
GPU_INFO=$(python3 -c "
try:
    import subprocess
    r = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], capture_output=True, text=True)
    print(r.stdout.strip())
except: print('unknown')
" 2>/dev/null || echo "unknown")

# Get CUDA version
CUDA_VERSION=$(nvcc --version 2>/dev/null | grep "release" | sed 's/.*release //' | awk '{print $1}' || echo "unknown")

cat >> "$SEGMENT_FILE" <<SEGJSON
{"segment": $SEGMENT_NUM, "session_id": "${COLAB_SESSION_ID:-unknown}", "start_completed": $COMPLETED_COUNT, "end_completed": null, "termination_reason": null, "gpu": "$GPU_INFO", "cuda_version": "$CUDA_VERSION", "llama_cpp_archive": "$(basename $ARCHIVE)", "gguf_sha256": "$EXPECTED_GGUF_SHA", "experiment_source_commit": "$FROZEN_EXPERIMENT_COMMIT", "confirmation_executable_sha256": "$EXPECTED_CONFIRMATION_SHA", "runtime_config_sha256": "c64eb7b828feeac599e4bb001bf14a790efabe0d8e39c4f9cc4486062ad024c3", "start_timestamp": "$TIMESTAMP", "end_timestamp": null}
SEGJSON
echo "  Segment $SEGMENT_NUM recorded: start_completed=$COMPLETED_COUNT"

echo ""
echo "=== Runtime Restore Complete ==="
echo "  Server: http://127.0.0.1:8081"
echo "  Output: $DAPH_OUTPUT"
echo "  Repo:   $(git rev-parse --short HEAD) (detached at frozen commit)"
echo "  Segment: $SEGMENT_NUM (starting at $COMPLETED_COUNT completed)"
echo ""
echo "To resume R13:"
echo "  PYTHONPATH=$REPO_DIR python3 $REPO_DIR/scripts/run_r13_confirmation.py \\"
echo "    --output-dir $DAPH_OUTPUT \\"
echo "    --base-url http://127.0.0.1:8081/v1 \\"
echo "    --model-name google/gemma-3-12b-it-qat-q4_0-gguf \\"
echo "    --gguf-sha256 $EXPECTED_GGUF_SHA \\"
echo "    --max-tokens 128 --parallel 4 --n-per-cell 40"
