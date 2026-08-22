#!/usr/bin/env bash
# DAPH Colab control plane — one-command operations
#
# Usage:
#   ./tools/colab/daph_colab.sh status       — show current state
#   ./tools/colab/daph_colab.sh provision    — create L4 session
#   ./tools/colab/daph_colab.sh restore      — restore runtime on fresh VM
#   ./tools/colab/daph_colab.sh launch-r13   — start R13 in tmux via SSH
#   ./tools/colab/daph_colab.sh recover-r13  — full recovery: provision + restore + resume
#   ./tools/colab/daph_colab.sh monitor      — check health + download checkpoint
#   ./tools/colab/daph_colab.sh attach       — SSH in and attach to tmux
#   ./tools/colab/daph_colab.sh stop         — stop session
#
# Architecture:
#   Control plane (this script, Mac/Windsurf):
#     - provision VMs
#     - upload artifacts
#     - monitor health
#     - download checkpoints
#     - trigger recovery
#
#   Data plane (Colab VM, tmux):
#     - llama-server
#     - R13 runner
#     - local result writes
#     - completely independent of SSH/Windsurf connection

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SESSION="${DAPH_SESSION:-daph}"
CHECKPOINT_DIR="$HOME/DAPH_CHECKPOINTS/R13"
ARCHIVE="$REPO_DIR/.cache/colab/llama-server-d775b8967a46-cuda-sm89.tar.gz"
GGUF_SHA="2ad4c9ce431a2d5b80af37983828c2cfb8f4909792ca5075e0370e3a71ca013d"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

cmd_status() {
    log "=== DAPH Colab Status ==="
    colab sessions 2>&1 || true
    echo ""
    if [ -f "$CHECKPOINT_DIR/progress.json" ]; then
        log "Local checkpoint:"
        python3 -c "import json; d=json.load(open('$CHECKPOINT_DIR/progress.json')); print(f'  completed={d[\"completed\"]}/{d[\"expected_trajectories\"]} failed={d[\"failed\"]}')"
    fi
    if [ -f "$CHECKPOINT_DIR/watcher_state.json" ]; then
        log "Last verification:"
        python3 -c "import json; d=json.load(open('$CHECKPOINT_DIR/watcher_state.json')); print(f'  unique_keys={d[\"unique_keys\"]} duplicates={d[\"duplicates\"]} errors={d[\"errors\"]} sha={d[\"results_sha256_prefix\"]}')"
    fi
    # Check if watcher is running
    if pgrep -f r13_checkpoint_loop.sh >/dev/null 2>&1; then
        log "Watcher: RUNNING (PID $(pgrep -f r13_checkpoint_loop.sh | head -1))"
    else
        log "Watcher: NOT RUNNING"
    fi
    # Check caffeinate
    if pgrep -x caffeinate >/dev/null 2>&1; then
        log "Caffeinate: ACTIVE"
    else
        log "Caffeinate: INACTIVE"
    fi
}

cmd_provision() {
    log "=== Provisioning Colab L4 session ==="
    colab sessions 2>&1 | grep -q "gpu-l4" && {
        log "GPU session already exists"
        colab sessions 2>&1
        return 0
    }
    colab new -s "$SESSION" --gpu L4
    log "Session created"
    colab sessions
}

cmd_restore() {
    log "=== Uploading runtime archive ==="
    colab upload "$ARCHIVE" /content/llama-server-d775b8967a46-cuda-sm89.tar.gz

    log "=== Uploading checkpoint ==="
    if [ -d "$CHECKPOINT_DIR" ] && [ -f "$CHECKPOINT_DIR/results.jsonl" ]; then
        # Upload checkpoint files to VM
        for f in results.jsonl progress.json identity_frozen.json run_manifest.json \
                 model_calls.jsonl mechanism_receipts.jsonl cognition_cost_receipts.jsonl \
                 errors.jsonl context_preflight.json; do
            if [ -f "$CHECKPOINT_DIR/$f" ]; then
                colab upload "$CHECKPOINT_DIR/$f" "/content/daph_r13/$f" 2>/dev/null || true
            fi
        done
        log "Checkpoint uploaded"
    else
        log "No local checkpoint to upload"
    fi

    log "=== Running restore script on VM ==="
    colab upload "$REPO_DIR/tools/colab/restore_daph_runtime.sh" /content/restore_daph_runtime.sh
    colab exec -s "$SESSION" -- bash /content/restore_daph_runtime.sh --timeout 600
}

cmd_launch_r13() {
    log "=== Launching R13 in tmux ==="
    # Upload the launch script
    cat > /tmp/r13_tmux_launch.sh <<'INNER'
#!/usr/bin/env bash
set -euo pipefail
cd /content/Daph-ex-research-gate-c2-beir-retrieval

# Kill any existing tmux session
tmux kill-session -t r13 2>/dev/null || true

# Start R13 in tmux
tmux new-session -d -s r13 "PYTHONPATH=/content/Daph-ex-research-gate-c2-beir-retrieval \
  python3 -u scripts/run_r13_confirmation.py \
    --output-dir /content/daph_r13 \
    --base-url http://127.0.0.1:8081/v1 \
    --model-name google/gemma-3-12b-it-qat-q4_0-gguf \
    --gguf-sha256 2ad4c9ce431a2d5b80af37983828c2cfb8f4909792ca5075e0370e3a71ca013d \
    --max-tokens 128 --parallel 4 --n-per-cell 40 \
    2>&1 | tee /content/daph_r13/r13.log"

echo "R13 launched in tmux session 'r13'"
echo "  Attach: colab ssh -s daph && tmux attach -t r13"
echo "  Log:    /content/daph_r13/r13.log"
INNER
    colab upload /tmp/r13_tmux_launch.sh /content/r13_tmux_launch.sh
    colab exec -s "$SESSION" -- bash /content/r13_tmux_launch.sh --timeout 30
    log "R13 is running in tmux (independent of SSH)"
}

cmd_recover_r13() {
    log "=== Full R13 Recovery ==="
    cmd_provision
    cmd_restore
    cmd_launch_r13
    log "=== Recovery Complete ==="
    log "R13 is running in tmux. Start watcher with: ./tools/colab/daph_colab.sh monitor"
}

cmd_monitor() {
    log "=== Starting checkpoint watcher ==="
    # Ensure caffeinate is running
    if ! pgrep -x caffeinate >/dev/null 2>&1; then
        caffeinate -dimsu &
        log "Started caffeinate (PID $!)"
    fi
    # Start watcher if not running
    if pgrep -f r13_checkpoint_loop.sh >/dev/null 2>&1; then
        log "Watcher already running"
    else
        nohup "$CHECKPOINT_DIR/r13_checkpoint_loop.sh" \
            > "$CHECKPOINT_DIR/checkpoint_loop.log" 2>&1 &
        log "Watcher started (PID $!)"
    fi
    log "Monitor active. Check: $CHECKPOINT_DIR/checkpoint_loop.log"
}

cmd_attach() {
    log "=== SSH to Colab and attach to tmux ==="
    colab ssh -s "$SESSION"
}

cmd_stop() {
    log "=== Stopping Colab session ==="
    colab stop -s "$SESSION" || true
    log "Session stopped"
}

# Main dispatch
case "${1:-status}" in
    status)       cmd_status ;;
    provision)    cmd_provision ;;
    restore)      cmd_restore ;;
    launch-r13)   cmd_launch_r13 ;;
    recover-r13)  cmd_recover_r13 ;;
    monitor)      cmd_monitor ;;
    attach)       cmd_attach ;;
    stop)         cmd_stop ;;
    *)
        echo "Usage: $0 {status|provision|restore|launch-r13|recover-r13|monitor|attach|stop}"
        exit 1
        ;;
esac
