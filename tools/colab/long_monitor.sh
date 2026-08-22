#!/usr/bin/env bash
# R13 Long Monitor — checks progress every 10 minutes
# Detects completion and triggers closure
# Handles session drops by reconnecting

set -euo pipefail

CHECKPOINT_DIR="$HOME/DAPH_CHECKPOINTS/R13"
REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
LOG="$CHECKPOINT_DIR/long_monitor.log"
POLL_INTERVAL=600  # 10 minutes

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "=== R13 Long Monitor Started ==="
log "  Poll interval: ${POLL_INTERVAL}s"

while true; do
    # Check if watcher is alive
    if ! pgrep -f r13_checkpoint_loop.sh >/dev/null 2>&1; then
        log "Watcher not running — restarting"
        nohup "$CHECKPOINT_DIR/r13_checkpoint_loop.sh" \
            > "$CHECKPOINT_DIR/checkpoint_loop.log" 2>&1 &
        log "  Watcher restarted (PID $!)"
    fi

    # Check caffeinate
    if ! pgrep -x caffeinate >/dev/null 2>&1; then
        caffeinate -dimsu &
        log "Caffeinate restarted (PID $!)"
    fi

    # Read latest checkpoint state
    if [ -f "$CHECKPOINT_DIR/watcher_state.json" ]; then
        COMPLETED=$(python3 -c "import json; print(json.load(open('$CHECKPOINT_DIR/watcher_state.json'))['completed'])" 2>/dev/null || echo 0)
        ERRORS=$(python3 -c "import json; print(json.load(open('$CHECKPOINT_DIR/watcher_state.json'))['errors'])" 2>/dev/null || echo 0)
        UNIQUE=$(python3 -c "import json; print(json.load(open('$CHECKPOINT_DIR/watcher_state.json'))['unique_keys'])" 2>/dev/null || echo 0)

        log "State: completed=$COMPLETED/1280 unique=$UNIQUE errors=$ERRORS"

        # Check for completion
        if [ "$COMPLETED" -ge 1280 ] && [ "$UNIQUE" -ge 1280 ] && [ "$ERRORS" -eq 0 ]; then
            log ""
            log "=== R13 COMPLETE: 1280/1280 ==="
            log "  Triggering dataset closure..."

            # Run closure
            python3 "$REPO_DIR/tools/colab/close_r13_dataset.py" \
                --checkpoint-dir "$CHECKPOINT_DIR" \
                --output-dir "$REPO_DIR/experiments/v2b_i3_15c/confirmation/r13" \
                2>&1 | tee -a "$LOG"

            log "=== R13 Closure Complete ==="
            log "=== Long Monitor exiting ==="
            exit 0
        fi

        # Alert on errors
        if [ "$ERRORS" -gt 0 ]; then
            log "ALERT: $ERRORS errors detected in checkpoint"
        fi
    else
        log "No watcher state yet"
    fi

    # Also do a direct Colab check every cycle
    cat > /tmp/_r13_quick_check.py <<'PY'
import json, os
if os.path.exists("/content/daph_r13/progress.json"):
    with open("/content/daph_r13/progress.json") as f:
        p = json.load(f)
    print(f"REMOTE: {p['completed']}/{p['expected_trajectories']} failed={p['failed']}")
else:
    print("REMOTE: no progress file")
PY
    REMOTE=$(timeout 45 colab exec -s daph -f /tmp/_r13_quick_check.py --timeout 30 2>/dev/null || echo "REMOTE: session unavailable")
    log "  $REMOTE"

    log "  Sleeping ${POLL_INTERVAL}s..."
    sleep "$POLL_INTERVAL"
done
