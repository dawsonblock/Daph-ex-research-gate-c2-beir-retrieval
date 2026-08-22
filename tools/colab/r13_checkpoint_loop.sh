#!/bin/bash
# R13 Checkpoint Watcher — robust off-VM persistence
#
# Downloads R13 checkpoint files from Colab every 5 minutes,
# verifies integrity, and alerts on operational issues.
#
# Does NOT inspect efficacy results (utility, success, R1 vs A1, T2).
# Monitors only infrastructure/progress.
#
# Alerts on:
#   - progress does not increase for 15 minutes (3 consecutive cycles)
#   - errors > 0
#   - duplicate trajectory key in results.jsonl
#   - Colab session disappears
#   - checkpoint download fails twice consecutively
#
# Usage: nohup ./r13_checkpoint_loop.sh > checkpoint_loop.log 2>&1 &

set -euo pipefail

DEST="$HOME/DAPH_CHECKPOINTS/R13"
SESSION="daph"
INTERVAL=300  # 5 minutes
STALE_THRESHOLD=3  # 3 consecutive cycles with no progress = 15 min
LOG="$DEST/checkpoint_loop.log"
ALERT_LOG="$DEST/alerts.log"
STATE_FILE="$DEST/watcher_state.json"

mkdir -p "$DEST"

# Files to download from Colab
FILES=(
    "progress.json"
    "results.jsonl"
    "model_calls.jsonl"
    "mechanism_receipts.jsonl"
    "cognition_cost_receipts.jsonl"
    "errors.jsonl"
    "identity_frozen.json"
    "run_manifest.json"
    "context_preflight.json"
)

# State tracking
LAST_COMPLETED=-1
STALE_COUNT=0
DOWNLOAD_FAIL_COUNT=0
CYCLE=0

# R12.9M: Expected confirmation executable SHA — reject checkpoints from wrong VM
EXPECTED_CONFIRMATION_SHA="c64eb7b828feeac599e4bb001bf14a790efabe0d8e39c4f9cc4486062ad024c3"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
}

alert() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ALERT: $*" | tee -a "$ALERT_LOG"
}

verify_checkpoint() {
    local results_path="$DEST/results.jsonl"
    local progress_path="$DEST/progress.json"
    local errors_path="$DEST/errors.jsonl"

    # Check files exist
    if [ ! -f "$results_path" ] || [ ! -f "$progress_path" ]; then
        log "  WARNING: results.jsonl or progress.json missing"
        return 1
    fi

    # Verify integrity using external Python script
    python3 "$DEST/verify_checkpoint.py" "$results_path" "$progress_path" "$errors_path" "$STATE_FILE"
    local rc=$?
    return $rc
}

check_colab_alive() {
    timeout --kill-after=3s 20s colab sessions 2>/dev/null | grep -q "$SESSION" || {
        # Try without session name filter
        timeout --kill-after=3s 20s colab sessions 2>/dev/null | grep -q "gpu-l4" && return 0
        return 1
    }
    return 0
}

check_server_alive() {
    cat > /tmp/_r13_health.py <<'PY'
import urllib.request, sys
try:
    urllib.request.urlopen("http://127.0.0.1:8081/health", timeout=10)
    print("SERVER_OK")
except Exception as e:
    print(f"SERVER_DOWN: {e}")
    sys.exit(1)
PY
    timeout --kill-after=5s 45s colab exec -s "$SESSION" -f /tmp/_r13_health.py --timeout 30 2>/dev/null | grep -q "SERVER_OK"
    return $?
}

# Main loop
log "=== R13 Checkpoint Watcher Started ==="
log "  Destination: $DEST"
log "  Session: $SESSION"
log "  Interval: ${INTERVAL}s"
log "  Stale threshold: ${STALE_THRESHOLD} cycles ($(( STALE_THRESHOLD * INTERVAL / 60 )) min)"

while true; do
    CYCLE=$((CYCLE + 1))
    log ""
    log "--- Checkpoint Cycle #$CYCLE ---"

    # Check Colab session
    if ! check_colab_alive; then
        alert "Colab session '$SESSION' not found — VM may have recycled"
        # Don't exit — session might come back or we need to restore
        STALE_COUNT=$((STALE_COUNT + 1))
    else
        # Check server health
        if ! check_server_alive; then
            alert "llama-server not responding on Colab"
            STALE_COUNT=$((STALE_COUNT + 1))
        fi
    fi

    # Download files (with per-file timeout + kill-after to prevent hangs)
    DOWNLOAD_OK=true
    for f in "${FILES[@]}"; do
        if timeout --kill-after=5s 60s colab download -s "$SESSION" "/content/daph_r13/$f" "$DEST/$f" 2>/dev/null; then
            log "  Downloaded $f ($(wc -c < "$DEST/$f" 2>/dev/null || echo 0) bytes)"
        else
            # File might not exist yet (e.g. confirmation SHA at end)
            if [ "$f" != "confirmation_executable_sha256.txt" ] && \
               [ "$f" != "semantic_error_attribution.json" ] && \
               [ "$f" != "mechanism_receipts_strengthened.jsonl" ]; then
                log "  Download FAILED or TIMEOUT: $f"
                DOWNLOAD_OK=false
            fi
        fi
    done

    if [ "$DOWNLOAD_OK" = false ]; then
        DOWNLOAD_FAIL_COUNT=$((DOWNLOAD_FAIL_COUNT + 1))
        if [ $DOWNLOAD_FAIL_COUNT -ge 2 ]; then
            alert "Checkpoint download failed $DOWNLOAD_FAIL_COUNT consecutive times"
        fi
    else
        DOWNLOAD_FAIL_COUNT=0
    fi

    # Verify checkpoint integrity
    if [ -f "$DEST/results.jsonl" ] && [ -f "$DEST/progress.json" ]; then
        # R12.9M: Verify remote run_manifest SHA before accepting checkpoint
        if [ -f "$DEST/run_manifest.json" ]; then
            REMOTE_SHA=$(python3 -c "import json; print(json.load(open('$DEST/run_manifest.json')).get('confirmation_executable_sha256',''))" 2>/dev/null || echo "")
            if [ -n "$REMOTE_SHA" ] && [ "$REMOTE_SHA" != "$EXPECTED_CONFIRMATION_SHA" ]; then
                alert "Remote confirmation_executable_sha mismatch: got ${REMOTE_SHA:0:16}... expected ${EXPECTED_CONFIRMATION_SHA:0:16}..."
                log "  REJECTING checkpoint — wrong VM or stale session"
                # Don't update progress or accept this checkpoint
                STALE_COUNT=$((STALE_COUNT + 1))
                log "  Sleeping ${INTERVAL}s..."
                sleep "$INTERVAL"
                continue
            fi
        fi

        if verify_checkpoint; then
            # Read current completed count
            CURRENT=$(python3 -c "import json; print(json.load(open('$DEST/progress.json'))['completed'])" 2>/dev/null || echo 0)

            if [ "$CURRENT" -eq "$LAST_COMPLETED" ] && [ "$CURRENT" -gt 0 ]; then
                STALE_COUNT=$((STALE_COUNT + 1))
                log "  No progress: still at $CURRENT (stale count: $STALE_COUNT)"
                if [ $STALE_COUNT -ge $STALE_THRESHOLD ]; then
                    alert "No progress for $(( STALE_COUNT * INTERVAL / 60 )) minutes (stuck at $CURRENT/$EXPECTED)"
                fi
            else
                STALE_COUNT=0
                log "  Progress OK: $CURRENT trajectories"
            fi
            LAST_COMPLETED=$CURRENT

            # Check for completion
            EXPECTED=$(python3 -c "import json; print(json.load(open('$DEST/progress.json'))['expected_trajectories'])" 2>/dev/null || echo 1280)
            if [ "$CURRENT" -ge "$EXPECTED" ]; then
                log ""
                log "=== R13 COMPLETE: $CURRENT/$EXPECTED trajectories ==="
                log "  Run final analysis locally against downloaded results"
                log "=== Watcher exiting ==="
                exit 0
            fi
        else
            RC=$?
            if [ $RC -eq 2 ]; then
                alert "Invariant violation: unique_keys != completed or duplicates found"
            elif [ $RC -eq 3 ]; then
                alert "Errors detected in errors.jsonl"
            else
                log "  Checkpoint verification returned $RC (may be incomplete)"
            fi
        fi
    else
        log "  Checkpoint files not yet available"
    fi

    log "  Sleeping ${INTERVAL}s..."
    sleep "$INTERVAL"
done
