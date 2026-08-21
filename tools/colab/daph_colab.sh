#!/usr/bin/env bash
set -euo pipefail

# DAPH Colab control script
# Usage: ./tools/colab/daph_colab.sh <command>

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SESSION="daph"
PORT=8080

log() { echo "[daph-colab] $*"; }
err() { echo "[daph-colab] ERROR: $*" >&2; }

check_session() {
    if ! colab sessions 2>&1 | grep -q "$SESSION"; then
        err "Session '$SESSION' does not exist. Run '$0 start' first."
        exit 1
    fi
}

cmd_status() {
    colab status -s "$SESSION" 2>&1
}

cmd_start() {
    if colab sessions 2>&1 | grep -q "$SESSION"; then
        log "Session '$SESSION' already exists."
        colab status -s "$SESSION"
        return 0
    fi
    log "Creating session '$SESSION' with T4 GPU..."
    colab new -s "$SESSION" --gpu T4
    colab status -s "$SESSION"
}

cmd_stop() {
    check_session
    log "Stopping session '$SESSION'..."
    colab stop -s "$SESSION"
}

cmd_bootstrap() {
    check_session
    log "Bootstrapping remote workspace..."
    local commit
    commit=$(cd "$REPO_DIR" && git rev-parse HEAD)
    log "Local commit: $commit"

    # Upload bootstrap script
    colab upload -s "$SESSION" "$SCRIPT_DIR/bootstrap_remote.py" bootstrap_remote.py

    # Run with DAPH_COMMIT env var
    colab exec -s "$SESSION" -f "$SCRIPT_DIR/bootstrap_remote.py" --timeout 600 2>&1 || {
        err "Bootstrap failed. Trying with env var..."
        colab exec -s "$SESSION" -f "$SCRIPT_DIR/bootstrap_remote.py" --timeout 600 2>&1
    }
}

cmd_preflight() {
    check_session
    log "Running no-LLM preflight checks..."

    # Upload and run preflight script
    colab upload -s "$SESSION" "$SCRIPT_DIR/preflight.py" preflight.py
    colab exec -s "$SESSION" -f "$SCRIPT_DIR/preflight.py" --timeout 300
}

cmd_r8() {
    check_session
    log "Running R8.1 retrieval qualification..."
    colab upload -s "$SESSION" "$REPO_DIR/scripts/i3_15c_prepare_evidence.py" prepare_evidence.py
    colab exec -s "$SESSION" -f "$SCRIPT_DIR/run_r8.py" --timeout 300
}

cmd_r9() {
    check_session
    log "Running R9 reasoning-budget qualification..."
    # Upload R9 script
    colab upload -s "$SESSION" "$SCRIPT_DIR/r9_reasoning_budget.py" r9_reasoning_budget.py
    # Upload and run the R9 setup+execution script
    colab upload -s "$SESSION" "$SCRIPT_DIR/run_r9.py" run_r9.py
    colab exec -s "$SESSION" -f "$SCRIPT_DIR/run_r9.py" --timeout 3600
}

cmd_confirm() {
    check_session
    log "Running R13 powered confirmation..."
    log "WARNING: This will consume significant GPU time."
    colab upload -s "$SESSION" "$SCRIPT_DIR/run_confirm.py" run_confirm.py
    colab exec -s "$SESSION" -f "$SCRIPT_DIR/run_confirm.py" --timeout 7200
}

cmd_fetch() {
    check_session
    log "Fetching results from Colab..."
    local dest="$REPO_DIR/experiments/v2b_i3_15c/confirmation"
    mkdir -p "$dest"

    # Download key result files
    for f in r9_results.json r9b_results.json preflight_report.json confirmation_results.jsonl confirmation_analysis.json; do
        colab download -s "$SESSION" "$f" "$dest/$f" 2>/dev/null || true
    done
    log "Results fetched to $dest"
}

# Main
case "${1:-help}" in
    status)   cmd_status ;;
    start)    cmd_start ;;
    stop)     cmd_stop ;;
    bootstrap) cmd_bootstrap ;;
    preflight) cmd_preflight ;;
    r8)       cmd_r8 ;;
    r9)       cmd_r9 ;;
    confirm)  cmd_confirm ;;
    fetch)    cmd_fetch ;;
    help|*)
        echo "DAPH Colab Control Script"
        echo "Usage: $0 <command>"
        echo ""
        echo "Commands:"
        echo "  status    Show Colab session status"
        echo "  start     Create T4 GPU session"
        echo "  stop      Stop session"
        echo "  bootstrap Clone repo + install deps on Colab"
        echo "  preflight Run no-LLM confirmation preflight"
        echo "  r8        Run R8.1 retrieval qualification"
        echo "  r9        Run R9 reasoning-budget qualification"
        echo "  confirm   Run R13 powered confirmation"
        echo "  fetch     Download results from Colab"
        ;;
esac
