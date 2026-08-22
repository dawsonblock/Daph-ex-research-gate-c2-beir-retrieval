#!/usr/bin/env python3
"""DAPH Colab runtime health monitor.

Runs on the Colab VM, checking every 60 seconds:
  - llama-server process alive?
  - HTTP /health responds?
  - GPU visible?
  - R13 runner process alive?
  - Progress increasing?
  - Disk space?

Outputs JSON state to /content/daph_r13/health.json
and prints alerts to stdout.

Usage: colab exec -f tools/colab/runtime_health.py --timeout 60
"""
import json, os, subprocess, time, sys, urllib.request

HEALTH_FILE = "/content/daph_r13/health.json"
R13_OUTPUT = "/content/daph_r13"
CHECK_INTERVAL = 60  # seconds
STALE_PROGRESS_THRESHOLD = 900  # 15 minutes

def check_server_alive():
    """Check if llama-server process exists."""
    r = subprocess.run(
        "pgrep -f llama-server | head -1",
        shell=True, capture_output=True, text=True)
    pid = r.stdout.strip()
    return bool(pid), pid

def check_server_http():
    """Check if server responds to /health."""
    try:
        urllib.request.urlopen("http://127.0.0.1:8081/health", timeout=10)
        return True
    except Exception:
        return False

def check_gpu():
    """Check GPU availability."""
    r = subprocess.run(
        "nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null",
        shell=True, capture_output=True, text=True)
    if r.returncode == 0 and r.stdout.strip():
        parts = r.stdout.strip().split(",")
        return {
            "available": True,
            "name": parts[0].strip() if parts else "unknown",
            "vram_used": parts[1].strip() if len(parts) > 1 else "?",
            "vram_total": parts[2].strip() if len(parts) > 2 else "?",
        }
    return {"available": False, "name": "none", "vram_used": "?", "vram_total": "?"}

def check_runner_alive():
    """Check if R13 runner process exists."""
    r = subprocess.run(
        "pgrep -f run_r13_confirmation | head -1",
        shell=True, capture_output=True, text=True)
    pid = r.stdout.strip()
    return bool(pid), pid

def check_progress():
    """Read progress.json and return completed count."""
    path = os.path.join(R13_OUTPUT, "progress.json")
    if not os.path.exists(path):
        return 0, 0, 0
    with open(path) as f:
        p = json.load(f)
    return p.get("completed", 0), p.get("expected_trajectories", 1280), p.get("failed", 0)

def check_disk():
    """Check disk space."""
    r = subprocess.run("df -h /content", shell=True, capture_output=True, text=True)
    lines = r.stdout.strip().split("\n")
    if len(lines) >= 2:
        parts = lines[1].split()
        return {
            "total": parts[1] if len(parts) > 1 else "?",
            "used": parts[2] if len(parts) > 2 else "?",
            "avail": parts[3] if len(parts) > 3 else "?",
            "percent": parts[4] if len(parts) > 4 else "?",
        }
    return {"total": "?", "used": "?", "avail": "?", "percent": "?"}

def main():
    prev_completed = 0
    prev_time = time.time()
    stale_count = 0

    print("=== DAPH Runtime Health Monitor ===")
    print(f"  Check interval: {CHECK_INTERVAL}s")
    print(f"  Stale threshold: {STALE_PROGRESS_THRESHOLD}s")

    while True:
        server_pid_ok, server_pid = check_server_alive()
        server_http_ok = check_server_http()
        gpu_info = check_gpu()
        runner_ok, runner_pid = check_runner_alive()
        completed, expected, failed = check_progress()
        disk = check_disk()

        now = time.time()
        time_since_progress = now - prev_time if completed == prev_completed else 0
        if completed != prev_completed:
            prev_completed = completed
            prev_time = now
            stale_count = 0
        else:
            stale_count += 1

        state = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "server_alive": server_pid_ok,
            "server_pid": server_pid,
            "server_http": server_http_ok,
            "gpu": gpu_info,
            "runner_alive": runner_ok,
            "runner_pid": runner_pid,
            "completed": completed,
            "expected": expected,
            "failed": failed,
            "stale_progress_seconds": int(time_since_progress),
            "stale_check_count": stale_count,
            "disk": disk,
        }

        # Write health file
        os.makedirs(R13_OUTPUT, exist_ok=True)
        with open(HEALTH_FILE, "w") as f:
            json.dump(state, f, indent=2)

        # Print status
        status_line = (
            f"[{state['timestamp']}] "
            f"server={'OK' if server_http_ok else 'DOWN'} "
            f"runner={'OK' if runner_ok else 'STOPPED'} "
            f"gpu={gpu_info['name']} "
            f"completed={completed}/{expected} "
            f"errors={failed} "
            f"disk={disk.get('percent', '?')}"
        )
        print(status_line, flush=True)

        # Alerts
        if not server_pid_ok:
            print(f"  ALERT: llama-server process not found", flush=True)
        if not server_http_ok and server_pid_ok:
            print(f"  ALERT: server process alive but HTTP /health not responding", flush=True)
        if not runner_ok and completed < expected:
            print(f"  ALERT: R13 runner not running but {completed}/{expected} completed", flush=True)
        if failed > 0:
            print(f"  ALERT: {failed} failures recorded", flush=True)
        if stale_count * CHECK_INTERVAL >= STALE_PROGRESS_THRESHOLD and completed < expected:
            print(f"  ALERT: No progress for {stale_count * CHECK_INTERVAL}s (stuck at {completed})", flush=True)

        # Check completion
        if completed >= expected and expected > 0:
            print(f"  R13 COMPLETE: {completed}/{expected}", flush=True)
            # Don't exit — keep monitoring until runner exits

        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()
