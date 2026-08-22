#!/usr/bin/env python3
"""R13 Checkpoint integrity verifier.

Usage: python3 verify_checkpoint.py <results.jsonl> <progress.json> <errors.jsonl> <state_file>

Verifies:
  - unique_keys == progress.completed
  - duplicates == 0
  - errors == 0
  - Records SHA256 and byte size for provenance

Exit codes:
  0 = OK
  1 = Files missing or incomplete
  2 = Invariant violation (unique_keys != completed or duplicates > 0)
  3 = Errors recorded in errors.jsonl
"""
import json, sys, hashlib, os
from datetime import datetime

results_path, progress_path, errors_path, state_file = sys.argv[1:5]

# Load progress
try:
    with open(progress_path) as f:
        progress = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    print("  WARNING: progress.json missing or invalid")
    sys.exit(1)

completed = progress.get("completed", 0)
expected = progress.get("expected_trajectories", 1280)
failed = progress.get("failed", 0)

# Count results lines and unique keys
try:
    with open(results_path) as f:
        lines = [l.strip() for l in f if l.strip()]
except FileNotFoundError:
    print("  WARNING: results.jsonl missing")
    sys.exit(1)

line_count = len(lines)
keys = set()
duplicates = 0
for line in lines:
    try:
        r = json.loads(line)
        key = r.get("trajectory_key", "")
        if key in keys:
            duplicates += 1
        keys.add(key)
    except json.JSONDecodeError:
        pass

unique_keys = len(keys)

# Count errors
error_count = 0
if os.path.exists(errors_path):
    with open(errors_path) as f:
        error_count = sum(1 for l in f if l.strip())

# Compute SHA256 of results.jsonl
sha = hashlib.sha256()
with open(results_path, "rb") as f:
    for chunk in iter(lambda: f.read(65536), b""):
        sha.update(chunk)
results_sha = sha.hexdigest()[:16]

# Byte size
results_bytes = os.path.getsize(results_path)

# Load previous state
prev_state = {}
if os.path.exists(state_file):
    try:
        with open(state_file) as f:
            prev_state = json.load(f)
    except (json.JSONDecodeError, IOError):
        pass

# Write new state
state = {
    "timestamp": datetime.now().isoformat(),
    "completed": completed,
    "expected": expected,
    "failed": failed,
    "results_lines": line_count,
    "unique_keys": unique_keys,
    "duplicates": duplicates,
    "errors": error_count,
    "results_bytes": results_bytes,
    "results_sha256_prefix": results_sha,
    "progress_increased": completed > prev_state.get("completed", -1),
}
with open(state_file, "w") as f:
    json.dump(state, f, indent=2)

# Print summary
print(f"  completed={completed}/{expected} lines={line_count} "
      f"unique_keys={unique_keys} duplicates={duplicates} "
      f"errors={error_count} bytes={results_bytes} sha={results_sha}")

# Check invariants
invariant_ok = (unique_keys == completed) and (duplicates == 0)
if not invariant_ok:
    print(f"  INVARIANT_VIOLATION: unique_keys({unique_keys}) != completed({completed}) "
          f"or duplicates({duplicates}) > 0")
    sys.exit(2)

if error_count > 0:
    print(f"  ERROR_COUNT_NONZERO: {error_count} errors recorded")
    sys.exit(3)

sys.exit(0)
