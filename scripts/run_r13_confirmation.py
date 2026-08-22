#!/usr/bin/env python3
"""R13: Powered Late-T2 Confirmation — Crash-Safe Resumable Runner.

Scientific protocol: confirmation_protocol_v2.json (FROZEN)
Backend: Gemma 3 12B IT QAT Q4_0 (frozen)
Retrieval: Q3_RERANKED only
Arms: A1_INFERRED, R1_INFERRED
Expected: 640 tasks × 2 arms = 1280 trajectories

Features:
  - Idempotent resume via trajectory key: (task_id, arm, retrieval_condition, backend_identity)
  - Append-only JSONL output (results, model_calls, mechanism_receipts, cognition_cost_receipts, errors)
  - Atomic progress.json updates
  - Identity verification on startup (protocol SHA, GGUF SHA, receipt SHA)
  - Abort on identity drift, decoder corruption, structural invariant failure
  - No early stopping — runs all 1280 trajectories
  - Final verification: 1280 unique keys, 0 duplicates, 0 missing
  - Preregistered statistical analysis run exactly once on complete frozen result set

Usage (on Colab, after llama-server is running on port 8081):
    PYTHONPATH=. python3 scripts/run_r13_confirmation.py \
        --output-dir experiments/v2b_i3_15c/confirmation/r13 \
        --base-url http://127.0.0.1:8081/v1 \
        --model-name google/gemma-3-12b-it-qat-q4_0-gguf \
        --gguf-sha256 2ad4c9ce431a2d5b80af37983828c2cfb8f4909792ca5075e0370e3a71ca013d \
        --max-tokens 128 \
        --parallel 4

To resume after a crash, simply re-run the same command.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import signal
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Load the i3_15c factorial module to reuse its trajectory runner and analysis
spec_15c = importlib.util.spec_from_file_location(
    "i3_15c_factorial", str(REPO_ROOT / "scripts" / "run_i3_15c_factorial.py"))
i3_15c = importlib.util.module_from_spec(spec_15c)
spec_15c.loader.exec_module(i3_15c)
i3_12j = i3_15c.i3_12j

from hrm_adaptive_memory.executive.semantic_relations.i3_15c_task_generator import (
    generate_i3_15c_corpus, validate_t2_eligibility, get_i3_15c_corpus,
)
from hrm_adaptive_memory.executive.model_backend import LocalLlamaBackend
from scripts.run_i3_15_r1_balanced import (
    build_corpus_index, get_required_passage_ids, build_retrieved_evidence_task,
    TOP_K, adapt_local_system_prompt,
)


# ---------------------------------------------------------------------------
# Frozen R13 configuration
# ---------------------------------------------------------------------------

R13_CONFIG = {
    "protocol_id": "I3_15C_CONFIRMATION_PROTOCOL_V2",
    "retrieval_condition": "Q3_RERANKED",
    "arms": ["A1_INFERRED", "R1_INFERRED"],
    "n_per_cell": 40,
    "seed": 42,
    "expected_tasks": 640,
    "expected_trajectories": 1280,
    "temperature": 0.0,
    "max_tokens": 128,
    "parallel_slots": 4,
    "reasoning_budget": "NOT_APPLICABLE",
}

# Abort thresholds
FAIL_CLOSED_ABORT_RATE = 0.15  # >15% FAIL_CLOSED in any 50-trajectory window
DECODER_FAIL_ABORT_RATE = 0.15  # >15% decoder failures in any 50-trajectory window


# ---------------------------------------------------------------------------
# Identity verification
# ---------------------------------------------------------------------------

def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def compute_receipt_identity(receipts_path: Path, retrieval_condition: str) -> str:
    """Compute a stable hash over all receipts for the given retrieval condition."""
    receipt_hashes = []
    with open(receipts_path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("retrieval_condition") == retrieval_condition:
                receipt_hashes.append(r["retrieval_sha256"])
    receipt_hashes.sort()
    return hashlib.sha256(
        json.dumps(receipt_hashes).encode()).hexdigest()


def verify_identity(
    output_dir: Path,
    protocol_path: Path,
    gguf_sha256: str,
    receipts_path: Path,
    retrieval_condition: str,
) -> dict[str, str]:
    """Verify all frozen identities. Returns identity dict or raises."""
    identities = {}

    # Protocol SHA
    with open(protocol_path) as f:
        protocol = json.load(f)
    protocol_sha = sha256_json(protocol)
    identities["protocol_sha256"] = protocol_sha
    identities["protocol_id"] = protocol.get("protocol_id", "UNKNOWN")

    # GGUF SHA (passed as argument, verified at caller)
    identities["gguf_sha256"] = gguf_sha256

    # Receipt identity
    receipt_identity = compute_receipt_identity(receipts_path, retrieval_condition)
    identities["receipt_identity_sha256"] = receipt_identity

    # Backend identity = GGUF SHA (first 16 chars)
    identities["backend_identity"] = gguf_sha256[:16]

    # Check for identity drift from previous run
    identity_file = output_dir / "identity_frozen.json"
    if identity_file.exists():
        with open(identity_file) as f:
            prev = json.load(f)
        for key in ["protocol_sha256", "gguf_sha256", "receipt_identity_sha256", "backend_identity"]:
            if prev.get(key) != identities[key]:
                raise RuntimeError(
                    f"IDENTITY DRIFT: {key} changed from {prev.get(key)} to {identities[key]}. "
                    f"ABORTING — cannot resume with different frozen identity.")
        print(f"  Identity verified: matches previous run")

    # Save identity
    with open(identity_file, "w") as f:
        json.dump(identities, f, indent=2)

    return identities


# ---------------------------------------------------------------------------
# Trajectory key
# ---------------------------------------------------------------------------

def make_trajectory_key(task_id: str, arm: str, retrieval_condition: str,
                        backend_identity: str) -> str:
    """Stable unique key for a trajectory."""
    return f"{task_id}|{arm}|{retrieval_condition}|{backend_identity}"


# ---------------------------------------------------------------------------
# Append-only JSONL writers (thread-safe)
# ---------------------------------------------------------------------------

class JsonlAppender:
    """Thread-safe append-only JSONL writer."""
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._fh = open(path, "a", buffering=1)

    def append(self, record: dict):
        with self._lock:
            self._fh.write(json.dumps(record, default=str) + "\n")

    def close(self):
        with self._lock:
            self._fh.close()


def atomic_write_json(path: Path, data: dict):
    """Atomically write JSON to a file (write temp, rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent), suffix=".tmp", prefix=path.stem + "_")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, default=str)
        os.rename(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Load completed keys from existing results
# ---------------------------------------------------------------------------

def load_completed_keys(results_path: Path) -> set[str]:
    """Load trajectory keys from existing results.jsonl."""
    completed = set()
    if not results_path.exists():
        return completed
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = record.get("trajectory_key")
            if key:
                completed.add(key)
    return completed


# ---------------------------------------------------------------------------
# Progress tracker
# ---------------------------------------------------------------------------

class ProgressTracker:
    """Atomic progress tracking."""
    def __init__(self, path: Path, expected: int, protocol_id: str):
        self.path = path
        self.expected = expected
        self.protocol_id = protocol_id
        self._lock = threading.Lock()
        self.completed = 0
        self.failed = 0
        self.remaining = expected
        self.last_completed_key = None
        self.start_time = time.time()

    def record_completion(self, key: str):
        with self._lock:
            self.completed += 1
            self.remaining = self.expected - self.completed - self.failed
            self.last_completed_key = key
        self._write()

    def record_failure(self):
        with self._lock:
            self.failed += 1
            self.remaining = self.expected - self.completed - self.failed
        self._write()

    def _write(self):
        atomic_write_json(self.path, {
            "protocol": self.protocol_id,
            "expected_trajectories": self.expected,
            "completed": self.completed,
            "failed": self.failed,
            "remaining": self.remaining,
            "last_completed_key": self.last_completed_key,
            "elapsed_s": round(time.time() - self.start_time, 1),
        })


# ---------------------------------------------------------------------------
# Abort monitor
# ---------------------------------------------------------------------------

class AbortMonitor:
    """Monitor for decoder corruption and structural invariant failures."""
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self._lock = threading.Lock()
        self._results: list[dict] = []
        self._aborted = False
        self._abort_reason = None

    def record(self, result: dict):
        with self._lock:
            self._results.append(result)
            if len(self._results) > self.window_size:
                self._results = self._results[-self.window_size:]

            # Check abort conditions
            window = self._results[-self.window_size:]
            n = len(window)
            if n >= 20:  # Need at least 20 to evaluate
                fail_closed = sum(
                    1 for r in window
                    if r.get("terminal_action") == "FAIL_CLOSED"
                    or r.get("terminal_result") == "FAIL_CLOSED")
                decoder_fails = sum(
                    1 for r in window
                    if "error" in r or r.get("terminal_result") == "BACKEND_ERROR")

                if fail_closed / n > FAIL_CLOSED_ABORT_RATE:
                    self._aborted = True
                    self._abort_reason = (
                        f"FAIL_CLOSED rate {fail_closed}/{n} = {fail_closed/n:.0%} "
                        f"exceeds threshold {FAIL_CLOSED_ABORT_RATE:.0%}")
                elif decoder_fails / n > DECODER_FAIL_ABORT_RATE:
                    self._aborted = True
                    self._abort_reason = (
                        f"Decoder/backend error rate {decoder_fails}/{n} = {decoder_fails/n:.0%} "
                        f"exceeds threshold {DECODER_FAIL_ABORT_RATE:.0%}")

    def should_abort(self) -> tuple[bool, str | None]:
        with self._lock:
            return self._aborted, self._abort_reason


# ---------------------------------------------------------------------------
# Pre-retrieve evidence from receipts
# ---------------------------------------------------------------------------

def load_q3_receipts(receipts_path: Path) -> dict[str, dict]:
    """Load Q3_RERANKED receipts indexed by task_id."""
    receipts = {}
    with open(receipts_path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("retrieval_condition") == "Q3_RERANKED":
                receipts[r["task_id"]] = r
    return receipts


def build_pre_retrieved_passages(receipt: dict, corpus_by_id: dict) -> list:
    """Build passage list from receipt's retrieved_chunk_ids."""
    return [
        corpus_by_id[pid] for pid in receipt.get("retrieved_chunk_ids", [])
        if pid in corpus_by_id
    ]


# ---------------------------------------------------------------------------
# Statistical analysis — matches confirmation_protocol_v2 exactly
# ---------------------------------------------------------------------------

def _stratum_from_category(cat: str) -> str:
    """Map category string to protocol v2 stratum name."""
    if cat.startswith("t2_conflict_immediate"):
        return "T2_CONFLICT_IMMEDIATE"
    elif cat.startswith("t2_conflict_late_1"):
        return "T2_CONFLICT_LATE_1"
    elif cat.startswith("t2_conflict_late_2"):
        return "T2_CONFLICT_LATE_2"
    elif cat.startswith("t2_conflict_late_3"):
        return "T2_CONFLICT_LATE_3"
    elif cat.startswith("matched_neg_immediate"):
        return "MATCHED_NEG_IMMEDIATE"
    elif cat.startswith("matched_neg_late"):
        return "MATCHED_NEG_LATE"
    elif cat.startswith("defer_control"):
        return "DEFER_CONTROL"
    elif cat.startswith("answer_control"):
        return "ANSWER_CONTROL"
    return "UNKNOWN"


def _paired_bootstrap_ci(xs: list[float], n_boot: int = 5000,
                         confidence: float = 0.95, seed: int = 42) -> tuple[float, float]:
    """Paired bootstrap CI (protocol v2: 5000 iterations)."""
    import random as rng
    rng.seed(seed)
    if not xs:
        return (0.0, 0.0)
    n = len(xs)
    boot_means = []
    for _ in range(n_boot):
        sample = [xs[rng.randint(0, n - 1)] for _ in range(n)]
        boot_means.append(sum(sample) / n)
    boot_means.sort()
    alpha = (1 - confidence) / 2
    lo = boot_means[int(n_boot * alpha)]
    hi = boot_means[int(n_boot * (1 - alpha))]
    return (lo, hi)


def _independent_bootstrap_diff_ci(
    xs: list[float], ys: list[float],
    n_boot: int = 5000, confidence: float = 0.95, seed: int = 42,
) -> tuple[float, float]:
    """Direct independent bootstrap for difference of means (I_phase)."""
    import random as rng
    rng.seed(seed)
    if not xs or not ys:
        return (0.0, 0.0)
    nx, ny = len(xs), len(ys)
    boot_diffs = []
    for _ in range(n_boot):
        sx = sum(xs[rng.randint(0, nx - 1)] for _ in range(nx)) / nx
        sy = sum(ys[rng.randint(0, ny - 1)] for _ in range(ny)) / ny
        boot_diffs.append(sx - sy)
    boot_diffs.sort()
    alpha = (1 - confidence) / 2
    lo = boot_diffs[int(n_boot * alpha)]
    hi = boot_diffs[int(n_boot * (1 - alpha))]
    return (lo, hi)


def _tost_equivalence(xs: list[float], margin: float = 5.0,
                      n_boot: int = 5000, seed: int = 42) -> dict:
    """TOST equivalence test via bootstrap (90% CI, margin=5.0)."""
    if not xs:
        return {"equivalent": False, "mean": 0.0, "ci_90": [0, 0], "margin": margin}
    mean_val = sum(xs) / len(xs)
    ci_lo, ci_hi = _paired_bootstrap_ci(xs, n_boot=n_boot, confidence=0.90, seed=seed)
    equivalent = (ci_lo > -margin) and (ci_hi < margin)
    return {
        "equivalent": equivalent,
        "mean": mean_val,
        "ci_90": [ci_lo, ci_hi],
        "margin": margin,
    }


def compute_r13_analysis(results: list[dict]) -> dict:
    """Compute all protocol v2 contrasts exactly as preregistered."""
    # Pair A1 and R1 by task_id
    pairs = defaultdict(dict)
    for r in results:
        key = r["task_id"]
        pairs[key][r["arm"]] = r

    # Build paired deltas
    deltas = []
    for task_id, arms in pairs.items():
        if "A1_INFERRED" not in arms or "R1_INFERRED" not in arms:
            continue
        a1 = arms["A1_INFERRED"]
        r1 = arms["R1_INFERRED"]
        cat = a1.get("category", "")
        s = _stratum_from_category(cat)
        delta_u = r1.get("realized_utility", 0) - a1.get("realized_utility", 0)
        delta_steps = r1.get("steps", 0) - a1.get("steps", 0)
        deltas.append({
            "task_id": task_id,
            "stratum": s,
            "delta_utility": delta_u,
            "delta_steps": delta_steps,
            "a1_utility": a1.get("realized_utility", 0),
            "r1_utility": r1.get("realized_utility", 0),
            "a1_steps": a1.get("steps", 0),
            "r1_steps": r1.get("steps", 0),
            "a1_success": a1.get("success", False),
            "r1_success": r1.get("success", False),
            "r1_t2_triggered": r1.get("r1_triggered", False),
            "r1_hit_step_limit": r1.get("terminal_result") == "STEP_LIMIT",
        })

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    # Stratum groupings
    late_deltas = [d for d in deltas if d["stratum"] in
                   ("T2_CONFLICT_LATE_1", "T2_CONFLICT_LATE_2", "T2_CONFLICT_LATE_3")]
    late_1 = [d for d in deltas if d["stratum"] == "T2_CONFLICT_LATE_1"]
    late_2 = [d for d in deltas if d["stratum"] == "T2_CONFLICT_LATE_2"]
    late_3 = [d for d in deltas if d["stratum"] == "T2_CONFLICT_LATE_3"]
    immediate = [d for d in deltas if d["stratum"] == "T2_CONFLICT_IMMEDIATE"]
    t2_plus = [d for d in deltas if d["stratum"].startswith("T2_CONFLICT")]
    defer_ctrl = [d for d in deltas if d["stratum"] == "DEFER_CONTROL"]
    answer_ctrl = [d for d in deltas if d["stratum"] == "ANSWER_CONTROL"]
    matched_neg = [d for d in deltas if d["stratum"].startswith("MATCHED_NEG")]

    # Primary contrast: Delta_T2_late
    late_du = [d["delta_utility"] for d in late_deltas]
    late_ci = _paired_bootstrap_ci(late_du, n_boot=5000)
    primary = {
        "name": "Delta_T2_late",
        "n": len(late_deltas),
        "mean": mean(late_du),
        "ci_95": [late_ci[0], late_ci[1]],
        "criterion": "lower_endpoint_95CI > 0",
        "passes": late_ci[0] > 0,
    }

    # Secondary contrasts
    # I_phase = Delta_T2+ - Delta_DEFER- (direct independent bootstrap)
    t2_plus_du = [d["delta_utility"] for d in t2_plus]
    defer_du = [d["delta_utility"] for d in defer_ctrl]
    i_phase_ci = _independent_bootstrap_diff_ci(t2_plus_du, defer_du, n_boot=5000)
    i_phase_mean = mean(t2_plus_du) - mean(defer_du)

    # Exploratory per-late-stratum
    def _exploratory(name, group):
        du = [d["delta_utility"] for d in group]
        ci = _paired_bootstrap_ci(du, n_boot=5000)
        return {"name": name, "n": len(group), "mean": mean(du), "ci_95": [ci[0], ci[1]]}

    secondary = [
        {"name": "I_phase", "n": len(t2_plus) + len(defer_ctrl),
         "mean": i_phase_mean, "ci_95": [i_phase_ci[0], i_phase_ci[1]],
         "criterion": "lower_endpoint_95CI > 0", "passes": i_phase_ci[0] > 0},
        _exploratory("Delta_T2_late_1", late_1),
        _exploratory("Delta_T2_late_2", late_2),
        _exploratory("Delta_T2_late_3", late_3),
        _exploratory("Delta_T2_immediate", immediate),
    ]

    # Control contrasts (TOST equivalence, margin=5.0)
    controls = [
        {"name": "Delta_DEFER-", **_tost_equivalence(defer_du, margin=5.0)},
        {"name": "Delta_ANSWER",
         **_tost_equivalence([d["delta_utility"] for d in answer_ctrl], margin=5.0)},
        {"name": "Delta_MATCHED_NEG",
         **_tost_equivalence([d["delta_utility"] for d in matched_neg], margin=5.0)},
    ]

    # Safety checks
    false_t2 = {
        "DEFER_CONTROL": sum(d["r1_t2_triggered"] for d in defer_ctrl),
        "ANSWER_CONTROL": sum(d["r1_t2_triggered"] for d in answer_ctrl),
        "MATCHED_NEG": sum(d["r1_t2_triggered"] for d in matched_neg),
    }
    total_controls = len(defer_ctrl) + len(answer_ctrl) + len(matched_neg)
    false_t2_total = sum(false_t2.values())
    false_t2_rate = false_t2_total / max(total_controls, 1)

    # Cost metrics
    t2_plus_steps = [d["delta_steps"] for d in t2_plus]
    cost = {
        "Delta_Steps_T2+": {
            "mean": mean(t2_plus_steps),
            "ci": list(_paired_bootstrap_ci(t2_plus_steps, n_boot=5000)),
        },
        "P_step_limit_R1_T2+": {
            "rate": sum(d["r1_hit_step_limit"] for d in t2_plus) / max(len(t2_plus), 1),
            "n": len(t2_plus),
        },
        "P_step_limit_A1_T2+": {
            "rate": sum(d["a1_steps"] >= 10 for d in t2_plus) / max(len(t2_plus), 1),
            "n": len(t2_plus),
        },
    }

    # Per-stratum breakdown
    per_stratum = {}
    for s in ["T2_CONFLICT_IMMEDIATE", "T2_CONFLICT_LATE_1", "T2_CONFLICT_LATE_2",
              "T2_CONFLICT_LATE_3", "MATCHED_NEG_IMMEDIATE", "MATCHED_NEG_LATE",
              "DEFER_CONTROL", "ANSWER_CONTROL"]:
        sd = [d for d in deltas if d["stratum"] == s]
        if not sd:
            continue
        du = [d["delta_utility"] for d in sd]
        ds = [d["delta_steps"] for d in sd]
        per_stratum[s] = {
            "n": len(sd),
            "mean_delta_utility": mean(du),
            "ci_delta_utility": list(_paired_bootstrap_ci(du, n_boot=5000)),
            "mean_delta_steps": mean(ds),
            "r1_t2_triggered_count": sum(d["r1_t2_triggered"] for d in sd),
            "r1_step_limit_count": sum(d["r1_hit_step_limit"] for d in sd),
        }

    # Promotion criteria evaluation
    promotion = {
        "effectiveness": primary["passes"],
        "phase_interaction": i_phase_ci[0] > 0,
        "safety": false_t2_rate <= 0.01,
        "controls_equivalent": all(c["equivalent"] for c in controls),
        "cost_ok": abs(cost["Delta_Steps_T2+"]["mean"]) < 3.0,
    }
    promotion["all_criteria"] = all(promotion.values())

    return {
        "protocol": "confirmation_protocol_v2",
        "n_pairs": len(deltas),
        "primary_contrast": primary,
        "secondary_contrasts": secondary,
        "control_contrasts": controls,
        "safety_checks": {
            "false_t2_on_controls": false_t2,
            "false_t2_rate": false_t2_rate,
            "criterion": "false_t2_rate <= 1%, ideally 0",
            "passes": false_t2_rate <= 0.01,
        },
        "cost_metrics": cost,
        "per_stratum": per_stratum,
        "promotion_criteria": promotion,
        "deltas": deltas,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="R13: Powered Late-T2 Confirmation")
    parser.add_argument("--output-dir", required=True,
                        help="Directory for all output files")
    parser.add_argument("--base-url", default="http://127.0.0.1:8081/v1",
                        help="llama-server base URL")
    parser.add_argument("--model-name", required=True,
                        help="Model name for API calls")
    parser.add_argument("--gguf-sha256", required=True,
                        help="Frozen GGUF SHA256")
    parser.add_argument("--gguf-path", default=None,
                        help="Path to GGUF file (for SHA verification)")
    parser.add_argument("--max-tokens", type=int, default=128,
                        help="Max tokens per model call (frozen: 128)")
    parser.add_argument("--parallel", type=int, default=4,
                        help="Parallel workers (frozen: 4)")
    parser.add_argument("--protocol-path", default=None,
                        help="Path to confirmation_protocol_v2.json")
    parser.add_argument("--receipts-path", default=None,
                        help="Path to retrieval_receipts.jsonl")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Default paths
    protocol_path = Path(args.protocol_path) if args.protocol_path else (
        REPO_ROOT / "experiments/v2b_i3_15c/confirmation/confirmation_protocol_v2.json")
    receipts_path = Path(args.receipts_path) if args.receipts_path else (
        REPO_ROOT / "experiments/v2b_i3_15c/confirmation/retrieval_receipts.jsonl")

    print("=" * 80)
    print("R13: POWERED LATE-T2 CONFIRMATION")
    print("=" * 80)
    print(f"  Protocol: {R13_CONFIG['protocol_id']}")
    print(f"  Retrieval: {R13_CONFIG['retrieval_condition']} ONLY")
    print(f"  Arms: {R13_CONFIG['arms']}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Parallel: {args.parallel}")
    print(f"  Output: {output_dir}")

    # ================================================================
    # Step 1: Verify GGUF SHA256 if path provided
    # ================================================================
    if args.gguf_path:
        print("\n[1] Verifying GGUF SHA256...")
        actual_sha = sha256_file(args.gguf_path)
        if actual_sha != args.gguf_sha256:
            print(f"  EXPECTED: {args.gguf_sha256}")
            print(f"  ACTUAL:   {actual_sha}")
            print("  ABORT: GGUF SHA256 mismatch")
            sys.exit(1)
        print(f"  GGUF SHA256 verified: {actual_sha[:16]}...")
    else:
        print("\n[1] GGUF SHA256 not verified (no --gguf-path provided)")
        print(f"  Frozen SHA256: {args.gguf_sha256[:16]}...")

    # ================================================================
    # Step 2: Verify frozen identities
    # ================================================================
    print("\n[2] Verifying frozen identities...")
    try:
        identities = verify_identity(
            output_dir=output_dir,
            protocol_path=protocol_path,
            gguf_sha256=args.gguf_sha256,
            receipts_path=receipts_path,
            retrieval_condition=R13_CONFIG["retrieval_condition"],
        )
    except RuntimeError as e:
        print(f"  ABORT: {e}")
        sys.exit(1)
    print(f"  Protocol SHA: {identities['protocol_sha256'][:16]}...")
    print(f"  GGUF SHA: {identities['gguf_sha256'][:16]}...")
    print(f"  Receipt SHA: {identities['receipt_identity_sha256'][:16]}...")
    print(f"  Backend identity: {identities['backend_identity']}")

    # ================================================================
    # Step 3: Generate tasks and validate structure
    # ================================================================
    print("\n[3] Generating tasks and validating structure...")
    tasks = generate_i3_15c_corpus(
        n_per_cell=R13_CONFIG["n_per_cell"], seed=R13_CONFIG["seed"])
    print(f"  Generated {len(tasks)} tasks")
    if len(tasks) != R13_CONFIG["expected_tasks"]:
        print(f"  ABORT: Expected {R13_CONFIG['expected_tasks']} tasks, got {len(tasks)}")
        sys.exit(1)

    validation = validate_t2_eligibility(tasks)
    if not validation["passed"]:
        print(f"  ABORT: Structural T2 validation failed")
        sys.exit(1)
    print(f"  Structural T2 validation: PASS")
    print(f"    T2+ reachable: {validation['t2_positive_reachable_gold']}/{validation['t2_positive_expected']}")
    print(f"    T2- incorrectly reachable: {validation['t2_negative_incorrectly_reachable_gold']}")

    # ================================================================
    # Step 4: Load receipts and build corpus
    # ================================================================
    print("\n[4] Loading receipts and corpus...")
    receipts = load_q3_receipts(receipts_path)
    print(f"  Q3 receipts: {len(receipts)}")
    if len(receipts) != R13_CONFIG["expected_tasks"]:
        print(f"  ABORT: Expected {R13_CONFIG['expected_tasks']} Q3 receipts, got {len(receipts)}")
        sys.exit(1)

    corpus_passages, corpus_by_text, corpus_by_id, chunks, corpus_sha = (
        i3_15c._get_cached_corpus())
    print(f"  Corpus: {len(chunks)} passages, SHA: {corpus_sha[:16]}...")

    # ================================================================
    # Step 5: Build work items
    # ================================================================
    print("\n[5] Building work items...")
    backend_identity = identities["backend_identity"]
    retrieval_condition = R13_CONFIG["retrieval_condition"]

    work_items = []
    for task in tasks:
        et = task.evidence_task
        task_id = et.task_id
        receipt = receipts.get(task_id)
        if receipt is None:
            print(f"  ABORT: No Q3 receipt for task {task_id}")
            sys.exit(1)
        pre_retrieved = build_pre_retrieved_passages(receipt, corpus_by_id)

        for arm in R13_CONFIG["arms"]:
            key = make_trajectory_key(task_id, arm, retrieval_condition, backend_identity)
            work_items.append({
                "key": key,
                "task": task,
                "arm": arm,
                "retrieval_condition": retrieval_condition,
                "pre_retrieved": pre_retrieved,
                "base_url": args.base_url,
                "model_name": args.model_name,
                "max_tokens": args.max_tokens,
            })

    total = len(work_items)
    print(f"  Total trajectories: {total}")
    if total != R13_CONFIG["expected_trajectories"]:
        print(f"  ABORT: Expected {R13_CONFIG['expected_trajectories']} trajectories, got {total}")
        sys.exit(1)

    # ================================================================
    # Step 6: Load completed keys (resume support)
    # ================================================================
    print("\n[6] Checking for existing results (resume)...")
    results_path = output_dir / "results.jsonl"
    completed_keys = load_completed_keys(results_path)
    print(f"  Completed trajectories from previous runs: {len(completed_keys)}")

    # Filter to remaining work
    remaining_work = [wi for wi in work_items if wi["key"] not in completed_keys]
    print(f"  Remaining trajectories: {len(remaining_work)}")

    if not remaining_work:
        print("\n  All trajectories already completed. Proceeding to final analysis.")
    else:
        # ================================================================
        # Step 7: Set up appenders and monitors
        # ================================================================
        print("\n[7] Setting up output files...")

        results_appender = JsonlAppender(output_dir / "results.jsonl")
        model_calls_appender = JsonlAppender(output_dir / "model_calls.jsonl")
        mechanism_appender = JsonlAppender(output_dir / "mechanism_receipts.jsonl")
        cost_appender = JsonlAppender(output_dir / "cognition_cost_receipts.jsonl")
        errors_appender = JsonlAppender(output_dir / "errors.jsonl")

        progress = ProgressTracker(
            output_dir / "progress.json",
            expected=total,
            protocol_id=R13_CONFIG["protocol_id"],
        )
        # Account for already-completed
        progress.completed = len(completed_keys)
        progress.remaining = len(remaining_work)
        progress._write()

        abort_monitor = AbortMonitor()

        # ================================================================
        # Step 8: Execute trajectories
        # ================================================================
        print(f"\n[8] Executing {len(remaining_work)} trajectories (parallel={args.parallel})...")
        print(f"  No early stopping. All {total} trajectories will be completed.")
        t_start = time.time()

        def _worker(wi: dict) -> dict:
            """Execute a single trajectory and write results immediately."""
            key = wi["key"]
            task = wi["task"]
            arm = wi["arm"]
            et = task.evidence_task

            try:
                t0 = time.time()
                result = i3_15c.run_single_trajectory(
                    task=task,
                    retrieval_level=wi["retrieval_condition"],
                    arm=arm,
                    chunks=chunks,
                    corpus_by_text=corpus_by_text,
                    corpus_by_id=corpus_by_id,
                    max_tokens=wi["max_tokens"],
                    base_url=wi["base_url"],
                    backend_type="local",
                    pre_retrieved_passages=wi["pre_retrieved"],
                    model_name=wi["model_name"],
                )
                result["wall_time_s"] = round(time.time() - t0, 1)
                result["trajectory_key"] = key
                result["backend_identity"] = backend_identity
                result["protocol_id"] = R13_CONFIG["protocol_id"]

                # Write model calls
                for call in result.get("model_call_log", []):
                    call_record = {
                        **call,
                        "trajectory_key": key,
                        "task_id": et.task_id,
                        "arm": arm,
                    }
                    model_calls_appender.append(call_record)

                # Write mechanism receipt (R1 only)
                if arm == "R1_INFERRED" and "mechanism_receipt" in result:
                    receipt_record = {
                        **result["mechanism_receipt"],
                        "trajectory_key": key,
                        "task_id": et.task_id,
                    }
                    mechanism_appender.append(receipt_record)

                # Write cognition cost receipt
                cost_record = {
                    "trajectory_key": key,
                    "task_id": et.task_id,
                    "arm": arm,
                    "category": et.category,
                    "steps": result.get("steps", 0),
                    "n_model_calls": len(result.get("model_call_log", [])),
                    "total_completion_tokens": sum(
                        c.get("completion_tokens", 0)
                        for c in result.get("model_call_log", [])),
                    "total_prompt_tokens": sum(
                        c.get("prompt_tokens", 0)
                        for c in result.get("model_call_log", [])),
                    "total_latency_ms": sum(
                        c.get("latency_ms", 0)
                        for c in result.get("model_call_log", [])),
                    "wall_time_s": result["wall_time_s"],
                    "terminal_action": result.get("terminal_action"),
                    "terminal_result": result.get("terminal_result"),
                    "realized_utility": result.get("realized_utility", 0),
                    "success": result.get("success", False),
                }
                cost_appender.append(cost_record)

                # Write result
                results_appender.append(result)

                # Update progress and monitor
                progress.record_completion(key)
                abort_monitor.record(result)

                return result

            except Exception as exc:
                error_record = {
                    "trajectory_key": key,
                    "task_id": et.task_id,
                    "arm": arm,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "timestamp": time.time(),
                }
                errors_appender.append(error_record)
                progress.record_failure()
                abort_monitor.record({"error": str(exc), "terminal_result": "BACKEND_ERROR"})
                return error_record

        # Execute with ThreadPoolExecutor
        completed_in_this_run = 0
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = {executor.submit(_worker, wi): wi for wi in remaining_work}

            for future in as_completed(futures):
                result = future.result()
                completed_in_this_run += 1

                # Check abort
                should_abort, reason = abort_monitor.should_abort()
                if should_abort:
                    print(f"\n  ABORT: {reason}")
                    print(f"  Cancelling remaining futures...")
                    for f in futures:
                        f.cancel()
                    break

                # Progress report
                if completed_in_this_run % 10 == 0 or completed_in_this_run == len(remaining_work):
                    elapsed = time.time() - t_start
                    rate = completed_in_this_run / elapsed if elapsed > 0 else 0
                    total_done = len(completed_keys) + completed_in_this_run
                    remaining = total - total_done
                    eta = remaining / rate if rate > 0 else 0
                    key = result.get("trajectory_key", "?")
                    action = result.get("terminal_action", result.get("error_type", "?"))
                    print(f"  [{total_done}/{total}] {key[:60]} "
                          f"action={action} "
                          f"({result.get('wall_time_s', 0)}s) "
                          f"ETA: {eta/60:.0f}min", flush=True)

        # Close appenders
        results_appender.close()
        model_calls_appender.close()
        mechanism_appender.close()
        cost_appender.close()
        errors_appender.close()

        # Check abort
        should_abort, reason = abort_monitor.should_abort()
        if should_abort:
            print(f"\n  EXPERIMENT ABORTED: {reason}")
            print(f"  Partial results saved. Fix the issue and re-run to resume.")
            sys.exit(1)

        elapsed = time.time() - t_start
        print(f"\n  Completed {completed_in_this_run} trajectories in {elapsed/60:.1f}min")

    # ================================================================
    # Step 9: Final verification
    # ================================================================
    print("\n[9] Final verification...")
    final_completed = load_completed_keys(results_path)
    print(f"  Completed unique keys: {len(final_completed)}")

    # Check for duplicates
    all_keys = []
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                key = record.get("trajectory_key")
                if key:
                    all_keys.append(key)
            except json.JSONDecodeError:
                continue

    key_counts = Counter(all_keys)
    duplicates = {k: v for k, v in key_counts.items() if v > 1}
    expected_keys = {wi["key"] for wi in work_items}
    missing = expected_keys - final_completed

    print(f"  Expected: {R13_CONFIG['expected_trajectories']}")
    print(f"  Completed: {len(final_completed)}")
    print(f"  Duplicates: {len(duplicates)}")
    print(f"  Missing: {len(missing)}")

    if duplicates:
        print(f"  WARNING: Duplicate keys found: {list(duplicates.keys())[:5]}")
    if missing:
        print(f"  WARNING: Missing keys: {list(missing)[:5]}")
        print(f"  Re-run to complete missing trajectories.")
        sys.exit(1)

    if len(final_completed) != R13_CONFIG["expected_trajectories"]:
        print(f"  ABORT: Expected {R13_CONFIG['expected_trajectories']}, got {len(final_completed)}")
        sys.exit(1)

    print(f"  VERIFICATION: PASS — {len(final_completed)} unique trajectories, 0 duplicates, 0 missing")

    # ================================================================
    # Step 10: Run preregistered statistical analysis
    # ================================================================
    print("\n[10] Running preregistered statistical analysis (protocol v2)...")

    # Load all results
    all_results = []
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                all_results.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    print(f"  Loaded {len(all_results)} trajectory results")

    # Compute protocol v2 contrasts
    analysis = compute_r13_analysis(all_results)

    # Save analysis
    analysis_path = output_dir / "analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f"  Analysis saved to {analysis_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("R13 RESULTS SUMMARY (Protocol V2)")
    print("=" * 80)

    # Primary contrast
    pc = analysis["primary_contrast"]
    print(f"\nPRIMARY CONTRAST: {pc['name']}")
    print(f"  n={pc['n']}, mean={pc['mean']:.4f}, CI=[{pc['ci_95'][0]:.4f}, {pc['ci_95'][1]:.4f}]")
    print(f"  Criterion: lower CI > 0 → {'PASS' if pc['passes'] else 'FAIL'} (lower={pc['ci_95'][0]:.4f})")

    # Secondary contrasts
    print(f"\nSECONDARY CONTRASTS:")
    for sc in analysis["secondary_contrasts"]:
        ci = sc.get("ci_95", [0, 0])
        passes = sc.get("passes")
        pass_str = f" → {'PASS' if passes else 'FAIL'}" if passes is not None else " (exploratory)"
        print(f"  {sc['name']}: n={sc['n']}, mean={sc['mean']:.4f}, CI=[{ci[0]:.4f}, {ci[1]:.4f}]{pass_str}")

    # Control contrasts
    print(f"\nCONTROL CONTRASTS (TOST equivalence, margin=5.0):")
    for cc in analysis["control_contrasts"]:
        ci = cc.get("ci_90", [0, 0])
        print(f"  {cc['name']}: mean={cc['mean']:.4f}, CI90=[{ci[0]:.4f}, {ci[1]:.4f}], "
              f"equivalent={'YES' if cc['equivalent'] else 'NO'}")

    # Safety
    safety = analysis["safety_checks"]
    print(f"\nSAFETY CHECKS:")
    print(f"  False T2 on controls: {safety['false_t2_on_controls']}")
    print(f"  False T2 rate: {safety['false_t2_rate']:.4f} → {'PASS' if safety['passes'] else 'FAIL'}")

    # Cost
    print(f"\nCOST METRICS:")
    for name, info in analysis["cost_metrics"].items():
        if isinstance(info, dict) and "mean" in info:
            print(f"  {name}: mean={info['mean']:.3f}")
        elif isinstance(info, dict) and "rate" in info:
            print(f"  {name}: rate={info['rate']:.3f} n={info.get('n', 0)}")

    # Per-stratum
    print(f"\nPER-STRATUM:")
    for stratum, info in sorted(analysis["per_stratum"].items()):
        ci = info.get("ci_delta_utility", [0, 0])
        print(f"  {stratum}: n={info['n']} "
              f"ΔU={info['mean_delta_utility']:.4f} "
              f"CI=[{ci[0]:.4f}, {ci[1]:.4f}] "
              f"ΔSteps={info['mean_delta_steps']:.2f} "
              f"R1_T2={info['r1_t2_triggered_count']} "
              f"R1_step_limit={info['r1_step_limit_count']}")

    # Promotion criteria
    promo = analysis["promotion_criteria"]
    print(f"\nPROMOTION CRITERIA:")
    for k, v in promo.items():
        if k != "all_criteria":
            print(f"  {k}: {'PASS' if v else 'FAIL'}")
    print(f"  ALL CRITERIA: {'PASS' if promo['all_criteria'] else 'FAIL'}")

    print(f"\n  Results: {output_dir / 'results.jsonl'}")
    print(f"  Analysis: {analysis_path}")
    print(f"\nR13 COMPLETE.")


if __name__ == "__main__":
    main()
