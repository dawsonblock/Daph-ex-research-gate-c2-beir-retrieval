#!/usr/bin/env python3
"""I3.5-PQ Phase 19: Six-arm executive experiment.

Runs six paired arms on the I3.5 benchmark tasks:
  P0:  no guidance (base MDSG packet)
  B0:  global prior
  B1:  phase x action observational baseline
  PS05: strong fixed challenge prior (shuffled B1)
  QOBS: state-conditioned observational estimator
  QCAUSAL: promoted causal pinned-policy estimator

All estimators are loaded from frozen artifacts. No .fit() in the live path.

The runner mirrors run_r2_dev_v2.py's control loop but adds action value
estimates to the MDSG packet based on the arm's estimator.

Usage:
  PYTHONPATH=. python3 scripts/run_i3_5_six_arm.py \
    --gguf-path /path/to/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
    --output-dir experiments/i3_5/six_arm \
    --resume

Output:
  experiments/i3_5/six_arm/trajectories_v1.jsonl
  experiments/i3_5/six_arm/receipts_v1.jsonl
  experiments/i3_5/six_arm/model_calls_v1.jsonl
  experiments/i3_5/six_arm/run_manifest_v1.json
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pickle
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


# ============================================================
# Feature extraction (must match freeze_i3_5_estimators.py)
# ============================================================

def extract_features(state_features: dict, action: str) -> dict:
    feats = {
        "n_live": state_features.get("n_live", 0),
        "n_eliminated": state_features.get("n_eliminated", 0),
        "n_untested": state_features.get("n_untested", 0),
        "n_total_hypotheses": state_features.get("n_total_hypotheses", 0),
        "n_visible_evidence": state_features.get("n_visible_evidence", 0),
        "n_verified": state_features.get("n_verified", 0),
        "n_supporting": state_features.get("n_supporting", 0),
        "n_contradicting": state_features.get("n_contradicting", 0),
        "n_stale": state_features.get("n_stale", 0),
        "retrieval_remaining": state_features.get("retrieval_remaining", 0),
        "search_remaining": state_features.get("search_remaining", 0),
        "verify_remaining": state_features.get("verify_remaining", 0),
        "steps_remaining": state_features.get("steps_remaining", 0),
        "can_retrieve": int(state_features.get("can_retrieve", False)),
        "can_search": int(state_features.get("can_search", False)),
        "can_verify": int(state_features.get("can_verify", False)),
        "searched": int(state_features.get("searched", False)),
        "reasoning_complete": int(state_features.get("reasoning_complete", False)),
        "same_action_run_length": state_features.get("same_action_run_length", 0),
        "retrieval_count": state_features.get("retrieval_count", 0),
        "search_count": state_features.get("search_count", 0),
        "verify_count": state_features.get("verify_count", 0),
    }
    for a in ["ANSWER", "DEFER", "RETRIEVE", "VERIFY", "SEARCH_MORE", "REASON_MORE"]:
        feats[f"a_{a}"] = int(action == a)
    feats["n_live_x_retrieve"] = feats["n_live"] * feats["a_RETRIEVE"]
    feats["n_live_x_verify"] = feats["n_live"] * feats["a_VERIFY"]
    feats["n_live_x_search"] = feats["n_live"] * feats["a_SEARCH_MORE"]
    feats["n_untested_x_retrieve"] = feats["n_untested"] * feats["a_RETRIEVE"]
    feats["n_untested_x_verify"] = feats["n_untested"] * feats["a_VERIFY"]
    feats["n_supporting_x_answer"] = feats["n_supporting"] * feats["a_ANSWER"]
    feats["n_eliminated_x_defer"] = feats["n_eliminated"] * feats["a_DEFER"]
    return feats


# ============================================================
# Frozen estimator loader
# ============================================================

class FrozenEstimator:
    """Loads and wraps a frozen estimator for live prediction."""

    def __init__(self, arm: str, estimator_dir: Path):
        self.arm = arm
        self.estimator_dir = estimator_dir
        self.feature_keys: list[str] = []
        self.b0_value: float = 0.0
        self.b1_table: dict = {}
        self.b1_global: dict = {}
        self.ps05_mapping: dict = {}
        self.qobs_model = None
        self.qcausal_model = None
        self._load()

    def _load(self):
        # Load feature schema
        with open(self.estimator_dir / "feature_schema.json") as f:
            schema = json.load(f)
        self.feature_keys = schema["feature_keys"]

        if self.arm == "P0":
            return  # No estimator

        if self.arm == "B0":
            with open(self.estimator_dir / "B0_global_mean.json") as f:
                data = json.load(f)
            self.b0_value = data["value"]

        elif self.arm == "B1":
            with open(self.estimator_dir / "B1_phase_action_table.json") as f:
                data = json.load(f)
            self.b1_table = data["phase_action_table"]
            self.b1_global = data["global_action_mean"]

        elif self.arm == "PS05":
            with open(self.estimator_dir / "PS05_shuffled_mapping.json") as f:
                data = json.load(f)
            self.ps05_mapping = data["mapping"]
            # Also load B1 for fallback
            with open(self.estimator_dir / "B1_phase_action_table.json") as f:
                b1_data = json.load(f)
            self.b1_table = b1_data["phase_action_table"]
            self.b1_global = b1_data["global_action_mean"]

        elif self.arm == "QOBS":
            with open(self.estimator_dir / "QOBS_gbt.pkl", "rb") as f:
                self.qobs_model = pickle.load(f)

        elif self.arm == "QCAUSAL":
            with open(self.estimator_dir / "QCAUSAL_gbt.pkl", "rb") as f:
                self.qcausal_model = pickle.load(f)

    def predict_action_values(self, state_features: dict, legal_actions: list[str],
                               phase: str = "UNKNOWN") -> dict[str, float]:
        """Predict Q values for each legal action."""
        import numpy as np

        if self.arm == "P0":
            return {}  # No values

        values = {}

        if self.arm == "B0":
            for a in legal_actions:
                values[a] = self.b0_value

        elif self.arm == "B1":
            phase_table = self.b1_table.get(phase, self.b1_global)
            for a in legal_actions:
                values[a] = phase_table.get(a, self.b1_global.get(a, 0.0))

        elif self.arm == "PS05":
            phase_mapping = self.ps05_mapping.get(phase, {})
            for a in legal_actions:
                values[a] = phase_mapping.get(a, self.b1_global.get(a, 0.0))

        elif self.arm == "QOBS":
            X = np.array([[extract_features(state_features, a)[k] for k in self.feature_keys]
                          for a in legal_actions])
            preds = self.qobs_model.predict(X)
            for a, p in zip(legal_actions, preds):
                values[a] = float(p)

        elif self.arm == "QCAUSAL":
            X = np.array([[extract_features(state_features, a)[k] for k in self.feature_keys]
                          for a in legal_actions])
            preds = self.qcausal_model.predict(X)
            for a, p in zip(legal_actions, preds):
                values[a] = float(p)

        return values

    def build_packet_values(self, state_features: dict, legal_actions: list[str],
                             phase: str = "UNKNOWN") -> tuple[dict | None, list[str] | None]:
        """Build action_value_estimates and ranking for the packet.

        Returns (estimates_dict, ranking_list) or (None, None) for P0.
        """
        if self.arm == "P0":
            return None, None

        raw_values = self.predict_action_values(state_features, legal_actions, phase)

        # Normalize to [0, 1]
        if raw_values:
            min_v = min(raw_values.values())
            max_v = max(raw_values.values())
            range_v = max_v - min_v + 1e-8
            normalized = {a: round((v - min_v) / range_v, 4) for a, v in raw_values.items()}
        else:
            normalized = {}

        estimates = {
            a: {"normalized_value": normalized.get(a, 0.0)}
            for a in legal_actions
        }
        ranking = sorted(legal_actions, key=lambda a: -normalized.get(a, 0.0))

        return estimates, ranking


# ============================================================
# Phase classification (simplified from i3_7e)
# ============================================================

def classify_phase_simple(state_features: dict) -> str:
    n_live = state_features.get("n_live", 0)
    n_eliminated = state_features.get("n_eliminated", 0)
    n_untested = state_features.get("n_untested", 0)
    n_supporting = state_features.get("n_supporting", 0)

    if n_eliminated > 0 and n_live == 0:
        return "T2"
    elif n_supporting > 0 and n_live <= 1:
        return "READY"
    elif n_untested > 0:
        return "EXPLORE"
    else:
        return "DISCRIMINATE"


# ============================================================
# Trajectory runner
# ============================================================

@dataclass
class SixArmWriter:
    output_dir: Path

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.traj_file = open(self.output_dir / "trajectories_v1.jsonl", "a")
        self.receipt_file = open(self.output_dir / "receipts_v1.jsonl", "a")
        self.model_call_file = open(self.output_dir / "model_calls_v1.jsonl", "a")
        self.error_file = open(self.output_dir / "errors_v1.jsonl", "a")

    def _write(self, fh, record: dict):
        fh.write(json.dumps(record, sort_keys=True, default=bool) + "\n")
        fh.flush()
        os.fsync(fh.fileno())

    def write_trajectory(self, r: dict): self._write(self.traj_file, r)
    def write_receipt(self, r: dict): self._write(self.receipt_file, r)
    def write_model_call(self, r: dict): self._write(self.model_call_file, r)
    def write_error(self, r: dict): self._write(self.error_file, r)

    def load_completed_keys(self) -> set[str]:
        completed = set()
        path = self.output_dir / "trajectories_v1.jsonl"
        if not path.exists():
            return completed
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    completed.add(f"{r['task_id']}:{r['arm']}")
                except (json.JSONDecodeError, KeyError):
                    continue
        return completed


def run_trajectory(
    task,
    arm: str,
    estimator: FrozenEstimator,
    backend,
    i3_7e,
    utility,
    budget,
    max_steps: int = 10,
    writer: SixArmWriter | None = None,
    run_id: str = "",
) -> dict:
    """Run a single trajectory with the given arm."""

    from hrm_adaptive_memory.executive.evidence_benchmark import build_evidence_snapshot
    from hrm_adaptive_memory.executive.evidence_benchmark.executor import EvidenceExecutor
    from hrm_adaptive_memory.executive.evidence_benchmark.schema import initial_evidence_runtime
    from hrm_adaptive_memory.executive.resources import ResourceState
    from hrm_adaptive_memory.cognitive_control.core import DecisionAction
    from hrm_adaptive_memory.executive.model_decoder import decode_output_strict
    from r2_schema import build_action_schema, schema_sha256, schema_action_enum
    from r2_allowed_actions import compute_allowed_actions, ActionState, EmptyAllowedActionSet, C0

    executor = EvidenceExecutor()
    resources = ResourceState(budget=budget)
    runtime = initial_evidence_runtime(task, resources)

    prior_actions: list[str] = []
    prior_outcomes: list[str] = []
    n_hypotheses = len(task.hypotheses)
    realized = 0.0
    success = False
    terminal = False
    terminal_action = None
    terminal_result = "STEP_LIMIT"
    premature_defer = False
    premature_answer = False
    model_calls = 0
    backend_errors = 0
    decoder_errors = 0
    steps_taken = 0
    actions_taken: list[str] = []
    system_prompt = i3_7e.MDSG_STATE_WITH_AFFORDANCES_SYSTEM_PROMPT

    for step_id in range(max_steps):
        # Build snapshot
        evidence_snapshot = build_evidence_snapshot(
            runtime,
            prior_actions=tuple(prior_actions),
            prior_outcomes=tuple(prior_outcomes),
        )

        # Compute T2
        viability = i3_7e._classify_from_snapshot(evidence_snapshot)
        eliminated = [h_id for h_id, info in viability.items()
                      if info["status"] == "ELIMINATED"]
        t2 = (len(eliminated) == n_hypotheses and n_hypotheses > 0)

        # Compute allowed actions (C0 = neutral, no gates)
        action_state = ActionState(
            t2=t2,
            executive_steps_remaining=max_steps - step_id,
            can_retrieve=evidence_snapshot.can_retrieve,
            can_search=evidence_snapshot.can_search,
            can_verify=evidence_snapshot.can_verify,
        )

        try:
            allowed_decision = compute_allowed_actions(action_state, C0)
        except EmptyAllowedActionSet:
            terminal_result = "EMPTY_ALLOWED_SET"
            break

        # Build MDSG packet
        packet = i3_7e.build_mdsg_state_with_affordances_packet(evidence_snapshot)

        # Add action value estimates based on arm
        state_features = evidence_snapshot.__dict__ if hasattr(evidence_snapshot, '__dict__') else {}
        # Use compute_state_features for consistent feature extraction
        from daph.intervention.checkpoint import compute_state_features
        sf = compute_state_features(runtime, tuple(prior_actions))
        phase = classify_phase_simple(sf)
        legal_actions = sorted(allowed_decision.allowed)

        estimates, ranking = estimator.build_packet_values(sf, legal_actions, phase)
        if estimates is not None:
            packet["action_value_estimates"] = estimates
        if ranking is not None:
            packet["action_value_ranking"] = ranking
        if arm != "P0":
            packet["epistemic_phase"] = phase

        # Build schema and prompts
        schema = build_action_schema(allowed_decision.allowed)
        schema_sha = schema_sha256(schema)
        user_prompt = i3_7e.evidence_packet_json(packet)

        # Call Qwen
        model_calls += 1
        try:
            call_result = backend.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.0,
                max_tokens=256,
                allowed_actions=allowed_decision.allowed,
            )
        except Exception as exc:
            backend_errors += 1
            if writer:
                writer.write_error({
                    "run_id": run_id, "task_id": task.task_id, "arm": arm,
                    "step": step_id, "error": "BackendError",
                    "error_type": type(exc).__name__, "error_message": str(exc),
                })
            terminal_result = "BACKEND_ERROR"
            break

        raw_output = call_result.raw_output
        outcome = decode_output_strict(raw_output)

        # Record model call
        if writer:
            writer.write_model_call({
                "run_id": run_id, "task_id": task.task_id, "arm": arm,
                "step": step_id, "raw_output": raw_output,
                "decoder_valid": bool(outcome.valid),
                "schema_sha256": schema_sha,
                "json_schema_sha256": getattr(call_result, "json_schema_sha256", ""),
                "system_prompt_sha256": getattr(call_result, "system_prompt_sha256", ""),
                "user_packet_sha256": getattr(call_result, "user_packet_sha256", ""),
                "prompt_tokens": getattr(call_result, "prompt_tokens", 0),
                "completion_tokens": getattr(call_result, "completion_tokens", 0),
                "latency_ms": getattr(call_result, "latency_ms", 0),
                "arm": arm, "phase": phase,
                "packet_has_values": estimates is not None,
                "packet_has_ranking": ranking is not None,
            })

        if not outcome.valid or not outcome.proposal:
            decoder_errors += 1
            terminal_result = "DECODER_ERROR"
            break

        proposal = outcome.proposal
        action_str = proposal.action.value if hasattr(proposal.action, "value") else str(proposal.action)
        target_id = getattr(proposal, "target_id", None)

        # Admissibility check
        if action_str not in allowed_decision.allowed:
            terminal_result = "ADMISSIBILITY_VIOLATION"
            break

        # Execute
        action = DecisionAction(action_str)
        resources_before = runtime.resources
        exec_res = executor.execute(runtime, action, target_evidence_id=target_id)
        runtime = exec_res.runtime
        resources_after = runtime.resources

        step_cost = utility.action_cost(resources_before, resources_after)
        realized -= step_cost

        actions_taken.append(action_str)
        prior_actions.append(action_str)
        prior_outcomes.append(exec_res.outcome_code)
        steps_taken += 1

        if exec_res.terminal:
            tr = utility.terminal_reward(exec_res.action, bool(exec_res.task_success))
            realized += tr
            success = bool(exec_res.task_success)
            terminal = True
            terminal_action = action_str
            terminal_result = exec_res.outcome_code
            if action is DecisionAction.DEFER and not exec_res.task_success:
                premature_defer = True
            if action is DecisionAction.ANSWER and not exec_res.task_success:
                premature_answer = True
            break

    if not terminal:
        realized -= 0.5  # step limit penalty

    result = {
        "run_id": run_id,
        "task_id": task.task_id,
        "arm": arm,
        "realized_utility": round(realized, 4),
        "success": bool(success),
        "steps": steps_taken,
        "terminal_action": terminal_action,
        "terminal_result": terminal_result,
        "model_calls": model_calls,
        "backend_errors": backend_errors,
        "decoder_errors": decoder_errors,
        "premature_defer": bool(premature_defer),
        "premature_answer": bool(premature_answer),
        "actions_taken": actions_taken,
        "n_hypotheses": n_hypotheses,
    }

    if writer:
        writer.write_trajectory(result)

    return result


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="I3.5-PQ Phase 19 Six-Arm Executive Experiment")
    parser.add_argument("--gguf-path", required=True)
    parser.add_argument("--model-name", default="qwen2.5-7b-instruct")
    parser.add_argument("--output-dir", default="experiments/i3_5/six_arm")
    parser.add_argument("--estimator-dir", default="experiments/i3_5/pinned_policy/frozen_estimators")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--n-per-subtype", type=int, default=24,
                        help="Number of tasks per one-live subtype")
    parser.add_argument("--n-per-two-live", type=int, default=20,
                        help="Number of tasks per two-live subtype")
    args = parser.parse_args()

    output_dir = REPO_ROOT / args.output_dir
    estimator_dir = REPO_ROOT / args.estimator_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load i3_7e
    print("Loading i3_7e module...")
    spec = importlib.util.spec_from_file_location(
        "i3_7e", str(REPO_ROOT / "scripts" / "run_i3_7e_compact_governor.py"))
    i3_7e = importlib.util.module_from_spec(spec)
    sys.modules["i3_7e"] = i3_7e
    spec.loader.exec_module(i3_7e)

    # Load tasks
    print("Loading benchmark tasks...")
    from hrm_adaptive_memory.executive.evidence_benchmark.i3_5_state_discrimination_generator import (
        generate_i3_5_state_discrimination_benchmark,
    )
    tasks = generate_i3_5_state_discrimination_benchmark(
        n_per_subtype=args.n_per_subtype,
        n_per_two_live_subtype=args.n_per_two_live,
        seed=9137,
    )
    print(f"  {len(tasks)} tasks")

    # Load utility
    from hrm_adaptive_memory.executive.metareasoning_utility import MetareasoningUtility
    utility = MetareasoningUtility.from_file(REPO_ROOT / "configs/v2b_i3_1_utility_v1.json")

    # Load budget
    from hrm_adaptive_memory.executive.resources import ResourceBudget
    budget = ResourceBudget(
        max_executive_steps=10, max_retrieval_calls=3,
        max_search_calls=2, max_verification_calls=5,
    )

    # Load frozen estimators
    print("\nLoading frozen estimators...")
    arms = ["P0", "B0", "B1", "PS05", "QOBS", "QCAUSAL"]
    estimators = {}
    for arm in arms:
        estimators[arm] = FrozenEstimator(arm, estimator_dir)
        print(f"  {arm} loaded")

    # Load estimator manifest
    with open(estimator_dir / "estimator_manifest.json") as f:
        estimator_manifest = json.load(f)

    # Initialize backend
    print(f"\nInitializing R2DirectLlamaBackend with {args.gguf_path}...")
    from hrm_adaptive_memory.executive.model_backend import R2DirectLlamaBackend
    backend = R2DirectLlamaBackend(
        model_name=args.model_name,
        model_path=args.gguf_path,
        n_ctx=args.n_ctx,
    )
    print("  Backend initialized.")

    # Compute backend SHA
    backend_id_content = json.dumps({
        "model_name": args.model_name,
        "gguf_path": args.gguf_path,
        "n_ctx": args.n_ctx,
        "temperature": 0.0,
        "max_tokens": 256,
        "seed": 42,
    }, sort_keys=True)
    backend_sha = hashlib.sha256(backend_id_content.encode()).hexdigest()

    # Build run manifest (before trajectory 1)
    timestamp = datetime.now(timezone.utc).isoformat()
    run_id = hashlib.sha256(f"i3_5_six_arm:{timestamp}".encode()).hexdigest()[:16]

    # Compute source SHAs
    source_shas = {}
    for fname in ["run_i3_7e_compact_governor.py", "r2_schema.py", "r2_allowed_actions.py"]:
        path = REPO_ROOT / "scripts" / fname
        if path.exists():
            source_shas[fname] = hashlib.sha256(path.read_bytes()).hexdigest()

    for fname in ["model_backend.py", "model_decoder.py", "metareasoning_utility.py"]:
        path = REPO_ROOT / "hrm_adaptive_memory" / "executive" / fname
        if path.exists():
            source_shas[fname] = hashlib.sha256(path.read_bytes()).hexdigest()

    # Benchmark SHA
    benchmark_path = REPO_ROOT / "experiments/i3_5/datasets/state_discrimination_v1.jsonl"
    benchmark_sha = hashlib.sha256(benchmark_path.read_bytes()).hexdigest() if benchmark_path.exists() else "unknown"

    # Utility SHA
    utility_path = REPO_ROOT / "configs/v2b_i3_1_utility_v1.json"
    utility_sha = hashlib.sha256(utility_path.read_bytes()).hexdigest()

    # Causal dataset SHA
    causal_dataset_path = REPO_ROOT / "experiments/i3_5/pinned_policy/pinned_causal_actions_v1.jsonl"
    causal_dataset_sha = hashlib.sha256(causal_dataset_path.read_bytes()).hexdigest() if causal_dataset_path.exists() else "unknown"

    # Packet builder SHA
    packet_builder_path = REPO_ROOT / "scripts/run_i3_7e_compact_governor.py"
    packet_builder_sha = hashlib.sha256(packet_builder_path.read_bytes()).hexdigest()

    run_manifest = {
        "run_id": run_id,
        "experiment": "I3.5-PQ Phase 19 Six-Arm Executive Experiment",
        "timestamp": timestamp,
        "arms": arms,
        "n_tasks": len(tasks),
        "n_trajectories": len(tasks) * len(arms),
        "binding": {
            "model_name": args.model_name,
            "gguf_path": args.gguf_path,
            "backend_sha256": backend_sha,
            "causal_dataset_sha256": causal_dataset_sha,
            "feature_schema_sha256": estimator_manifest["feature_schema"]["sha256"],
            "packet_builder_sha256": packet_builder_sha,
            "benchmark_sha256": benchmark_sha,
            "utility_sha256": utility_sha,
            "estimator_manifest_sha256": hashlib.sha256(
                (estimator_dir / "estimator_manifest.json").read_bytes()).hexdigest(),
        },
        "source_shas": source_shas,
        "estimator_manifest": estimator_manifest,
        "preregistered_endpoints": [
            "success_rate (per arm)",
            "paired_rescues_breaks (McNemar exact test)",
            "premature_defer_rate",
            "premature_answer_rate",
            "loop_rate",
            "resource_exhaustion_rate",
            "mean_steps",
            "mean_tokens",
            "mean_causal_regret_of_chosen_action",
            "near_optimal_action_rate_epsilon3",
            "stratified_by_gap_bucket (clear >10, moderate 3-10, near_tie <=3)",
            "delta_U_QCAUSAL_vs_P0",
            "delta_U_QCAUSAL_vs_B0",
            "delta_U_QCAUSAL_vs_B1",
            "delta_U_QCAUSAL_vs_PS05",
            "delta_U_QCAUSAL_vs_QOBS",
        ],
        "primary_contrasts": [
            "QCAUSAL > B0 (paired 95% CI excluding zero)",
            "QCAUSAL > QOBS (paired 95% CI excluding zero)",
        ],
        "promotion_rule": [
            "delta_U_QCAUSAL-B0 > 0 with paired 95% CI excluding zero",
            "improved or non-inferior success rate",
            "no increase in premature DEFER or ANSWER",
            "materially better near-optimal action selection on gap>3 states",
        ],
        "invariants": [
            "No .fit() in the live path",
            "All estimators loaded from frozen artifacts",
            "Same Qwen backend for all arms",
            "Same system prompt for all arms",
            "Same schema enforcement for all arms",
            "Same utility for all arms",
            "temperature=0 for deterministic generation",
        ],
    }
    manifest_path = output_dir / "run_manifest_v1.json"
    with open(manifest_path, "w") as f:
        json.dump(run_manifest, f, indent=2, sort_keys=True)
    print(f"\nRun manifest: {manifest_path}")
    print(f"  Run ID: {run_id}")
    print(f"  Arms: {arms}")
    print(f"  Tasks: {len(tasks)}")
    print(f"  Total trajectories: {len(tasks) * len(arms)}")

    # Setup writer
    writer = SixArmWriter(output_dir)
    completed_keys = writer.load_completed_keys() if args.resume else set()
    if completed_keys:
        print(f"Resume: {len(completed_keys)} trajectories already completed")

    # Run trajectories
    n_total = len(tasks) * len(arms)
    n_done = len(completed_keys)
    n_success_by_arm = defaultdict(int)
    start_time = time.time()

    print(f"\nStarting six-arm experiment: {n_total - n_done} trajectories remaining")
    print()

    for task in tasks:
        for arm in arms:
            traj_key = f"{task.task_id}:{arm}"
            if traj_key in completed_keys:
                continue

            result = run_trajectory(
                task=task,
                arm=arm,
                estimator=estimators[arm],
                backend=backend,
                i3_7e=i3_7e,
                utility=utility,
                budget=budget,
                max_steps=args.max_steps,
                writer=writer,
                run_id=run_id,
            )

            n_done += 1
            if result["success"]:
                n_success_by_arm[arm] += 1

            if n_done % 50 == 0:
                elapsed = time.time() - start_time
                rate = (n_done - len(completed_keys)) / elapsed if elapsed > 0 else 0
                remaining = n_total - n_done
                eta = remaining / rate if rate > 0 else 0
                print(f"  Progress: {n_done}/{n_total} ({rate:.1f}/s, ETA {eta:.0f}s) "
                      f"last: {task.task_id}:{arm} U={result['realized_utility']:.1f} "
                      f"success={result['success']}")

    # Summary
    print(f"\n{'='*70}")
    print(f"Six-Arm Experiment Complete")
    print(f"{'='*70}")
    print(f"  Total trajectories: {n_done}")
    print(f"  By arm:")
    for arm in arms:
        print(f"    {arm:10s}: {n_success_by_arm[arm]} successes")

    print(f"\n  Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
