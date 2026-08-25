#!/usr/bin/env python3
"""I3.5-PQ Phase 21: Interface-ablation experiment.

Tests five interface variants against the current normalized interface,
using the FROZEN QCAUSAL_V1 estimator (no retraining).

Arms:
  C0    — no guidance (baseline)
  I0    — current normalized values (known problematic)
  I1    — raw centered advantages: A(s,a) = Q(s,a) - max_b Q(s,b)
  I2    — epsilon near-optimal set (no artificial numerical distinction)
  I3    — confidence-aware recommendation (gap/confidence-aware)
  I4    — single recommendation only on clear-choice states (gap > 3)

The QCAUSAL estimator is loaded from frozen artifacts and never retrained.
Only the packet representation changes.

Usage:
  PYTHONPATH=. python3 scripts/run_i3_5_interface_ablation.py \
    --gguf-path /path/to/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
    --output-dir experiments/i3_5/interface_ablation \
    --resume
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

import numpy as np

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
# Phase classification
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
# Interface variants
# ============================================================

class InterfaceVariant:
    """Builds the action-value portion of the MDSG packet for a given interface."""

    def __init__(self, variant: str, qcausal_model, feature_keys: list[str],
                 epsilon: float = 3.0, tau: float = 3.0):
        self.variant = variant
        self.model = qcausal_model
        self.feature_keys = feature_keys
        self.epsilon = epsilon  # for I2 (near-optimal set) and I4 (clear-choice threshold)
        self.tau = tau          # for I3 (confidence threshold)

    def predict_q(self, state_features: dict, legal_actions: list[str]) -> dict[str, float]:
        """Get raw Q predictions from the frozen QCAUSAL model."""
        X = np.array([[extract_features(state_features, a)[k] for k in self.feature_keys]
                      for a in legal_actions])
        preds = self.model.predict(X)
        return dict(zip(legal_actions, [float(p) for p in preds]))

    def build_packet_fields(self, state_features: dict, legal_actions: list[str],
                             phase: str) -> tuple[dict | None, list[str] | None,
                                                  dict | None]:
        """Build the action-value fields for the packet.

        Returns (extra_fields, ranking_or_none, estimates_or_none).
        extra_fields are added directly to the packet.
        """
        if self.variant == "C0":
            return None, None, None

        raw_q = self.predict_q(state_features, legal_actions)
        q_max = max(raw_q.values())
        q_sorted = sorted(raw_q.items(), key=lambda x: -x[1])
        q_best = q_sorted[0][0]
        q_second = q_sorted[1][1] if len(q_sorted) > 1 else q_max
        gap = q_max - q_second

        if self.variant == "I0":
            # Current normalized values (min-max normalize)
            min_v = min(raw_q.values())
            max_v = max(raw_q.values())
            range_v = max_v - min_v + 1e-8
            normalized = {a: round((v - min_v) / range_v, 4) for a, v in raw_q.items()}
            estimates = {a: {"normalized_value": normalized[a]} for a in legal_actions}
            ranking = sorted(legal_actions, key=lambda a: -normalized[a])
            return None, ranking, estimates

        elif self.variant == "I1":
            # Raw centered advantages: A(s,a) = Q(s,a) - max_b Q(s,b)
            advantages = {a: round(v - q_max, 4) for a, v in raw_q.items()}
            estimates = {a: {"advantage": advantages[a]} for a in legal_actions}
            ranking = sorted(legal_actions, key=lambda a: -advantages[a])
            return None, ranking, estimates

        elif self.variant == "I2":
            # Epsilon near-optimal set: bucket actions into NEAR_OPTIMAL vs LOWER_VALUE
            near_optimal = [a for a, q in raw_q.items() if q >= q_max - self.epsilon]
            lower_value = [a for a in legal_actions if a not in near_optimal]
            extra = {
                "near_optimal_actions": sorted(near_optimal),
                "lower_value_actions": sorted(lower_value),
            }
            # No numeric estimates, no ranking — just the bucket
            return extra, None, None

        elif self.variant == "I3":
            # Confidence-aware recommendation
            if gap > self.tau:
                # High confidence: single recommendation
                extra = {
                    "recommendation_confidence": "HIGH",
                    "recommended_action": q_best,
                }
            else:
                # Low confidence: near-optimal set
                near_optimal = [a for a, q in raw_q.items() if q >= q_max - self.epsilon]
                extra = {
                    "recommendation_confidence": "LOW",
                    "near_optimal_actions": sorted(near_optimal),
                }
            return extra, None, None

        elif self.variant == "I4":
            # Single recommendation only on clear-choice states (gap > 3)
            if gap > self.tau:
                extra = {"recommended_action": q_best}
            else:
                extra = {"recommended_action": None}
            return extra, None, None

        return None, None, None


# ============================================================
# Trajectory runner
# ============================================================

@dataclass
class AblationWriter:
    output_dir: Path

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.traj_file = open(self.output_dir / "trajectories_v1.jsonl", "a")
        self.model_call_file = open(self.output_dir / "model_calls_v1.jsonl", "a")
        self.error_file = open(self.output_dir / "errors_v1.jsonl", "a")

    def _write(self, fh, record: dict):
        fh.write(json.dumps(record, sort_keys=True, default=bool) + "\n")
        fh.flush()
        os.fsync(fh.fileno())

    def write_trajectory(self, r: dict): self._write(self.traj_file, r)
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
    interface: InterfaceVariant,
    backend,
    i3_7e,
    utility,
    budget,
    max_steps: int = 10,
    writer: AblationWriter | None = None,
    run_id: str = "",
) -> dict:
    from hrm_adaptive_memory.executive.evidence_benchmark import build_evidence_snapshot
    from hrm_adaptive_memory.executive.evidence_benchmark.executor import EvidenceExecutor, initial_evidence_runtime
    from hrm_adaptive_memory.executive.resources import ResourceState
    from hrm_adaptive_memory.cognitive_control.core import DecisionAction
    from hrm_adaptive_memory.executive.model_decoder import decode_output_strict
    from r2_schema import build_action_schema, schema_sha256, schema_action_enum
    from r2_allowed_actions import compute_allowed_actions, ActionState, EmptyAllowedActionSet, C0
    from daph.intervention.checkpoint import compute_state_features

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
        evidence_snapshot = build_evidence_snapshot(
            runtime,
            prior_actions=tuple(prior_actions),
            prior_outcomes=tuple(prior_outcomes),
        )

        viability = i3_7e._classify_from_snapshot(evidence_snapshot)
        eliminated = [h_id for h_id, info in viability.items()
                      if info["status"] == "ELIMINATED"]
        t2 = (len(eliminated) == n_hypotheses and n_hypotheses > 0)

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

        packet = i3_7e.build_mdsg_state_with_affordances_packet(evidence_snapshot)

        # Get state features for the interface
        sf = compute_state_features(runtime, tuple(prior_actions))
        phase = classify_phase_simple(sf)
        legal_actions = sorted(allowed_decision.allowed)

        # Build interface-specific packet fields
        extra_fields, ranking, estimates = interface.build_packet_fields(
            sf, legal_actions, phase)

        # Add phase for all guided arms
        if arm != "C0":
            packet["epistemic_phase"] = phase

        # Add interface-specific fields
        if estimates is not None:
            packet["action_value_estimates"] = estimates
        if ranking is not None:
            packet["action_value_ranking"] = ranking
        if extra_fields:
            packet.update(extra_fields)

        schema = build_action_schema(allowed_decision.allowed)
        schema_sha = schema_sha256(schema)
        user_prompt = i3_7e.evidence_packet_json(packet)

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

        if writer:
            writer.write_model_call({
                "run_id": run_id, "task_id": task.task_id, "arm": arm,
                "step": step_id, "raw_output": raw_output,
                "decoder_valid": bool(outcome.valid),
                "schema_sha256": schema_sha,
                "arm": arm, "phase": phase,
                "packet_has_values": estimates is not None,
                "packet_has_ranking": ranking is not None,
                "packet_has_extra": extra_fields is not None,
            })

        if not outcome.valid or not outcome.proposal:
            decoder_errors += 1
            terminal_result = "DECODER_ERROR"
            break

        proposal = outcome.proposal
        action_str = proposal.action.value if hasattr(proposal.action, "value") else str(proposal.action)
        target_id = getattr(proposal, "target_id", None)

        if action_str not in allowed_decision.allowed:
            terminal_result = "ADMISSIBILITY_VIOLATION"
            break

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
        realized -= 0.5

    # Compute repeated-action metrics
    from collections import Counter
    action_counts = Counter(actions_taken)
    max_repeat = max(action_counts.values()) if action_counts else 0
    retrieve_count = action_counts.get("RETRIEVE", 0)
    verify_count = action_counts.get("VERIFY", 0)
    search_count = action_counts.get("SEARCH_MORE", 0)
    # Count consecutive repeats of any action
    max_consecutive = 0
    current_consecutive = 0
    prev_action = None
    for a in actions_taken:
        if a == prev_action:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 1
            prev_action = a
    max_consecutive = max(max_consecutive, 1) if actions_taken else 0

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
        "retrieve_count": retrieve_count,
        "verify_count": verify_count,
        "search_count": search_count,
        "max_action_repeat": max_repeat,
        "max_consecutive_repeat": max_consecutive,
        "resource_exhausted": terminal_result in ("STEP_LIMIT", "BACKEND_ERROR") and not terminal,
    }

    if writer:
        writer.write_trajectory(result)

    return result


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="I3.5-PQ Phase 21 Interface Ablation")
    parser.add_argument("--gguf-path", required=True)
    parser.add_argument("--model-name", default="qwen2.5-7b-instruct")
    parser.add_argument("--output-dir", default="experiments/i3_5/interface_ablation")
    parser.add_argument("--estimator-dir", default="experiments/i3_5/pinned_policy/frozen_estimators")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--n-per-subtype", type=int, default=24)
    parser.add_argument("--n-per-two-live", type=int, default=20)
    parser.add_argument("--epsilon", type=float, default=3.0, help="Near-optimal threshold")
    parser.add_argument("--tau", type=float, default=3.0, help="Confidence/clear-choice threshold")
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

    # Load frozen QCAUSAL model
    print("Loading frozen QCAUSAL model...")
    with open(estimator_dir / "QCAUSAL_gbt.pkl", "rb") as f:
        qcausal_model = pickle.load(f)
    with open(estimator_dir / "feature_schema.json") as f:
        feature_schema = json.load(f)
    feature_keys = feature_schema["feature_keys"]
    print(f"  QCAUSAL loaded, {len(feature_keys)} features")

    # Build interface variants
    arms = ["C0", "I0", "I1", "I2", "I3", "I4"]
    arm_descriptions = {
        "C0": "no guidance (baseline)",
        "I0": "current normalized values (min-max normalize)",
        "I1": "raw centered advantages: A(s,a) = Q(s,a) - max_b Q(s,b)",
        "I2": "epsilon near-optimal set (bucket, no numeric values)",
        "I3": "confidence-aware recommendation (gap/confidence-aware)",
        "I4": "single recommendation only on clear-choice states (gap > tau)",
    }
    interfaces = {}
    for arm in arms:
        if arm == "C0":
            interfaces[arm] = InterfaceVariant("C0", None, feature_keys)
        else:
            interfaces[arm] = InterfaceVariant(
                arm, qcausal_model, feature_keys,
                epsilon=args.epsilon, tau=args.tau)

    # Initialize backend
    print(f"\nInitializing R2DirectLlamaBackend with {args.gguf_path}...")
    from hrm_adaptive_memory.executive.model_backend import R2DirectLlamaBackend
    backend = R2DirectLlamaBackend(
        model_name=args.model_name,
        model_path=args.gguf_path,
        n_ctx=args.n_ctx,
    )
    print("  Backend initialized.")

    # Build run manifest
    timestamp = datetime.now(timezone.utc).isoformat()
    run_id = hashlib.sha256(f"i3_5_interface_ablation:{timestamp}".encode()).hexdigest()[:16]

    # Compute SHAs
    source_shas = {}
    for fname in ["run_i3_7e_compact_governor.py", "r2_schema.py", "r2_allowed_actions.py"]:
        path = REPO_ROOT / "scripts" / fname
        if path.exists():
            source_shas[fname] = hashlib.sha256(path.read_bytes()).hexdigest()
    for fname in ["model_backend.py", "model_decoder.py", "metareasoning_utility.py"]:
        path = REPO_ROOT / "hrm_adaptive_memory" / "executive" / fname
        if path.exists():
            source_shas[fname] = hashlib.sha256(path.read_bytes()).hexdigest()

    causal_dataset_path = REPO_ROOT / "experiments/i3_5/pinned_policy/pinned_causal_actions_v1.jsonl"
    causal_dataset_sha = hashlib.sha256(causal_dataset_path.read_bytes()).hexdigest()
    qcausal_pkl_sha = hashlib.sha256((estimator_dir / "QCAUSAL_gbt.pkl").read_bytes()).hexdigest()
    feature_schema_sha = hashlib.sha256((estimator_dir / "feature_schema.json").read_bytes()).hexdigest()
    utility_sha = hashlib.sha256((REPO_ROOT / "configs/v2b_i3_1_utility_v1.json").read_bytes()).hexdigest()

    run_manifest = {
        "run_id": run_id,
        "experiment": "I3.5-PQ Phase 21 Interface Ablation",
        "timestamp": timestamp,
        "arms": arms,
        "arm_descriptions": arm_descriptions,
        "n_tasks": len(tasks),
        "n_trajectories": len(tasks) * len(arms),
        "parameters": {
            "epsilon": args.epsilon,
            "tau": args.tau,
            "max_steps": args.max_steps,
        },
        "binding": {
            "model_name": args.model_name,
            "gguf_path": args.gguf_path,
            "causal_dataset_sha256": causal_dataset_sha,
            "qcausal_model_sha256": qcausal_pkl_sha,
            "feature_schema_sha256": feature_schema_sha,
            "utility_sha256": utility_sha,
        },
        "source_shas": source_shas,
        "invariants": [
            "QCAUSAL_V1 is frozen — no retraining",
            "Only the packet representation changes between arms",
            "Same Qwen backend for all arms",
            "Same system prompt for all arms",
            "Same schema enforcement for all arms",
            "Same utility for all arms",
            "temperature=0 for deterministic generation",
        ],
        "preregistered_endpoints": [
            "mean_utility",
            "success_rate",
            "mean_retrieve_count (especially ol_retrieve)",
            "resource_exhaustion_rate",
            "stop_after_exhaustion_rate",
            "near_optimal_selected_action_rate_epsilon3",
            "repeated_near_optimal_action_rate",
            "loop_rate",
            "premature_answer_rate",
            "premature_defer_rate",
            "max_consecutive_repeat",
            "stratified_by_subtype",
        ],
        "primary_target": "E[#RETRIEVE] on ol_retrieve should fall from ~3 toward ~1 without reducing success",
    }
    manifest_path = output_dir / "run_manifest_v1.json"
    with open(manifest_path, "w") as f:
        json.dump(run_manifest, f, indent=2, sort_keys=True)
    print(f"\nRun manifest: {manifest_path}")
    print(f"  Run ID: {run_id}")
    print(f"  Arms: {arms}")
    print(f"  Tasks: {len(tasks)}")
    print(f"  Total trajectories: {len(tasks) * len(arms)}")
    print(f"  epsilon={args.epsilon}, tau={args.tau}")

    # Setup writer
    writer = AblationWriter(output_dir)
    completed_keys = writer.load_completed_keys() if args.resume else set()
    if completed_keys:
        print(f"Resume: {len(completed_keys)} trajectories already completed")

    # Run trajectories
    n_total = len(tasks) * len(arms)
    n_done = len(completed_keys)
    start_time = time.time()

    print(f"\nStarting interface ablation: {n_total - n_done} trajectories remaining")
    print()

    for task in tasks:
        for arm in arms:
            traj_key = f"{task.task_id}:{arm}"
            if traj_key in completed_keys:
                continue

            result = run_trajectory(
                task=task,
                arm=arm,
                interface=interfaces[arm],
                backend=backend,
                i3_7e=i3_7e,
                utility=utility,
                budget=budget,
                max_steps=args.max_steps,
                writer=writer,
                run_id=run_id,
            )

            n_done += 1
            if n_done % 50 == 0:
                elapsed = time.time() - start_time
                rate = (n_done - len(completed_keys)) / elapsed if elapsed > 0 else 0
                remaining = n_total - n_done
                eta = remaining / rate if rate > 0 else 0
                print(f"  Progress: {n_done}/{n_total} ({rate:.1f}/s, ETA {eta:.0f}s) "
                      f"last: {task.task_id}:{arm} U={result['realized_utility']:.1f} "
                      f"success={result['success']} retr={result['retrieve_count']}")

    # Summary
    print(f"\n{'='*70}")
    print(f"Interface Ablation Complete")
    print(f"{'='*70}")
    print(f"  Total trajectories: {n_done}")
    print(f"  Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
