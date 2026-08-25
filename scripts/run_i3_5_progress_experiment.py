#!/usr/bin/env python3
"""I3.5-PQ Phase 24: VP executive = Q_CAUSAL_V1 + I2 + PROGRESS_RULE_V1.

Architecture:
  1. Compute Q(s,a) for all legal actions using frozen QCAUSAL_V1
  2. Compute A_epsilon(s) = {a : Q(s,a) >= Q_max - epsilon_Q}  (I2)
  3. Compute Progress(s,a) for each action in A_epsilon using PROGRESS_RULE_V1
  4. If progress can distinguish actions (gap > epsilon_P):
     - Narrow to A*(s) = {a in A_epsilon : P(a) >= P_max - epsilon_P}
  5. If progress cannot distinguish: keep full A_epsilon
  6. Send categorical packet to Qwen (no numbers, no ranking)

The model-facing packet is always:
  {
    "near_optimal_actions": [...],  # A*(s) or A_epsilon(s)
    "lower_value_actions": [...],   # everything else
    "guidance_confidence": "clear" | "ambiguous"
  }

Three arms:
  C0:  no guidance
  V1:  Q_CAUSAL_V1 + I2 (no progress)
  VP:  Q_CAUSAL_V1 + I2 + PROGRESS_RULE_V1

Usage:
  PYTHONPATH=. python3 scripts/run_i3_5_progress_experiment.py \
    --gguf-path /path/to/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
    --output-dir experiments/i3_5/progress_experiment \
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

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def extract_features(state_features: dict, action: str) -> dict:
    """Extract the 35 features for QCAUSAL_V1 prediction."""
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


def classify_phase_simple(sf: dict) -> str:
    """Phase classifier (frozen from interface ablation)."""
    n_live = sf.get("n_live", 0)
    n_eliminated = sf.get("n_eliminated", 0)
    n_total = sf.get("n_total_hypotheses", 0)
    n_supporting = sf.get("n_supporting", 0)
    n_verified = sf.get("n_verified", 0)
    n_visible = sf.get("n_visible_evidence", 0)
    if n_live == 0 and n_eliminated == n_total and n_total > 0:
        return "T2"
    if n_supporting > 0 and n_verified > 0:
        return "DISCRIMINATE"
    if n_visible > 0:
        return "EXPLORE"
    return "READY"


@dataclass
class QCAUSALModel:
    """Frozen QCAUSAL_V1 model wrapper."""
    model: object
    feature_keys: list[str]

    @classmethod
    def load(cls, est_dir: Path):
        with open(est_dir / "QCAUSAL_gbt.pkl", "rb") as f:
            model = pickle.load(f)
        with open(est_dir / "feature_schema.json") as f:
            schema = json.load(f)
        return cls(model=model, feature_keys=schema["feature_keys"])

    def predict_q(self, state_features: dict, legal_actions: list[str]) -> dict[str, float]:
        X = np.array([[extract_features(state_features, a)[k] for k in self.feature_keys]
                      for a in legal_actions])
        preds = self.model.predict(X)
        return dict(zip(legal_actions, [float(p) for p in preds]))


def compute_near_optimal_set(q_values: dict[str, float], epsilon_q: float = 3.0) -> tuple[list[str], list[str]]:
    """I2: Compute near-optimal set and lower-value set."""
    q_max = max(q_values.values())
    near_optimal = sorted([a for a, q in q_values.items() if q >= q_max - epsilon_q])
    lower_value = sorted([a for a, q in q_values.items() if q < q_max - epsilon_q])
    return near_optimal, lower_value


def compute_progress_scores(
    runtime,
    task,
    legal_actions: list[str],
    utility,
    executor,
) -> dict[str, dict]:
    """Compute progress scores for each legal action by simulating execution.

    For each action, we:
    1. Execute it on a copy of the runtime
    2. Compute Progress(s, a, s') using PROGRESS_RULE_V1
    3. Restore the runtime (the simulation doesn't affect the real runtime)
    """
    from daph.progress.progress_rule_v1 import compute_progress
    from hrm_adaptive_memory.cognitive_control.core import DecisionAction
    from hrm_adaptive_memory.executive.evidence_benchmark.executor import valid_verify_targets

    progress_scores = {}
    for action_str in legal_actions:
        action = DecisionAction(action_str)
        target_eid = None
        if action is DecisionAction.VERIFY:
            valid_targets = valid_verify_targets(runtime)
            if valid_targets:
                target_eid = valid_targets[0]
            else:
                progress_scores[action_str] = {
                    "progress": -0.2,
                    "state_changed": False,
                    "new_evidence_ids": [],
                    "new_verified_ids": [],
                }
                continue

        try:
            result = executor.execute(runtime, action, target_evidence_id=target_eid)
            progress = compute_progress(runtime, result, utility)
            progress_scores[action_str] = progress.as_dict()
        except Exception:
            progress_scores[action_str] = {
                "progress": -0.2,
                "state_changed": False,
                "new_evidence_ids": [],
                "new_verified_ids": [],
            }

    return progress_scores


def apply_progress_tiebreak(
    near_optimal: list[str],
    progress_scores: dict[str, dict],
    epsilon_p: float = 0.05,
) -> tuple[list[str], str]:
    """Use progress to break ties within the near-optimal set.

    Returns (refined_set, confidence).
    If progress can distinguish actions, narrow the set.
    If not, return the full near-optimal set.
    """
    if len(near_optimal) <= 1:
        return near_optimal, "clear"

    # Get progress scores for near-optimal actions
    scores = {a: progress_scores.get(a, {}).get("progress", -0.2) for a in near_optimal}
    p_max = max(scores.values())
    p_min = min(scores.values())
    gap = p_max - p_min

    if gap < epsilon_p:
        # Progress cannot distinguish — return full set
        return near_optimal, "ambiguous"

    # Narrow to actions within epsilon_p of the best progress
    refined = sorted([a for a, p in scores.items() if p >= p_max - epsilon_p])

    if len(refined) == len(near_optimal):
        # No narrowing happened
        return near_optimal, "ambiguous"

    return refined, "clear"


def build_vp_packet(
    near_optimal: list[str],
    lower_value: list[str],
    confidence: str,
    phase: str,
) -> dict:
    """Build the model-facing packet (categorical, no numbers)."""
    return {
        "near_optimal_actions": near_optimal,
        "lower_value_actions": lower_value,
        "guidance_confidence": confidence,
        "epistemic_phase": phase,
    }


def run_trajectory(
    task,
    backend,
    i3_7e,
    utility,
    q_model: QCAUSALModel,
    arm: str,
    max_steps: int = 10,
    epsilon_q: float = 3.0,
    epsilon_p: float = 0.05,
) -> dict:
    """Run a single trajectory for one arm."""
    from hrm_adaptive_memory.cognitive_control.core import DecisionAction
    from hrm_adaptive_memory.executive.evidence_benchmark import (
        initial_evidence_runtime, build_evidence_snapshot,
    )
    from hrm_adaptive_memory.executive.evidence_benchmark.executor import EvidenceExecutor
    from hrm_adaptive_memory.executive.resources import ResourceState, ResourceBudget
    from hrm_adaptive_memory.executive.model_decoder import decode_output_strict
    from r2_schema import build_action_schema, schema_sha256
    from r2_allowed_actions import compute_allowed_actions, ActionState, EmptyAllowedActionSet, C0
    from daph.intervention.checkpoint import compute_state_features

    budget = ResourceBudget(
        max_executive_steps=10, max_retrieval_calls=3,
        max_search_calls=2, max_verification_calls=5,
    )
    resources = ResourceState(budget=budget)
    runtime = initial_evidence_runtime(task, resources)
    executor = EvidenceExecutor()

    prior_actions = []
    prior_outcomes = []
    realized = 0.0
    success = False
    terminal = False
    terminal_action = None
    terminal_result = "STEP_LIMIT"
    premature_defer = False
    premature_answer = False
    actions_taken = []
    progress_log = []
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
        n_hypotheses = len(task.hypotheses)
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

        legal_actions = sorted(allowed_decision.allowed)
        sf = compute_state_features(runtime, tuple(prior_actions))
        phase = classify_phase_simple(sf)

        # Build arm-specific packet
        if arm == "C0":
            # No guidance — just the evidence packet
            extra_fields = {}
        elif arm == "V1":
            # Q + I2 (no progress)
            q_values = q_model.predict_q(sf, legal_actions)
            near_optimal, lower_value = compute_near_optimal_set(q_values, epsilon_q)
            extra_fields = {
                "near_optimal_actions": near_optimal,
                "lower_value_actions": lower_value,
                "guidance_confidence": "clear" if len(near_optimal) == 1 else "ambiguous",
                "epistemic_phase": phase,
            }
        elif arm == "VP":
            # Q + I2 + Progress
            q_values = q_model.predict_q(sf, legal_actions)
            near_optimal, lower_value = compute_near_optimal_set(q_values, epsilon_q)

            # Compute progress scores for all legal actions
            progress_scores = compute_progress_scores(
                runtime, task, legal_actions, utility, executor)

            # Apply progress tiebreak within near-optimal set
            refined_set, confidence = apply_progress_tiebreak(
                near_optimal, progress_scores, epsilon_p)

            extra_fields = build_vp_packet(refined_set, lower_value, confidence, phase)

            # Log progress scores for analysis
            progress_log.append({
                "step": step_id,
                "q_values": {a: round(q, 2) for a, q in q_values.items()},
                "near_optimal_before_progress": near_optimal,
                "near_optimal_after_progress": refined_set,
                "progress_scores": {a: round(s.get("progress", 0), 4)
                                    for a, s in progress_scores.items()},
                "confidence": confidence,
            })
        else:
            extra_fields = {}

        packet = i3_7e.build_mdsg_state_with_affordances_packet(evidence_snapshot)
        schema = build_action_schema(allowed_decision.allowed)
        schema_sha = schema_sha256(schema)

        # Inject extra fields into the packet
        if extra_fields:
            packet_dict = json.loads(i3_7e.evidence_packet_json(packet))
            packet_dict["executive_guidance"] = extra_fields
            user_prompt = json.dumps(packet_dict, indent=2)
        else:
            user_prompt = i3_7e.evidence_packet_json(packet)

        try:
            call_result = backend.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.0,
                max_tokens=256,
                allowed_actions=allowed_decision.allowed,
            )
        except Exception:
            terminal_result = "BACKEND_ERROR"
            break

        raw_output = call_result.raw_output
        outcome = decode_output_strict(raw_output)

        if not outcome.valid or not outcome.proposal:
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

    # Compute action statistics
    action_counts = defaultdict(int)
    for a in actions_taken:
        action_counts[a] += 1

    # Compute repeated action stats
    max_consecutive_repeat = 0
    current_run = 1
    for i in range(1, len(actions_taken)):
        if actions_taken[i] == actions_taken[i - 1]:
            current_run += 1
            max_consecutive_repeat = max(max_consecutive_repeat, current_run)
        else:
            current_run = 1

    return {
        "task_id": task.task_id,
        "arm": arm,
        "realized_utility": round(realized, 4),
        "success": bool(success),
        "terminal_action": terminal_action,
        "terminal_result": terminal_result,
        "actions_taken": actions_taken,
        "retrieve_count": action_counts.get("RETRIEVE", 0),
        "verify_count": action_counts.get("VERIFY", 0),
        "search_count": action_counts.get("SEARCH_MORE", 0),
        "answer_count": action_counts.get("ANSWER", 0),
        "defer_count": action_counts.get("DEFER", 0),
        "max_consecutive_repeat": max_consecutive_repeat,
        "premature_defer": bool(premature_defer),
        "premature_answer": bool(premature_answer),
        "steps": len(actions_taken),
        "progress_log": progress_log if arm == "VP" else [],
    }


@dataclass
class ProgressWriter:
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

    def write_traj(self, r: dict): self._write(self.traj_file, r)
    def write_error(self, r: dict): self._write(self.error_file, r)

    def load_completed(self) -> set[str]:
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


def main():
    parser = argparse.ArgumentParser(description="I3.5-PQ Phase 24 Progress Experiment")
    parser.add_argument("--gguf-path", required=True)
    parser.add_argument("--model-name", default="qwen2.5-7b-instruct")
    parser.add_argument("--output-dir", default="experiments/i3_5/progress_experiment")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--epsilon-q", type=float, default=3.0)
    parser.add_argument("--epsilon-p", type=float, default=0.05)
    parser.add_argument("--n-per-subtype", type=int, default=24)
    parser.add_argument("--n-per-two-live", type=int, default=20)
    args = parser.parse_args()

    output_dir = REPO_ROOT / args.output_dir
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
    utility = MetareasoningUtility.from_file(
        REPO_ROOT / "configs" / "v2b_i3_1_utility_v1.json")

    # Load frozen QCAUSAL_V1
    print("Loading frozen QCAUSAL_V1...")
    est_dir = REPO_ROOT / "experiments/i3_5/pinned_policy/frozen_estimators"
    q_model = QCAUSALModel.load(est_dir)
    print(f"  QCAUSAL loaded, {len(q_model.feature_keys)} features")

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
    run_id = hashlib.sha256(f"i3_5_progress:{timestamp}".encode()).hexdigest()[:16]

    arms = ["C0", "V1", "VP"]
    arm_descriptions = {
        "C0": "no guidance (baseline)",
        "V1": "Q_CAUSAL_V1 + I2 epsilon interface",
        "VP": "Q_CAUSAL_V1 + I2 + PROGRESS_RULE_V1 tie-breaking",
    }

    manifest = {
        "run_id": run_id,
        "experiment": "I3.5-PQ Phase 24 Progress Experiment",
        "timestamp": timestamp,
        "arms": arms,
        "arm_descriptions": arm_descriptions,
        "parameters": {
            "max_steps": args.max_steps,
            "epsilon_q": args.epsilon_q,
            "epsilon_p": args.epsilon_p,
            "n_per_subtype": args.n_per_subtype,
            "n_per_two_live": args.n_per_two_live,
        },
        "frozen_components": {
            "q_model": "QCAUSAL_V1 (frozen, no retraining)",
            "interface": "I2 epsilon near-optimal set",
            "progress": "PROGRESS_RULE_V1 (deterministic structural)",
            "model": "Qwen2.5-7B-Instruct-Q4_K_M at temperature=0",
        },
        "hypothesis": (
            "Adding PROGRESS_RULE_V1 as a tie-breaker within A_epsilon "
            "will preserve V1's success while reducing redundant actions "
            "and increasing utility on repeated-action traps."
        ),
        "promotion_criteria": {
            "success": "VP success >= V1 success",
            "utility": "paired Delta_U(VP - V1) > 0, CI excludes 0",
            "repeated_actions": "VP repeated-action count < V1",
            "no_regression": "no increase in premature DEFER/ANSWER or resource exhaustion",
        },
    }
    manifest_path = output_dir / "run_manifest_v1.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"\nRun manifest: {manifest_path}")
    print(f"  Run ID: {run_id}")
    print(f"  Arms: {arms}")
    print(f"  Tasks: {len(tasks)}")
    print(f"  Total trajectories: {len(arms) * len(tasks)}")
    print(f"  epsilon_q={args.epsilon_q}, epsilon_p={args.epsilon_p}")

    # Setup writer
    writer = ProgressWriter(output_dir)
    completed = writer.load_completed() if args.resume else set()
    if completed:
        print(f"Resume: {len(completed)} trajectories already completed")

    # Run experiment
    n_done = len(completed)
    start_time = time.time()
    total = len(arms) * len(tasks)

    print(f"\nStarting progress experiment: {total - n_done} trajectories remaining")

    for arm in arms:
        for task in tasks:
            key = f"{task.task_id}:{arm}"
            if key in completed:
                continue

            try:
                result = run_trajectory(
                    task, backend, i3_7e, utility, q_model, arm,
                    max_steps=args.max_steps,
                    epsilon_q=args.epsilon_q,
                    epsilon_p=args.epsilon_p,
                )
                writer.write_traj(result)
                n_done += 1

                if n_done % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = (n_done - len(completed)) / elapsed if elapsed > 0 else 0
                    print(f"  Progress: {n_done}/{total} ({rate:.1f}/s) "
                          f"last: {task.task_id}:{arm} "
                          f"U={result['realized_utility']:.1f} "
                          f"success={result['success']} "
                          f"retr={result['retrieve_count']}")
            except Exception as exc:
                writer.write_error({
                    "task_id": task.task_id,
                    "arm": arm,
                    "error": type(exc).__name__,
                    "error_message": str(exc),
                    "timestamp": timestamp,
                })
                n_done += 1

    print(f"\n{'='*70}")
    print(f"Progress Experiment Complete")
    print(f"{'='*70}")
    print(f"  Total trajectories: {n_done}")
    print(f"  Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
