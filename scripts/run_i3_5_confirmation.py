#!/usr/bin/env python3
"""I3.5-PQ Phase 25: Confirmation benchmark runner.

4 arms: C0, B0, V1, VP on 180 frozen confirmation tasks.
No modification to any frozen component permitted.

Usage:
  PYTHONPATH=. python3 scripts/run_i3_5_confirmation.py \
    --gguf-path /path/to/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
    --output-dir experiments/i3_5/confirmation \
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

from hrm_adaptive_memory.executive.evidence_benchmark.i3_5_confirmation_generator import (
    generate_confirmation_benchmark,
    CONFIRMATION_BUDGET_PROFILES,
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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


def classify_phase_simple(sf: dict) -> str:
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


def compute_near_optimal_set(q_values: dict[str, float], epsilon_q: float = 3.0):
    q_max = max(q_values.values())
    near_optimal = sorted([a for a, q in q_values.items() if q >= q_max - epsilon_q])
    lower_value = sorted([a for a, q in q_values.items() if q < q_max - epsilon_q])
    return near_optimal, lower_value


def compute_progress_scores(runtime, legal_actions, utility, executor):
    from daph.progress.progress_rule_v1 import compute_progress
    from hrm_adaptive_memory.cognitive_control.core import DecisionAction
    from hrm_adaptive_memory.executive.evidence_benchmark.executor import valid_verify_targets

    scores = {}
    for action_str in legal_actions:
        action = DecisionAction(action_str)
        target_eid = None
        if action is DecisionAction.VERIFY:
            valid = valid_verify_targets(runtime)
            if valid:
                target_eid = valid[0]
            else:
                scores[action_str] = {"progress": -0.2, "state_changed": False,
                                      "new_evidence_ids": [], "new_verified_ids": []}
                continue
        try:
            result = executor.execute(runtime, action, target_evidence_id=target_eid)
            progress = compute_progress(runtime, result, utility)
            scores[action_str] = progress.as_dict()
        except Exception:
            scores[action_str] = {"progress": -0.2, "state_changed": False,
                                  "new_evidence_ids": [], "new_verified_ids": []}
    return scores


def apply_progress_tiebreak(near_optimal, progress_scores, epsilon_p=0.05):
    if len(near_optimal) <= 1:
        return near_optimal, "clear"
    scores = {a: progress_scores.get(a, {}).get("progress", -0.2) for a in near_optimal}
    p_max = max(scores.values())
    p_min = min(scores.values())
    gap = p_max - p_min
    if gap < epsilon_p:
        return near_optimal, "ambiguous"
    refined = sorted([a for a, p in scores.items() if p >= p_max - epsilon_p])
    if len(refined) == len(near_optimal):
        return near_optimal, "ambiguous"
    return refined, "clear"


def get_budget_for_profile(profile: str):
    from hrm_adaptive_memory.executive.resources import ResourceBudget
    params = CONFIRMATION_BUDGET_PROFILES[profile]
    return ResourceBudget(
        max_executive_steps=params["max_executive_steps"],
        max_retrieval_calls=params["max_retrieval_calls"],
        max_verification_calls=params["max_verification_calls"],
        max_search_calls=params["max_search_calls"],
        max_reasoning_tokens=params.get("max_reasoning_tokens", 256),
        max_elapsed_ms=params.get("max_elapsed_ms", 10_000),
    )


def run_trajectory(task, backend, i3_7e, utility, q_model, b0_value, arm,
                   max_steps_override=None):
    from hrm_adaptive_memory.cognitive_control.core import DecisionAction
    from hrm_adaptive_memory.executive.evidence_benchmark import (
        initial_evidence_runtime, build_evidence_snapshot,
    )
    from hrm_adaptive_memory.executive.evidence_benchmark.executor import EvidenceExecutor
    from hrm_adaptive_memory.executive.resources import ResourceState
    from hrm_adaptive_memory.executive.model_decoder import decode_output_strict
    from r2_schema import build_action_schema, schema_sha256
    from r2_allowed_actions import compute_allowed_actions, ActionState, EmptyAllowedActionSet, C0
    from daph.intervention.checkpoint import compute_state_features

    budget = get_budget_for_profile(task.budget_profile)
    max_steps = max_steps_override or budget.max_executive_steps
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
    resource_exhaustion = False
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

        # Check resource exhaustion
        if sf.get("steps_remaining", 0) <= 0:
            resource_exhaustion = True
        if sf.get("retrieval_remaining", 0) == 0 and sf.get("search_remaining", 0) == 0:
            if sf.get("verify_remaining", 0) == 0 and not t2:
                resource_exhaustion = True

        # Build arm-specific packet
        if arm == "C0":
            extra_fields = {}
        elif arm == "B0":
            # B0: global prior — same value for all actions
            extra_fields = {
                "near_optimal_actions": legal_actions,
                "lower_value_actions": [],
                "guidance_confidence": "ambiguous",
                "epistemic_phase": phase,
            }
        elif arm == "V1":
            q_values = q_model.predict_q(sf, legal_actions)
            near_optimal, lower_value = compute_near_optimal_set(q_values, 3.0)
            extra_fields = {
                "near_optimal_actions": near_optimal,
                "lower_value_actions": lower_value,
                "guidance_confidence": "clear" if len(near_optimal) == 1 else "ambiguous",
                "epistemic_phase": phase,
            }
        elif arm == "VP":
            q_values = q_model.predict_q(sf, legal_actions)
            near_optimal, lower_value = compute_near_optimal_set(q_values, 3.0)
            progress_scores = compute_progress_scores(runtime, legal_actions, utility, executor)
            refined_set, confidence = apply_progress_tiebreak(near_optimal, progress_scores, 0.05)
            extra_fields = {
                "near_optimal_actions": refined_set,
                "lower_value_actions": lower_value,
                "guidance_confidence": confidence,
                "epistemic_phase": phase,
            }
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
        if resource_exhaustion:
            terminal_result = "RESOURCE_EXHAUSTION"

    action_counts = defaultdict(int)
    for a in actions_taken:
        action_counts[a] += 1

    max_consecutive = 0
    cur = 1
    for i in range(1, len(actions_taken)):
        if actions_taken[i] == actions_taken[i - 1]:
            cur += 1
            max_consecutive = max(max_consecutive, cur)
        else:
            cur = 1

    return {
        "task_id": task.task_id,
        "category": task.category,
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
        "max_consecutive_repeat": max_consecutive,
        "premature_defer": bool(premature_defer),
        "premature_answer": bool(premature_answer),
        "resource_exhaustion": bool(resource_exhaustion),
        "steps": len(actions_taken),
        "budget_profile": task.budget_profile,
        "expected_terminal": task.expected_terminal.value,
        "progress_log": progress_log if arm == "VP" else [],
    }


@dataclass
class ConfirmationWriter:
    output_dir: Path

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.traj_file = open(self.output_dir / "trajectories_v1.jsonl", "a")
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
    parser = argparse.ArgumentParser(description="I3.5-PQ Phase 25 Confirmation")
    parser.add_argument("--gguf-path", required=True)
    parser.add_argument("--model-name", default="qwen2.5-7b-instruct")
    parser.add_argument("--output-dir", default="experiments/i3_5/confirmation")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--n-ctx", type=int, default=4096)
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

    # Load frozen confirmation tasks
    print("Loading frozen confirmation benchmark...")
    tasks = generate_confirmation_benchmark(n_per_subtype=12, seed=4287)
    print(f"  {len(tasks)} tasks")

    # Verify hash
    task_json = json.dumps([{
        "task_id": t.task_id, "category": t.category,
        "expected_terminal": t.expected_terminal.value,
        "budget_profile": t.budget_profile,
        "correct_hypothesis_id": t.correct_hypothesis_id,
    } for t in tasks], sort_keys=True)
    bench_hash = hashlib.sha256(task_json.encode()).hexdigest()
    print(f"  Benchmark hash: {bench_hash}")

    # Load utility
    from hrm_adaptive_memory.executive.metareasoning_utility import MetareasoningUtility
    utility = MetareasoningUtility.from_file(
        REPO_ROOT / "configs" / "v2b_i3_1_utility_v1.json")

    # Load frozen QCAUSAL_V1
    print("Loading frozen QCAUSAL_V1...")
    est_dir = REPO_ROOT / "experiments/i3_5/pinned_policy/frozen_estimators"
    q_model = QCAUSALModel.load(est_dir)

    # Load B0 global mean
    with open(est_dir / "B0_global_mean.json") as f:
        b0_data = json.load(f)
    b0_value = b0_data["value"]
    print(f"  B0 global mean: {b0_value:.2f}")

    # Initialize backend
    print(f"\nInitializing backend with {args.gguf_path}...")
    from hrm_adaptive_memory.executive.model_backend import R2DirectLlamaBackend
    backend = R2DirectLlamaBackend(
        model_name=args.model_name,
        model_path=args.gguf_path,
        n_ctx=args.n_ctx,
    )
    print("  Backend initialized.")

    # Build run manifest
    timestamp = datetime.now(timezone.utc).isoformat()
    run_id = hashlib.sha256(f"i3_5_confirmation:{timestamp}".encode()).hexdigest()[:16]

    arms = ["C0", "B0", "V1", "VP"]
    manifest = {
        "run_id": run_id,
        "experiment": "I3.5-PQ Phase 25 Confirmation Benchmark",
        "timestamp": timestamp,
        "arms": arms,
        "benchmark_hash": bench_hash,
        "n_tasks": len(tasks),
        "n_trajectories": len(arms) * len(tasks),
        "frozen_components": {
            "q_model": "QCAUSAL_V1 (frozen)",
            "interface": "I2 epsilon near-optimal set (epsilon_q=3.0)",
            "progress": "PROGRESS_RULE_V1 (frozen, epsilon_p=0.05)",
            "model": "Qwen2.5-7B-Instruct-Q4_K_M at temperature=0",
            "utility": "v2b_i3_1_utility_v1.json (frozen)",
            "benchmark": "PHASE25_CONFIRMATION_BENCHMARK (frozen, hash recorded)",
        },
        "prohibition": (
            "No modification to VP, Q, I2, Progress, prompts, thresholds, "
            "utility, or benchmark tasks permitted. This is attempted falsification."
        ),
    }
    manifest_path = output_dir / "run_manifest_v1.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"\nRun manifest: {manifest_path}")
    print(f"  Run ID: {run_id}")
    print(f"  Arms: {arms}")
    print(f"  Tasks: {len(tasks)}")
    print(f"  Total trajectories: {len(arms) * len(tasks)}")

    # Setup writer
    writer = ConfirmationWriter(output_dir)
    completed = writer.load_completed() if args.resume else set()
    if completed:
        print(f"Resume: {len(completed)} trajectories already completed")

    # Run experiment
    n_done = len(completed)
    start_time = time.time()
    total = len(arms) * len(tasks)

    print(f"\nStarting confirmation: {total - n_done} trajectories remaining")

    for arm in arms:
        for task in tasks:
            key = f"{task.task_id}:{arm}"
            if key in completed:
                continue

            try:
                result = run_trajectory(
                    task, backend, i3_7e, utility, q_model, b0_value, arm,
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
                          f"steps={result['steps']}")
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
    print(f"Confirmation Complete")
    print(f"{'='*70}")
    print(f"  Total trajectories: {n_done}")
    print(f"  Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
