#!/usr/bin/env python3
"""I3.26 Stage B: VP vs VS (VP + SelectiveSearch) development experiment.

Two arms only:
  VP = frozen DAPH_PROGRESS_EXECUTIVE_V1
  VS = VP + SelectiveSearch (challenger)

Runs on the frozen I3.26 development benchmark (seed 7719).
No modification to any frozen VP component permitted.

Usage:
  PYTHONPATH=. python3 scripts/run_i3_26_stage_b.py \
    --gguf-path /path/to/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
    --output-dir experiments/i3_26/stage_b \
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

from hrm_adaptive_memory.executive.evidence_benchmark.i3_26_development_generator import (
    generate_development_benchmark, compute_benchmark_hash,
)
from hrm_adaptive_memory.executive.evidence_benchmark.i3_5_confirmation_generator import (
    CONFIRMATION_BUDGET_PROFILES,
)


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


def run_trajectory(task, backend, i3_7e, utility, q_model, arm,
                   max_steps_override=None, search_config=None):
    """Run a single trajectory for VP or VS arm.

    VP: frozen DAPH_PROGRESS_EXECUTIVE_V1 (Q + I2 + Progress tiebreak)
    VS: VP + SelectiveSearch (challenger)
    """
    from hrm_adaptive_memory.cognitive_control.core import DecisionAction
    from hrm_adaptive_memory.executive.evidence_benchmark import (
        initial_evidence_runtime, build_evidence_snapshot,
    )
    from hrm_adaptive_memory.executive.evidence_benchmark.executor import EvidenceExecutor
    from hrm_adaptive_memory.executive.resources import ResourceState
    from hrm_adaptive_memory.executive.model_decoder import decode_output_strict
    from r2_schema import build_action_schema, schema_sha256
    from r2_allowed_actions import compute_allowed_actions, ActionState, EmptyAllowedActionSet, C0
    from daph.intervention.checkpoint import (
        compute_state_features, create_checkpoint,
    )

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
    search_log = []
    system_prompt = i3_7e.MDSG_STATE_WITH_AFFORDANCES_SYSTEM_PROMPT

    # For VS arm: set up search planner
    search_planner = None
    if arm == "VS":
        from daph.search.types import SearchConfig
        from daph.search.planner import SearchPlanner
        from daph.search.trigger import decide_search
        config = search_config or SearchConfig()
        search_planner = SearchPlanner(task, utility, config)

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

        if sf.get("steps_remaining", 0) <= 0:
            resource_exhaustion = True
        if sf.get("retrieval_remaining", 0) == 0 and sf.get("search_remaining", 0) == 0:
            if sf.get("verify_remaining", 0) == 0 and not t2:
                resource_exhaustion = True

        # === VP baseline (frozen) ===
        q_values = q_model.predict_q(sf, legal_actions)
        near_optimal, lower_value = compute_near_optimal_set(q_values, 3.0)
        progress_scores = compute_progress_scores(runtime, legal_actions, utility, executor)
        refined_set, confidence = apply_progress_tiebreak(near_optimal, progress_scores, 0.05)

        progress_log.append({
            "step": step_id,
            "q_values": {a: round(q, 2) for a, q in q_values.items()},
            "near_optimal_before_progress": near_optimal,
            "near_optimal_after_progress": refined_set,
            "progress_scores": {a: round(s.get("progress", 0), 4)
                                for a, s in progress_scores.items()},
            "confidence": confidence,
        })

        if arm == "VP":
            # Frozen VP guidance
            extra_fields = {
                "near_optimal_actions": refined_set,
                "lower_value_actions": lower_value,
                "guidance_confidence": confidence,
                "epistemic_phase": phase,
            }
        elif arm == "VS":
            # VP + SelectiveSearch
            from daph.search.types import SearchConfig
            config = search_config or SearchConfig()
            checkpoint = create_checkpoint(
                runtime, step=step_id, phase=phase,
                prior_actions=tuple(prior_actions),
                prior_outcomes=tuple(prior_outcomes),
            )

            # Decide whether to search
            trigger_result = decide_search(
                state_features=sf,
                near_optimal_actions=tuple(refined_set),
                pav_selected=tuple(refined_set) if confidence == "clear" else None,
                pav_abstained=(confidence == "ambiguous"),
                q_values=q_values,
                config=config,
            )

            search_triggered = trigger_result.should_search
            search_winner = None
            search_abstained = False
            search_fallback_reason = None
            search_receipt = None

            if search_triggered and search_planner is not None:
                try:
                    search_result = search_planner.plan(
                        checkpoint=checkpoint,
                        candidate_actions=tuple(refined_set),
                        q_values=q_values,
                        trigger_reasons=trigger_result.reasons,
                    )
                    search_abstained = search_result.abstained
                    if not search_result.abstained:
                        search_winner = search_result.winner
                    else:
                        search_fallback_reason = search_result.fallback_reason
                    search_receipt = search_result.receipt
                except Exception as e:
                    search_abstained = True
                    search_fallback_reason = f"search_error: {e}"

            search_log.append({
                "step": step_id,
                "search_triggered": search_triggered,
                "search_abstained": search_abstained,
                "search_winner": search_winner,
                "search_fallback_reason": search_fallback_reason,
                "trigger_reasons": list(trigger_result.reasons),
                "near_optimal_set": refined_set,
            })

            if search_triggered and not search_abstained and search_winner:
                # Search found a winner — recommend it
                extra_fields = {
                    "near_optimal_actions": [search_winner],
                    "lower_value_actions": [a for a in refined_set if a != search_winner] + lower_value,
                    "guidance_confidence": "high",
                    "epistemic_phase": phase,
                    "metacognitive_guidance": {
                        "mode": "SEARCH_SUPPORTED",
                        "recommended_actions": [search_winner],
                        "alternatives": [a for a in refined_set if a != search_winner],
                        "confidence": "HIGH",
                    },
                }
            else:
                # Fall back to VP guidance
                extra_fields = {
                    "near_optimal_actions": refined_set,
                    "lower_value_actions": lower_value,
                    "guidance_confidence": confidence,
                    "epistemic_phase": phase,
                }
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
        "progress_log": progress_log,
        "search_log": search_log if arm == "VS" else [],
    }


@dataclass
class StageBWriter:
    output_dir: Path

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.traj_file = open(self.output_dir / "trajectories.jsonl", "a")
        self.error_file = open(self.output_dir / "errors.jsonl", "a")

    def _write(self, fh, record: dict):
        fh.write(json.dumps(record, sort_keys=True, default=bool) + "\n")
        fh.flush()
        os.fsync(fh.fileno())

    def write_traj(self, r: dict): self._write(self.traj_file, r)
    def write_error(self, r: dict): self._write(self.error_file, r)

    def load_completed(self) -> set[str]:
        completed = set()
        path = self.output_dir / "trajectories.jsonl"
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
    parser = argparse.ArgumentParser(description="I3.26 Stage B: VP vs VS")
    parser.add_argument("--gguf-path", required=True)
    parser.add_argument("--model-name", default="qwen2.5-7b-instruct")
    parser.add_argument("--output-dir", default="experiments/i3_26/stage_b")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--max-tasks", type=int, default=None,
                       help="Limit number of tasks (for testing)")
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

    # Load frozen development benchmark
    print("Loading frozen I3.26 development benchmark...")
    tasks = generate_development_benchmark(seed=7719)
    bench_hash = compute_benchmark_hash(tasks)
    print(f"  {len(tasks)} tasks, hash: {bench_hash}")

    if args.max_tasks:
        tasks = tasks[:args.max_tasks]
        print(f"  Limited to {len(tasks)} tasks")

    # Load utility
    from hrm_adaptive_memory.executive.metareasoning_utility import MetareasoningUtility
    utility = MetareasoningUtility.from_file(
        REPO_ROOT / "configs" / "v2b_i3_1_utility_v1.json")

    # Load frozen QCAUSAL_V1
    print("Loading frozen QCAUSAL_V1...")
    est_dir = REPO_ROOT / "experiments/i3_5/pinned_policy/frozen_estimators"
    q_model = QCAUSALModel.load(est_dir)

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
    run_id = hashlib.sha256(f"i3_26_stage_b:{timestamp}".encode()).hexdigest()[:16]

    arms = ["VP", "VS"]
    manifest = {
        "run_id": run_id,
        "timestamp": timestamp,
        "benchmark_hash": bench_hash,
        "n_tasks": len(tasks),
        "arms": arms,
        "benchmark_seed": 7719,
        "benchmark_version": "I3.26_DEVELOPMENT_BENCHMARK_V1",
        "search_config": "DAPH_SEARCH_CONFIG_V0",
        "frozen_components": [
            "QCAUSAL_V1", "PROGRESS_RULE_V1", "VP prompts",
            "VP thresholds", "utility", "executor", "phase/state computation",
        ],
    }
    with open(output_dir / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    writer = StageBWriter(output_dir)
    completed = writer.load_completed() if args.resume else set()

    total = len(tasks) * len(arms)
    done = 0
    errors = 0
    start_time = time.time()

    print(f"\nRunning {total} trajectories ({len(tasks)} tasks x {len(arms)} arms)")
    print(f"Already completed: {len(completed)}")

    for arm in arms:
        for task in tasks:
            key = f"{task.task_id}:{arm}"
            if key in completed:
                done += 1
                continue

            try:
                result = run_trajectory(task, backend, i3_7e, utility, q_model, arm)
                writer.write_traj(result)
                done += 1
                completed.add(key)

                if done % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = done / elapsed if elapsed > 0 else 0
                    eta = (total - done) / rate if rate > 0 else 0
                    print(f"  [{done}/{total}] {arm} {task.task_id} "
                          f"success={result['success']} "
                          f"steps={result['steps']} "
                          f"({rate:.1f}/s, ETA {eta:.0f}s)")

            except Exception as e:
                errors += 1
                writer.write_error({
                    "task_id": task.task_id, "arm": arm,
                    "error": str(e), "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                print(f"  ERROR: {arm} {task.task_id}: {e}")

    elapsed = time.time() - start_time
    print(f"\nDone: {done}/{total} trajectories, {errors} errors, {elapsed:.1f}s")

    # Quick summary
    traj_path = output_dir / "trajectories.jsonl"
    if traj_path.exists():
        records = [json.loads(line) for line in open(traj_path)]
        by_arm = defaultdict(list)
        for r in records:
            by_arm[r["arm"]].append(r)

        print(f"\n{'='*60}")
        print("STAGE B PRELIMINARY SUMMARY")
        print(f"{'='*60}")
        for arm in arms:
            arm_records = by_arm.get(arm, [])
            if not arm_records:
                continue
            n = len(arm_records)
            successes = sum(1 for r in arm_records if r["success"])
            mean_u = np.mean([r["realized_utility"] for r in arm_records])
            mean_steps = np.mean([r["steps"] for r in arm_records])
            print(f"\n  {arm}:")
            print(f"    N: {n}")
            print(f"    Success: {successes}/{n} ({successes/n:.2%})")
            print(f"    Mean utility: {mean_u:.2f}")
            print(f"    Mean steps: {mean_steps:.2f}")

        # Paired comparison if both arms have data
        if "VP" in by_arm and "VS" in by_arm:
            vp_by_task = {r["task_id"]: r for r in by_arm["VP"]}
            vs_by_task = {r["task_id"]: r for r in by_arm["VS"]}
            common = set(vp_by_task.keys()) & set(vs_by_task.keys())

            if common:
                vp_u = np.array([vp_by_task[t]["realized_utility"] for t in common])
                vs_u = np.array([vs_by_task[t]["realized_utility"] for t in common])
                delta = vs_u - vp_u
                mean_delta = np.mean(delta)
                ci = 1.96 * np.std(delta) / np.sqrt(len(delta))

                vp_s = sum(1 for t in common if vp_by_task[t]["success"])
                vs_s = sum(1 for t in common if vs_by_task[t]["success"])

                rescues = sum(1 for t in common
                             if not vp_by_task[t]["success"] and vs_by_task[t]["success"])
                breaks = sum(1 for t in common
                            if vp_by_task[t]["success"] and not vs_by_task[t]["success"])

                print(f"\n  Paired comparison ({len(common)} tasks):")
                print(f"    Delta U (VS-VP): {mean_delta:.4f} "
                      f"(95% CI [{mean_delta-ci:.4f}, {mean_delta+ci:.4f}])")
                print(f"    Success VP: {vp_s}, VS: {vs_s}")
                print(f"    Rescues: {rescues}, Breaks: {breaks}")

                # Search stats
                vs_search_triggered = sum(
                    1 for r in by_arm["VS"]
                    if any(s.get("search_triggered") for s in r.get("search_log", []))
                )
                vs_search_rate = vs_search_triggered / len(by_arm["VS"]) if by_arm["VS"] else 0
                print(f"    Search triggered: {vs_search_triggered}/{len(by_arm['VS'])} "
                      f"({vs_search_rate:.1%})")


if __name__ == "__main__":
    main()
