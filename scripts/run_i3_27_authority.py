#!/usr/bin/env python3
"""I3.27 Track 2: Authority calibration experiment.

Three arms on chain + defer tasks (the solvable failure types):
  VP    = A0 advisory (current behavior, all legal actions in schema)
  VP_A1 = schema narrowed to near-optimal set when confidence is clear
  VP_A2 = hard select single action when confidence is clear AND Q gap > threshold

Control law:
  Q gap large + confidence clear -> strong authority (A2)
  Q gap small + confidence ambiguous -> return choice to LLM (A0)

The scientific question:
  When executive confidence is very high, should DAPH advise or decide?

Usage:
  PYTHONPATH=. python3 scripts/run_i3_27_authority.py \
    --gguf-path /Users/dawsonblock/Downloads/qwen_gguf/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
    --output-dir experiments/i3_27/authority
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


def compute_q_gap(q_values: dict[str, float], near_optimal: list[str]) -> float:
    """Compute the Q gap between the best and second-best actions."""
    if len(q_values) < 2:
        return 0.0
    sorted_q = sorted(q_values.values(), reverse=True)
    return sorted_q[0] - sorted_q[1]


def run_trajectory(task, backend, i3_7e, utility, q_model, arm,
                   max_steps_override=None, authority_threshold=10.0):
    """Run a single trajectory with the specified authority mode.

    arm options:
      VP    = A0 advisory (all legal actions in schema)
      VP_A1 = schema narrowed to near-optimal set when confidence is clear
      VP_A2 = hard select single action when confidence is clear AND Q gap > threshold
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
    authority_log = []
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

        if sf.get("steps_remaining", 0) <= 0:
            resource_exhaustion = True
        if sf.get("retrieval_remaining", 0) == 0 and sf.get("search_remaining", 0) == 0:
            if sf.get("verify_remaining", 0) == 0 and not t2:
                resource_exhaustion = True

        # === VP guidance (frozen, identical across arms) ===
        # C0 and B0 do not use Q/progress — skip computation for those arms
        if arm in ("C0", "B0"):
            q_values = {}
            near_optimal = []
            lower_value = []
            refined_set = []
            confidence = "ambiguous"
            q_gap = 0.0
            progress_scores = {}
        else:
            q_values = q_model.predict_q(sf, legal_actions)
            near_optimal, lower_value = compute_near_optimal_set(q_values, 3.0)
            progress_scores = compute_progress_scores(runtime, legal_actions, utility, executor)
            refined_set, confidence = apply_progress_tiebreak(near_optimal, progress_scores, 0.05)
            q_gap = compute_q_gap(q_values, refined_set)

        progress_log.append({
            "step": step_id,
            "q_values": {a: round(q, 2) for a, q in q_values.items()},
            "near_optimal_before_progress": near_optimal,
            "near_optimal_after_progress": refined_set,
            "progress_scores": {a: round(s.get("progress", 0), 4)
                                for a, s in progress_scores.items()},
            "confidence": confidence,
            "q_gap": round(q_gap, 2),
        })

        # === Authority mode selection ===
        # Determine the schema actions based on the arm
        authority_mode = "A0_advisory"
        schema_actions = allowed_decision.allowed  # default: all legal

        if arm == "C0":
            # C0: no guidance, all legal actions, no extra fields
            schema_actions = allowed_decision.allowed
            authority_mode = "C0_no_guidance"

        elif arm == "B0":
            # B0: global prior — all actions near-optimal, confidence ambiguous
            schema_actions = allowed_decision.allowed
            authority_mode = "B0_global_prior"

        elif arm == "VP":
            # A0: advisory, all legal actions
            schema_actions = allowed_decision.allowed
            authority_mode = "A0_advisory"

        elif arm == "VP_A1":
            # A1: schema narrowed to near-optimal set when confidence is clear
            if confidence == "clear" and len(refined_set) >= 1:
                schema_actions = frozenset(refined_set) & allowed_decision.allowed
                if len(schema_actions) == 0:
                    schema_actions = allowed_decision.allowed  # fallback
                    authority_mode = "A0_advisory"
                else:
                    authority_mode = "A1_schema_narrowed"
            else:
                schema_actions = allowed_decision.allowed
                authority_mode = "A0_advisory"

        elif arm == "VP_A2":
            # A2: hard select single action when confidence is clear AND Q gap > threshold
            if (confidence == "clear" and len(refined_set) == 1
                    and q_gap > authority_threshold):
                schema_actions = frozenset(refined_set) & allowed_decision.allowed
                if len(schema_actions) == 0:
                    schema_actions = allowed_decision.allowed  # fallback
                    authority_mode = "A0_advisory"
                else:
                    authority_mode = "A2_hard_select"
            elif confidence == "clear" and len(refined_set) >= 1:
                # A1: narrow but don't hard-select
                schema_actions = frozenset(refined_set) & allowed_decision.allowed
                if len(schema_actions) == 0:
                    schema_actions = allowed_decision.allowed
                    authority_mode = "A0_advisory"
                else:
                    authority_mode = "A1_schema_narrowed"
            else:
                schema_actions = allowed_decision.allowed
                authority_mode = "A0_advisory"

        elif arm == "VP_A2R":
            # A2 Restricted: hard select ONLY for terminal resolution actions
            # (ANSWER, DEFER) when confidence is clear AND Q gap > threshold.
            # Do NOT narrow for intermediate actions (STOP, RETRIEVE, VERIFY, etc.)
            TERMINAL_RESOLUTION = frozenset({"ANSWER", "DEFER"})
            if (confidence == "clear" and len(refined_set) == 1
                    and q_gap > authority_threshold
                    and refined_set[0] in TERMINAL_RESOLUTION):
                schema_actions = frozenset(refined_set) & allowed_decision.allowed
                if len(schema_actions) == 0:
                    schema_actions = allowed_decision.allowed
                    authority_mode = "A0_advisory"
                else:
                    authority_mode = "A2_hard_select"
            else:
                schema_actions = allowed_decision.allowed
                authority_mode = "A0_advisory"

        elif arm == "VP_A2A":
            # A2 Answer-only: hard select ONLY for ANSWER when confidence is clear
            # AND Q gap > threshold. Do NOT hard-select DEFER (Q can't distinguish
            # defer-correct from contradiction-incorrect states).
            if (confidence == "clear" and len(refined_set) == 1
                    and q_gap > authority_threshold
                    and refined_set[0] == "ANSWER"):
                schema_actions = frozenset(refined_set) & allowed_decision.allowed
                if len(schema_actions) == 0:
                    schema_actions = allowed_decision.allowed
                    authority_mode = "A0_advisory"
                else:
                    authority_mode = "A2_hard_select"
            else:
                schema_actions = allowed_decision.allowed
                authority_mode = "A0_advisory"

        authority_log.append({
            "step": step_id,
            "authority_mode": authority_mode,
            "legal_actions": legal_actions,
            "schema_actions": sorted(schema_actions),
            "refined_set": refined_set,
            "confidence": confidence,
            "q_gap": round(q_gap, 2),
        })

        # Build packet with guidance
        if arm == "C0":
            extra_fields = {}
        elif arm == "B0":
            extra_fields = {
                "near_optimal_actions": legal_actions,
                "lower_value_actions": [],
                "guidance_confidence": "ambiguous",
                "epistemic_phase": phase,
            }
        else:
            extra_fields = {
                "near_optimal_actions": refined_set,
                "lower_value_actions": lower_value,
                "guidance_confidence": confidence,
                "epistemic_phase": phase,
            }

        packet = i3_7e.build_mdsg_state_with_affordances_packet(evidence_snapshot)
        schema = build_action_schema(schema_actions)
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
                allowed_actions=schema_actions,
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

        if action_str not in schema_actions:
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
        "authority_log": authority_log,
    }


@dataclass
class AuthorityWriter:
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


# Frozen identity hashes
EXPECTED_IDENTITIES = {
    "gguf_sha256": "65b8fcd92af6b4fefa935c625d1ac27ea29dcb6ee14589c55a8f115ceaaa1423",
    "qcausal_v1_sha256": "d90d72dab250ba7ce032afcc9d430fabfd4e84dace3c4dbe4deb77de0f8c9729",
    "progress_rule_v1_sha256": "9f0bfc5eea1d24f97cb65020d0ba2569888c07110c1be189c2383ae9d62349b9",
    "packet_builder_sha256": "93e1b5767589143a7d7fde4c4cfb8260151dfa9b1fa30ffb41c4328c587cc8ef",
}


def _sha256_file(path: Path) -> str:
    import hashlib as _hl
    h = _hl.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_text(path: Path) -> str:
    import hashlib as _hl
    h = _hl.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def verify_frozen_identities(gguf_path: Path) -> dict:
    print("=" * 70)
    print("FROZEN IDENTITY VERIFICATION (fail-closed)")
    print("=" * 70)
    identities = {}

    print("  Computing GGUF SHA256...", flush=True)
    computed = _sha256_file(gguf_path)
    identities["gguf_sha256"] = computed
    expected = EXPECTED_IDENTITIES["gguf_sha256"]
    print(f"    GGUF:       {computed[:16]}... {'OK' if computed == expected else 'MISMATCH'}")
    if computed != expected:
        raise SystemExit(1)

    computed = _sha256_file(REPO_ROOT / "experiments/i3_5/pinned_policy/frozen_estimators/QCAUSAL_gbt.pkl")
    identities["qcausal_v1_sha256"] = computed
    expected = EXPECTED_IDENTITIES["qcausal_v1_sha256"]
    print(f"    QCAUSAL_V1: {computed[:16]}... {'OK' if computed == expected else 'MISMATCH'}")
    if computed != expected:
        raise SystemExit(1)

    computed = _sha256_text(REPO_ROOT / "daph/progress/progress_rule_v1.py")
    identities["progress_rule_v1_sha256"] = computed
    expected = EXPECTED_IDENTITIES["progress_rule_v1_sha256"]
    print(f"    Progress:   {computed[:16]}... {'OK' if computed == expected else 'MISMATCH'}")
    if computed != expected:
        raise SystemExit(1)

    computed = _sha256_text(REPO_ROOT / "daph/executive/packet_builder.py")
    identities["packet_builder_sha256"] = computed
    expected = EXPECTED_IDENTITIES["packet_builder_sha256"]
    print(f"    PktBuilder:  {computed[:16]}... {'OK' if computed == expected else 'MISMATCH'}")
    if computed != expected:
        raise SystemExit(1)

    import subprocess, platform
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()
    identities["source_commit"] = source_commit
    runtime_version = f"Python {platform.python_version()} ({platform.machine()})"
    identities["runtime_version"] = runtime_version
    print(f"    Commit:     {source_commit[:16]}...")
    print(f"    Runtime:    {runtime_version}")
    print("  ALL IDENTITIES VERIFIED.")
    print("=" * 70)
    print()
    return identities


def main():
    parser = argparse.ArgumentParser(description="I3.27 Authority calibration")
    parser.add_argument("--gguf-path", required=True)
    parser.add_argument("--model-name", default="qwen2.5-7b-instruct")
    parser.add_argument("--output-dir", default="experiments/i3_27/authority")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--authority-threshold", type=float, default=10.0,
                       help="Q gap threshold for A2 hard selection")
    parser.add_argument("--categories", default="chain,defer",
                       help="Comma-separated categories to run on")
    parser.add_argument("--arms", default="VP,VP_A1,VP_A2,VP_A2R",
                       help="Comma-separated arms to run")
    parser.add_argument("--benchmark", default="development",
                       choices=["development", "confirmation"],
                       help="Which benchmark to use (development=seed7719, confirmation=fresh seed)")
    args = parser.parse_args()

    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Identity verification
    gguf_path = Path(args.gguf_path)
    if not gguf_path.exists():
        print(f"FATAL: GGUF file not found: {gguf_path}")
        raise SystemExit(1)
    identities = verify_frozen_identities(gguf_path)

    # Load i3_7e
    print("Loading i3_7e module...")
    spec = importlib.util.spec_from_file_location(
        "i3_7e", str(REPO_ROOT / "scripts" / "run_i3_7e_compact_governor.py"))
    i3_7e = importlib.util.module_from_spec(spec)
    sys.modules["i3_7e"] = i3_7e
    spec.loader.exec_module(i3_7e)

    # Load benchmark
    if args.benchmark == "development":
        print("Loading development benchmark (seed=7719)...")
        all_tasks = generate_development_benchmark(seed=7719)
        bench_hash = compute_benchmark_hash(all_tasks)
        bench_version = "I3.26_DEVELOPMENT_BENCHMARK_V1"
    else:
        # Fresh confirmation benchmark with derived seed
        import hashlib as _hl
        fresh_seed = int(_hl.sha256(b"i3_27_authority_confirmation_v1").hexdigest()[:8], 16) % (2**31)
        print(f"Loading fresh confirmation benchmark (seed={fresh_seed})...")
        from hrm_adaptive_memory.executive.evidence_benchmark.i3_5_confirmation_generator import (
            generate_confirmation_benchmark,
        )
        all_tasks = generate_confirmation_benchmark(seed=fresh_seed)
        # Compute hash
        task_json = json.dumps([t.as_dict() for t in all_tasks], sort_keys=True)
        bench_hash = _hl.sha256(task_json.encode()).hexdigest()
        bench_version = "I3.27_CONFIRMATION_BENCHMARK_V1"
    requested_cats = set(args.categories.split(","))
    tasks = [t for t in all_tasks if t.category in requested_cats]
    print(f"  {len(tasks)} tasks (categories: {requested_cats})")
    print(f"  Benchmark hash: {bench_hash}")
    print(f"  Benchmark version: {bench_version}")

    # Load utility
    from hrm_adaptive_memory.executive.metareasoning_utility import MetareasoningUtility
    utility = MetareasoningUtility.from_file(
        REPO_ROOT / "configs" / "v2b_i3_1_utility_v1.json")

    # Load frozen QCAUSAL_V1
    print("Loading frozen QCAUSAL_V1...")
    est_dir = REPO_ROOT / "experiments/i3_5/pinned_policy/frozen_estimators"
    q_model = QCAUSALModel.load(est_dir)

    # Initialize backend
    print(f"Initializing backend with {args.gguf_path}...")
    from hrm_adaptive_memory.executive.model_backend import R2DirectLlamaBackend
    backend = R2DirectLlamaBackend(
        model_name=args.model_name,
        model_path=args.gguf_path,
        n_ctx=args.n_ctx,
    )
    print("  Backend initialized.")

    # Manifest
    timestamp = datetime.now(timezone.utc).isoformat()
    run_id = hashlib.sha256(f"i3_27_authority:{timestamp}".encode()).hexdigest()[:16]
    arms = args.arms.split(",")
    manifest = {
        "run_id": run_id,
        "timestamp": timestamp,
        "benchmark_hash": bench_hash,
        "benchmark_version": bench_version,
        "n_tasks": len(tasks),
        "arms": arms,
        "categories": list(requested_cats),
        "authority_threshold": args.authority_threshold,
        "identities": identities,
    }
    with open(output_dir / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    writer = AuthorityWriter(output_dir)
    completed = writer.load_completed() if args.resume else set()

    total = len(tasks) * len(arms)
    done = 0
    errors = 0
    start_time = time.time()

    print(f"\nRunning {total} trajectories ({len(tasks)} tasks x {len(arms)} arms)")
    print(f"Authority threshold: {args.authority_threshold}")

    for arm in arms:
        for task in tasks:
            key = f"{task.task_id}:{arm}"
            if key in completed:
                done += 1
                continue

            try:
                result = run_trajectory(task, backend, i3_7e, utility, q_model, arm,
                                       authority_threshold=args.authority_threshold)
                writer.write_traj(result)
                done += 1
                completed.add(key)

                if done % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = done / elapsed if elapsed > 0 else 0
                    eta = (total - done) / rate if rate > 0 else 0
                    print(f"  [{done}/{total}] {arm} {task.task_id} "
                          f"success={result['success']} "
                          f"({rate:.1f}/s, ETA {eta:.0f}s)")

            except Exception as e:
                errors += 1
                writer.write_error({
                    "task_id": task.task_id, "arm": arm,
                    "error": str(e),
                })
                print(f"  ERROR: {arm} {task.task_id}: {e}")

    elapsed = time.time() - start_time
    print(f"\nDone: {done}/{total} trajectories, {errors} errors, {elapsed:.1f}s")

    # Summary
    traj_path = output_dir / "trajectories.jsonl"
    if traj_path.exists():
        records = [json.loads(line) for line in open(traj_path)]
        by_arm = defaultdict(list)
        for r in records:
            by_arm[r["arm"]].append(r)

        print(f"\n{'='*70}")
        print("AUTHORITY CALIBRATION SUMMARY")
        print(f"{'='*70}")
        for arm in arms:
            arm_records = by_arm.get(arm, [])
            if not arm_records:
                continue
            n = len(arm_records)
            successes = sum(1 for r in arm_records if r["success"])
            mean_u = np.mean([r["realized_utility"] for r in arm_records])
            print(f"\n  {arm}:")
            print(f"    N: {n}")
            print(f"    Success: {successes}/{n} ({successes/n:.2%})")
            print(f"    Mean utility: {mean_u:.2f}")

        # Per-category
        print(f"\n  Per-category:")
        for cat in sorted(requested_cats):
            print(f"    {cat}:")
            for arm in arms:
                arm_cat = [r for r in by_arm.get(arm, []) if r["category"] == cat]
                if arm_cat:
                    s = sum(1 for r in arm_cat if r["success"])
                    print(f"      {arm}: {s}/{len(arm_cat)}")

        # Paired comparisons
        if "VP" in by_arm:
            vp_by_task = {r["task_id"]: r for r in by_arm["VP"]}
            for challenger in ["VP_A1", "VP_A2"]:
                if challenger not in by_arm:
                    continue
                ch_by_task = {r["task_id"]: r for r in by_arm[challenger]}
                common = set(vp_by_task.keys()) & set(ch_by_task.keys())
                if not common:
                    continue

                vp_u = np.array([vp_by_task[t]["realized_utility"] for t in common])
                ch_u = np.array([ch_by_task[t]["realized_utility"] for t in common])
                delta = ch_u - vp_u
                mean_delta = np.mean(delta)
                ci = 1.96 * np.std(delta) / np.sqrt(len(delta))

                vp_s = sum(1 for t in common if vp_by_task[t]["success"])
                ch_s = sum(1 for t in common if ch_by_task[t]["success"])
                rescues = sum(1 for t in common
                             if not vp_by_task[t]["success"] and ch_by_task[t]["success"])
                breaks = sum(1 for t in common
                            if vp_by_task[t]["success"] and not ch_by_task[t]["success"])

                print(f"\n  Paired: {challenger} vs VP ({len(common)} tasks):")
                print(f"    Delta U: {mean_delta:.4f} (95% CI [{mean_delta-ci:.4f}, {mean_delta+ci:.4f}])")
                print(f"    Success VP: {vp_s}, {challenger}: {ch_s}")
                print(f"    Rescues: {rescues}, Breaks: {breaks}")

                # Authority mode stats
                if challenger in by_arm:
                    a2_count = sum(1 for r in by_arm[challenger]
                                  if any(a["authority_mode"] == "A2_hard_select"
                                         for a in r.get("authority_log", [])))
                    a1_count = sum(1 for r in by_arm[challenger]
                                  if any(a["authority_mode"] == "A1_schema_narrowed"
                                         for a in r.get("authority_log", [])))
                    print(f"    A2 hard-select triggered: {a2_count}/{len(by_arm[challenger])}")
                    print(f"    A1 schema-narrowed triggered: {a1_count}/{len(by_arm[challenger])}")


if __name__ == "__main__":
    main()
