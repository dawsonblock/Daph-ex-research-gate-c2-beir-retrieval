#!/usr/bin/env python3
"""I3.30: Targeted Live Study with 5 Strata (D1-D5).

Compares frozen V1 against candidate V3 (Q_V3R + positive structural certificates).

Five strata:
  D1: safe resource-exhausted DEFER
  D2: post-verification DEFER
  D3: unresolved/contradictory control
  D4: post-verification ANSWER
  D5: post-verification ambiguous/continue (competing verified support)

Pre-registered live promotion rule:
  FalseAuthorityRate_ANSWER = 0
  AND FalseAuthorityRate_DEFER = 0
  AND rescues > breaks
  AND breaks = 0 for hard-authority interventions
  AND success_V3 >= success_V1
  AND ΔU > 0

Usage:
  PYTHONPATH=. python3 scripts/run_i3_30_live_study.py \
    --gguf-path /Users/dawsonblock/Downloads/qwen_gguf/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
    --output-dir experiments/i3_30/live_study \
    --resume
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from hrm_adaptive_memory.cognitive_control.core import DecisionAction
from hrm_adaptive_memory.cognitive_control.state import VerificationState
from hrm_adaptive_memory.executive.evidence_benchmark import (
    initial_evidence_runtime, build_evidence_snapshot,
)
from hrm_adaptive_memory.executive.evidence_benchmark.executor import (
    EvidenceExecutor, valid_verify_targets,
)
from hrm_adaptive_memory.executive.resources import ResourceState
from hrm_adaptive_memory.executive.model_decoder import decode_output_strict
from hrm_adaptive_memory.executive.metareasoning_utility import MetareasoningUtility
from daph.intervention.checkpoint import compute_state_features
from daph.authority.policy import (
    AuthorityMode, StructuralState, decide_authority, build_receipt,
    AUTHORITY_THRESHOLD,
)
from daph.authority.policy_v3 import (
    StructuralStateV3, decide_authority_v3,
)

# Import from I3.29 runner
from run_i3_29_live_safety import (
    QModelV1, QModelV2R,
    classify_phase_simple, compute_near_optimal_set,
    compute_progress_scores, apply_progress_tiebreak,
    compute_q_gap, get_structural_state,
    run_trajectory as run_trajectory_v1v2,
    _make_result,
)
from run_i3_28_rep_repair import (
    extract_v1_features, compute_structural_features,
)
from run_i3_30_train_v3r import extract_v3_features, get_v3_feature_keys


# ============================================================
# Q_V3R model
# ============================================================

class QModelV3R:
    """Q_CAUSAL_V3R — V3 model with post-verification features (56 features)."""
    def __init__(self, model, feature_keys):
        self.model = model
        self.feature_keys = feature_keys

    @classmethod
    def load(cls, model_path: Path, schema_path: Path):
        model = pickle.loads(model_path.read_bytes())
        with open(schema_path) as f:
            schema = json.load(f)
        return cls(model, schema["featurekeys"])

    def predict_q(self, sf: dict, legal_actions: list[str],
                  v3_struct: dict) -> dict[str, float]:
        X = np.array([[extract_v3_features(sf, a, v3_struct)[k]
                      for k in self.feature_keys]
                      for a in legal_actions])
        preds = self.model.predict(X)
        return dict(zip(legal_actions, [float(p) for p in preds]))


# ============================================================
# V3 structural state
# ============================================================

def get_structural_state_v3(runtime) -> StructuralStateV3:
    """Build StructuralStateV3 from runtime."""
    visible_ev = []
    for ev in runtime.visible_evidence:
        visible_ev.append({
            "evidence_id": ev.evidence_id,
            "supports": list(ev.supports),
            "contradicts": list(ev.contradicts),
            "verification_state": ev.verification_state.name,
            "retrieved": ev.retrieved,
        })
    v2_struct = compute_structural_features(visible_ev)

    # Compute V3 features
    from run_i3_30_v3_coverage import compute_v3_features
    hyps_list = [{"hypothesis_id": h.hypothesis_id, "answer_action": h.answer_action.value}
                 for h in runtime.task.hypotheses]
    v3 = compute_v3_features(visible_ev, hyps_list)

    can_verify = bool(valid_verify_targets(runtime))
    verify_budget_exhausted = (
        runtime.resources.verification_calls_used >= runtime.resources.budget.max_verification_calls
    )
    all_evidence_verified = all(
        ev.verification_state != VerificationState.UNVERIFIED
        for ev in runtime.visible_evidence
        if ev.retrieved
    )

    return StructuralStateV3(
        has_competing_unverified_support=bool(v2_struct["has_competing_unverified_support"]),
        n_hyp_unverified_support=v2_struct["n_hyp_unverified_support"],
        n_hyp_unverified_contradiction=v2_struct["n_hyp_unverified_contradiction"],
        can_verify=can_verify,
        verify_budget_exhausted=verify_budget_exhausted,
        all_evidence_verified=all_evidence_verified,
        n_hyp_with_verified_support=v3["n_hyp_with_verified_support"],
        n_hyp_with_verified_contradiction=v3["n_hyp_with_verified_contradiction"],
        n_hyp_with_mixed_verified=v3["n_hyp_with_mixed_verified"],
        n_viable_hypotheses=v3["n_viable_hypotheses"],
        n_eliminated_hypotheses=v3["n_eliminated_hypotheses"],
        has_unique_verified_supported_hypothesis=bool(v3["has_unique_verified_supported_hypothesis"]),
        has_verified_unresolved_competition=bool(v3["has_verified_unresolved_competition"]),
        verified_hyp_action_is_answer=bool(v3["verified_hyp_action_is_answer"]),
        verified_hyp_action_is_defer=bool(v3["verified_hyp_action_is_defer"]),
    )


# ============================================================
# Trajectory runner with V3 arm
# ============================================================

def run_trajectory_v3(task, backend, i3_7e, utility,
                      q_v1: QModelV1, q_v3r: QModelV3R, arm: str,
                      d2_pre_verify: bool = False,
                      d5_pre_verify: bool = False):
    """Run a single trajectory with V1 or V3 arm."""
    from r2_schema import build_action_schema, schema_sha256
    from r2_allowed_actions import compute_allowed_actions, ActionState, EmptyAllowedActionSet, C0
    from hrm_adaptive_memory.executive.evidence_benchmark.i3_29_safety_generator import get_budget_for_task

    budget = get_budget_for_task(task)
    max_steps = budget.max_executive_steps
    resources = ResourceState(budget=budget)
    runtime = initial_evidence_runtime(task, resources)
    executor = EvidenceExecutor()

    # D2: pre-verify first evidence
    if d2_pre_verify:
        valid = valid_verify_targets(runtime)
        if valid:
            res = executor.execute(runtime, DecisionAction.VERIFY,
                                   target_evidence_id=valid[0])
            runtime = res.runtime
            if res.terminal:
                return _make_result(task, arm, 0.0, False, "PRE_VERIFY_TERMINAL",
                                    [], [], [], False, False)

    # D5: pre-verify to create competing verified support
    if d5_pre_verify:
        valid = valid_verify_targets(runtime)
        if valid:
            # Verify first two evidence items to create competing support
            for target_id in valid[:2]:
                try:
                    res = executor.execute(runtime, DecisionAction.VERIFY,
                                           target_evidence_id=target_id)
                    runtime = res.runtime
                    if res.terminal:
                        return _make_result(task, arm, 0.0, False, "PRE_VERIFY_TERMINAL",
                                            [], [], [], False, False)
                except Exception:
                    break

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
    authority_log = []
    receipts = []
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

        # === Q prediction ===
        if arm == "V1":
            q_values = q_v1.predict_q(sf, legal_actions)
        else:
            # V3: use V3R model with V3 structural features
            structural_v3 = get_structural_state_v3(runtime)
            v3_struct_dict = structural_v3.as_dict()
            v3_struct_dict["n_hyp_unverified_support"] = structural_v3.n_hyp_unverified_support
            v3_struct_dict["n_hyp_unverified_contradiction"] = structural_v3.n_hyp_unverified_contradiction
            v3_struct_dict["has_competing_unverified_support"] = int(structural_v3.has_competing_unverified_support)
            q_values = q_v3r.predict_q(sf, legal_actions, v3_struct_dict)

        near_optimal, lower_value = compute_near_optimal_set(q_values, 3.0)
        progress_scores = compute_progress_scores(runtime, legal_actions, utility, executor)
        refined_set, confidence = apply_progress_tiebreak(near_optimal, progress_scores, 0.05)
        q_gap = compute_q_gap(q_values)

        # === Authority decision ===
        schema_actions = allowed_decision.allowed
        authority_mode_str = "A0_advisory"
        forced_action = None

        if arm == "V1":
            # V1: ANSWER-only hard authority (A2A_RULE_V1)
            AUTHORITATIVE = frozenset({"ANSWER"})
            if (confidence == "clear" and len(refined_set) == 1
                    and q_gap >= AUTHORITY_THRESHOLD
                    and refined_set[0] in AUTHORITATIVE):
                schema_actions = frozenset({refined_set[0]}) & allowed_decision.allowed
                if schema_actions:
                    authority_mode_str = "A2A_hard_select"
                    forced_action = refined_set[0]
                else:
                    schema_actions = allowed_decision.allowed

        elif arm == "V3":
            # V3: Positive structural certificate authority (A2AD_V3)
            structural_v3 = get_structural_state_v3(runtime)
            decision = decide_authority_v3(
                q_values=q_values,
                legal_actions=legal_actions,
                structural=structural_v3,
            )

            if decision.mode == AuthorityMode.HARD_ANSWER:
                schema_actions = frozenset({"ANSWER"}) & allowed_decision.allowed
                if schema_actions:
                    authority_mode_str = "A2AD_hard_ANSWER"
                    forced_action = "ANSWER"
                else:
                    schema_actions = allowed_decision.allowed
            elif decision.mode == AuthorityMode.HARD_DEFER:
                schema_actions = frozenset({"DEFER"}) & allowed_decision.allowed
                if schema_actions:
                    authority_mode_str = "A2AD_hard_DEFER"
                    forced_action = "DEFER"
                else:
                    schema_actions = allowed_decision.allowed
            else:
                schema_actions = allowed_decision.allowed

            # Build receipt
            receipt = build_receipt(
                state_features=sf,
                legal_actions=legal_actions,
                q_values=q_values,
                structural=structural_v3.to_v2(),
                decision=decision,
            )
            receipt["step"] = step_id
            receipt["arm"] = arm
            receipt["authority_mode"] = authority_mode_str
            receipt["v3_structural"] = structural_v3.as_dict()
            receipts.append(receipt)

        authority_log.append({
            "step": step_id,
            "authority_mode": authority_mode_str,
            "legal_actions": legal_actions,
            "schema_actions": sorted(schema_actions),
            "refined_set": refined_set,
            "confidence": confidence,
            "q_gap": round(q_gap, 2),
            "forced_action": forced_action,
        })

        # === Build packet ===
        extra_fields = {
            "near_optimal_actions": refined_set,
            "lower_value_actions": lower_value,
            "guidance_confidence": confidence,
            "epistemic_phase": phase,
        }

        packet = i3_7e.build_mdsg_state_with_affordances_packet(evidence_snapshot)
        schema = build_action_schema(schema_actions)
        schema_sha = schema_sha256(schema)

        packet_dict = json.loads(i3_7e.evidence_packet_json(packet))
        packet_dict["executive_guidance"] = extra_fields
        user_prompt = json.dumps(packet_dict, indent=2)

        # === Call Qwen ===
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

        prior_actions.append(action_str)
        prior_outcomes.append("EXECUTED")
        actions_taken.append(action_str)

        if exec_res.terminal:
            terminal = True
            terminal_action = action_str
            success = bool(exec_res.task_success)
            tr = utility.terminal_reward(action, success)
            realized += tr
            if success:
                terminal_result = "SUCCESS"
            else:
                terminal_result = "TERMINAL_WRONG"
                if action_str == "DEFER":
                    premature_defer = (sf.get("n_verified", 0) == 0)
                elif action_str == "ANSWER":
                    premature_answer = (sf.get("n_verified", 0) == 0)
            break

    if not terminal:
        realized -= 0.5  # step limit penalty

    return _make_result(
        task, arm, realized, success, terminal_result,
        actions_taken, authority_log, receipts,
        premature_defer, premature_answer,
    )


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="I3.30 Live Study")
    parser.add_argument("--gguf-path", required=True)
    parser.add_argument("--output-dir", default="experiments/i3_30/live_study")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--n-per-stratum", type=int, default=30)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    q_v1 = QModelV1.load(
        REPO_ROOT / "experiments/i3_5/pinned_policy/Q_CAUSAL_V1.pkl",
        REPO_ROOT / "experiments/i3_5/pinned_policy/feature_schema_v1.json",
    )
    q_v3r = QModelV3R.load(
        REPO_ROOT / "experiments/i3_30/Q_V3R_postverify.pkl",
        REPO_ROOT / "experiments/i3_30/v3_feature_schema.json",
    )

    # Generate benchmark (reuse I3.29 generator for D1-D4, add D5)
    from hrm_adaptive_memory.executive.evidence_benchmark.i3_29_safety_generator import (
        generate_i3_29_benchmark, get_budget_for_task,
    )
    tasks = generate_i3_29_benchmark(seed=9817)

    # For now, use I3.29 benchmark (D1-D4). D5 will be added.
    # D5 tasks: post-verification ambiguous/continue (competing verified support)
    # These can be derived from D3 tasks by pre-verifying evidence

    print(f"Tasks: {len(tasks)}")
    print(f"Arms: V1, V3")
    print(f"Output: {output_dir}")

    # The live study requires the Qwen model.
    # This is a placeholder for the actual run — the user should execute it.
    print("\nTo run the live study:")
    print(f"  PYTHONPATH=. python3 scripts/run_i3_30_live_study.py \\")
    print(f"    --gguf-path {args.gguf_path} \\")
    print(f"    --output-dir {output_dir} \\")
    print(f"    --resume")


if __name__ == "__main__":
    main()
