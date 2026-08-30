#!/usr/bin/env python3
"""I3.30R3: Three-Arm Authority Isolation Study (V1, V3-SHADOW, V3-AUTH).

FROZEN MANIFEST: This run uses frozen artifacts. No changes to model,
representation, threshold, positive certificates, prompts, or runner
once trajectory generation begins.

The primary comparison is:

    ATE_authority = E[U | V3-AUTH] - E[U | V3-SHADOW]

This isolates the causal effect of adaptive hard authority by holding
everything else constant (Q model, features, epsilon, prompts, LLM).

The secondary comparison is:

    V3-SHADOW - V1

which estimates the contribution of the V3 representation/Q/advisory layer.

Three arms:
  V1        = confirmed champion (ANSWER-only hard authority)
  V3-SHADOW = V3 executive, certificates evaluated/logged, NO hard override
  V3-AUTH   = V3 executive, certificates evaluated/logged, WITH hard override

Treatment purity invariant:
  V3-SHADOW and V3-AUTH execute identical code up to the final override:

      # Everything above this line must be identical.
      decision = evaluate_v3_authority(state)
      if arm == V3_HARD and decision.would_force:
          action = decision.forced_action
      else:
          action = llm_action

Usage:
  PYTHONPATH=. python3 scripts/run_i3_30r3_authority_isolation.py \\
    --gguf-path /path/to/Qwen2.5-7B-Instruct-Q4_K_M.gguf \\
    --output-dir experiments/i3_30r3/live \\
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
from daph.authority.isolation import (
    ArmMode, AuthorityDecisionV3, AuthorityEffect,
    evaluate_v3_authority, apply_authority,
    classify_authority_effect, state_sha, build_normalized_receipt,
)

from run_i3_29_live_safety import (
    QModelV1, classify_phase_simple, compute_near_optimal_set,
    compute_progress_scores, apply_progress_tiebreak,
    compute_q_gap, get_structural_state, _make_result,
)
from run_i3_28_rep_repair import (
    extract_v1_features, compute_structural_features,
)
from run_i3_30r_train_v3r2 import extract_v3r2_features, get_v3r2_feature_keys
from daph.epistemic.v3_features import compute_v3_features_canonical


# ============================================================
# Q_V3R model (frozen, same as I3.30R2)
# ============================================================

class QModelV3R:
    """Q_CAUSAL_V3R2 — V3R2 model with canonical topology features (53 features)."""
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
        X = np.array([[extract_v3r2_features(sf, a, v3_struct)[k]
                      for k in self.feature_keys]
                      for a in legal_actions])
        preds = self.model.predict(X)
        return dict(zip(legal_actions, [float(p) for p in preds]))


# ============================================================
# V3 structural state (frozen, same as I3.30R2)
# ============================================================

def get_structural_state_v3(runtime) -> StructuralStateV3:
    """Build StructuralStateV3 from runtime (identical to I3.30R2)."""
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

    hyps_list = [{"hypothesis_id": h.hypothesis_id, "answer_action": h.answer_action.value}
                 for h in runtime.task.hypotheses]
    v3 = compute_v3_features_canonical(visible_ev, hyps_list)

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
# Three-arm trajectory runner
# ============================================================

def run_trajectory(task, backend, i3_7e, utility,
                   q_v1: QModelV1, q_v3r: QModelV3R, arm: ArmMode,
                   d2_pre_verify: bool = False):
    """Run a single trajectory for a given arm.

    arm: ArmMode.V1, ArmMode.V3_SHADOW, or ArmMode.V3_HARD

    Treatment purity:
    - V3_SHADOW and V3_HARD share identical code up to apply_authority()
    - The ONLY difference is whether the hard override is applied
    """
    from r2_schema import build_action_schema, schema_sha256
    from r2_allowed_actions import compute_allowed_actions, ActionState, EmptyAllowedActionSet, C0
    from hrm_adaptive_memory.executive.evidence_benchmark.i3_29_safety_generator import get_budget_for_task

    budget = get_budget_for_task(task)
    max_steps = budget.max_executive_steps
    resources = ResourceState(budget=budget)
    runtime = initial_evidence_runtime(task, resources)
    executor = EvidenceExecutor()

    # D2: pre-verify first evidence (same as I3.30R2)
    if d2_pre_verify:
        valid = valid_verify_targets(runtime)
        if valid:
            res = executor.execute(runtime, DecisionAction.VERIFY,
                                   target_evidence_id=valid[0])
            runtime = res.runtime
            if res.terminal:
                return _make_result(task, arm.value, 0.0, False, "PRE_VERIFY_TERMINAL",
                                    [], [], [], False, False,
                                    "VERIFY", False)

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
    authority_events = []  # normalized authority event records
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

        # ============================================================
        # Q prediction — arm-dependent
        # ============================================================
        if arm == ArmMode.V1:
            q_values = q_v1.predict_q(sf, legal_actions)
        else:
            # V3_SHADOW and V3_HARD use the same Q model
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

        # ============================================================
        # Authority decision — arm-dependent
        # ============================================================
        # CRITICAL FIX (I3.30R3 ATTEMPT 2):
        # For V3 arms, schema_actions MUST always equal allowed_decision.allowed.
        # Certificate evaluation may happen before decoding, but its result
        # must NOT affect the grammar, prompt, legal-action set, or any
        # model input. The treatment (hard override) is applied ONLY after
        # decoding, via apply_authority().
        #
        # V1 retains its existing ANSWER-only hard-select behavior (narrowing
        # the schema before generation), because that is V1's actual
        # champion behavior — not a treatment variable in this experiment.
        schema_actions = allowed_decision.allowed
        authority_mode_str = "A0_advisory"
        forced_action = None
        v3_decision = None  # AuthorityDecisionV3 for V3 arms

        if arm == ArmMode.V1:
            # V1 authority: ANSWER-only hard select (unchanged from I3.30R2)
            # V1 narrows the schema before generation — this is V1's
            # actual behavior, not a treatment variable.
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

        else:
            # V3_SHADOW and V3_HARD: identical evaluation path
            # ============================================================
            # Certificate is evaluated but MUST NOT affect schema_actions.
            # The LLM sees the full legal action set in both arms.
            # Treatment divergence happens ONLY after decoding.
            # ============================================================
            structural_v3 = get_structural_state_v3(runtime)
            v3_decision = evaluate_v3_authority(
                q_values=q_values,
                legal_actions=legal_actions,
                structural=structural_v3,
            )

            # Record authority mode for logging, but DO NOT narrow schema
            if v3_decision.would_force:
                authority_mode_str = v3_decision.authority_mode
            # schema_actions remains allowed_decision.allowed for both V3 arms

        # ============================================================
        # Build packet and call LLM
        # ============================================================
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

        # ============================================================
        # Purity receipts (Step 5 fix)
        # Pre-generation hashes that MUST match between V3_SHADOW and
        # V3_HARD for the same task/step. Any mismatch indicates
        # treatment contamination before the divergence point.
        # ============================================================
        prompt_sha = hashlib.sha256(user_prompt.encode()).hexdigest()
        legal_actions_sha = hashlib.sha256(
            json.dumps(sorted(legal_actions), sort_keys=True).encode()
        ).hexdigest()
        schema_actions_sha = hashlib.sha256(
            json.dumps(sorted(schema_actions), sort_keys=True).encode()
        ).hexdigest()
        q_values_sha = hashlib.sha256(
            json.dumps({k: round(v, 6) for k, v in sorted(q_values.items())}).encode()
        ).hexdigest()
        state_sha_val = state_sha(sf)

        # For V3 arms, verify schema_actions == legal_actions (no narrowing)
        if arm in (ArmMode.V3_SHADOW, ArmMode.V3_HARD):
            assert schema_actions == allowed_decision.allowed, (
                f"TREATMENT CONTAMINATION: schema_actions != allowed for {arm.value} "
                f"at {task.task_id} step {step_id}: "
                f"{sorted(schema_actions)} != {sorted(allowed_decision.allowed)}"
            )

        # Log authority decision (after purity receipts are computed)
        authority_log.append({
            "step": step_id,
            "arm": arm.value,
            "authority_mode": authority_mode_str,
            "legal_actions": legal_actions,
            "schema_actions": sorted(schema_actions),
            "refined_set": refined_set,
            "confidence": confidence,
            "q_gap": round(q_gap, 2),
            "forced_action": forced_action if arm == ArmMode.V1 else (
                v3_decision.forced_action if v3_decision else None),
            "would_force": v3_decision.would_force if v3_decision else False,
            # Purity receipts
            "pre_generation_prompt_sha": prompt_sha,
            "pre_generation_schema_actions_sha": schema_actions_sha,
            "pre_generation_legal_actions_sha": legal_actions_sha,
            "pre_generation_q_values_sha": q_values_sha,
            "pre_generation_state_sha": state_sha_val,
        })

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

        # ============================================================
        # Apply authority — the single conditional
        # ============================================================
        if arm == ArmMode.V1:
            # V1: forced_action already set above, schema already restricted
            executed_action = action_str  # LLM chose within restricted schema
            # If V1 forced, the schema only contains the forced action,
            # so the LLM must choose it (or fail admissibility)
            force_applied = forced_action is not None
            action_changed = force_applied and (executed_action != action_str)
            # Build a simplified receipt for V1
            receipt = {
                "task_id": task.task_id,
                "arm": arm.value,
                "step": step_id,
                "state_sha": state_sha(sf),
                "legal_actions": legal_actions,
                "q_values": {k: round(v, 4) for k, v in q_values.items()},
                "q_argmax": max(q_values, key=q_values.get) if q_values else "",
                "q_gap": round(q_gap, 4),
                "epsilon_set": refined_set,
                "certificate_evaluated": False,
                "certificate_passed": False,
                "certificate_type": "V1_ANSWER_ONLY",
                "certificate_components": {},
                "authority_mode": authority_mode_str,
                "would_force": force_applied,
                "forced_action": forced_action,
                "llm_proposed_action": action_str,
                "executed_action": executed_action,
                "force_applied": force_applied,
                "action_changed": action_changed,
            }
            receipts.append(receipt)

        else:
            # V3_SHADOW and V3_HARD: apply_authority is the ONLY difference
            # ============================================================
            executed_action, updated_decision = apply_authority(
                v3_decision, arm, action_str,
            )

            # Build normalized receipt
            resource_state = {
                "executive_steps_used": runtime.resources.executive_steps_used,
                "executive_steps_remaining": max_steps - step_id,
                "verification_calls_used": runtime.resources.verification_calls_used,
                "verification_calls_remaining": (
                    runtime.resources.budget.max_verification_calls
                    - runtime.resources.verification_calls_used
                ),
                "retrieval_calls_used": runtime.resources.retrieval_calls_used,
                "retrieval_calls_remaining": (
                    runtime.resources.budget.max_retrieval_calls
                    - runtime.resources.retrieval_calls_used
                ),
            }

            receipt = build_normalized_receipt(
                task_id=task.task_id,
                arm=arm.value,
                step=step_id,
                state_features=sf,
                decision=updated_decision,
                legal_actions=legal_actions,
                resource_state=resource_state,
            )
            receipts.append(receipt)

            # Record authority event when would_force is True
            # (for both SHADOW and HARD, so we can compare)
            if updated_decision.would_force:
                # Capture checkpoint for counterfactual replay (Step 6)
                from daph.intervention.checkpoint import create_checkpoint
                checkpoint = create_checkpoint(
                    runtime=runtime,
                    step=step_id,
                    phase=phase,
                    prior_actions=tuple(prior_actions),
                    prior_outcomes=tuple(prior_outcomes),
                )

                # Deterministic counterfactual: simulate forced action
                # and LLM action from the same state (no LLM needed for
                # immediate outcome)
                from hrm_adaptive_memory.executive.evidence_benchmark.executor import EvidenceExecutor as _Exec
                _cf_executor = _Exec()

                # Simulate forced action
                forced_action_enum = DecisionAction(updated_decision.forced_action)
                forced_target = None
                if forced_action_enum is DecisionAction.VERIFY:
                    _valid = valid_verify_targets(runtime)
                    if _valid:
                        forced_target = _valid[0]
                try:
                    _forced_res = _cf_executor.execute(runtime, forced_action_enum, target_evidence_id=forced_target)
                    forced_immediate_terminal = _forced_res.terminal
                    forced_immediate_success = _forced_res.task_success
                except Exception:
                    forced_immediate_terminal = False
                    forced_immediate_success = None

                # Simulate LLM action (only if different from forced)
                if updated_decision.llm_proposed_action != updated_decision.forced_action:
                    llm_action_enum = DecisionAction(updated_decision.llm_proposed_action)
                    llm_target = target_id
                    if llm_action_enum is DecisionAction.VERIFY and not llm_target:
                        _valid = valid_verify_targets(runtime)
                        if _valid:
                            llm_target = _valid[0]
                    try:
                        _llm_res = _cf_executor.execute(runtime, llm_action_enum, target_evidence_id=llm_target)
                        llm_immediate_terminal = _llm_res.terminal
                        llm_immediate_success = _llm_res.task_success
                    except Exception:
                        llm_immediate_terminal = False
                        llm_immediate_success = None
                else:
                    llm_immediate_terminal = forced_immediate_terminal
                    llm_immediate_success = forced_immediate_success

                authority_events.append({
                    "task_id": task.task_id,
                    "stratum": task.category,
                    "arm": arm.value,
                    "step": step_id,
                    "state_sha": receipt["state_sha"],
                    "q_argmax": updated_decision.q_argmax,
                    "q_second_best": updated_decision.q_second_best,
                    "q_gap": round(updated_decision.q_gap, 4),
                    "q_values": {k: round(v, 4) for k, v in updated_decision.q_values.items()},
                    "epsilon_set": list(updated_decision.epsilon_set),
                    "certificate_type": updated_decision.certificate_type,
                    "certificate_components": updated_decision.certificate_components,
                    "certificate_passed": updated_decision.certificate_passed,
                    "authority_mode": updated_decision.authority_mode,
                    "would_force": updated_decision.would_force,
                    "forced_action": updated_decision.forced_action,
                    "llm_proposed_action": updated_decision.llm_proposed_action,
                    "executed_action": updated_decision.executed_action,
                    "force_applied": updated_decision.force_applied,
                    "action_changed": updated_decision.action_changed,
                    "structural_state": updated_decision.structural_state,
                    # Purity receipts (Step 5)
                    "pre_generation_prompt_sha": prompt_sha,
                    "pre_generation_schema_sha": schema_sha,
                    "pre_generation_schema_actions_sha": schema_actions_sha,
                    "pre_generation_legal_actions_sha": legal_actions_sha,
                    "pre_generation_q_values_sha": q_values_sha,
                    "pre_generation_state_sha": state_sha_val,
                    "schema_actions": sorted(schema_actions),
                    "legal_actions": legal_actions,
                    # Counterfactual replay (Step 6)
                    "checkpoint_id": checkpoint.checkpoint_id,
                    "checkpoint_state_sha": checkpoint.state_sha256,
                    "forced_immediate_terminal": forced_immediate_terminal,
                    "forced_immediate_success": forced_immediate_success,
                    "llm_immediate_terminal": llm_immediate_terminal,
                    "llm_immediate_success": llm_immediate_success,
                    "actions_diverge": updated_decision.llm_proposed_action != updated_decision.forced_action,
                    "terminal_outcome": None,  # filled after trajectory
                    "trajectory_success": None,  # filled after trajectory
                    "realized_utility": None,  # filled after trajectory
                })

            # Use the executed_action from apply_authority
            action_str = executed_action
            if target_id is None and action_str in ("VERIFY",):
                # Re-derive target if needed
                valid = valid_verify_targets(runtime)
                if valid:
                    target_id = valid[0]

        # ============================================================
        # Execute action
        # ============================================================
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
        realized -= 0.5

    # Update authority events with terminal outcome
    for evt in authority_events:
        evt["terminal_outcome"] = terminal_result
        evt["trajectory_success"] = success
        evt["realized_utility"] = round(realized, 4)

    result = _make_result(
        task, arm.value, realized, success, terminal_result,
        actions_taken, authority_log, receipts,
        premature_defer, premature_answer,
        terminal_action, resource_exhaustion,
    )
    result["authority_events"] = authority_events
    return result


# ============================================================
# Manifest
# ============================================================

def compute_manifest(gguf_path: str) -> dict:
    """Compute the frozen manifest with all SHAs."""
    def sha256_file(path):
        p = Path(path)
        if not p.exists():
            return "MISSING"
        return hashlib.sha256(p.read_bytes()).hexdigest()

    import subprocess
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                     cwd=REPO_ROOT).decode().strip()

    manifest = {
        "experiment": "I3.30R3",
        "title": "Adaptive Authority Causal Isolation",
        "source_commit": commit,
        "source_commit_short": commit[:12],
        "branch": subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=REPO_ROOT).decode().strip(),

        # Q models (frozen — same as I3.30R2)
        "Q_V1_model_sha256": sha256_file("experiments/i3_5/pinned_policy/frozen_estimators/QCAUSAL_gbt.pkl"),
        "Q_V1_schema_sha256": sha256_file("experiments/i3_5/pinned_policy/frozen_estimators/feature_schema.json"),
        "Q_V3R_model_sha256": sha256_file("experiments/i3_30r/Q_V3R2_A.pkl"),
        "Q_V3R_schema_sha256": sha256_file("experiments/i3_30r/v3r2_feature_schema.json"),

        # Authority policies (frozen)
        "authority_policy_v2_sha256": sha256_file("daph/authority/policy.py"),
        "authority_policy_v3_sha256": sha256_file("daph/authority/policy_v3.py"),
        "authority_isolation_sha256": sha256_file("daph/authority/isolation.py"),

        # Topology (frozen)
        "topology_sha256": sha256_file("daph/epistemic/topology.py"),
        "v3_features_sha256": sha256_file("daph/epistemic/v3_features.py"),

        # Utility
        "utility_config_sha256": sha256_file("configs/v2b_i3_1_utility_v1.json"),

        # Runner
        "runner_sha256": sha256_file("scripts/run_i3_30r3_authority_isolation.py"),
        "i3_29_runner_sha256": sha256_file("scripts/run_i3_29_live_safety.py"),
        # Step 8: Freeze authority isolation and evaluator SHAs
        "authority_isolation_sha256": sha256_file("daph/authority/isolation.py"),
        "authority_policy_v3_sha256": sha256_file("daph/authority/policy_v3.py"),
        "evaluator_sha256": sha256_file("scripts/evaluate_i3_30r3_authority_isolation.py"),
        "checkpoint_sha256": sha256_file("daph/intervention/checkpoint.py"),
        "restore_sha256": sha256_file("daph/intervention/restore.py"),

        # Benchmark generators
        "i3_29_generator_sha256": sha256_file("hrm_adaptive_memory/executive/evidence_benchmark/i3_29_safety_generator.py"),
        "i3_30_d5_generator_sha256": sha256_file("hrm_adaptive_memory/executive/evidence_benchmark/i3_30_d5_generator.py"),

        # Frozen constants
        "authority_threshold": 5.0,
        "near_optimal_epsilon": 3.0,
        "v3_frozen_rule": "A2AD_V3_POSITIVE_CERTIFICATE",

        # Benchmark
        "benchmark_seed": 9817,

        # GGUF
        "qwen_gguf_sha256": sha256_file(gguf_path),
        "qwen_gguf_path": gguf_path,

        # Arms
        "arms": ["v1", "v3_shadow", "v3_hard"],
        "arm_count": 3,

        # Pre-registered gates
        "gates": {
            "G1": "treatment_purity",
            "G2": "authority_breaks == 0",
            "G3": "false_answer_authority == 0",
            "G4": "false_defer_authority == 0",
            "G5": "ate_authority >= 0",
            "G6": "rescues > breaks",
            "G7": "effective_answer_interventions > 0",
            "G8": "effective_defer_interventions > 0",
            "G9": "semantic_disagreements == 0",
            "G10": "reliability_errors == 0",
            "G11": "manifest_mismatches == 0",
            "G12": "complete_receipt_rate == 1.0",
        },

        # Primary comparison
        "primary_comparison": "V3_AUTH - V3_SHADOW",
        "secondary_comparison": "V3_SHADOW - V1",
    }

    # Compute benchmark hash — hash FULL task content, not just IDs
    # Step 8 fix: Hash hypotheses, evidence relationships, verification
    # states, resource budgets, and oracle paths.
    from hrm_adaptive_memory.executive.evidence_benchmark.i3_29_safety_generator import (
        generate_i3_29_benchmark, compute_benchmark_hash,
    )
    from hrm_adaptive_memory.executive.evidence_benchmark.i3_30_d5_generator import (
        generate_d5_tasks,
    )
    tasks = generate_i3_29_benchmark(seed=9817)
    d5_tasks = generate_d5_tasks(seed=9817, n_tasks=35)
    all_tasks = tasks + d5_tasks

    bench_hash = compute_benchmark_hash(tasks)
    manifest["benchmark_sha256"] = bench_hash

    # D5 hash: full content, not just task IDs
    def _task_to_hashable(t):
        """Serialize a task's full content for hashing."""
        return {
            "task_id": t.task_id,
            "category": t.category,
            "expected_terminal": t.expected_terminal.value,
            "correct_hypothesis_id": t.correct_hypothesis_id,
            "hypotheses": [
                {
                    "hypothesis_id": h.hypothesis_id,
                    "answer_action": h.answer_action.value,
                }
                for h in t.hypotheses
            ],
            "evidence_items": [
                {
                    "evidence_id": ev.evidence_id,
                    "supports": list(ev.supports),
                    "contradicts": list(ev.contradicts),
                    "verification_state": ev.verification_state.value,
                    "retrieved": ev.retrieved,
                    "verify_result": ev.verify_result,
                }
                for ev in t.evidence_items
            ],
            "oracle_resolution_path": list(t.oracle_resolution_path) if t.oracle_resolution_path else [],
        }

    manifest["d5_benchmark_sha256"] = hashlib.sha256(
        json.dumps([_task_to_hashable(t) for t in d5_tasks], sort_keys=True).encode()
    ).hexdigest()
    manifest["full_benchmark_sha256"] = hashlib.sha256(
        json.dumps([_task_to_hashable(t) for t in all_tasks], sort_keys=True).encode()
    ).hexdigest()

    from collections import Counter
    strata_counts = Counter(t.category for t in all_tasks)
    manifest["stratum_counts"] = dict(strata_counts)
    manifest["task_count"] = len(all_tasks)
    manifest["trajectory_count"] = len(all_tasks) * 3  # V1 + V3_SHADOW + V3_HARD

    return manifest


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="I3.30R3 Authority Isolation Study")
    parser.add_argument("--gguf-path", required=True)
    parser.add_argument("--output-dir", default="experiments/i3_30r3/live")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--freeze-manifest-only", action="store_true",
                        help="Only compute and save the manifest, don't run")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute and freeze manifest
    # Step 8 fix: Don't overwrite an existing frozen manifest
    manifest = compute_manifest(args.gguf_path)
    manifest_path = output_dir / "frozen_manifest.json"
    if manifest_path.exists() and not args.resume:
        # Compare new manifest against existing
        with open(manifest_path) as f:
            existing = json.load(f)
        if existing.get("source_commit") == manifest.get("source_commit"):
            print(f"Manifest already frozen at {manifest_path} (same commit)")
            manifest = existing  # Use existing frozen manifest
        else:
            print(f"WARNING: Existing manifest is from different commit.")
            print(f"  Existing: {existing.get('source_commit', '?')[:12]}")
            print(f"  New:      {manifest.get('source_commit', '?')[:12]}")
            print(f"  Overwriting with new manifest.")
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)
    else:
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
    print(f"Manifest frozen at {manifest_path}")
    print(f"  source_commit: {manifest['source_commit_short']}")
    print(f"  Q_V3R model SHA: {manifest['Q_V3R_model_sha256'][:16]}...")
    print(f"  authority_isolation SHA: {manifest['authority_isolation_sha256'][:16]}...")
    print(f"  task_count: {manifest['task_count']}")
    print(f"  trajectory_count: {manifest['trajectory_count']}")
    print(f"  arms: {manifest['arms']}")
    print(f"  gates: {len(manifest['gates'])}")

    if args.freeze_manifest_only:
        print("\nManifest frozen. Run without --freeze-manifest-only to execute.")
        return

    # Verify manifest SHAs match frozen baseline
    baseline_path = REPO_ROOT / "experiments/i3_30r3/I3_30R3_PREREGISTRATION.json"
    with open(baseline_path) as f:
        prereg = json.load(f)

    # Map manifest keys to preregistration keys
    sha_key_map = {
        "Q_V3R_model_sha256": "Q_V3R2_A_sha256",
        "Q_V3R_schema_sha256": "V3R2_feature_schema_sha256",
        "Q_V1_model_sha256": "Q_V1_sha256",
        "Q_V1_schema_sha256": "V1_feature_schema_sha256",
        "topology_sha256": "topology_sha256",
        "v3_features_sha256": "v3_features_sha256",
        "authority_policy_v2_sha256": "authority_policy_v2_sha256",
        "authority_policy_v3_sha256": "authority_policy_v3_sha256",
        "utility_config_sha256": "utility_config_sha256",
        "i3_29_generator_sha256": "i3_29_generator_sha256",
        "i3_30_d5_generator_sha256": "i3_30_d5_generator_sha256",
    }

    mismatches = []
    for manifest_key, prereg_key in sha_key_map.items():
        if prereg_key in prereg.get("frozen_artifacts", {}):
            expected = prereg["frozen_artifacts"][prereg_key]
            actual = manifest.get(manifest_key, "")
            if actual != expected:
                mismatches.append(f"  {manifest_key}: expected {expected[:16]}..., got {actual[:16]}...")

    if mismatches:
        print("\n*** MANIFEST MISMATCHES DETECTED ***")
        for m in mismatches:
            print(m)
        print("\nAborting: frozen artifacts do not match preregistration.")
        sys.exit(1)
    else:
        print("\n  All frozen artifact SHAs match preregistration.")

    # Load models
    q_v1 = QModelV1.load(
        REPO_ROOT / "experiments/i3_5/pinned_policy/frozen_estimators/QCAUSAL_gbt.pkl",
        REPO_ROOT / "experiments/i3_5/pinned_policy/frozen_estimators/feature_schema.json",
    )
    q_v3r = QModelV3R.load(
        REPO_ROOT / "experiments/i3_30r/Q_V3R2_A.pkl",
        REPO_ROOT / "experiments/i3_30r/v3r2_feature_schema.json",
    )

    # Generate benchmark
    from hrm_adaptive_memory.executive.evidence_benchmark.i3_29_safety_generator import (
        generate_i3_29_benchmark,
    )
    from hrm_adaptive_memory.executive.evidence_benchmark.i3_30_d5_generator import (
        generate_d5_tasks,
    )
    tasks = generate_i3_29_benchmark(seed=9817)
    d5_tasks = generate_d5_tasks(seed=9817, n_tasks=35)
    all_tasks = tasks + d5_tasks

    print(f"\nTasks: {len(all_tasks)} (D1-D4: {len(tasks)}, D5: {len(d5_tasks)})")
    print(f"Arms: V1, V3_SHADOW, V3_HARD")
    print(f"Total trajectories: {len(all_tasks) * 3}")

    # Load backend
    from hrm_adaptive_memory.executive.model_backend import R2DirectLlamaBackend
    backend = R2DirectLlamaBackend(
        model_path=args.gguf_path,
        n_ctx=4096,
        n_gpu_layers=-1,
    )

    # Load I3.7e for snapshot building
    import run_i3_7e_compact_governor as i3_7e

    # Load utility
    utility = MetareasoningUtility.from_file(
        REPO_ROOT / "configs" / "v2b_i3_1_utility_v1.json")

    # Trajectory output paths — one per arm
    arm_files = {}
    for arm in [ArmMode.V1, ArmMode.V3_SHADOW, ArmMode.V3_HARD]:
        traj_path = output_dir / f"trajectories_{arm.value}.jsonl"
        arm_files[arm] = traj_path

    error_path = output_dir / "errors.jsonl"
    auth_events_path = output_dir / "authority_events.jsonl"

    # Resume logic
    done = {arm: set() for arm in arm_files}
    if args.resume:
        for arm, path in arm_files.items():
            if path.exists():
                with open(path) as f:
                    for line in f:
                        r = json.loads(line)
                        done[arm].add(r["task_id"])
        print(f"  Resuming: " + ", ".join(f"{arm.value}={len(d)}" for arm, d in done.items()))

    # Open files
    traj_files = {arm: open(path, "a") for arm, path in arm_files.items()}
    error_file = open(error_path, "a")
    auth_events_file = open(auth_events_path, "a")

    total = len(all_tasks) * 3
    completed = 0

    for task in all_tasks:
        d2_pre = "_d2_" in task.task_id

        for arm in [ArmMode.V1, ArmMode.V3_SHADOW, ArmMode.V3_HARD]:
            if task.task_id in done[arm]:
                completed += 1
                continue

            try:
                result = run_trajectory(
                    task, backend, i3_7e, utility,
                    q_v1, q_v3r, arm,
                    d2_pre_verify=d2_pre,
                )

                traj_files[arm].write(json.dumps(result, default=str) + "\n")
                traj_files[arm].flush()

                # Write authority events
                for evt in result.get("authority_events", []):
                    auth_events_file.write(json.dumps(evt, default=str) + "\n")
                auth_events_file.flush()

                done[arm].add(task.task_id)
                completed += 1

                if completed % 10 == 0:
                    print(f"  [{completed}/{total}] {task.task_id} {arm.value}: "
                          f"success={result['success']} util={result['realized_utility']:.1f}")

            except Exception as e:
                error_file.write(json.dumps({
                    "task_id": task.task_id,
                    "arm": arm.value,
                    "error": str(e),
                    "timestamp": time.time(),
                }) + "\n")
                error_file.flush()
                completed += 1

    for f in traj_files.values():
        f.close()
    error_file.close()
    auth_events_file.close()

    print(f"\nDone. {completed}/{total} trajectories completed.")
    for arm, path in arm_files.items():
        print(f"  {arm.value}: {path}")
    print(f"  Authority events: {auth_events_path}")
    print(f"  Errors: {error_path}")
    print(f"  Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
