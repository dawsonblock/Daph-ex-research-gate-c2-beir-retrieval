#!/usr/bin/env python3
"""I3.30R3 Phase 8: D5 State-Level Causal Truth Audit.

For every D5 task, dumps every relevant decision state and computes:
  - canonical topology
  - legal actions
  - epistemically admissible actions
  - remaining resources
  - Q_V1 and Q_V3R2 for each legal action
  - certificate evaluation
  - terminal readiness classification

For d5_0026 specifically, reconstructs the full divergence state and
answers the diagnostic questions:
  1. What is the canonical topology at the divergence state?
  2. Is ANSWER_READY(s) actually true?
  3. What is Q*(s, ANSWER) vs Q*(s, REASON_MORE) vs Q*(s, VERIFY)?
  4. Would forced ANSWER succeed?
  5. Would advisory REASON_MORE fail?
  6. Did the certificate fail structurally, or did the Q gap fail threshold?

Usage:
  PYTHONPATH=. python3 scripts/audit_d5_state_truth.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from collections import defaultdict

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
from hrm_adaptive_memory.executive.evidence_benchmark.i3_29_safety_generator import (
    generate_i3_29_benchmark, get_budget_for_task,
)
from hrm_adaptive_memory.executive.evidence_benchmark.i3_30_d5_generator import (
    generate_d5_tasks,
)
from hrm_adaptive_memory.executive.resources import ResourceState
from hrm_adaptive_memory.executive.metareasoning_utility import MetareasoningUtility

from daph.epistemic import derive_hypothesis_topology, is_answer_ready
from daph.epistemic.topology import TerminalReadiness
from daph.intervention.checkpoint import compute_state_features
from daph.authority.policy import AUTHORITY_THRESHOLD
from daph.authority.policy_v3 import (
    StructuralStateV3, decide_authority_v3,
    answer_structural_certificate, defer_structural_certificate,
)
from daph.authority.isolation import evaluate_v3_authority

from run_i3_29_live_safety import (
    QModelV1, classify_phase_simple, compute_near_optimal_set,
    compute_progress_scores, apply_progress_tiebreak,
    compute_q_gap,
)
from run_i3_28_rep_repair import compute_structural_features
from run_i3_30r_train_v3r2 import extract_v3r2_features
from daph.epistemic.v3_features import compute_v3_features_canonical

import numpy as np
import pickle


# ============================================================
# Helpers
# ============================================================

def get_structural_state_v3(runtime) -> StructuralStateV3:
    """Build StructuralStateV3 from runtime (same as runner)."""
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


def classify_terminal_readiness(topology, structural_v3, runtime) -> str:
    """Classify a state as ANSWER_READY, DEFER_READY, CONTINUE_REQUIRED, or AMBIGUOUS."""
    if is_answer_ready(topology):
        return "ANSWER_READY"

    # Check if DEFER_READY
    # DEFER requires: no answer_ready, no continuation can resolve
    has_unverified_discriminating = topology.unverified_evidence_exists
    has_hidden = topology.hidden_evidence_count > 0
    rs = runtime.resources.as_dict()
    can_verify = rs.get("verification_calls_remaining", 0) > 0 and len(valid_verify_targets(runtime)) > 0
    can_retrieve = rs.get("retrieval_calls_remaining", 0) > 0 and has_hidden
    can_search = rs.get("search_calls_remaining", 0) > 0 and not runtime.searched

    if can_verify and has_unverified_discriminating:
        return "CONTINUE_REQUIRED"
    if can_retrieve and has_hidden:
        return "CONTINUE_REQUIRED"
    if can_search:
        return "CONTINUE_REQUIRED"

    return "DEFER_READY"


def simulate_verify(runtime, executor, target_id):
    """Simulate VERIFY and return the new runtime."""
    res = executor.execute(runtime, DecisionAction.VERIFY, target_evidence_id=target_id)
    return res.runtime, res.terminal, res.task_success


def simulate_answer(runtime, executor):
    """Simulate ANSWER and return success."""
    res = executor.execute(runtime, DecisionAction.ANSWER)
    return res.task_success


def load_q_models():
    """Load frozen Q models."""
    q_v1 = QModelV1.load(
        REPO_ROOT / "experiments/i3_5/pinned_policy/frozen_estimators/QCAUSAL_gbt.pkl",
        REPO_ROOT / "experiments/i3_5/pinned_policy/frozen_estimators/feature_schema.json",
    )
    with open(REPO_ROOT / "experiments/i3_30r/Q_V3R2_A.pkl", "rb") as f:
        v3r_model = pickle.load(f)
    with open(REPO_ROOT / "experiments/i3_30r/v3r2_feature_schema.json") as f:
        v3r_schema = json.load(f)
    return q_v1, v3r_model, v3r_schema["featurekeys"]


def predict_q_v3r(q_v3r_model, feature_keys, sf, legal_actions, structural_v3):
    """Predict Q using V3R2 model."""
    v3_struct_dict = structural_v3.as_dict()
    v3_struct_dict["n_hyp_unverified_support"] = structural_v3.n_hyp_unverified_support
    v3_struct_dict["n_hyp_unverified_contradiction"] = structural_v3.n_hyp_unverified_contradiction
    v3_struct_dict["has_competing_unverified_support"] = int(structural_v3.has_competing_unverified_support)
    X = np.array([[extract_v3r2_features(sf, a, v3_struct_dict)[k]
                   for k in feature_keys]
                  for a in legal_actions])
    preds = q_v3r_model.predict(X)
    return dict(zip(legal_actions, [float(p) for p in preds]))


def audit_state(runtime, task, executor, q_v1, q_v3r_model, v3r_keys, utility, step_label=""):
    """Audit a single decision state."""
    from r2_allowed_actions import compute_allowed_actions, ActionState, EmptyAllowedActionSet, C0

    # Build snapshot for affordances
    evidence_snapshot = build_evidence_snapshot(runtime, prior_actions=(), prior_outcomes=())

    # We need the i3_7e viability classification for action_state
    # But we can compute legal actions directly from the snapshot
    action_state = ActionState(
        t2=False,  # not needed for D5 audit
        executive_steps_remaining=runtime.resources.budget.max_executive_steps - runtime.resources.executive_steps_used,
        can_retrieve=evidence_snapshot.can_retrieve,
        can_search=evidence_snapshot.can_search,
        can_verify=evidence_snapshot.can_verify,
    )
    try:
        allowed_decision = compute_allowed_actions(action_state, C0)
        legal_actions = sorted(allowed_decision.allowed)
    except EmptyAllowedActionSet:
        legal_actions = []

    sf = compute_state_features(runtime, ())
    phase = classify_phase_simple(sf)

    # Canonical topology
    visible_ev = []
    for ev in runtime.visible_evidence:
        visible_ev.append({
            "evidence_id": ev.evidence_id,
            "supports": list(ev.supports),
            "contradicts": list(ev.contradicts),
            "verification_state": ev.verification_state,
            "temporal_status": ev.temporal_status,
            "retrieved": ev.retrieved,
        })
    hyp_ids = [h.hypothesis_id for h in task.hypotheses]
    topology = derive_hypothesis_topology(visible_ev, hyp_ids)

    # Structural state V3
    structural_v3 = get_structural_state_v3(runtime)

    # Terminal readiness
    readiness = classify_terminal_readiness(topology, structural_v3, runtime)

    # Q values
    q_v1_values = q_v1.predict_q(sf, legal_actions) if legal_actions else {}
    q_v3r_values = predict_q_v3r(q_v3r_model, v3r_keys, sf, legal_actions, structural_v3) if legal_actions else {}

    # Near-optimal sets
    no_v1, _ = compute_near_optimal_set(q_v1_values, 3.0)
    no_v3, _ = compute_near_optimal_set(q_v3r_values, 3.0)

    # Progress tiebreak
    progress_scores = {}
    for action_str in legal_actions:
        action = DecisionAction(action_str)
        target_eid = None
        if action is DecisionAction.VERIFY:
            valid = valid_verify_targets(runtime)
            if valid:
                target_eid = valid[0]
            else:
                progress_scores[action_str] = {"progress": -0.2, "state_changed": False}
                continue
        try:
            result = executor.execute(runtime, action, target_evidence_id=target_eid)
            from daph.progress.progress_rule_v1 import compute_progress
            progress = compute_progress(runtime, result, utility)
            progress_scores[action_str] = progress.as_dict()
        except Exception:
            progress_scores[action_str] = {"progress": -0.2, "state_changed": False}

    refined_v1, conf_v1 = apply_progress_tiebreak(no_v1, progress_scores, 0.05)
    refined_v3, conf_v3 = apply_progress_tiebreak(no_v3, progress_scores, 0.05)

    q_gap_v1 = compute_q_gap(q_v1_values)
    q_gap_v3r = compute_q_gap(q_v3r_values)

    # V3 authority evaluation
    v3_decision = evaluate_v3_authority(
        q_values=q_v3r_values,
        legal_actions=legal_actions,
        structural=structural_v3,
    )

    # V1 authority evaluation
    AUTHORITATIVE = frozenset({"ANSWER"})
    v1_would_force = (conf_v1 == "clear" and len(refined_v1) == 1
                      and q_gap_v1 >= AUTHORITY_THRESHOLD
                      and refined_v1[0] in AUTHORITATIVE)

    # Certificate evaluation details
    answer_cert = answer_structural_certificate(structural_v3)
    defer_cert = defer_structural_certificate(structural_v3)

    # Simulate what would happen with each action
    action_outcomes = {}
    for action_str in legal_actions:
        action = DecisionAction(action_str)
        target_eid = None
        if action is DecisionAction.VERIFY:
            valid = valid_verify_targets(runtime)
            if valid:
                target_eid = valid[0]
        try:
            sim_res = executor.execute(runtime, action, target_evidence_id=target_eid)
            if sim_res.terminal:
                action_outcomes[action_str] = {
                    "terminal": True,
                    "success": sim_res.task_success,
                    "outcome_code": sim_res.outcome_code,
                }
            else:
                action_outcomes[action_str] = {
                    "terminal": False,
                    "success": None,
                    "outcome_code": sim_res.outcome_code,
                }
        except Exception as e:
            action_outcomes[action_str] = {"terminal": True, "success": False, "error": str(e)}

    return {
        "step_label": step_label,
        "legal_actions": legal_actions,
        "phase": phase,
        "state_features": {k: v for k, v in sf.items() if not isinstance(v, (list, tuple))},
        "topology": {
            "n_hypotheses": len(hyp_ids),
            "hypothesis_states": {k: v.value for k, v in topology.hypothesis_states.items()},
            "n_supported": topology.n_viable_hypotheses,
            "n_contradicted": topology.n_eliminated_hypotheses,
            "n_untested": topology.n_untested_hypotheses,
            "n_viable": topology.n_viable_hypotheses + topology.n_untested_hypotheses,
            "unique_supported_hypothesis": topology.unique_supported_hypothesis,
            "has_verified_unresolved_competition": topology.has_verified_unresolved_competition,
            "has_unique_verified_supported": topology.has_unique_verified_supported,
            "unverified_evidence_exists": topology.unverified_evidence_exists,
            "hidden_evidence_count": topology.hidden_evidence_count,
            "verification_complete": topology.verification_complete,
            "is_answer_ready": is_answer_ready(topology),
        },
        "structural_v3": structural_v3.as_dict(),
        "terminal_readiness": readiness,
        "q_v1": {k: round(v, 4) for k, v in q_v1_values.items()},
        "q_v3r": {k: round(v, 4) for k, v in q_v3r_values.items()},
        "q_gap_v1": round(q_gap_v1, 4),
        "q_gap_v3r": round(q_gap_v3r, 4),
        "near_optimal_v1": no_v1,
        "near_optimal_v3": no_v3,
        "refined_v1": refined_v1,
        "refined_v3": refined_v3,
        "confidence_v1": conf_v1,
        "confidence_v3": conf_v3,
        "v1_would_force": v1_would_force,
        "v3_would_force": v3_decision.would_force,
        "v3_forced_action": v3_decision.forced_action,
        "v3_authority_mode": v3_decision.authority_mode,
        "v3_certificate_passed": v3_decision.certificate_passed,
        "v3_certificate_type": v3_decision.certificate_type,
        "answer_certificate_detail": {
            "has_unique_verified_supported_hypothesis": structural_v3.has_unique_verified_supported_hypothesis,
            "verified_hyp_action_is_answer": structural_v3.verified_hyp_action_is_answer,
            "has_verified_unresolved_competition": structural_v3.has_verified_unresolved_competition,
            "answer_cert_passes": answer_cert,
        },
        "defer_certificate_detail": {
            "defer_cert_passes": defer_cert,
        },
        "action_outcomes": action_outcomes,
        "resources": {
            "executive_steps_used": runtime.resources.executive_steps_used,
            "executive_steps_remaining": runtime.resources.budget.max_executive_steps - runtime.resources.executive_steps_used,
            "verification_calls_used": runtime.resources.verification_calls_used,
            "verification_calls_remaining": runtime.resources.budget.max_verification_calls - runtime.resources.verification_calls_used,
            "retrieval_calls_used": runtime.resources.retrieval_calls_used,
            "retrieval_calls_remaining": runtime.resources.budget.max_retrieval_calls - runtime.resources.retrieval_calls_used,
        },
        "evidence_state": [
            {
                "evidence_id": ev.evidence_id,
                "verification_state": ev.verification_state.value,
                "supports": list(ev.supports),
                "contradicts": list(ev.contradicts),
                "retrieved": ev.retrieved,
                "verify_result": ev.verify_result,
            }
            for ev in runtime.visible_evidence
        ],
        "valid_verify_targets": list(valid_verify_targets(runtime)),
    }


def audit_d5_task(task, executor, q_v1, q_v3r_model, v3r_keys, utility):
    """Audit all decision states in a D5 task's oracle path."""
    budget = get_budget_for_task(task)
    resources = ResourceState(budget=budget)
    runtime = initial_evidence_runtime(task, resources)

    states = []

    # State 0: initial state
    states.append(audit_state(runtime, task, executor, q_v1, q_v3r_model, v3r_keys, utility, "s0_initial"))

    # State 1: after VERIFY(E3) — the oracle's first action
    valid = valid_verify_targets(runtime)
    if valid:
        # Find E3 (the discriminator)
        e3_target = None
        for eid in valid:
            for ev in runtime.visible_evidence:
                if ev.evidence_id == eid and ev.contradicts:
                    e3_target = eid
                    break
            if e3_target:
                break
        if not e3_target:
            e3_target = valid[0]

        runtime_after_verify, terminal, success = simulate_verify(runtime, executor, e3_target)
        if not terminal:
            states.append(audit_state(runtime_after_verify, task, executor, q_v1, q_v3r_model, v3r_keys, utility, "s1_after_verify_discriminator"))

            # State 2: what if we ANSWER now?
            answer_success = simulate_answer(runtime_after_verify, executor)
            states[-1]["simulate_answer_after_verify"] = {
                "would_succeed": answer_success,
            }

    return states


def main():
    output_dir = REPO_ROOT / "experiments/i3_30r3/d5_state_truth"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    q_v1, q_v3r_model, v3r_keys = load_q_models()
    executor = EvidenceExecutor()
    utility = MetareasoningUtility.from_file(
        REPO_ROOT / "configs" / "v2b_i3_1_utility_v1.json")

    # Generate D5 tasks
    d5_tasks = generate_d5_tasks(seed=9817, n_tasks=35)
    print(f"Generated {len(d5_tasks)} D5 tasks")

    # Audit all D5 tasks
    all_audits = {}
    readiness_counts = defaultdict(int)

    for task in d5_tasks:
        states = audit_d5_task(task, executor, q_v1, q_v3r_model, v3r_keys, utility)
        all_audits[task.task_id] = states

        # Count initial state readiness
        if states:
            readiness = states[0]["terminal_readiness"]
            readiness_counts[readiness] += 1

    # Print summary
    print("\n" + "=" * 70)
    print("D5 INITIAL STATE TERMINAL READINESS CLASSIFICATION")
    print("=" * 70)
    for readiness, count in sorted(readiness_counts.items()):
        print(f"  {readiness:<25} {count:>3} tasks")

    # Print s0 vs s1 comparison for all tasks
    print("\n" + "=" * 70)
    print("D5 STATE TRANSITION: s0 (initial) → s1 (after VERIFY discriminator)")
    print("=" * 70)
    print(f"  {'Task':<20} {'s0 readiness':<25} {'s1 readiness':<25} {'s1 answer_ready':<15}")
    for tid in sorted(all_audits.keys()):
        states = all_audits[tid]
        s0 = states[0] if len(states) > 0 else {}
        s1 = states[1] if len(states) > 1 else {}
        s0_r = s0.get("terminal_readiness", "?")
        s1_r = s1.get("terminal_readiness", "?")
        s1_ar = s1.get("topology", {}).get("is_answer_ready", "?")
        print(f"  {tid:<20} {s0_r:<25} {s1_r:<25} {str(s1_ar):<15}")

    # ============================================================
    # Deep dive: d5_0026
    # ============================================================
    print("\n" + "=" * 70)
    print("DEEP DIVE: d5_0026 — DIVERGENCE STATE ANALYSIS")
    print("=" * 70)

    d5_0026_states = all_audits.get("i3_29_d5_0026", [])
    for i, state in enumerate(d5_0026_states):
        print(f"\n--- State {i}: {state.get('step_label', '')} ---")
        print(f"  Terminal readiness: {state['terminal_readiness']}")
        print(f"  is_answer_ready: {state['topology']['is_answer_ready']}")
        print(f"  unique_supported_hyp: {state['topology']['unique_supported_hypothesis']}")
        print(f"  has_verified_unresolved_competition: {state['topology']['has_verified_unresolved_competition']}")
        print(f"  unverified_evidence_exists: {state['topology']['unverified_evidence_exists']}")
        print(f"  valid_verify_targets: {state['valid_verify_targets']}")
        print(f"  Legal actions: {state['legal_actions']}")
        print(f"  Q_V1: {state['q_v1']}")
        print(f"  Q_V3R: {state['q_v3r']}")
        print(f"  Q gap V1: {state['q_gap_v1']}")
        print(f"  Q gap V3R: {state['q_gap_v3r']}")
        print(f"  Near-optimal V1: {state['near_optimal_v1']}")
        print(f"  Near-optimal V3: {state['near_optimal_v3']}")
        print(f"  Refined V1: {state['refined_v1']} (conf={state['confidence_v1']})")
        print(f"  Refined V3: {state['refined_v3']} (conf={state['confidence_v3']})")
        print(f"  V1 would force: {state['v1_would_force']}")
        print(f"  V3 would force: {state['v3_would_force']}")
        print(f"  V3 forced action: {state['v3_forced_action']}")
        print(f"  V3 authority mode: {state['v3_authority_mode']}")
        print(f"  V3 certificate passed: {state['v3_certificate_passed']}")
        print(f"  V3 certificate type: {state['v3_certificate_type']}")
        print(f"  Answer cert detail: {state['answer_certificate_detail']}")
        print(f"  Action outcomes: {json.dumps(state['action_outcomes'], indent=2)}")

        if "simulate_answer_after_verify" in state:
            print(f"  ANSWER after VERIFY would succeed: {state['simulate_answer_after_verify']['would_succeed']}")

    # ============================================================
    # Answer the diagnostic questions for d5_0026
    # ============================================================
    print("\n" + "=" * 70)
    print("DIAGNOSTIC ANSWERS FOR d5_0026")
    print("=" * 70)

    if len(d5_0026_states) >= 2:
        s0 = d5_0026_states[0]
        s1 = d5_0026_states[1]  # after VERIFY

        print(f"\n  Q1: Canonical topology at divergence state (s1, after VERIFY)?")
        print(f"      n_supported: {s1['topology']['n_supported']}")
        print(f"      n_contradicted: {s1['topology']['n_contradicted']}")
        print(f"      unique_supported_hyp: {s1['topology']['unique_supported_hypothesis']}")
        print(f"      has_verified_unresolved_competition: {s1['topology']['has_verified_unresolved_competition']}")

        print(f"\n  Q2: Is ANSWER_READY(s1) actually true?")
        print(f"      is_answer_ready: {s1['topology']['is_answer_ready']}")
        print(f"      terminal_readiness: {s1['terminal_readiness']}")

        print(f"\n  Q3: Q values at s1?")
        print(f"      Q_V1: {s1['q_v1']}")
        print(f"      Q_V3R: {s1['q_v3r']}")

        print(f"\n  Q4: Would forced ANSWER succeed at s1?")
        answer_outcome = s1['action_outcomes'].get('ANSWER', {})
        print(f"      ANSWER outcome: {answer_outcome}")
        if "simulate_answer_after_verify" in s1:
            print(f"      Simulated ANSWER success: {s1['simulate_answer_after_verify']['would_succeed']}")

        print(f"\n  Q5: Would advisory REASON_MORE fail at s1?")
        reason_outcome = s1['action_outcomes'].get('REASON_MORE', {})
        print(f"      REASON_MORE outcome: {reason_outcome}")

        print(f"\n  Q6: Did the certificate fail structurally, or did the Q gap fail threshold?")
        print(f"      V3 certificate passed: {s1['v3_certificate_passed']}")
        print(f"      V3 certificate type: {s1['v3_certificate_type']}")
        print(f"      V3 would force: {s1['v3_would_force']}")
        print(f"      V3 Q gap: {s1['q_gap_v3r']}")
        print(f"      Threshold: {AUTHORITY_THRESHOLD}")
        print(f"      Answer cert detail:")
        for k, v in s1['answer_certificate_detail'].items():
            print(f"        {k}: {v}")

        # Diagnosis
        print(f"\n  DIAGNOSIS:")
        if s1['topology']['is_answer_ready'] and s1['v3_certificate_passed']:
            if s1['q_gap_v3r'] < AUTHORITY_THRESHOLD:
                print(f"      → Q-GAP THRESHOLD FAILURE: certificate passes but Q gap {s1['q_gap_v3r']} < {AUTHORITY_THRESHOLD}")
            elif not s1['v3_would_force']:
                print(f"      → NOT SOLE NEAR-OPTIMAL: certificate passes but V3 not sole near-optimal")
            else:
                print(f"      → UNKNOWN: certificate passes, Q gap sufficient, but force not applied")
        elif s1['topology']['is_answer_ready'] and not s1['v3_certificate_passed']:
            print(f"      → CERTIFICATE RECALL BUG: state is ANSWER_READY but certificate fails")
            print(f"        has_unique_verified_supported_hypothesis: {s1['answer_certificate_detail']['has_unique_verified_supported_hypothesis']}")
            print(f"        verified_hyp_action_is_answer: {s1['answer_certificate_detail']['verified_hyp_action_is_answer']}")
            print(f"        has_verified_unresolved_competition: {s1['answer_certificate_detail']['has_verified_unresolved_competition']}")
        elif not s1['topology']['is_answer_ready']:
            print(f"      → STATE NOT ANSWER_READY: V1's forced ANSWER may exploit the evaluator")
            print(f"        V1 succeeds by forcing ANSWER on a non-answer-ready state")

    # Write full audit to file
    audit_path = output_dir / "d5_state_audit.json"
    with open(audit_path, "w") as f:
        json.dump(all_audits, f, indent=2, default=str)
    print(f"\nFull audit written to: {audit_path}")

    # Write summary
    summary_path = output_dir / "d5_readiness_summary.json"
    summary = {
        "task_count": len(d5_tasks),
        "initial_readiness_counts": dict(readiness_counts),
        "d5_0026_diagnosis": {
            "s0_readiness": d5_0026_states[0]["terminal_readiness"] if d5_0026_states else None,
            "s1_readiness": d5_0026_states[1]["terminal_readiness"] if len(d5_0026_states) > 1 else None,
            "s1_is_answer_ready": d5_0026_states[1]["topology"]["is_answer_ready"] if len(d5_0026_states) > 1 else None,
            "s1_v3_certificate_passed": d5_0026_states[1]["v3_certificate_passed"] if len(d5_0026_states) > 1 else None,
            "s1_v3_would_force": d5_0026_states[1]["v3_would_force"] if len(d5_0026_states) > 1 else None,
            "s1_q_gap_v3r": d5_0026_states[1]["q_gap_v3r"] if len(d5_0026_states) > 1 else None,
        },
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
