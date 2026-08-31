#!/usr/bin/env python3
"""I3.30R3: CERT-only and Q-only ablations on the OOD pool.

Tests where the +63 utility actually comes from:
  SHADOW:    no authority (baseline)
  Q-only:    Q gap >= 5.0 + sole near-optimal, NO certificate required
  CERT-only: certificate passes, NO Q gap/sole near-optimal required
  Q+CERT:    full V3R2 authority (both Q and CERT required)

If CERT-only ≈ Q+CERT, the certificate is the decisive mechanism.
If Q-only << Q+CERT, Q alone is insufficient without structural verification.

Usage:
    PYTHONPATH=. python3 scripts/run_ablations.py \\
        --gguf-path /path/to/Qwen2.5-7B-Instruct-Q4_K_M.gguf \\
        --output-dir experiments/i3_30r3/ablations
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from enum import Enum
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


# ============================================================
# Ablation arm modes
# ============================================================

class AblationArm(str, Enum):
    SHADOW = "shadow"
    Q_ONLY = "q_only"
    CERT_ONLY = "cert_only"
    Q_CERT = "q_cert"  # full V3R2 authority (same as V3_HARD)


# ============================================================
# Ablation authority decisions
# ============================================================

from daph.authority.policy import AuthorityMode, AuthorityDecision, AUTHORITY_THRESHOLD, I2_EPSILON_Q
from daph.authority.policy_v3 import (
    StructuralStateV3,
    answer_structural_certificate,
    defer_structural_certificate,
)


def decide_q_only(q_values, legal_actions):
    """Q-only: force when Q gap >= 5.0 and sole near-optimal, NO certificate."""
    legal_q = {a: q_values[a] for a in legal_actions if a in q_values}
    if not legal_q:
        return AuthorityDecision(mode=AuthorityMode.ADVISORY, action=None, reason_codes=["NO_LEGAL_Q"])

    sorted_q = sorted(legal_q.items(), key=lambda x: -x[1])
    q_argmax = sorted_q[0][0]
    q_max = sorted_q[0][1]
    q_second = sorted_q[1][1] if len(sorted_q) > 1 else q_max
    q_gap = q_max - q_second
    near_optimal = [a for a, q in legal_q.items() if q >= q_max - I2_EPSILON_Q]

    if q_argmax not in ("ANSWER", "DEFER"):
        return AuthorityDecision(mode=AuthorityMode.ADVISORY, action=None, reason_codes=["ARGMAX_NOT_TERMINAL"])

    if q_gap < AUTHORITY_THRESHOLD:
        return AuthorityDecision(mode=AuthorityMode.ADVISORY, action=None, reason_codes=["GAP_TOO_SMALL"], q_gap=q_gap)

    if len(near_optimal) != 1 or near_optimal[0] != q_argmax:
        return AuthorityDecision(mode=AuthorityMode.ADVISORY, action=None, reason_codes=["NOT_SOLE_NEAR_OPT"], q_gap=q_gap)

    # Q-only: NO certificate check
    if q_argmax == "ANSWER":
        return AuthorityDecision(mode=AuthorityMode.HARD_ANSWER, action="ANSWER",
                                 reason_codes=["Q_ONLY_ANSWER"], q_gap=q_gap, q_argmax=q_argmax)
    else:
        return AuthorityDecision(mode=AuthorityMode.HARD_DEFER, action="DEFER",
                                 reason_codes=["Q_ONLY_DEFER"], q_gap=q_gap, q_argmax=q_argmax)


def decide_cert_only(structural, legal_actions):
    """CERT-only: force when certificate passes, NO Q gap/sole near-optimal required."""
    # Check ANSWER certificate
    if "ANSWER" in legal_actions and answer_structural_certificate(structural):
        return AuthorityDecision(mode=AuthorityMode.HARD_ANSWER, action="ANSWER",
                                 reason_codes=["CERT_ONLY_ANSWER"])
    # Check DEFER certificate
    if "DEFER" in legal_actions and defer_structural_certificate(structural):
        return AuthorityDecision(mode=AuthorityMode.HARD_DEFER, action="DEFER",
                                 reason_codes=["CERT_ONLY_DEFER"])
    return AuthorityDecision(mode=AuthorityMode.ADVISORY, action=None, reason_codes=["NO_CERTIFICATE"])


# ============================================================
# Ablation trajectory runner (adapted from run_i3_30r3_authority_isolation)
# ============================================================

def run_ablation_trajectory(task, backend, i3_7e, utility, q_v3r, arm: AblationArm):
    """Run a single trajectory for an ablation arm."""
    from r2_schema import build_action_schema, schema_sha256
    from r2_allowed_actions import compute_allowed_actions, ActionState, EmptyAllowedActionSet, C0
    from hrm_adaptive_memory.executive.evidence_benchmark.i3_29_safety_generator import get_budget_for_task
    from daph.intervention.checkpoint import compute_state_features
    from run_i3_29_live_safety import classify_phase_simple
    from hrm_adaptive_memory.executive.evidence_benchmark.schema import initial_evidence_runtime
    from hrm_adaptive_memory.executive.evidence_benchmark.executor import EvidenceExecutor
    from hrm_adaptive_memory.executive.resources import ResourceState
    from scripts.run_i3_30r3_authority_isolation import (
        compute_near_optimal_set, compute_progress_scores, apply_progress_tiebreak,
        compute_q_gap, decode_output_strict, get_structural_state_v3,
        build_evidence_snapshot, valid_verify_targets,
    )

    budget = get_budget_for_task(task)
    max_steps = budget.max_executive_steps
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
    actions_taken = []
    authority_events = []
    system_prompt = i3_7e.MDSG_STATE_WITH_AFFORDANCES_SYSTEM_PROMPT

    for step_id in range(max_steps):
        evidence_snapshot = build_evidence_snapshot(
            runtime, prior_actions=tuple(prior_actions), prior_outcomes=tuple(prior_outcomes))

        viability = i3_7e._classify_from_snapshot(evidence_snapshot)
        eliminated = [h_id for h_id, info in viability.items() if info["status"] == "ELIMINATED"]
        n_hypotheses = len(task.hypotheses)
        t2 = (len(eliminated) == n_hypotheses and n_hypotheses > 0)

        action_state = ActionState(
            t2=t2, executive_steps_remaining=max_steps - step_id,
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

        # Q prediction (V3R2 model)
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

        # Ablation authority decision
        schema_actions = allowed_decision.allowed
        forced_action = None
        would_force = False
        cert_type = "NONE"

        if arm == AblationArm.SHADOW:
            pass  # no authority

        elif arm == AblationArm.Q_CERT:
            # Full V3R2 authority
            from daph.authority.policy_v3 import decide_authority_v3
            decision = decide_authority_v3(q_values=q_values, legal_actions=legal_actions, structural=structural_v3)
            if decision.mode in (AuthorityMode.HARD_ANSWER, AuthorityMode.HARD_DEFER):
                would_force = True
                forced_action = decision.action
                if decision.mode == AuthorityMode.HARD_ANSWER:
                    cert_type = "unique_verified_support_answer"
                else:
                    cert_type = "defer_certificate"

        elif arm == AblationArm.Q_ONLY:
            decision = decide_q_only(q_values, legal_actions)
            if decision.mode in (AuthorityMode.HARD_ANSWER, AuthorityMode.HARD_DEFER):
                would_force = True
                forced_action = decision.action
                cert_type = "q_only_no_cert"

        elif arm == AblationArm.CERT_ONLY:
            decision = decide_cert_only(structural_v3, legal_actions)
            if decision.mode in (AuthorityMode.HARD_ANSWER, AuthorityMode.HARD_DEFER):
                would_force = True
                forced_action = decision.action
                if decision.action == "ANSWER":
                    cert_type = "cert_only_answer"
                else:
                    cert_type = "cert_only_defer"

        # Build packet and call LLM
        extra_fields = {
            "near_optimal_actions": refined_set,
            "lower_value_actions": lower_value,
            "guidance_confidence": confidence,
            "epistemic_phase": phase,
        }

        packet = i3_7e.build_mdsg_state_with_affordances_packet(evidence_snapshot)
        schema = build_action_schema(schema_actions)
        packet_dict = json.loads(i3_7e.evidence_packet_json(packet))
        packet_dict["executive_guidance"] = extra_fields
        user_prompt = json.dumps(packet_dict, indent=2)

        try:
            call_result = backend.generate(
                system_prompt=system_prompt, user_prompt=user_prompt,
                temperature=0.0, max_tokens=256, allowed_actions=schema_actions)
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

        # Apply authority
        executed_action = action_str
        force_applied = False
        if would_force and forced_action:
            executed_action = forced_action
            force_applied = True

        action_changed = (executed_action != action_str)

        # Record authority event
        if would_force:
            authority_events.append({
                "task_id": task.task_id,
                "arm": arm.value,
                "step": step_id,
                "q_argmax": max(q_values, key=q_values.get) if q_values else "",
                "q_gap": round(q_gap, 4),
                "q_values": {k: round(v, 4) for k, v in q_values.items()},
                "certificate_type": cert_type,
                "would_force": would_force,
                "forced_action": forced_action,
                "llm_proposed": action_str,
                "executed": executed_action,
                "force_applied": force_applied,
                "action_changed": action_changed,
            })

        # Execute
        from hrm_adaptive_memory.cognitive_control.core import DecisionAction
        action_enum = DecisionAction(executed_action)

        resources_before = runtime.resources
        if executed_action in ("ANSWER", "DEFER"):
            res = executor.execute(runtime, action_enum)
            runtime = res.runtime
            resources_after = runtime.resources
            step_cost = utility.action_cost(resources_before, resources_after)
            realized -= step_cost
            success = bool(res.task_success)
            terminal = res.terminal
            terminal_action = executed_action
            tr = utility.terminal_reward(action_enum, success)
            realized += tr
            terminal_result = "SUCCESS" if success else "TERMINAL_WRONG"
            actions_taken.append(executed_action)
            prior_actions.append(executed_action)
            prior_outcomes.append("terminal")
            break
        else:
            kwargs = {}
            if executed_action == "VERIFY" and target_id:
                kwargs["target_evidence_id"] = target_id
            elif executed_action == "SEARCH" and target_id:
                kwargs["target_hypothesis_id"] = target_id
            elif executed_action == "VERIFY" and not target_id:
                valid = valid_verify_targets(runtime)
                if valid:
                    kwargs["target_evidence_id"] = valid[0]

            res = executor.execute(runtime, action_enum, **kwargs)
            runtime = res.runtime
            resources_after = runtime.resources
            step_cost = utility.action_cost(resources_before, resources_after)
            realized -= step_cost
            actions_taken.append(executed_action)
            prior_actions.append(executed_action)
            prior_outcomes.append("ok" if not res.terminal else "terminal")

            if res.terminal:
                terminal = True
                success = bool(res.task_success)
                terminal_action = executed_action
                tr = utility.terminal_reward(action_enum, success)
                realized += tr
                terminal_result = "SUCCESS" if success else "TERMINAL_WRONG"
                break

    if not terminal:
        realized -= 0.5

    result = {
        "task_id": task.task_id,
        "arm": arm.value,
        "realized_utility": realized,
        "success": success,
        "terminal_action": terminal_action,
        "terminal_result": terminal_result,
        "actions_taken": actions_taken,
        "n_steps": len(actions_taken),
        "authority_events": authority_events,
        "hard_force_count": sum(1 for e in authority_events if e.get("force_applied")),
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="CERT-only and Q-only ablations on OOD pool")
    parser.add_argument("--gguf-path", required=True)
    parser.add_argument("--output-dir", default="experiments/i3_30r3/ablations")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load OOD pool
    ood_pool_path = REPO_ROOT / "experiments/i3_30r3/structural_ood/ood_pool.json"
    with open(ood_pool_path) as f:
        ood_pool = json.load(f)
    print(f"OOD pool: {len(ood_pool)} tasks")

    # Rebuild tasks
    from scripts.build_structural_ood_pool import (
        OOD_DOMAIN_TEMPLATES, generate_ood_candidate, compute_task_signature,
    )
    from hrm_adaptive_memory.executive.evidence_benchmark.i3_29_safety_generator import _BUDGET_OVERRIDES
    from hrm_adaptive_memory.executive.resources import ResourceBudget

    dev_sigs_path = REPO_ROOT / "experiments/i3_30r3/structural_ood/development_signatures.json"
    with open(dev_sigs_path) as f:
        dev_signatures = set(json.load(f)["signatures"])

    tasks = []
    for template in OOD_DOMAIN_TEMPLATES:
        for i in range(20):
            candidate = generate_ood_candidate(template, i)
            sig = compute_task_signature(candidate)
            if sig and sig not in dev_signatures:
                tasks.append(candidate)

    # Register budgets
    for task in tasks:
        parts = task.budget_profile.split("_")
        _BUDGET_OVERRIDES[task.task_id] = ResourceBudget(
            max_executive_steps=int(parts[1]), max_reasoning_tokens=256,
            max_retrieval_calls=0, max_verification_calls=int(parts[2]),
            max_search_calls=int(parts[3]), max_elapsed_ms=10000)

    print(f"Rebuilt {len(tasks)} OOD tasks")

    # Load models
    from run_i3_30r3_authority_isolation import QModelV3R
    from hrm_adaptive_memory.executive.model_backend import R2DirectLlamaBackend
    from hrm_adaptive_memory.executive.metareasoning_utility import MetareasoningUtility
    import run_i3_7e_compact_governor as i3_7e

    q_v3r = QModelV3R.load(
        REPO_ROOT / "experiments/i3_30r/Q_V3R2_A.pkl",
        REPO_ROOT / "experiments/i3_30r/v3r2_feature_schema.json",
    )
    backend = R2DirectLlamaBackend(model_path=args.gguf_path, n_ctx=4096, n_gpu_layers=-1)
    utility = MetareasoningUtility.from_file(REPO_ROOT / "configs" / "v2b_i3_1_utility_v1.json")

    # Run all 4 arms
    arms = [AblationArm.SHADOW, AblationArm.Q_ONLY, AblationArm.CERT_ONLY, AblationArm.Q_CERT]
    total = len(tasks) * len(arms)
    completed = 0

    for arm in arms:
        arm_file = output_dir / f"trajectories_{arm.value}.jsonl"
        done_ids = set()
        if args.resume and arm_file.exists():
            with open(arm_file) as f:
                for line in f:
                    done_ids.add(json.loads(line)["task_id"])

        with open(arm_file, "a") as f:
            for task in tasks:
                if task.task_id in done_ids:
                    completed += 1
                    continue
                try:
                    result = run_ablation_trajectory(task, backend, i3_7e, utility, q_v3r, arm)
                    f.write(json.dumps(result, default=str) + "\n")
                    f.flush()
                    completed += 1
                    if completed % 10 == 0:
                        print(f"  [{completed}/{total}] {arm.value} {task.task_id}: "
                              f"success={result.get('success')} util={result.get('realized_utility',0):.1f}")
                except Exception as e:
                    print(f"  ERROR [{completed}/{total}] {arm.value} {task.task_id}: {e}")
                    completed += 1

    print(f"\nAblations complete. {completed}/{total} trajectories.")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
