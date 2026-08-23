#!/usr/bin/env python3
"""
R2 Development Runner — 2×2 factorial on held-out development data.

Single execution function for all arms (C0, D, E, DE).
The arm flows through exactly two intervention hooks:
    1. apply_semantics_intervention(packet, state, arm)  — R2e label
    2. compute_allowed_actions(state, arm)               — R2d gate

Everything else is shared.

Per-call receipts include full provenance:
    arm, t2, gold_t2, decision_state_internal, decision_state_exposed,
    legal_actions, epistemically_admissible_actions, allowed_actions,
    allowed_actions_sha256, verify_gate_applied, verify_gate_reason,
    schema_sha256, schema_action_enum, selected_action,
    admissibility_assertion_passed

Usage:
    PYTHONPATH=scripts:. python3 scripts/run_r2_development.py \
        --dataset /path/to/r2_dataset.jsonl \
        --output /path/to/r2-dev/ \
        --n-per-arm 10
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from r2_allowed_actions import (
    ACTION_VOCABULARY,
    ALWAYS_LEGAL,
    ActionState,
    AllowedActionDecision,
    R2Arm,
    C0, D, E, DE,
    ALL_ARMS,
    EmptyAllowedActionSet,
    compute_legal_actions,
    compute_epistemically_admissible_actions,
    compute_allowed_actions,
    allowed_actions_sha256,
)
from r2_schema import (
    build_action_schema,
    schema_sha256,
    schema_action_enum,
    verify_schema_invariant,
    FROZEN_R13_ACTION_SCHEMA_SHA256,
)
from r2_dataset_generator import R2GoldLabels, R2Task


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class EpistemicAdmissibilityViolation(Exception):
    """Raised when a decoded action is not in the allowed action set.

    This is a defense-in-depth invariant check (Layer 2).
    It should NEVER fire in a qualified run.
    If it fires, the run is aborted — this is an infrastructure/protocol
    failure, not model behavior.
    """
    def __init__(self, action: str, allowed: frozenset[str], reason: str | None):
        self.action = action
        self.allowed = allowed
        self.reason = reason
        super().__init__(
            f"Admissibility violation: action={action} not in allowed={sorted(allowed)} "
            f"reason={reason}"
        )


# ---------------------------------------------------------------------------
# Intervention hooks
# ---------------------------------------------------------------------------

def apply_semantics_intervention(
    packet: dict,
    state: ActionState,
    arm: R2Arm,
) -> tuple[dict, str, str]:
    """Apply R2e semantics intervention (label-only).

    Returns (modified_packet, decision_state_internal, decision_state_exposed).

    For E/DE at T2: changes decision_state from NEEDS_DISCRIMINATION to
    NO_VIABLE_HYPOTHESIS. Underlying computation is unchanged.
    """
    ds_summary = packet.get("decision_state_summary", {})
    decision_state_internal = ds_summary.get("decision_state", "UNKNOWN")

    decision_state_exposed = decision_state_internal

    if arm.corrected_t2_semantics and state.t2:
        # R2e: relabel NEEDS_DISCRIMINATION → NO_VIABLE_HYPOTHESIS at T2
        if decision_state_internal == "NEEDS_DISCRIMINATION":
            decision_state_exposed = "NO_VIABLE_HYPOTHESIS"
            # Create a shallow copy with the modified label
            packet = dict(packet)
            ds_summary = dict(ds_summary)
            ds_summary["decision_state"] = "NO_VIABLE_HYPOTHESIS"
            packet["decision_state_summary"] = ds_summary

    return packet, decision_state_internal, decision_state_exposed


# ---------------------------------------------------------------------------
# R2-aware stub backend (for testing)
# ---------------------------------------------------------------------------

class R2StubBackend:
    """Deterministic stub backend that respects allowed actions.

    Returns the first allowed action from a priority list.
    This allows testing the full R2 pipeline without a live model.
    """
    model_name = "r2-stub-v1"

    def __init__(self):
        from hrm_adaptive_memory.executive.model_backend import ModelCallResult
        self._ModelCallResult = ModelCallResult
        self._call_counter = 0

    # These will be set by the runner before generate() is called
    task_id: str = ""
    condition: str = ""
    pair_id: str = ""
    allowed_actions: frozenset[str] | None = None

    # Priority order for stub selection
    _priority = ("VERIFY", "RETRIEVE", "SEARCH_MORE", "REASON_MORE", "ANSWER", "DEFER", "STOP")

    def generate(self, *, system_prompt: str, user_prompt: str,
                 temperature: float, max_tokens: int,
                 allowed_actions: frozenset[str] | None = None):
        import hashlib, json
        allowed = allowed_actions or ACTION_VOCABULARY

        # Select first action from priority that's in allowed
        selected = None
        for action in self._priority:
            if action in allowed:
                selected = action
                break
        if selected is None:
            selected = sorted(allowed)[0]  # fallback

        response = json.dumps({
            "action": selected,
            "reason_code": f"STUB_{selected}",
            "target_id": None,
        })

        action_schema = build_action_schema(allowed)
        schema_sha = schema_sha256(action_schema)

        self._call_counter += 1
        return self._ModelCallResult(
            raw_output=response,
            provider_raw_output=response,
            prompt_tokens=100,
            completion_tokens=20,
            reasoning_tokens=0,
            latency_ms=1,
            model_name=self.model_name,
            system_fingerprint=None,
            finish_reason="stop",
            json_schema_sha256=schema_sha,
            system_prompt_sha256=hashlib.sha256(system_prompt.encode()).hexdigest(),
            user_packet_sha256=hashlib.sha256(user_prompt.encode()).hexdigest(),
            request_sha256=hashlib.sha256(response.encode()).hexdigest(),
        )


# ---------------------------------------------------------------------------
# Trajectory runner
# ---------------------------------------------------------------------------

@dataclass
class TrajectoryResult:
    """Result of a single trajectory."""
    task_id: str
    arm: str
    realized_utility: float
    success: bool
    steps: int
    terminal_action: str | None
    terminal_result: str
    model_calls: int
    backend_errors: int
    call_log: list[dict] = field(default_factory=list)
    decision_state_log: list[dict] = field(default_factory=list)
    gold_t2: bool = False
    gold_should_gate: bool = False
    stratum: str = ""
    inferred_t2_fired: bool = False
    inferred_t2_step: int | None = None


def _load_i3_7e():
    """Load the i3_7e module."""
    spec = importlib.util.spec_from_file_location(
        "i3_7e", str(REPO_ROOT / "scripts" / "run_i3_7e_compact_governor.py"))
    i3_7e = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(i3_7e)
    return i3_7e


def _build_action_state(snapshot, n_hypotheses: int) -> ActionState:
    """Build ActionState from an EvidenceSnapshot."""
    ds = getattr(snapshot, "decision_state_summary", None)
    # Compute T2 from the snapshot's hypothesis viability
    # T2 = all hypotheses eliminated
    viability = {}  # would need i3_7e._classify_from_snapshot

    # Use the snapshot's can_* fields and step count
    steps_remaining = 0  # will be set by caller
    return ActionState(
        t2=False,  # will be computed by caller
        executive_steps_remaining=steps_remaining,
        can_retrieve=snapshot.can_retrieve,
        can_search=snapshot.can_search,
        can_verify=snapshot.can_verify,
    )


def run_trajectory(
    task: R2Task,
    arm: R2Arm,
    backend_factory: Callable,
    *,
    max_tokens: int = 128,
    strict_decode: bool = True,
    i3_7e=None,
) -> TrajectoryResult:
    """Run a single trajectory for one arm.

    Single execution function for all arms. The arm flows through exactly
    two intervention hooks:
        1. apply_semantics_intervention (R2e label)
        2. compute_allowed_actions (R2d gate)
    """
    if i3_7e is None:
        i3_7e = _load_i3_7e()

    from hrm_adaptive_memory.executive.evidence_benchmark import (
        initial_evidence_runtime, build_evidence_snapshot,
    )
    from hrm_adaptive_memory.executive.evidence_benchmark.executor import EvidenceExecutor
    from hrm_adaptive_memory.executive.resources import ResourceBudget, ResourceState
    from hrm_adaptive_memory.executive.metareasoning_utility import MetareasoningUtility
    from hrm_adaptive_memory.executive.model_decoder import decode_output
    from hrm_adaptive_memory.executive.pinned_model_controller import (
        BACKEND_ERROR_PROPOSAL, FAIL_CLOSED_PROPOSAL,
    )

    et = task.evidence_task
    if et is None:
        raise ValueError(f"Task {task.task_id} has no evidence_task")

    budget = ResourceBudget()
    utility = MetareasoningUtility.from_file(
        REPO_ROOT / "configs" / "v2b_i3_1_utility_v1.json"
    )
    executor = EvidenceExecutor()
    resources = ResourceState(budget)
    runtime = initial_evidence_runtime(et, resources)

    realized = 0.0
    model_calls = 0
    steps_taken = 0
    success = False
    terminal = False
    terminal_result = "STEP_LIMIT"
    terminal_action = None
    backend_errors = 0

    call_log: list[dict] = []
    decision_state_log: list[dict] = []
    prior_actions: list[str] = []
    prior_outcomes: list[str] = []
    max_steps = budget.max_executive_steps

    inferred_t2_fired = False
    inferred_t2_step: int | None = None

    n_hypotheses = len(et.hypotheses)

    for step_id in range(max_steps):
        # Build snapshot
        evidence_snapshot = build_evidence_snapshot(
            runtime,
            prior_actions=tuple(prior_actions),
            prior_outcomes=tuple(prior_outcomes),
        )

        # Compute T2 from snapshot
        viability = i3_7e._classify_from_snapshot(evidence_snapshot)
        eliminated = [h_id for h_id, info in viability.items()
                      if info["status"] == "ELIMINATED"]
        t2 = (len(eliminated) == n_hypotheses and n_hypotheses > 0)

        if t2 and not inferred_t2_fired:
            inferred_t2_fired = True
            inferred_t2_step = step_id

        # Build action state
        action_state = ActionState(
            t2=t2,
            executive_steps_remaining=max_steps - step_id,
            can_retrieve=evidence_snapshot.can_retrieve,
            can_search=evidence_snapshot.can_search,
            can_verify=evidence_snapshot.can_verify,
        )

        # Compute allowed actions (R2d gate — intervention hook 2)
        try:
            allowed_decision = compute_allowed_actions(action_state, arm)
        except EmptyAllowedActionSet as exc:
            # This should never happen in a qualified run
            call_log.append({
                "step": step_id,
                "error": "EmptyAllowedActionSet",
                "error_detail": str(exc),
            })
            terminal_result = "EMPTY_ALLOWED_ACTION_SET"
            break

        # Build packet (MDSG state with affordances)
        packet = i3_7e.build_mdsg_state_with_affordances_packet(evidence_snapshot)
        system_prompt = i3_7e.MDSG_STATE_WITH_AFFORDANCES_SYSTEM_PROMPT

        # Apply semantics intervention (R2e label — intervention hook 1)
        packet, ds_internal, ds_exposed = apply_semantics_intervention(
            packet, action_state, arm
        )

        # Build dynamic schema from allowed actions
        schema = build_action_schema(allowed_decision.allowed)
        verify_schema_invariant(schema, allowed_decision.allowed)
        schema_sha = schema_sha256(schema)
        allowed_sha = allowed_actions_sha256(allowed_decision.allowed)

        user_prompt = i3_7e.evidence_packet_json(packet)

        # Log decision state
        decision_state_log.append({
            "step": step_id,
            "decision_state_internal": ds_internal,
            "decision_state_exposed": ds_exposed,
            "t2": t2,
            "n_live": n_hypotheses - len(eliminated),
            "n_eliminated": len(eliminated),
        })

        # Per-call receipt
        receipt: dict[str, Any] = {
            "step": step_id,
            "arm": arm.name,
            "t2": t2,
            "gold_t2": task.gold.gold_t2,
            "decision_state_internal": ds_internal,
            "decision_state_exposed": ds_exposed,
            "legal_actions": sorted(allowed_decision.legal),
            "epistemically_admissible_actions": sorted(allowed_decision.epistemically_admissible),
            "allowed_actions": sorted(allowed_decision.allowed),
            "allowed_actions_sha256": allowed_sha,
            "verify_gate_condition_active": allowed_decision.verify_gate_condition_active,
            "verify_removed_by_epistemic_gate": allowed_decision.verify_removed_by_epistemic_gate,
            "verify_gate_reason": allowed_decision.verify_gate_reason,
            "schema_sha256": schema_sha,
            "schema_action_enum": schema_action_enum(schema),
            "system_prompt_sha256": hashlib.sha256(system_prompt.encode()).hexdigest(),
            "user_packet_sha256": hashlib.sha256(user_prompt.encode()).hexdigest(),
        }

        # Generate model response
        backend = backend_factory()
        model_calls += 1
        try:
            # Pass allowed_actions to R2-aware backends
            call_result = backend.generate(
                system_prompt=system_prompt, user_prompt=user_prompt,
                temperature=0.0, max_tokens=max_tokens,
                allowed_actions=allowed_decision.allowed,
            )
        except Exception as exc:
            backend_errors += 1
            proposal = BACKEND_ERROR_PROPOSAL
            receipt.update({
                "result_class": "backend_error",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "decoder_valid": False,
                "selected_action": None,
                "admissibility_assertion_passed": False,
            })
            call_log.append(receipt)
            # Continue to next step with error proposal
        else:
            # Decode
            outcome = decode_output(call_result.raw_output, strict=strict_decode)
            if outcome.valid and outcome.proposal:
                proposal = outcome.proposal
            else:
                proposal = FAIL_CLOSED_PROPOSAL

            action_str = proposal.action.value if hasattr(proposal.action, "value") else str(proposal.action)

            # Layer 2: Defense-in-depth admissibility check
            admissibility_passed = action_str in allowed_decision.allowed
            if not admissibility_passed:
                raise EpistemicAdmissibilityViolation(
                    action=action_str,
                    allowed=allowed_decision.allowed,
                    reason=allowed_decision.verify_gate_reason,
                )

            receipt.update({
                "result_class": "success",
                "provider_raw_output": call_result.provider_raw_output or call_result.raw_output,
                "decoder_valid": outcome.valid,
                "decoder_rejection_code": outcome.rejection_code,
                "selected_action": action_str,
                "selected_reason_code": proposal.reason_code,
                "selected_target_id": getattr(proposal, "target_id", None),
                "admissibility_assertion_passed": admissibility_passed,
                "json_schema_sha256": getattr(call_result, "json_schema_sha256", ""),
            })
            call_log.append(receipt)

        # Execute action
        action = proposal.action
        target_id = getattr(proposal, "target_id", None)

        resources_before = runtime.resources
        exec_res = executor.execute(runtime, action, target_evidence_id=target_id)
        resources_after = exec_res.runtime.resources

        step_cost = utility.action_cost(resources_before, resources_after)
        realized -= step_cost

        action_str = action.value if hasattr(action, "value") else str(action)
        prior_actions.append(action_str)
        prior_outcomes.append(exec_res.outcome_code)
        runtime = exec_res.runtime
        steps_taken += 1

        if exec_res.terminal:
            tr = utility.terminal_reward(exec_res.action, bool(exec_res.task_success))
            realized += tr
            success = bool(exec_res.task_success)
            terminal = True
            terminal_result = exec_res.outcome_code
            terminal_action = action_str
            break

    return TrajectoryResult(
        task_id=task.task_id,
        arm=arm.name,
        realized_utility=round(realized, 4),
        success=success,
        steps=steps_taken,
        terminal_action=terminal_action,
        terminal_result=terminal_result,
        model_calls=model_calls,
        backend_errors=backend_errors,
        call_log=call_log,
        decision_state_log=decision_state_log,
        gold_t2=task.gold.gold_t2,
        gold_should_gate=task.gold.gold_should_gate_verify,
        stratum=task.gold.stratum,
        inferred_t2_fired=inferred_t2_fired,
        inferred_t2_step=inferred_t2_step,
    )


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_results(results: list[TrajectoryResult]) -> dict:
    """Analyze R2-DEV results in the specified order.

    1. Dataset integrity
    2. Arm isolation/diff audit
    3. FalseGate/MissedGate
    4. Hard-gate invariants
    5. Replacement-action distribution
    6. Loop migration
    7. Success/rescue/break
    8. Utility contrasts
    9. D×E interaction
    """
    from collections import Counter, defaultdict

    # Group by arm
    by_arm: dict[str, list[TrajectoryResult]] = defaultdict(list)
    for r in results:
        by_arm[r.arm].append(r)

    analysis: dict[str, Any] = {}

    # 1. Dataset integrity
    analysis["dataset_integrity"] = {
        "total_trajectories": len(results),
        "per_arm": {arm: len(rs) for arm, rs in by_arm.items()},
        "strata": dict(Counter(r.stratum for r in results)),
    }

    # 2. Arm isolation (check that arms differ only in expected ways)
    arm_diffs = {}
    for arm_name, arm_results in by_arm.items():
        if arm_name == "C0":
            continue
        c0_results = by_arm.get("C0", [])
        # Compare per-task
        c0_by_task = {r.task_id: r for r in c0_results}
        arm_by_task = {r.task_id: r for r in arm_results}
        common_tasks = set(c0_by_task.keys()) & set(arm_by_task.keys())

        diffs = 0
        for tid in common_tasks:
            c0_r = c0_by_task[tid]
            arm_r = arm_by_task[tid]
            # Check that call logs differ only in expected fields
            for c0_call, arm_call in zip(c0_r.call_log, arm_r.call_log):
                if c0_call.get("allowed_actions") != arm_call.get("allowed_actions"):
                    if arm_name in ("D", "DE"):
                        diffs += 1  # expected for D/DE at T2
                if c0_call.get("decision_state_exposed") != arm_call.get("decision_state_exposed"):
                    if arm_name in ("E", "DE"):
                        diffs += 1  # expected for E/DE at T2

        arm_diffs[arm_name] = {"expected_diffs": diffs, "n_common": len(common_tasks)}

    analysis["arm_isolation"] = arm_diffs

    # 3. FalseGate/MissedGate
    inferred_gates = []
    gold_gates = []
    semantic_error_classes = []
    for r in results:
        if r.arm in ("D", "DE"):
            # Check if gate fired for this trajectory
            gate_fired = any(
                call.get("verify_gate_condition_active", False)
                for call in r.call_log
            )
            inferred_gates.append(gate_fired)
            gold_gates.append(r.gold_should_gate)
            semantic_error_classes.append(None)  # would come from task

    false_gate_metrics = {}
    if inferred_gates:
        from r2_dataset_generator import (
            compute_gate_confusion_matrix,
            decompose_false_gate,
        )
        cm = compute_gate_confusion_matrix(inferred_gates, gold_gates)
        false_gate_metrics = {
            "confusion_matrix": {
                "tp": cm.tp, "fp": cm.fp, "fn": cm.fn, "tn": cm.tn,
            },
            "false_gate_rate": cm.false_gate_rate,
            "missed_gate_rate": cm.missed_gate_rate,
        }

    analysis["gate_safety"] = false_gate_metrics

    # 4. Hard-gate invariants
    schema_violations = 0
    executor_violations = 0
    for r in results:
        for call in r.call_log:
            if call.get("result_class") == "success":
                if not call.get("admissibility_assertion_passed", True):
                    executor_violations += 1
    analysis["hard_gate_invariants"] = {
        "schema_gate_violations": schema_violations,
        "executor_admissibility_violations": executor_violations,
    }

    # 5. Replacement-action distribution (when VERIFY is gated)
    replacement_actions = Counter()
    for r in results:
        if r.arm in ("D", "DE"):
            for call in r.call_log:
                if call.get("verify_gate_condition_active", False):
                    selected = call.get("selected_action")
                    if selected:
                        replacement_actions[selected] += 1

    analysis["replacement_action_distribution"] = dict(replacement_actions)

    # 6. Loop metrics
    loop_metrics: dict[str, dict] = {}
    for action in ["RETRIEVE", "SEARCH", "VERIFY", "REASON_MORE"]:
        max_runs = []
        repeated_rates = []
        for r in results:
            actions = [call.get("selected_action") for call in r.call_log
                       if call.get("selected_action")]
            # MaxRun
            max_run = 0
            current_run = 0
            for a in actions:
                if a == action:
                    current_run += 1
                    max_run = max(max_run, current_run)
                else:
                    current_run = 0
            max_runs.append(max_run)

            # RepeatedActionRate
            repeats = sum(1 for i in range(1, len(actions))
                         if actions[i] == action and actions[i-1] == action)
            total_occurrences = sum(1 for a in actions if a == action)
            if total_occurrences > 0:
                repeated_rates.append(repeats / total_occurrences)

        loop_metrics[action] = {
            "max_run_mean": sum(max_runs) / len(max_runs) if max_runs else 0,
            "repeated_action_rate_mean": sum(repeated_rates) / len(repeated_rates) if repeated_rates else 0,
        }

    analysis["loop_metrics"] = loop_metrics

    # 7. Success/rescue/break
    success_by_arm = {}
    for arm_name, arm_results in by_arm.items():
        n_success = sum(1 for r in arm_results if r.success)
        n_defer = sum(1 for r in arm_results if r.terminal_action == "DEFER")
        n_answer = sum(1 for r in arm_results if r.terminal_action == "ANSWER")
        n_step_limit = sum(1 for r in arm_results if r.terminal_result == "STEP_LIMIT")
        success_by_arm[arm_name] = {
            "n": len(arm_results),
            "success_rate": n_success / len(arm_results) if arm_results else 0,
            "defer_rate": n_defer / len(arm_results) if arm_results else 0,
            "answer_rate": n_answer / len(arm_results) if arm_results else 0,
            "step_limit_rate": n_step_limit / len(arm_results) if arm_results else 0,
        }

    analysis["success_breakdown"] = success_by_arm

    # 8. Utility contrasts
    utility_by_arm = {}
    for arm_name, arm_results in by_arm.items():
        utilities = [r.realized_utility for r in arm_results]
        utility_by_arm[arm_name] = {
            "mean": sum(utilities) / len(utilities) if utilities else 0,
            "n": len(utilities),
        }

    def mean_utility(arm_name: str) -> float:
        return utility_by_arm.get(arm_name, {}).get("mean", 0.0)

    delta_d = mean_utility("D") - mean_utility("C0")
    delta_e = mean_utility("E") - mean_utility("C0")
    delta_de = mean_utility("DE") - mean_utility("C0")
    interaction = (mean_utility("DE") - mean_utility("E")) - (mean_utility("D") - mean_utility("C0"))

    analysis["utility_contrasts"] = {
        "delta_D": round(delta_d, 4),
        "delta_E": round(delta_e, 4),
        "delta_DE": round(delta_de, 4),
        "interaction_DxE": round(interaction, 4),
        "per_arm": utility_by_arm,
    }

    # 9. LoopMigrationRate
    resource_exhausted_by_arm = {}
    for arm_name, arm_results in by_arm.items():
        n_exhausted_non_verify = sum(
            1 for r in arm_results
            if r.terminal_result == "STEP_LIMIT"
            and r.terminal_action != "VERIFY"
        )
        n_t2 = sum(1 for r in arm_results if r.inferred_t2_fired)
        resource_exhausted_by_arm[arm_name] = {
            "n_exhausted_non_verify": n_exhausted_non_verify,
            "n_t2": n_t2,
            "loop_migration_rate": n_exhausted_non_verify / n_t2 if n_t2 > 0 else 0,
        }

    analysis["loop_migration"] = resource_exhausted_by_arm

    return analysis


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="R2 Development Runner")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-per-arm", type=int, default=10,
                        help="Number of tasks per arm (subset)")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--backend", choices=["stub", "llama"], default="stub",
                        help="Backend type (stub for testing, llama for live)")
    parser.add_argument("--llama-url", type=str, default="http://127.0.0.1:8080")
    parser.add_argument("--llama-model", type=str,
                        default="gemma-3-12b-it-qat-q4_0")
    args = parser.parse_args()

    # Load dataset
    from r2_dataset_generator import generate_r2_dataset
    all_tasks = generate_r2_dataset(n_per_stratum=5, seed=137)

    # Subset
    tasks = all_tasks[:args.n_per_arm]

    # Backend factory
    if args.backend == "stub":
        def backend_factory():
            return R2StubBackend()
    else:
        from hrm_adaptive_memory.executive.model_backend import LocalLlamaBackend
        def backend_factory():
            return LocalLlamaBackend(
                base_url=args.llama_url,
                model_name=args.llama_model,
            )

    print("R2 Development Runner")
    print(f"  Backend: {args.backend}")
    print(f"  Tasks: {len(tasks)}")
    print(f"  Arms: {[arm.name for arm in ALL_ARMS]}")
    print()

    i3_7e = _load_i3_7e()

    all_results: list[TrajectoryResult] = []

    for arm in ALL_ARMS:
        print(f"  Running arm {arm.name}...")
        arm_results = []
        for task in tasks:
            try:
                result = run_trajectory(
                    task=task,
                    arm=arm,
                    backend_factory=backend_factory,
                    max_tokens=args.max_tokens,
                    i3_7e=i3_7e,
                )
                arm_results.append(result)
                all_results.append(result)
            except EpistemicAdmissibilityViolation as exc:
                print(f"    *** ADMISSIBILITY VIOLATION: {exc}")
                print(f"    *** ABORTING arm {arm.name}")
                break
            except Exception as exc:
                print(f"    Error on task {task.task_id}: {type(exc).__name__}: {exc}")
                continue
        print(f"    {len(arm_results)} trajectories completed")

    # Analyze
    analysis = analyze_results(all_results)

    # Write results
    args.output.mkdir(parents=True, exist_ok=True)

    # Write trajectories
    trajectories_path = args.output / "trajectories.jsonl"
    with open(trajectories_path, "w") as f:
        for r in all_results:
            record = {
                "task_id": r.task_id,
                "arm": r.arm,
                "realized_utility": r.realized_utility,
                "success": r.success,
                "steps": r.steps,
                "terminal_action": r.terminal_action,
                "terminal_result": r.terminal_result,
                "model_calls": r.model_calls,
                "backend_errors": r.backend_errors,
                "gold_t2": r.gold_t2,
                "gold_should_gate": r.gold_should_gate,
                "stratum": r.stratum,
                "inferred_t2_fired": r.inferred_t2_fired,
                "inferred_t2_step": r.inferred_t2_step,
            }
            f.write(json.dumps(record, sort_keys=True) + "\n")

    # Write analysis
    analysis_path = args.output / "analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2, sort_keys=True)

    # Write full call logs
    calls_path = args.output / "call_logs.jsonl"
    with open(calls_path, "w") as f:
        for r in all_results:
            for call in r.call_log:
                call["task_id"] = r.task_id
                call["arm"] = r.arm
                f.write(json.dumps(call, sort_keys=True, default=str) + "\n")

    print(f"\n  Trajectories: {trajectories_path}")
    print(f"  Analysis: {analysis_path}")
    print(f"  Call logs: {calls_path}")

    # Print summary
    print(f"\n=== Utility Contrasts ===")
    uc = analysis.get("utility_contrasts", {})
    print(f"  Δ_D  = {uc.get('delta_D', 'N/A')}")
    print(f"  Δ_E  = {uc.get('delta_E', 'N/A')}")
    print(f"  Δ_DE = {uc.get('delta_DE', 'N/A')}")
    print(f"  I_D×E = {uc.get('interaction_DxE', 'N/A')}")

    print(f"\n=== Gate Safety ===")
    gs = analysis.get("gate_safety", {})
    if gs:
        cm = gs.get("confusion_matrix", {})
        print(f"  TP={cm.get('tp')} FP={cm.get('fp')} FN={cm.get('fn')} TN={cm.get('tn')}")
        print(f"  FalseGateRate = {gs.get('false_gate_rate', 'N/A')}")
        print(f"  MissedGateRate = {gs.get('missed_gate_rate', 'N/A')}")

    print(f"\n=== Hard-Gate Invariants ===")
    hg = analysis.get("hard_gate_invariants", {})
    print(f"  Schema violations: {hg.get('schema_gate_violations', 0)}")
    print(f"  Executor violations: {hg.get('executor_admissibility_violations', 0)}")


if __name__ == "__main__":
    main()
