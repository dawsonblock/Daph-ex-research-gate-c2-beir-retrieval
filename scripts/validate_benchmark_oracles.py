#!/usr/bin/env python3
"""Oracle-path semantic validator for benchmark tasks.

For every generated task, executes the declared oracle path deterministically
and asserts that the terminal state matches the expected terminal.

Checks:
  G_B1: oracle actions all valid for the state
  G_B2: nonterminal oracle states are continuation-required
  G_B3: terminal oracle state matches expected terminal
  G_B4: correct hypothesis is structurally justified at terminal

Usage:
    python scripts/validate_benchmark_oracles.py
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from hrm_adaptive_memory.cognitive_control.core import DecisionAction
from hrm_adaptive_memory.cognitive_control.state import (
    TemporalStatus, VerificationState,
)
from hrm_adaptive_memory.executive.evidence_benchmark.schema import (
    EvidenceHypothesis, EvidenceItem, EvidenceTask,
    initial_evidence_runtime,
)
from hrm_adaptive_memory.executive.evidence_benchmark.executor import EvidenceExecutor
from hrm_adaptive_memory.executive.resources import ResourceBudget, ResourceState


@dataclass
class OracleValidationResult:
    task_id: str
    category: str
    valid: bool
    g_b1_actions_legal: bool
    g_b2_continuation_required: bool
    g_b3_terminal_matches: bool
    g_b4_hypothesis_justified: bool
    failure_reason: str | None = None
    oracle_path: tuple[str, ...] = ()
    expected_terminal: str = ""
    actual_terminal_readiness: str = ""
    final_state_summary: str = ""


def get_readiness(runtime) -> str:
    """Determine the canonical readiness of the current state."""
    task = runtime.task
    hypotheses = task.hypotheses
    evidence = runtime.evidence

    n_viable = 0
    n_eliminated = 0
    n_verified_support = 0

    for h in hypotheses:
        hyp_id = h.hypothesis_id
        vs = False
        vc = False
        for e in evidence:
            if e.verification_state == VerificationState.SUFFICIENT:
                if hyp_id in e.supports:
                    vs = True
                if hyp_id in e.contradicts:
                    vc = True
            elif e.verification_state == VerificationState.FALSIFIED:
                if hyp_id in e.supports:
                    vc = True
                if hyp_id in e.contradicts:
                    vs = True

        if vc and not vs:
            n_eliminated += 1
        elif vs and not vc:
            n_viable += 1
            n_verified_support += 1
        elif not vs and not vc:
            n_viable += 1
        else:
            n_viable += 1

    if n_verified_support == 1:
        return "ANSWER_READY"
    if n_verified_support >= 2:
        return "COMPETING_VERIFIED"
    if n_viable == 0:
        return "ALL_ELIMINATED"
    return "CONTINUE_REQUIRED"


def validate_task_oracle(task: EvidenceTask) -> OracleValidationResult:
    """Validate a single task's oracle path by simulating it."""
    parts = task.budget_profile.split("_")
    budget = ResourceBudget(
        max_executive_steps=int(parts[1]) if len(parts) > 1 else 4,
        max_reasoning_tokens=256,
        max_retrieval_calls=0,
        max_verification_calls=int(parts[2]) if len(parts) > 2 else 2,
        max_search_calls=int(parts[3]) if len(parts) > 3 else 0,
        max_elapsed_ms=10000,
    )
    resources = ResourceState(budget=budget)
    runtime = initial_evidence_runtime(task, resources)
    executor = EvidenceExecutor()

    oracle_path = task.oracle_resolution_path
    expected_terminal = task.expected_terminal.value

    result = OracleValidationResult(
        task_id=task.task_id,
        category=task.category,
        valid=True,
        g_b1_actions_legal=True,
        g_b2_continuation_required=True,
        g_b3_terminal_matches=True,
        g_b4_hypothesis_justified=True,
        oracle_path=oracle_path,
        expected_terminal=expected_terminal,
    )

    for i, action_name in enumerate(oracle_path):
        is_terminal = action_name in ("ANSWER", "DEFER")

        if is_terminal:
            readiness = get_readiness(runtime)

            if action_name == "ANSWER" and readiness != "ANSWER_READY":
                result.g_b3_terminal_matches = False
                result.failure_reason = (
                    f"Oracle ANSWER at step {i} but readiness={readiness} "
                    f"(not ANSWER_READY). "
                    f"n_verified_support > 1 or n_viable > 1."
                )
                result.actual_terminal_readiness = readiness
                result.valid = False
                break

            # G_B4: Check correct hypothesis is uniquely justified
            if action_name == "ANSWER":
                correct_id = task.correct_hypothesis_id
                has_support = False
                other_support = 0

                for h in runtime.task.hypotheses:
                    h_id = h.hypothesis_id
                    for e in runtime.evidence:
                        if e.verification_state == VerificationState.SUFFICIENT:
                            if h_id in e.supports and h_id == correct_id:
                                has_support = True
                            elif h_id in e.supports and h_id != correct_id:
                                # Check if this other hypothesis is eliminated
                                other_eliminated = False
                                for e2 in runtime.evidence:
                                    if e2.verification_state in (VerificationState.FALSIFIED, VerificationState.SUFFICIENT):
                                        if h_id in e2.contradicts:
                                            other_eliminated = True
                                if not other_eliminated:
                                    other_support += 1

                if not has_support or other_support > 0:
                    result.g_b4_hypothesis_justified = False
                    result.failure_reason = (
                        f"Correct hypothesis {correct_id} not uniquely justified. "
                        f"has_support={has_support}, other_support={other_support}"
                    )
                    result.valid = False
                    break

            break

        else:
            # Non-terminal action — check continuation is required
            readiness = get_readiness(runtime)
            if readiness == "ANSWER_READY":
                result.g_b2_continuation_required = False
                result.failure_reason = (
                    f"Oracle continues with {action_name} at step {i} "
                    f"but readiness=ANSWER_READY (should have terminated)"
                )
                result.valid = False
                break

            # Execute the step
            action_map = {
                "VERIFY": DecisionAction.VERIFY,
                "REASON_MORE": DecisionAction.REASON_MORE,
                "SEARCH_MORE": DecisionAction.SEARCH_MORE,
                "RETRIEVE": DecisionAction.RETRIEVE,
            }
            action = action_map.get(action_name)
            if action is None:
                result.g_b1_actions_legal = False
                result.failure_reason = f"Unknown oracle action: {action_name}"
                result.valid = False
                break

            try:
                exec_result = executor.execute(runtime, action)
                # The execute method returns an EvidenceActionExecution
                # We need to update the runtime from it
                # Check if execution succeeded
                if hasattr(exec_result, 'next_runtime'):
                    runtime = exec_result.next_runtime
                elif hasattr(exec_result, 'runtime'):
                    runtime = exec_result.runtime
                else:
                    # The executor may mutate runtime in place or return a new one
                    # Let's check what it returns
                    result.g_b1_actions_legal = False
                    result.failure_reason = f"Cannot extract runtime from exec result: {type(exec_result)}"
                    result.valid = False
                    break
            except Exception as e:
                result.g_b1_actions_legal = False
                result.failure_reason = f"Execution failed at step {i}: {e}"
                result.valid = False
                break

    # Build final state summary
    n_viable = 0
    n_verified_support = 0
    for h in runtime.task.hypotheses:
        h_id = h.hypothesis_id
        vs = False
        vc = False
        for e in runtime.evidence:
            if e.verification_state == VerificationState.SUFFICIENT:
                if h_id in e.supports:
                    vs = True
                if h_id in e.contradicts:
                    vc = True
            if e.verification_state == VerificationState.FALSIFIED:
                if h_id in e.supports:
                    vc = True
                if h_id in e.contradicts:
                    vs = True
        if vs and not vc:
            n_verified_support += 1
        if not vc or vs:
            n_viable += 1

    result.final_state_summary = (
        f"n_viable={n_viable}, n_verified_support={n_verified_support}, "
        f"readiness={get_readiness(runtime)}"
    )

    return result


def validate_all_templates():
    """Validate all OOD domain templates."""
    from build_structural_ood_pool import OOD_DOMAIN_TEMPLATES, generate_ood_candidate

    results = []
    valid_count = 0
    invalid_count = 0

    print(f"{'='*80}")
    print("ORACLE-PATH SEMANTIC VALIDATION")
    print(f"{'='*80}")
    print(f"{'Category':>35} {'Valid':>6} {'G_B1':>5} {'G_B2':>5} {'G_B3':>5} {'G_B4':>5} {'Reason'}")
    print("-" * 110)

    for template in OOD_DOMAIN_TEMPLATES:
        task = generate_ood_candidate(template, 0)
        result = validate_task_oracle(task)
        results.append(result)

        if result.valid:
            valid_count += 1
        else:
            invalid_count += 1

        reason = (result.failure_reason or "")[:50]
        print(f"{result.category:>35} {str(result.valid):>6} "
              f"{str(result.g_b1_actions_legal):>5} "
              f"{str(result.g_b2_continuation_required):>5} "
              f"{str(result.g_b3_terminal_matches):>5} "
              f"{str(result.g_b4_hypothesis_justified):>5} "
              f"{reason}")

    print(f"\n{'='*80}")
    print(f"Valid: {valid_count}, Invalid: {invalid_count}")
    print(f"{'='*80}")

    # Show details for invalid
    if invalid_count > 0:
        print(f"\nINVALID TEMPLATE DETAILS:")
        for r in results:
            if not r.valid:
                print(f"\n  {r.category}:")
                print(f"    Oracle path: {r.oracle_path}")
                print(f"    Expected: {r.expected_terminal}")
                print(f"    Readiness: {r.actual_terminal_readiness}")
                print(f"    Final state: {r.final_state_summary}")
                print(f"    Reason: {r.failure_reason}")

    # Save
    output = {
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "results": [
            {
                "task_id": r.task_id,
                "category": r.category,
                "valid": r.valid,
                "g_b1_actions_legal": r.g_b1_actions_legal,
                "g_b2_continuation_required": r.g_b2_continuation_required,
                "g_b3_terminal_matches": r.g_b3_terminal_matches,
                "g_b4_hypothesis_justified": r.g_b4_hypothesis_justified,
                "failure_reason": r.failure_reason,
                "oracle_path": list(r.oracle_path),
                "expected_terminal": r.expected_terminal,
                "actual_terminal_readiness": r.actual_terminal_readiness,
                "final_state_summary": r.final_state_summary,
            }
            for r in results
        ],
    }

    output_path = REPO_ROOT / "experiments/i3_30r3/structural_ood/oracle_validation.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")

    return output


if __name__ == "__main__":
    results = validate_all_templates()
    sys.exit(0 if results["invalid_count"] == 0 else 1)
