#!/usr/bin/env python3
"""Oracle-path semantic validator for benchmark tasks.

Uses CANONICAL topology semantics from daph.epistemic.topology — the same
implementation used by the executive. No duplicated epistemic-state logic.

For every generated task, executes the declared oracle path deterministically
and asserts that the terminal state matches the expected terminal.

Checks:
  G_B1: oracle actions all legal at each step
  G_B2: nonterminal oracle states are exactly CONTINUE_REQUIRED
  G_B3: terminal oracle state matches expected terminal (ANSWER_READY or DEFER_READY)
  G_B4: correct hypothesis is the canonical unique supported hypothesis

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

# CANONICAL imports — same semantics as the executive
from daph.epistemic.topology import (
    derive_hypothesis_topology,
    classify_terminal_readiness,
    is_answer_ready,
)
from daph.epistemic.types import HypothesisState, TerminalReadiness


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


def canonical_topology_and_readiness(runtime) -> tuple:
    """Derive canonical topology and readiness from runtime.

    This is the SINGLE canonical implementation — same as the executive.
    No duplicated epistemic-state logic.
    """
    task = runtime.task
    hypothesis_ids = [h.hypothesis_id for h in task.hypotheses]
    evidence = runtime.evidence

    # Derive canonical topology
    topology = derive_hypothesis_topology(
        evidence_items=evidence,
        hypothesis_ids=hypothesis_ids,
        hidden_evidence_count=0,  # All evidence is visible in oracle validation
    )

    # Determine action admissibility for readiness classification
    resources = runtime.resources
    can_verify = (
        resources.budget.max_verification_calls > 0
        and topology.unverified_evidence_exists
    )
    can_retrieve = (
        resources.budget.max_retrieval_calls > 0
        and topology.hidden_evidence_count > 0
    )
    can_search = (
        resources.budget.max_search_calls > 0
        and not runtime.searched
    )

    # Check for discriminating evidence
    has_unverified_discriminating = topology.unverified_evidence_exists
    has_hidden_evidence = topology.hidden_evidence_count > 0
    search_could_discriminate = can_search  # Simplified

    readiness = classify_terminal_readiness(
        topology,
        can_verify=can_verify,
        can_retrieve=can_retrieve,
        can_search=can_search,
        has_unverified_discriminating_evidence=has_unverified_discriminating,
        has_hidden_evidence=has_hidden_evidence,
        search_could_discriminate=search_could_discriminate,
    )

    return topology, readiness


def validate_task_oracle(task: EvidenceTask) -> OracleValidationResult:
    """Validate a single task's oracle path using canonical semantics."""
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
            # G_B3: Check terminal readiness matches expected terminal
            topology, readiness = canonical_topology_and_readiness(runtime)
            result.actual_terminal_readiness = readiness.value

            if action_name == "ANSWER":
                if readiness != TerminalReadiness.ANSWER_READY:
                    result.g_b3_terminal_matches = False
                    result.failure_reason = (
                        f"Oracle ANSWER at step {i} but canonical readiness="
                        f"{readiness.value} (not ANSWER_READY). "
                        f"Topology: n_supported={topology.n_viable_hypotheses}, "
                        f"n_contradicted={topology.n_eliminated_hypotheses}, "
                        f"unique_supported={topology.unique_supported_hypothesis}, "
                        f"has_competition={topology.has_verified_unresolved_competition}"
                    )
                    result.valid = False
                    break

            elif action_name == "DEFER":
                if readiness != TerminalReadiness.DEFER_READY:
                    result.g_b3_terminal_matches = False
                    result.failure_reason = (
                        f"Oracle DEFER at step {i} but canonical readiness="
                        f"{readiness.value} (not DEFER_READY). "
                        f"Topology: n_supported={topology.n_viable_hypotheses}, "
                        f"n_contradicted={topology.n_eliminated_hypotheses}, "
                        f"can_verify={topology.unverified_evidence_exists}"
                    )
                    result.valid = False
                    break

            # G_B4: Check correct hypothesis is the canonical unique supported
            if action_name == "ANSWER":
                correct_id = task.correct_hypothesis_id
                canonical_unique = topology.unique_supported_hypothesis

                if canonical_unique != correct_id:
                    result.g_b4_hypothesis_justified = False
                    result.failure_reason = (
                        f"Correct hypothesis {correct_id} is not the canonical "
                        f"unique supported hypothesis. "
                        f"Canonical unique: {canonical_unique}. "
                        f"Topology states: {dict(topology.hypothesis_states)}"
                    )
                    result.valid = False
                    break

            break

        else:
            # G_B2: Non-terminal action — readiness must be EXACTLY CONTINUE_REQUIRED
            topology, readiness = canonical_topology_and_readiness(runtime)

            if readiness != TerminalReadiness.CONTINUE_REQUIRED:
                result.g_b2_continuation_required = False
                result.failure_reason = (
                    f"Oracle continues with {action_name} at step {i} "
                    f"but canonical readiness={readiness.value} "
                    f"(must be exactly CONTINUE_REQUIRED). "
                    f"Topology: n_supported={topology.n_viable_hypotheses}, "
                    f"unique_supported={topology.unique_supported_hypothesis}"
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
                if hasattr(exec_result, 'next_runtime'):
                    runtime = exec_result.next_runtime
                elif hasattr(exec_result, 'runtime'):
                    runtime = exec_result.runtime
                else:
                    result.g_b1_actions_legal = False
                    result.failure_reason = f"Cannot extract runtime from exec result: {type(exec_result)}"
                    result.valid = False
                    break
            except Exception as e:
                result.g_b1_actions_legal = False
                result.failure_reason = f"Execution failed at step {i}: {e}"
                result.valid = False
                break

    # Build final state summary from canonical topology
    topology, readiness = canonical_topology_and_readiness(runtime)
    result.final_state_summary = (
        f"n_supported={topology.n_viable_hypotheses}, "
        f"n_contradicted={topology.n_eliminated_hypotheses}, "
        f"n_weakened={topology.n_weakened_hypotheses}, "
        f"unique_supported={topology.unique_supported_hypothesis}, "
        f"readiness={readiness.value}"
    )

    return result


def validate_all_templates():
    """Validate all OOD domain templates."""
    from build_structural_ood_pool import OOD_DOMAIN_TEMPLATES, generate_ood_candidate

    results = []
    valid_count = 0
    invalid_count = 0

    print(f"{'='*90}")
    print("ORACLE-PATH SEMANTIC VALIDATION (CANONICAL TOPOLOGY)")
    print(f"{'='*90}")
    print(f"{'Category':>40} {'Valid':>6} {'G_B1':>5} {'G_B2':>5} {'G_B3':>5} {'G_B4':>5} {'Reason'}")
    print("-" * 120)

    for template in OOD_DOMAIN_TEMPLATES:
        task = generate_ood_candidate(template, 0)
        result = validate_task_oracle(task)
        results.append(result)

        if result.valid:
            valid_count += 1
        else:
            invalid_count += 1

        reason = (result.failure_reason or "")[:60]
        print(f"{result.category:>40} {str(result.valid):>6} "
              f"{str(result.g_b1_actions_legal):>5} "
              f"{str(result.g_b2_continuation_required):>5} "
              f"{str(result.g_b3_terminal_matches):>5} "
              f"{str(result.g_b4_hypothesis_justified):>5} "
              f"{reason}")

    print(f"\n{'='*90}")
    print(f"Valid: {valid_count}, Invalid: {invalid_count}")
    print(f"{'='*90}")

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
        "validator": "canonical_topology",
        "topology_source": "daph.epistemic.topology.derive_hypothesis_topology",
        "readiness_source": "daph.epistemic.topology.classify_terminal_readiness",
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
