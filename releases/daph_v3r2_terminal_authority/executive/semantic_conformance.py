"""Semantic conformance checker.

For every targeted decision state, mechanically verifies:

    Topology = Q representation = Authority = Executor = Benchmark truth

Each component must independently agree on:
- canonical readiness (ANSWER_READY / DEFER_READY / CONTINUE_REQUIRED)
- viable hypotheses
- executor truth for ANSWER/DEFER
- certificate outputs
- causal best action
- available continuation

If any two components disagree, the state fails conformance.

This is the mechanical invariant that prevents silent semantic drift
between the topology classifier, Q model, authority certificate, executor,
and benchmark ground truth.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Sequence

from hrm_adaptive_memory.cognitive_control.core import DecisionAction
from hrm_adaptive_memory.cognitive_control.state import (
    TemporalStatus, VerificationState,
)
from hrm_adaptive_memory.executive.evidence_benchmark.schema import (
    EvidenceRuntime, EvidenceTask, EvidenceHypothesis,
)
from hrm_adaptive_memory.executive.evidence_benchmark.executor import (
    EvidenceExecutor, valid_verify_targets,
)
from hrm_adaptive_memory.executive.resources import ResourceState

from daph.epistemic.topology import (
    derive_hypothesis_topology,
    classify_terminal_readiness,
    is_answer_ready,
    is_defer_ready,
    is_continue_required,
)
from daph.epistemic.types import (
    HypothesisState,
    HypothesisTopology,
    TerminalReadiness,
)
from daph.authority.policy_v3 import (
    StructuralStateV3,
    answer_structural_certificate,
    defer_structural_certificate,
)


@dataclass(frozen=True)
class ConformanceRecord:
    """A single conformance check for one decision state.

    Records what each component says about the state, and whether they agree.
    """
    task_id: str
    step: int
    state_sha256: str

    # Topology says
    topology_readiness: str  # ANSWER_READY / DEFER_READY / CONTINUE_REQUIRED
    topology_unique_supported: str | None
    topology_n_viable: int
    topology_n_eliminated: int

    # Authority certificate says
    cert_answer: bool
    cert_defer: bool
    cert_readiness: str  # derived from certificate

    # Executor truth (what actually happens)
    executor_answer_success: bool | None
    executor_defer_success: bool | None
    executor_answer_terminal: bool
    executor_defer_terminal: bool
    executor_truth_readiness: str  # derived from executor outcomes

    # Benchmark truth
    benchmark_expected_terminal: str
    benchmark_correct_hypothesis: str | None
    benchmark_truth_readiness: str  # derived from benchmark

    # Causal best action (from executor truth)
    causal_best_action: str  # ANSWER / DEFER / CONTINUE

    # Available continuation
    can_verify: bool
    can_retrieve: bool
    can_search: bool
    has_unverified_evidence: bool
    has_hidden_evidence: bool

    # Conformance
    conformant: bool
    disagreements: tuple[str, ...]

    # Disagreement classification (P1.2: separate safe abstention from unsafe)
    # safe_abstention: certificate abstains (CONTINUE_REQUIRED) when topology
    #   says terminal — conservative, not unsafe
    # unsafe_disagreement: certificate forces a terminal action that disagrees
    #   with topology/executor/benchmark — potentially unsafe
    # no_disagreement: all components agree
    disagreement_type: str  # "no_disagreement" / "safe_abstention" / "unsafe_disagreement"

    def as_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "step": self.step,
            "state_sha256": self.state_sha256,
            "topology_readiness": self.topology_readiness,
            "topology_unique_supported": self.topology_unique_supported,
            "topology_n_viable": self.topology_n_viable,
            "topology_n_eliminated": self.topology_n_eliminated,
            "cert_answer": self.cert_answer,
            "cert_defer": self.cert_defer,
            "cert_readiness": self.cert_readiness,
            "executor_answer_success": self.executor_answer_success,
            "executor_defer_success": self.executor_defer_success,
            "executor_answer_terminal": self.executor_answer_terminal,
            "executor_defer_terminal": self.executor_defer_terminal,
            "executor_truth_readiness": self.executor_truth_readiness,
            "benchmark_expected_terminal": self.benchmark_expected_terminal,
            "benchmark_correct_hypothesis": self.benchmark_correct_hypothesis,
            "benchmark_truth_readiness": self.benchmark_truth_readiness,
            "causal_best_action": self.causal_best_action,
            "can_verify": self.can_verify,
            "can_retrieve": self.can_retrieve,
            "can_search": self.can_search,
            "has_unverified_evidence": self.has_unverified_evidence,
            "has_hidden_evidence": self.has_hidden_evidence,
            "conformant": self.conformant,
            "disagreements": list(self.disagreements),
            "disagreement_type": self.disagreement_type,
        }


def _compute_state_sha(
    task_id: str,
    step: int,
    visible_evidence: Sequence,
    resources_dict: dict,
) -> str:
    """Compute deterministic state hash."""
    evidence_data = []
    for ev in visible_evidence:
        if isinstance(ev, dict):
            evidence_data.append(ev)
        else:
            evidence_data.append({
                "evidence_id": ev.evidence_id,
                "verification_state": ev.verification_state.value,
                "temporal_status": ev.temporal_status.value,
                "supports": list(ev.supports),
                "contradicts": list(ev.contradicts),
                "retrieved": ev.retrieved,
            })
    content = json.dumps({
        "task_id": task_id,
        "step": step,
        "evidence": evidence_data,
        "resources": resources_dict,
    }, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()


def _build_structural_state_v3(
    runtime: EvidenceRuntime,
    topology: HypothesisTopology,
) -> StructuralStateV3:
    """Build StructuralStateV3 from runtime and topology."""
    rs = runtime.resources.as_dict()
    visible = runtime.visible_evidence

    has_unverified = any(
        ev.verification_state == VerificationState.UNVERIFIED
        for ev in visible
    )

    # Determine verified hyp action
    verified_hyp_action_is_answer = False
    verified_hyp_action_is_defer = False

    if topology.unique_supported_hypothesis is not None:
        for h in runtime.task.hypotheses:
            if h.hypothesis_id == topology.unique_supported_hypothesis:
                if h.answer_action == DecisionAction.ANSWER:
                    verified_hyp_action_is_answer = True
                elif h.answer_action == DecisionAction.DEFER:
                    verified_hyp_action_is_defer = True

    return StructuralStateV3(
        has_competing_unverified_support=any(
            len(topology.unverified_support_by_hypothesis.get(h, ())) > 0
            for h in topology.hypothesis_states
            if topology.hypothesis_states[h] == HypothesisState.SUPPORTED
        ),
        n_hyp_unverified_support=sum(
            1 for h in topology.hypothesis_states
            if h in topology.unverified_support_by_hypothesis
        ),
        n_hyp_unverified_contradiction=sum(
            1 for h in topology.hypothesis_states
            if h in topology.unverified_contradiction_by_hypothesis
        ),
        can_verify=rs.get("verification_calls_remaining", 0) > 0 and has_unverified,
        verify_budget_exhausted=rs.get("verification_calls_remaining", 0) == 0,
        all_evidence_verified=topology.verification_complete,
        n_hyp_with_verified_support=topology.n_hyp_with_verified_support,
        n_hyp_with_verified_contradiction=topology.n_hyp_with_verified_contradiction,
        n_hyp_with_mixed_verified=topology.n_hyp_with_mixed_verified,
        n_viable_hypotheses=topology.n_viable_hypotheses,
        n_eliminated_hypotheses=topology.n_eliminated_hypotheses,
        has_unique_verified_supported_hypothesis=topology.has_unique_verified_supported,
        has_verified_unresolved_competition=topology.has_verified_unresolved_competition,
        verified_hyp_action_is_answer=verified_hyp_action_is_answer,
        verified_hyp_action_is_defer=verified_hyp_action_is_defer,
    )


def _cert_to_readiness(cert_answer: bool, cert_defer: bool) -> str:
    """Derive readiness from certificate outputs."""
    if cert_answer:
        return "ANSWER_READY"
    if cert_defer:
        return "DEFER_READY"
    return "CONTINUE_REQUIRED"


def _executor_truth(
    runtime: EvidenceRuntime,
    executor: EvidenceExecutor,
) -> tuple[bool | None, bool, bool | None, bool, str]:
    """Execute ANSWER and DEFER to determine executor truth.

    Returns:
        (answer_success, answer_terminal, defer_success, defer_terminal, readiness)
    """
    # Try ANSWER
    try:
        ans_result = executor.execute(runtime, DecisionAction.ANSWER)
        answer_success = bool(ans_result.task_success) if ans_result.task_success is not None else False
        answer_terminal = ans_result.terminal
    except Exception:
        answer_success = None
        answer_terminal = False

    # Try DEFER
    try:
        defer_result = executor.execute(runtime, DecisionAction.DEFER)
        defer_success = bool(defer_result.task_success) if defer_result.task_success is not None else False
        defer_terminal = defer_result.terminal
    except Exception:
        defer_success = None
        defer_terminal = False

    # Derive readiness from executor truth
    if answer_terminal and answer_success:
        readiness = "ANSWER_READY"
    elif defer_terminal and defer_success:
        readiness = "DEFER_READY"
    elif answer_terminal and not answer_success and defer_terminal and not defer_success:
        # Both terminal, neither success — state is terminal but wrong
        readiness = "DEFER_READY"  # DEFER is safer
    else:
        readiness = "CONTINUE_REQUIRED"

    return answer_success, answer_terminal, defer_success, defer_terminal, readiness


def _benchmark_truth(task: EvidenceTask, runtime: EvidenceRuntime) -> str:
    """Derive state-conditioned readiness from benchmark ground truth.

    CRITICAL: task.expected_terminal is the EVENTUAL correct terminal
    action for the whole task. It is NOT the same as the current state's
    readiness. For example, a D5 task may have expected_terminal=ANSWER
    but the initial state is CONTINUE_REQUIRED because the discriminator
    hasn't been verified yet.

    This function checks the oracle_resolution_path to determine whether
    the current state is ready for the expected terminal action, or
    whether continuation is still needed.

    Args:
        task: The evidence task
        runtime: The current runtime state

    Returns:
        State-conditioned readiness: ANSWER_READY / DEFER_READY / CONTINUE_REQUIRED
    """
    expected = task.expected_terminal
    oracle_path = getattr(task, 'oracle_resolution_path', ())

    # If the oracle path starts with the expected terminal action,
    # the state is ready for that action NOW.
    if oracle_path:
        first_step = oracle_path[0]
        if isinstance(first_step, str):
            if first_step.startswith("ANSWER") or first_step == "ANSWER":
                return "ANSWER_READY"
            if first_step.startswith("DEFER") or first_step == "DEFER":
                return "DEFER_READY"
            # VERIFY, RETRIEVE, SEARCH_MORE, REASON_MORE → continuation needed
            return "CONTINUE_REQUIRED"

    # Fallback: check if all evidence is verified and the expected
    # terminal action would succeed from this state
    visible = runtime.visible_evidence
    all_verified = all(
        ev.verification_state in (VerificationState.SUFFICIENT, VerificationState.FALSIFIED)
        for ev in visible
    ) and len(visible) > 0

    if all_verified:
        if expected == DecisionAction.ANSWER:
            return "ANSWER_READY"
        if expected == DecisionAction.DEFER:
            return "DEFER_READY"

    # If not all verified and there's unverified evidence, continuation may help
    has_unverified = any(
        ev.verification_state == VerificationState.UNVERIFIED
        for ev in visible
    )
    rs = runtime.resources.as_dict()
    can_verify = rs.get("verification_calls_remaining", 0) > 0 and has_unverified

    if can_verify:
        return "CONTINUE_REQUIRED"

    # No continuation possible — state is terminal-ready
    if expected == DecisionAction.ANSWER:
        return "ANSWER_READY"
    if expected == DecisionAction.DEFER:
        return "DEFER_READY"
    return "CONTINUE_REQUIRED"


def _causal_best_action(
    answer_success: bool | None,
    defer_success: bool | None,
    can_continue: bool,
) -> str:
    """Determine the causal best action from executor truth."""
    if answer_success:
        return "ANSWER"
    if defer_success:
        return "DEFER"
    if can_continue:
        return "CONTINUE"
    # No good option — DEFER is safer than wrong ANSWER
    return "DEFER"


def check_conformance(
    runtime: EvidenceRuntime,
    step: int,
    executor: EvidenceExecutor,
) -> ConformanceRecord:
    """Check semantic conformance for a single decision state.

    Verifies that topology, authority certificate, executor truth, and
    benchmark truth all agree on the state's readiness and best action.

    Args:
        runtime: The evidence runtime at the decision point
        step: The step number
        executor: The evidence executor (for truth checks)

    Returns:
        ConformanceRecord with all component outputs and disagreements
    """
    task = runtime.task
    visible = runtime.visible_evidence
    rs = runtime.resources.as_dict()

    # 1. Topology
    topology = derive_hypothesis_topology(
        visible,
        [h.hypothesis_id for h in task.hypotheses],
        hidden_evidence_count=len(runtime.hidden_evidence),
    )

    can_verify = rs.get("verification_calls_remaining", 0) > 0 and any(
        ev.verification_state == VerificationState.UNVERIFIED
        for ev in visible
    )
    can_retrieve = rs.get("retrieval_calls_remaining", 0) > 0 and len(runtime.hidden_evidence) > 0
    can_search = rs.get("search_calls_remaining", 0) > 0
    has_unverified = any(
        ev.verification_state == VerificationState.UNVERIFIED
        for ev in visible
    )
    has_hidden = len(runtime.hidden_evidence) > 0

    # Check if unverified evidence is discriminating
    has_unverified_discriminating = any(
        ev.verification_state == VerificationState.UNVERIFIED
        and (ev.supports or ev.contradicts)
        for ev in visible
    )

    topology_readiness = classify_terminal_readiness(
        topology,
        can_verify=can_verify,
        can_retrieve=can_retrieve,
        can_search=can_search,
        has_unverified_discriminating_evidence=has_unverified_discriminating,
        has_hidden_evidence=has_hidden,
    ).value

    # 2. Authority certificate
    structural = _build_structural_state_v3(runtime, topology)
    cert_answer = answer_structural_certificate(structural)
    cert_defer = defer_structural_certificate(structural)
    cert_readiness = _cert_to_readiness(cert_answer, cert_defer)

    # 3. Executor truth
    answer_success, answer_terminal, defer_success, defer_terminal, executor_readiness = \
        _executor_truth(runtime, executor)

    # 4. Benchmark truth
    benchmark_readiness = _benchmark_truth(task, runtime)
    correct_hyp = task.correct_hypothesis_id if hasattr(task, 'correct_hypothesis_id') else None

    # 5. Causal best action
    can_continue = can_verify or can_retrieve or can_search
    causal_best = _causal_best_action(answer_success, defer_success, can_continue)

    # 6. State hash
    state_sha = _compute_state_sha(
        task.task_id, step, visible, rs,
    )

    # 7. Check conformance
    disagreements = []

    # Topology vs certificate
    if topology_readiness != cert_readiness:
        disagreements.append(
            f"topology({topology_readiness}) != cert({cert_readiness})"
        )

    # Topology vs executor truth
    if topology_readiness != executor_readiness:
        disagreements.append(
            f"topology({topology_readiness}) != executor({executor_readiness})"
        )

    # Topology vs benchmark truth
    if topology_readiness != benchmark_readiness:
        disagreements.append(
            f"topology({topology_readiness}) != benchmark({benchmark_readiness})"
        )

    # Certificate vs executor truth
    if cert_readiness != executor_readiness:
        disagreements.append(
            f"cert({cert_readiness}) != executor({executor_readiness})"
        )

    # Certificate vs benchmark truth
    if cert_readiness != benchmark_readiness:
        disagreements.append(
            f"cert({cert_readiness}) != benchmark({benchmark_readiness})"
        )

    # Executor truth vs benchmark truth
    if executor_readiness != benchmark_readiness:
        disagreements.append(
            f"executor({executor_readiness}) != benchmark({benchmark_readiness})"
        )

    conformant = len(disagreements) == 0

    # Classify disagreement type (P1.2)
    # safe_abstention: certificate says CONTINUE_REQUIRED when others say terminal
    #   — conservative, not unsafe
    # unsafe_disagreement: certificate forces a terminal action that disagrees
    #   with topology/executor/benchmark — potentially unsafe
    if conformant:
        disagreement_type = "no_disagreement"
    elif cert_readiness == "CONTINUE_REQUIRED" and (
        topology_readiness != "CONTINUE_REQUIRED"
        or executor_readiness != "CONTINUE_REQUIRED"
        or benchmark_readiness != "CONTINUE_REQUIRED"
    ):
        disagreement_type = "safe_abstention"
    elif cert_readiness != "CONTINUE_REQUIRED" and (
        cert_readiness != topology_readiness
        or cert_readiness != executor_readiness
        or cert_readiness != benchmark_readiness
    ):
        disagreement_type = "unsafe_disagreement"
    else:
        disagreement_type = "other_disagreement"

    return ConformanceRecord(
        task_id=task.task_id,
        step=step,
        state_sha256=state_sha,
        topology_readiness=topology_readiness,
        topology_unique_supported=topology.unique_supported_hypothesis,
        topology_n_viable=topology.n_viable_hypotheses,
        topology_n_eliminated=topology.n_eliminated_hypotheses,
        cert_answer=cert_answer,
        cert_defer=cert_defer,
        cert_readiness=cert_readiness,
        executor_answer_success=answer_success,
        executor_defer_success=defer_success,
        executor_answer_terminal=answer_terminal,
        executor_defer_terminal=defer_terminal,
        executor_truth_readiness=executor_readiness,
        benchmark_expected_terminal=task.expected_terminal.value,
        benchmark_correct_hypothesis=correct_hyp,
        benchmark_truth_readiness=benchmark_readiness,
        causal_best_action=causal_best,
        can_verify=can_verify,
        can_retrieve=can_retrieve,
        can_search=can_search,
        has_unverified_evidence=has_unverified,
        has_hidden_evidence=has_hidden,
        conformant=conformant,
        disagreements=tuple(disagreements),
        disagreement_type=disagreement_type,
    )


def check_conformance_for_task(
    task: EvidenceTask,
    resources: ResourceState,
    executor: EvidenceExecutor,
    pre_verify: bool = True,
) -> list[ConformanceRecord]:
    """Check conformance at multiple decision points within a task.

    By default, checks at:
    - Initial state
    - After first VERIFY (if admissible)
    - After second VERIFY (if admissible)

    Args:
        task: The evidence task
        resources: Initial resource state
        executor: The evidence executor
        pre_verify: Whether to also check after VERIFY steps

    Returns:
        List of ConformanceRecords for each checked state
    """
    from hrm_adaptive_memory.executive.evidence_benchmark.schema import initial_evidence_runtime

    records = []
    runtime = initial_evidence_runtime(task, resources)

    # Check initial state
    records.append(check_conformance(runtime, step=0, executor=executor))

    if not pre_verify:
        return records

    # Verify up to 2 times, checking conformance after each
    for step in range(1, 3):
        valid = valid_verify_targets(runtime)
        if not valid:
            break

        rs = runtime.resources.as_dict()
        if rs.get("verification_calls_remaining", 0) <= 0:
            break

        result = executor.execute(runtime, DecisionAction.VERIFY, target_evidence_id=valid[0])
        runtime = result.runtime

        if result.terminal:
            break

        records.append(check_conformance(runtime, step=step, executor=executor))

    return records
