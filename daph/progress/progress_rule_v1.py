"""DAPH PROGRESS_RULE_V1: Deterministic structural progress function.

Measures whether an action actually improves the epistemic situation
enough to justify its cost. This is separate from Q_CAUSAL_V1, which
measures recoverable long-horizon outcome.

Progress(s,a,s') = Phi(s') - Phi(s) - C(a)

Phi components (all controller-visible, deterministic):
  - verification_coverage: fraction of visible evidence that is verified
  - evidence_novelty: new evidence IDs exposed by this action
  - hypothesis_resolution: change in eliminated hypotheses
  - terminal_readiness: movement toward ANSWER_READY or justified DEFER
  - contradiction_resolution: change in contradicting evidence

C(a) = action_cost from MetareasoningUtility (already defined)

Key property: A repeated RETRIEVE that exposes no new evidence gets
Progress ~= -C(a) < 0, even if Qwen can still recover afterward.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hrm_adaptive_memory.cognitive_control.state import (
    VerificationState, TemporalStatus,
)
from hrm_adaptive_memory.executive.evidence_benchmark.schema import (
    EvidenceRuntime, EvidenceItem, EvidenceActionExecution,
)
from hrm_adaptive_memory.executive.metareasoning_utility import MetareasoningUtility


@dataclass(frozen=True)
class ProgressComponents:
    """Individual components of the progress score."""
    delta_verification_coverage: float
    delta_evidence_novelty: float
    delta_hypothesis_resolution: float
    delta_terminal_readiness: float
    delta_contradiction_resolution: float
    action_cost: float
    raw_progress: float  # Phi(s') - Phi(s) before cost
    progress: float  # Phi(s') - Phi(s) - C(a)
    state_changed: bool
    new_evidence_ids: tuple[str, ...]
    new_verified_ids: tuple[str, ...]

    def as_dict(self) -> dict:
        return {
            "delta_verification_coverage": round(self.delta_verification_coverage, 4),
            "delta_evidence_novelty": round(self.delta_evidence_novelty, 4),
            "delta_hypothesis_resolution": round(self.delta_hypothesis_resolution, 4),
            "delta_terminal_readiness": round(self.delta_terminal_readiness, 4),
            "delta_contradiction_resolution": round(self.delta_contradiction_resolution, 4),
            "action_cost": round(self.action_cost, 4),
            "raw_progress": round(self.raw_progress, 4),
            "progress": round(self.progress, 4),
            "state_changed": self.state_changed,
            "new_evidence_ids": list(self.new_evidence_ids),
            "new_verified_ids": list(self.new_verified_ids),
        }


def _compute_phi(runtime: EvidenceRuntime) -> dict[str, float]:
    """Compute the epistemic progress potential Phi(s).

    Returns a dict of components, each in [0, 1] or similar normalized range.
    Higher is better.
    """
    visible = runtime.visible_evidence
    n_visible = len(visible)
    n_verified = sum(1 for ev in visible
                     if ev.verification_state in (VerificationState.SUFFICIENT,
                                                   VerificationState.FALSIFIED))
    n_supporting = sum(1 for ev in visible
                       if ev.verification_state == VerificationState.SUFFICIENT
                       and ev.supports)
    n_contradicting = sum(1 for ev in visible
                          if ev.verification_state == VerificationState.FALSIFIED
                          and ev.supports)

    # Verification coverage: fraction of visible evidence that is verified
    verification_coverage = n_verified / max(n_visible, 1)

    # Evidence novelty: fraction of total evidence that is visible
    total_evidence = len(runtime.evidence)
    evidence_novelty = n_visible / max(total_evidence, 1)

    # Hypothesis resolution: how many hypotheses have been eliminated
    hyp_status = _classify_hypotheses(runtime)
    n_eliminated = sum(1 for s in hyp_status.values() if s == "ELIMINATED")
    n_total = len(hyp_status)
    hypothesis_resolution = n_eliminated / max(n_total, 1)

    # Terminal readiness: is the state close to a justified terminal action?
    terminal_readiness = _compute_terminal_readiness(runtime)

    # Contradiction resolution: fraction of contradicting evidence that is verified
    # (resolving contradictions is progress)
    contradiction_resolution = n_contradicting / max(n_visible, 1)

    return {
        "verification_coverage": verification_coverage,
        "evidence_novelty": evidence_novelty,
        "hypothesis_resolution": hypothesis_resolution,
        "terminal_readiness": terminal_readiness,
        "contradiction_resolution": contradiction_resolution,
    }


def _classify_hypotheses(runtime: EvidenceRuntime) -> dict[str, str]:
    """Classify each hypothesis as LIVE, ELIMINATED, or UNTESTED."""
    visible = runtime.visible_evidence
    status = {}
    for hyp in runtime.task.hypotheses:
        has_support = any(
            ev.verification_state == VerificationState.SUFFICIENT
            and hyp.hypothesis_id in ev.supports
            for ev in visible
        )
        has_contradiction = any(
            ev.verification_state == VerificationState.FALSIFIED
            and hyp.hypothesis_id in ev.supports
            for ev in visible
        )
        if has_contradiction and not has_support:
            status[hyp.hypothesis_id] = "ELIMINATED"
        elif has_support:
            status[hyp.hypothesis_id] = "LIVE"
        else:
            status[hyp.hypothesis_id] = "UNTESTED"
    return status


def _compute_terminal_readiness(runtime: EvidenceRuntime) -> float:
    """Compute how close the state is to a justified terminal action.

    Returns a float in [0, 1]:
    - 0.0: no evidence, far from any terminal
    - 0.5: some evidence but not conclusive
    - 1.0: ready to ANSWER or DEFER with justification
    """
    visible = runtime.visible_evidence
    if not visible:
        return 0.0

    hyp_status = _classify_hypotheses(runtime)
    n_total = len(hyp_status)
    n_eliminated = sum(1 for s in hyp_status.values() if s == "ELIMINATED")
    n_live = sum(1 for s in hyp_status.values() if s == "LIVE")

    # If exactly one hypothesis is live and others are eliminated -> ready to ANSWER
    if n_live == 1 and n_eliminated == n_total - 1:
        # Check if the live hypothesis has sufficient supporting evidence
        for hyp_id, status in hyp_status.items():
            if status == "LIVE":
                has_sufficient = any(
                    ev.verification_state == VerificationState.SUFFICIENT
                    and ev.temporal_status == TemporalStatus.CURRENT
                    and hyp_id in ev.supports
                    for ev in visible
                )
                if has_sufficient:
                    return 1.0
                return 0.7  # Live but not fully verified

    # If all hypotheses eliminated -> ready to DEFER
    if n_eliminated == n_total and n_total > 0:
        return 0.8

    # Partial progress
    if n_eliminated > 0:
        return 0.3 + 0.2 * (n_eliminated / n_total)

    return 0.1


def compute_progress(
    runtime_before: EvidenceRuntime,
    action_result: EvidenceActionExecution,
    utility: MetareasoningUtility,
    weights: dict[str, float] | None = None,
) -> ProgressComponents:
    """Compute the progress score for a single action execution.

    Progress(s,a,s') = Phi(s') - Phi(s) - C(a)

    Args:
        runtime_before: The runtime state before the action
        action_result: The execution result (contains runtime_after)
        utility: The utility function for computing action cost
        weights: Optional weight overrides for Phi components

    Returns:
        ProgressComponents with all individual deltas and the final score.
    """
    if weights is None:
        # Default weights: normalized components, simple fixed coefficients
        weights = {
            "verification_coverage": 1.0,
            "evidence_novelty": 1.0,
            "hypothesis_resolution": 1.5,
            "terminal_readiness": 2.0,
            "contradiction_resolution": 0.5,
        }

    runtime_after = action_result.runtime

    phi_before = _compute_phi(runtime_before)
    phi_after = _compute_phi(runtime_after)

    # Compute deltas
    delta_vc = phi_after["verification_coverage"] - phi_before["verification_coverage"]
    delta_en = phi_after["evidence_novelty"] - phi_before["evidence_novelty"]
    delta_hr = phi_after["hypothesis_resolution"] - phi_before["hypothesis_resolution"]
    delta_tr = phi_after["terminal_readiness"] - phi_before["terminal_readiness"]
    delta_cr = phi_after["contradiction_resolution"] - phi_before["contradiction_resolution"]

    # Weighted raw progress
    raw_progress = (
        weights["verification_coverage"] * delta_vc
        + weights["evidence_novelty"] * delta_en
        + weights["hypothesis_resolution"] * delta_hr
        + weights["terminal_readiness"] * delta_tr
        + weights["contradiction_resolution"] * delta_cr
    )

    # Action cost
    cost = utility.action_cost(runtime_before.resources, runtime_after.resources)

    # Final progress = raw progress - cost (normalized)
    # Cost is in utility units (~10-20 per action), raw_progress is in [0,1] range
    # We need to normalize: divide cost by a scale factor to make them comparable
    # A typical action costs ~10-15 utility points
    # A meaningful progress delta is ~0.1-0.3
    # So we scale: progress = raw_progress - cost / 100
    # This means: an action that costs 10 and produces no progress gets -0.10
    # An action that costs 10 and produces 0.3 progress gets +0.20
    cost_normalized = cost / 100.0
    progress = raw_progress - cost_normalized

    # Did the state actually change?
    state_changed = (
        len(action_result.evidence_exposed) > 0
        or len(action_result.evidence_verified) > 0
        or runtime_after.searched != runtime_before.searched
        or runtime_after.reasoning_complete != runtime_before.reasoning_complete
    )

    return ProgressComponents(
        delta_verification_coverage=delta_vc,
        delta_evidence_novelty=delta_en,
        delta_hypothesis_resolution=delta_hr,
        delta_terminal_readiness=delta_tr,
        delta_contradiction_resolution=delta_cr,
        action_cost=cost,
        raw_progress=raw_progress,
        progress=progress,
        state_changed=state_changed,
        new_evidence_ids=action_result.evidence_exposed,
        new_verified_ids=action_result.evidence_verified,
    )


def compute_progress_from_features(
    sf_before: dict,
    sf_after: dict,
    action: str,
    action_cost: float = 10.0,
    weights: dict[str, float] | None = None,
) -> ProgressComponents:
    """Compute progress from state feature dicts (for offline labeling).

    This is used when we don't have full runtime objects, just the
    computed state features from checkpoints.
    """
    if weights is None:
        weights = {
            "verification_coverage": 1.0,
            "evidence_novelty": 1.0,
            "hypothesis_resolution": 1.5,
            "terminal_readiness": 2.0,
            "contradiction_resolution": 0.5,
        }

    # Compute Phi from features
    def phi_from_sf(sf: dict) -> dict[str, float]:
        n_visible = sf.get("n_visible_evidence", 0)
        n_verified = sf.get("n_verified", 0)
        n_supporting = sf.get("n_supporting", 0)
        n_contradicting = sf.get("n_contradicting", 0)
        n_total_hyps = sf.get("n_total_hypotheses", 0)
        n_eliminated = sf.get("n_eliminated", 0)
        n_live = sf.get("n_live", 0)

        verification_coverage = n_verified / max(n_visible, 1)
        # Evidence novelty: we can't compute total evidence from features,
        # but we can use n_visible as a proxy
        evidence_novelty = n_visible / 10.0  # approximate normalization
        hypothesis_resolution = n_eliminated / max(n_total_hyps, 1)

        # Terminal readiness from features
        if n_live == 1 and n_eliminated == n_total_hyps - 1 and n_supporting > 0:
            terminal_readiness = 1.0
        elif n_live == 1 and n_eliminated == n_total_hyps - 1:
            terminal_readiness = 0.7
        elif n_eliminated == n_total_hyps and n_total_hyps > 0:
            terminal_readiness = 0.8
        elif n_eliminated > 0:
            terminal_readiness = 0.3 + 0.2 * (n_eliminated / max(n_total_hyps, 1))
        else:
            terminal_readiness = 0.1

        contradiction_resolution = n_contradicting / max(n_visible, 1)

        return {
            "verification_coverage": min(verification_coverage, 1.0),
            "evidence_novelty": min(evidence_novelty, 1.0),
            "hypothesis_resolution": min(hypothesis_resolution, 1.0),
            "terminal_readiness": min(terminal_readiness, 1.0),
            "contradiction_resolution": min(contradiction_resolution, 1.0),
        }

    phi_before = phi_from_sf(sf_before)
    phi_after = phi_from_sf(sf_after)

    delta_vc = phi_after["verification_coverage"] - phi_before["verification_coverage"]
    delta_en = phi_after["evidence_novelty"] - phi_before["evidence_novelty"]
    delta_hr = phi_after["hypothesis_resolution"] - phi_before["hypothesis_resolution"]
    delta_tr = phi_after["terminal_readiness"] - phi_before["terminal_readiness"]
    delta_cr = phi_after["contradiction_resolution"] - phi_before["contradiction_resolution"]

    raw_progress = (
        weights["verification_coverage"] * delta_vc
        + weights["evidence_novelty"] * delta_en
        + weights["hypothesis_resolution"] * delta_hr
        + weights["terminal_readiness"] * delta_tr
        + weights["contradiction_resolution"] * delta_cr
    )

    cost_normalized = action_cost / 100.0
    progress = raw_progress - cost_normalized

    state_changed = (
        sf_before.get("n_visible_evidence", 0) != sf_after.get("n_visible_evidence", 0)
        or sf_before.get("n_verified", 0) != sf_after.get("n_verified", 0)
        or sf_before.get("n_supporting", 0) != sf_after.get("n_supporting", 0)
        or sf_before.get("n_eliminated", 0) != sf_after.get("n_eliminated", 0)
        or sf_before.get("searched", False) != sf_after.get("searched", False)
    )

    return ProgressComponents(
        delta_verification_coverage=delta_vc,
        delta_evidence_novelty=delta_en,
        delta_hypothesis_resolution=delta_hr,
        delta_terminal_readiness=delta_tr,
        delta_contradiction_resolution=delta_cr,
        action_cost=action_cost,
        raw_progress=raw_progress,
        progress=progress,
        state_changed=state_changed,
        new_evidence_ids=(),
        new_verified_ids=(),
    )
