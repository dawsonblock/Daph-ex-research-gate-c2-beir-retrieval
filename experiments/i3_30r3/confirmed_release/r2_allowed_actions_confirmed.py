#!/usr/bin/env python3
"""
R2 Allowed-Action Computation — Pure Logic.

This module is part of the scientific intervention. It contains ONLY pure logic:
no backend, no executor mutation, no counterfactual simulator.

Architecture:
    Legal(a, s)                  = executable under budgets, targets, and runtime
    EpistemicallyAdmissible(a, s) = allowed by public epistemic structure
    Allowed(a, s)                = Legal(a, s) ∩ EpistemicallyAdmissible(a, s)

EpistemicallyAdmissible is defined independently of Legal (not as Legal AND ...),
otherwise legality would be applied twice. The intersection combines them.

For VERIFY under R2d:
    EpistemicallyAdmissible(VERIFY, s) = ¬T2(s)

where T2(s) = (|H| > 0 ∧ ∀h ∈ H, status(h) = ELIMINATED).
"""

from __future__ import annotations

from dataclasses import dataclass


# Frozen seven-action vocabulary
ACTION_VOCABULARY = frozenset({
    "ANSWER",
    "RETRIEVE",
    "VERIFY",
    "SEARCH_MORE",
    "REASON_MORE",
    "DEFER",
    "STOP",
})

# Actions that are always legal (not budget-constrained)
ALWAYS_LEGAL = frozenset({"ANSWER", "DEFER", "STOP", "REASON_MORE"})

# Gate reason constants
GATE_REASON_ALL_ELIMINATED = "ALL_HYPOTHESES_ELIMINATED"


class EmptyAllowedActionSet(Exception):
    """Raised when the allowed action set is empty before a model call.

    This is a scientific invariant violation. ANSWER, DEFER, STOP, and
    REASON_MORE are always legal (when steps remain) and always epistemically
    admissible under current rules, so this should never fire. If it does,
    the run must abort.
    """
    def __init__(self, state: "ActionState", arm: "R2Arm"):
        self.state = state
        self.arm = arm
        super().__init__(
            f"Empty allowed action set for arm={arm.name} "
            f"t2={state.t2} steps_remaining={state.executive_steps_remaining}"
        )


@dataclass(frozen=True)
class R2Arm:
    """R2 arm configuration.

    structural_verify_gate: R2d — remove VERIFY from Allowed at T2
    corrected_t2_semantics: R2e — relabel NEEDS_DISCRIMINATION → NO_VIABLE_HYPOTHESIS at T2
    """
    name: str
    structural_verify_gate: bool
    corrected_t2_semantics: bool


# The four arms of the 2×2 factorial
C0 = R2Arm("C0", False, False)  # current behavior (baseline control)
D  = R2Arm("D",  True,  False)  # structural gate only
E  = R2Arm("E",  False, True)   # semantics correction only
DE = R2Arm("DE", True,  True)   # both interventions

ALL_ARMS = (C0, D, E, DE)


@dataclass(frozen=True)
class ActionState:
    """Visible structural state for allowed-action computation.

    All fields are derivable from the controller-visible EvidenceSnapshot
    and runtime budget state. No hidden task effects.
    """
    t2: bool
    executive_steps_remaining: int
    can_retrieve: bool
    can_search: bool
    can_verify: bool


@dataclass(frozen=True)
class AllowedActionDecision:
    """Result of allowed-action computation.

    legal:                        actions executable under budgets/targets/runtime
    epistemically_admissible:     actions allowed by public epistemic structure
    allowed:                      legal ∩ epistemically_admissible (the actual action set)
    verify_gate_condition_active: did the R2d structural rule condition fire?
                                  (arm.structural_verify_gate AND state.t2)
    verify_removed_by_epistemic_gate: did R2d actually change the legal action set?
                                     (condition active AND VERIFY was legal AND
                                      VERIFY not in allowed)
    verify_gate_reason:           why VERIFY was gated (or None)
    """
    legal: frozenset[str]
    epistemically_admissible: frozenset[str]
    allowed: frozenset[str]
    verify_gate_condition_active: bool
    verify_removed_by_epistemic_gate: bool
    verify_gate_reason: str | None


def compute_legal_actions(state: ActionState) -> frozenset[str]:
    """Compute legally executable actions from budgets and visible state.

    Legal(a, s) = executable under budgets, targets, and runtime.
    Returns empty set if no executive steps remain (fail closed).
    """
    if state.executive_steps_remaining <= 0:
        return frozenset()
    legal = set(ALWAYS_LEGAL)
    if state.can_retrieve:
        legal.add("RETRIEVE")
    if state.can_search:
        legal.add("SEARCH_MORE")
    if state.can_verify:
        legal.add("VERIFY")
    return frozenset(legal)


def compute_epistemically_admissible_actions(
    state: ActionState,
    arm: R2Arm,
) -> tuple[frozenset[str], bool, str | None]:
    """Compute epistemically admissible actions from public structural state.

    EpistemicallyAdmissible(a, s) = allowed by public epistemic structure.
    Defined independently of Legal.

    Returns (admissible_set, verify_gate_condition_active, verify_gate_reason).

    verify_gate_condition_active = arm.structural_verify_gate AND state.t2
    This is True even if VERIFY was already illegal (e.g. budget exhausted).
    """
    actions = set(ACTION_VOCABULARY)

    verify_gate_condition_active = False
    verify_gate_reason: str | None = None

    if arm.structural_verify_gate and state.t2:
        actions.discard("VERIFY")
        verify_gate_condition_active = True
        verify_gate_reason = GATE_REASON_ALL_ELIMINATED

    return frozenset(actions), verify_gate_condition_active, verify_gate_reason


def compute_allowed_actions(state: ActionState, arm: R2Arm) -> AllowedActionDecision:
    """Compute the full allowed-action decision.

    Allowed(a, s) = Legal(a, s) ∩ EpistemicallyAdmissible(a, s)

    Raises EmptyAllowedActionSet if the intersection is empty.
    """
    legal = compute_legal_actions(state)
    epistemic, verify_gate_condition_active, verify_gate_reason = (
        compute_epistemically_admissible_actions(state, arm)
    )
    allowed = legal & epistemic

    # Did R2d actually change the legal action set?
    # This is True only when: condition active AND VERIFY was legal AND
    # VERIFY is not in allowed (i.e. the epistemic gate removed it).
    verify_removed_by_epistemic_gate = (
        verify_gate_condition_active
        and "VERIFY" in legal
        and "VERIFY" not in allowed
    )

    if not allowed:
        raise EmptyAllowedActionSet(state, arm)

    return AllowedActionDecision(
        legal=legal,
        epistemically_admissible=epistemic,
        allowed=allowed,
        verify_gate_condition_active=verify_gate_condition_active,
        verify_removed_by_epistemic_gate=verify_removed_by_epistemic_gate,
        verify_gate_reason=verify_gate_reason,
    )


def allowed_actions_sha256(allowed: frozenset[str]) -> str:
    """Compute a deterministic SHA256 over the allowed action set."""
    import hashlib
    return hashlib.sha256(
        "|".join(sorted(allowed)).encode()
    ).hexdigest()
