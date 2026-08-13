"""Canonical Markov-state projection for the V2B-I3.1 oracle.

The runtime retains complete resource and audit instrumentation.  The oracle
uses only this frozen projection: every field here can independently alter a
future transition, legality decision, or utility.  Elapsed time and monetary
cost are deliberately excluded: elapsed time is derivable from the remaining
action budgets in the current deterministic cost table and all I3.1 monetary
costs are zero.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from .metareasoning_executor import I3Runtime
from .resources import DEFAULT_ACTION_COSTS, POLICY_REJECTION_COST, ResourceState


@dataclass(frozen=True)
class OracleState:
    verification_state: str
    temporal_state: str
    conflict_state: bool
    composition_state: bool
    provenance_count: int
    retrieved: bool
    searched: bool
    steps_remaining: int
    reasoning_units_remaining: int
    retrievals_remaining: int
    verifications_remaining: int
    searches_remaining: int
    policy_rejections_used: int = 0

    def __post_init__(self) -> None:
        if min(self.steps_remaining, self.reasoning_units_remaining,
               self.retrievals_remaining, self.verifications_remaining,
               self.searches_remaining) < 0:
            raise ValueError("oracle state remaining resources must be nonnegative")

    def state_id(self) -> str:
        return hashlib.sha256(json.dumps(self.as_dict(), sort_keys=True,
                                         separators=(",", ":")).encode()).hexdigest()

    def as_dict(self) -> dict[str, object]:
        return {
            "verification_state": self.verification_state,
            "temporal_state": self.temporal_state,
            "conflict_state": self.conflict_state,
            "composition_state": self.composition_state,
            "provenance_count": self.provenance_count,
            "retrieved": self.retrieved,
            "searched": self.searched,
            "steps_remaining": self.steps_remaining,
            "reasoning_units_remaining": self.reasoning_units_remaining,
            "retrievals_remaining": self.retrievals_remaining,
            "verifications_remaining": self.verifications_remaining,
            "searches_remaining": self.searches_remaining,
            "policy_rejections_used": self.policy_rejections_used,
        }

    def capacity(self) -> tuple[int, int, int, int, int]:
        """A strict finite-horizon rank; every executable action reduces steps."""
        return (self.steps_remaining, self.reasoning_units_remaining,
                self.retrievals_remaining, self.verifications_remaining,
                self.searches_remaining)


def canonicalize_runtime_state(runtime: I3Runtime) -> OracleState:
    remaining = runtime.resources.as_dict()
    reasoning_unit = DEFAULT_ACTION_COSTS[next(action for action in DEFAULT_ACTION_COSTS
                                                if action.value == "REASON_MORE")].reasoning_tokens
    units, remainder = divmod(remaining["reasoning_tokens_remaining"], reasoning_unit)
    if remainder:
        raise ValueError("I3.1 canonical state requires 128-token reasoning units")
    return OracleState(
        verification_state=runtime.verification_state.value,
        temporal_state=runtime.temporal_status.value,
        conflict_state=runtime.unresolved_conflict,
        composition_state=runtime.composition_complete,
        provenance_count=runtime.provenance_count,
        retrieved=runtime.retrieved,
        searched=runtime.searched,
        steps_remaining=remaining["executive_steps_remaining"],
        reasoning_units_remaining=units,
        retrievals_remaining=remaining["retrieval_calls_remaining"],
        verifications_remaining=remaining["verification_calls_remaining"],
        searches_remaining=remaining["search_calls_remaining"],
        policy_rejections_used=remaining.get("policy_rejections_used", 0),
    )


def runtime_from_oracle_state(template: I3Runtime, state: OracleState) -> I3Runtime:
    """Reconstruct only the deterministic runtime data needed by policy/execution.

    The current nonterminal action set makes elapsed milliseconds derivable
    from resource use.  Reconstructing it here is intentionally explicit so
    runtime/oracle resource parity is testable.
    """
    budget = template.resources.budget
    steps_used = budget.max_executive_steps - state.steps_remaining
    reasoning_unit = DEFAULT_ACTION_COSTS[next(action for action in DEFAULT_ACTION_COSTS
                                                if action.value == "REASON_MORE")].reasoning_tokens
    tokens_used = budget.max_reasoning_tokens - state.reasoning_units_remaining * reasoning_unit
    retrieval_used = budget.max_retrieval_calls - state.retrievals_remaining
    verification_used = budget.max_verification_calls - state.verifications_remaining
    search_used = budget.max_search_calls - state.searches_remaining
    # Only nonterminal actions can precede an oracle state, therefore this is
    # uniquely determined by the used resource counters.
    from hrm_adaptive_memory.cognitive_control.core import DecisionAction
    elapsed = (
        retrieval_used * DEFAULT_ACTION_COSTS[DecisionAction.RETRIEVE].elapsed_ms
        + verification_used * DEFAULT_ACTION_COSTS[DecisionAction.VERIFY].elapsed_ms
        + search_used * DEFAULT_ACTION_COSTS[DecisionAction.SEARCH_MORE].elapsed_ms
        + (tokens_used // reasoning_unit) * DEFAULT_ACTION_COSTS[DecisionAction.REASON_MORE].elapsed_ms
        + state.policy_rejections_used * POLICY_REJECTION_COST.elapsed_ms
    )
    resources = ResourceState(
        budget, executive_steps_used=steps_used, reasoning_tokens_used=tokens_used,
        retrieval_calls_used=retrieval_used, verification_calls_used=verification_used,
        search_calls_used=search_used, elapsed_ms=elapsed, monetary_cost_microusd=0,
        policy_rejections_used=state.policy_rejections_used)
    from dataclasses import replace
    from hrm_adaptive_memory.cognitive_control.state import TemporalStatus, VerificationState
    return replace(template, resources=resources,
                   verification_state=VerificationState(state.verification_state),
                   temporal_status=TemporalStatus(state.temporal_state),
                   unresolved_conflict=state.conflict_state,
                   composition_complete=state.composition_state,
                   provenance_count=state.provenance_count,
                   retrieved=state.retrieved, searched=state.searched)
