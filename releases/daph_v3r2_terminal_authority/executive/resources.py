"""Explicit, deterministic resource accounting for V2B-I2 actions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from hrm_adaptive_memory.cognitive_control.actions import validate_v2b_action
from hrm_adaptive_memory.cognitive_control.core import DecisionAction


class ResourceExhausted(ValueError):
    pass


@dataclass(frozen=True)
class ResourceBudget:
    max_executive_steps: int = 12
    max_reasoning_tokens: int = 512
    max_retrieval_calls: int = 4
    max_verification_calls: int = 3
    max_search_calls: int = 3
    max_elapsed_ms: int = 10_000
    max_monetary_cost_microusd: int = 0

    def __post_init__(self) -> None:
        if any(value < 0 for value in (
            self.max_executive_steps, self.max_reasoning_tokens, self.max_retrieval_calls,
            self.max_verification_calls, self.max_search_calls, self.max_elapsed_ms,
            self.max_monetary_cost_microusd,
        )):
            raise ValueError("resource budgets must be nonnegative")


@dataclass(frozen=True)
class ActionCost:
    reasoning_tokens: int = 0
    retrieval_calls: int = 0
    verification_calls: int = 0
    search_calls: int = 0
    elapsed_ms: int = 0
    monetary_cost_microusd: int = 0

    def __post_init__(self) -> None:
        if any(value < 0 for value in (
            self.reasoning_tokens, self.retrieval_calls, self.verification_calls,
            self.search_calls, self.elapsed_ms, self.monetary_cost_microusd,
        )):
            raise ValueError("action costs must be nonnegative")


DEFAULT_ACTION_COSTS: dict[DecisionAction, ActionCost] = {
    DecisionAction.ANSWER: ActionCost(elapsed_ms=1),
    DecisionAction.RETRIEVE: ActionCost(retrieval_calls=1, elapsed_ms=5),
    DecisionAction.VERIFY: ActionCost(verification_calls=1, elapsed_ms=8),
    DecisionAction.SEARCH_MORE: ActionCost(search_calls=1, elapsed_ms=6),
    DecisionAction.REASON_MORE: ActionCost(reasoning_tokens=128, elapsed_ms=4),
    DecisionAction.DEFER: ActionCost(elapsed_ms=1),
    DecisionAction.STOP: ActionCost(elapsed_ms=1),
}

# I3.2 freezes policy feedback as a visible, bounded control event. It is not
# an executive action and therefore remains outside the seven-action contract,
# but it consumes one control step and the same minimal timing cost as a
# terminal control decision. Both runtime and information-state oracle import
# this single definition.
POLICY_REJECTION_COST = ActionCost(elapsed_ms=DEFAULT_ACTION_COSTS[DecisionAction.DEFER].elapsed_ms)


@dataclass(frozen=True)
class ResourceState:
    budget: ResourceBudget
    executive_steps_used: int = 0
    reasoning_tokens_used: int = 0
    retrieval_calls_used: int = 0
    verification_calls_used: int = 0
    search_calls_used: int = 0
    elapsed_ms: int = 0
    monetary_cost_microusd: int = 0
    policy_rejections_used: int = 0

    def can_execute(self, action: DecisionAction) -> bool:
        try:
            self.consume(action)
        except ResourceExhausted:
            return False
        return True

    def consume(self, action: DecisionAction, *,
                costs: Mapping[DecisionAction, ActionCost] | None = None) -> "ResourceState":
        action = validate_v2b_action(action)
        costs = DEFAULT_ACTION_COSTS if costs is None else costs
        cost = costs[action]
        next_state = ResourceState(
            self.budget,
            executive_steps_used=self.executive_steps_used + 1,
            reasoning_tokens_used=self.reasoning_tokens_used + cost.reasoning_tokens,
            retrieval_calls_used=self.retrieval_calls_used + cost.retrieval_calls,
            verification_calls_used=self.verification_calls_used + cost.verification_calls,
            search_calls_used=self.search_calls_used + cost.search_calls,
            elapsed_ms=self.elapsed_ms + cost.elapsed_ms,
            monetary_cost_microusd=self.monetary_cost_microusd + cost.monetary_cost_microusd,
            policy_rejections_used=self.policy_rejections_used,
        )
        if any((
            next_state.executive_steps_used > self.budget.max_executive_steps,
            next_state.reasoning_tokens_used > self.budget.max_reasoning_tokens,
            next_state.retrieval_calls_used > self.budget.max_retrieval_calls,
            next_state.verification_calls_used > self.budget.max_verification_calls,
            next_state.search_calls_used > self.budget.max_search_calls,
            next_state.elapsed_ms > self.budget.max_elapsed_ms,
            next_state.monetary_cost_microusd > self.budget.max_monetary_cost_microusd,
        )):
            raise ResourceExhausted(f"resource budget prevents {action.value}")
        return next_state

    def consume_policy_rejection(self) -> "ResourceState":
        """Record a visible denied proposal under the frozen I3.2 semantics."""
        cost = POLICY_REJECTION_COST
        next_state = ResourceState(
            self.budget, executive_steps_used=self.executive_steps_used + 1,
            reasoning_tokens_used=self.reasoning_tokens_used,
            retrieval_calls_used=self.retrieval_calls_used,
            verification_calls_used=self.verification_calls_used,
            search_calls_used=self.search_calls_used,
            elapsed_ms=self.elapsed_ms + cost.elapsed_ms,
            monetary_cost_microusd=self.monetary_cost_microusd + cost.monetary_cost_microusd)
        next_state = ResourceState(
            next_state.budget, executive_steps_used=next_state.executive_steps_used,
            reasoning_tokens_used=next_state.reasoning_tokens_used,
            retrieval_calls_used=next_state.retrieval_calls_used,
            verification_calls_used=next_state.verification_calls_used,
            search_calls_used=next_state.search_calls_used, elapsed_ms=next_state.elapsed_ms,
            monetary_cost_microusd=next_state.monetary_cost_microusd,
            policy_rejections_used=self.policy_rejections_used + 1)
        if (next_state.executive_steps_used > self.budget.max_executive_steps
                or next_state.elapsed_ms > self.budget.max_elapsed_ms
                or next_state.monetary_cost_microusd > self.budget.max_monetary_cost_microusd):
            raise ResourceExhausted("resource budget prevents policy rejection feedback")
        return next_state

    def as_dict(self) -> dict[str, int]:
        return {
            "executive_steps_used": self.executive_steps_used,
            "executive_steps_remaining": self.budget.max_executive_steps - self.executive_steps_used,
            "reasoning_tokens_used": self.reasoning_tokens_used,
            "reasoning_tokens_remaining": self.budget.max_reasoning_tokens - self.reasoning_tokens_used,
            "retrieval_calls_used": self.retrieval_calls_used,
            "retrieval_calls_remaining": self.budget.max_retrieval_calls - self.retrieval_calls_used,
            "verification_calls_used": self.verification_calls_used,
            "verification_calls_remaining": self.budget.max_verification_calls - self.verification_calls_used,
            "search_calls_used": self.search_calls_used,
            "search_calls_remaining": self.budget.max_search_calls - self.search_calls_used,
            "elapsed_ms": self.elapsed_ms,
            "elapsed_ms_remaining": self.budget.max_elapsed_ms - self.elapsed_ms,
            "monetary_cost_microusd": self.monetary_cost_microusd,
            "monetary_cost_microusd_remaining": (
                self.budget.max_monetary_cost_microusd - self.monetary_cost_microusd),
            "policy_rejections_used": self.policy_rejections_used,
        }
