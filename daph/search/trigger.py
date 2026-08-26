"""Search trigger: decide whether to invoke selective search.

Pure deterministic function. Does NOT search if:
- Terminal-ready and clear action
- Fewer than min_steps_remaining steps remain
- Only one legal action
- Q gap is large and uncertainty low

Searches if at least 2 near-optimal actions exist AND one or more:
- Resource pressure high (steps remaining <= threshold)
- Repeated-action trap detected
- Contradiction unresolved
- Irreversible/high-cost action present
- PAV signals disagree on near-optimal set
"""
from __future__ import annotations

from daph.search.types import SearchConfig, SearchTriggerResult


def decide_search(
    state_features: dict,
    near_optimal_actions: tuple[str, ...],
    pav_selected: tuple[str, ...] | None,
    pav_abstained: bool,
    q_values: dict[str, float],
    config: SearchConfig,
) -> SearchTriggerResult:
    """Decide whether to invoke selective search.

    Args:
        state_features: Controller-visible state features
        near_optimal_actions: Q epsilon near-optimal set
        pav_selected: PAV preferred set (or None if PAV not run)
        pav_abstained: Whether PAV abstained
        q_values: Q values for all legal actions
        config: Search configuration

    Returns:
        SearchTriggerResult with should_search and reasons.
    """
    reasons = []

    # --- Hard exclusions: do NOT search ---

    # Only one near-optimal action — no ambiguity
    if len(near_optimal_actions) < config.trigger_min_near_optimal:
        return SearchTriggerResult(
            should_search=False,
            reasons=("single_near_optimal",),
            config=config,
        )

    # Too few steps remaining to benefit from lookahead
    steps_remaining = state_features.get("steps_remaining", 0)
    if steps_remaining < config.min_steps_remaining:
        return SearchTriggerResult(
            should_search=False,
            reasons=("insufficient_steps",),
            config=config,
        )

    # Large Q gap — clear choice, no need to search
    if len(q_values) >= 2:
        q_sorted = sorted(q_values.values(), reverse=True)
        q_gap = q_sorted[0] - q_sorted[1]
        if q_gap > config.q_epsilon * 2:
            return SearchTriggerResult(
                should_search=False,
                reasons=("large_q_gap",),
                config=config,
            )

    # --- Trigger conditions: search if any fire ---

    # Resource pressure: few steps remaining
    if steps_remaining <= config.trigger_resource_pressure_steps:
        reasons.append("resource_pressure")

    # Repeated-action trap: same action repeated
    same_action_run = state_features.get("same_action_run_length", 0)
    if same_action_run >= config.trigger_repeated_action_run:
        reasons.append("repeated_action_trap")

    # Contradiction unresolved: contradicting evidence present
    n_contradicting = state_features.get("n_contradicting", 0)
    if n_contradicting > 0:
        reasons.append("contradiction_unresolved")

    # PAV disagreement: PAV selected a strict subset
    if pav_selected is not None and not pav_abstained:
        if len(pav_selected) < len(near_optimal_actions):
            reasons.append("pav_disagreement")

    # PAV abstained but multiple near-optimal — search might help
    if pav_abstained and len(near_optimal_actions) >= 2:
        reasons.append("pav_ambiguous_multi_candidate")

    # High-cost/irreversible action in near-optimal set
    if "ANSWER" in near_optimal_actions or "DEFER" in near_optimal_actions:
        # Terminal actions are irreversible — search if also continuation actions
        has_continuation = any(
            a in near_optimal_actions for a in ["RETRIEVE", "VERIFY", "SEARCH_MORE"]
        )
        if has_continuation:
            reasons.append("irreversible_vs_continuation_choice")

    # Hidden evidence remaining — retrieval might matter
    n_hidden = state_features.get("n_hidden_evidence", 0)
    retrieval_remaining = state_features.get("retrieval_remaining", 0)
    if n_hidden > 0 and retrieval_remaining > 0:
        reasons.append("retrieval_available_with_hidden_evidence")

    # Search if any trigger fired
    should_search = len(reasons) > 0

    return SearchTriggerResult(
        should_search=should_search,
        reasons=tuple(reasons) if reasons else ("no_trigger",),
        config=config,
    )
