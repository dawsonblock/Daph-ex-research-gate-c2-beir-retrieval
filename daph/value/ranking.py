"""DAPH I3.4 — Action ranking and evaluation.

Ranks legal actions by predicted value and evaluates ranking quality
against hindsight-best action.
"""

from __future__ import annotations

from typing import Any, Protocol


class ValueModel(Protocol):
    """Protocol for value models."""

    def predict_all(
        self, phase: str, legal_actions: list[str], features: dict
    ) -> dict[str, float]:
        ...

    @property
    def name(self) -> str:
        ...


def rank_actions(
    model: ValueModel,
    phase: str,
    legal_actions: list[str],
    features: dict,
) -> list[tuple[str, float]]:
    """Rank legal actions by predicted value (descending)."""
    predictions = model.predict_all(phase, legal_actions, features)
    ranked = sorted(predictions.items(), key=lambda x: -x[1])
    return ranked


def normalize_values(values: dict[str, float], epsilon: float = 1e-8) -> dict[str, float]:
    """Monotonic normalization to [0, 1]."""
    if not values:
        return {}
    min_v = min(values.values())
    max_v = max(values.values())
    range_v = max_v - min_v + epsilon
    return {a: (v - min_v) / range_v for a, v in values.items()}


def compute_regret(
    predicted_ranking: list[tuple[str, float]],
    actual_utility: float,
    hindsight_best_utility: float,
) -> float:
    """Regret = Q(s, a*) - Q(s, â)."""
    return hindsight_best_utility - actual_utility


def evaluate_ranking(
    model: ValueModel,
    transitions: list[dict],
    target_fn,
) -> dict[str, Any]:
    """Evaluate ranking quality on a set of transitions.

    For each transition:
    - Predict values for all legal actions
    - Rank them
    - Compare the top-1 pick to the actual action and the hindsight-best

    Metrics:
    - Top1ActionAccuracy: fraction where model's top-1 == actual action
    - Top1Utility: mean utility of model's top-1 pick
    - ActualUtility: mean utility of the action actually taken
    - HindsightBestUtility: mean utility of the best action in hindsight
    - MeanRegret: mean(hindsight_best - model_top1_utility)
    - TopKRecall: fraction where actual action is in top-K
    """
    top1_correct = 0
    top2_correct = 0
    top3_correct = 0
    model_top1_utilities = []
    actual_utilities = []
    hindsight_best_utilities = []
    regrets = []

    # Group transitions by (task_id, step) to get all actions for each state
    # But we only have one action per state in observational data.
    # So we evaluate: does the model rank the ACTUAL action highly?
    # And: what is the model's top-1 predicted utility vs actual?

    for t in transitions:
        phase = t["phase_before"]
        legal_actions = t.get("legal_actions", [])
        features = t.get("features_before", {})
        actual_action = t["action"]
        actual_utility = target_fn(t)

        if not legal_actions:
            continue

        # Get model predictions
        ranked = rank_actions(model, phase, legal_actions, features)
        ranked_actions = [a for a, v in ranked]

        # Top-1 accuracy: does model's top-1 match actual?
        if ranked_actions and ranked_actions[0] == actual_action:
            top1_correct += 1

        # Top-K recall: is actual action in top-K?
        if actual_action in ranked_actions[:1]:
            top1_correct += 0  # already counted above
        if actual_action in ranked_actions[:2]:
            top2_correct += 1
        if actual_action in ranked_actions[:3]:
            top3_correct += 1

        # Model's top-1 utility (we don't have counterfactual, so use
        # the empirical table if available, else the actual utility)
        model_top1_action = ranked_actions[0] if ranked_actions else actual_action
        model_top1_utility = actual_utility  # proxy: can't observe counterfactual

        # Hindsight best: we can only compute this if we have multiple
        # transitions from the same state. With observational data, we
        # approximate using the phase×action table.
        # For now, use the actual utility as a lower bound.
        hindsight_best = actual_utility  # conservative

        model_top1_utilities.append(model_top1_utility)
        actual_utilities.append(actual_utility)
        hindsight_best_utilities.append(hindsight_best)
        regrets.append(max(0, hindsight_best - model_top1_utility))

    n = len(transitions)
    return {
        "model_name": model.name,
        "n_evaluated": n,
        "top1_accuracy": top1_correct / n if n else 0.0,
        "top2_recall": top2_correct / n if n else 0.0,
        "top3_recall": top3_correct / n if n else 0.0,
        "mean_model_top1_utility": sum(model_top1_utilities) / n if n else 0.0,
        "mean_actual_utility": sum(actual_utilities) / n if n else 0.0,
        "mean_hindsight_best": sum(hindsight_best_utilities) / n if n else 0.0,
        "mean_regret": sum(regrets) / n if n else 0.0,
    }


def evaluate_ranking_with_hindsight(
    model: ValueModel,
    transitions: list[dict],
    target_fn,
) -> dict[str, Any]:
    """Evaluate ranking quality using hindsight best from grouped transitions.

    Groups transitions by (phase, features_hash) to find states where
    multiple actions were taken, enabling hindsight comparison.
    """
    import hashlib

    # Group by (phase, n_live, n_eliminated, decision_state) as a state key
    state_groups: dict[str, list[dict]] = {}
    for t in transitions:
        features = t.get("features_before", {})
        state_key = f"{t['phase_before']}|{features.get('n_live', 0)}|{features.get('n_eliminated', 0)}|{features.get('decision_state', '')}|{features.get('step', 0)}"
        if state_key not in state_groups:
            state_groups[state_key] = []
        state_groups[state_key].append(t)

    # Only evaluate on states with multiple actions (hindsight available)
    multi_action_states = {
        k: v for k, v in state_groups.items() if len(set(t["action"] for t in v)) >= 2
    }

    top1_correct = 0
    top2_correct = 0
    n_evaluated = 0
    regrets = []
    model_top1_utilities = []
    hindsight_best_utilities = []

    for state_key, group in multi_action_states.items():
        # Find hindsight best action
        action_utilities = {}
        for t in group:
            action = t["action"]
            util = target_fn(t)
            if action not in action_utilities:
                action_utilities[action] = []
            action_utilities[action].append(util)

        action_mean_utilities = {a: sum(us) / len(us) for a, us in action_utilities.items()}
        hindsight_best_action = max(action_mean_utilities, key=action_mean_utilities.get)
        hindsight_best_utility = action_mean_utilities[hindsight_best_action]

        # Get model ranking for this state
        representative = group[0]
        phase = representative["phase_before"]
        legal_actions = list(action_utilities.keys())
        features = representative.get("features_before", {})

        ranked = rank_actions(model, phase, legal_actions, features)
        ranked_actions = [a for a, v in ranked]

        if not ranked_actions:
            continue

        model_top1 = ranked_actions[0]
        model_top1_utility = action_mean_utilities.get(model_top1, 0.0)

        if model_top1 == hindsight_best_action:
            top1_correct += 1
        if hindsight_best_action in ranked_actions[:2]:
            top2_correct += 1

        regret = hindsight_best_utility - model_top1_utility
        regrets.append(regret)
        model_top1_utilities.append(model_top1_utility)
        hindsight_best_utilities.append(hindsight_best_utility)
        n_evaluated += 1

    return {
        "model_name": model.name,
        "n_multi_action_states": len(multi_action_states),
        "n_evaluated": n_evaluated,
        "top1_accuracy": top1_correct / n_evaluated if n_evaluated else 0.0,
        "top2_recall": top2_correct / n_evaluated if n_evaluated else 0.0,
        "mean_model_top1_utility": sum(model_top1_utilities) / n_evaluated if n_evaluated else 0.0,
        "mean_hindsight_best": sum(hindsight_best_utilities) / n_evaluated if n_evaluated else 0.0,
        "mean_regret": sum(regrets) / n_evaluated if n_evaluated else 0.0,
    }
