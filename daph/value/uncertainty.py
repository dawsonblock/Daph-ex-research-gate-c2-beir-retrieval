"""DAPH I3.4 — Uncertainty estimation for action-value predictions.

Uses ensemble variance (random forest tree variance) as the primary
uncertainty measure. Returns Q(a) and σ(a) for each action.
"""

from __future__ import annotations

from typing import Any

from daph.value.model import RandomForestValueModel


def compute_uncertainty(
    model: RandomForestValueModel,
    phase: str,
    legal_actions: list[str],
    features: dict,
) -> dict[str, dict[str, float]]:
    """Compute Q(a) and σ(a) for each legal action using ensemble variance."""
    result = {}
    for action in legal_actions:
        mean, std = model.predict_with_uncertainty(phase, action, features)
        result[action] = {
            "mean": round(mean, 4),
            "uncertainty": round(std, 4),
            "lcb": round(mean - 1.96 * std, 4),
            "ucb": round(mean + 1.96 * std, 4),
        }
    return result


def rank_with_uncertainty(
    model: RandomForestValueModel,
    phase: str,
    legal_actions: list[str],
    features: dict,
    *,
    lambda_: float = 1.0,
) -> list[dict[str, Any]]:
    """Rank actions by LCB: Q(a) - λ·σ(a).

    This is a conservative ranking that penalizes uncertain actions.
    """
    uncertainties = compute_uncertainty(model, phase, legal_actions, features)
    ranked = []
    for action in legal_actions:
        q = uncertainties[action]["mean"]
        sigma = uncertainties[action]["uncertainty"]
        lcb = q - lambda_ * sigma
        ranked.append({
            "action": action,
            "Q": q,
            "sigma": sigma,
            "LCB": lcb,
        })
    ranked.sort(key=lambda x: -x["LCB"])
    return ranked


def build_value_packet(
    model: RandomForestValueModel,
    phase: str,
    legal_actions: list[str],
    features: dict,
    *,
    normalize: bool = True,
    epsilon: float = 1e-8,
) -> dict[str, Any]:
    """Build the P2 action-value packet for the LLM.

    Returns normalized values in [0, 1] with uncertainty.
    """
    uncertainties = compute_uncertainty(model, phase, legal_actions, features)

    if normalize:
        # Normalize means to [0, 1]
        means = {a: uncertainties[a]["mean"] for a in legal_actions}
        if means:
            min_m = min(means.values())
            max_m = max(means.values())
            range_m = max_m - min_m + epsilon
            normalized = {a: (v - min_m) / range_m for a, v in means.items()}
        else:
            normalized = {}
    else:
        normalized = {a: uncertainties[a]["mean"] for a in legal_actions}

    # Build ranking
    ranked = sorted(legal_actions, key=lambda a: -normalized.get(a, 0.0))

    return {
        "phase": phase,
        "action_value_estimates": {
            a: {
                "normalized_value": round(normalized.get(a, 0.0), 4),
                "expected_utility": round(uncertainties[a]["mean"], 4),
                "uncertainty": round(uncertainties[a]["uncertainty"], 4),
            }
            for a in legal_actions
        },
        "ranking": ranked,
    }
