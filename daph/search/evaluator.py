"""Branch evaluator: scores complete or partial branches.

Score(τ) = w_Q * Q_leaf + w_P * Σ γ^t * PAV_t - w_C * C_branch - w_U * U_branch

Components are logged separately for transparency.
"""
from __future__ import annotations

from daph.search.types import BranchNode, BranchResult


def evaluate_branch(
    first_action: str,
    nodes: tuple[BranchNode, ...],
    q_values: dict[str, float],
    *,
    w_q: float = 1.0,
    w_pav: float = 1.0,
    gamma: float = 0.9,
    w_cost: float = 0.01,
    w_uncertainty: float = 0.0,
) -> BranchResult:
    """Evaluate a single branch and compute its score.

    Args:
        first_action: The first action in the branch
        nodes: All nodes in the branch (in order)
        q_values: Q values at the root state (for the first action)
        w_q: Weight for Q value
        w_pav: Weight for cumulative PAV
        gamma: Discount factor for PAV
        w_cost: Weight for cumulative cost
        w_uncertainty: Weight for uncertainty penalty

    Returns:
        BranchResult with score and metadata.
    """
    if not nodes:
        return BranchResult(
            first_action=first_action,
            nodes=(),
            score=-999.0,
            terminal=False,
            success=None,
            terminal_utility=None,
            cumulative_cost=0.0,
            cumulative_pav=0.0,
            depth_reached=0,
        )

    # Q value of the first action
    q_first = q_values.get(first_action, 0.0)

    # Cumulative PAV with discount
    cumulative_pav = 0.0
    cumulative_cost = 0.0
    for t, node in enumerate(nodes):
        cumulative_pav += (gamma ** t) * node.pav_score
        cumulative_cost += node.cumulative_cost

    # Terminal info
    last_node = nodes[-1]
    terminal = last_node.terminal
    success = last_node.success
    terminal_utility = last_node.terminal_utility

    # Score: Q + discounted PAV - cost penalty
    # If terminal, use terminal utility as the primary score component
    if terminal and terminal_utility is not None:
        score = terminal_utility + w_pav * cumulative_pav - w_cost * cumulative_cost
    else:
        score = w_q * q_first + w_pav * cumulative_pav - w_cost * cumulative_cost

    return BranchResult(
        first_action=first_action,
        nodes=nodes,
        score=score,
        terminal=terminal,
        success=success,
        terminal_utility=terminal_utility,
        cumulative_cost=cumulative_cost,
        cumulative_pav=cumulative_pav,
        depth_reached=len(nodes),
    )
