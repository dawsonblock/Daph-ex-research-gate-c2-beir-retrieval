"""Counterfactual fork engine for DAPH-X.

Forks a checkpoint into multiple branches, each executing a different
first action, then rolling forward with the same downstream policy.

Causal invariant:
  s_0^(1) = s_0^(2) = ... = s_0^(k)

The first action is the ONLY manipulated variable.
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from daph_x.actions.typed_actions import Action, ActionType
from daph_x.receipts.checkpoint import Checkpoint, _serialize_graph
from daph_x.world_model.transition_model import transition_model, ObservationOutcome


@dataclass(frozen=True)
class ForkResult:
    """Result of a single counterfactual fork."""
    checkpoint_hash: str
    first_action: str
    first_action_hash: str
    outcome: str
    next_state_hash: str
    terminal_outcome: str
    success: bool
    utility: float
    action_cost: float
    total_cost: float
    steps_used: int
    steps_remaining: int
    verify_remaining: int
    runtime_errors: tuple[str, ...] = ()
    downstream_policy_id: str = ""
    seed: int = 42

    def to_dict(self) -> dict:
        return {
            "checkpoint_hash": self.checkpoint_hash,
            "first_action": self.first_action,
            "first_action_hash": self.first_action_hash,
            "outcome": self.outcome,
            "next_state_hash": self.next_state_hash,
            "terminal_outcome": self.terminal_outcome,
            "success": self.success,
            "utility": self.utility,
            "action_cost": self.action_cost,
            "total_cost": self.total_cost,
            "steps_used": self.steps_used,
            "steps_remaining": self.steps_remaining,
            "verify_remaining": self.verify_remaining,
            "runtime_errors": list(self.runtime_errors),
            "downstream_policy_id": self.downstream_policy_id,
            "seed": self.seed,
        }


def fork_and_run(
    checkpoint: Checkpoint,
    first_action: Action,
    seed: int | None = None,
) -> ForkResult:
    """Fork a checkpoint and run a single action.

    Returns the result of executing first_action from the checkpoint,
    then rolling forward with the downstream policy.

    The causal quantity being estimated is:
      Q^π(s,a) = E[U(τ) | do(a_0=a), π_downstream]

    Where π_downstream is the frozen downstream policy.
    """
    if seed is None:
        seed = checkpoint.seed

    # Deep copy the graph to avoid mutation
    graph = copy.deepcopy(checkpoint.graph)

    # Validate the action is legal
    action_hash = hashlib.sha256(str(first_action).encode()).hexdigest()

    # Execute the first action
    runtime_errors = []
    try:
        transitions = transition_model(graph, first_action)
        if not transitions:
            runtime_errors.append(f"No transitions for action {first_action}")
            return _make_result(checkpoint, first_action, action_hash, "ERROR", 0.0, 0.0, runtime_errors, seed)

        # Compute expected utility over ALL stochastic outcomes:
        #   Q(s,a) = Σ_o P(o|s,a) * U(s'_o)
        # This replaces the previous argmax-probability approach which
        # systematically distorted action values when minority outcomes matter.
        outcome_utilities = []
        for t in transitions:
            u = _compute_utility(checkpoint, first_action, t.outcome.value, t.next_graph)
            outcome_utilities.append((t, u))

        expected_utility = sum(t.probability * u for t, u in outcome_utilities)

        # Use the most likely transition for state hashing and outcome reporting
        best_transition = max(transitions, key=lambda t: t.probability)
        next_graph = best_transition.next_graph
        outcome = best_transition.outcome.value

    except Exception as e:
        runtime_errors.append(f"Execution error: {e}")
        return _make_result(checkpoint, first_action, action_hash, "ERROR", 0.0, 0.0, runtime_errors, seed)

    # Compute next state hash
    next_state_hash = hashlib.sha256(
        json.dumps(_serialize_graph(next_graph), sort_keys=True, default=str).encode()
    ).hexdigest()

    # Use expected utility over all outcomes
    utility = expected_utility

    # Determine success (based on most likely outcome)
    success = _compute_success(checkpoint, first_action, outcome, next_graph)

    # Terminal outcome
    terminal_outcome = _compute_terminal_outcome(first_action, outcome, next_graph)

    return ForkResult(
        checkpoint_hash=checkpoint.checkpoint_hash,
        first_action=str(first_action),
        first_action_hash=action_hash,
        outcome=outcome,
        next_state_hash=next_state_hash,
        terminal_outcome=terminal_outcome,
        success=success,
        utility=utility,
        action_cost=first_action.expected_cost,
        total_cost=first_action.expected_cost,
        steps_used=checkpoint.graph.steps_remaining - next_graph.steps_remaining,
        steps_remaining=next_graph.steps_remaining,
        verify_remaining=next_graph.verify_remaining,
        runtime_errors=tuple(runtime_errors),
        downstream_policy_id=checkpoint.downstream_policy_id,
        seed=seed,
    )


def evaluate_all_actions(
    checkpoint: Checkpoint,
    actions: Sequence[Action],
    seed: int | None = None,
) -> list[ForkResult]:
    """Evaluate all actions from the same checkpoint.

    Returns one ForkResult per action, all from the same initial state.
    This is the core counterfactual evaluation.

    For state s, obtains:
      A_L(s) = {a_1, ..., a_n}
    Then:
      results = {(a_i, U_i)} for i=1..n
    From this:
      a* = argmax_i U_i
      Regret(s,a) = U(s,a*) - U(s,a)
    """
    if seed is None:
        seed = checkpoint.seed

    results = []
    for action in actions:
        result = fork_and_run(checkpoint, action, seed=seed)
        results.append(result)

    return results


def compute_oracle_action(results: list[ForkResult]) -> tuple[str, float]:
    """Compute the oracle action (highest utility) from fork results."""
    if not results:
        return "", 0.0
    best = max(results, key=lambda r: r.utility)
    return best.first_action, best.utility


def compute_regret(results: list[ForkResult], action_str: str) -> float:
    """Compute regret for a specific action."""
    oracle_action, oracle_utility = compute_oracle_action(results)
    for r in results:
        if r.first_action == action_str:
            return oracle_utility - r.utility
    return float('inf')  # Action not found


def compute_near_optimal_set(results: list[ForkResult], epsilon: float = 3.0) -> list[str]:
    """Compute the set of near-optimal actions."""
    if not results:
        return []
    _, oracle_utility = compute_oracle_action(results)
    return [
        r.first_action for r in results
        if oracle_utility - r.utility <= epsilon
    ]


def _make_result(
    checkpoint: Checkpoint,
    action: Action,
    action_hash: str,
    outcome: str,
    utility: float,
    cost: float,
    errors: list[str],
    seed: int,
) -> ForkResult:
    """Create a ForkResult for error cases."""
    return ForkResult(
        checkpoint_hash=checkpoint.checkpoint_hash,
        first_action=str(action),
        first_action_hash=action_hash,
        outcome=outcome,
        next_state_hash="",
        terminal_outcome="ERROR",
        success=False,
        utility=utility,
        action_cost=cost,
        total_cost=cost,
        steps_used=0,
        steps_remaining=checkpoint.graph.steps_remaining,
        verify_remaining=checkpoint.graph.verify_remaining,
        runtime_errors=tuple(errors),
        downstream_policy_id=checkpoint.downstream_policy_id,
        seed=seed,
    )


def _compute_utility(
    checkpoint: Checkpoint,
    action: Action,
    outcome: str,
    next_graph,
) -> float:
    """Compute utility for a fork result.

    Utility function is TARGET-AWARE:
      - Correct ANSWER: 100 - total_cost
      - Wrong ANSWER: -50
      - Correct DEFER: 50 - total_cost
      - Wrong DEFER: -20
      - VERIFY(e) that discriminates: 40 - cost (resolves competition)
      - VERIFY(e) that supports correct: 30 - cost
      - VERIFY(e) redundant: 5 - cost
      - VERIFY(e) irrelevant: 0 - cost
      - SEARCH/RETRIEVE: 5 - cost
      - Other: -cost
    """
    cost = action.expected_cost

    if action.action_type == ActionType.ANSWER:
        if action.target == checkpoint.correct_hypothesis_id:
            return 100.0 - cost
        else:
            return -50.0

    if action.action_type == ActionType.DEFER:
        if checkpoint.expected_terminal == "DEFER":
            return 50.0 - cost
        else:
            return -20.0

    if action.action_type == ActionType.VERIFY:
        # Target-aware VERIFY utility
        evidence_id = action.target
        if not isinstance(evidence_id, str):
            return 0.0 - cost

        # Get the evidence node from the ORIGINAL graph
        node = checkpoint.graph.nodes.get(evidence_id)
        if node is None:
            return 0.0 - cost

        # Determine what this evidence supports/contradicts
        supports = [e.target_id for e in checkpoint.graph.edges
                   if e.source_id == evidence_id and e.edge_type.value == "supports"]
        contradicts = [e.target_id for e in checkpoint.graph.edges
                      if e.source_id == evidence_id and e.edge_type.value == "contradicts"]

        # Check if verifying this evidence discriminates between hypotheses
        # (supports/contradicts a viable hypothesis)
        from daph.epistemic.topology import derive_hypothesis_topology
        evidence_items = checkpoint.graph.to_legacy_evidence_items()
        hypothesis_ids = checkpoint.graph.hypothesis_ids()
        topo = derive_hypothesis_topology(
            evidence_items=evidence_items,
            hypothesis_ids=hypothesis_ids,
        )

        # Count how many SUPPORTED hypotheses this evidence affects
        n_supported_affected = 0
        for h_id in supports + contradicts:
            if topo.hypothesis_states.get(h_id, None) is not None:
                state = topo.hypothesis_states[h_id]
                if state.value in ("SUPPORTED", "WEAKENED", "UNTESTED"):
                    n_supported_affected += 1

        # Check if this evidence supports the correct hypothesis
        supports_correct = checkpoint.correct_hypothesis_id in supports
        contradicts_correct = checkpoint.correct_hypothesis_id in contradicts

        # Utility based on target quality
        if outcome == "SUFFICIENT":
            if supports_correct:
                # Verifying evidence that supports the correct hypothesis
                return 40.0 - cost  # High value — resolves toward correct answer
            elif contradicts_correct:
                # Verifying evidence that contradicts the correct hypothesis
                return 5.0 - cost  # Low value — might mislead
            elif n_supported_affected >= 2:
                # Evidence discriminates between multiple hypotheses
                return 35.0 - cost
            elif n_supported_affected == 1:
                # Evidence affects one hypothesis
                return 20.0 - cost
            else:
                # Evidence doesn't affect any viable hypothesis
                return 5.0 - cost
        elif outcome == "FALSIFIED":
            # Falsified evidence eliminates a hypothesis
            if contradicts_correct:
                return 30.0 - cost  # Good — eliminates wrong hypothesis
            elif n_supported_affected >= 2:
                return 25.0 - cost
            else:
                return 10.0 - cost
        else:
            return 5.0 - cost

    if action.action_type in (ActionType.SEARCH, ActionType.RETRIEVE):
        return 5.0 - cost

    if action.action_type in (ActionType.COMPARE, ActionType.CHECK_CONSISTENCY):
        return 5.0 - cost

    if action.action_type == ActionType.STOP:
        return -10.0

    return -cost


def _compute_success(
    checkpoint: Checkpoint,
    action: Action,
    outcome: str,
    next_graph,
) -> bool:
    """Determine if the fork result is a success."""
    if action.action_type == ActionType.ANSWER:
        return action.target == checkpoint.correct_hypothesis_id
    if action.action_type == ActionType.DEFER:
        return checkpoint.expected_terminal == "DEFER"
    return False


def _compute_terminal_outcome(
    action: Action,
    outcome: str,
    next_graph,
) -> str:
    """Determine the terminal outcome of a fork."""
    if action.action_type in (ActionType.ANSWER, ActionType.DEFER, ActionType.STOP):
        return action.action_type.value
    return "CONTINUED"
