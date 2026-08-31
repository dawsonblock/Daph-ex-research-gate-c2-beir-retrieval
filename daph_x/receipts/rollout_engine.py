"""Multi-step counterfactual rollout engine for DAPH-X M4.

Implements true Q^π(s,a) = E[U(τ) | do(a_0=a), π_downstream].

Unlike the one-step fork engine, this executes the full trajectory:
  checkpoint → force first action → enumerate/sample observation
  → update graph/belief/resources → invoke frozen downstream policy
  → execute next action → repeat until terminal or horizon

For small synthetic state spaces, enumerates outcome branches exactly:
  Q^π(s,a) = Σ_τ P(τ|s,a,π) U(τ)

The first-action intervention is the ONLY manipulated variable.
Everything else (checkpoint, RNG, world model, budget, downstream policy)
remains identical across forks.
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
from daph_x.graph.epistemic_graph import (
    EpistemicGraph, GraphNode, GraphEdge, NodeType, EdgeType,
)
from daph_x.receipts.checkpoint import Checkpoint, _serialize_graph
from daph_x.world_model.transition_model import (
    transition_model, ObservationOutcome, Transition,
)
from daph_x.actions.candidate_generator import generate_and_prune
from daph_x.benchmark.novelty_signatures import compute_all_signatures


@dataclass(frozen=True)
class TrajectoryStep:
    """One step in a rollout trajectory."""
    step_index: int
    action_str: str
    action_type: str
    outcome: str
    outcome_probability: float
    state_hash_before: str
    state_hash_after: str
    step_cost: float
    is_first_action: bool
    is_terminal: bool
    terminal_reason: str


@dataclass(frozen=True)
class RolloutResult:
    """Result of a multi-step counterfactual rollout.

    This is RolloutResultV1 — the full causal record of executing
    a first action from a checkpoint, then rolling forward with
    the frozen downstream policy until terminal or horizon.
    """
    # Identity
    checkpoint_hash: str
    runtime_hash: str
    topology_signature: str
    mechanism_signature: str

    # First action (the manipulated variable)
    first_action: str
    first_action_hash: str
    first_action_type: str

    # Trajectory
    trajectory: tuple[TrajectoryStep, ...]
    observation_path: tuple[str, ...]
    terminal_reason: str  # "ANSWER", "DEFER", "STOP", "RESOURCE_EXHAUSTION", "HORIZON", "RUNTIME_ERROR"
    terminal_state_hash: str

    # Outcome
    utility: float
    total_cost: float
    steps_used: int
    success: bool

    # Provenance
    downstream_policy_id: str
    downstream_policy_hash: str
    world_model_hash: str
    seed: int

    # Errors (kept separate from utility)
    runtime_errors: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "checkpoint_hash": self.checkpoint_hash,
            "runtime_hash": self.runtime_hash,
            "topology_signature": self.topology_signature,
            "mechanism_signature": self.mechanism_signature,
            "first_action": self.first_action,
            "first_action_hash": self.first_action_hash,
            "first_action_type": self.first_action_type,
            "trajectory": [
                {
                    "step_index": s.step_index,
                    "action_str": s.action_str,
                    "action_type": s.action_type,
                    "outcome": s.outcome,
                    "outcome_probability": s.outcome_probability,
                    "state_hash_before": s.state_hash_before,
                    "state_hash_after": s.state_hash_after,
                    "step_cost": s.step_cost,
                    "is_first_action": s.is_first_action,
                    "is_terminal": s.is_terminal,
                    "terminal_reason": s.terminal_reason,
                }
                for s in self.trajectory
            ],
            "observation_path": list(self.observation_path),
            "terminal_reason": self.terminal_reason,
            "terminal_state_hash": self.terminal_state_hash,
            "utility": self.utility,
            "total_cost": self.total_cost,
            "steps_used": self.steps_used,
            "success": self.success,
            "downstream_policy_id": self.downstream_policy_id,
            "downstream_policy_hash": self.downstream_policy_hash,
            "world_model_hash": self.world_model_hash,
            "seed": self.seed,
            "runtime_errors": list(self.runtime_errors),
        }


class DownstreamPolicy:
    """Frozen downstream policy for multi-step rollouts.

    This is a simple deterministic policy that:
    - ANSWERs if there's a unique supported hypothesis with ANSWER action
    - DEFERs if there's competing support or no clear answer
    - VERIFYs unverified evidence if budget allows and it could discriminate
    - STOPs if resources are exhausted

    The policy is frozen — its behavior never changes across forks.
    """

    POLICY_VERSION = "downstream_v1_frozen"

    def __init__(self):
        self.policy_hash = hashlib.sha256(
            self.POLICY_VERSION.encode()
        ).hexdigest()[:16]

    def select_action(self, graph: EpistemicGraph) -> Action | None:
        """Select the next action according to the frozen downstream policy."""
        from daph.epistemic.topology import derive_hypothesis_topology
        from daph.epistemic.types import TerminalReadiness

        hypothesis_ids = graph.hypothesis_ids()
        evidence_items = graph.to_legacy_evidence_items()
        topology = derive_hypothesis_topology(
            evidence_items=evidence_items,
            hypothesis_ids=hypothesis_ids,
        )

        # If unique supported hypothesis with ANSWER action → ANSWER
        if topology.unique_supported_hypothesis:
            hyp_id = topology.unique_supported_hypothesis
            hyp_node = graph.nodes.get(hyp_id)
            if hyp_node and hyp_node.answer_action == "ANSWER":
                from daph_x.actions.typed_actions import answer
                return answer(hyp_id)

        # If readiness is ANSWER_READY but no unique support (shouldn't happen)
        # If readiness is DEFER_READY → DEFER
        from daph.epistemic.topology import classify_terminal_readiness
        readiness = classify_terminal_readiness(
            topology,
            can_verify=graph.verify_remaining > 0 and topology.unverified_evidence_exists,
            can_retrieve=graph.retrieve_remaining > 0,
            can_search=graph.search_remaining > 0,
            has_unverified_discriminating_evidence=topology.unverified_evidence_exists,
            has_hidden_evidence=False,
            search_could_discriminate=graph.search_remaining > 0,
        )

        if readiness == TerminalReadiness.DEFER_READY:
            from daph_x.actions.typed_actions import defer
            return defer("downstream_defer")

        # If we can verify and there's unverified evidence → VERIFY the first one
        if graph.verify_remaining > 0 and topology.unverified_evidence_exists:
            for nid, node in sorted(graph.nodes.items()):
                if (node.node_type == NodeType.EVIDENCE
                        and node.verification_state == "UNVERIFIED"):
                    from daph_x.actions.typed_actions import verify
                    return verify(nid)

        # If no verify budget or no unverified evidence → DEFER
        from daph_x.actions.typed_actions import defer
        return defer("downstream_no_action")


def _compute_rollout_utility(
    checkpoint: Checkpoint,
    first_action: Action,
    trajectory: list[TrajectoryStep],
    final_graph: EpistemicGraph,
) -> float:
    """Compute utility for a completed rollout trajectory.

    Utility is based on the terminal outcome:
      - Correct ANSWER: 100 - total_cost
      - Wrong ANSWER: -50
      - Correct DEFER (expected_terminal == DEFER): 50 - total_cost
      - Wrong DEFER (expected_terminal != DEFER): -20 - total_cost
      - STOP: -10
      - Resource/horizon exhaustion: -5 - total_cost
      - Runtime error: 0 (recorded separately, not as utility)
    """
    total_cost = sum(s.step_cost for s in trajectory)

    if not trajectory:
        return 0.0

    last_step = trajectory[-1]
    terminal_reason = last_step.terminal_reason

    if terminal_reason == "RUNTIME_ERROR":
        return 0.0  # Errors get utility 0 but are tracked separately

    if terminal_reason == "ANSWER":
        # Check if the answered hypothesis is correct
        # The ANSWER action's target is the hypothesis ID
        if first_action.action_type == ActionType.ANSWER:
            answered_hyp = first_action.target
        else:
            # Find the ANSWER step in the trajectory
            answered_hyp = None
            for step in trajectory:
                if step.action_type == "ANSWER":
                    # Parse the action string to get the target
                    # Format: "ANSWER(H1)"
                    action_str = step.action_str
                    if "(" in action_str and ")" in action_str:
                        answered_hyp = action_str.split("(")[1].rstrip(")")
                    break

        if answered_hyp == checkpoint.correct_hypothesis_id:
            return 100.0 - total_cost
        else:
            return -50.0

    if terminal_reason == "DEFER":
        if checkpoint.expected_terminal == "DEFER":
            return 50.0 - total_cost
        else:
            return -20.0 - total_cost

    if terminal_reason == "STOP":
        return -10.0

    if terminal_reason in ("RESOURCE_EXHAUSTION", "HORIZON"):
        return -5.0 - total_cost

    return -total_cost


def _is_terminal_action(action: Action) -> bool:
    """Check if an action is terminal."""
    return action.action_type in (ActionType.ANSWER, ActionType.DEFER, ActionType.STOP)


def _graph_state_hash(graph: EpistemicGraph) -> str:
    """Compute a state hash for the graph (for trajectory tracking)."""
    return hashlib.sha256(
        json.dumps(_serialize_graph(graph), sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


def rollout(
    checkpoint: Checkpoint,
    first_action: Action,
    downstream_policy: DownstreamPolicy,
    world_model_config: dict | None = None,
    max_steps: int = 10,
    seed: int | None = None,
) -> RolloutResult:
    """Execute a multi-step counterfactual rollout.

    Forces first_action from the checkpoint, then rolls forward with
    the frozen downstream policy until terminal or horizon.

    For stochastic actions, enumerates all outcome branches and computes
    exact expected utility:
      Q^π(s,a) = Σ_τ P(τ|s,a,π) U(τ)

    Args:
        checkpoint: The initial state
        first_action: The action to force (the manipulated variable)
        downstream_policy: The frozen policy for subsequent steps
        world_model_config: Optional world model configuration
        max_steps: Maximum trajectory length
        seed: RNG seed (defaults to checkpoint seed)
    """
    if seed is None:
        seed = checkpoint.seed

    first_action_hash = hashlib.sha256(str(first_action).encode()).hexdigest()[:16]

    # Enumerate all outcome branches for exact expected utility
    trajectories = _enumerate_branches(
        checkpoint, first_action, downstream_policy,
        world_model_config, max_steps, seed,
    )

    if not trajectories:
        # No valid trajectories — runtime error
        return RolloutResult(
            checkpoint_hash=checkpoint.checkpoint_hash,
            runtime_hash=checkpoint.runtime_hash(),
            topology_signature=checkpoint.topology_signature(),
            mechanism_signature="",  # Set by caller
            first_action=str(first_action),
            first_action_hash=first_action_hash,
            first_action_type=first_action.action_type.value,
            trajectory=(),
            observation_path=(),
            terminal_reason="RUNTIME_ERROR",
            terminal_state_hash="",
            utility=0.0,
            total_cost=0.0,
            steps_used=0,
            success=False,
            downstream_policy_id=downstream_policy.POLICY_VERSION,
            downstream_policy_hash=downstream_policy.policy_hash,
            world_model_hash=_world_model_hash(world_model_config),
            seed=seed,
            runtime_errors=("No valid trajectories",),
        )

    # Compute expected utility over all branches
    total_prob = sum(t["probability"] for t in trajectories)
    if total_prob == 0:
        expected_utility = 0.0
    else:
        expected_utility = sum(t["probability"] * t["utility"] for t in trajectories) / total_prob

    # Use the most likely trajectory for reporting
    best_traj = max(trajectories, key=lambda t: t["probability"])

    return RolloutResult(
        checkpoint_hash=checkpoint.checkpoint_hash,
        runtime_hash=checkpoint.runtime_hash(),
        topology_signature=checkpoint.topology_signature(),
        mechanism_signature="",  # Set by caller
        first_action=str(first_action),
        first_action_hash=first_action_hash,
        first_action_type=first_action.action_type.value,
        trajectory=tuple(best_traj["steps"]),
        observation_path=tuple(best_traj["observations"]),
        terminal_reason=best_traj["terminal_reason"],
        terminal_state_hash=best_traj["terminal_state_hash"],
        utility=expected_utility,
        total_cost=sum(s.step_cost for s in best_traj["steps"]),
        steps_used=len(best_traj["steps"]),
        success=best_traj["success"],
        downstream_policy_id=downstream_policy.POLICY_VERSION,
        downstream_policy_hash=downstream_policy.policy_hash,
        world_model_hash=_world_model_hash(world_model_config),
        seed=seed,
        runtime_errors=tuple(best_traj["errors"]),
    )


def _enumerate_branches(
    checkpoint: Checkpoint,
    first_action: Action,
    downstream_policy: DownstreamPolicy,
    world_model_config: dict | None,
    max_steps: int,
    seed: int,
) -> list[dict]:
    """Enumerate all outcome branches for exact expected utility.

    For small synthetic state spaces, this is feasible because the
    branching factor is small (VERIFY has 3 outcomes, SEARCH/RETRIEVE
    have 2, terminal actions have 1).
    """
    # Each branch is: (probability, steps, observations, terminal_reason, utility, success, errors)
    branches = []

    # Start with the first action
    initial_graph = copy.deepcopy(checkpoint.graph)
    initial_state_hash = _graph_state_hash(initial_graph)

    try:
        transitions = transition_model(initial_graph, first_action, world_model_config)
    except Exception as e:
        return [{
            "probability": 1.0,
            "steps": [TrajectoryStep(
                step_index=0,
                action_str=str(first_action),
                action_type=first_action.action_type.value,
                outcome="ERROR",
                outcome_probability=1.0,
                state_hash_before=initial_state_hash,
                state_hash_after="",
                step_cost=first_action.expected_cost,
                is_first_action=True,
                is_terminal=True,
                terminal_reason="RUNTIME_ERROR",
            )],
            "observations": ["ERROR"],
            "terminal_reason": "RUNTIME_ERROR",
            "terminal_state_hash": "",
            "utility": 0.0,
            "success": False,
            "errors": [f"First action error: {e}"],
        }]

    if not transitions:
        return []

    # Recursively enumerate all branches
    for trans in transitions:
        _enumerate_branch_recursive(
            checkpoint=checkpoint,
            graph=trans.next_graph,
            first_action=first_action,
            first_outcome=trans.outcome.value,
            first_outcome_prob=trans.probability,
            first_graph_before=initial_graph,
            downstream_policy=downstream_policy,
            world_model_config=world_model_config,
            max_steps=max_steps,
            current_step=1,
            current_prob=trans.probability,
            steps_so_far=[TrajectoryStep(
                step_index=0,
                action_str=str(first_action),
                action_type=first_action.action_type.value,
                outcome=trans.outcome.value,
                outcome_probability=trans.probability,
                state_hash_before=initial_state_hash,
                state_hash_after=_graph_state_hash(trans.next_graph),
                step_cost=first_action.expected_cost,
                is_first_action=True,
                is_terminal=_is_terminal_action(first_action),
                terminal_reason=(_terminal_reason_for_action(first_action)
                                 if _is_terminal_action(first_action) else ""),
            )],
            observations_so_far=[trans.outcome.value],
            branches=branches,
            errors=[],
        )

    return branches


def _enumerate_branch_recursive(
    checkpoint: Checkpoint,
    graph: EpistemicGraph,
    first_action: Action,
    first_outcome: str,
    first_outcome_prob: float,
    first_graph_before: EpistemicGraph,
    downstream_policy: DownstreamPolicy,
    world_model_config: dict | None,
    max_steps: int,
    current_step: int,
    current_prob: float,
    steps_so_far: list[TrajectoryStep],
    observations_so_far: list[str],
    branches: list[dict],
    errors: list[str],
):
    """Recursively enumerate branches until terminal or horizon."""
    # Check termination conditions
    last_step = steps_so_far[-1]
    if last_step.is_terminal:
        terminal_reason = last_step.terminal_reason
        utility = _compute_rollout_utility(checkpoint, first_action, steps_so_far, graph)
        success = _compute_success(checkpoint, first_action, steps_so_far, terminal_reason)
        branches.append({
            "probability": current_prob,
            "steps": steps_so_far,
            "observations": observations_so_far,
            "terminal_reason": terminal_reason,
            "terminal_state_hash": _graph_state_hash(graph),
            "utility": utility,
            "success": success,
            "errors": errors,
        })
        return

    if current_step >= max_steps:
        utility = _compute_rollout_utility(checkpoint, first_action, steps_so_far, graph)
        branches.append({
            "probability": current_prob,
            "steps": steps_so_far,
            "observations": observations_so_far,
            "terminal_reason": "HORIZON",
            "terminal_state_hash": _graph_state_hash(graph),
            "utility": utility,
            "success": False,
            "errors": errors,
        })
        return

    if graph.steps_remaining <= 0:
        utility = _compute_rollout_utility(checkpoint, first_action, steps_so_far, graph)
        branches.append({
            "probability": current_prob,
            "steps": steps_so_far,
            "observations": observations_so_far,
            "terminal_reason": "RESOURCE_EXHAUSTION",
            "terminal_state_hash": _graph_state_hash(graph),
            "utility": utility,
            "success": False,
            "errors": errors,
        })
        return

    # Select next action using downstream policy
    try:
        next_action = downstream_policy.select_action(graph)
    except Exception as e:
        errors.append(f"Downstream policy error: {e}")
        branches.append({
            "probability": current_prob,
            "steps": steps_so_far,
            "observations": observations_so_far,
            "terminal_reason": "RUNTIME_ERROR",
            "terminal_state_hash": _graph_state_hash(graph),
            "utility": 0.0,
            "success": False,
            "errors": errors,
        })
        return

    if next_action is None:
        errors.append("Downstream policy returned None")
        branches.append({
            "probability": current_prob,
            "steps": steps_so_far,
            "observations": observations_so_far,
            "terminal_reason": "RUNTIME_ERROR",
            "terminal_state_hash": _graph_state_hash(graph),
            "utility": 0.0,
            "success": False,
            "errors": errors,
        })
        return

    state_hash_before = _graph_state_hash(graph)

    try:
        transitions = transition_model(graph, next_action, world_model_config)
    except Exception as e:
        errors.append(f"Transition error: {e}")
        branches.append({
            "probability": current_prob,
            "steps": steps_so_far + [TrajectoryStep(
                step_index=current_step,
                action_str=str(next_action),
                action_type=next_action.action_type.value,
                outcome="ERROR",
                outcome_probability=1.0,
                state_hash_before=state_hash_before,
                state_hash_after="",
                step_cost=next_action.expected_cost,
                is_first_action=False,
                is_terminal=True,
                terminal_reason="RUNTIME_ERROR",
            )],
            "observations": observations_so_far + ["ERROR"],
            "terminal_reason": "RUNTIME_ERROR",
            "terminal_state_hash": "",
            "utility": 0.0,
            "success": False,
            "errors": errors,
        })
        return

    if not transitions:
        errors.append(f"No transitions for action {next_action}")
        branches.append({
            "probability": current_prob,
            "steps": steps_so_far,
            "observations": observations_so_far,
            "terminal_reason": "RUNTIME_ERROR",
            "terminal_state_hash": state_hash_before,
            "utility": 0.0,
            "success": False,
            "errors": errors,
        })
        return

    # Recurse into each outcome branch
    for trans in transitions:
        new_step = TrajectoryStep(
            step_index=current_step,
            action_str=str(next_action),
            action_type=next_action.action_type.value,
            outcome=trans.outcome.value,
            outcome_probability=trans.probability,
            state_hash_before=state_hash_before,
            state_hash_after=_graph_state_hash(trans.next_graph),
            step_cost=next_action.expected_cost,
            is_first_action=False,
            is_terminal=_is_terminal_action(next_action),
            terminal_reason=(_terminal_reason_for_action(next_action)
                             if _is_terminal_action(next_action) else ""),
        )
        _enumerate_branch_recursive(
            checkpoint=checkpoint,
            graph=trans.next_graph,
            first_action=first_action,
            first_outcome=first_outcome,
            first_outcome_prob=first_outcome_prob,
            first_graph_before=first_graph_before,
            downstream_policy=downstream_policy,
            world_model_config=world_model_config,
            max_steps=max_steps,
            current_step=current_step + 1,
            current_prob=current_prob * trans.probability,
            steps_so_far=steps_so_far + [new_step],
            observations_so_far=observations_so_far + [trans.outcome.value],
            branches=branches,
            errors=list(errors),  # Copy to avoid mutation
        )


def _terminal_reason_for_action(action: Action) -> str:
    """Get the terminal reason for a terminal action."""
    if action.action_type == ActionType.ANSWER:
        return "ANSWER"
    if action.action_type == ActionType.DEFER:
        return "DEFER"
    if action.action_type == ActionType.STOP:
        return "STOP"
    return ""


def _compute_success(
    checkpoint: Checkpoint,
    first_action: Action,
    steps: list[TrajectoryStep],
    terminal_reason: str,
) -> bool:
    """Determine if the rollout was successful."""
    if terminal_reason == "ANSWER":
        # Find the ANSWER action in the trajectory
        for step in steps:
            if step.action_type == "ANSWER":
                action_str = step.action_str
                if "(" in action_str and ")" in action_str:
                    answered_hyp = action_str.split("(")[1].rstrip(")")
                    return answered_hyp == checkpoint.correct_hypothesis_id
        return False
    if terminal_reason == "DEFER":
        return checkpoint.expected_terminal == "DEFER"
    return False


def _world_model_hash(world_model_config: dict | None) -> str:
    """Compute a hash for the world model configuration."""
    if world_model_config is None:
        return "default"
    return hashlib.sha256(
        json.dumps(world_model_config, sort_keys=True).encode()
    ).hexdigest()[:16]


def evaluate_all_actions_rollout(
    checkpoint: Checkpoint,
    actions: Sequence[Action],
    downstream_policy: DownstreamPolicy | None = None,
    world_model_config: dict | None = None,
    max_steps: int = 10,
    seed: int | None = None,
) -> list[RolloutResult]:
    """Evaluate all actions from the same checkpoint using multi-step rollout.

    This is the core counterfactual evaluation for M4.

    For state s, obtains:
      A(s) = {a_1, ..., a_n}
    Then for each action:
      Q^π(s,a_i) = E[U(τ) | do(a_0=a_i), π_downstream]

    The first-action intervention is the ONLY manipulated variable.
    """
    if downstream_policy is None:
        downstream_policy = DownstreamPolicy()
    if seed is None:
        seed = checkpoint.seed

    results = []
    for action in actions:
        result = rollout(
            checkpoint=checkpoint,
            first_action=action,
            downstream_policy=downstream_policy,
            world_model_config=world_model_config,
            max_steps=max_steps,
            seed=seed,
        )
        results.append(result)

    return results
