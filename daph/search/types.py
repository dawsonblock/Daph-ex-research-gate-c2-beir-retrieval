"""Search type definitions."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SearchConfig:
    """Configuration for bounded best-first search.

    Defaults are deliberately conservative:
      max_depth=2, branching_factor=2, max_nodes=6, max_model_calls=4
    """
    max_depth: int = 2
    branching_factor: int = 2
    max_nodes: int = 6
    max_model_calls: int = 4
    max_wall_ms: int = 5000
    min_steps_remaining: int = 3
    q_epsilon: float = 3.0
    # Early stop margin: if best branch leads by this much, stop
    early_stop_margin: float = 5.0
    # Search trigger thresholds
    trigger_min_near_optimal: int = 2  # need >= 2 near-optimal to search
    trigger_resource_pressure_steps: int = 4  # <= this many steps left
    trigger_repeated_action_run: int = 2  # same action repeated this many times

    def as_dict(self) -> dict:
        return {
            "max_depth": self.max_depth,
            "branching_factor": self.branching_factor,
            "max_nodes": self.max_nodes,
            "max_model_calls": self.max_model_calls,
            "max_wall_ms": self.max_wall_ms,
            "min_steps_remaining": self.min_steps_remaining,
            "q_epsilon": self.q_epsilon,
            "early_stop_margin": self.early_stop_margin,
            "trigger_min_near_optimal": self.trigger_min_near_optimal,
            "trigger_resource_pressure_steps": self.trigger_resource_pressure_steps,
            "trigger_repeated_action_run": self.trigger_repeated_action_run,
        }


@dataclass(frozen=True)
class SearchTriggerResult:
    """Result of the search trigger decision.

    Attributes:
        should_search: Whether to invoke selective search
        reasons: List of trigger reasons that fired
        config: The search config that will be used
    """
    should_search: bool
    reasons: tuple[str, ...]
    config: SearchConfig

    def as_dict(self) -> dict:
        return {
            "should_search": self.should_search,
            "reasons": list(self.reasons),
            "config": self.config.as_dict(),
        }


@dataclass
class BranchNode:
    """A node in the search tree.

    Attributes:
        action: The action taken to reach this node
        depth: Depth in the search tree (0 = root)
        parent_id: ID of parent node, or None for root
        node_id: Unique ID for this node
        checkpoint_id: Checkpoint ID at this node's state
        state_sha: State hash at this node
        q_value: Q value of the action taken
        pav_score: PAV score of the action taken
        children: Child node IDs
        terminal: Whether this node is terminal
        terminal_utility: Utility at terminal (if terminal)
        success: Whether task succeeded at terminal
        cumulative_cost: Cumulative action cost to reach this node
        cumulative_pav: Cumulative PAV score to reach this node
    """
    action: str
    depth: int
    parent_id: str | None
    node_id: str
    checkpoint_id: str
    state_sha: str
    q_value: float = 0.0
    pav_score: float = 0.0
    children: list[str] = field(default_factory=list)
    terminal: bool = False
    terminal_utility: float | None = None
    success: bool | None = None
    cumulative_cost: float = 0.0
    cumulative_pav: float = 0.0

    def as_dict(self) -> dict:
        return {
            "action": self.action,
            "depth": self.depth,
            "parent_id": self.parent_id,
            "node_id": self.node_id,
            "checkpoint_id": self.checkpoint_id,
            "state_sha": self.state_sha,
            "q_value": round(self.q_value, 4),
            "pav_score": round(self.pav_score, 4),
            "children": list(self.children),
            "terminal": self.terminal,
            "terminal_utility": self.terminal_utility,
            "success": self.success,
            "cumulative_cost": round(self.cumulative_cost, 4),
            "cumulative_pav": round(self.cumulative_pav, 4),
        }


@dataclass(frozen=True)
class BranchResult:
    """Result of evaluating a single branch.

    Attributes:
        first_action: The first action in the branch (what we'd execute)
        nodes: All nodes in this branch
        score: Branch score (weighted Q + PAV - cost)
        terminal: Whether the branch reached a terminal
        success: Whether the branch succeeded
        terminal_utility: Utility at terminal (if terminal)
        cumulative_cost: Total action cost
        cumulative_pav: Total PAV score
        depth_reached: Actual depth reached (may be < max_depth)
    """
    first_action: str
    nodes: tuple[BranchNode, ...]
    score: float
    terminal: bool
    success: bool | None
    terminal_utility: float | None
    cumulative_cost: float
    cumulative_pav: float
    depth_reached: int

    def as_dict(self) -> dict:
        return {
            "first_action": self.first_action,
            "nodes": [n.as_dict() for n in self.nodes],
            "score": round(self.score, 4),
            "terminal": self.terminal,
            "success": self.success,
            "terminal_utility": self.terminal_utility,
            "cumulative_cost": round(self.cumulative_cost, 4),
            "cumulative_pav": round(self.cumulative_pav, 4),
            "depth_reached": self.depth_reached,
        }


@dataclass(frozen=True)
class SearchResult:
    """Result of a complete search.

    Attributes:
        abstained: Whether search declined and fell back to VP
        winner: The winning first action, or None if abstained
        branches: All evaluated branches
        nodes_expanded: Total nodes expanded
        model_calls: Total model calls used
        wall_time_ms: Wall time in milliseconds
        config: Search config used
        trigger_reasons: Why search was triggered
        fallback_reason: Why search abstained (if it did)
        receipt: Full provenance receipt
    """
    abstained: bool
    winner: str | None
    branches: tuple[BranchResult, ...]
    nodes_expanded: int
    model_calls: int
    wall_time_ms: float
    config: SearchConfig
    trigger_reasons: tuple[str, ...]
    fallback_reason: str | None
    receipt: dict

    def as_dict(self) -> dict:
        return {
            "abstained": self.abstained,
            "winner": self.winner,
            "branches": [b.as_dict() for b in self.branches],
            "nodes_expanded": self.nodes_expanded,
            "model_calls": self.model_calls,
            "wall_time_ms": round(self.wall_time_ms, 2),
            "config": self.config.as_dict(),
            "trigger_reasons": list(self.trigger_reasons),
            "fallback_reason": self.fallback_reason,
            "receipt": self.receipt,
        }


# Type alias for search decision
SearchDecision = SearchResult
