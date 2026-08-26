"""DAPH selective search package.

Bounded best-first lookahead for high-consequence, high-ambiguity decisions.
Does NOT run unconditional MCTS. Only searches when triggered by
consequence + ambiguity + budget criteria.

Defaults: depth=2, branching=2, max_nodes=6, max_model_calls=4.
"""
from daph.search.types import (
    SearchConfig, SearchDecision, SearchTriggerResult,
    BranchNode, BranchResult, SearchResult,
)
from daph.search.trigger import decide_search
from daph.search.budget import SearchBudget
from daph.search.planner import SearchPlanner

__all__ = [
    "SearchConfig", "SearchDecision", "SearchTriggerResult",
    "BranchNode", "BranchResult", "SearchResult",
    "decide_search", "SearchBudget", "SearchPlanner",
]
