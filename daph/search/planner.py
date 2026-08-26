"""Search planner: bounded best-first lookahead.

1. Expand candidate first actions (from Q epsilon set)
2. Evaluate resulting states
3. Expand best nonterminal branches to depth 2
4. Score complete/partial branches
5. Choose winning first action
6. Execute only that first action
7. Replan after actual transition

No UCT/MCTS. Simple bounded best-first.
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from hrm_adaptive_memory.cognitive_control.core import DecisionAction
from hrm_adaptive_memory.executive.evidence_benchmark.schema import (
    EvidenceTask, EvidenceRuntime,
)
from hrm_adaptive_memory.executive.evidence_benchmark.executor import (
    EvidenceExecutor, valid_verify_targets,
)
from hrm_adaptive_memory.executive.metareasoning_utility import MetareasoningUtility

from daph.intervention.checkpoint import StateCheckpoint, create_checkpoint
from daph.intervention.restore import restore_runtime
from daph.progress.progress_rule_v1 import compute_progress
from daph.search.types import (
    SearchConfig, BranchNode, BranchResult, SearchResult,
)
from daph.search.budget import SearchBudget
from daph.search.branch import simulate_branch_step
from daph.search.evaluator import evaluate_branch


class SearchPlanner:
    """Bounded best-first search planner.

    Expands candidate actions from the Q epsilon set, evaluates branches
    to depth 2, and returns the winning first action.
    """

    def __init__(
        self,
        task: EvidenceTask,
        utility: MetareasoningUtility,
        config: SearchConfig,
    ):
        self.task = task
        self.utility = utility
        self.config = config
        self._executor = EvidenceExecutor()

    def plan(
        self,
        checkpoint: StateCheckpoint,
        candidate_actions: tuple[str, ...],
        q_values: dict[str, float],
        trigger_reasons: tuple[str, ...],
    ) -> SearchResult:
        """Run bounded best-first search from a checkpoint.

        Args:
            checkpoint: The root checkpoint
            candidate_actions: Actions to branch over (from Q epsilon set, top-k)
            q_values: Q values at the root state
            trigger_reasons: Why search was triggered

        Returns:
            SearchResult with winner or abstention.
        """
        start_time = time.time()
        budget = SearchBudget(self.config)

        # Limit candidates to branching factor
        candidates = list(candidate_actions[: self.config.branching_factor])

        if not candidates:
            return self._abstain(
                "no_candidates", trigger_reasons, budget, start_time,
            )

        # Phase 1: Expand first actions
        phase1_nodes: dict[str, tuple[BranchNode, EvidenceRuntime | None]] = {}
        phase1_checkpoints: dict[str, StateCheckpoint] = {}

        for action in candidates:
            if not budget.can_expand():
                break
            node, post_runtime = simulate_branch_step(
                checkpoint, self.task, action, self.utility,
                self._executor, parent_id=None, depth=1,
            )
            node = BranchNode(
                **{**node.__dict__, "q_value": q_values.get(action, 0.0)},
            )
            phase1_nodes[action] = (node, post_runtime)
            budget.consume_node()

            # Create checkpoint for phase 2 expansion
            if post_runtime is not None and not node.terminal:
                post_cp = create_checkpoint(
                    post_runtime,
                    step=checkpoint.step + 1,
                    phase=checkpoint.phase,
                    prior_actions=checkpoint.prior_actions + (action,),
                    prior_outcomes=checkpoint.prior_outcomes + ("OK",),
                )
                phase1_checkpoints[action] = post_cp

        # Phase 2: Expand best nonterminal branches to depth 2
        phase2_results: dict[str, tuple[BranchNode, ...]] = {}

        for action, (node, post_runtime) in phase1_nodes.items():
            if node.terminal or post_runtime is None:
                phase2_results[action] = (node,)
                continue

            if action not in phase1_checkpoints:
                phase2_results[action] = (node,)
                continue

            if not budget.can_expand():
                phase2_results[action] = (node,)
                continue

            # Get legal actions at the post-action state
            post_cp = phase1_checkpoints[action]
            post_runtime_restore = restore_runtime(post_cp, self.task)

            # Determine legal actions at depth 2
            from daph.intervention.checkpoint import compute_legal_actions
            legal_d2 = compute_legal_actions(post_runtime_restore)

            # Pick the best action at depth 2 (by Q if available, else by PAV)
            # For simplicity, pick the first legal non-terminal action
            # that isn't the same as the parent (avoid loops)
            best_d2_action = None
            best_d2_score = -999.0

            for d2_action in legal_d2:
                if d2_action == action:  # Avoid same-action loops
                    continue
                # Quick PAV evaluation
                try:
                    d2_node, _ = simulate_branch_step(
                        post_cp, self.task, d2_action, self.utility,
                        self._executor, parent_id=node.node_id, depth=2,
                    )
                    if d2_node.pav_score > best_d2_score:
                        best_d2_score = d2_node.pav_score
                        best_d2_action = d2_action
                except Exception:
                    continue

            if best_d2_action is not None and budget.can_expand():
                d2_node, d2_post = simulate_branch_step(
                    post_cp, self.task, best_d2_action, self.utility,
                    self._executor, parent_id=node.node_id, depth=2,
                )
                budget.consume_node()
                phase2_results[action] = (node, d2_node)
            else:
                phase2_results[action] = (node,)

        # Evaluate all branches
        branches = []
        for action in candidates:
            if action in phase2_results:
                branch = evaluate_branch(
                    first_action=action,
                    nodes=phase2_results[action],
                    q_values=q_values,
                )
                branches.append(branch)

        if not branches:
            return self._abstain(
                "no_branches_evaluated", trigger_reasons, budget, start_time,
            )

        # Early stop: if one branch dominates by early_stop_margin
        branches.sort(key=lambda b: b.score, reverse=True)
        if len(branches) >= 2:
            margin = branches[0].score - branches[1].score
            if margin >= self.config.early_stop_margin:
                # Clear winner
                pass  # Still return it normally

        # Check if all branches converge to same first action (trivially true here)
        winner = branches[0].first_action if branches else None

        # Check for early stop conditions
        # If winner reached terminal success, that's great
        # If all branches failed, abstain
        all_terminal = all(b.terminal for b in branches)
        any_success = any(b.success for b in branches)

        if all_terminal and not any_success:
            # All branches lead to failure — abstain, let VP handle it
            return self._abstain(
                "all_branches_fail", trigger_reasons, budget, start_time,
                branches=branches,
            )

        wall_ms = (time.time() - start_time) * 1000

        receipt = {
            "planner": "SearchPlanner",
            "checkpoint_id": checkpoint.checkpoint_id,
            "candidates": candidates,
            "branches": [b.as_dict() for b in branches],
            "winner": winner,
            "budget": budget.as_dict(),
            "trigger_reasons": list(trigger_reasons),
            "wall_ms": round(wall_ms, 2),
        }

        return SearchResult(
            abstained=False,
            winner=winner,
            branches=tuple(branches),
            nodes_expanded=budget.nodes_expanded,
            model_calls=budget.model_calls,
            wall_time_ms=wall_ms,
            config=self.config,
            trigger_reasons=trigger_reasons,
            fallback_reason=None,
            receipt=receipt,
        )

    def _abstain(
        self,
        reason: str,
        trigger_reasons: tuple[str, ...],
        budget: SearchBudget,
        start_time: float,
        branches: list[BranchResult] | None = None,
    ) -> SearchResult:
        """Return an abstention result (fall back to VP)."""
        wall_ms = (time.time() - start_time) * 1000
        return SearchResult(
            abstained=True,
            winner=None,
            branches=tuple(branches) if branches else (),
            nodes_expanded=budget.nodes_expanded,
            model_calls=budget.model_calls,
            wall_time_ms=wall_ms,
            config=self.config,
            trigger_reasons=trigger_reasons,
            fallback_reason=reason,
            receipt={
                "planner": "SearchPlanner",
                "abstained": True,
                "fallback_reason": reason,
                "budget": budget.as_dict(),
                "trigger_reasons": list(trigger_reasons),
                "wall_ms": round(wall_ms, 2),
            },
        )
