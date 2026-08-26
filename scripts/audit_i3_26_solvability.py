#!/usr/bin/env python3
"""I3.26 search-solvability audit: prove VP failures are actually solvable.

For each of the 21 VP failures from Phase 25 confirmation, run a bounded
oracle-style BFS over all legal action sequences using the deterministic
EvidenceExecutor (no LLM needed). Find the minimum successful sequence
and its depth.

Classify each failure as:
  SEARCH_SOLVABLE: A successful legal sequence exists within budget
  SEARCH_UNSOLVABLE_WITHIN_BUDGET: No successful sequence exists
  UNKNOWN: Search space too large to exhaustively explore

Also measure d* = minimum lookahead depth needed to distinguish the
winning first action from the losing one.

Output: experiments/i3_26/solvability_audit.json
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from hrm_adaptive_memory.cognitive_control.core import DecisionAction
from hrm_adaptive_memory.cognitive_control.state import (
    VerificationState, TemporalStatus,
)
from hrm_adaptive_memory.executive.evidence_benchmark.schema import (
    EvidenceTask, EvidenceRuntime, initial_evidence_runtime,
)
from hrm_adaptive_memory.executive.evidence_benchmark.executor import (
    EvidenceExecutor, valid_verify_targets,
)
from hrm_adaptive_memory.executive.resources import ResourceBudget, ResourceState
from hrm_adaptive_memory.executive.metareasoning_utility import MetareasoningUtility

from hrm_adaptive_memory.executive.evidence_benchmark.i3_5_confirmation_generator import (
    generate_confirmation_benchmark, CONFIRMATION_BUDGET_PROFILES,
)
from daph.intervention.checkpoint import (
    create_checkpoint, compute_state_features, compute_legal_actions,
)


@dataclass
class SolvabilityResult:
    task_id: str
    category: str
    vp_actions: list[str]
    vp_terminal_result: str
    budget_profile: str
    max_steps: int
    solvable: bool
    min_successful_sequence: list[str] | None
    min_successful_depth: int | None
    successful_first_action: str | None
    resource_usage: dict | None
    terminal_utility: float | None
    classification: str  # SEARCH_SOLVABLE, SEARCH_UNSOLVABLE_WITHIN_BUDGET, UNKNOWN
    all_successful_first_actions: list[str]
    d_star: int | None  # minimum depth to distinguish winning first action
    n_sequences_explored: int
    timing_ms: float


def get_budget_for_profile(profile: str) -> ResourceBudget:
    params = CONFIRMATION_BUDGET_PROFILES[profile]
    return ResourceBudget(
        max_executive_steps=params["max_executive_steps"],
        max_retrieval_calls=params["max_retrieval_calls"],
        max_verification_calls=params["max_verification_calls"],
        max_search_calls=params["max_search_calls"],
        max_reasoning_tokens=params.get("max_reasoning_tokens", 256),
        max_elapsed_ms=params.get("max_elapsed_ms", 10_000),
    )


def bfs_solvability(
    task: EvidenceTask,
    budget: ResourceBudget,
    max_steps: int,
    max_sequences: int = 5000,
) -> tuple[bool, list[str] | None, int | None, dict | None, float | None,
           list[str], int, set[str]]:
    """BFS over all legal action sequences to find minimum successful path.

    Returns:
        (solvable, min_sequence, min_depth, resource_usage, terminal_utility,
         successful_first_actions, n_explored, visited_states)
    """
    utility = MetareasoningUtility.from_file(
        REPO_ROOT / "configs" / "v2b_i3_1_utility_v1.json",
    )
    executor = EvidenceExecutor()

    initial_runtime = initial_evidence_runtime(task, ResourceState(budget=budget))

    # BFS state: (runtime, actions_taken, prior_outcomes)
    queue = deque()
    queue.append((initial_runtime, [], []))

    visited = set()
    n_explored = 0
    successful_first_actions = set()

    min_successful = None
    min_successful_depth = None
    min_successful_resources = None
    min_successful_utility = None

    while queue and n_explored < max_sequences:
        runtime, actions, outcomes = queue.popleft()
        n_explored += 1

        if len(actions) >= max_steps:
            continue

        # Compute legal actions at this state
        legal = compute_legal_actions(runtime)
        if not legal:
            continue

        # State hash for deduplication
        state_key = (
            tuple(ev.evidence_id for ev in runtime.visible_evidence),
            tuple(sorted(ev.verification_state.value for ev in runtime.visible_evidence)),
            runtime.resources.as_dict().get("executive_steps_remaining", 0),
            runtime.resources.as_dict().get("retrieval_calls_remaining", 0),
            runtime.resources.as_dict().get("verification_calls_remaining", 0),
            runtime.resources.as_dict().get("search_calls_remaining", 0),
            runtime.searched,
            runtime.reasoning_complete,
        )
        if state_key in visited:
            continue
        visited.add(state_key)

        for action_str in legal:
            action = DecisionAction(action_str)

            # Determine verify target
            target_eid = None
            if action is DecisionAction.VERIFY:
                valid = valid_verify_targets(runtime)
                if valid:
                    target_eid = valid[0]
                else:
                    continue

            try:
                exec_result = executor.execute(runtime, action, target_evidence_id=target_eid)
            except Exception:
                continue

            new_actions = actions + [action_str]
            new_outcomes = outcomes + [exec_result.outcome_code]

            if exec_result.terminal:
                if exec_result.task_success:
                    # Found a successful path!
                    if not successful_first_actions:
                        min_successful = new_actions
                        min_successful_depth = len(new_actions)
                        min_successful_resources = exec_result.runtime.resources.as_dict()
                        # Compute terminal utility
                        terminal_u = utility.terminal_reward(
                            exec_result.action, True,
                        )
                        # Subtract action costs
                    successful_first_actions.add(new_actions[0])

                    # Track minimum
                    if min_successful_depth is None or len(new_actions) < min_successful_depth:
                        min_successful = new_actions
                        min_successful_depth = len(new_actions)
                        min_successful_resources = exec_result.runtime.resources.as_dict()
                        terminal_u = utility.terminal_reward(exec_result.action, True)
                        min_successful_utility = terminal_u
            else:
                # Continue BFS
                if len(new_actions) < max_steps:
                    queue.append((exec_result.runtime, new_actions, new_outcomes))

    solvable = len(successful_first_actions) > 0
    return (
        solvable,
        min_successful,
        min_successful_depth,
        min_successful_resources,
        min_successful_utility,
        sorted(successful_first_actions),
        n_explored,
        visited,
    )


def compute_d_star(
    task: EvidenceTask,
    budget: ResourceBudget,
    max_steps: int,
    successful_first_actions: set[str],
    vp_first_action: str,
) -> int | None:
    """Compute minimum lookahead depth d* needed to distinguish winning first action.

    For each depth d from 1 to max_steps:
    - For each first action, compute the best achievable outcome at depth d
    - Check if the winning first action is distinguishable from the losing one

    Returns d* or None if not distinguishable within max_steps.
    """
    if not successful_first_actions:
        return None

    utility = MetareasoningUtility.from_file(
        REPO_ROOT / "configs" / "v2b_i3_1_utility_v1.json",
    )
    executor = EvidenceExecutor()

    for d in range(1, max_steps + 1):
        # For each first action, compute best outcome at depth d
        first_action_outcomes = {}

        initial_runtime = initial_evidence_runtime(task, ResourceState(budget=budget))
        legal = compute_legal_actions(initial_runtime)

        for action_str in legal:
            action = DecisionAction(action_str)
            target_eid = None
            if action is DecisionAction.VERIFY:
                valid = valid_verify_targets(initial_runtime)
                if valid:
                    target_eid = valid[0]
                else:
                    first_action_outcomes[action_str] = -999.0
                    continue

            try:
                exec_result = executor.execute(initial_runtime, action, target_evidence_id=target_eid)
            except Exception:
                first_action_outcomes[action_str] = -999.0
                continue

            if exec_result.terminal:
                if exec_result.task_success:
                    first_action_outcomes[action_str] = 100.0  # Success
                else:
                    first_action_outcomes[action_str] = -100.0  # Failure
            elif d == 1:
                # At depth 1, use Q-like proxy (progress)
                from daph.progress.progress_rule_v1 import compute_progress
                try:
                    progress = compute_progress(initial_runtime, exec_result, utility)
                    first_action_outcomes[action_str] = progress.progress
                except Exception:
                    first_action_outcomes[action_str] = 0.0
            else:
                # Expand to depth d-1 more
                best_at_depth = _best_outcome_at_depth(
                    exec_result.runtime, d - 1, max_steps, utility, executor,
                )
                first_action_outcomes[action_str] = best_at_depth

        # Check if winning actions are distinguishable from losing actions
        winning_scores = [first_action_outcomes.get(a, -999) for a in successful_first_actions]
        losing_scores = [first_action_outcomes.get(a, -999)
                        for a in first_action_outcomes if a not in successful_first_actions]

        if not losing_scores:
            return d  # Only winning actions exist

        best_winning = max(winning_scores) if winning_scores else -999
        best_losing = max(losing_scores) if losing_scores else -999

        if best_winning > best_losing:
            return d

    return None


def _best_outcome_at_depth(
    runtime: EvidenceRuntime,
    depth: int,
    max_total_steps: int,
    utility: MetareasoningUtility,
    executor: EvidenceExecutor,
) -> float:
    """Compute the best achievable outcome at a given depth."""
    if depth == 0:
        # Use progress as proxy
        from daph.progress.progress_rule_v1 import _compute_phi
        phi = _compute_phi(runtime)
        return sum(phi.values())

    legal = compute_legal_actions(runtime)
    best = -999.0

    for action_str in legal:
        action = DecisionAction(action_str)
        target_eid = None
        if action is DecisionAction.VERIFY:
            valid = valid_verify_targets(runtime)
            if valid:
                target_eid = valid[0]
            else:
                continue

        try:
            exec_result = executor.execute(runtime, action, target_evidence_id=target_eid)
        except Exception:
            continue

        if exec_result.terminal:
            if exec_result.task_success:
                return 100.0
            else:
                score = -100.0
            if score > best:
                best = score
        else:
            score = _best_outcome_at_depth(
                exec_result.runtime, depth - 1, max_total_steps, utility, executor,
            )
            if score > best:
                best = score

    return best


def main():
    print("Loading confirmation trajectories...")
    conf_path = REPO_ROOT / "experiments/i3_5/confirmation/trajectories_v1.jsonl"
    records = [json.loads(line) for line in open(conf_path)]

    by_task_arm = {}
    for r in records:
        by_task_arm[(r["task_id"], r["arm"])] = r

    # Load failure audit
    audit_path = REPO_ROOT / "experiments/i3_26/failure_audit.json"
    audit = json.load(open(audit_path))

    # Load confirmation tasks
    print("Loading confirmation benchmark...")
    tasks = generate_confirmation_benchmark(n_per_subtype=12, seed=4287)
    task_by_id = {t.task_id: t for t in tasks}

    # Get VP failures
    vp_failures = audit["failures"]
    print(f"VP failures to audit: {len(vp_failures)}")

    results = []
    print(f"\n{'='*80}")
    print("SEARCH-SOLVABILITY AUDIT")
    print(f"{'='*80}")

    for f in vp_failures:
        tid = f["task_id"]
        task = task_by_id.get(tid)

        if task is None:
            print(f"\n  {tid}: TASK NOT FOUND")
            continue

        budget = get_budget_for_profile(task.budget_profile)
        max_steps = budget.max_executive_steps

        vp_record = by_task_arm.get((tid, "VP"))
        vp_first_action = vp_record["actions_taken"][0] if vp_record and vp_record["actions_taken"] else None

        print(f"\n  {tid} ({f['category']}, {f['failure_type']}):")
        print(f"    VP sequence: {f['actions']}")
        print(f"    Budget: {task.budget_profile} (max_steps={max_steps})")

        start_time = time.time()
        solvable, min_seq, min_depth, resources, terminal_u, \
            successful_first, n_explored, visited = bfs_solvability(
                task, budget, max_steps,
            )
        timing_ms = (time.time() - start_time) * 1000

        # Compute d* if solvable
        d_star = None
        if solvable and successful_first:
            d_star = compute_d_star(
                task, budget, max_steps,
                set(successful_first), vp_first_action or "",
            )

        if solvable:
            classification = "SEARCH_SOLVABLE"
            print(f"    SOLVABLE: min sequence = {min_seq}")
            print(f"    Min depth: {min_depth}")
            print(f"    Successful first actions: {successful_first}")
            print(f"    d* (min distinguishing depth): {d_star}")
            print(f"    Explored: {n_explored} sequences in {timing_ms:.0f}ms")
        else:
            classification = "SEARCH_UNSOLVABLE_WITHIN_BUDGET"
            print(f"    UNSOLVABLE: no successful path within budget")
            print(f"    Explored: {n_explored} sequences in {timing_ms:.0f}ms")

        results.append(SolvabilityResult(
            task_id=tid,
            category=f["category"],
            vp_actions=f["actions"],
            vp_terminal_result=f["terminal_result"],
            budget_profile=task.budget_profile,
            max_steps=max_steps,
            solvable=solvable,
            min_successful_sequence=min_seq,
            min_successful_depth=min_depth,
            successful_first_action=successful_first[0] if successful_first else None,
            resource_usage=resources,
            terminal_utility=terminal_u,
            classification=classification,
            all_successful_first_actions=successful_first,
            d_star=d_star,
            n_sequences_explored=n_explored,
            timing_ms=timing_ms,
        ))

    # Summary
    print(f"\n{'='*80}")
    print("SOLVABILITY SUMMARY")
    print(f"{'='*80}")

    by_class = defaultdict(int)
    by_category_solvable = defaultdict(lambda: {"solvable": 0, "unsolvable": 0})
    d_stars = []

    for r in results:
        by_class[r.classification] += 1
        cat = r.category
        if r.solvable:
            by_category_solvable[cat]["solvable"] += 1
        else:
            by_category_solvable[cat]["unsolvable"] += 1
        if r.d_star is not None:
            d_stars.append(r.d_star)

    print(f"\nClassification:")
    for cls, count in sorted(by_class.items()):
        print(f"  {cls}: {count}")

    print(f"\nBy category:")
    for cat, counts in sorted(by_category_solvable.items()):
        print(f"  {cat}: {counts['solvable']} solvable, {counts['unsolvable']} unsolvable")

    print(f"\nMinimum distinguishing depth (d*):")
    if d_stars:
        print(f"  Values: {sorted(d_stars)}")
        print(f"  Min: {min(d_stars)}, Max: {max(d_stars)}, Median: {sorted(d_stars)[len(d_stars)//2]}")
        from collections import Counter
        d_counts = Counter(d_stars)
        print(f"  Distribution:")
        for d, count in sorted(d_counts.items()):
            print(f"    d*={d}: {count} failures")
    else:
        print(f"  No solvable failures with computed d*")

    # Save
    output = {
        "total_failures": len(results),
        "classification_counts": dict(by_class),
        "by_category": {k: dict(v) for k, v in by_category_solvable.items()},
        "d_star_values": d_stars,
        "d_star_distribution": dict(Counter(d_stars)) if d_stars else {},
        "results": [r.__dict__ if hasattr(r, '__dict__') else dict(r) for r in results],
    }

    # Convert dataclass objects
    output["results"] = []
    for r in results:
        output["results"].append({
            "task_id": r.task_id,
            "category": r.category,
            "vp_actions": r.vp_actions,
            "vp_terminal_result": r.vp_terminal_result,
            "budget_profile": r.budget_profile,
            "max_steps": r.max_steps,
            "solvable": r.solvable,
            "min_successful_sequence": r.min_successful_sequence,
            "min_successful_depth": r.min_successful_depth,
            "successful_first_action": r.successful_first_action,
            "resource_usage": r.resource_usage,
            "terminal_utility": r.terminal_utility,
            "classification": r.classification,
            "all_successful_first_actions": r.all_successful_first_actions,
            "d_star": r.d_star,
            "n_sequences_explored": r.n_sequences_explored,
            "timing_ms": r.timing_ms,
        })

    output_path = REPO_ROOT / "experiments/i3_26/solvability_audit.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, sort_keys=True)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
