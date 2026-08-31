#!/usr/bin/env python3
"""Phase 14: Authority modes with FORCE shadowed.

Implements all five authority modes. FORCE is computed but NOT executed.
Logs what FORCE would have done for every disagreement.

Usage:
    python scripts/authority_shadow.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from daph_x.actions.typed_actions import Action, ActionType, answer, defer, verify, stop
from daph_x.authority.executive import select_action, ExecutiveConfig
from daph_x.authority.authority_policy import determine_authority_mode, AuthorityConfig, constrain_actions
from daph_x.belief.belief_engine import compute_belief_state
from daph_x.graph.epistemic_graph import build_graph_from_evidence_task
from daph_x.receipts.checkpoint import checkpoint_from_task_and_runtime
from daph_x.receipts.fork_engine import evaluate_all_actions, compute_oracle_action
from daph_x.actions.candidate_generator import generate_and_prune

from build_structural_ood_pool import OOD_DOMAIN_TEMPLATES, generate_ood_candidate


def simulate_llm_proposal(task) -> str:
    """Simulate LLM proposal."""
    for e in task.evidence_items:
        if e.verification_state.value == "UNVERIFIED":
            return "VERIFY"
    verified_support = sum(1 for e in task.evidence_items
                          if e.verification_state.value == "SUFFICIENT" and e.supports)
    if verified_support >= 2:
        return "DEFER"
    return "ANSWER"


def run_shadow_analysis(task, seed=42):
    """Run a task with all authority modes shadowed."""
    checkpoint = checkpoint_from_task_and_runtime(task, None, seed=seed)
    graph = build_graph_from_evidence_task(task)
    belief = compute_belief_state(graph)
    candidates = generate_and_prune(graph)

    if not candidates:
        return None

    # Get executive decision
    config = ExecutiveConfig()
    decision = select_action(graph, config=config)

    # Evaluate all actions for ground truth
    results = evaluate_all_actions(checkpoint, candidates, seed=seed)
    oracle_action, oracle_utility = compute_oracle_action(results)

    # Get LLM proposal
    llm_proposal = simulate_llm_proposal(task)

    # Compute ΔQ between executive and LLM
    executive_score = decision.expected_value
    llm_score = 0.0  # Placeholder
    for r in results:
        if r.first_action.split("(")[0] == llm_proposal:
            llm_score = r.utility
            break

    delta_q = executive_score - llm_score

    # Determine authority mode (shadowed)
    auth_config = AuthorityConfig(force_enabled=False)  # FORCE disabled
    authority_mode = determine_authority_mode(
        belief=belief,
        selected_action=decision.selected_action,
        selected_score=executive_score,
        selected_sigma=0.1,  # Placeholder
        next_best_score=llm_score,
        next_best_sigma=0.1,
        intervention_risk=0.05,  # Placeholder
        llm_proposal=None,
        config=auth_config,
    )

    # What would FORCE do?
    would_force = (
        decision.authority_mode.value == "FORCE"
        or (delta_q > 5.0 and decision.structural_certificate)
    )

    # Is FORCE correct?
    executive_correct = str(decision.selected_action) == oracle_action
    would_force_correct = executive_correct if would_force else None

    return {
        "task_id": task.task_id,
        "category": task.category,
        "llm_proposal": llm_proposal,
        "executive_action": str(decision.selected_action),
        "oracle_action": oracle_action,
        "executive_correct": executive_correct,
        "authority_mode": authority_mode.value,
        "would_force": would_force,
        "would_force_correct": would_force_correct,
        "delta_q": delta_q,
        "executive_score": executive_score,
        "llm_score": llm_score,
        "oracle_utility": oracle_utility,
        "structural_certificate": decision.structural_certificate,
        "readiness": belief.readiness.value,
        "n_supported": belief.n_supported,
        "n_contradicted": belief.n_contradicted,
    }


def main():
    print("=" * 80)
    print("PHASE 14: AUTHORITY MODES (FORCE SHADOWED)")
    print("=" * 80)

    all_results = []
    for template in OOD_DOMAIN_TEMPLATES:
        for i in range(10):
            task = generate_ood_candidate(template, i)
            result = run_shadow_analysis(task)
            if result:
                all_results.append(result)

    print(f"\nTotal tasks: {len(all_results)}")

    # Authority mode distribution
    auth_counts = defaultdict(int)
    for r in all_results:
        auth_counts[r["authority_mode"]] += 1

    print(f"\nAuthority mode distribution:")
    for mode, count in sorted(auth_counts.items()):
        print(f"  {mode}: {count}")

    # FORCE analysis
    force_cases = [r for r in all_results if r["would_force"]]
    force_correct = sum(1 for r in force_cases if r["would_force_correct"])
    force_harmful = sum(1 for r in force_cases if r["would_force_correct"] is False)

    print(f"\nFORCE analysis (shadowed):")
    print(f"  Would FORCE: {len(force_cases)}/{len(all_results)}")
    print(f"  Correct: {force_correct}")
    print(f"  Harmful: {force_harmful}")
    print(f"  Precision: {force_correct/max(len(force_cases),1):.3f}")

    # Disagreement analysis
    disagreements = [r for r in all_results
                     if r["executive_action"].split("(")[0] != r["llm_proposal"]]
    print(f"\nDisagreements: {len(disagreements)}/{len(all_results)}")

    # Where executive is correct but LLM is wrong
    executive_better = [r for r in disagreements
                        if r["executive_correct"] and r["llm_proposal"] != r["oracle_action"].split("(")[0]]
    print(f"Executive correct, LLM wrong: {len(executive_better)}")

    # Where executive is wrong but LLM is correct
    llm_better = [r for r in disagreements
                  if not r["executive_correct"] and r["llm_proposal"] == r["oracle_action"].split("(")[0]]
    print(f"LLM correct, executive wrong: {len(llm_better)}")

    # Both correct
    both_correct = [r for r in all_results
                    if r["executive_correct"] and r["llm_proposal"] == r["oracle_action"].split("(")[0]]
    print(f"Both correct: {len(both_correct)}")

    # Both wrong
    both_wrong = [r for r in all_results
                  if not r["executive_correct"] and r["llm_proposal"] != r["oracle_action"].split("(")[0]]
    print(f"Both wrong: {len(both_wrong)}")

    # Save
    output_path = REPO_ROOT / "experiments/daph_x/authority_shadow.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
