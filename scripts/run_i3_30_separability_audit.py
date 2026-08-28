#!/usr/bin/env python3
"""I3.30 Phase 2: Full separability audit across all I3.29 trajectories.

Check whether the candidate V3 representation separates terminal decisions
across ALL trajectory states, not just the 11 known false-authority cases.

Search for collisions:
  f(s_i) = f(s_j) AND a_i* != a_j*
especially where a* in {ANSWER, DEFER}.

Candidate V3 features (observable, verified-evidence topology):
  n_hyp_with_verified_support
  n_hyp_with_verified_contradiction
  n_hyp_with_mixed_verified
  n_viable_hypotheses
  n_eliminated_hypotheses
  has_unique_verified_supported_hypothesis
  has_verified_unresolved_competition
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

from hrm_adaptive_memory.cognitive_control.core import DecisionAction
from hrm_adaptive_memory.cognitive_control.state import VerificationState
from hrm_adaptive_memory.executive.evidence_benchmark import initial_evidence_runtime
from hrm_adaptive_memory.executive.evidence_benchmark.executor import (
    EvidenceExecutor, valid_verify_targets,
)
from hrm_adaptive_memory.executive.resources import ResourceState
from hrm_adaptive_memory.executive.evidence_benchmark.i3_29_safety_generator import (
    generate_i3_29_benchmark, get_budget_for_task,
)
from daph.intervention.checkpoint import compute_state_features
from run_i3_28_rep_repair import compute_structural_features

OUTPUT_DIR = REPO_ROOT / "experiments/i3_30"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAJ_PATH = REPO_ROOT / "experiments/i3_29/live_safety/trajectories_v1.jsonl"


def compute_v3_structural(runtime):
    """Compute V3 post-verification structural features.

    Observable: uses only visible evidence supports/contradicts and
    verification_state. No verify_result oracle, no hidden evidence,
    no future outcomes.
    """
    hyp_verified_support = defaultdict(set)
    hyp_verified_contradiction = defaultdict(set)
    hyp_unverified_support = defaultdict(set)
    hyp_unverified_contradiction = defaultdict(set)

    for ev in runtime.visible_evidence:
        if not ev.retrieved:
            continue
        is_verified = ev.verification_state in (
            VerificationState.SUFFICIENT, VerificationState.FALSIFIED,
        )
        for h_id in ev.supports:
            if is_verified:
                hyp_verified_support[h_id].add(ev.evidence_id)
            else:
                hyp_unverified_support[h_id].add(ev.evidence_id)
        for h_id in ev.contradicts:
            if is_verified:
                hyp_verified_contradiction[h_id].add(ev.evidence_id)
            else:
                hyp_unverified_contradiction[h_id].add(ev.evidence_id)

    all_hyp_ids = {h.hypothesis_id for h in runtime.task.hypotheses}

    n_hyp_with_verified_support = 0
    n_hyp_with_verified_contradiction = 0
    n_hyp_with_mixed_verified = 0
    n_viable = 0
    n_eliminated = 0

    for h_id in all_hyp_ids:
        has_vs = len(hyp_verified_support.get(h_id, set())) > 0
        has_vc = len(hyp_verified_contradiction.get(h_id, set())) > 0
        if has_vs:
            n_hyp_with_verified_support += 1
        if has_vc:
            n_hyp_with_verified_contradiction += 1
        if has_vs and has_vc:
            n_hyp_with_mixed_verified += 1
        # Eliminated: has verified contradiction but no verified support
        if has_vc and not has_vs:
            n_eliminated += 1
        # Viable: not eliminated
        elif not (has_vc and not has_vs):
            n_viable += 1

    has_unique_verified_supported = n_hyp_with_verified_support == 1
    has_verified_unresolved_competition = n_hyp_with_verified_support > 1

    # Also compute V2R features for comparison
    v2r_structural = compute_structural_features([
        {
            "evidence_id": ev.evidence_id,
            "supports": list(ev.supports),
            "contradicts": list(ev.contradicts),
            "verification_state": ev.verification_state.name,
            "retrieved": ev.retrieved,
        }
        for ev in runtime.visible_evidence
    ])

    return {
        # V3 new features
        "n_hyp_with_verified_support": n_hyp_with_verified_support,
        "n_hyp_with_verified_contradiction": n_hyp_with_verified_contradiction,
        "n_hyp_with_mixed_verified": n_hyp_with_mixed_verified,
        "n_viable_hypotheses": n_viable,
        "n_eliminated_hypotheses": n_eliminated,
        "has_unique_verified_supported_hypothesis": int(has_unique_verified_supported),
        "has_verified_unresolved_competition": int(has_verified_unresolved_competition),
        # V2R features (for comparison)
        "n_hyp_unverified_support": v2r_structural["n_hyp_unverified_support"],
        "n_hyp_unverified_contradiction": v2r_structural["n_hyp_unverified_contradiction"],
        "has_competing_unverified_support": v2r_structural["has_competing_unverified_support"],
    }


def reconstruct_runtime_at_step(task, actions_before, d2_pre_verify=False):
    """Reconstruct runtime state after executing actions_before."""
    budget = get_budget_for_task(task)
    runtime = initial_evidence_runtime(task, ResourceState(budget=budget))
    executor = EvidenceExecutor()

    if d2_pre_verify:
        valid = valid_verify_targets(runtime)
        if valid:
            res = executor.execute(runtime, DecisionAction.VERIFY,
                                   target_evidence_id=valid[0])
            runtime = res.runtime

    for a_name in actions_before:
        try:
            action = DecisionAction(a_name)
            target = None
            if action == DecisionAction.VERIFY:
                valid = valid_verify_targets(runtime)
                if valid:
                    target = valid[0]
            res = executor.execute(runtime, action, target_evidence_id=target)
            runtime = res.runtime
            if res.terminal:
                break
        except Exception:
            break

    return runtime


def main():
    print("=" * 70)
    print("I3.30 Phase 2: Full Separability Audit")
    print("=" * 70)

    # Load I3.29 trajectories
    trajs = []
    with open(TRAJ_PATH) as f:
        for line in f:
            trajs.append(json.loads(line))

    # Load benchmark
    tasks = generate_i3_29_benchmark(seed=9817)
    task_by_id = {t.task_id: t for t in tasks}

    # Reconstruct every state from every trajectory
    # For each state, record:
    #   - V3 feature vector
    #   - V2R feature vector
    #   - correct terminal action (from task expected_terminal)
    #   - stratum
    #   - whether this was a terminal decision point

    print("\nReconstructing all trajectory states...")

    all_states = []
    for t in trajs:
        task = task_by_id.get(t["task_id"])
        if task is None:
            continue

        d2_pre = "_d2_" in t["task_id"]

        # Reconstruct state at each step
        for step in range(len(t["actions_taken"]) + 1):
            actions_before = t["actions_taken"][:step]
            runtime = reconstruct_runtime_at_step(task, actions_before, d2_pre)

            v3_features = compute_v3_structural(runtime)
            sf = compute_state_features(runtime, tuple(actions_before))

            # Determine if this is a terminal decision point
            # (i.e., the action taken at this step was terminal)
            is_terminal_step = False
            action_taken = None
            if step < len(t["actions_taken"]):
                action_taken = t["actions_taken"][step]
                is_terminal_step = action_taken in ("ANSWER", "DEFER")

            # Record the correct terminal action for this task
            correct_terminal = t["expected_terminal"]

            state_record = {
                "task_id": t["task_id"],
                "arm": t["arm"],
                "stratum": t["stratum"],
                "step": step,
                "action_taken": action_taken,
                "is_terminal_step": is_terminal_step,
                "correct_terminal": correct_terminal,
                "success": t["success"],
                "v3_features": v3_features,
                "state_features": {
                    k: sf.get(k, 0) for k in [
                        "n_live", "n_eliminated", "n_untested", "n_total_hypotheses",
                        "n_visible_evidence", "n_verified", "n_supporting",
                        "n_contradicting", "n_stale", "retrieval_remaining",
                        "search_remaining", "verify_remaining", "steps_remaining",
                        "can_retrieve", "can_search", "can_verify",
                    ]
                },
            }
            all_states.append(state_record)

    print(f"  Total states reconstructed: {len(all_states)}")

    # ============================================================
    # Collision audit: f(s_i) = f(s_j) AND a_i* != a_j*
    # ============================================================
    print("\n" + "=" * 70)
    print("Collision audit: V3 structural features")
    print("=" * 70)

    # Focus on terminal decision points where ANSWER or DEFER was taken
    terminal_states = [s for s in all_states if s["is_terminal_step"]]

    print(f"  Terminal decision points: {len(terminal_states)}")

    # Group by V3 structural feature vector
    v3_keys = [
        "n_hyp_with_verified_support",
        "n_hyp_with_verified_contradiction",
        "n_hyp_with_mixed_verified",
        "n_viable_hypotheses",
        "n_eliminated_hypotheses",
        "has_unique_verified_supported_hypothesis",
        "has_verified_unresolved_competition",
    ]

    def v3_signature(state):
        return tuple(state["v3_features"][k] for k in v3_keys)

    def v2r_signature(state):
        v2r_keys = [
            "n_hyp_unverified_support",
            "n_hyp_unverified_contradiction",
            "has_competing_unverified_support",
        ]
        return tuple(state["v3_features"][k] for k in v2r_keys)

    # Group terminal states by V3 signature
    by_v3 = defaultdict(list)
    for s in terminal_states:
        by_v3[v3_signature(s)].append(s)

    # Find collisions: same V3 signature, different correct terminal
    v3_collisions = 0
    v3_collision_groups = []
    for sig, states in by_v3.items():
        terminals = set(s["correct_terminal"] for s in states)
        if len(terminals) > 1:
            v3_collisions += 1
            v3_collision_groups.append((sig, states))

    print(f"\n  V3 collisions (same features, different correct terminal): {v3_collisions}")
    for sig, states in v3_collision_groups:
        terminals = defaultdict(list)
        for s in states:
            terminals[s["correct_terminal"]].append(s)
        print(f"\n  Signature: {dict(zip(v3_keys, sig))}")
        for term, term_states in terminals.items():
            strata = defaultdict(int)
            for ts in term_states:
                strata[ts["stratum"]] += 1
            print(f"    {term}: {len(term_states)} states, strata={dict(strata)}")

    # Compare with V2R collisions
    print("\n" + "=" * 70)
    print("Collision audit: V2R structural features (for comparison)")
    print("=" * 70)

    by_v2r = defaultdict(list)
    for s in terminal_states:
        by_v2r[v2r_signature(s)].append(s)

    v2r_collisions = 0
    v2r_collision_groups = []
    for sig, states in by_v2r.items():
        terminals = set(s["correct_terminal"] for s in states)
        if len(terminals) > 1:
            v2r_collisions += 1
            v2r_collision_groups.append((sig, states))

    v2r_keys = [
        "n_hyp_unverified_support",
        "n_hyp_unverified_contradiction",
        "has_competing_unverified_support",
    ]

    print(f"\n  V2R collisions: {v2r_collisions}")
    for sig, states in v2r_collision_groups:
        terminals = defaultdict(list)
        for s in states:
            terminals[s["correct_terminal"]].append(s)
        print(f"\n  Signature: {dict(zip(v2r_keys, sig))}")
        for term, term_states in terminals.items():
            strata = defaultdict(int)
            for ts in term_states:
                strata[ts["stratum"]] += 1
            print(f"    {term}: {len(term_states)} states, strata={dict(strata)}")

    # ============================================================
    # Minimal separating feature analysis
    # ============================================================
    print("\n" + "=" * 70)
    print("Minimal separating feature analysis")
    print("=" * 70)

    # For each candidate feature, check if it separates ANSWER-correct from DEFER-correct
    answer_states = [s for s in terminal_states if s["correct_terminal"] == "ANSWER"]
    defer_states = [s for s in terminal_states if s["correct_terminal"] == "DEFER"]

    print(f"\n  ANSWER-correct terminal states: {len(answer_states)}")
    print(f"  DEFER-correct terminal states: {len(defer_states)}")

    candidate_features = v3_keys + v2r_keys

    print(f"\n{'Feature':<50} {'ANSWER-correct':<25} {'DEFER-correct':<25} {'Separates?'}")
    print("-" * 130)

    separating_features = []
    for feat in candidate_features:
        answer_vals = set(s["v3_features"][feat] for s in answer_states)
        defer_vals = set(s["v3_features"][feat] for s in defer_states)
        separates = len(answer_vals & defer_vals) == 0 and len(answer_vals) > 0 and len(defer_vals) > 0
        print(f"{feat:<50} {str(sorted(answer_vals)):<25} {str(sorted(defer_vals)):<25} {'YES' if separates else 'NO'}")
        if separates:
            separating_features.append(feat)

    print(f"\n  Separating features: {separating_features}")

    # ============================================================
    # Check if minimal subset of separating features suffices
    # ============================================================
    if separating_features:
        print("\n" + "=" * 70)
        print("Minimal separating subset analysis")
        print("=" * 70)

        # Try each single separating feature
        for feat in separating_features:
            answer_vals = set(s["v3_features"][feat] for s in answer_states)
            defer_vals = set(s["v3_features"][feat] for s in defer_states)
            print(f"\n  {feat}:")
            print(f"    ANSWER-correct values: {sorted(answer_vals)}")
            print(f"    DEFER-correct values: {sorted(defer_vals)}")

            # Check for collisions across ALL terminal states using just this feature
            by_feat = defaultdict(list)
            for s in terminal_states:
                by_feat[s["v3_features"][feat]].append(s)

            collisions = 0
            for val, states in by_feat.items():
                terminals = set(s["correct_terminal"] for s in states)
                if len(terminals) > 1:
                    collisions += 1
                    term_counts = defaultdict(int)
                    for ts in states:
                        term_counts[ts["correct_terminal"]] += 1
                    print(f"    Collision at value={val}: {dict(term_counts)}")

            if collisions == 0:
                print(f"    NO collisions — this single feature separates all terminal states")

    # ============================================================
    # Also check non-terminal states for aliasing
    # ============================================================
    print("\n" + "=" * 70)
    print("Non-terminal state aliasing check")
    print("=" * 70)

    # Check all states (not just terminal) for V3 collisions where
    # the correct action differs
    # We use the action actually taken as a proxy for "correct" at non-terminal steps
    # This is weaker but still informative

    all_by_v3 = defaultdict(list)
    for s in all_states:
        all_by_v3[v3_signature(s)].append(s)

    non_terminal_collisions = 0
    for sig, states in all_by_v3.items():
        # Check if states in this group have different expected terminals
        terminals = set(s["correct_terminal"] for s in states)
        if len(terminals) > 1:
            # Check if any of these are terminal decision points
            terminal_in_group = [s for s in states if s["is_terminal_step"]]
            if terminal_in_group:
                non_terminal_collisions += 1

    print(f"  V3 groups with mixed correct terminals AND terminal decisions: {non_terminal_collisions}")

    # Save results
    results = {
        "total_states": len(all_states),
        "terminal_states": len(terminal_states),
        "v3_collisions": v3_collisions,
        "v2r_collisions": v2r_collisions,
        "separating_features": separating_features,
        "answer_correct_count": len(answer_states),
        "defer_correct_count": len(defer_states),
    }
    with open(OUTPUT_DIR / "separability_audit.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_DIR / 'separability_audit.json'}")


if __name__ == "__main__":
    main()
