#!/usr/bin/env python3
"""I3.30 Phase 1: Post-Verification Aliasing Audit.

Dump all 11 I3.29 false-authority states and matched safe controls.
Identify the minimal observable distinction between:
  - POST-VERIFY DEFER-CORRECT (D2)
  - POST-VERIFY ANSWER-CORRECT (D3 after verify, D4)
  - POST-VERIFY CONTINUE-CORRECT (D3 before verify resolves)

Do NOT add features yet. First demonstrate what separates the failure states.
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
from run_i3_28_rep_repair import (
    extract_v1_features, extract_v2r_features,
    get_v1_feature_keys, get_v2r_feature_keys,
    compute_structural_features,
)

OUTPUT_DIR = REPO_ROOT / "experiments/i3_30"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAJ_PATH = REPO_ROOT / "experiments/i3_29/live_safety/trajectories_v1.jsonl"


def get_visible_evidence_topology(runtime):
    """Dump complete visible evidence topology for a runtime state."""
    topology = []
    for ev in runtime.visible_evidence:
        topology.append({
            "evidence_id": ev.evidence_id,
            "supports": list(ev.supports),
            "contradicts": list(ev.contradicts),
            "verification_state": ev.verification_state.name,
            "retrieved": ev.retrieved,
        })
    return topology


def get_hypothesis_resolution_state(runtime):
    """Compute per-hypothesis verified evidence topology.

    This is the candidate new representation — but we're auditing first,
    not adding it yet.
    """
    hyp_verified_support = defaultdict(set)  # hyp_id -> set of evidence_ids
    hyp_verified_contradiction = defaultdict(set)
    hyp_unverified_support = defaultdict(set)
    hyp_unverified_contradiction = defaultdict(set)

    for ev in runtime.visible_evidence:
        if not ev.retrieved:
            continue
        is_verified = ev.verification_state in (VerificationState.SUFFICIENT, VerificationState.FALSIFIED)
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

    # Per-hypothesis summary
    hyps = {}
    all_hyp_ids = set(hyp_verified_support.keys()) | set(hyp_verified_contradiction.keys()) | \
                  set(hyp_unverified_support.keys()) | set(hyp_unverified_contradiction.keys()) | \
                  {h.hypothesis_id for h in runtime.task.hypotheses}

    for h_id in sorted(all_hyp_ids):
        n_vs = len(hyp_verified_support.get(h_id, set()))
        n_vc = len(hyp_verified_contradiction.get(h_id, set()))
        n_us = len(hyp_unverified_support.get(h_id, set()))
        n_uc = len(hyp_unverified_contradiction.get(h_id, set()))
        hyps[h_id] = {
            "n_verified_support": n_vs,
            "n_verified_contradiction": n_vc,
            "n_unverified_support": n_us,
            "n_unverified_contradiction": n_uc,
            "has_verified_support": n_vs > 0,
            "has_verified_contradiction": n_vc > 0,
            "is_eliminated": n_vc > 0 and n_vs == 0,  # contradicted with no support
            "is_confirmed": n_vs > 0 and n_vc == 0,   # supported with no contradiction
        }

    # Aggregate features (candidates for V3)
    n_hyp_with_verified_support = sum(1 for h in hyps.values() if h["has_verified_support"])
    n_hyp_with_verified_contradiction = sum(1 for h in hyps.values() if h["has_verified_contradiction"])
    n_hyp_with_mixed_verified = sum(1 for h in hyps.values()
                                     if h["has_verified_support"] and h["has_verified_contradiction"])
    n_viable = sum(1 for h in hyps.values() if not h["is_eliminated"])
    n_eliminated = sum(1 for h in hyps.values() if h["is_eliminated"])
    has_unique_verified_supported = n_hyp_with_verified_support == 1
    has_verified_unresolved_competition = n_hyp_with_verified_support > 1

    return {
        "per_hypothesis": hyps,
        "aggregate": {
            "n_hyp_with_verified_support": n_hyp_with_verified_support,
            "n_hyp_with_verified_contradiction": n_hyp_with_verified_contradiction,
            "n_hyp_with_mixed_verified": n_hyp_with_mixed_verified,
            "n_viable_hypotheses": n_viable,
            "n_eliminated_hypotheses": n_eliminated,
            "has_unique_verified_supported_hypothesis": has_unique_verified_supported,
            "has_verified_unresolved_competition": has_verified_unresolved_competition,
        }
    }


def reconstruct_state(task, step_receipt, d2_pre_verify=False):
    """Reconstruct the runtime state at a given step from the receipt."""
    budget = get_budget_for_task(task)
    runtime = initial_evidence_runtime(task, ResourceState(budget=budget))
    executor = EvidenceExecutor()

    # D2: pre-verify
    if d2_pre_verify:
        valid = valid_verify_targets(runtime)
        if valid:
            res = executor.execute(runtime, DecisionAction.VERIFY,
                                   target_evidence_id=valid[0])
            runtime = res.runtime

    # Replay actions from the receipt's step
    # We need the actions taken before this step
    # For now, just return the initial state (or post-D2-verify state)
    return runtime


def main():
    print("=" * 70)
    print("I3.30 Phase 1: Post-Verification Aliasing Audit")
    print("=" * 70)

    # Load I3.29 trajectories
    trajs = []
    with open(TRAJ_PATH) as f:
        for line in f:
            trajs.append(json.loads(line))

    # Identify the 11 false-authority states
    false_authority_states = []
    for t in trajs:
        if t["arm"] != "V2":
            continue
        for entry in t.get("authority_log", []):
            mode = entry.get("authority_mode", "")
            if mode.startswith("A2AD_hard"):
                forced = entry.get("forced_action", "")
                if not t["success"]:
                    false_authority_states.append({
                        "task_id": t["task_id"],
                        "stratum": t["stratum"],
                        "step": entry["step"],
                        "mode": mode,
                        "forced_action": forced,
                        "expected_terminal": t["expected_terminal"],
                        "success": t["success"],
                        "receipt": None,
                    })

    print(f"\nFalse authority states: {len(false_authority_states)}")
    for fa in false_authority_states:
        print(f"  {fa['task_id']} step={fa['step']} forced={fa['forced_action']} "
              f"expected={fa['expected_terminal']} mode={fa['mode']}")

    # Load the benchmark to reconstruct states
    tasks = generate_i3_29_benchmark(seed=9817)
    task_by_id = {t.task_id: t for t in tasks}

    # For each false-authority state, find the matching receipt
    v2_trajs = {t["task_id"]: t for t in trajs if t["arm"] == "V2"}
    for fa in false_authority_states:
        t = v2_trajs[fa["task_id"]]
        for r in t.get("receipts", []):
            if r.get("step") == fa["step"]:
                fa["receipt"] = r
                break

    # Now find matched safe controls
    # For each false-authority state, find a V2 task in the same stratum
    # where the same forced action was correct
    safe_controls = []
    for fa in false_authority_states:
        stratum = fa["stratum"]
        forced = fa["forced_action"]
        # Find a successful V2 task in the same stratum with the same forced action
        for t in trajs:
            if t["arm"] != "V2" or t["stratum"] != stratum:
                continue
            if not t["success"]:
                continue
            for entry in t.get("authority_log", []):
                if entry.get("forced_action") == forced and entry.get("authority_mode", "").startswith("A2AD_hard"):
                    safe_controls.append({
                        "task_id": t["task_id"],
                        "stratum": t["stratum"],
                        "step": entry["step"],
                        "mode": entry["authority_mode"],
                        "forced_action": forced,
                        "expected_terminal": t["expected_terminal"],
                        "success": t["success"],
                        "receipt": None,
                    })
                    break
            else:
                continue
            break

    print(f"\nMatched safe controls: {len(safe_controls)}")
    for sc in safe_controls[:5]:
        print(f"  {sc['task_id']} step={sc['step']} forced={sc['forced_action']} "
              f"expected={sc['expected_terminal']}")

    # Get receipts for safe controls
    for sc in safe_controls:
        t = v2_trajs[sc["task_id"]]
        for r in t.get("receipts", []):
            if r.get("step") == sc["step"]:
                sc["receipt"] = r
                break

    # ============================================================
    # Compare feature vectors and evidence topology
    # ============================================================
    print("\n" + "=" * 70)
    print("Feature comparison: false-authority vs safe controls")
    print("=" * 70)

    # Group by forced action
    for forced_action in ["DEFER", "ANSWER"]:
        fa_group = [fa for fa in false_authority_states if fa["forced_action"] == forced_action]
        sc_group = [sc for sc in safe_controls if sc["forced_action"] == forced_action]

        print(f"\n--- {forced_action} authority ---")
        print(f"  False authority: {len(fa_group)} cases")
        print(f"  Safe controls: {len(sc_group)} cases")

        # Compare V2R feature vectors
        v2r_keys = get_v2r_feature_keys()

        print(f"\n  V2R structural features (the ones that collapse):")
        structural_keys = [
            "n_hyp_unverified_support",
            "n_hyp_unverified_contradiction",
            "has_competing_unverified_support",
        ]

        for label, group in [("FALSE", fa_group), ("SAFE", sc_group)]:
            print(f"\n  {label} authority states:")
            for item in group[:3]:
                r = item["receipt"]
                if r is None:
                    continue
                sf = r.get("q_values", {})  # We need state features from the receipt
                structural = r.get("structural", {})
                print(f"    {item['task_id']} (expected={item['expected_terminal']}):")
                print(f"      structural: {structural}")
                print(f"      q_values: {r.get('q_values', {})}")
                print(f"      q_gap: {r.get('q_gap', {})}")
                print(f"      authority: {r.get('authority', {})}")

    # ============================================================
    # Now reconstruct actual runtime states and dump full topology
    # ============================================================
    print("\n" + "=" * 70)
    print("Full evidence topology reconstruction")
    print("=" * 70)

    audit_results = []

    for label, group in [("FALSE_AUTHORITY", false_authority_states),
                         ("SAFE_CONTROL", safe_controls)]:
        print(f"\n=== {label} ===")
        for item in group:
            task = task_by_id.get(item["task_id"])
            if task is None:
                continue

            d2_pre = "_d2_" in item["task_id"]
            budget = get_budget_for_task(task)
            runtime = initial_evidence_runtime(task, ResourceState(budget=budget))
            executor = EvidenceExecutor()

            # D2: pre-verify
            if d2_pre:
                valid = valid_verify_targets(runtime)
                if valid:
                    res = executor.execute(runtime, DecisionAction.VERIFY,
                                           target_evidence_id=valid[0])
                    runtime = res.runtime

            # Replay actions up to the step
            v2_traj = v2_trajs[item["task_id"]]
            actions_before = v2_traj["actions_taken"][:item["step"]]
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

            # Dump state
            sf = compute_state_features(runtime, tuple(actions_before))
            topology = get_visible_evidence_topology(runtime)
            hyp_resolution = get_hypothesis_resolution_state(runtime)
            structural = compute_structural_features(topology)

            result = {
                "label": label,
                "task_id": item["task_id"],
                "stratum": item["stratum"],
                "step": item["step"],
                "forced_action": item["forced_action"],
                "expected_terminal": item["expected_terminal"],
                "success": item["success"],
                "state_features": sf,
                "structural_v2r": structural,
                "evidence_topology": topology,
                "hypothesis_resolution": hyp_resolution,
            }
            audit_results.append(result)

            print(f"\n  {item['task_id']} (stratum={item['stratum']}, "
                  f"forced={item['forced_action']}, expected={item['expected_terminal']}, "
                  f"success={item['success']})")
            print(f"  V2R structural: {structural}")
            print(f"  Hypothesis resolution aggregate: {hyp_resolution['aggregate']}")
            print(f"  Per-hypothesis:")
            for h_id, h_info in hyp_resolution["per_hypothesis"].items():
                print(f"    {h_id}: vs={h_info['n_verified_support']} "
                      f"vc={h_info['n_verified_contradiction']} "
                      f"us={h_info['n_unverified_support']} "
                      f"uc={h_info['n_unverified_contradiction']} "
                      f"eliminated={h_info['is_eliminated']} "
                      f"confirmed={h_info['is_confirmed']}")
            print(f"  Evidence topology:")
            for ev in topology:
                print(f"    {ev['evidence_id']}: vstate={ev['verification_state']} "
                      f"supports={ev['supports']} contradicts={ev['contradicts']}")

    # ============================================================
    # Identify the minimal separating representation
    # ============================================================
    print("\n" + "=" * 70)
    print("Minimal separating representation analysis")
    print("=" * 70)

    # Group by correct terminal action
    by_correct_terminal = defaultdict(list)
    for r in audit_results:
        correct = r["expected_terminal"]
        by_correct_terminal[correct].append(r)

    print(f"\nStates grouped by correct terminal action:")
    for terminal, states in by_correct_terminal.items():
        false_count = sum(1 for s in states if s["label"] == "FALSE_AUTHORITY")
        safe_count = sum(1 for s in states if s["label"] == "SAFE_CONTROL")
        print(f"  {terminal}: {len(states)} total ({false_count} false, {safe_count} safe)")

    # Check which candidate features separate the groups
    candidate_features = [
        "n_hyp_with_verified_support",
        "n_hyp_with_verified_contradiction",
        "n_hyp_with_mixed_verified",
        "n_viable_hypotheses",
        "n_eliminated_hypotheses",
        "has_unique_verified_supported_hypothesis",
        "has_verified_unresolved_competition",
    ]

    print(f"\nCandidate separating features:")
    print(f"{'Feature':<50} {'DEFER-correct':<20} {'ANSWER-correct':<20} {'Separates?'}")
    print("-" * 110)

    defer_states = by_correct_terminal.get("DEFER", [])
    answer_states = by_correct_terminal.get("ANSWER", [])

    for feat in candidate_features:
        defer_vals = set()
        answer_vals = set()
        for s in defer_states:
            val = s["hypothesis_resolution"]["aggregate"].get(feat)
            defer_vals.add(val)
        for s in answer_states:
            val = s["hypothesis_resolution"]["aggregate"].get(feat)
            answer_vals.add(val)

        # Check if the sets are disjoint
        separates = len(defer_vals & answer_vals) == 0 and len(defer_vals) > 0 and len(answer_vals) > 0
        print(f"{feat:<50} {str(defer_vals):<20} {str(answer_vals):<20} {'YES' if separates else 'NO'}")

    # Also check V2R structural features
    print(f"\nV2R structural features (current representation):")
    print(f"{'Feature':<50} {'DEFER-correct':<20} {'ANSWER-correct':<20} {'Separates?'}")
    print("-" * 110)

    v2r_structural = [
        "n_hyp_unverified_support",
        "n_hyp_unverified_contradiction",
        "has_competing_unverified_support",
    ]

    for feat in v2r_structural:
        defer_vals = set()
        answer_vals = set()
        for s in defer_states:
            val = s["structural_v2r"].get(feat)
            defer_vals.add(val)
        for s in answer_states:
            val = s["structural_v2r"].get(feat)
            answer_vals.add(val)
        separates = len(defer_vals & answer_vals) == 0 and len(defer_vals) > 0 and len(answer_vals) > 0
        print(f"{feat:<50} {str(defer_vals):<20} {str(answer_vals):<20} {'YES' if separates else 'NO'}")

    # Save audit results
    with open(OUTPUT_DIR / "aliasing_audit.json", "w") as f:
        json.dump(audit_results, f, indent=2, default=str)
    print(f"\nAudit saved to {OUTPUT_DIR / 'aliasing_audit.json'}")


if __name__ == "__main__":
    main()
