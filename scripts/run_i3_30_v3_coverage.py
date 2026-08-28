#!/usr/bin/env python3
"""I3.30 Phase 3b: Reconstruct V3 features for all causal records and build
the real coverage matrix.

The causal records reference checkpoint_ids. The checkpoints have evidence
and hypotheses with answer_action. We can reconstruct V3 features from those.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from hrm_adaptive_memory.cognitive_control.state import VerificationState

OUTPUT_DIR = REPO_ROOT / "experiments/i3_30"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CKPT_PATH = REPO_ROOT / "experiments/i3_5/datasets/checkpoints_v1.jsonl"
CAUSAL_DATA_PATHS = [
    REPO_ROOT / "experiments/i3_5/pinned_policy/pinned_causal_actions_v1.jsonl",
    REPO_ROOT / "experiments/i3_28b/boundary_causal_actions_v1.jsonl",
    REPO_ROOT / "experiments/i3_28c/strata_causal_actions_v1.jsonl",
]


def load_checkpoints(path: Path):
    ckpts = {}
    if not path.exists():
        return ckpts
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            ckpts[r["checkpoint_id"]] = r
    return ckpts


def compute_v3_features(evidence: list, hypotheses: list):
    """Compute V3 structural features from checkpoint evidence and hypotheses."""
    hyp_verified_support = defaultdict(set)
    hyp_verified_contradiction = defaultdict(set)
    hyp_unverified_support = defaultdict(set)
    hyp_unverified_contradiction = defaultdict(set)

    for ev in evidence:
        if not ev.get("retrieved", False):
            continue
        vstate = ev.get("verification_state", "UNVERIFIED")
        is_verified = vstate in ("SUFFICIENT", "FALSIFIED")
        for h_id in ev.get("supports", []):
            if is_verified:
                hyp_verified_support[h_id].add(ev["evidence_id"])
            else:
                hyp_unverified_support[h_id].add(ev["evidence_id"])
        for h_id in ev.get("contradicts", []):
            if is_verified:
                hyp_verified_contradiction[h_id].add(ev["evidence_id"])
            else:
                hyp_unverified_contradiction[h_id].add(ev["evidence_id"])

    all_hyp_ids = {h["hypothesis_id"] for h in hypotheses}

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
        if has_vc and not has_vs:
            n_eliminated += 1
        else:
            n_viable += 1

    has_unique_verified_supported = n_hyp_with_verified_support == 1
    has_verified_unresolved_competition = n_hyp_with_verified_support > 1

    # verified_hyp_action
    verified_hyp_action = None
    supported_hyps = [h_id for h_id, evs in hyp_verified_support.items() if len(evs) > 0]
    if len(supported_hyps) == 1:
        h_id = supported_hyps[0]
        for h in hypotheses:
            if h["hypothesis_id"] == h_id:
                verified_hyp_action = h.get("answer_action", None)
                break

    return {
        "n_hyp_with_verified_support": n_hyp_with_verified_support,
        "n_hyp_with_verified_contradiction": n_hyp_with_verified_contradiction,
        "n_hyp_with_mixed_verified": n_hyp_with_mixed_verified,
        "n_viable_hypotheses": n_viable,
        "n_eliminated_hypotheses": n_eliminated,
        "has_unique_verified_supported_hypothesis": int(has_unique_verified_supported),
        "has_verified_unresolved_competition": int(has_verified_unresolved_competition),
        "verified_hyp_action": verified_hyp_action,
        "verified_hyp_action_is_answer": int(verified_hyp_action == "ANSWER"),
        "verified_hyp_action_is_defer": int(verified_hyp_action == "DEFER"),
        # V2R features for comparison
        "n_hyp_unverified_support": len(hyp_unverified_support),
        "n_hyp_unverified_contradiction": len(hyp_unverified_contradiction),
        "has_competing_unverified_support": int(len(hyp_unverified_support) > 1),
    }


def classify_v3_cell(v3: dict, expected_terminal: str, forced_action: str):
    """Classify a record into a coverage cell."""
    n_verified_support = v3["n_hyp_with_verified_support"]
    n_verified_contradiction = v3["n_hyp_with_verified_contradiction"]
    vha = v3["verified_hyp_action"]

    # Verification state
    if n_verified_support == 0 and n_verified_contradiction == 0:
        vstate = "none"
    elif n_verified_support > 0 and n_verified_contradiction == 0:
        vstate = "complete_supported"
    elif n_verified_support == 0 and n_verified_contradiction > 0:
        vstate = "complete_eliminated"
    else:
        vstate = "complete_mixed"

    # Topology
    if n_verified_support == 0 and n_verified_contradiction == 0:
        topo = "no_verified"
    elif n_verified_support == 1 and n_verified_contradiction == 0:
        topo = "unique_supported"
    elif n_verified_support == 1 and n_verified_contradiction >= 1:
        topo = "unique_supported_with_elim"
    elif n_verified_support > 1:
        topo = "competing_support"
    elif n_verified_support == 0 and n_verified_contradiction >= 1:
        topo = "only_eliminated"
    else:
        topo = "other"

    return (vstate, topo, vha, expected_terminal, forced_action)


def main():
    print("=" * 70)
    print("I3.30 Phase 3b: V3 Coverage Matrix (Reconstructed from Checkpoints)")
    print("=" * 70)

    # Load checkpoints
    ckpts = load_checkpoints(CKPT_PATH)
    print(f"  Checkpoints loaded: {len(ckpts)}")

    # Load all causal records
    all_records = []
    for path in CAUSAL_DATA_PATHS:
        records = []
        with open(path) as f:
            for line in f:
                records.append(json.loads(line))
        print(f"  {path.name}: {len(records)} records")
        all_records.extend(records)
    print(f"  Total: {len(all_records)} records")

    # Reconstruct V3 features for each record
    v3_records = []
    missing_ckpt = 0
    for r in all_records:
        ckpt_id = r.get("checkpoint_id")
        ckpt = ckpts.get(ckpt_id)
        if ckpt is None:
            missing_ckpt += 1
            continue
        v3 = compute_v3_features(ckpt.get("evidence", []), ckpt.get("hypotheses", []))
        v3_records.append({
            "checkpoint_id": ckpt_id,
            "task_id": r.get("task_id"),
            "source": r.get("source", "i3_5"),
            "forced_action": r.get("forced_action"),
            "expected_terminal": r.get("expected_terminal"),
            "correct_first_action": r.get("correct_first_action"),
            "pinned_policy_utility": r.get("pinned_policy_utility"),
            "pinned_policy_success": r.get("pinned_policy_success"),
            "v3_features": v3,
            "state_features": r.get("state_features", {}),
        })

    print(f"  Records with V3 features: {len(v3_records)}")
    print(f"  Missing checkpoints: {missing_ckpt}")

    # Build coverage matrix
    coverage = defaultdict(int)
    for r in v3_records:
        cell = classify_v3_cell(r["v3_features"], r["expected_terminal"], r["forced_action"])
        coverage[cell] += 1

    # Print coverage matrix focusing on safety-critical cells
    print("\n" + "=" * 70)
    print("V3 Coverage Matrix (verification_state, topology, verified_hyp_action, expected_terminal, forced_action)")
    print("=" * 70)

    # Group by (vstate, topo, vha, expected_terminal) and show all forced_actions
    grouped = defaultdict(lambda: defaultdict(int))
    for r in v3_records:
        v3 = r["v3_features"]
        vstate, topo, vha, et, fa = classify_v3_cell(v3, r["expected_terminal"], r["forced_action"])
        grouped[(vstate, topo, vha, et)][fa] += 1

    print(f"\n{'VState':<20} {'Topology':<25} {'VHA':<10} {'Expected':<10} {'Forced actions'}")
    print("-" * 120)

    for (vstate, topo, vha, et), actions in sorted(grouped.items()):
        action_str = ", ".join(f"{a}:{n}" for a, n in sorted(actions.items()))
        print(f"{vstate:<20} {topo:<25} {str(vha):<10} {et:<10} {action_str}")

    # Identify safety-critical cells with zero support
    print("\n" + "=" * 70)
    print("Safety-Critical Coverage Check")
    print("=" * 70)

    # The critical cells for V3 authority:
    # 1. Post-verify, unique supported, vha=ANSWER, expected=ANSWER → ANSWER authority should fire
    # 2. Post-verify, unique supported, vha=DEFER, expected=DEFER → DEFER authority should fire
    # 3. Post-verify, unique supported, vha=ANSWER, expected=DEFER → should NOT fire ANSWER (D2 false)
    # 4. Post-verify, unique supported, vha=DEFER, expected=ANSWER → should NOT fire DEFER (D3 false)
    # 5. Post-verify, only eliminated, expected=DEFER → DEFER authority candidate
    # 6. Post-verify, competing support → should NOT fire any terminal authority
    # 7. No verify, no verified evidence, expected=DEFER → DEFER authority (D1)
    # 8. No verify, no verified evidence, expected=ANSWER → should NOT fire ANSWER (premature)

    critical_cells = [
        ("complete_supported", "unique_supported", "ANSWER", "ANSWER",
         "D4: ANSWER authority should fire"),
        ("complete_supported", "unique_supported", "DEFER", "DEFER",
         "D2: DEFER authority should fire"),
        ("complete_supported", "unique_supported", "ANSWER", "DEFER",
         "D2 false: should NOT fire ANSWER"),
        ("complete_supported", "unique_supported", "DEFER", "ANSWER",
         "D3 false: should NOT fire DEFER"),
        ("complete_eliminated", "only_eliminated", None, "DEFER",
         "D2 elim: DEFER authority candidate"),
        ("complete_mixed", "unique_supported_with_elim", "ANSWER", "ANSWER",
         "D3 post-verify: ANSWER authority"),
        ("complete_mixed", "unique_supported_with_elim", "DEFER", "DEFER",
         "D2 post-verify with elim: DEFER authority"),
        ("complete_supported", "competing_support", None, "ANSWER",
         "D3 competing: should NOT fire terminal"),
        ("none", "no_verified", None, "DEFER",
         "D1: DEFER authority (no verify)"),
        ("none", "no_verified", None, "ANSWER",
         "D3 pre-verify: should NOT fire ANSWER"),
    ]

    print(f"\n{'VState':<22} {'Topology':<25} {'VHA':<8} {'Expected':<10} {'Count':<6} {'Description'}")
    print("-" * 120)

    zero_cells = []
    for vstate, topo, vha, et, desc in critical_cells:
        # Count all records in this cell (across all forced_actions)
        count = sum(n for (vs, t, v, e), actions in grouped.items()
                    if vs == vstate and t == topo and v == vha and e == et
                    for n in actions.values())
        marker = " *** ZERO ***" if count == 0 else (" * TINY *" if count < 10 else "")
        print(f"{vstate:<22} {topo:<25} {str(vha):<8} {et:<10} {count:<6} {desc}{marker}")
        if count < 10:
            zero_cells.append({
                "vstate": vstate, "topology": topo, "vha": vha,
                "expected": et, "count": count, "description": desc,
            })

    print(f"\n  Safety-critical cells with < 10 support: {len(zero_cells)}")

    # Save results
    results = {
        "total_records": len(v3_records),
        "missing_checkpoints": missing_ckpt,
        "coverage": {str(k): v for k, v in coverage.items()},
        "zero_support_cells": zero_cells,
    }
    with open(OUTPUT_DIR / "v3_coverage_matrix.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_DIR / 'v3_coverage_matrix.json'}")


if __name__ == "__main__":
    main()
