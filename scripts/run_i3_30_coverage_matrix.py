#!/usr/bin/env python3
"""I3.30 Phase 3: Training support coverage matrix.

Check whether the existing causal training data covers the safety-critical
cells of the V3 representation. This lesson has now occurred twice:
  - I3.28B: representation had the features but training data lacked the interaction
  - I3.29: V2R had post-verify features but training data lacked post-verify contrast

Build a coverage matrix:
  Verification state | Verified topology | Correct terminal regime | Training support
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
from run_i3_28_rep_repair import compute_structural_features

OUTPUT_DIR = REPO_ROOT / "experiments/i3_30"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Training data sources
CAUSAL_DATA_PATHS = [
    REPO_ROOT / "experiments/i3_5/pinned_policy/pinned_causal_actions_v1.jsonl",
    REPO_ROOT / "experiments/i3_28b/boundary_causal_actions_v1.jsonl",
    REPO_ROOT / "experiments/i3_28c/strata_causal_actions_v1.jsonl",
]


def load_causal_records(path: Path):
    records = []
    if not path.exists():
        return records
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def classify_verification_state(record):
    """Classify a causal record's verification state."""
    evidence = record.get("evidence_items", [])
    if not evidence:
        return "none"
    
    has_verified = any(
        ev.get("verification_state") in ("SUFFICIENT", "FALSIFIED")
        for ev in evidence if ev.get("retrieved", False)
    )
    if not has_verified:
        return "none"
    
    all_verified = all(
        ev.get("verification_state") != "UNVERIFIED"
        for ev in evidence if ev.get("retrieved", False)
    )
    if all_verified:
        return "complete"
    return "partial"


def classify_verified_topology(record):
    """Classify the verified evidence topology."""
    evidence = record.get("evidence_items", [])
    
    hyp_verified_support = defaultdict(set)
    hyp_verified_contradiction = defaultdict(set)
    
    for ev in evidence:
        if not ev.get("retrieved", False):
            continue
        if ev.get("verification_state") not in ("SUFFICIENT", "FALSIFIED"):
            continue
        if ev.get("verification_state") == "SUFFICIENT":
            for h_id in ev.get("supports", []):
                hyp_verified_support[h_id].add(ev["evidence_id"])
        elif ev.get("verification_state") == "FALSIFIED":
            for h_id in ev.get("contradicts", []):
                hyp_verified_contradiction[h_id].add(ev["evidence_id"])
    
    n_vs = len(hyp_verified_support)
    n_vc = len(hyp_verified_contradiction)
    
    if n_vs == 0 and n_vc == 0:
        return "no_verified_evidence"
    if n_vs == 1 and n_vc == 0:
        return "unique_supported"
    if n_vs == 1 and n_vc >= 1:
        return "unique_supported_with_eliminated"
    if n_vs > 1:
        return "competing_support"
    if n_vs == 0 and n_vc >= 1:
        return "only_eliminated"
    return "other"


def classify_correct_terminal(record):
    """Classify the correct terminal regime from the causal record."""
    # The causal record has an action and utility
    action = record.get("action", "")
    utility = record.get("utility", 0.0)
    
    # Terminal actions
    if action == "ANSWER":
        if utility > 0:
            return "ANSWER_correct"
        else:
            return "ANSWER_wrong"
    elif action == "DEFER":
        if utility > 0:
            return "DEFER_correct"
        else:
            return "DEFER_wrong"
    elif action in ("VERIFY", "RETRIEVE", "SEARCH_MORE", "REASON_MORE"):
        if utility > 0:
            return "CONTINUE_correct"
        else:
            return "CONTINUE_wrong"
    elif action == "STOP":
        return "STOP"
    return "other"


def main():
    print("=" * 70)
    print("I3.30 Phase 3: Training Support Coverage Matrix")
    print("=" * 70)

    # Load all causal records
    all_records = []
    for path in CAUSAL_DATA_PATHS:
        records = load_causal_records(path)
        print(f"  {path.name}: {len(records)} records")
        all_records.extend(records)
    
    print(f"\n  Total causal records: {len(all_records)}")

    # Build coverage matrix
    # Rows: (verification_state, verified_topology, correct_terminal_regime)
    # Cells: count of training records

    coverage = defaultdict(int)
    
    for record in all_records:
        vstate = classify_verification_state(record)
        topology = classify_verified_topology(record)
        terminal = classify_correct_terminal(record)
        coverage[(vstate, topology, terminal)] += 1

    # Print coverage matrix
    print("\n" + "=" * 70)
    print("Coverage Matrix")
    print("=" * 70)

    vstates = ["none", "partial", "complete"]
    topologies = [
        "no_verified_evidence", "unique_supported", "unique_supported_with_eliminated",
        "competing_support", "only_eliminated", "other",
    ]
    terminals = [
        "ANSWER_correct", "ANSWER_wrong",
        "DEFER_correct", "DEFER_wrong",
        "CONTINUE_correct", "CONTINUE_wrong",
        "STOP", "other",
    ]

    print(f"\n{'Verification':<12} {'Topology':<35} {'Terminal regime':<20} {'Count'}")
    print("-" * 90)

    for vs in vstates:
        for topo in topologies:
            for term in terminals:
                count = coverage.get((vs, topo, term), 0)
                if count > 0:
                    print(f"{vs:<12} {topo:<35} {term:<20} {count}")

    # Identify safety-critical cells with zero or tiny support
    print("\n" + "=" * 70)
    print("Safety-Critical Cells with Zero or Tiny Support")
    print("=" * 70)

    # These are the cells that V3 needs to learn correctly
    safety_critical = [
        # (verification_state, topology, terminal_regime, description)
        ("none", "no_verified_evidence", "DEFER_correct",
         "D1: safe DEFER, no verification possible"),
        ("none", "no_verified_evidence", "ANSWER_correct",
         "D3 pre-verify: ANSWER correct but premature"),
        ("complete", "unique_supported", "ANSWER_correct",
         "D4: post-verify, unique supported hypothesis, ANSWER correct"),
        ("complete", "unique_supported", "DEFER_correct",
         "D2: post-verify, unique supported hypothesis, DEFER correct"),
        ("complete", "unique_supported_with_eliminated", "DEFER_correct",
         "D2 with eliminated: post-verify, supported + eliminated, DEFER correct"),
        ("complete", "unique_supported_with_eliminated", "ANSWER_correct",
         "D3 post-verify: supported + eliminated, ANSWER correct"),
        ("complete", "competing_support", "CONTINUE_correct",
         "Post-verify with competing support, continue"),
        ("complete", "only_eliminated", "DEFER_correct",
         "Post-verify, only eliminated, DEFER correct"),
        ("partial", "unique_supported", "ANSWER_correct",
         "Partial verify, unique supported, ANSWER correct"),
        ("partial", "unique_supported", "DEFER_correct",
         "Partial verify, unique supported, DEFER correct"),
        ("partial", "unique_supported", "CONTINUE_correct",
         "Partial verify, unique supported, continue"),
        ("partial", "competing_support", "CONTINUE_correct",
         "Partial verify, competing support, continue"),
    ]

    print(f"\n{'Verification':<12} {'Topology':<35} {'Terminal':<20} {'Count':<6} {'Description'}")
    print("-" * 120)

    zero_support_cells = []
    for vs, topo, term, desc in safety_critical:
        count = coverage.get((vs, topo, term), 0)
        marker = " *** ZERO ***" if count == 0 else (" * TINY *" if count < 10 else "")
        print(f"{vs:<12} {topo:<35} {term:<20} {count:<6} {desc}{marker}")
        if count < 10:
            zero_support_cells.append({
                "verification_state": vs,
                "topology": topo,
                "terminal_regime": term,
                "count": count,
                "description": desc,
            })

    print(f"\n  Cells with < 10 support: {len(zero_support_cells)}")

    # Also check: does the training data have the verified_hyp_action feature?
    print("\n" + "=" * 70)
    print("Verified hypothesis action mapping in training data")
    print("=" * 70)

    vha_counts = defaultdict(int)
    for record in all_records:
        vstate = classify_verification_state(record)
        if vstate == "none":
            vha_counts["none"] += 1
            continue
        
        # Check if there's a uniquely verified-supported hypothesis
        evidence = record.get("evidence_items", [])
        hyp_verified_support = defaultdict(set)
        for ev in evidence:
            if not ev.get("retrieved", False):
                continue
            if ev.get("verification_state") != "SUFFICIENT":
                continue
            for h_id in ev.get("supports", []):
                hyp_verified_support[h_id].add(ev["evidence_id"])
        
        supported = [h_id for h_id, evs in hyp_verified_support.items() if len(evs) > 0]
        if len(supported) == 1:
            # Find the hypothesis's answer_action
            hyps = record.get("hypotheses", [])
            for h in hyps:
                if h.get("hypothesis_id") == supported[0]:
                    action = h.get("answer_action", "unknown")
                    vha_counts[f"unique_{action}"] += 1
                    break
            else:
                vha_counts["unique_unknown"] += 1
        elif len(supported) > 1:
            vha_counts["multiple"] += 1
        else:
            vha_counts["no_supported"] += 1

    print(f"\n  Verified hypothesis action distribution:")
    for k, v in sorted(vha_counts.items()):
        print(f"    {k}: {v}")

    # Save results
    results = {
        "total_records": len(all_records),
        "coverage_matrix": {str(k): v for k, v in coverage.items()},
        "zero_support_cells": zero_support_cells,
        "verified_hyp_action_distribution": dict(vha_counts),
    }
    with open(OUTPUT_DIR / "coverage_matrix.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_DIR / 'coverage_matrix.json'}")


if __name__ == "__main__":
    main()
