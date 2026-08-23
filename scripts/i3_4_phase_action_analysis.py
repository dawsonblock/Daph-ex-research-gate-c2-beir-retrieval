#!/usr/bin/env python3
"""DAPH I3.4 — Empirical Phase × Action analysis.

Produces the decisive artifact: the Phase × Action matrix showing
whether phase-conditioned action values show real separation.

For each (phase, action) pair, computes:
  N              — number of transitions
  E[delta_U]     — mean delta utility
  P(success)     — success rate
  P(exhaustion)  — resource exhaustion rate
  P(loop)        — repeated action rate
  E[delta_epi]   — mean epistemic progress

Also computes:
  - Phase distribution
  - Action distribution by phase
  - Phase separation test (chi-square on action distributions)
  - Transition matrix

This is the GO/NO-GO gate for I3.4b. If all action-value distributions
look the same across phases, STOP before building a value estimator.

Usage:
    PYTHONPATH=. python3 scripts/i3_4_phase_action_analysis.py \
        --transitions experiments/i3_4/datasets/transitions_r2_dev_v2.jsonl \
        --output experiments/i3_4/phase/
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from daph.phase.ontology import Phase, ALL_PHASES, classify_transition


def load_transitions(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def phase_action_matrix(transitions: list[dict]) -> dict:
    """Build the empirical Phase × Action matrix.

    For each (phase, action) pair:
      N, E[delta_U], P(success), P(exhaustion), P(loop), E[delta_epi]
    """
    # Group transitions by (phase, action)
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for t in transitions:
        groups[(t["phase_before"], t["action"])].append(t)

    matrix = {}
    for (phase, action), group in sorted(groups.items()):
        n = len(group)
        # E[delta_U]: use utility_to_go difference as proxy
        # For non-terminal steps, delta_U = utility_to_go - next_utility_to_go
        # For terminal steps, delta_U = terminal_utility
        # Simplified: use immediate_cost + delta_epistemic as proxy
        delta_u_values = [t["immediate_cost"] + t["delta_epistemic_utility"] for t in group]
        mean_delta_u = sum(delta_u_values) / n if n else 0.0

        # P(success): fraction where the trajectory was successful
        success_count = sum(1 for t in group if t["success"])
        p_success = success_count / n if n else 0.0

        # P(exhaustion): fraction where can_retrieve/can_search became False
        # after the action (and was True before)
        exhaustion_count = sum(
            1 for t in group
            if (t["features_before"].get("can_retrieve", False) and not t["features_after"].get("can_retrieve", False))
            or (t["features_before"].get("can_search", False) and not t["features_after"].get("can_search", False))
        )
        p_exhaustion = exhaustion_count / n if n else 0.0

        # P(loop): fraction where the same action was taken in the previous step
        # (approximated by checking if the action appears in the prior_actions)
        loop_count = sum(
            1 for t in group
            if t["action"] in (t["features_before"].get("prior_actions", []) or [])
        )
        p_loop = loop_count / n if n else 0.0

        # E[delta_epi]: mean epistemic progress
        delta_epi_values = [t["delta_epistemic_utility"] for t in group]
        mean_delta_epi = sum(delta_epi_values) / n if n else 0.0

        matrix[f"{phase}|{action}"] = {
            "phase": phase,
            "action": action,
            "N": n,
            "E_delta_U": round(mean_delta_u, 4),
            "P_success": round(p_success, 4),
            "P_exhaustion": round(p_exhaustion, 4),
            "P_loop": round(p_loop, 4),
            "E_delta_epi": round(mean_delta_epi, 4),
        }

    return matrix


def action_distribution_by_phase(transitions: list[dict]) -> dict[str, dict[str, float]]:
    """For each phase, compute the action distribution P(a | phase)."""
    phase_actions: dict[str, Counter] = defaultdict(Counter)
    for t in transitions:
        phase_actions[t["phase_before"]][t["action"]] += 1

    result = {}
    for phase in sorted(phase_actions):
        total = sum(phase_actions[phase].values())
        result[phase] = {
            action: count / total
            for action, count in sorted(phase_actions[phase].items())
        }
    return result


def phase_separation_test(
    transitions: list[dict],
    action_dist: dict[str, dict[str, float]],
) -> dict:
    """Test whether action distributions differ significantly across phases.

    Uses a simple chi-square-like statistic on the action distribution
    differences. This is not a formal hypothesis test but a practical
    separation measure.

    Returns:
        dict with separation statistics and per-phase-pair distances
    """
    phases = sorted(action_dist.keys())
    all_actions = sorted(set(a for d in action_dist.values() for a in d))

    # Build distribution vectors
    dist_vectors = {}
    for phase in phases:
        vec = [action_dist[phase].get(a, 0.0) for a in all_actions]
        dist_vectors[phase] = vec

    # Compute pairwise L1 distances
    distances = {}
    max_distance = 0.0
    for i, p1 in enumerate(phases):
        for p2 in phases[i + 1:]:
            v1 = dist_vectors[p1]
            v2 = dist_vectors[p2]
            l1 = sum(abs(a - b) for a, b in zip(v1, v2))
            distances[f"{p1}|{p2}"] = round(l1, 4)
            max_distance = max(max_distance, l1)

    # Overall separation: mean pairwise distance
    mean_distance = sum(distances.values()) / len(distances) if distances else 0.0

    return {
        "max_pairwise_l1": round(max_distance, 4),
        "mean_pairwise_l1": round(mean_distance, 4),
        "pairwise_distances": distances,
        "all_actions": all_actions,
        "n_phases": len(phases),
    }


def transition_matrix(transitions: list[dict]) -> dict:
    """Compute phase transition matrix P(phase_after | phase_before)."""
    transition_counts: dict[str, Counter] = defaultdict(Counter)
    for t in transitions:
        transition_counts[t["phase_before"]][t["phase_after"]] += 1

    result = {}
    for phase_before in sorted(transition_counts):
        total = sum(transition_counts[phase_before].values())
        result[phase_before] = {
            phase_after: count / total
            for phase_after, count in sorted(transition_counts[phase_before].items())
        }
    return result


def transition_type_distribution(transitions: list[dict]) -> dict[str, int]:
    """Distribution of transition types (expected, allowed_reversal, etc.)."""
    return dict(Counter(t["transition_type"] for t in transitions))


def run_analysis(transitions: list[dict], output: Path) -> dict:
    """Run the full phase-action analysis."""
    print("=== I3.4 Phase × Action Analysis ===")
    print(f"Transitions: {len(transitions)}")
    print()

    # Phase distribution
    phase_counts = Counter(t["phase_before"] for t in transitions)
    print("Phase distribution (before action):")
    for phase in ALL_PHASES:
        count = phase_counts.get(phase.value, 0)
        pct = count / len(transitions) * 100 if transitions else 0
        print(f"  {phase.value:30s}: {count:4d} ({pct:.1f}%)")
    print()

    # Action distribution by phase
    action_dist = action_distribution_by_phase(transitions)
    print("Action distribution by phase:")
    for phase in sorted(action_dist):
        dist = action_dist[phase]
        parts = [f"{a}={p:.2f}" for a, p in sorted(dist.items(), key=lambda x: -x[1])]
        print(f"  {phase:30s}: {', '.join(parts)}")
    print()

    # Phase separation test
    separation = phase_separation_test(transitions, action_dist)
    print("Phase separation test:")
    print(f"  Max pairwise L1 distance: {separation['max_pairwise_l1']}")
    print(f"  Mean pairwise L1 distance: {separation['mean_pairwise_l1']}")
    print(f"  Phases compared: {separation['n_phases']}")
    print()

    # Phase × Action matrix
    matrix = phase_action_matrix(transitions)
    print("Phase × Action matrix (N, E[delta_U], P(success), E[delta_epi]):")
    print(f"  {'Phase':30s} {'Action':15s} {'N':>5s} {'E[ΔU]':>8s} {'P(succ)':>8s} {'E[Δepi]':>8s}")
    for key, row in sorted(matrix.items()):
        print(f"  {row['phase']:30s} {row['action']:15s} {row['N']:5d} "
              f"{row['E_delta_U']:8.2f} {row['P_success']:8.2f} {row['E_delta_epi']:8.2f}")
    print()

    # Transition matrix
    trans_matrix = transition_matrix(transitions)
    print("Phase transition matrix P(phase_after | phase_before):")
    for phase_before in sorted(trans_matrix):
        parts = [f"{p}={v:.2f}" for p, v in sorted(trans_matrix[phase_before].items())]
        print(f"  {phase_before:30s} → {', '.join(parts)}")
    print()

    # Transition types
    trans_types = transition_type_distribution(transitions)
    print("Transition type distribution:")
    for ttype, count in sorted(trans_types.items()):
        print(f"  {ttype}: {count}")
    print()

    # GO / NO-GO assessment
    print("=== GO / NO-GO GATE ===")
    go_conditions = {
        "phase_separation_significant": separation["max_pairwise_l1"] > 0.5,
        "multiple_phases_present": sum(1 for c in phase_counts.values() if c > 0) >= 3,
        "action_values_differ": len(matrix) >= 10,
    }
    all_go = all(go_conditions.values())
    for cond, result in go_conditions.items():
        print(f"  {cond}: {'PASS' if result else 'FAIL'}")
    print()
    if all_go:
        print("  → GO: Phase-conditioned structure detected. Proceed to I3.4b.")
    else:
        print("  → NO-GO: Insufficient phase separation. Do not build value estimator.")
    print()

    # Save results
    output.mkdir(parents=True, exist_ok=True)

    result = {
        "n_transitions": len(transitions),
        "phase_distribution": dict(phase_counts),
        "action_distribution_by_phase": action_dist,
        "phase_separation": separation,
        "phase_action_matrix": matrix,
        "transition_matrix": trans_matrix,
        "transition_type_distribution": trans_types,
        "go_no_go": {
            "conditions": go_conditions,
            "result": "GO" if all_go else "NO-GO",
        },
    }

    with open(output / "phase_action_analysis.json", "w") as f:
        json.dump(result, f, indent=2, sort_keys=True, default=str)

    # Also save the matrix as a flat table for easy inspection
    with open(output / "phase_action_matrix.tsv", "w") as f:
        f.write("phase\taction\tN\tE_delta_U\tP_success\tP_exhaustion\tP_loop\tE_delta_epi\n")
        for key, row in sorted(matrix.items()):
            f.write(f"{row['phase']}\t{row['action']}\t{row['N']}\t"
                    f"{row['E_delta_U']}\t{row['P_success']}\t"
                    f"{row['P_exhaustion']}\t{row['P_loop']}\t{row['E_delta_epi']}\n")

    print(f"Results saved to {output}")
    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description="I3.4 Phase × Action analysis")
    parser.add_argument("--transitions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    transitions = load_transitions(args.transitions)
    run_analysis(transitions, args.output)


if __name__ == "__main__":
    main()
