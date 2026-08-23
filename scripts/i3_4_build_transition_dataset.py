#!/usr/bin/env python3
"""DAPH I3.4 — Build transition dataset from R2-DEV-V2 trajectories.

Converts mechanism receipts into (s_t, a_t, r_t, s_{t+1}) records with
phase classifications and feature vectors.

Each transition record contains:
  - trajectory_key, task_id, arm, step
  - phase_before, features_before
  - action, execution_outcome
  - phase_after, features_after
  - immediate_cost (step cost)
  - terminal_utility (from trajectory result)
  - delta_epistemic_utility (estimated)
  - success (terminal task success)
  - provenance: backend_id, experiment_id, dataset_id

Usage:
    PYTHONPATH=. python3 scripts/i3_4_build_transition_dataset.py \
        --receipts experiments/v2b_i3_15c/development/r2-dev-v2/merged/mechanism_receipts.jsonl \
        --results experiments/v2b_i3_15c/development/r2-dev-v2/merged/results.jsonl \
        --dataset experiments/v2b_i3_15c/development/r2-dev-v2/balanced_dataset.jsonl \
        --output experiments/i3_4/datasets/transitions_r2_dev_v2.jsonl
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from daph.phase.ontology import Phase, classify_transition
from daph.phase.features import PhaseFeatures, features_from_receipt
from daph.phase.classifier import classify_from_receipt, classify_phase


# Step costs (from resources.py DEFAULT_ACTION_COSTS, elapsed_ms only)
STEP_COSTS = {
    "ANSWER": 1.0,
    "RETRIEVE": 5.0,
    "VERIFY": 8.0,
    "SEARCH_MORE": 6.0,
    "REASON_MORE": 4.0,
    "DEFER": 1.0,
    "STOP": 1.0,
}


def build_transitions(
    receipts: list[dict],
    results: list[dict],
    dataset_sha: str,
    backend_id: str,
    experiment_id: str = "r2-dev-v2",
) -> list[dict]:
    """Build transition records from receipts and results.

    Groups receipts by trajectory_key, then creates one transition per
    consecutive receipt pair within each trajectory.
    """
    # Build result lookup
    result_lookup = {}
    for r in results:
        result_lookup[r["trajectory_key"]] = r

    # Group receipts by trajectory
    traj_receipts: dict[str, list[dict]] = defaultdict(list)
    for r in receipts:
        traj_receipts[r["trajectory_key"]].append(r)

    # Sort each trajectory's receipts by step
    for key in traj_receipts:
        traj_receipts[key].sort(key=lambda r: r.get("step", 0))

    transitions = []

    for traj_key, recs in traj_receipts.items():
        result = result_lookup.get(traj_key, {})
        task_id = recs[0].get("task_id", "")
        arm = recs[0].get("arm", "")
        terminal_utility = float(result.get("realized_utility", 0.0))
        success = bool(result.get("success", False))
        terminal_action = result.get("terminal_action", "")
        n_steps = len(recs)

        for i, rec in enumerate(recs):
            action = rec.get("selected_action", "")
            step = int(rec.get("step", i))

            # Phase before action
            phase_before = classify_from_receipt(rec)
            features_before = features_from_receipt(rec, phase=phase_before.phase.value)

            # Phase after action (from next receipt, or from post_state)
            if i + 1 < len(recs):
                next_rec = recs[i + 1]
                phase_after = classify_from_receipt(next_rec)
                features_after = features_from_receipt(next_rec, phase=phase_after.phase.value)
            else:
                # Terminal step — use post_state from current receipt
                post_state = rec.get("post_state", {})
                phase_after = classify_phase(
                    decision_state=post_state.get("decision_state", ""),
                    n_live=post_state.get("n_live", 0),
                    n_eliminated=post_state.get("n_eliminated", 0),
                    n_total=post_state.get("n_live", 0) + post_state.get("n_eliminated", 0),
                    t2=post_state.get("t2", False),
                )
                features_after = features_from_receipt(
                    {**rec, "step": step + 1, "n_live_hypotheses": post_state.get("n_live", 0),
                     "n_eliminated_hypotheses": post_state.get("n_eliminated", 0),
                     "decision_state_exposed": post_state.get("decision_state", ""),
                     "t2": post_state.get("t2", False)},
                    phase=phase_after.phase.value,
                )

            # Immediate cost (step cost)
            immediate_cost = -STEP_COSTS.get(action, 1.0)

            # Delta epistemic utility (simple proxy: change in n_live * weight)
            # Positive = progress (fewer live hypotheses or reached answer-ready)
            delta_epistemic = 0.0
            if features_before.n_live > features_after.n_live:
                delta_epistemic += (features_before.n_live - features_after.n_live) * 1.0
            if features_after.decision_state == "READY_TO_ANSWER" and features_before.decision_state != "READY_TO_ANSWER":
                delta_epistemic += 2.0
            if phase_after.phase == Phase.NO_VIABLE_HYPOTHESIS and phase_before.phase != Phase.NO_VIABLE_HYPOTHESIS:
                delta_epistemic += 1.0  # reaching terminal state is progress

            # Transition classification
            transition_type = classify_transition(phase_before.phase, phase_after.phase)

            # Utility-to-go: remaining utility from this step to terminal
            # For the last step, this is the terminal utility.
            # For earlier steps, we use the terminal utility minus accumulated costs.
            steps_from_here = n_steps - step - 1
            utility_to_go = terminal_utility  # simplified: full terminal utility

            transition = {
                # Provenance
                "trajectory_key": traj_key,
                "task_id": task_id,
                "arm": arm,
                "step": step,
                "backend_id": backend_id,
                "experiment_id": experiment_id,
                "dataset_id": dataset_sha,

                # State before
                "phase_before": phase_before.phase.value,
                "phase_before_confidence": phase_before.confidence,
                "features_before": features_before.as_dict(),

                # Action
                "action": action,
                "execution_outcome": rec.get("execution_outcome", ""),
                "selected_reason_code": rec.get("selected_reason_code", ""),
                "selected_target_id": rec.get("selected_target_id"),

                # State after
                "phase_after": phase_after.phase.value,
                "features_after": features_after.as_dict(),

                # Transition
                "transition_type": transition_type,

                # Rewards
                "immediate_cost": immediate_cost,
                "delta_epistemic_utility": delta_epistemic,
                "utility_to_go": utility_to_go,
                "terminal_utility": terminal_utility,
                "success": success,
                "terminal_action": terminal_action,
                "is_terminal_step": (i == n_steps - 1),

                # Legal context
                "legal_actions": rec.get("legal_actions", []),
                "allowed_actions": rec.get("allowed_actions", []),
                "can_verify_before": features_before.can_verify,
                "can_retrieve_before": features_before.can_retrieve,
                "can_search_before": features_before.can_search,
            }

            transitions.append(transition)

    return transitions


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build I3.4 transition dataset")
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend-id", type=str, default="qwen2.5-7b-instruct-q4_k_m")
    args = parser.parse_args()

    # Load inputs
    print(f"Loading receipts from {args.receipts}")
    with open(args.receipts) as f:
        receipts = [json.loads(line) for line in f]
    print(f"  {len(receipts)} receipts")

    print(f"Loading results from {args.results}")
    with open(args.results) as f:
        results = [json.loads(line) for line in f]
    print(f"  {len(results)} results")

    # Dataset SHA
    dataset_sha = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    print(f"Dataset SHA: {dataset_sha}")

    # Build transitions
    print("Building transitions...")
    transitions = build_transitions(
        receipts=receipts,
        results=results,
        dataset_sha=dataset_sha,
        backend_id=args.backend_id,
    )
    print(f"  {len(transitions)} transitions")

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for t in transitions:
            f.write(json.dumps(t, sort_keys=True) + "\n")
    print(f"Wrote {args.output}")

    # Summary stats
    from collections import Counter
    phase_counts = Counter(t["phase_before"] for t in transitions)
    action_counts = Counter(t["action"] for t in transitions)
    phase_action_counts = Counter((t["phase_before"], t["action"]) for t in transitions)

    print(f"\nPhase distribution (before action):")
    for phase in sorted(phase_counts):
        print(f"  {phase}: {phase_counts[phase]}")

    print(f"\nAction distribution:")
    for action in sorted(action_counts):
        print(f"  {action}: {action_counts[action]}")

    print(f"\nPhase × Action combinations: {len(phase_action_counts)}")


if __name__ == "__main__":
    main()
