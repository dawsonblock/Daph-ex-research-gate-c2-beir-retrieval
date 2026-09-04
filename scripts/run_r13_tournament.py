#!/usr/bin/env python3
"""R13 Stage 6: Execute full matched-state operator tournament.

For every frozen checkpoint s_i, execute every admissible operator a ∈ A_0.
Record U_terminal(s_i, a) and C(s_i, a).

This is the core R13-A experiment. All operators start from the exact
same serialized state, ensuring fair comparison.

Usage:
    python scripts/run_r13_tournament.py --checkpoints experiments/daph_x/r13/r13a_checkpoints.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from daph_x.operators.base import CheckpointState, Observation, compute_normalized_cost
from daph_x.operators.stop import StopOperator
from daph_x.operators.sample import SampleStandardOperator
from daph_x.operators.diverse import SampleDiverseOperator
from daph_x.operators.critique import CritiqueRetryOperator
from daph_x.operators.verify import VerifyTargetedOperator
from daph_x.coding.reasoning_tasks import check_answer


def load_checkpoints(path: Path) -> list:
    """Load frozen checkpoints."""
    checkpoints = []
    with open(path) as f:
        for line in f:
            checkpoints.append(json.loads(line))
    return checkpoints


def restore_state(state_dict: dict) -> CheckpointState:
    """Restore a CheckpointState from serialized form."""
    return CheckpointState(
        task_id=state_dict["task_id"],
        task_prompt=state_dict["task_prompt"],
        correct_answer=state_dict["correct_answer"],
        answer_type=state_dict["answer_type"],
        difficulty=state_dict["difficulty"],
        category=state_dict["category"],
        candidates=state_dict["candidates"],
        k=state_dict["k"],
        features=state_dict["features"],
        maxcal_answer=state_dict["maxcal_answer"],
        maxcal_correct=state_dict["maxcal_correct"],
        maxcal_confidence=state_dict["maxcal_confidence"],
    )


def evaluate_terminal_utility(
    state: CheckpointState, obs: Observation,
    all_candidates: list,
) -> dict:
    """Evaluate terminal utility after applying operator.

    After an operator executes, we have new candidates. We recompute
    MaxCal over the combined set (original + new) and check correctness.

    For STOP, the utility is just the current MaxCal correctness.
    """
    if obs.operator_name == "STOP":
        return {
            "terminal_answer": state.maxcal_answer,
            "terminal_correct": state.maxcal_correct,
            "terminal_k": state.k,
            "n_candidates_after": state.k,
        }

    # Get new candidates from the observation
    new_cands = []
    if "new_candidates" in obs.evidence:
        new_cands = obs.evidence["new_candidates"]
    elif obs.evidence.get("new_candidate"):
        new_cands = [obs.evidence["new_candidate"]]

    # Combine original + new candidates
    combined = list(state.candidates) + new_cands

    # Recompute MaxCal (majority vote)
    from collections import Counter
    answers = [c["answer"] for c in combined]
    answer_counts = Counter(answers)
    majority_answer = answer_counts.most_common(1)[0][0]

    # Check if majority answer is correct
    majority_cand = next((c for c in combined if c["answer"] == majority_answer), None)
    terminal_correct = majority_cand["is_correct"] if majority_cand else False

    return {
        "terminal_answer": majority_answer,
        "terminal_correct": terminal_correct,
        "terminal_k": len(combined),
        "n_candidates_after": len(combined),
        "n_new_candidates": len(new_cands),
    }


def run_tournament(checkpoints: list, operators: list, output_path: Path):
    """Execute every operator from every checkpoint."""
    results = []
    n = len(checkpoints)
    n_ops = len(operators)

    print(f"R13 Tournament: {n} checkpoints × {n_ops} operators = {n * n_ops} executions")

    # Group checkpoints by task_id to share model loading
    from collections import defaultdict
    by_task = defaultdict(list)
    for i, cp in enumerate(checkpoints):
        by_task[cp["state"]["task_id"]].append((i, cp))

    t_start = time.monotonic()

    for task_idx, (task_id, cps) in enumerate(by_task.items()):
        print(f"\n[{task_idx+1}/{len(by_task)}] {task_id} ({len(cps)} checkpoints)")

        for cp_idx, cp in cps:
            state = restore_state(cp["state"])
            baseline_correct = state.maxcal_correct

            for op in operators:
                if not op.is_admissible(state):
                    results.append({
                        "checkpoint_idx": cp_idx,
                        "task_id": state.task_id,
                        "k": state.k,
                        "operator": op.name,
                        "admissible": False,
                        "executed": False,
                    })
                    continue

                try:
                    t0 = time.monotonic()
                    obs = op.execute(state)
                    exec_time = time.monotonic() - t0

                    terminal = evaluate_terminal_utility(state, obs, [])

                    # Classify event
                    if obs.operator_name == "STOP":
                        event = "NEUTRAL"  # No change
                    elif terminal["terminal_correct"] and not baseline_correct:
                        event = "RESCUE"
                    elif not terminal["terminal_correct"] and baseline_correct:
                        event = "BREAK"
                    elif terminal["terminal_correct"] == baseline_correct:
                        event = "WASTE"
                    else:
                        event = "UNKNOWN"

                    result = {
                        "checkpoint_idx": cp_idx,
                        "task_id": state.task_id,
                        "k": state.k,
                        "operator": op.name,
                        "admissible": True,
                        "executed": True,
                        "baseline_correct": baseline_correct,
                        "terminal_correct": terminal["terminal_correct"],
                        "event": event,
                        "terminal_answer": terminal["terminal_answer"],
                        "n_candidates_after": terminal["n_candidates_after"],
                        "n_new_candidates": terminal.get("n_new_candidates", 0),
                        "cost": obs.cost.to_dict(),
                        "exec_time_s": exec_time,
                        "observation": obs.to_dict(),
                        "state_features": state.features,
                        "classification": cp["classification"],
                    }
                    results.append(result)

                    status = {"RESCUE": "RESCUE!", "BREAK": "BREAK!", "WASTE": "waste", "NEUTRAL": "stop"}[event]
                    print(f"  K={state.k} {op.name:20s}: {status:8s} ({exec_time:.1f}s)")

                except Exception as e:
                    import traceback
                    print(f"  K={state.k} {op.name:20s}: ERROR — {e}")
                    results.append({
                        "checkpoint_idx": cp_idx,
                        "task_id": state.task_id,
                        "k": state.k,
                        "operator": op.name,
                        "admissible": True,
                        "executed": False,
                        "error": str(e),
                    })

        # Save after each task
        with open(output_path, "w") as f:
            for r in results:
                f.write(json.dumps(r, default=str) + "\n")

    elapsed = time.monotonic() - t_start
    print(f"\nTournament complete: {len(results)} results in {elapsed:.0f}s")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", default="experiments/daph_x/r13/r13a_checkpoints.jsonl")
    parser.add_argument("--output", default="experiments/daph_x/r13/r13a_tournament.jsonl")
    parser.add_argument("--operators", default="all")
    args = parser.parse_args()

    cp_path = REPO_ROOT / args.checkpoints
    out_path = REPO_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoints = load_checkpoints(cp_path)
    print(f"Loaded {len(checkpoints)} checkpoints")

    # Initialize operators (shared model to avoid reloading)
    from daph_x.coding.model_interface import CodingModelInterface
    model = CodingModelInterface(
        model_path="/Users/dawsonblock/Downloads/qwen_gguf/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        n_gpu_layers=-1, seed=42,
    )

    all_operators = [
        StopOperator(),
        SampleStandardOperator(model=model),
        SampleDiverseOperator(model=model),
        CritiqueRetryOperator(model=model),
        VerifyTargetedOperator(model=model),
    ]

    if args.operators == "all":
        operators = all_operators
    else:
        op_names = args.operators.split(",")
        name_to_op = {op.name: op for op in all_operators}
        operators = [name_to_op[n] for n in op_names if n in name_to_op]

    print(f"Operators: {[op.name for op in operators]}")
    print()

    results = run_tournament(checkpoints, operators, out_path)

    # Quick summary
    from collections import Counter
    events = Counter(r.get("event", "N/A") for r in results if r.get("executed"))
    by_op = {}
    for r in results:
        if not r.get("executed"):
            continue
        op = r["operator"]
        by_op.setdefault(op, Counter())[r.get("event", "N/A")] += 1

    print(f"\n=== TOURNAMENT SUMMARY ===")
    print(f"  Total executions: {sum(1 for r in results if r.get('executed'))}")
    print(f"  Events: {dict(events)}")
    print(f"\n  By operator:")
    for op, counts in sorted(by_op.items()):
        total = sum(counts.values())
        print(f"    {op:20s}: {dict(counts)} (n={total})")


if __name__ == "__main__":
    main()
