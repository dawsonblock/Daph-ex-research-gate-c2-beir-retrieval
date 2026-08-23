#!/usr/bin/env python3
"""
R2-DEV-V2 Balanced Dataset Selector.

Constructs an explicitly balanced structural set for R2-DEV-V2, NOT
the first N tasks. Ensures coverage of all required gold categories:

    Gold state                           Purpose
    ---------------------------------------------------------------
    true T2 / VERIFY dead-end            D should gate
    one live hypothesis                  D must not gate
    >=2 live hypotheses / discrimination D must not gate
    false-contradiction inferred T2      measures upstream false gating
    missed contradiction                 measures missed structural dead-end
    retrieval available                  replacement-action behavior
    retrieval exhausted                  tests fallback behavior
    search available                     replacement-action behavior
    search exhausted                     tests fallback behavior
    DEFER control                        termination calibration
    ANSWER control                       answer preservation

The selector draws from the existing 800-task dataset and synthesizes
budget-exhausted variants by duplicating tasks with modified budget
fields. The actual budget enforcement happens at runtime in the
executive loop.

Output: r2_dev_v2_balanced_dataset.jsonl
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


# Target counts per gold category for R2-DEV-V2
# Small enough for one Colab session, large enough for mechanism analysis
TARGET_COUNTS = {
    # True T2 cases (D should gate)
    "T2_IMMEDIATE": 5,
    "T2_LATE_1": 5,
    "T2_LATE_2": 5,

    # One live hypothesis (D must not gate)
    "ONE_LIVE_NEAR_BOUNDARY": 5,

    # Two live hypotheses (D must not gate)
    "TWO_LIVE_DISCRIMINATION": 5,

    # Semantic error cases
    "FALSE_CONTRADICTION": 5,
    "MISSED_CONTRADICTION": 5,

    # False T2 / non-trigger
    "T2_LATE_3_NONTRIGGER": 5,

    # Matched negatives
    "MATCHED_NEG_IMMEDIATE": 5,
    "MATCHED_NEG_LATE": 5,

    # Controls
    "DEFER_CONTROL": 5,
    "ANSWER_CONTROL": 5,
}

# Budget variants to synthesize from T2 cases
# Each T2 task gets duplicated with retrieval/search exhausted
BUDGET_VARIANTS = {
    "retrieval_available": {"retrieval_budget_case": "available", "search_budget_case": "available"},
    "retrieval_exhausted": {"retrieval_budget_case": "exhausted", "search_budget_case": "available"},
    "search_exhausted": {"retrieval_budget_case": "available", "search_budget_case": "exhausted"},
    "both_exhausted": {"retrieval_budget_case": "exhausted", "search_budget_case": "exhausted"},
}


def select_balanced_dataset(
    source_dataset_path: Path,
    seed: int = 137,
) -> list[dict]:
    """Select a balanced structural subset from the full dataset.

    Also synthesizes budget-exhausted variants for T2 cases.
    """
    with open(source_dataset_path) as f:
        all_tasks = [json.loads(line) for line in f]

    rng = random.Random(seed)

    # Group by stratum
    by_stratum: dict[str, list[dict]] = {}
    for task in all_tasks:
        stratum = task["stratum"]
        by_stratum.setdefault(stratum, []).append(task)

    selected = []

    # Select from each stratum
    for stratum, count in TARGET_COUNTS.items():
        pool = by_stratum.get(stratum, [])
        if len(pool) < count:
            print(f"WARNING: stratum {stratum} has only {len(pool)} tasks, "
                  f"need {count}. Using all available.")
            chosen = pool
        else:
            chosen = rng.sample(pool, count)

        for task in chosen:
            selected.append(dict(task))

    # Synthesize budget-exhausted variants for T2 cases
    # Take T2_IMMEDIATE tasks and create budget variants
    t2_tasks = [t for t in selected if t["gold_t2"] and t["stratum"] in ("T2_IMMEDIATE", "T2_LATE_1")]
    budget_variants = []

    for task in t2_tasks[:5]:  # Take first 5 T2 tasks
        for variant_name, budget_overrides in BUDGET_VARIANTS.items():
            variant = dict(task)
            variant.update(budget_overrides)
            variant["task_id"] = f"{task['task_id']}__{variant_name}"
            variant["stratum"] = f"{task['stratum']}__{variant_name}"
            budget_variants.append(variant)

    selected.extend(budget_variants)

    return selected


def print_dataset_summary(tasks: list[dict]):
    """Print a summary of the selected dataset."""
    print(f"\nTotal tasks: {len(tasks)}")
    print()

    strata = Counter(t["stratum"] for t in tasks)
    print("Strata:")
    for s, c in sorted(strata.items()):
        print(f"  {s}: {c}")

    print()
    print("Gold T2:", Counter(t["gold_t2"] for t in tasks))
    print("Gold should gate verify:", Counter(t["gold_should_gate_verify"] for t in tasks))
    print("Gold n_live:", Counter(t["gold_n_live"] for t in tasks))
    print("Retrieval budget:", Counter(t["retrieval_budget_case"] for t in tasks))
    print("Search budget:", Counter(t["search_budget_case"] for t in tasks))
    print("Expected terminal:", Counter(t["expected_terminal"] for t in tasks))

    # Verify all required gold categories are covered
    print()
    print("=== Gold category coverage ===")
    coverage = {
        "true T2 / VERIFY dead-end (D should gate)": (
            sum(1 for t in tasks if t["gold_t2"] and t["gold_should_gate_verify"]) > 0
        ),
        "one live hypothesis (D must not gate)": (
            sum(1 for t in tasks if t["gold_n_live"] == 1 and not t["gold_t2"]) > 0
        ),
        ">=2 live hypotheses (D must not gate)": (
            sum(1 for t in tasks if t["stratum"] == "TWO_LIVE_DISCRIMINATION") > 0
        ),
        "false-contradiction inferred T2": (
            sum(1 for t in tasks if t.get("semantic_error_class") == "FALSE_CONTRADICTION") > 0
        ),
        "missed contradiction": (
            sum(1 for t in tasks if t.get("semantic_error_class") == "MISSED_CONTRADICTION") > 0
        ),
        "retrieval available": (
            sum(1 for t in tasks if t["retrieval_budget_case"] == "available") > 0
        ),
        "retrieval exhausted": (
            sum(1 for t in tasks if t["retrieval_budget_case"] == "exhausted") > 0
        ),
        "search available": (
            sum(1 for t in tasks if t["search_budget_case"] == "available") > 0
        ),
        "search exhausted": (
            sum(1 for t in tasks if t["search_budget_case"] == "exhausted") > 0
        ),
        "DEFER control": (
            sum(1 for t in tasks if t["stratum"] == "DEFER_CONTROL") > 0
        ),
        "ANSWER control": (
            sum(1 for t in tasks if t["stratum"] == "ANSWER_CONTROL") > 0
        ),
    }

    all_covered = True
    for cat, covered in coverage.items():
        status = "OK" if covered else "MISSING"
        print(f"  [{status}] {cat}")
        if not covered:
            all_covered = False

    print()
    if all_covered:
        print("All required gold categories covered.")
    else:
        print("WARNING: Some gold categories missing!")

    return all_covered


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="R2-DEV-V2 Balanced Dataset Selector")
    parser.add_argument("--source", type=Path,
                        default=REPO_ROOT / "experiments/v2b_i3_15c/development/r2-dev/r2_dataset.jsonl",
                        help="Path to source R2 dataset JSONL")
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "experiments/v2b_i3_15c/development/r2-dev-v2/balanced_dataset.jsonl",
                        help="Output path for balanced dataset JSONL")
    parser.add_argument("--seed", type=int, default=137,
                        help="Random seed for task selection")
    args = parser.parse_args()

    print(f"Source dataset: {args.source}")
    print(f"Output: {args.output}")
    print(f"Seed: {args.seed}")

    tasks = select_balanced_dataset(args.source, seed=args.seed)
    all_covered = print_dataset_summary(tasks)

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for task in tasks:
            f.write(json.dumps(task, sort_keys=True) + "\n")

    print(f"\nDataset written to: {args.output}")
    print(f"Total tasks: {len(tasks)}")

    # With 4 arms (C0/D/E/DE), total trajectories
    print(f"Total trajectories (4 arms): {len(tasks) * 4}")


if __name__ == "__main__":
    main()
