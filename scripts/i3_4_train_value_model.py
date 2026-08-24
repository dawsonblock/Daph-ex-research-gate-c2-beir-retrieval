#!/usr/bin/env python3
"""DAPH I3.4b — Train and evaluate action-value models.

Trains the model ladder:
  B0: Global action mean
  B1: Phase × Action empirical table
  B2: Linear/Ridge regression
  B3: Gradient-boosted trees
  B3b: Random forest (with uncertainty)

Evaluates ranking quality on held-out tasks:
  - Top-1 accuracy (vs hindsight best)
  - Top-2 recall
  - Mean regret
  - Mean model top-1 utility vs hindsight best

Splits by task_id to prevent leakage. No trajectory from one task
can appear in multiple splits.

Usage:
    PYTHONPATH=. python3 scripts/i3_4_train_value_model.py \
        --transitions experiments/i3_4/datasets/transitions_r2_dev_v2.jsonl \
        --output experiments/i3_4/value/
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from daph.value.dataset import (
    load_transitions, split_by_task,
    get_action_value_target, get_epistemic_target, get_success_target,
    get_dataset_hash,
)
from daph.value.empirical import GlobalActionMean, PhaseActionTable
from daph.value.model import LinearValueModel, GBTValueModel, RandomForestValueModel
from daph.value.ranking import evaluate_ranking, evaluate_ranking_with_hindsight


def main():
    import argparse

    parser = argparse.ArgumentParser(description="I3.4b Train value models")
    parser.add_argument("--transitions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-frac", type=float, default=0.6)
    parser.add_argument("--dev-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load transitions
    print(f"Loading transitions from {args.transitions}")
    transitions = load_transitions(args.transitions)
    print(f"  {len(transitions)} transitions")

    # Split by task
    train, dev, test = split_by_task(
        transitions,
        train_frac=args.train_frac,
        dev_frac=args.dev_frac,
        seed=args.seed,
    )
    train_task_ids = set(t["task_id"] for t in train)
    dev_task_ids = set(t["task_id"] for t in dev)
    test_task_ids = set(t["task_id"] for t in test)

    print(f"\nTask-level split:")
    print(f"  Train: {len(train_task_ids)} tasks, {len(train)} transitions")
    print(f"  Dev:   {len(dev_task_ids)} tasks, {len(dev)} transitions")
    print(f"  Test:  {len(test_task_ids)} tasks, {len(test)} transitions")
    assert not (train_task_ids & dev_task_ids), "task leakage train→dev"
    assert not (train_task_ids & test_task_ids), "task leakage train→test"
    assert not (dev_task_ids & test_task_ids), "task leakage dev→test"
    print("  No task leakage detected.")

    # Target function
    target_fn = get_action_value_target

    # Train models
    print(f"\n{'='*60}")
    print("Training model ladder")
    print(f"{'='*60}")

    models = {}

    # B0: Global action mean
    print("\n[B0] Global action mean...")
    b0 = GlobalActionMean()
    b0.fit(train, target_fn)
    models["B0"] = b0
    print(f"  Values: {b0._values}")

    # B1: Phase × Action table
    print("\n[B1] Phase × Action empirical table...")
    b1 = PhaseActionTable(min_samples=3, fallback=b0)
    b1.fit(train, target_fn)
    models["B1"] = b1
    print(f"  Table:")
    for phase, actions in sorted(b1.table().items()):
        print(f"    {phase}: {actions}")

    # B2: Linear/Ridge
    print("\n[B2] Linear Ridge regression...")
    b2 = LinearValueModel()
    b2.fit(train, target_fn)
    models["B2"] = b2
    print(f"  Trained on {len(train)} samples, dim={b2._feature_dim}")

    # B3: Gradient-boosted trees
    print("\n[B3] Gradient-boosted trees...")
    b3 = GBTValueModel(n_estimators=100, max_depth=4)
    b3.fit(train, target_fn)
    models["B3"] = b3
    print(f"  Trained on {len(train)} samples")

    # B3b: Random forest (for uncertainty)
    print("\n[B3b] Random forest (uncertainty)...")
    b3b = RandomForestValueModel(n_estimators=100, max_depth=6)
    b3b.fit(train, target_fn)
    models["B3b"] = b3b
    print(f"  Trained on {len(train)} samples, {len(b3b._model.estimators_)} trees")

    # Evaluate on test set
    print(f"\n{'='*60}")
    print("Evaluation on held-out test set")
    print(f"{'='*60}")

    results = {}
    for name, model in models.items():
        print(f"\n[{name}] {model.name}")

        # Standard evaluation (no hindsight)
        eval_std = evaluate_ranking(model, test, target_fn)
        print(f"  Standard:")
        print(f"    Top-1 accuracy:  {eval_std['top1_accuracy']:.4f}")
        print(f"    Top-2 recall:    {eval_std['top2_recall']:.4f}")
        print(f"    Top-3 recall:    {eval_std['top3_recall']:.4f}")
        print(f"    Mean actual util:{eval_std['mean_actual_utility']:.4f}")

        # Hindsight evaluation (only on multi-action states)
        eval_hind = evaluate_ranking_with_hindsight(model, test, target_fn)
        print(f"  Hindsight (multi-action states):")
        print(f"    Multi-action states: {eval_hind['n_multi_action_states']}")
        print(f"    Evaluated:          {eval_hind['n_evaluated']}")
        print(f"    Top-1 accuracy:     {eval_hind['top1_accuracy']:.4f}")
        print(f"    Top-2 recall:       {eval_hind['top2_recall']:.4f}")
        print(f"    Mean model top-1:   {eval_hind['mean_model_top1_utility']:.4f}")
        print(f"    Mean hindsight best:{eval_hind['mean_hindsight_best']:.4f}")
        print(f"    Mean regret:        {eval_hind['mean_regret']:.4f}")

        results[name] = {
            "standard": eval_std,
            "hindsight": eval_hind,
        }

    # Compare against baselines
    print(f"\n{'='*60}")
    print("Baseline comparison")
    print(f"{'='*60}")

    # Random baseline
    import random
    rng = random.Random(42)
    random_top1 = 0
    random_n = 0
    for t in test:
        legal = t.get("legal_actions", [])
        actual = t["action"]
        if legal:
            random_pick = rng.choice(legal)
            if random_pick == actual:
                random_top1 += 1
            random_n += 1
    random_acc = random_top1 / random_n if random_n else 0.0
    print(f"\n  Random baseline:     Top-1 = {random_acc:.4f}")

    # Always-pick-most-common baseline
    from collections import Counter
    action_counts = Counter(t["action"] for t in train)
    most_common = action_counts.most_common(1)[0][0]
    always_top1 = sum(1 for t in test if t["action"] == most_common) / len(test)
    print(f"  Always-{most_common}:  Top-1 = {always_top1:.4f}")

    # Phase-frequency heuristic: pick most common action per phase
    phase_action_freq: dict[str, str] = {}
    phase_actions: dict[str, Counter] = defaultdict(Counter)
    for t in train:
        phase_actions[t["phase_before"]][t["action"]] += 1
    for phase, counter in phase_actions.items():
        phase_action_freq[phase] = counter.most_common(1)[0][0]
    phase_freq_top1 = sum(
        1 for t in test
        if phase_action_freq.get(t["phase_before"], "") == t["action"]
    ) / len(test)
    print(f"  Phase-frequency:     Top-1 = {phase_freq_top1:.4f}")

    # GO/NO-GO Gate B
    print(f"\n{'='*60}")
    print("GO / NO-GO Gate B")
    print(f"{'='*60}")

    b1_top1 = results["B1"]["standard"]["top1_accuracy"]
    b3_top1 = results["B3"]["standard"]["top1_accuracy"]
    b3_hind_top1 = results["B3"]["hindsight"]["top1_accuracy"]
    b3_regret = results["B3"]["hindsight"]["mean_regret"]

    gate_conditions = {
        "b3_beats_random": b3_top1 > random_acc,
        "b3_beats_phase_freq": b3_top1 > phase_freq_top1,
        "b3_beats_b1": b3_top1 > b1_top1,
        "b3_hindsight_top1_positive": b3_hind_top1 > 0.0,
        "b3_regret_finite": b3_regret == b3_regret,  # not NaN
    }

    all_pass = all(gate_conditions.values())
    for cond, result in gate_conditions.items():
        print(f"  {cond}: {'PASS' if result else 'FAIL'}")
    print()
    if all_pass:
        print("  → GO: Learned model beats baselines. Proceed to I3.4c.")
    else:
        print("  → NO-GO: Learned model does not beat baselines.")
        print("    Keep the simpler system (B1 phase×action table).")

    # Save results
    args.output.mkdir(parents=True, exist_ok=True)

    # Save model comparison
    output = {
        "dataset_hash": get_dataset_hash(transitions),
        "n_transitions": len(transitions),
        "split": {
            "train_tasks": len(train_task_ids),
            "dev_tasks": len(dev_task_ids),
            "test_tasks": len(test_task_ids),
            "train_transitions": len(train),
            "dev_transitions": len(dev),
            "test_transitions": len(test),
            "seed": args.seed,
        },
        "baselines": {
            "random_top1": random_acc,
            f"always_{most_common}_top1": always_top1,
            "phase_frequency_top1": phase_freq_top1,
        },
        "models": results,
        "b1_table": b1.table(),
        "b1_sample_counts": b1.sample_counts(),
        "go_no_go": {
            "conditions": gate_conditions,
            "result": "GO" if all_pass else "NO-GO",
        },
    }

    with open(args.output / "value_model_results.json", "w") as f:
        json.dump(output, f, indent=2, sort_keys=True, default=str)

    # Save B1 table as TSV
    with open(args.output / "b1_phase_action_table.tsv", "w") as f:
        f.write("phase\taction\tvalue\tn_samples\n")
        for (phase, action), value in sorted(b1._values.items()):
            n = b1._counts.get((phase, action), 0)
            f.write(f"{phase}\t{action}\t{value:.4f}\t{n}\n")

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
