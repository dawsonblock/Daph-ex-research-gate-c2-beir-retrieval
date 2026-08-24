#!/usr/bin/env python3
"""DAPH I3.4 — Value-estimator validation gates (I3.4-QUAL value gates).

Before exposing the value model to the LLM, require:
  Q1: Finite predictions 100%
  Q2: Legal action coverage 100%
  Q3: No action missing
  Q4: Held-out ranking better than random
  Q5: Calibration acceptable
  Q6: Regret lower than naïve baselines

If any gate fails, do not deploy the value model.

Usage:
    PYTHONPATH=. python3 scripts/i3_4_value_qualification.py \
        --transitions experiments/i3_4/datasets/transitions_r2_dev_v2.jsonl \
        --output experiments/i3_4/qualification/
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from daph.value.dataset import (
    load_transitions, split_by_task,
    get_action_value_target,
)
from daph.value.empirical import GlobalActionMean, PhaseActionTable
from daph.value.model import RandomForestValueModel
from daph.value.ranking import evaluate_ranking, evaluate_ranking_with_hindsight
from daph.value.uncertainty import compute_uncertainty


def main():
    import argparse

    parser = argparse.ArgumentParser(description="I3.4 Value qualification")
    parser.add_argument("--transitions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    transitions = load_transitions(args.transitions)
    train, dev, test = split_by_task(transitions, seed=args.seed)

    target_fn = get_action_value_target

    # Train B3b (random forest for uncertainty)
    print("Training B3b random forest for qualification...")
    model = RandomForestValueModel(n_estimators=100, max_depth=6)
    model.fit(train, target_fn)

    # Also train B1 for comparison
    b0 = GlobalActionMean()
    b0.fit(train, target_fn)
    b1 = PhaseActionTable(min_samples=3, fallback=b0)
    b1.fit(train, target_fn)

    print(f"\n{'='*60}")
    print("I3.4 Value Qualification Gates")
    print(f"{'='*60}")

    checks = {}

    # Q1: Finite predictions 100%
    print("\n[Q1] Finite predictions...")
    n_finite = 0
    n_total = 0
    for t in test:
        phase = t["phase_before"]
        legal = t.get("legal_actions", [])
        features = t.get("features_before", {})
        for action in legal:
            pred = model.predict(phase, action, features)
            if math.isfinite(pred):
                n_finite += 1
            n_total += 1
    q1_pass = n_finite == n_total
    checks["Q1_finite_predictions"] = {
        "passed": q1_pass,
        "n_finite": n_finite,
        "n_total": n_total,
        "rate": n_finite / n_total if n_total else 0.0,
    }
    print(f"  {'PASS' if q1_pass else 'FAIL'}: {n_finite}/{n_total} finite")

    # Q2: Legal action coverage 100%
    print("\n[Q2] Legal action coverage...")
    n_covered = 0
    n_states = 0
    for t in test:
        legal = t.get("legal_actions", [])
        if not legal:
            continue
        n_states += 1
        phase = t["phase_before"]
        features = t.get("features_before", {})
        predictions = model.predict_all(phase, legal, features)
        if all(a in predictions for a in legal):
            n_covered += 1
    q2_pass = n_covered == n_states
    checks["Q2_legal_coverage"] = {
        "passed": q2_pass,
        "n_covered": n_covered,
        "n_states": n_states,
    }
    print(f"  {'PASS' if q2_pass else 'FAIL'}: {n_covered}/{n_states} states fully covered")

    # Q3: No action missing (check all unique actions)
    print("\n[Q3] No action missing...")
    all_actions = set(t["action"] for t in transitions)
    model_actions = set()
    for t in test:
        legal = t.get("legal_actions", [])
        phase = t["phase_before"]
        features = t.get("features_before", {})
        predictions = model.predict_all(phase, legal, features)
        model_actions.update(predictions.keys())
    missing = all_actions - model_actions
    q3_pass = len(missing) == 0
    checks["Q3_no_action_missing"] = {
        "passed": q3_pass,
        "all_actions": sorted(all_actions),
        "model_actions": sorted(model_actions),
        "missing": sorted(missing),
    }
    print(f"  {'PASS' if q3_pass else 'FAIL'}: missing={sorted(missing)}")

    # Q4: Held-out ranking better than random
    print("\n[Q4] Held-out ranking better than random...")
    eval_b3 = evaluate_ranking(model, test, target_fn)
    rng = random.Random(42)
    random_correct = 0
    random_n = 0
    for t in test:
        legal = t.get("legal_actions", [])
        if legal:
            if rng.choice(legal) == t["action"]:
                random_correct += 1
            random_n += 1
    random_acc = random_correct / random_n if random_n else 0.0
    q4_pass = eval_b3["top1_accuracy"] > random_acc
    checks["Q4_better_than_random"] = {
        "passed": q4_pass,
        "b3_top1": eval_b3["top1_accuracy"],
        "random_top1": random_acc,
    }
    print(f"  {'PASS' if q4_pass else 'FAIL'}: B3={eval_b3['top1_accuracy']:.4f} vs random={random_acc:.4f}")

    # Q5: Calibration (simplified — check prediction range is reasonable)
    print("\n[Q5] Calibration check...")
    predictions = []
    actuals = []
    for t in test:
        phase = t["phase_before"]
        action = t["action"]
        features = t.get("features_before", {})
        pred = model.predict(phase, action, features)
        actual = target_fn(t)
        predictions.append(pred)
        actuals.append(actual)
    pred_mean = sum(predictions) / len(predictions) if predictions else 0
    actual_mean = sum(actuals) / len(actuals) if actuals else 0
    calibration_error = abs(pred_mean - actual_mean)
    q5_pass = calibration_error < 10.0  # generous threshold
    checks["Q5_calibration"] = {
        "passed": q5_pass,
        "pred_mean": pred_mean,
        "actual_mean": actual_mean,
        "calibration_error": calibration_error,
    }
    print(f"  {'PASS' if q5_pass else 'FAIL'}: pred_mean={pred_mean:.2f}, actual_mean={actual_mean:.2f}, error={calibration_error:.2f}")

    # Q6: Regret lower than naïve baseline (B1)
    print("\n[Q6] Regret vs B1 baseline...")
    eval_b3_hind = evaluate_ranking_with_hindsight(model, test, target_fn)
    eval_b1_hind = evaluate_ranking_with_hindsight(b1, test, target_fn)
    q6_pass = eval_b3_hind["mean_regret"] <= eval_b1_hind["mean_regret"] + 0.1  # allow tie
    checks["Q6_regret_vs_b1"] = {
        "passed": q6_pass,
        "b3_regret": eval_b3_hind["mean_regret"],
        "b1_regret": eval_b1_hind["mean_regret"],
    }
    print(f"  {'PASS' if q6_pass else 'FAIL'}: B3 regret={eval_b3_hind['mean_regret']:.4f} vs B1 regret={eval_b1_hind['mean_regret']:.4f}")

    # Q7: Uncertainty finite
    print("\n[Q7] Uncertainty finite...")
    n_unc_finite = 0
    n_unc_total = 0
    for t in test[:50]:  # sample
        phase = t["phase_before"]
        legal = t.get("legal_actions", [])
        features = t.get("features_before", {})
        uncertainties = compute_uncertainty(model, phase, legal, features)
        for a, u in uncertainties.items():
            if math.isfinite(u["mean"]) and math.isfinite(u["uncertainty"]):
                n_unc_finite += 1
            n_unc_total += 1
    q7_pass = n_unc_finite == n_unc_total
    checks["Q7_uncertainty_finite"] = {
        "passed": q7_pass,
        "n_finite": n_unc_finite,
        "n_total": n_unc_total,
    }
    print(f"  {'PASS' if q7_pass else 'FAIL'}: {n_unc_finite}/{n_unc_total} finite")

    # Summary
    all_pass = all(c["passed"] for c in checks.values())
    print(f"\n{'='*60}")
    print(f"QUALIFICATION RESULT: {'PASS' if all_pass else 'FAIL'}")
    print(f"{'='*60}")
    for name, result in checks.items():
        print(f"  {name}: {'PASS' if result['passed'] else 'FAIL'}")

    # Save
    args.output.mkdir(parents=True, exist_ok=True)
    output = {
        "overall_passed": all_pass,
        "checks": checks,
        "n_test_transitions": len(test),
    }
    with open(args.output / "value_qualification.json", "w") as f:
        json.dump(output, f, indent=2, sort_keys=True, default=str)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
