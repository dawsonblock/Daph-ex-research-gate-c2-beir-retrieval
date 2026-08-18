#!/usr/bin/env python3
"""Train the Q^{π_B} intervention gate predictor for V2B-I3.5.3.

Builds a training dataset from two sources:
1. OFF trajectory data from I3.5.2c: Q^{π_B}(s, a_taken) = realized utility from s onward
2. I3.5.2d fork data: Q^{π_B}(s, a_G) = fork B realized utility (gov action + OFF continuation)

The target is the realized utility from state s onward under the OFF model policy.
Features are the controller-visible intervention features + action one-hot.

Trains a GradientBoostingRegressor and performs fold-isolated cross-validation.

Usage:
    python scripts/train_q_pib_gate.py \\
        --i352c-results experiments/v2b_i3_5_2/development/i352c_55f93130e87c/results.json \\
        --i352d-results experiments/v2b_i3_5_2/development/i352d/intervention_values_v1.jsonl \\
        --output-dir experiments/v2b_i3_5_2/development/i352d
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hrm_adaptive_memory.cognitive_control.core import DecisionAction
from hrm_adaptive_memory.cognitive_control.state import DecisionSummary
from hrm_adaptive_memory.executive.executor import (
    DeterministicActionExecutor, initial_runtime as init_task_runtime,
)
from hrm_adaptive_memory.executive.governor.assessor import GeneralGovernor
from hrm_adaptive_memory.executive.i3_5_1.conditions import ConditionID, get_condition
from hrm_adaptive_memory.executive.i3_5_1.observation_builder import build_observation
from hrm_adaptive_memory.executive.i3_5_1.trajectory_runner import _I3TaskAdapter
from hrm_adaptive_memory.executive.metareasoning_benchmark import (
    load_metareasoning_benchmark,
)
from hrm_adaptive_memory.executive.metareasoning_executor import (
    DeterministicMetareasoningExecutor, initial_i3_runtime,
)
from hrm_adaptive_memory.executive.metareasoning_transition_table import (
    build_oracle_policy_table,
)
from hrm_adaptive_memory.executive.metareasoning_utility import MetareasoningUtility
from hrm_adaptive_memory.executive.policy import load_frozen_policy
from hrm_adaptive_memory.executive.resources import ResourceState
from hrm_adaptive_memory.executive.selective_governor.features import (
    InterventionFeatures, extract_features, FEATURE_NAMES,
)
from hrm_adaptive_memory.executive.selective_governor.q_pib_predictor import (
    QPiBInterventionPredictor, ACTION_NAMES, EXTENDED_FEATURE_NAMES,
    features_action_vector,
)

ACTION_SET = {a for a in DecisionAction if a.value in ACTION_NAMES}


def build_training_data_from_off_trajectories(
    i352c_results_path: str,
    benchmark_manifest: str,
    split: str,
    utility_config: str,
    policy_config: str,
) -> list[dict[str, Any]]:
    """Build training data from OFF trajectories.

    For each step in each OFF trajectory:
      - Extract features x(s)
      - The action taken a = a_taken
      - The target y = realized utility from s onward under OFF continuation

    This gives us Q^{π_B}(s, a_taken) samples.
    """
    print(f"Loading benchmark from {benchmark_manifest}...")
    benchmark = load_metareasoning_benchmark(benchmark_manifest, verify_oracle_cache=False)
    split_bm = benchmark.for_split(split)
    task_map = {t.task_id: t for t in split_bm.tasks}

    print(f"Loading I3.5.2c results from {i352c_results_path}...")
    data = json.loads(Path(i352c_results_path).read_text())
    blocks = data["results"]
    print(f"Loaded {len(blocks)} task blocks")

    utility = MetareasoningUtility.from_file(ROOT / utility_config)
    policy = load_frozen_policy(policy_config)
    cond = get_condition(ConditionID.AWARE_GOVERNOR)

    oracle_executor = DeterministicMetareasoningExecutor()
    task_executor = DeterministicActionExecutor()
    governor = GeneralGovernor()

    training_records: list[dict[str, Any]] = []

    for i, block in enumerate(blocks):
        task_id = block["task_id"]
        if task_id not in task_map:
            continue
        task = task_map[task_id]
        budget = split_bm.budget_for(task)

        off_traj = block["trajectories"]["OFF"]
        off_steps = off_traj["steps"]
        off_utility = off_traj["realized_utility"]

        # Replay OFF trajectory to extract features at each step
        resources = ResourceState(budget)
        i3_runtime = initial_i3_runtime(task, resources)
        adapter = _I3TaskAdapter(task)
        t_runtime = init_task_runtime(adapter, ResourceState(budget))

        prior_decisions: list[DecisionSummary] = []
        prior_outcomes: list[str] = []

        # Compute cumulative utility from each step to terminal
        # realized_utility = sum of step costs + terminal reward
        # We need to compute the utility from step k onward
        # We'll replay and track the utility at each step

        step_utilities: list[float] = []
        cumulative_util = 0.0

        # First pass: replay and collect step costs
        replay_runtime = initial_i3_runtime(task, ResourceState(budget))
        replay_t = init_task_runtime(adapter, ResourceState(budget))
        replay_pd: list[DecisionSummary] = []
        replay_po: list[str] = []

        step_costs: list[float] = []
        terminal_reward = 0.0

        for step_idx, step_data in enumerate(off_steps):
            a = DecisionAction(step_data["executed_action"])
            resources_before = replay_runtime.resources
            exec_res = oracle_executor.execute(replay_runtime, a)
            resources_after = exec_res.runtime.resources
            step_cost = utility.action_cost(resources_before, resources_after)
            step_costs.append(-step_cost)  # utility is negative of cost

            if exec_res.terminal:
                terminal_reward = utility.terminal_reward(a, bool(exec_res.task_success))

            replay_runtime = exec_res.runtime
            replay_t = task_executor.execute(replay_t, a).runtime
            replay_pd.append(DecisionSummary(
                f"{task_id}:step:{step_idx}", a.value,
                step_data["reason_code"], exec_res.outcome_code))
            replay_po.append(exec_res.outcome_code)

            if exec_res.terminal:
                break

        # Compute utility-from-step-k: sum of step_costs[k:] + terminal_reward
        n_steps = len(step_costs)
        utility_from_step = []
        for k in range(n_steps):
            future_util = sum(step_costs[k:]) + terminal_reward
            utility_from_step.append(future_util)

        # Second pass: extract features at each step
        feat_runtime = initial_i3_runtime(task, ResourceState(budget))
        feat_t = init_task_runtime(adapter, ResourceState(budget))
        feat_pd: list[DecisionSummary] = []
        feat_po: list[str] = []

        for step_idx, step_data in enumerate(off_steps):
            a_str = step_data["executed_action"]
            a = DecisionAction(a_str)

            # Extract features at this state
            obs = build_observation(feat_t, task, cond, tuple(feat_pd), tuple(feat_po))
            p_actions = tuple(d.selected_action if isinstance(d.selected_action, str)
                              else d.selected_action.value for d in feat_pd)
            features = extract_features(
                obs, remaining_steps=24 - step_idx,
                prior_actions=p_actions, prior_outcomes=tuple(feat_po),
            )

            training_records.append({
                "task_id": task_id,
                "step_id": step_idx,
                "action": a_str,
                "features": features,
                "q_pib_target": round(utility_from_step[step_idx], 4),
                "source": "off_trajectory",
            })

            # Step forward
            exec_res = oracle_executor.execute(feat_runtime, a)
            feat_runtime = exec_res.runtime
            feat_t = task_executor.execute(feat_t, a).runtime
            feat_pd.append(DecisionSummary(
                f"{task_id}:step:{step_idx}", a.value,
                step_data["reason_code"], exec_res.outcome_code))
            feat_po.append(exec_res.outcome_code)

            if exec_res.terminal:
                break

        if (i + 1) % 50 == 0:
            print(f"  Processed [{i+1}/{len(blocks)}] tasks, {len(training_records)} samples")

    return training_records


def build_training_data_from_i352d(
    i352d_results_path: str,
    i352c_results_path: str,
    benchmark_manifest: str,
    split: str,
    utility_config: str,
    policy_config: str,
) -> list[dict[str, Any]]:
    """Build training data from I3.5.2d fork results.

    For each intervention state, we have:
      - Fork A: a_base + OFF continuation → u_base_continuation = Q^{π_B}(s, a_base)
      - Fork B: a_gov + OFF continuation → u_gov_off_continuation = Q^{π_B}(s, a_gov)

    These give us Q^{π_B} samples for the governor action.
    """
    print(f"Loading I3.5.2d results from {i352d_results_path}...")
    i352d_records = []
    with open(i352d_results_path) as f:
        for line in f:
            i352d_records.append(json.loads(line))
    print(f"Loaded {len(i352d_records)} I3.5.2d records")

    # We need to replay OFF trajectories to get features at the intervention states
    print(f"Loading benchmark from {benchmark_manifest}...")
    benchmark = load_metareasoning_benchmark(benchmark_manifest, verify_oracle_cache=False)
    split_bm = benchmark.for_split(split)
    task_map = {t.task_id: t for t in split_bm.tasks}

    print(f"Loading I3.5.2c results from {i352c_results_path}...")
    data = json.loads(Path(i352c_results_path).read_text())
    blocks = data["results"]

    utility = MetareasoningUtility.from_file(ROOT / utility_config)
    cond = get_condition(ConditionID.AWARE_GOVERNOR)

    oracle_executor = DeterministicMetareasoningExecutor()
    task_executor = DeterministicActionExecutor()

    # Build a map: (task_id, step_id) -> I3.5.2d record
    i352d_map = {}
    for rec in i352d_records:
        key = (rec["task_id"], rec["step_id"])
        i352d_map[key] = rec

    training_records: list[dict[str, Any]] = []

    for i, block in enumerate(blocks):
        task_id = block["task_id"]
        if task_id not in task_map:
            continue
        task = task_map[task_id]
        budget = split_bm.budget_for(task)

        off_traj = block["trajectories"]["OFF"]
        off_steps = off_traj["steps"]

        # Replay OFF trajectory
        resources = ResourceState(budget)
        i3_runtime = initial_i3_runtime(task, resources)
        adapter = _I3TaskAdapter(task)
        t_runtime = init_task_runtime(adapter, ResourceState(budget))

        prior_decisions: list[DecisionSummary] = []
        prior_outcomes: list[str] = []

        for step_idx, step_data in enumerate(off_steps):
            # Check if this step has an I3.5.2d record
            key = (task_id, step_idx)
            if key in i352d_map:
                rec = i352d_map[key]

                # Extract features
                obs = build_observation(t_runtime, task, cond,
                                        tuple(prior_decisions), tuple(prior_outcomes))
                p_actions = tuple(d.selected_action if isinstance(d.selected_action, str)
                                  else d.selected_action.value for d in prior_decisions)
                features = extract_features(
                    obs, remaining_steps=24 - step_idx,
                    prior_actions=p_actions, prior_outcomes=tuple(prior_outcomes),
                )

                # Fork A: Q^{π_B}(s, a_base) = u_base_continuation
                training_records.append({
                    "task_id": task_id,
                    "step_id": step_idx,
                    "action": rec["base_action"],
                    "features": features,
                    "q_pib_target": rec["u_base_continuation"],
                    "source": "i352d_fork_a",
                })

                # Fork B: Q^{π_B}(s, a_gov) = u_gov_off_continuation
                if not rec["same_action"]:
                    training_records.append({
                        "task_id": task_id,
                        "step_id": step_idx,
                        "action": rec["gov_action"],
                        "features": features,
                        "q_pib_target": rec["u_gov_off_continuation"],
                        "source": "i352d_fork_b",
                    })

            # Step forward
            a = DecisionAction(step_data["executed_action"])
            exec_res = oracle_executor.execute(i3_runtime, a)
            i3_runtime = exec_res.runtime
            t_runtime = task_executor.execute(t_runtime, a).runtime
            prior_decisions.append(DecisionSummary(
                f"{task_id}:step:{step_idx}", a.value,
                step_data["reason_code"], exec_res.outcome_code))
            prior_outcomes.append(exec_res.outcome_code)

            if exec_res.terminal:
                break

        if (i + 1) % 50 == 0:
            print(f"  Processed [{i+1}/{len(blocks)}] tasks, {len(training_records)} I3.5.2d samples")

    return training_records


def train_and_evaluate(
    training_records: list[dict[str, Any]],
    output_dir: Path,
    n_splits: int = 5,
) -> dict[str, Any]:
    """Train the Q^{π_B} regression model with fold-isolated CV."""

    # Convert to feature matrix and target vector
    X = []
    y = []
    task_ids = []
    sources = []
    actions = []

    for rec in training_records:
        vec = features_action_vector(rec["features"], rec["action"])
        X.append(vec)
        y.append(rec["q_pib_target"])
        task_ids.append(rec["task_id"])
        sources.append(rec["source"])
        actions.append(rec["action"])

    X = np.array(X)
    y = np.array(y)
    task_ids = np.array(task_ids)

    print(f"\nTraining data: {len(X)} samples, {X.shape[1]} features")
    print(f"  Target range: [{y.min():.2f}, {y.max():.2f}], mean={y.mean():.2f}")
    print(f"  Sources: {Counter(sources)}")
    print(f"  Actions: {Counter(actions)}")

    # Fold-isolated cross-validation by task
    unique_tasks = sorted(set(task_ids))
    print(f"  Unique tasks: {len(unique_tasks)}")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []
    all_predictions = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(unique_tasks)):
        train_tasks = set(unique_tasks[t] for t in train_idx)
        test_tasks = set(unique_tasks[t] for t in test_idx)

        train_mask = np.array([t in train_tasks for t in task_ids])
        test_mask = np.array([t in test_tasks for t in task_ids])

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        # Train GradientBoostingRegressor
        model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        )
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Compute intervention advantage predictions
        # For each test sample, predict Q^{π_B}(s, a) for all actions
        # and check if the best action advantage is positive
        advantage_correct = 0
        advantage_total = 0
        harm_rate = 0  # fraction where predicted best action leads to worse actual

        # Group test samples by (task_id, step_id) to compute advantage
        test_groups = defaultdict(list)
        for i, mask_i in enumerate(np.where(test_mask)[0]):
            rec = training_records[mask_i]
            key = (rec["task_id"], rec["step_id"])
            test_groups[key].append((rec["action"], y_pred[i], y_test[i]))

        for key, group in test_groups.items():
            if len(group) < 2:
                continue
            # Find predicted best action
            pred_best_idx = max(range(len(group)), key=lambda j: group[j][1])
            pred_best_action = group[pred_best_idx][0]
            pred_best_q = group[pred_best_idx][1]

            # Find actual best action
            actual_best_idx = max(range(len(group)), key=lambda j: group[j][2])
            actual_best_q = group[actual_best_idx][2]

            # Did predicted best action match actual best?
            advantage_total += 1
            if group[pred_best_idx][2] >= group[actual_best_idx][2] - 5.0:
                advantage_correct += 1

            # Check if predicted best action is actually harmful
            # (actual Q of predicted best < actual Q of predicted worst)
            actual_qs = [g[2] for g in group]
            if group[pred_best_idx][2] < max(actual_qs) - 10.0:
                harm_rate += 1

        advantage_accuracy = advantage_correct / advantage_total if advantage_total > 0 else 0
        harm_rate_frac = harm_rate / advantage_total if advantage_total > 0 else 0

        fold_results.append({
            "fold": fold_idx,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "mse": round(mse, 4),
            "r2": round(r2, 4),
            "advantage_accuracy": round(advantage_accuracy, 4),
            "harm_rate": round(harm_rate_frac, 4),
        })
        print(f"  Fold {fold_idx}: n_train={len(X_train)}, n_test={len(X_test)}, "
              f"mse={mse:.2f}, r2={r2:.4f}, "
              f"advantage_acc={advantage_accuracy:.2%}, harm_rate={harm_rate_frac:.2%}")

    # Train final model on all data
    print("\nTraining final model on all data...")
    final_model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
    )
    final_model.fit(X, y)

    # Feature importance
    feature_importance = sorted(
        zip(EXTENDED_FEATURE_NAMES, final_model.feature_importances_),
        key=lambda x: -x[1]
    )
    print("\nTop 10 features:")
    for name, imp in feature_importance[:10]:
        print(f"  {name}: {imp:.4f}")

    # Save model
    predictor = QPiBInterventionPredictor(
        model=final_model,
        delta_q_threshold=5.0,
        max_harm_probability=0.15,
        min_confidence=0.60,
    )
    model_path = output_dir / "q_pib_gate_v1.pkl"
    predictor.save(model_path)
    print(f"\nSaved model: {model_path}")

    # Summary
    mean_mse = sum(f["mse"] for f in fold_results) / len(fold_results)
    mean_r2 = sum(f["r2"] for f in fold_results) / len(fold_results)
    mean_adv_acc = sum(f["advantage_accuracy"] for f in fold_results) / len(fold_results)
    mean_harm = sum(f["harm_rate"] for f in fold_results) / len(fold_results)

    summary = {
        "schema": "DAPH_V2B_I3_5_3_QPIB_GATE_TRAINING_V1",
        "n_samples": len(X),
        "n_features": X.shape[1],
        "n_tasks": len(unique_tasks),
        "sources": dict(Counter(sources)),
        "actions": dict(Counter(actions)),
        "target_stats": {
            "min": round(float(y.min()), 4),
            "max": round(float(y.max()), 4),
            "mean": round(float(y.mean()), 4),
            "std": round(float(y.std()), 4),
        },
        "fold_results": fold_results,
        "mean_mse": round(mean_mse, 4),
        "mean_r2": round(mean_r2, 4),
        "mean_advantage_accuracy": round(mean_adv_acc, 4),
        "mean_harm_rate": round(mean_harm, 4),
        "top_features": [{"name": n, "importance": round(float(i), 6)}
                         for n, i in feature_importance[:15]],
        "model_path": str(model_path),
    }

    summary_path = output_dir / "q_pib_gate_training_summary_v1.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"Saved summary: {summary_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Train Q^{π_B} intervention gate")
    parser.add_argument("--split", default="structure_dev_v2")
    parser.add_argument(
        "--i352c-results",
        default="experiments/v2b_i3_5_2/development/i352c_55f93130e87c/results.json",
    )
    parser.add_argument(
        "--i352d-results",
        default="experiments/v2b_i3_5_2/development/i352d/intervention_values_v1.jsonl",
    )
    parser.add_argument(
        "--benchmark-manifest",
        default="experiments/v2b_i3_5/manifests/v2b_i3_5_benchmark_manifest_v2.json",
    )
    parser.add_argument("--utility", default="configs/v2b_i3_1_utility_v1.json")
    parser.add_argument("--policy", default="configs/v2b_i3_policy_v1.json")
    parser.add_argument("--output-dir", default="experiments/v2b_i3_5_2/development/i352d")
    parser.add_argument("--n-splits", type=int, default=5)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build training data from OFF trajectories
    print("\n" + "=" * 78)
    print("BUILDING TRAINING DATA FROM OFF TRAJECTORIES")
    print("=" * 78)
    off_records = build_training_data_from_off_trajectories(
        i352c_results_path=args.i352c_results,
        benchmark_manifest=args.benchmark_manifest,
        split=args.split,
        utility_config=args.utility,
        policy_config=args.policy,
    )
    print(f"\nTotal OFF trajectory samples: {len(off_records)}")

    # Build training data from I3.5.2d forks
    print("\n" + "=" * 78)
    print("BUILDING TRAINING DATA FROM I3.5.2d FORKS")
    print("=" * 78)
    i352d_records = build_training_data_from_i352d(
        i352d_results_path=args.i352d_results,
        i352c_results_path=args.i352c_results,
        benchmark_manifest=args.benchmark_manifest,
        split=args.split,
        utility_config=args.utility,
        policy_config=args.policy,
    )
    print(f"\nTotal I3.5.2d fork samples: {len(i352d_records)}")

    # Combine
    all_records = off_records + i352d_records
    print(f"\nTotal training samples: {len(all_records)}")

    # Train and evaluate
    print("\n" + "=" * 78)
    print("TRAINING Q^{π_B} GATE")
    print("=" * 78)
    summary = train_and_evaluate(all_records, output_dir, n_splits=args.n_splits)

    print("\n" + "=" * 78)
    print("TRAINING COMPLETE")
    print("=" * 78)
    print(f"  Samples: {summary['n_samples']}")
    print(f"  Mean R²: {summary['mean_r2']:.4f}")
    print(f"  Mean advantage accuracy: {summary['mean_advantage_accuracy']:.2%}")
    print(f"  Mean harm rate: {summary['mean_harm_rate']:.2%}")
    print(f"  Model: {summary['model_path']}")


if __name__ == "__main__":
    main()
