#!/usr/bin/env python3
"""I3.27 Track 1: Q_CAUSAL_V1 error audit on chain tasks.

For every chain task at step 0, compare predicted Q against actual
forced-action causal returns for every legal action.

Produces a table:
  Action | Predicted Q | Actual causal return | Error

Then diagnoses why RETRIEVE is catastrophically underestimated:
- Feature support for chain states in training data
- Phase distribution
- Retrieval-state feature distribution
- OOD check: are chain states out-of-distribution relative to the
  1,056 causal records?
"""
from __future__ import annotations

import json
import pickle
import sys
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from hrm_adaptive_memory.cognitive_control.core import DecisionAction
from hrm_adaptive_memory.cognitive_control.state import (
    VerificationState, TemporalStatus,
)
from hrm_adaptive_memory.executive.evidence_benchmark.schema import (
    EvidenceTask, initial_evidence_runtime,
)
from hrm_adaptive_memory.executive.evidence_benchmark.executor import (
    EvidenceExecutor, valid_verify_targets,
)
from hrm_adaptive_memory.executive.evidence_benchmark.i3_26_development_generator import (
    generate_development_benchmark,
)
from hrm_adaptive_memory.executive.evidence_benchmark.i3_5_confirmation_generator import (
    CONFIRMATION_BUDGET_PROFILES,
)
from hrm_adaptive_memory.executive.resources import ResourceBudget, ResourceState
from hrm_adaptive_memory.executive.metareasoning_utility import MetareasoningUtility

from daph.intervention.checkpoint import (
    create_checkpoint, compute_state_features, compute_legal_actions,
)
from daph.intervention.force_action import force_action_with_rollout


def extract_features(state_features: dict, action: str) -> dict:
    feats = {
        "n_live": state_features.get("n_live", 0),
        "n_eliminated": state_features.get("n_eliminated", 0),
        "n_untested": state_features.get("n_untested", 0),
        "n_total_hypotheses": state_features.get("n_total_hypotheses", 0),
        "n_visible_evidence": state_features.get("n_visible_evidence", 0),
        "n_verified": state_features.get("n_verified", 0),
        "n_supporting": state_features.get("n_supporting", 0),
        "n_contradicting": state_features.get("n_contradicting", 0),
        "n_stale": state_features.get("n_stale", 0),
        "retrieval_remaining": state_features.get("retrieval_remaining", 0),
        "search_remaining": state_features.get("search_remaining", 0),
        "verify_remaining": state_features.get("verify_remaining", 0),
        "steps_remaining": state_features.get("steps_remaining", 0),
        "can_retrieve": int(state_features.get("can_retrieve", False)),
        "can_search": int(state_features.get("can_search", False)),
        "can_verify": int(state_features.get("can_verify", False)),
        "searched": int(state_features.get("searched", False)),
        "reasoning_complete": int(state_features.get("reasoning_complete", False)),
        "same_action_run_length": state_features.get("same_action_run_length", 0),
        "retrieval_count": state_features.get("retrieval_count", 0),
        "search_count": state_features.get("search_count", 0),
        "verify_count": state_features.get("verify_count", 0),
    }
    for a in ["ANSWER", "DEFER", "RETRIEVE", "VERIFY", "SEARCH_MORE", "REASON_MORE"]:
        feats[f"a_{a}"] = int(action == a)
    feats["n_live_x_retrieve"] = feats["n_live"] * feats["a_RETRIEVE"]
    feats["n_live_x_verify"] = feats["n_live"] * feats["a_VERIFY"]
    feats["n_live_x_search"] = feats["n_live"] * feats["a_SEARCH_MORE"]
    feats["n_untested_x_retrieve"] = feats["n_untested"] * feats["a_RETRIEVE"]
    feats["n_untested_x_verify"] = feats["n_untested"] * feats["a_VERIFY"]
    feats["n_supporting_x_answer"] = feats["n_supporting"] * feats["a_ANSWER"]
    feats["n_eliminated_x_defer"] = feats["n_eliminated"] * feats["a_DEFER"]
    return feats


def load_q_model():
    """Load frozen QCAUSAL_V1 model."""
    est_dir = REPO_ROOT / "experiments/i3_5/pinned_policy/frozen_estimators"
    with open(est_dir / "QCAUSAL_gbt.pkl", "rb") as f:
        model = pickle.load(f)
    with open(est_dir / "feature_schema.json") as f:
        schema = json.load(f)
    return model, schema["feature_keys"]


def load_training_data():
    """Load the 1,056 causal training records."""
    path = REPO_ROOT / "experiments/i3_5/pinned_policy/pinned_causal_actions_v1.jsonl"
    records = [json.loads(line) for line in open(path)]
    return records


def get_budget_for_profile(profile: str) -> ResourceBudget:
    params = CONFIRMATION_BUDGET_PROFILES[profile]
    return ResourceBudget(
        max_executive_steps=params["max_executive_steps"],
        max_retrieval_calls=params["max_retrieval_calls"],
        max_verification_calls=params["max_verification_calls"],
        max_search_calls=params["max_search_calls"],
        max_reasoning_tokens=params.get("max_reasoning_tokens", 256),
        max_elapsed_ms=params.get("max_elapsed_ms", 10_000),
    )


def compute_actual_causal_return(
    task: EvidenceTask,
    runtime,
    action_str: str,
    utility: MetareasoningUtility,
    executor: EvidenceExecutor,
    max_steps: int = 8,
) -> dict:
    """Compute actual causal return for forcing an action from a state.

    Uses force_action_with_rollout with a deterministic no-op policy
    (just executes the forced action and measures immediate outcome).
    For non-terminal actions, continues with a simple greedy policy
    that picks the highest-progress action.
    """
    action = DecisionAction(action_str)

    # Determine verify target
    target_eid = None
    if action is DecisionAction.VERIFY:
        valid = valid_verify_targets(runtime)
        if valid:
            target_eid = valid[0]
        else:
            return {"causal_return": -20.0, "success": False, "terminal": False,
                    "outcome": "NO_VERIFY_TARGET"}

    # Execute the forced action
    try:
        exec_result = executor.execute(runtime, action, target_evidence_id=target_eid)
    except Exception as e:
        return {"causal_return": -20.0, "success": False, "terminal": False,
                "outcome": f"EXEC_ERROR: {e}"}

    if exec_result.terminal:
        tr = utility.terminal_reward(exec_result.action, bool(exec_result.task_success))
        # Subtract action cost
        cost = utility.action_cost(runtime.resources, exec_result.runtime.resources)
        causal_return = tr - cost
        return {
            "causal_return": round(causal_return, 2),
            "success": bool(exec_result.task_success),
            "terminal": True,
            "outcome": exec_result.outcome_code,
        }

    # For non-terminal: continue with greedy progress policy
    post_runtime = exec_result.runtime
    total_cost = utility.action_cost(runtime.resources, post_runtime.resources)
    steps = 1

    while steps < max_steps:
        legal = compute_legal_actions(post_runtime)
        if not legal:
            break

        # Greedy: pick the action with highest progress
        best_action = None
        best_progress = -999.0
        best_target = None

        for a_str in legal:
            a = DecisionAction(a_str)
            tgt = None
            if a is DecisionAction.VERIFY:
                valid = valid_verify_targets(post_runtime)
                if valid:
                    tgt = valid[0]
                else:
                    continue
            try:
                from daph.progress.progress_rule_v1 import compute_progress
                res = executor.execute(post_runtime, a, target_evidence_id=tgt)
                prog = compute_progress(post_runtime, res, utility)
                if prog.progress > best_progress:
                    best_progress = prog.progress
                    best_action = a
                    best_target = tgt
            except Exception:
                continue

        if best_action is None:
            break

        try:
            res = executor.execute(post_runtime, best_action, target_evidence_id=best_target)
        except Exception:
            break

        total_cost += utility.action_cost(post_runtime.resources, res.runtime.resources)
        post_runtime = res.runtime
        steps += 1

        if res.terminal:
            tr = utility.terminal_reward(res.action, bool(res.task_success))
            causal_return = tr - total_cost
            return {
                "causal_return": round(causal_return, 2),
                "success": bool(res.task_success),
                "terminal": True,
                "outcome": res.outcome_code,
                "downstream_steps": steps,
            }

    # Step limit
    return {
        "causal_return": round(-0.5 - total_cost, 2),
        "success": False,
        "terminal": False,
        "outcome": "STEP_LIMIT",
        "downstream_steps": steps,
    }


def main():
    print("=" * 80)
    print("I3.27 TRACK 1: Q_CAUSAL_V1 ERROR AUDIT ON CHAIN TASKS")
    print("=" * 80)

    # Load Q model
    print("\nLoading frozen QCAUSAL_V1...")
    q_model, feature_keys = load_q_model()
    print(f"  Features: {len(feature_keys)}")
    print(f"  Training records: 1056")

    # Load utility
    utility = MetareasoningUtility.from_file(
        REPO_ROOT / "configs" / "v2b_i3_1_utility_v1.json")

    # Load training data for OOD analysis
    print("\nLoading training data for OOD analysis...")
    training_records = load_training_data()
    print(f"  {len(training_records)} records")

    # Load development benchmark chain tasks
    print("\nLoading chain tasks from development benchmark...")
    tasks = generate_development_benchmark(seed=7719)
    chain_tasks = [t for t in tasks if t.category == "chain"]
    print(f"  {len(chain_tasks)} chain tasks")

    executor = EvidenceExecutor()

    # === Q ERROR TABLE ===
    print(f"\n{'='*80}")
    print("Q ERROR TABLE: Predicted Q vs Actual Causal Return")
    print(f"{'='*80}")

    error_records = []
    per_action_errors = defaultdict(list)

    for task in chain_tasks:
        budget = get_budget_for_profile(task.budget_profile)
        runtime = initial_evidence_runtime(task, ResourceState(budget=budget))
        sf = compute_state_features(runtime, ())
        legal = list(compute_legal_actions(runtime))

        # Predict Q for each legal action
        X = np.array([[extract_features(sf, a)[k] for k in feature_keys]
                      for a in legal])
        q_preds = q_model.predict(X)
        q_dict = {a: float(q) for a, q in zip(legal, q_preds)}

        print(f"\n  {task.task_id}:")
        print(f"    {'Action':<15} {'Predicted Q':>12} {'Actual return':>14} {'Error':>10} {'Success':>8}")
        print(f"    {'-'*65}")

        for action_str in legal:
            # Compute actual causal return
            actual = compute_actual_causal_return(
                task, runtime, action_str, utility, executor,
                max_steps=budget.max_executive_steps,
            )

            predicted_q = q_dict[action_str]
            actual_return = actual["causal_return"]
            error = predicted_q - actual_return

            per_action_errors[action_str].append({
                "predicted": predicted_q,
                "actual": actual_return,
                "error": error,
                "success": actual["success"],
            })

            print(f"    {action_str:<15} {predicted_q:>12.2f} {actual_return:>14.2f} {error:>10.2f} {str(actual['success']):>8}")

            error_records.append({
                "task_id": task.task_id,
                "action": action_str,
                "predicted_q": round(predicted_q, 2),
                "actual_causal_return": actual_return,
                "error": round(error, 2),
                "success": actual["success"],
                "terminal": actual["terminal"],
                "outcome": actual["outcome"],
            })

    # === ERROR SUMMARY ===
    print(f"\n{'='*80}")
    print("ERROR SUMMARY BY ACTION")
    print(f"{'='*80}")
    print(f"  {'Action':<15} {'N':>5} {'Mean pred Q':>12} {'Mean actual':>12} {'Mean error':>12} {'MAE':>10}")
    print(f"  {'-'*70}")

    for action_str in sorted(per_action_errors.keys()):
        errors = per_action_errors[action_str]
        n = len(errors)
        mean_pred = np.mean([e["predicted"] for e in errors])
        mean_actual = np.mean([e["actual"] for e in errors])
        mean_error = np.mean([e["error"] for e in errors])
        mae = np.mean([abs(e["error"]) for e in errors])
        print(f"  {action_str:<15} {n:>5} {mean_pred:>12.2f} {mean_actual:>12.2f} {mean_error:>12.2f} {mae:>10.2f}")

    # === RETRIEVE DEEP DIVE ===
    print(f"\n{'='*80}")
    print("RETRIEVE DEEP DIVE: Why is it underestimated?")
    print(f"{'='*80}")

    retrieve_errors = per_action_errors.get("RETRIEVE", [])
    if retrieve_errors:
        print(f"\n  RETRIEVE predictions: {[e['predicted'] for e in retrieve_errors[:5]]}")
        print(f"  RETRIEVE actual:     {[e['actual'] for e in retrieve_errors[:5]]}")
        print(f"  RETRIEVE errors:     {[e['error'] for e in retrieve_errors[:5]]}")
        print(f"  RETRIEVE successes:  {sum(1 for e in retrieve_errors if e['success'])}/{len(retrieve_errors)}")

    # === OOD ANALYSIS ===
    print(f"\n{'='*80}")
    print("OOD ANALYSIS: Are chain states out-of-distribution?")
    print(f"{'='*80}")

    # Compare chain state features to training data features
    # Key features for chain states at step 0:
    chain_features = []
    for task in chain_tasks:
        budget = get_budget_for_profile(task.budget_profile)
        runtime = initial_evidence_runtime(task, ResourceState(budget=budget))
        sf = compute_state_features(runtime, ())
        chain_features.append(sf)

    # Training data state features
    train_sf = [r["state_features"] for r in training_records]

    print(f"\n  Chain state feature distribution (step 0):")
    key_features = ["n_live", "n_eliminated", "n_untested", "n_visible_evidence",
                    "n_verified", "n_supporting", "n_contradicting", "n_hidden_evidence",
                    "retrieval_remaining", "verify_remaining", "search_remaining",
                    "steps_remaining", "can_retrieve", "can_verify", "can_search"]

    print(f"  {'Feature':<25} {'Chain mean':>12} {'Train mean':>12} {'Chain range':>15} {'Train range':>15}")
    print(f"  {'-'*85}")

    for feat in key_features:
        chain_vals = [sf.get(feat, 0) for sf in chain_features]
        train_vals = [sf.get(feat, 0) for sf in train_sf]
        chain_mean = np.mean(chain_vals) if chain_vals else 0
        train_mean = np.mean(train_vals) if train_vals else 0
        chain_range = f"[{min(chain_vals)},{max(chain_vals)}]" if chain_vals else "N/A"
        train_range = f"[{min(train_vals)},{max(train_vals)}]" if train_vals else "N/A"
        print(f"  {feat:<25} {chain_mean:>12.2f} {train_mean:>12.2f} {chain_range:>15} {train_range:>15}")

    # Check: how many training records have RETRIEVE as forced action in chain-like states?
    print(f"\n  Training data: RETRIEVE records in chain-like states:")
    retrieve_train = [r for r in training_records if r["forced_action"] == "RETRIEVE"]
    print(f"    Total RETRIEVE records: {len(retrieve_train)}/{len(training_records)}")

    # Chain-like: n_hidden_evidence > 0, can_retrieve=True, n_untested > 0
    chain_like_retrieve = [
        r for r in retrieve_train
        if r["state_features"].get("n_hidden_evidence", 0) > 0
        and r["state_features"].get("can_retrieve", False)
        and r["state_features"].get("n_untested", 0) > 0
    ]
    print(f"    Chain-like RETRIEVE records (hidden>0, can_retrieve, untested>0): {len(chain_like_retrieve)}")

    # What categories are in the training data?
    train_cats = Counter(r.get("category", "unknown") for r in training_records)
    print(f"\n  Training data categories:")
    for cat, count in sorted(train_cats.items(), key=lambda x: -x[1]):
        print(f"    {cat}: {count}")

    # Check if any training records match chain state profile
    chain_profile = {
        "n_live": 1, "n_eliminated": 1, "n_untested": 0,
        "n_visible_evidence": 2, "n_hidden_evidence": 3,
        "can_retrieve": True, "can_verify": True, "can_search": True,
    }
    matching = []
    for r in training_records:
        sf = r["state_features"]
        match = all(sf.get(k, 0) == v for k, v in chain_profile.items())
        if match:
            matching.append(r)
    print(f"\n  Training records matching chain step-0 profile: {len(matching)}")
    if matching:
        print(f"    Categories: {Counter(r.get('category', 'unknown') for r in matching)}")
        print(f"    Forced actions: {Counter(r['forced_action'] for r in matching)}")
        print(f"    Mean utility: {np.mean([r['pinned_policy_utility'] for r in matching]):.2f}")

    # === SAVE ===
    output = {
        "n_chain_tasks": len(chain_tasks),
        "per_action_summary": {
            a: {
                "n": len(errs),
                "mean_predicted_q": round(float(np.mean([e["predicted"] for e in errs])), 2),
                "mean_actual_return": round(float(np.mean([e["actual"] for e in errs])), 2),
                "mean_error": round(float(np.mean([e["error"] for e in errs])), 2),
                "mae": round(float(np.mean([abs(e["error"]) for e in errs])), 2),
            }
            for a, errs in per_action_errors.items()
        },
        "error_records": error_records,
        "retrieve_deep_dive": {
            "predictions": [e["predicted"] for e in retrieve_errors],
            "actuals": [e["actual"] for e in retrieve_errors],
            "errors": [e["error"] for e in retrieve_errors],
            "successes": sum(1 for e in retrieve_errors if e["success"]),
            "total": len(retrieve_errors),
        },
        "ood_analysis": {
            "training_records_total": len(training_records),
            "training_retrieve_records": len(retrieve_train),
            "chain_like_retrieve_records": len(chain_like_retrieve),
            "matching_chain_profile": len(matching),
            "training_categories": dict(train_cats),
        },
    }

    output_path = REPO_ROOT / "experiments/i3_27/q_error_audit.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, sort_keys=True)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
