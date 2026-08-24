#!/usr/bin/env python3
"""I3.5 Phase 14: Formal causal vs observational learning comparison.

Compares Q_CAUSAL (trained on all forced-action data) vs Q_OBS (trained on
observational subset where only "naturally selected" actions appear).

The key scientific question:
  Does causal data (do(a)) produce better Q estimates than observational data?

This is the core distinction:
  Q_CAUSAL: Q(s,a) = E[U | do(a), s]   (forced interventions)
  Q_OBS:    Q(s,a) = E[U | observed(a), s]  (policy behavior)

With observational data, the policy selects actions based on state, creating
confounding: the policy picks ANSWER when it sees supporting evidence, so
Q_OBS learns "ANSWER is good when supporting evidence exists" — but this
conflates the policy's selection bias with the action's causal effect.

Causal data breaks this confounding by forcing all actions from each state.

Output:
  experiments/i3_5/models/causal_vs_observational_v1.json
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.train_i3_5_model_ladder import (
    B0GlobalPrior, B1PhaseConditioned, StateActionModel,
    extract_features, load_causal_data, paired_bootstrap_ci,
    ACTION_NAMES, FEATURE_KEYS,
)


def main():
    output_dir = REPO_ROOT / "experiments/i3_5/models"

    # Load causal data
    records = load_causal_data(REPO_ROOT / "experiments/i3_5/causal/causal_actions_v1.jsonl")
    print(f"Loaded {len(records)} causal records")

    # Build observational subset (same logic as training script)
    obs_records = []
    for r in records:
        sf = r["state_features"]
        if sf.get("n_supporting", 0) > 0 and sf.get("n_contradicting", 0) == 0:
            obs_action = "ANSWER"
        elif sf.get("n_eliminated", 0) > 0 and sf.get("n_live", 0) == 0:
            obs_action = "DEFER"
        elif sf.get("can_retrieve", False):
            obs_action = "RETRIEVE"
        elif sf.get("can_verify", False):
            obs_action = "VERIFY"
        elif sf.get("can_search", False):
            obs_action = "SEARCH_MORE"
        else:
            obs_action = "DEFER"
        if r["forced_action"] == obs_action:
            obs_records.append(r)

    print(f"Observational subset: {len(obs_records)} records (of {len(records)} causal)")

    # Analyze observational coverage
    obs_by_action = Counter(r["forced_action"] for r in obs_records)
    causal_by_action = Counter(r["forced_action"] for r in records)
    print(f"\nAction coverage:")
    print(f"  {'Action':15s} {'Causal':>8s} {'Observational':>14s} {'Coverage':>10s}")
    for action in ACTION_NAMES:
        c = causal_by_action.get(action, 0)
        o = obs_by_action.get(action, 0)
        cov = f"{o}/{c}" if c > 0 else "N/A"
        print(f"  {action:15s} {c:8d} {o:14d} {cov:>10s}")

    # Analyze observational coverage by subtype
    obs_by_subtype = Counter(r["category"] for r in obs_records)
    causal_by_subtype = Counter(r["category"] for r in records)
    print(f"\nSubtype coverage:")
    print(f"  {'Subtype':15s} {'Causal':>8s} {'Observational':>14s}")
    for subtype in sorted(causal_by_subtype.keys()):
        c = causal_by_subtype.get(subtype, 0)
        o = obs_by_subtype.get(subtype, 0)
        print(f"  {subtype:15s} {c:8d} {o:14d}")

    # Train both models
    X_causal = np.array([extract_features(r) for r in records])
    y_causal = np.array([r["terminal_utility"] for r in records])

    X_obs = np.array([extract_features(r) for r in obs_records])
    y_obs = np.array([r["terminal_utility"] for r in obs_records])

    gbt_params = {
        "n_estimators": 200, "max_depth": 4, "learning_rate": 0.1,
        "subsample": 0.8, "random_state": 42,
    }

    q_causal = StateActionModel(GradientBoostingRegressor(**gbt_params), "Q_CAUSAL")
    q_causal.fit(X_causal, y_causal)

    q_obs = StateActionModel(GradientBoostingRegressor(**gbt_params), "Q_OBS")
    q_obs.fit(X_obs, y_obs)

    b0 = B0GlobalPrior()
    b0.fit(records)

    # Group by task
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_task[r["task_id"]].append(r)

    q_star: dict[str, dict[str, float]] = {}
    for task_id, task_records in by_task.items():
        q_star[task_id] = {}
        for r in task_records:
            q_star[task_id][r["forced_action"]] = r["terminal_utility"]

    # ---- Comparison 1: Per-task selected action utility ----
    print("\n=== Per-Task Selected Action Utility ===")
    comparisons = []
    for model_name, model in [("B0", b0), ("Q_CAUSAL", q_causal), ("Q_OBS", q_obs)]:
        utilities = []
        for task_id, task_records in by_task.items():
            sf = task_records[0]["state_features"]
            preds = model.predict_all(sf)
            selected = max(preds, key=preds.get) if preds else "UNKNOWN"
            u = q_star[task_id].get(selected, 0.0)
            utilities.append(u)
        mean_u = np.mean(utilities)
        print(f"  {model_name:15s}: mean selected utility = {mean_u:+.4f}")

    # Paired bootstrap: Q_CAUSAL vs Q_OBS
    causal_minus_obs = []
    causal_minus_b0 = []
    obs_minus_b0 = []
    for task_id, task_records in by_task.items():
        sf = task_records[0]["state_features"]

        qc_preds = q_causal.predict_all(sf)
        qc_action = max(qc_preds, key=qc_preds.get) if qc_preds else "UNKNOWN"

        qo_preds = q_obs.predict_all(sf)
        qo_action = max(qo_preds, key=qo_preds.get) if qo_preds else "UNKNOWN"

        b0_preds = b0.predict_all(sf)
        b0_action = max(b0_preds, key=b0_preds.get) if b0_preds else "UNKNOWN"

        qc_u = q_star[task_id].get(qc_action, 0.0)
        qo_u = q_star[task_id].get(qo_action, 0.0)
        b0_u = q_star[task_id].get(b0_action, 0.0)

        causal_minus_obs.append(qc_u - qo_u)
        causal_minus_b0.append(qc_u - b0_u)
        obs_minus_b0.append(qo_u - b0_u)

    print("\n=== Paired Bootstrap Comparisons ===")
    for name, diffs in [("Q_CAUSAL - Q_OBS", causal_minus_obs),
                         ("Q_CAUSAL - B0", causal_minus_b0),
                         ("Q_OBS - B0", obs_minus_b0)]:
        mean_d, ci_lo, ci_hi = paired_bootstrap_ci(diffs)
        sig = ci_lo > 0 or ci_hi < 0
        print(f"  {name:25s}: mean={mean_d:+.4f}, CI=[{ci_lo:+.4f}, {ci_hi:+.4f}], significant={sig}")
        comparisons.append({
            "comparison": name,
            "mean_diff": round(mean_d, 4),
            "ci_lower": round(ci_lo, 4),
            "ci_upper": round(ci_hi, 4),
            "significant": sig,
            "n_tasks": len(by_task),
        })

    # ---- Comparison 2: Confounding analysis ----
    print("\n=== Confounding Analysis ===")
    # In observational data, the policy selects ANSWER when n_supporting > 0.
    # This means Q_OBS only sees ANSWER in states with supporting evidence.
    # Q_OBS never sees ANSWER in states WITHOUT supporting evidence.
    # So Q_OBS cannot learn Q(s, ANSWER) for states without support.

    # Check: what fraction of (s, ANSWER) pairs have n_supporting > 0 in observational data?
    obs_answer_records = [r for r in obs_records if r["forced_action"] == "ANSWER"]
    causal_answer_records = [r for r in records if r["forced_action"] == "ANSWER"]

    obs_answer_with_support = sum(
        1 for r in obs_answer_records
        if r["state_features"].get("n_supporting", 0) > 0
    )
    causal_answer_with_support = sum(
        1 for r in causal_answer_records
        if r["state_features"].get("n_supporting", 0) > 0
    )

    confounding = {
        "observational_answer_n": len(obs_answer_records),
        "observational_answer_with_support": obs_answer_with_support,
        "observational_answer_coverage_fraction": round(
            obs_answer_with_support / max(len(obs_answer_records), 1), 4
        ),
        "causal_answer_n": len(causal_answer_records),
        "causal_answer_with_support": causal_answer_with_support,
        "causal_answer_coverage_fraction": round(
            causal_answer_with_support / max(len(causal_answer_records), 1), 4
        ),
        "confounding_present": obs_answer_with_support == len(obs_answer_records),
    }
    print(f"  Observational ANSWER records: {len(obs_answer_records)}")
    print(f"    With n_supporting > 0: {obs_answer_with_support} ({confounding['observational_answer_coverage_fraction']:.2%})")
    print(f"  Causal ANSWER records: {len(causal_answer_records)}")
    print(f"    With n_supporting > 0: {causal_answer_with_support} ({confounding['causal_answer_coverage_fraction']:.2%})")
    print(f"  Confounding present: {confounding['confounding_present']}")

    # ---- Comparison 3: Q estimate accuracy vs ground truth Q* ----
    print("\n=== Q Estimate Accuracy (MSE vs Q*) ===")
    # For each (s,a) pair, compare predicted Q to actual Q*
    causal_preds = []
    obs_preds = []
    b0_preds_list = []
    q_star_vals = []

    for r in records:
        sf = r["state_features"]
        action = r["forced_action"]
        q_true = r["terminal_utility"]

        qc = q_causal.predict(sf, action)
        qo = q_obs.predict(sf, action)
        b0q = b0.predict(sf, action)

        causal_preds.append(qc)
        obs_preds.append(qo)
        b0_preds_list.append(b0q)
        q_star_vals.append(q_true)

    causal_mse = np.mean((np.array(causal_preds) - np.array(q_star_vals))**2)
    obs_mse = np.mean((np.array(obs_preds) - np.array(q_star_vals))**2)
    b0_mse = np.mean((np.array(b0_preds_list) - np.array(q_star_vals))**2)

    q_accuracy = {
        "B0_mse": round(float(b0_mse), 6),
        "Q_CAUSAL_mse": round(float(causal_mse), 6),
        "Q_OBS_mse": round(float(obs_mse), 6),
        "Q_CAUSAL_better_than_Q_OBS": bool(causal_mse < obs_mse),
        "Q_CAUSAL_better_than_B0": bool(causal_mse < b0_mse),
    }
    print(f"  B0 MSE:       {b0_mse:.6f}")
    print(f"  Q_CAUSAL MSE: {causal_mse:.6f}")
    print(f"  Q_OBS MSE:    {obs_mse:.6f}")
    print(f"  Q_CAUSAL < Q_OBS: {causal_mse < obs_mse}")
    print(f"  Q_CAUSAL < B0: {causal_mse < b0_mse}")

    # ---- Save ----
    result = {
        "n_causal_records": len(records),
        "n_observational_records": len(obs_records),
        "observational_coverage": {
            "by_action": dict(obs_by_action),
            "by_subtype": dict(obs_by_subtype),
        },
        "causal_coverage": {
            "by_action": dict(causal_by_action),
            "by_subtype": dict(causal_by_subtype),
        },
        "comparisons": comparisons,
        "confounding_analysis": confounding,
        "q_estimate_accuracy": q_accuracy,
        "gbt_params": gbt_params,
    }

    result_content = json.dumps(result, sort_keys=True)
    result_sha = hashlib.sha256(result_content.encode()).hexdigest()

    result_path = output_dir / "causal_vs_observational_v1.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print(f"\nWritten to {result_path}")
    print(f"SHA256: {result_sha[:16]}...")


if __name__ == "__main__":
    main()
