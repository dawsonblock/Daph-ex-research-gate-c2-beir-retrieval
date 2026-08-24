#!/usr/bin/env python3
"""I3.5 offline validation: per-subtype breakdown, regret, mechanism audit.

Evaluates each model in the ladder on:
  1. Per-subtype top-action accuracy
  2. Per-subtype action regret
  3. Best-action agreement (does the model pick the correct first action?)
  4. Per-subtype rescues and breaks vs B0
  5. Mechanism audit: does Q_CAUSAL increase the correct action per subtype?

Output:
  experiments/i3_5/models/offline_validation_v1.json
  experiments/i3_5/models/offline_validation_v1_manifest.json
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
from sklearn.linear_model import Ridge
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
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load causal data
    causal_path = REPO_ROOT / "experiments/i3_5/causal/causal_actions_v1.jsonl"
    records = load_causal_data(causal_path)
    print(f"Loaded {len(records)} causal records")

    # Load model ladder
    ladder_path = output_dir / "model_ladder_v1.json"
    with open(ladder_path) as f:
        ladder = json.load(f)

    # Retrain models (need the actual model objects for prediction)
    X = np.array([extract_features(r) for r in records])
    y = np.array([r["terminal_utility"] for r in records])

    b0 = B0GlobalPrior()
    b0.fit(records)

    b1 = B1PhaseConditioned()
    b1.fit(records)

    gbt_params = {
        "n_estimators": 200, "max_depth": 4, "learning_rate": 0.1,
        "subsample": 0.8, "random_state": 42,
    }
    q_causal = StateActionModel(GradientBoostingRegressor(**gbt_params), "Q_CAUSAL")
    q_causal.fit(X, y)

    # Observational subset
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

    q_obs = StateActionModel(GradientBoostingRegressor(**gbt_params), "Q_OBS")
    if len(obs_records) > 10:
        X_obs = np.array([extract_features(r) for r in obs_records])
        y_obs = np.array([r["terminal_utility"] for r in obs_records])
        q_obs.fit(X_obs, y_obs)

    models = {
        "B0": b0,
        "B1": b1,
        "Q_CAUSAL": q_causal,
        "Q_OBS": q_obs,
    }

    # Group records by task
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_task[r["task_id"]].append(r)

    # Build per-task Q* lookup: q_star[task_id][action] = utility
    q_star: dict[str, dict[str, float]] = {}
    for task_id, task_records in by_task.items():
        q_star[task_id] = {}
        for r in task_records:
            q_star[task_id][r["forced_action"]] = r["terminal_utility"]

    # ---- Per-subtype analysis ----
    print("\n=== Per-Subtype Top-Action Accuracy ===")
    subtypes = sorted(set(r["category"] for r in records))

    per_subtype: dict[str, dict] = {}
    for subtype in subtypes:
        subtype_tasks = [tid for tid, trs in by_task.items()
                         if trs[0]["category"] == subtype]
        print(f"\n  {subtype} ({len(subtype_tasks)} tasks):")

        subtype_results = {}
        for model_name, model in models.items():
            correct = 0
            selected_actions = []
            for task_id in subtype_tasks:
                sf = by_task[task_id][0]["state_features"]
                preds = model.predict_all(sf)
                selected = max(preds, key=preds.get) if preds else "UNKNOWN"
                selected_actions.append(selected)
                correct_action = by_task[task_id][0]["correct_first_action"]
                if selected == correct_action:
                    correct += 1

            acc = correct / len(subtype_tasks) if subtype_tasks else 0
            subtype_results[model_name] = {
                "top_action_accuracy": round(acc, 4),
                "n_correct": correct,
                "n_tasks": len(subtype_tasks),
                "selected_actions": dict(Counter(selected_actions)),
            }
            print(f"    {model_name:15s}: acc={acc:.4f} ({correct}/{len(subtype_tasks)})")

        per_subtype[subtype] = subtype_results

    # ---- Per-subtype regret ----
    print("\n=== Per-Subtype Regret (vs Q* of correct action) ===")
    regret_by_subtype: dict[str, dict] = {}
    for subtype in subtypes:
        subtype_tasks = [tid for tid, trs in by_task.items()
                         if trs[0]["category"] == subtype]
        subtype_regrets = {}
        for model_name, model in models.items():
            regrets = []
            for task_id in subtype_tasks:
                sf = by_task[task_id][0]["state_features"]
                preds = model.predict_all(sf)
                selected = max(preds, key=preds.get) if preds else "UNKNOWN"
                correct_action = by_task[task_id][0]["correct_first_action"]
                q_correct = q_star[task_id].get(correct_action, 0.0)
                q_selected = q_star[task_id].get(selected, 0.0)
                regrets.append(q_correct - q_selected)
            mean_regret = np.mean(regrets) if regrets else 0
            subtype_regrets[model_name] = round(float(mean_regret), 4)
            print(f"  {subtype:15s} × {model_name:15s}: regret={mean_regret:+.4f}")
        regret_by_subtype[subtype] = subtype_regrets

    # ---- Rescues and breaks vs B0 ----
    print("\n=== Rescues and Breaks vs B0 ===")
    rescue_break: dict[str, dict] = {}
    for model_name, model in models.items():
        if model_name == "B0":
            continue
        rescues = 0
        breaks = 0
        ties = 0
        for task_id, task_records in by_task.items():
            sf = task_records[0]["state_features"]
            model_preds = model.predict_all(sf)
            model_action = max(model_preds, key=model_preds.get) if model_preds else "UNKNOWN"
            b0_preds = b0.predict_all(sf)
            b0_action = max(b0_preds, key=b0_preds.get) if b0_preds else "UNKNOWN"

            model_u = q_star[task_id].get(model_action, 0.0)
            b0_u = q_star[task_id].get(b0_action, 0.0)

            if model_u > b0_u:
                rescues += 1
            elif model_u < b0_u:
                breaks += 1
            else:
                ties += 1

        rescue_break[model_name] = {
            "rescues": rescues,
            "breaks": breaks,
            "ties": ties,
            "n_tasks": len(by_task),
        }
        print(f"  {model_name:15s}: rescues={rescues}, breaks={breaks}, ties={ties}")

    # ---- Mechanism audit: does Q_CAUSAL increase the correct action per subtype? ----
    print("\n=== Mechanism Audit: Q_CAUSAL action selection per subtype ===")
    mechanism: dict[str, dict] = {}
    desired_actions = {
        "ol_answer": "ANSWER",
        "ol_defer": "DEFER",
        "ol_retrieve": "RETRIEVE",
        "ol_verify": "VERIFY",
        "ol_search": "SEARCH_MORE",
        "tl_answer": "ANSWER",
        "tl_defer": "DEFER",
        "tl_retrieve": "RETRIEVE",
        "tl_verify": "VERIFY",
        "tl_search": "SEARCH_MORE",
    }
    for subtype in subtypes:
        subtype_tasks = [tid for tid, trs in by_task.items()
                         if trs[0]["category"] == subtype]
        desired = desired_actions.get(subtype, "UNKNOWN")

        b0_actions = []
        qcausal_actions = []
        for task_id in subtype_tasks:
            sf = by_task[task_id][0]["state_features"]
            b0_preds = b0.predict_all(sf)
            b0_a = max(b0_preds, key=b0_preds.get) if b0_preds else "UNKNOWN"
            b0_actions.append(b0_a)

            qc_preds = q_causal.predict_all(sf)
            qc_a = max(qc_preds, key=qc_preds.get) if qc_preds else "UNKNOWN"
            qcausal_actions.append(qc_a)

        b0_correct_rate = sum(1 for a in b0_actions if a == desired) / len(b0_actions)
        qc_correct_rate = sum(1 for a in qcausal_actions if a == desired) / len(qcausal_actions)

        mechanism[subtype] = {
            "desired_action": desired,
            "b0_correct_rate": round(b0_correct_rate, 4),
            "qcausal_correct_rate": round(qc_correct_rate, 4),
            "b0_action_distribution": dict(Counter(b0_actions)),
            "qcausal_action_distribution": dict(Counter(qcausal_actions)),
            "qcausal_increases_desired": qc_correct_rate > b0_correct_rate,
        }
        print(f"  {subtype:15s}: desired={desired:15s} B0={b0_correct_rate:.2f} Q_C={qc_correct_rate:.2f} increases={qc_correct_rate > b0_correct_rate}")

    # ---- Overall summary ----
    print("\n=== Overall Summary ===")
    overall: dict[str, dict] = {}
    for model_name, model in models.items():
        all_correct = 0
        all_regrets = []
        for task_id, task_records in by_task.items():
            sf = task_records[0]["state_features"]
            preds = model.predict_all(sf)
            selected = max(preds, key=preds.get) if preds else "UNKNOWN"
            correct_action = task_records[0]["correct_first_action"]
            if selected == correct_action:
                all_correct += 1
            q_correct = q_star[task_id].get(correct_action, 0.0)
            q_selected = q_star[task_id].get(selected, 0.0)
            all_regrets.append(q_correct - q_selected)

        overall[model_name] = {
            "top_action_accuracy": round(all_correct / len(by_task), 4),
            "mean_regret": round(float(np.mean(all_regrets)), 4),
            "n_tasks": len(by_task),
        }
        print(f"  {model_name:15s}: acc={all_correct/len(by_task):.4f}, regret={np.mean(all_regrets):+.4f}")

    # ---- Save ----
    validation = {
        "per_subtype_accuracy": per_subtype,
        "per_subtype_regret": regret_by_subtype,
        "rescue_break_vs_b0": rescue_break,
        "mechanism_audit": mechanism,
        "overall": overall,
        "n_tasks": len(by_task),
        "n_records": len(records),
        "subtypes": subtypes,
    }

    val_content = json.dumps(validation, sort_keys=True)
    val_sha = hashlib.sha256(val_content.encode()).hexdigest()

    manifest = {
        "validation_sha256": val_sha,
        "n_tasks": len(by_task),
        "n_records": len(records),
        "model_ladder_sha256": ladder.get("model_ladder_sha256"),
    }

    val_path = output_dir / "offline_validation_v1.json"
    with open(val_path, "w") as f:
        json.dump(validation, f, indent=2, sort_keys=True)
    print(f"\nWritten validation to {val_path}")

    manifest_path = output_dir / "offline_validation_v1_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"Written manifest to {manifest_path}")


if __name__ == "__main__":
    main()
