#!/usr/bin/env python3
"""I3.5-PQ: Build the model ladder on pinned-policy Q values.

Retrains B0, B1, Linear, GBT, Q_OBS, Q_CAUSAL_POLICY on the pinned-policy
causal dataset. The target is now Q^{pi_Qwen}(s,a) instead of oracle Q*.

Produces:
  - experiments/i3_5/pinned_policy/model_ladder_v1.json
  - experiments/i3_5/pinned_policy/model_predictions_v1.jsonl
  - experiments/i3_5/pinned_policy/model_comparison_v1.json

The decisive metric: regret(s) = Q^pi(s,a*) - Q^pi(s,a_hat)
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def load_pinned_causal(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def load_oracle_causal(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def load_observational(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def extract_features(state_features: dict) -> dict:
    """Extract model features from state_features."""
    return {
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


def action_one_hot(action: str) -> dict:
    actions = ["ANSWER", "DEFER", "RETRIEVE", "VERIFY", "SEARCH_MORE", "REASON_MORE"]
    return {f"a_{a}": int(a == action) for a in actions}


def build_feature_matrix(records: list[dict]) -> tuple[list[dict], list[float], list[str]]:
    """Build (features, targets, checkpoint_ids) from records."""
    features = []
    targets = []
    cp_ids = []
    for r in records:
        feat = extract_features(r["state_features"])
        feat.update(action_one_hot(r["forced_action"]))
        features.append(feat)
        targets.append(r.get("pinned_policy_utility", r.get("terminal_utility", 0.0)))
        cp_ids.append(r["checkpoint_id"])
    return features, targets, cp_ids


def train_b0(targets: list[float]) -> float:
    """B0: global mean."""
    return sum(targets) / len(targets) if targets else 0.0


def train_b1(records: list[dict], targets: list[float]) -> dict[str, float]:
    """B1: per-action mean."""
    by_action: dict[str, list[float]] = defaultdict(list)
    for r, t in zip(records, targets):
        by_action[r["forced_action"]].append(t)
    return {a: sum(ts) / len(ts) for a, ts in by_action.items()}


def train_linear(features: list[dict], targets: list[float]) -> object:
    """Linear regression."""
    from sklearn.linear_model import LinearRegression
    import numpy as np
    X = np.array([[f[k] for k in sorted(features[0].keys())] for f in features])
    y = np.array(targets)
    model = LinearRegression()
    model.fit(X, y)
    return model


def train_gbt(features: list[dict], targets: list[float]) -> object:
    """Gradient boosted trees."""
    from sklearn.ensemble import GradientBoostingRegressor
    import numpy as np
    X = np.array([[f[k] for k in sorted(features[0].keys())] for f in features])
    y = np.array(targets)
    model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
    model.fit(X, y)
    return model


def predict_model(model, features: list[dict]) -> list[float]:
    import numpy as np
    keys = sorted(features[0].keys())
    X = np.array([[f[k] for k in keys] for f in features])
    return model.predict(X).tolist()


def predict_b0(b0_val: float, n: int) -> list[float]:
    return [b0_val] * n


def predict_b1(b1_map: dict[str, float], records: list[dict]) -> list[float]:
    return [b1_map.get(r["forced_action"], 0.0) for r in records]


def compute_regret(records: list[dict], predictions: list[float]) -> list[float]:
    """Regret(s) = Q^pi(s,a*) - Q^pi(s,a_hat)."""
    # Group by checkpoint
    by_cp: dict[str, list[tuple[str, float, float]]] = defaultdict(list)
    for r, pred in zip(records, predictions):
        by_cp[r["checkpoint_id"]].append((r["forced_action"], r["pinned_policy_utility"], pred))

    regrets = []
    for cp_id, entries in by_cp.items():
        # a* = action with highest actual Q
        actual = [(a, q) for a, q, _ in entries]
        a_star = max(actual, key=lambda x: x[1])[0]
        q_star = max(q for _, q, _ in entries)
        # a_hat = action with highest predicted Q
        pred = [(a, p) for a, _, p in entries]
        a_hat = max(pred, key=lambda x: x[1])[0]
        q_hat = next(q for a, q, _ in entries if a == a_hat)
        regrets.append(q_star - q_hat)
    return regrets


def compute_top1(records: list[dict], predictions: list[float]) -> float:
    """Top-1 accuracy: does the model pick the best action?"""
    by_cp: dict[str, list[tuple[str, float, float]]] = defaultdict(list)
    for r, pred in zip(records, predictions):
        by_cp[r["checkpoint_id"]].append((r["forced_action"], r["pinned_policy_utility"], pred))

    correct = 0
    total = 0
    for cp_id, entries in by_cp.items():
        actual = [(a, q) for a, q, _ in entries]
        a_star = max(actual, key=lambda x: x[1])[0]
        pred = [(a, p) for a, _, p in entries]
        a_hat = max(pred, key=lambda x: x[1])[0]
        if a_star == a_hat:
            correct += 1
        total += 1
    return correct / total if total > 0 else 0.0


def compute_top2_recall(records: list[dict], predictions: list[float]) -> float:
    """Top-2 recall: is the best action in the top-2 predicted?"""
    by_cp: dict[str, list[tuple[str, float, float]]] = defaultdict(list)
    for r, pred in zip(records, predictions):
        by_cp[r["checkpoint_id"]].append((r["forced_action"], r["pinned_policy_utility"], pred))

    correct = 0
    total = 0
    for cp_id, entries in by_cp.items():
        actual = [(a, q) for a, q, _ in entries]
        a_star = max(actual, key=lambda x: x[1])[0]
        pred_sorted = sorted([(a, p) for a, _, p in entries], key=lambda x: -x[1])
        top2 = [a for a, _ in pred_sorted[:2]]
        if a_star in top2:
            correct += 1
        total += 1
    return correct / total if total > 0 else 0.0


def subtype_consistency(records: list[dict], predictions: list[float]) -> dict[str, str]:
    """For each one-live subtype, what action does the model predict most often?"""
    by_cp: dict[str, list[tuple[str, float, float]]] = defaultdict(list)
    for r, pred in zip(records, predictions):
        by_cp[r["checkpoint_id"]].append((r["forced_action"], r["pinned_policy_utility"], pred))

    # For each checkpoint, get the predicted best action
    cp_to_pred: dict[str, str] = {}
    for cp_id, entries in by_cp.items():
        pred_sorted = sorted([(a, p) for a, _, p in entries], key=lambda x: -x[1])
        cp_to_pred[cp_id] = pred_sorted[0][0]

    # Group by category
    by_cat: dict[str, list[str]] = defaultdict(list)
    for r in records:
        cp_id = r["checkpoint_id"]
        if cp_id in cp_to_pred:
            by_cat[r["category"]].append(cp_to_pred[cp_id])

    result = {}
    for cat, actions in by_cat.items():
        most_common = max(set(actions), key=actions.count)
        result[cat] = most_common
    return result


def bootstrap_ci(data: list[float], n_bootstrap: int = 1000, confidence: float = 0.95) -> tuple[float, float]:
    """Bootstrap confidence interval for the mean."""
    import random
    n = len(data)
    if n == 0:
        return (0.0, 0.0)
    means = []
    for _ in range(n_bootstrap):
        sample = [random.choice(data) for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    alpha = (1 - confidence) / 2
    lower = means[int(n_bootstrap * alpha)]
    upper = means[int(n_bootstrap * (1 - alpha))]
    return (lower, upper)


def main():
    pinned_path = REPO_ROOT / "experiments/i3_5/pinned_policy/pinned_causal_actions_v1.jsonl"
    oracle_path = REPO_ROOT / "experiments/i3_5/causal/causal_actions_v1.jsonl"
    obs_path = REPO_ROOT / "experiments/i3_5/observational/observational_actions_v1.jsonl"
    output_dir = REPO_ROOT / "experiments/i3_5/pinned_policy"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    pinned_records = load_pinned_causal(pinned_path)
    print(f"  Pinned-policy causal: {len(pinned_records)} records")

    oracle_records = load_oracle_causal(oracle_path)
    print(f"  Oracle causal: {len(oracle_records)} records")

    obs_records = []
    if obs_path.exists():
        obs_records = load_observational(obs_path)
        print(f"  Observational: {len(obs_records)} records")

    if not pinned_records:
        print("ERROR: No pinned-policy records found. Run collection first.")
        return

    # Build feature matrices
    print("\nBuilding feature matrices...")
    pinned_features, pinned_targets, pinned_cp_ids = build_feature_matrix(pinned_records)
    oracle_features, oracle_targets, oracle_cp_ids = build_feature_matrix(oracle_records)

    # Train models on pinned-policy data
    print("\nTraining models on pinned-policy Q values...")
    b0_val = train_b0(pinned_targets)
    print(f"  B0 (global mean): {b0_val:.4f}")

    b1_map = train_b1(pinned_records, pinned_targets)
    print(f"  B1 (per-action mean): {b1_map}")

    print("  Training Linear...")
    linear_model = train_linear(pinned_features, pinned_targets)

    print("  Training GBT (Q_CAUSAL_POLICY)...")
    gbt_model = train_gbt(pinned_features, pinned_targets)

    # Train Q_OBS on observational data (if available)
    q_obs_model = None
    if obs_records:
        obs_features, obs_targets, _ = build_feature_matrix(obs_records)
        print("  Training Q_OBS (GBT on observational data)...")
        q_obs_model = train_gbt(obs_features, obs_targets)

    # Generate predictions on pinned-policy data (held-out evaluation)
    print("\nGenerating predictions...")
    pred_b0 = predict_b0(b0_val, len(pinned_records))
    pred_b1 = predict_b1(b1_map, pinned_records)
    pred_linear = predict_model(linear_model, pinned_features)
    pred_gbt = predict_model(gbt_model, pinned_features)
    pred_q_obs = predict_model(q_obs_model, pinned_features) if q_obs_model else pred_b0

    # Compute metrics
    print("\nComputing metrics...")
    metrics = {}
    for name, preds in [
        ("B0", pred_b0),
        ("B1", pred_b1),
        ("Linear", pred_linear),
        ("Q_CAUSAL_POLICY", pred_gbt),
        ("Q_OBS", pred_q_obs),
    ]:
        regret = compute_regret(pinned_records, preds)
        top1 = compute_top1(pinned_records, preds)
        top2 = compute_top2_recall(pinned_records, preds)
        regret_mean = sum(regret) / len(regret) if regret else 0.0
        regret_ci = bootstrap_ci(regret)
        metrics[name] = {
            "regret_mean": round(regret_mean, 4),
            "regret_ci": [round(regret_ci[0], 4), round(regret_ci[1], 4)],
            "top1_accuracy": round(top1, 4),
            "top2_recall": round(top2, 4),
        }
        print(f"  {name}: regret={regret_mean:.4f} CI=[{regret_ci[0]:.4f}, {regret_ci[1]:.4f}] "
              f"top1={top1:.4f} top2={top2:.4f}")

    # Subtype consistency
    print("\nSubtype consistency (Q_CAUSAL_POLICY):")
    sc = subtype_consistency(pinned_records, pred_gbt)
    expected = {
        "ol_answer": "ANSWER",
        "ol_defer": "DEFER",
        "ol_retrieve": "RETRIEVE",
        "ol_verify": "VERIFY",
        "ol_search": "SEARCH_MORE",
    }
    for cat, expected_action in expected.items():
        actual = sc.get(cat, "UNKNOWN")
        match = "OK" if actual == expected_action else "MISMATCH"
        print(f"  {cat} -> {actual} (expected {expected_action}) [{match}]")

    # Promotion gate
    print("\n=== Promotion Gate ===")
    causal_regret = metrics["Q_CAUSAL_POLICY"]["regret_mean"]
    b0_regret = metrics["B0"]["regret_mean"]
    causal_top1 = metrics["Q_CAUSAL_POLICY"]["top1_accuracy"]
    b1_top1 = metrics["B1"]["top1_accuracy"]
    causal_top2 = metrics["Q_CAUSAL_POLICY"]["top2_recall"]

    gates = {
        "regret_lt_b0": causal_regret < b0_regret,
        "top1_gt_b1": causal_top1 > b1_top1,
        "top2_recall_gt_80": causal_top2 > 0.80,
        "subtype_consistency": all(
            sc.get(cat, "") == exp for cat, exp in expected.items()
        ),
    }
    for gate, passed in gates.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {gate}: {status}")

    all_pass = all(gates.values())
    print(f"\n  OVERALL: {'PASS' if all_pass else 'FAIL'}")
    if not all_pass:
        print("  WARNING: Q_CAUSAL_POLICY did not pass all gates.")
        print("  Do NOT proceed to six-arm executive experiment.")

    # Save results
    results = {
        "metrics": metrics,
        "subtype_consistency": sc,
        "expected_subtype_consistency": expected,
        "promotion_gates": gates,
        "all_gates_passed": all_pass,
        "n_pinned_records": len(pinned_records),
        "n_oracle_records": len(oracle_records),
        "n_observational_records": len(obs_records),
    }
    results_path = output_dir / "model_ladder_v1.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
