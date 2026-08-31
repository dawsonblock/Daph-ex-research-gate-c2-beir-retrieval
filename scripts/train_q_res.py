#!/usr/bin/env python3
"""Train the first Q_res model for DAPH-X.

Q_res(s,a) = Q_oracle(s,a) - Q_MB(s,a)

Where Q_MB is the model-based value (from the world model) and
Q_oracle is the actual utility from counterfactual evaluation.

The first model is deliberately simple: gradient-boosted trees
over graph-derived/action features.

Usage:
    python scripts/train_q_res.py [--corpus experiments/daph_x/causal_corpus/causal_corpus_v1.jsonl]
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def extract_features(record: dict, topology_cache: dict) -> dict[str, float]:
    """Extract features from a CausalActionRecord for Q_res training.

    ONLY uses features available at decision time (pre-action).
    Does NOT use outcome, success, or any post-hoc information.
    """
    feats = {}

    # Action type one-hot
    action_type = record["action_type"]
    for at in ["ANSWER", "DEFER", "VERIFY", "COMPARE", "STOP", "SEARCH", "RETRIEVE"]:
        feats[f"action_{at}"] = 1.0 if action_type == at else 0.0

    # Action cost (known before action)
    feats["action_cost"] = record["action_cost"]

    # Resource state (known before action)
    feats["steps_remaining"] = float(record["steps_remaining"])
    feats["verify_remaining"] = float(record["verify_remaining"])

    # Action target features
    feats["has_target"] = 1.0 if record["action_target"] else 0.0

    # Canonical topology features (stored in record, available at decision time)
    feats["n_supported"] = float(record.get("topo_n_supported", 0))
    feats["n_contradicted"] = float(record.get("topo_n_contradicted", 0))
    feats["n_weakened"] = float(record.get("topo_n_weakened", 0))
    feats["n_untested"] = float(record.get("topo_n_untested", 0))
    feats["has_unique_supported"] = 1.0 if record.get("topo_unique_supported") else 0.0
    feats["has_competition"] = 1.0 if record.get("topo_has_competition") else 0.0
    feats["unverified_exists"] = 1.0 if record.get("topo_unverified_exists") else 0.0

    # For ANSWER actions: is the target the unique supported hypothesis?
    if action_type == "ANSWER":
        feats["answer_is_unique_supported"] = (
            1.0 if record["action_target"] == record.get("topo_unique_supported", "") else 0.0
        )
    else:
        feats["answer_is_unique_supported"] = 0.0

    # For VERIFY actions: does unverified evidence exist?
    if action_type == "VERIFY":
        feats["verify_useful"] = 1.0 if record.get("topo_unverified_exists") else 0.0
    else:
        feats["verify_useful"] = 0.0

    return feats


def build_topology_cache(corpus_path: str) -> dict:
    """Build a cache of canonical topology for each checkpoint.

    Loads each unique checkpoint and derives its canonical topology.
    """
    from daph_x.receipts.checkpoint import Checkpoint
    from daph.epistemic.topology import derive_hypothesis_topology

    cache = {}
    seen_hashes = set()

    with open(corpus_path) as f:
        for line in f:
            record = json.loads(line)
            ch = record["checkpoint_hash"]
            if ch in seen_hashes:
                continue
            seen_hashes.add(ch)

            # We don't have the checkpoint data in the record itself
            # For now, use the graph hash as a proxy
            # In future: store full checkpoint data or topology in the record
            cache[ch] = {
                "n_viable_hypotheses": 0,
                "n_eliminated_hypotheses": 0,
                "n_weakened_hypotheses": 0,
                "n_untested_hypotheses": 0,
                "unique_supported_hypothesis": None,
                "has_verified_unresolved_competition": False,
                "unverified_evidence_exists": False,
            }

    return cache


def compute_q_mb(record: dict) -> float:
    """Compute the model-based Q value for a record.

    For now, uses the same heuristic as the executive scorer.
    In future, this will use the actual world model.
    """
    action_type = record["action_type"]

    # Base values (same as executive scorer)
    base_values = {
        "ANSWER": 100.0,
        "DEFER": 50.0,
        "STOP": -10.0,
        "VERIFY": 30.0,
        "COMPARE": 10.0,
        "SEARCH": 15.0,
        "RETRIEVE": 20.0,
    }
    base = base_values.get(action_type, 0.0)

    # Adjust for cost
    cost = record["action_cost"]
    return base - 0.5 * cost


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="experiments/daph_x/causal_corpus/causal_corpus_v1.jsonl")
    args = parser.parse_args()

    # Load corpus
    records = []
    with open(args.corpus) as f:
        for line in f:
            records.append(json.loads(line))

    print(f"Loaded {len(records)} records")
    print(f"Counterfactual groups: {len(set(r['counterfactual_group_id'] for r in records))}")

    # Extract features and targets (topology is now stored in each record)
    feature_dicts = [extract_features(r, {}) for r in records]
    feature_keys = sorted(feature_dicts[0].keys())
    X = np.array([[f[k] for k in feature_keys] for f in feature_dicts])

    # Q_oracle = actual utility
    y_oracle = np.array([r["utility"] for r in records])

    # Q_MB = model-based prediction
    q_mb = np.array([compute_q_mb(r) for r in records])

    # Q_res = Q_oracle - Q_MB (the residual we want to learn)
    y_res = y_oracle - q_mb

    print(f"\nFeature keys: {feature_keys}")
    print(f"X shape: {X.shape}")
    print(f"y_oracle: min={y_oracle.min():.1f}, max={y_oracle.max():.1f}, mean={y_oracle.mean():.1f}")
    print(f"q_mb: min={q_mb.min():.1f}, max={q_mb.max():.1f}, mean={q_mb.mean():.1f}")
    print(f"y_res: min={y_res.min():.1f}, max={y_res.max():.1f}, mean={y_res.mean():.1f}")

    # Split by counterfactual_group_id (no leakage)
    group_ids = list(set(r["counterfactual_group_id"] for r in records))
    np.random.seed(42)
    np.random.shuffle(group_ids)
    split = int(0.8 * len(group_ids))
    train_groups = set(group_ids[:split])
    test_groups = set(group_ids[split:])

    train_mask = np.array([r["counterfactual_group_id"] in train_groups for r in records])
    test_mask = np.array([r["counterfactual_group_id"] in test_groups for r in records])

    X_train, X_test = X[train_mask], X[test_mask]
    y_res_train, y_res_test = y_res[train_mask], y_res[test_mask]
    y_oracle_test = y_oracle[test_mask]
    q_mb_test = q_mb[test_mask]

    print(f"\nTrain: {X_train.shape[0]} records ({len(train_groups)} groups)")
    print(f"Test: {X_test.shape[0]} records ({len(test_groups)} groups)")

    # Train Q_res model
    print(f"\nTraining GradientBoostingRegressor...")
    model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
    )
    model.fit(X_train, y_res_train)

    # Evaluate
    q_res_pred = model.predict(X_test)
    q_hybrid = q_mb_test + q_res_pred

    # MAE comparison
    mae_mb = mean_absolute_error(y_oracle_test, q_mb_test)
    mae_hybrid = mean_absolute_error(y_oracle_test, q_hybrid)
    rmse_mb = np.sqrt(mean_squared_error(y_oracle_test, q_mb_test))
    rmse_hybrid = np.sqrt(mean_squared_error(y_oracle_test, q_hybrid))

    print(f"\n{'='*50}")
    print(f"RESULTS")
    print(f"{'='*50}")
    print(f"MAE(Q_MB):           {mae_mb:.2f}")
    print(f"MAE(Q_MB + Q_res):   {mae_hybrid:.2f}")
    print(f"Improvement:         {mae_mb - mae_hybrid:.2f} ({100*(mae_mb - mae_hybrid)/mae_mb:.1f}%)")
    print(f"RMSE(Q_MB):          {rmse_mb:.2f}")
    print(f"RMSE(Q_MB + Q_res):  {rmse_hybrid:.2f}")

    # Action regret comparison — CORRECTED
    # For each test group, find the best action by Q_MB and by Q_hybrid
    # CRITICAL: use test_records indices (aligned with X_test/q_res_pred),
    # NOT group-relative indices.
    test_records = [r for r, m in zip(records, test_mask) if m]
    test_groups_map = defaultdict(list)
    for i, r in enumerate(test_records):
        test_groups_map[r["counterfactual_group_id"]].append((i, r))

    regret_mb = []
    regret_hybrid = []
    for gid, group in test_groups_map.items():
        if len(group) < 2:
            continue
        # Oracle
        oracle_utility = max(r["utility"] for _, r in group)

        # Q_MB best (recomputed per record, consistent with training)
        q_mb_group = [compute_q_mb(r) for _, r in group]
        best_mb_idx = np.argmax(q_mb_group)
        regret_mb.append(oracle_utility - group[best_mb_idx][1]["utility"])

        # Q_hybrid best — use CORRECT indices into q_res_pred
        # i is the index in test_records (aligned with X_test/q_res_pred)
        q_hybrid_group = [q_mb_test[i] + q_res_pred[i] for i, _ in group]
        best_hybrid_idx = np.argmax(q_hybrid_group)
        regret_hybrid.append(oracle_utility - group[best_hybrid_idx][1]["utility"])

    # Invariant: if predictions are perfect, regret must be zero
    max_pred_error = np.max(np.abs(q_hybrid - y_oracle_test))
    if max_pred_error < 1e-9:
        assert all(r == 0.0 for r in regret_hybrid), (
            f"Perfect predictions but nonzero regret: {regret_hybrid}"
        )

    if regret_mb:
        print(f"\nAction regret (test groups with >1 action):")
        print(f"  Groups: {len(regret_mb)}")
        print(f"  Regret(Q_MB):          mean={np.mean(regret_mb):.2f}")
        print(f"  Regret(Q_MB + Q_res):  mean={np.mean(regret_hybrid):.2f}")
        print(f"  Improvement:           {np.mean(regret_mb) - np.mean(regret_hybrid):.2f}")

    # Feature importance
    print(f"\nFeature importance:")
    importances = model.feature_importances_
    for k, imp in sorted(zip(feature_keys, importances), key=lambda x: -x[1])[:10]:
        print(f"  {k}: {imp:.3f}")

    # Save model
    import joblib
    model_path = Path(args.corpus).parent / "q_res_v1.pkl"
    joblib.dump({
        "model": model,
        "feature_keys": feature_keys,
        "mae_mb": mae_mb,
        "mae_hybrid": mae_hybrid,
        "rmse_mb": rmse_mb,
        "rmse_hybrid": rmse_hybrid,
    }, model_path)
    print(f"\nSaved model to {model_path}")


if __name__ == "__main__":
    main()
