#!/usr/bin/env python3
"""Train Q_res for DAPH-X M4 on multi-step rollout targets.

Q_res(s,a) = Q^π_oracle(s,a) - Q_MB(s,a)

Where:
  Q^π_oracle(s,a) = actual multi-step rollout utility
  Q_MB(s,a) = model-based heuristic (belief + topology, can be wrong)

Trains on the M4 train split, evaluates on structural_ood and mechanism_ood.

Usage:
    python scripts/train_m4_q_res.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

import joblib

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

M4_DIR = REPO_ROOT / "experiments/daph_x/m4"


# ─── Feature Extraction ───

# Forbidden fields that must NEVER appear in features (anti-leakage)
FORBIDDEN_FEATURES = {
    "utility", "oracle_utility", "regret", "delta_u", "is_harmful_intervention",
    "executive_utility", "executive_is_oracle", "base_utility",
    "correct_hypothesis_id", "terminal_reason", "success",
    "runtime_errors", "trajectory", "observation_path",
    "terminal_state_hash", "steps_used", "total_cost",
    "harm_mechanism", "mechanism_family",  # Labels, not features
}


def extract_m4_features(record: dict) -> dict[str, float]:
    """Extract pre-decision features from an M4 rollout record.

    ONLY uses information available BEFORE the action is executed.
    Does NOT use: utility, oracle_utility, regret, delta_u, terminal_reason,
    trajectory outcomes, or any post-action information.

    Uses embedded graph_features if available (M4 corpus v2+).
    Falls back to basic features for legacy records without graph_features.
    """
    feats = {}

    # Action type one-hot
    action_type = record["first_action_type"]
    for at in ["ANSWER", "DEFER", "VERIFY", "COMPARE", "STOP", "SEARCH", "RETRIEVE"]:
        feats[f"action_{at}"] = 1.0 if action_type == at else 0.0

    # Action target features (pre-decision)
    feats["has_target"] = 1.0 if record.get("first_action", "") != "" else 0.0

    # Resource state (pre-decision, from checkpoint)
    gen_params = record.get("generator_params", {})
    if isinstance(gen_params, str):
        import json as _json
        try:
            gen_params = _json.loads(gen_params.replace("'", '"'))
        except Exception:
            gen_params = {}
    feats["steps_remaining"] = float(gen_params.get("steps", 0))
    feats["verify_remaining"] = float(gen_params.get("verify_budget", 0))
    feats["search_remaining"] = float(gen_params.get("search_budget", 0))
    feats["n_hyp"] = float(gen_params.get("n_hyp", 0))
    feats["n_ev"] = float(gen_params.get("n_ev", 0))

    # World model config features (pre-decision)
    wm_config = record.get("world_model_config", {})
    if isinstance(wm_config, str):
        import json as _json
        try:
            wm_config = _json.loads(wm_config.replace("'", '"'))
        except Exception:
            wm_config = {}
    feats["wm_verify_sufficient"] = float(wm_config.get("verify_sufficient_prob", 0.7))
    feats["wm_verify_falsified"] = float(wm_config.get("verify_falsified_prob", 0.2))

    # Graph-structural features (replaces topo_hash_prefix)
    # These are embedded in the record during corpus building.
    # If graph_features is missing (legacy record), use basic fallbacks.
    graph_feats = record.get("graph_features", {})
    if graph_feats:
        feats.update(graph_feats)
    else:
        # Legacy fallback: minimal features (no graph structure available)
        # This should not happen with v2+ corpus, but prevents crashes.
        from daph_x.features.graph_features import _ensure_action_features
        _ensure_action_features(feats, action_type)
        # Set all graph feature defaults to 0
        for k in [
            "topo_n_supported", "topo_n_contradicted", "topo_n_untested",
            "topo_n_weakened", "topo_n_stale", "topo_n_mixed_verified",
            "topo_has_unique_supported", "topo_has_competition",
            "topo_verification_complete", "topo_unverified_exists",
            "graph_edge_density", "graph_support_edge_ratio", "graph_contradict_edge_ratio",
            "graph_mean_support_degree", "graph_std_support_degree",
            "graph_mean_contradict_degree", "graph_std_contradict_degree",
            "graph_support_entropy",
            "ev_n_verified", "ev_n_unverified", "ev_n_falsified", "ev_n_stale",
            "ev_verify_ratio", "ev_unverified_ratio",
            "ev_mean_source_reliability", "ev_mean_independence", "ev_std_source_reliability",
            "belief_entropy", "belief_confidence", "belief_top_two_margin",
            "belief_normalized_entropy",
            "resource_verify_ratio", "resource_steps_ratio",
            "resource_search_ratio", "resource_verify_per_hyp",
        ]:
            if k not in feats:
                feats[k] = 0.0

    # Q_MB score (pre-decision model prediction, stored in record)
    feats["q_mb_score"] = float(record.get("q_mb_score", 0.0))

    return feats


def compute_q_mb_from_record(record: dict) -> float:
    """Compute Q_MB for a record using the same heuristic as the corpus builder."""
    return float(record.get("q_mb_score", 0.0))


def load_m4_split(split_name: str) -> list[dict]:
    """Load an M4 split from JSONL."""
    path = M4_DIR / f"m4_{split_name}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in open(path)]


def train_q_res():
    """Train Q_res on M4 train split, evaluate on OOD splits.

    Trains two models:
    1. Q_res_value: predicts Q_oracle - Q_MB (value correction)
       — with boundary-weighted sample weights to prioritize near-boundary examples
    2. Q_pairwise: predicts ΔU = U(exec) - U(base) directly (pairwise advantage)
       — trained on group-level features, targets the advantage
    """
    # Load data
    train_records = load_m4_split("train")
    struct_ood_records = load_m4_split("structural_ood")
    mech_ood_records = load_m4_split("mechanism_ood")

    print(f"Train: {len(train_records)} records")
    print(f"Structural OOD: {len(struct_ood_records)} records")
    print(f"Mechanism OOD: {len(mech_ood_records)} records")

    # Extract features
    feature_dicts_train = [extract_m4_features(r) for r in train_records]
    feature_keys = sorted(feature_dicts_train[0].keys())

    # Verify no forbidden features
    for k in feature_keys:
        for forbidden in FORBIDDEN_FEATURES:
            assert forbidden not in k.lower(), f"Forbidden feature: {k}"

    X_train = np.array([[f[k] for k in feature_keys] for f in feature_dicts_train])

    # Targets: Q_oracle = rollout utility, Q_res = Q_oracle - Q_MB
    y_oracle_train = np.array([r["utility"] for r in train_records])
    q_mb_train = np.array([compute_q_mb_from_record(r) for r in train_records])
    y_res_train = y_oracle_train - q_mb_train

    print(f"\nFeature keys ({len(feature_keys)}): {feature_keys}")
    print(f"y_oracle: min={y_oracle_train.min():.1f}, max={y_oracle_train.max():.1f}, mean={y_oracle_train.mean():.1f}")
    print(f"q_mb: min={q_mb_train.min():.1f}, max={q_mb_train.max():.1f}, mean={q_mb_train.mean():.1f}")
    print(f"y_res: min={y_res_train.min():.1f}, max={y_res_train.max():.1f}, mean={y_res_train.mean():.1f}")

    # ── Compute boundary-weighted sample weights ──
    # Weight examples near the decision boundary more heavily.
    # For each group, compute |ΔU| = |U(exec) - U(base)|.
    # Examples in groups with small |ΔU| are near-boundary and matter more for FORCE.
    groups_train = defaultdict(list)
    for i, r in enumerate(train_records):
        groups_train[r["counterfactual_group_id"]].append((i, r))

    sample_weights = np.ones(len(train_records))
    delta_u_values = []
    for gid, group in groups_train.items():
        if len(group) < 2:
            continue
        # Find exec and base
        q_mb_group = [q_mb_train[i] for i, _ in group]
        exec_idx = max(range(len(group)), key=lambda j: q_mb_group[j])
        base_idx = None
        for j, (_, rec) in enumerate(group):
            if "DEFER" in rec["first_action"]:
                base_idx = j
                break
        if base_idx is None:
            base_idx = 0
        delta_u = group[exec_idx][1]["utility"] - group[base_idx][1]["utility"]
        delta_u_values.append(abs(delta_u))

    if delta_u_values:
        scale = np.median(delta_u_values)
        for gid, group in groups_train.items():
            if len(group) < 2:
                continue
            q_mb_group = [q_mb_train[i] for i, _ in group]
            exec_idx = max(range(len(group)), key=lambda j: q_mb_group[j])
            base_idx = None
            for j, (_, rec) in enumerate(group):
                if "DEFER" in rec["first_action"]:
                    base_idx = j
                    break
            if base_idx is None:
                base_idx = 0
            delta_u = abs(group[exec_idx][1]["utility"] - group[base_idx][1]["utility"])
            # Weight: 1.0 at boundary, ~0.33 far from boundary
            w = 1.0 / (1.0 + delta_u / max(scale, 1.0))
            for idx, _ in group:
                sample_weights[idx] = w

    print(f"\nBoundary-weighted training:")
    print(f"  |ΔU| scale (median): {scale:.2f}" if delta_u_values else "  No groups found")
    print(f"  Weight range: [{sample_weights.min():.3f}, {sample_weights.max():.3f}]")
    print(f"  Mean weight: {sample_weights.mean():.3f}")

    # Train Q_res value model with boundary weighting
    print(f"\nTraining Q_res value model (boundary-weighted)...")
    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X_train, y_res_train, sample_weight=sample_weights)

    # ── Train pairwise advantage model ──
    # This model directly predicts ΔU = U(exec) - U(base) from state features.
    # It captures the pairwise ordering that matters for FORCE decisions.
    print(f"\nTraining pairwise advantage model...")

    pairwise_features = []
    pairwise_targets = []
    pairwise_weights = []

    for gid, group in groups_train.items():
        if len(group) < 2:
            continue
        # Find exec (argmax Q_MB) and base (DEFER)
        q_mb_group = [q_mb_train[i] for i, _ in group]
        exec_idx = max(range(len(group)), key=lambda j: q_mb_group[j])
        base_idx = None
        for j, (_, rec) in enumerate(group):
            if "DEFER" in rec["first_action"]:
                base_idx = j
                break
        if base_idx is None:
            base_idx = 0

        exec_rec = group[exec_idx][1]
        base_rec = group[base_idx][1]
        delta_u = exec_rec["utility"] - base_rec["utility"]

        # Features: executive action's features + delta_q_mb
        exec_feats = extract_m4_features(exec_rec)
        feats = dict(exec_feats)
        feats["delta_q_mb"] = float(q_mb_group[exec_idx] - q_mb_group[base_idx])
        # Add Q_res prediction for exec and base as features
        exec_x = np.array([[exec_feats[k] for k in feature_keys]])
        base_feats_dict = extract_m4_features(base_rec)
        base_x = np.array([[base_feats_dict[k] for k in feature_keys]])
        feats["q_res_exec_pred"] = float(model.predict(exec_x)[0])
        feats["q_res_base_pred"] = float(model.predict(base_x)[0])
        feats["delta_q_res_pred"] = feats["q_res_exec_pred"] - feats["q_res_base_pred"]

        pairwise_features.append(feats)
        pairwise_targets.append(delta_u)
        # Weight near-boundary examples more
        pairwise_weights.append(1.0 / (1.0 + abs(delta_u) / max(scale, 1.0)))

    pairwise_feature_keys = sorted(pairwise_features[0].keys())
    X_pairwise = np.array([[f[k] for k in pairwise_feature_keys] for f in pairwise_features])
    y_pairwise = np.array(pairwise_targets)
    w_pairwise = np.array(pairwise_weights)

    # Verify no forbidden features in pairwise model
    for k in pairwise_feature_keys:
        for forbidden in FORBIDDEN_FEATURES:
            assert forbidden not in k.lower(), f"Forbidden pairwise feature: {k}"

    pairwise_model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    pairwise_model.fit(X_pairwise, y_pairwise, sample_weight=w_pairwise)

    # Evaluate pairwise model
    print(f"\n  Pairwise model feature keys ({len(pairwise_feature_keys)}): {pairwise_feature_keys}")
    print(f"  Pairwise targets: min={y_pairwise.min():.1f}, max={y_pairwise.max():.1f}, mean={y_pairwise.mean():.1f}")

    # Evaluate on each split
    results = {}
    for split_name, records in [
        ("train", train_records),
        ("structural_ood", struct_ood_records),
        ("mechanism_ood", mech_ood_records),
    ]:
        if not records:
            continue

        feature_dicts = [extract_m4_features(r) for r in records]
        X = np.array([[f[k] for k in feature_keys] for f in feature_dicts])
        y_oracle = np.array([r["utility"] for r in records])
        q_mb = np.array([compute_q_mb_from_record(r) for r in records])
        q_res_pred = model.predict(X)
        q_hybrid = q_mb + q_res_pred

        # MAE
        mae_mb = mean_absolute_error(y_oracle, q_mb)
        mae_hybrid = mean_absolute_error(y_oracle, q_hybrid)

        # Regret and pairwise evaluation
        groups = defaultdict(list)
        for i, r in enumerate(records):
            groups[r["counterfactual_group_id"]].append((i, r))

        regret_mb = []
        regret_hybrid = []
        top1_mb = 0
        top1_hybrid = 0
        n_groups = 0

        # Pairwise sign accuracy
        pairwise_preds = []
        pairwise_truths = []
        pairwise_maes = []

        for gid, group in groups.items():
            if len(group) < 2:
                continue
            n_groups += 1
            oracle_utility = max(r["utility"] for _, r in group)

            # Q_MB best
            q_mb_group = [q_mb[i] for i, _ in group]
            best_mb_idx = np.argmax(q_mb_group)
            regret_mb.append(oracle_utility - group[best_mb_idx][1]["utility"])
            if group[best_mb_idx][1]["utility"] == oracle_utility:
                top1_mb += 1

            # Q_hybrid best
            q_hybrid_group = [q_hybrid[i] for i, _ in group]
            best_hybrid_idx = np.argmax(q_hybrid_group)
            regret_hybrid.append(oracle_utility - group[best_hybrid_idx][1]["utility"])
            if group[best_hybrid_idx][1]["utility"] == oracle_utility:
                top1_hybrid += 1

            # Pairwise evaluation
            exec_idx = max(range(len(group)), key=lambda j: q_mb_group[j])
            base_idx = None
            for j, (_, rec) in enumerate(group):
                if "DEFER" in rec["first_action"]:
                    base_idx = j
                    break
            if base_idx is None:
                base_idx = 0

            exec_rec = group[exec_idx][1]
            base_rec = group[base_idx][1]
            delta_u = exec_rec["utility"] - base_rec["utility"]

            # Build pairwise features
            exec_feats = extract_m4_features(exec_rec)
            pw_feats = dict(exec_feats)
            pw_feats["delta_q_mb"] = float(q_mb_group[exec_idx] - q_mb_group[base_idx])
            exec_x = np.array([[exec_feats[k] for k in feature_keys]])
            base_feats_dict = extract_m4_features(base_rec)
            base_x = np.array([[base_feats_dict[k] for k in feature_keys]])
            pw_feats["q_res_exec_pred"] = float(model.predict(exec_x)[0])
            pw_feats["q_res_base_pred"] = float(model.predict(base_x)[0])
            pw_feats["delta_q_res_pred"] = pw_feats["q_res_exec_pred"] - pw_feats["q_res_base_pred"]

            pw_x = np.array([[pw_feats.get(k, 0.0) for k in pairwise_feature_keys]])
            pw_pred = pairwise_model.predict(pw_x)[0]

            pairwise_preds.append(pw_pred)
            pairwise_truths.append(delta_u)
            pairwise_maes.append(abs(pw_pred - delta_u))

        # Pairwise sign accuracy: fraction where sign(pred) == sign(truth)
        pairwise_preds = np.array(pairwise_preds)
        pairwise_truths = np.array(pairwise_truths)
        sign_acc = np.mean(np.sign(pairwise_preds) == np.sign(pairwise_truths)) if len(pairwise_preds) > 0 else 0.0
        # Sign accuracy on non-ties (|ΔU| > 5, meaningful interventions)
        non_tie_mask = np.abs(pairwise_truths) > 5.0
        sign_acc_nontie = np.mean(np.sign(pairwise_preds[non_tie_mask]) == np.sign(pairwise_truths[non_tie_mask])) if non_tie_mask.sum() > 0 else 0.0
        pairwise_mae = np.mean(pairwise_maes) if pairwise_maes else 0.0

        result = {
            "n_records": len(records),
            "n_groups": n_groups,
            "mae_mb": round(mae_mb, 2),
            "mae_hybrid": round(mae_hybrid, 2),
            "mae_improvement": round(mae_mb - mae_hybrid, 2),
            "mae_improvement_pct": round(100 * (mae_mb - mae_hybrid) / max(mae_mb, 0.01), 1),
            "regret_mb": round(float(np.mean(regret_mb)), 2) if regret_mb else 0.0,
            "regret_hybrid": round(float(np.mean(regret_hybrid)), 2) if regret_hybrid else 0.0,
            "regret_improvement": round(float(np.mean(regret_mb) - np.mean(regret_hybrid)), 2) if regret_hybrid else 0.0,
            "top1_mb": round(top1_mb / max(n_groups, 1), 3),
            "top1_hybrid": round(top1_hybrid / max(n_groups, 1), 3),
            "pairwise_mae": round(float(pairwise_mae), 2),
            "pairwise_sign_acc": round(float(sign_acc), 4),
            "pairwise_sign_acc_nontie": round(float(sign_acc_nontie), 4),
            "pairwise_n_nontie": int(non_tie_mask.sum()),
        }
        results[split_name] = result

        print(f"\n{'='*50}")
        print(f"  {split_name.upper()}")
        print(f"{'='*50}")
        print(f"  Records: {len(records)}, Groups: {n_groups}")
        print(f"  MAE(Q_MB):           {mae_mb:.2f}")
        print(f"  MAE(Q_MB + Q_res):   {mae_hybrid:.2f}")
        print(f"  MAE Improvement:     {mae_mb - mae_hybrid:.2f} ({100*(mae_mb - mae_hybrid)/max(mae_mb, 0.01):.1f}%)")
        if regret_hybrid:
            print(f"  Regret(Q_MB):        {np.mean(regret_mb):.2f}")
            print(f"  Regret(Q_hybrid):    {np.mean(regret_hybrid):.2f}")
            print(f"  Regret Improvement:  {np.mean(regret_mb) - np.mean(regret_hybrid):.2f}")
            print(f"  Top-1(Q_MB):         {top1_mb}/{n_groups} ({top1_mb/max(n_groups,1):.3f})")
            print(f"  Top-1(Q_hybrid):     {top1_hybrid}/{n_groups} ({top1_hybrid/max(n_groups,1):.3f})")
        print(f"  Pairwise MAE:        {pairwise_mae:.2f}")
        print(f"  Pairwise Sign Acc:   {sign_acc:.4f}")
        print(f"  Pairwise Sign Acc (non-tie, n={int(non_tie_mask.sum())}): {sign_acc_nontie:.4f}")

    # Feature importance
    print(f"\nQ_res value model feature importance:")
    importances = model.feature_importances_
    for k, imp in sorted(zip(feature_keys, importances), key=lambda x: -x[1])[:10]:
        print(f"  {k}: {imp:.4f}")

    print(f"\nPairwise model feature importance:")
    pw_importances = pairwise_model.feature_importances_
    for k, imp in sorted(zip(pairwise_feature_keys, pw_importances), key=lambda x: -x[1])[:10]:
        print(f"  {k}: {imp:.4f}")

    # Save models
    model_path = M4_DIR / "q_res_m4.pkl"
    joblib.dump({
        "model": model,
        "feature_keys": feature_keys,
        "results": results,
        "train_size": len(train_records),
        "boundary_weighted": True,
    }, model_path)
    print(f"\nSaved Q_res model to {model_path}")

    pairwise_model_path = M4_DIR / "pairwise_model_m4.pkl"
    joblib.dump({
        "model": pairwise_model,
        "feature_keys": pairwise_feature_keys,
        "train_size": len(pairwise_targets),
    }, pairwise_model_path)
    print(f"Saved pairwise model to {pairwise_model_path}")

    # Save results
    results_path = M4_DIR / "q_res_m4_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {results_path}")

    return results


if __name__ == "__main__":
    train_q_res()
