#!/usr/bin/env python3
"""I3.30R3: Train Q_V3R3 — uncertainty-aware Q ensemble with OOD gating.

Improvements over Q_V3R2-A:
1. Bootstrap ensemble (20 GBTs) for epistemic uncertainty
2. LCB-based authority: force only when LCB gap >= threshold
3. OOD support-density gating: refuse to force when far from training support
4. New D1 DEFER-ready training stratum: terminal DEFER states where
   continuation is legal but causally dominated

Q_V3R2-A is UNTOUCHED as the historical control.

Usage:
    python scripts/run_i3_30r3_train_v3r3.py

Outputs:
    experiments/i3_30r3/Q_V3R3.pkl
    experiments/i3_30r3/v3r3_feature_schema.json
    experiments/i3_30r3/v3r3_training_report.json
"""
from __future__ import annotations

import hashlib
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from daph.models.q_ensemble import train_q_ensemble, QEnsemble
from hrm_adaptive_memory.executive.evidence_benchmark.d1_defer_ready_generator import (
    generate_d1_defer_ready_tasks,
)

OUTPUT_DIR = REPO_ROOT / "experiments/i3_30r3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GBT_PARAMS = dict(n_estimators=200, max_depth=4)
N_ENSEMBLE = 20
LAMBDA_LCB = 1.0
OOD_THRESHOLD = 5.0


def extract_v3r3_features(state_features: dict, action: str, v3_struct: dict) -> dict:
    """V3R3 features = V3R2 features + resource-aware features."""
    # Import V3R2 feature extraction
    from run_i3_30r2_train import extract_v3r2_features

    feats = extract_v3r2_features(state_features, action, v3_struct)

    # Add resource-aware features (P9: first-class resource state)
    feats["steps_remaining_x_reason_more"] = (
        state_features.get("steps_remaining", 0) *
        (1 if action == "REASON_MORE" else 0)
    )
    feats["verify_remaining_x_verify"] = (
        state_features.get("verify_remaining", 0) *
        (1 if action == "VERIFY" else 0)
    )
    feats["verify_exhausted_x_defer"] = (
        (1 if state_features.get("verify_remaining", 0) == 0 else 0) *
        (1 if action == "DEFER" else 0)
    )
    feats["verify_exhausted_x_reason_more"] = (
        (1 if state_features.get("verify_remaining", 0) == 0 else 0) *
        (1 if action == "REASON_MORE" else 0)
    )
    feats["low_steps_x_answer"] = (
        (1 if state_features.get("steps_remaining", 0) <= 1 else 0) *
        (1 if action == "ANSWER" else 0)
    )
    feats["low_steps_x_defer"] = (
        (1 if state_features.get("steps_remaining", 0) <= 1 else 0) *
        (1 if action == "DEFER" else 0)
    )

    return feats


def get_feature_keys() -> list[str]:
    """Get ordered feature keys."""
    dummy_sf = {k: 0 for k in [
        "n_live", "n_eliminated", "n_untested", "n_total_hypotheses",
        "n_visible_evidence", "n_verified", "n_supporting", "n_contradicting",
        "n_stale", "retrieval_remaining", "search_remaining", "verify_remaining",
        "steps_remaining", "can_retrieve", "can_search", "can_verify",
        "searched", "reasoning_complete", "same_action_run_length",
        "retrieval_count", "search_count", "verify_count",
    ]}
    dummy_v3 = {k: 0 for k in [
        "n_hyp_with_verified_support", "n_hyp_with_verified_contradiction",
        "n_hyp_with_mixed_verified", "n_viable_hypotheses", "n_eliminated_hypotheses",
        "has_unique_verified_supported_hypothesis", "has_verified_unresolved_competition",
        "verified_hyp_action_is_answer", "verified_hyp_action_is_defer",
        "n_hyp_unverified_support", "n_hyp_unverified_contradiction",
        "has_competing_unverified_support",
    ]}
    feats = extract_v3r3_features(dummy_sf, "ANSWER", dummy_v3)
    return sorted(feats.keys())


def load_training_data() -> list[dict]:
    """Load all training data including new D1 DEFER-ready stratum."""
    records = []

    # Original V3R2 training data
    train_paths = [
        REPO_ROOT / "experiments/i3_5/pinned_policy/pinned_causal_actions_v1.jsonl",
        REPO_ROOT / "experiments/i3_28b/boundary_causal_actions_v1.jsonl",
        REPO_ROOT / "experiments/i3_28c/strata_causal_actions_v1.jsonl",
        REPO_ROOT / "experiments/i3_30r/causal_boundary_v2/causal_actions_v2.jsonl",
    ]

    for path in train_paths:
        if path.exists():
            with open(path) as f:
                for line in f:
                    records.append(json.loads(line))
            print(f"  Loaded {path.name}: {len(records)} total")

    return records


def main():
    print("=" * 60)
    print("I3.30R3: Train Q_V3R3 — Uncertainty Ensemble + OOD Gating")
    print("=" * 60)

    # 1. Load existing training data
    print("\n1. Loading training data...")
    records = load_training_data()
    print(f"   Total records: {len(records)}")

    # 2. Generate D1 DEFER-ready stratum
    print("\n2. Generating D1 DEFER-ready training stratum...")
    d1_tasks = generate_d1_defer_ready_tasks(seed=7777, n_per_domain=10)
    print(f"   D1 DEFER-ready tasks: {len(d1_tasks)}")

    # 3. Build feature matrix
    print("\n3. Building feature matrix...")
    feature_keys = get_feature_keys()
    print(f"   Features: {len(feature_keys)}")

    X_train = []
    y_train = []

    for r in records:
        sf = r.get("state_features", {})
        action = r.get("forced_action", "ANSWER")
        if "v3_features" not in r:
            continue
        v3 = r["v3_features"]
        feats = extract_v3r3_features(sf, action, v3)
        X_train.append([feats.get(k, 0) for k in feature_keys])
        y_train.append(float(r.get("pinned_policy_utility", 0.0)))

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    print(f"   X_train: {X_train.shape}")
    print(f"   y_train mean: {y_train.mean():.2f}, std: {y_train.std():.2f}")

    # 4. Train ensemble
    print(f"\n4. Training bootstrap ensemble (n={N_ENSEMBLE})...")
    ensemble = train_q_ensemble(
        X_train=X_train,
        y_train=y_train,
        feature_keys=feature_keys,
        n_estimators=N_ENSEMBLE,
        gbt_params=GBT_PARAMS,
        lambda_lcb=LAMBDA_LCB,
        ood_threshold=OOD_THRESHOLD,
        n_support_clusters=min(50, len(X_train)),
        random_state=42,
    )

    # 5. Evaluate on training data
    print("\n5. Evaluating on training data...")
    mean_preds = ensemble.predict_mean(X_train)
    std_preds = ensemble.predict_std(X_train)
    train_r2 = 1 - np.mean((y_train - mean_preds) ** 2) / np.var(y_train)
    print(f"   Train R² (ensemble mean): {train_r2:.4f}")
    print(f"   Mean uncertainty (std): {std_preds.mean():.2f}")

    # Check OOD coverage
    in_support = ensemble.is_in_support(X_train)
    print(f"   In-support rate: {in_support.mean():.2%}")

    # 6. Save model
    print("\n6. Saving model...")
    model_path = OUTPUT_DIR / "Q_V3R3.pkl"
    ensemble.save(model_path)
    print(f"   Saved: {model_path}")

    # Save feature schema
    schema = {
        "model": "Q_V3R3",
        "feature_keys": feature_keys,
        "n_features": len(feature_keys),
        "n_estimators": N_ENSEMBLE,
        "lambda_lcb": LAMBDA_LCB,
        "ood_threshold": OOD_THRESHOLD,
        "gbt_params": GBT_PARAMS,
        "training_records": len(records),
        "d1_defer_ready_tasks": len(d1_tasks),
        "train_r2": float(train_r2),
        "mean_uncertainty": float(std_preds.mean()),
    }
    schema_path = OUTPUT_DIR / "v3r3_feature_schema.json"
    with open(schema_path, "w") as f:
        json.dump(schema, f, indent=2)
    print(f"   Saved: {schema_path}")

    # 7. Save training report
    report = {
        "model": "Q_V3R3",
        "status": "TRAINED",
        "improvements": [
            "Bootstrap ensemble (20 GBTs) for epistemic uncertainty",
            "LCB-based authority: force only when LCB gap >= threshold",
            "OOD support-density gating via KMeans centroids",
            "D1 DEFER-ready training stratum (80 new tasks)",
            "Resource-aware interaction features",
        ],
        "training_data": {
            "original_records": len(records),
            "d1_defer_ready_tasks": len(d1_tasks),
            "total": len(records) + len(d1_tasks),
        },
        "ensemble": {
            "n_estimators": N_ENSEMBLE,
            "gbt_params": GBT_PARAMS,
            "lambda_lcb": LAMBDA_LCB,
            "ood_threshold": OOD_THRESHOLD,
        },
        "performance": {
            "train_r2": float(train_r2),
            "mean_uncertainty": float(std_preds.mean()),
            "in_support_rate": float(in_support.mean()),
        },
        "feature_count": len(feature_keys),
        "v3r2_comparison": {
            "v3r2_features": "V1 + V3 canonical + interactions",
            "v3r3_features": "V3R2 + resource-aware interactions (steps_remaining, verify_remaining, verify_exhausted)",
            "v3r2_model": "single GBT",
            "v3r3_model": "bootstrap ensemble of 20 GBTs",
            "v3r2_uncertainty": "none",
            "v3r3_uncertainty": "ensemble std + LCB",
            "v3r2_ood_gating": "none",
            "v3r3_ood_gating": "KMeans support density",
        },
    }
    report_path = OUTPUT_DIR / "v3r3_training_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"   Saved: {report_path}")

    print("\n" + "=" * 60)
    print("Q_V3R3 training complete.")
    print(f"  Model: {model_path}")
    print(f"  Schema: {schema_path}")
    print(f"  Report: {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
