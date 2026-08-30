#!/usr/bin/env python3
"""I3.30R3: Train Q_V3R3 — uncertainty-aware Q ensemble with OOD gating.

Improvements over Q_V3R2-A:
1. Bootstrap ensemble of GBTs for epistemic uncertainty
2. LCB-based authority: force only when LCB gap >= threshold
3. OOD support-density gating: refuse to force when far from training support
4. New D1 DEFER-ready training stratum: terminal DEFER states where
   continuation is legal but causally dominated — ACTUALLY ROLLED OUT
   and converted to causal action training records

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
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from daph.models.q_ensemble import train_q_ensemble, QEnsemble
from hrm_adaptive_memory.executive.evidence_benchmark.d1_defer_ready_generator import (
    generate_d1_defer_ready_tasks,
)
from hrm_adaptive_memory.executive.evidence_benchmark.schema import (
    initial_evidence_runtime,
)
from hrm_adaptive_memory.executive.evidence_benchmark.executor import (
    EvidenceExecutor, valid_verify_targets,
)
from hrm_adaptive_memory.cognitive_control.core import DecisionAction
from hrm_adaptive_memory.executive.resources import ResourceState, ResourceBudget
from hrm_adaptive_memory.executive.metareasoning_utility import MetareasoningUtility
from daph.epistemic.v3_features import compute_v3_features_canonical
from daph.intervention.checkpoint import compute_state_features

OUTPUT_DIR = REPO_ROOT / "experiments/i3_30r3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GBT_PARAMS = dict(n_estimators=200, max_depth=4)
N_ENSEMBLE = 20
LAMBDA_LCB = 1.0
OOD_THRESHOLD = 5.0


def extract_v3r3_features(state_features: dict, action: str, v3_struct: dict) -> dict:
    """V3R3 features = V3R2 features + resource-aware features."""
    from run_i3_30r2_train import extract_v3r2_features

    feats = extract_v3r2_features(state_features, action, v3_struct)

    # Add resource-aware interaction features (P9: first-class resource state)
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


def load_training_data() -> tuple[list[dict], dict]:
    """Load all training data. Return records and per-source accounting.

    Returns:
        (records, accounting) where accounting is:
            {source_name: {"loaded": int, "has_v3_features": int, "skipped": int}}
    """
    records = []
    accounting = {}

    train_sources = [
        ("i3_5", REPO_ROOT / "experiments/i3_5/pinned_policy/pinned_causal_actions_v1.jsonl"),
        ("i3_28b", REPO_ROOT / "experiments/i3_28b/boundary_causal_actions_v1.jsonl"),
        ("i3_28c", REPO_ROOT / "experiments/i3_28c/strata_causal_actions_v1.jsonl"),
        ("i3_30r", REPO_ROOT / "experiments/i3_30r/causal_boundary_v2/causal_actions_v2.jsonl"),
    ]

    for source_name, path in train_sources:
        source_records = []
        if path.exists():
            with open(path) as f:
                for line in f:
                    source_records.append(json.loads(line))

        has_v3 = sum(1 for r in source_records if "v3_features" in r)
        skipped = len(source_records) - has_v3

        accounting[source_name] = {
            "loaded": len(source_records),
            "has_v3_features": has_v3,
            "skipped_no_v3": skipped,
        }

        records.extend(source_records)
        print(f"  {source_name}: {len(source_records)} records "
              f"({has_v3} with v3_features, {skipped} skipped)")

    return records, accounting


def rollout_d1_task_to_causal_records(
    task,
    executor: EvidenceExecutor,
    utility: MetareasoningUtility,
) -> list[dict]:
    """Roll out a D1 DEFER-ready task and convert to causal action records.

    For each D1 task, we generate training records by executing each
    legal action from the initial state and recording the realized utility.
    This produces the same format as the existing causal_actions_v2.jsonl.

    Args:
        task: D1 DEFER-ready task
        executor: Evidence executor
        utility: Utility function

    Returns:
        List of causal action records with state_features, v3_features,
        forced_action, and pinned_policy_utility
    """
    # Reconstruct budget from task
    parts = task.budget_profile.split("_")
    budget = ResourceBudget(
        max_executive_steps=int(parts[1]) if len(parts) > 1 else 2,
        max_reasoning_tokens=256,
        max_retrieval_calls=0,
        max_verification_calls=int(parts[2]) if len(parts) > 2 else 0,
        max_search_calls=int(parts[3]) if len(parts) > 3 else 0,
        max_elapsed_ms=10000,
    )
    resources = ResourceState(budget=budget)
    runtime = initial_evidence_runtime(task, resources)

    # Compute state features
    sf = compute_state_features(runtime, prior_actions=())

    # Compute V3 features
    evidence_dicts = []
    for ev in runtime.visible_evidence:
        evidence_dicts.append({
            "evidence_id": ev.evidence_id,
            "supports": list(ev.supports),
            "contradicts": list(ev.contradicts),
            "verification_state": ev.verification_state.value,
            "temporal_status": ev.temporal_status.value,
            "retrieved": ev.retrieved,
        })
    hyp_dicts = []
    for h in task.hypotheses:
        hyp_dicts.append({
            "hypothesis_id": h.hypothesis_id,
            "answer_action": h.answer_action.value,
        })
    v3 = compute_v3_features_canonical(evidence_dicts, hyp_dicts)

    # Legal actions from this state
    rs = runtime.resources.as_dict()
    legal_actions = []
    if rs.get("retrieval_calls_remaining", 0) > 0 and len(runtime.hidden_evidence) > 0:
        legal_actions.append("RETRIEVE")
    if rs.get("search_calls_remaining", 0) > 0:
        legal_actions.append("SEARCH_MORE")
    if rs.get("verification_calls_remaining", 0) > 0 and any(
        ev.verification_state.value == "UNVERIFIED" for ev in runtime.visible_evidence
    ):
        legal_actions.append("VERIFY")
    legal_actions.append("REASON_MORE")
    legal_actions.append("ANSWER")
    legal_actions.append("DEFER")

    records = []
    for action_str in legal_actions:
        # Execute action on a fresh runtime
        fresh_runtime = initial_evidence_runtime(task, resources)
        action = DecisionAction(action_str)

        resources_before = fresh_runtime.resources
        try:
            if action_str == "VERIFY":
                valid = valid_verify_targets(fresh_runtime)
                if valid:
                    result = executor.execute(fresh_runtime, action, target_evidence_id=valid[0])
                else:
                    continue
            else:
                result = executor.execute(fresh_runtime, action)
            resources_after = result.runtime.resources
        except Exception:
            continue

        # Compute realized utility
        realized = 0.0
        realized -= utility.action_cost(resources_before, resources_after)
        if result.terminal:
            success = bool(result.task_success)
            realized += utility.terminal_reward(action, success)

        records.append({
            "checkpoint_id": f"d1dr_{task.task_id}_{action_str}",
            "task_id": task.task_id,
            "state_features": sf,
            "v3_features": v3,
            "forced_action": action_str,
            "pinned_policy_utility": realized,
            "source": "d1_defer_ready",
        })

    return records


def main():
    print("=" * 60)
    print("I3.30R3: Train Q_V3R3 — Uncertainty Ensemble + OOD Gating")
    print("=" * 60)

    # 1. Load existing training data with per-source accounting
    print("\n1. Loading training data...")
    records, accounting = load_training_data()
    total_loaded = sum(a["loaded"] for a in accounting.values())
    total_with_v3 = sum(a["has_v3_features"] for a in accounting.values())
    print(f"   Total loaded: {total_loaded}")
    print(f"   With v3_features: {total_with_v3}")
    print(f"   Skipped (no v3_features): {total_loaded - total_with_v3}")

    # 2. Generate D1 DEFER-ready stratum
    print("\n2. Generating D1 DEFER-ready training stratum...")
    d1_tasks = generate_d1_defer_ready_tasks(seed=7777, n_per_domain=10)
    print(f"   D1 DEFER-ready tasks: {len(d1_tasks)}")

    # 3. Roll out D1 tasks to causal action records
    print("\n3. Rolling out D1 tasks to causal action records...")
    utility = MetareasoningUtility.from_file(REPO_ROOT / "configs" / "v2b_i3_1_utility_v1.json")
    executor = EvidenceExecutor()

    d1_records = []
    d1_rollout_accounting = {"tasks": len(d1_tasks), "records_generated": 0, "tasks_failed": 0}

    for task in d1_tasks:
        try:
            task_records = rollout_d1_task_to_causal_records(task, executor, utility)
            d1_records.extend(task_records)
        except Exception as e:
            d1_rollout_accounting["tasks_failed"] += 1
            print(f"   WARNING: {task.task_id} rollout failed: {e}")

    d1_rollout_accounting["records_generated"] = len(d1_records)
    print(f"   D1 records generated: {len(d1_records)}")
    print(f"   D1 tasks failed: {d1_rollout_accounting['tasks_failed']}")

    # 4. Build feature matrix with fail-closed accounting
    print("\n4. Building feature matrix...")
    feature_keys = get_feature_keys()
    print(f"   Features: {len(feature_keys)}")

    X_train = []
    y_train = []
    included_by_source = defaultdict(int)
    skipped_by_source = defaultdict(int)

    # Process historical records
    for r in records:
        source = "unknown"
        sf = r.get("state_features", {})
        action = r.get("forced_action", "ANSWER")
        if "v3_features" not in r:
            skipped_by_source["historical_no_v3"] += 1
            continue
        v3 = r["v3_features"]
        feats = extract_v3r3_features(sf, action, v3)
        X_train.append([feats.get(k, 0) for k in feature_keys])
        y_train.append(float(r.get("pinned_policy_utility", 0.0)))
        included_by_source["historical"] += 1

    # Process D1 DEFER-ready records
    for r in d1_records:
        sf = r.get("state_features", {})
        action = r.get("forced_action", "ANSWER")
        if "v3_features" not in r:
            skipped_by_source["d1_no_v3"] += 1
            continue
        v3 = r["v3_features"]
        feats = extract_v3r3_features(sf, action, v3)
        X_train.append([feats.get(k, 0) for k in feature_keys])
        y_train.append(float(r.get("pinned_policy_utility", 0.0)))
        included_by_source["d1_defer_ready"] += 1

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Fail-closed accounting
    declared_total = total_with_v3 + len(d1_records)
    actual_total = len(X_train)
    print(f"\n   DECLARED usable records: {declared_total}")
    print(f"   ACTUAL training rows:    {actual_total}")
    print(f"   Included by source: {dict(included_by_source)}")
    print(f"   Skipped by source:  {dict(skipped_by_source)}")

    assert actual_total == declared_total, (
        f"ROW ACCOUNTING MISMATCH: declared={declared_total} actual={actual_total}"
    )
    print(f"   ASSERTION PASSED: declared == actual")

    print(f"\n   X_train: {X_train.shape}")
    print(f"   y_train mean: {y_train.mean():.2f}, std: {y_train.std():.2f}")

    # 5. Train ensemble with STANDARDIZED features for OOD
    print(f"\n5. Training bootstrap ensemble (n={N_ENSEMBLE})...")
    print(f"   Using StandardScaler for OOD distance computation")

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
        use_standardization=True,  # NEW: standardize before OOD distance
    )

    # 6. Evaluate on training data
    print("\n6. Evaluating on training data...")
    mean_preds = ensemble.predict_mean(X_train)
    std_preds = ensemble.predict_std(X_train)
    train_r2 = 1 - np.mean((y_train - mean_preds) ** 2) / np.var(y_train)
    print(f"   Train R² (ensemble mean): {train_r2:.4f}")
    print(f"   Mean uncertainty (std): {std_preds.mean():.2f}")

    # Check OOD coverage with standardized distance
    in_support = ensemble.is_in_support(X_train)
    print(f"   In-support rate: {in_support.mean():.2%}")

    # 7. Save model
    print("\n7. Saving model...")
    model_path = OUTPUT_DIR / "Q_V3R3.pkl"
    ensemble.save(model_path)
    print(f"   Saved: {model_path}")

    # Save feature schema with CORRECT accounting
    schema = {
        "model": "Q_V3R3",
        "feature_keys": feature_keys,
        "n_features": len(feature_keys),
        "n_estimators": N_ENSEMBLE,
        "lambda_lcb": LAMBDA_LCB,
        "ood_threshold": OOD_THRESHOLD,
        "gbt_params": GBT_PARAMS,
        "training_rows_actual": actual_total,
        "training_rows_declared": declared_total,
        "accounting": {
            "historical_sources": accounting,
            "historical_with_v3": total_with_v3,
            "historical_skipped": total_loaded - total_with_v3,
            "d1_defer_ready": d1_rollout_accounting,
            "included_by_source": dict(included_by_source),
            "skipped_by_source": dict(skipped_by_source),
        },
        "ood_standardization": True,
        "train_r2": float(train_r2),
        "mean_uncertainty": float(std_preds.mean()),
    }
    schema_path = OUTPUT_DIR / "v3r3_feature_schema.json"
    with open(schema_path, "w") as f:
        json.dump(schema, f, indent=2)
    print(f"   Saved: {schema_path}")

    # 8. Save training report with CORRECT accounting
    report = {
        "model": "Q_V3R3",
        "status": "TRAINED",
        "lineage": "DAPH_ADAPTIVE_AUTHORITY_V3R3_DEVELOPMENT",
        "warning": "Q_V3R3 is a DEVELOPMENT candidate. Q_V3R2-A (tag v3r2-confirmed) is the confirmed historical control.",
        "improvements": [
            "Bootstrap ensemble (20 GBTs) for epistemic uncertainty",
            "LCB-based authority: force only when LCB gap >= threshold",
            "OOD support-density gating via KMeans centroids with StandardScaler",
            "D1 DEFER-ready training stratum — ACTUALLY ROLLED OUT to causal records",
            "Resource-aware interaction features",
        ],
        "training_data": {
            "historical_loaded": total_loaded,
            "historical_with_v3_features": total_with_v3,
            "historical_skipped_no_v3": total_loaded - total_with_v3,
            "d1_defer_ready_tasks": len(d1_tasks),
            "d1_defer_ready_records": len(d1_records),
            "actual_training_rows": actual_total,
            "per_source_accounting": accounting,
            "d1_rollout_accounting": d1_rollout_accounting,
        },
        "ensemble": {
            "n_estimators": N_ENSEMBLE,
            "gbt_params": GBT_PARAMS,
            "lambda_lcb": LAMBDA_LCB,
            "ood_threshold": OOD_THRESHOLD,
            "ood_standardization": True,
        },
        "performance": {
            "train_r2": float(train_r2),
            "mean_uncertainty": float(std_preds.mean()),
            "in_support_rate": float(in_support.mean()),
        },
        "feature_count": len(feature_keys),
        "heldout_evaluation": "NOT YET RUN — required before live use",
        "v3r2_comparison": {
            "v3r2_features": "V1 + V3 canonical + interactions",
            "v3r3_features": "V3R2 + resource-aware interactions",
            "v3r2_model": "single GBT",
            "v3r3_model": "bootstrap ensemble of 20 GBTs",
            "v3r2_uncertainty": "none",
            "v3r3_uncertainty": "ensemble std + LCB",
            "v3r2_ood_gating": "none",
            "v3r3_ood_gating": "KMeans support density with StandardScaler",
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
    print(f"  Training rows: {actual_total} (declared={declared_total})")
    print(f"  HELDOUT EVALUATION: NOT YET RUN — required before live use")
    print("=" * 60)


if __name__ == "__main__":
    main()
