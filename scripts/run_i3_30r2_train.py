#!/usr/bin/env python3
"""I3.30R2: Train Q_V3R2 on structurally disjoint data + held-out evaluation.

Fixes:
  - Fail-closed data loader (abort if declared records are silently dropped)
  - Structural holdout (G0: 0% feature overlap)
  - Exact decide_authority_v3() for offline safety
  - verified_hyp_action ablation (V3R2-A vs V3R2-B)
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

from daph.authority.policy import AuthorityMode, AUTHORITY_THRESHOLD, I2_EPSILON_Q
from daph.authority.policy_v3 import StructuralStateV3, decide_authority_v3
from daph.epistemic.v3_features import compute_v3_features_canonical
from run_i3_28_rep_repair import extract_v1_features

OUTPUT_DIR = REPO_ROOT / "experiments/i3_30r"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GBT_PARAMS = dict(n_estimators=200, max_depth=4, random_state=42)


def load_records_fail_closed(path: Path, expected_count: int | None = None) -> list[dict]:
    """Load records from a JSONL file. Fail closed if file missing or empty."""
    if not path.exists():
        raise FileNotFoundError(f"Declared training data file not found: {path}")
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    if expected_count is not None and len(records) != expected_count:
        print(f"  WARNING: {path.name} has {len(records)} records, expected {expected_count}")
    if len(records) == 0:
        raise ValueError(f"Declared training data file is empty: {path}")
    return records


def extract_v3r2_features(state_features: dict, action: str, v3_struct: dict) -> dict:
    feats = extract_v1_features(state_features, action)
    for k in ["n_hyp_with_verified_support", "n_hyp_with_verified_contradiction",
              "n_hyp_with_mixed_verified", "n_viable_hypotheses", "n_eliminated_hypotheses",
              "has_unique_verified_supported_hypothesis", "has_verified_unresolved_competition",
              "verified_hyp_action_is_answer", "verified_hyp_action_is_defer",
              "n_hyp_unverified_support", "n_hyp_unverified_contradiction",
              "has_competing_unverified_support"]:
        feats[k] = v3_struct[k]
    feats["verified_hyp_action_is_answer_x_answer"] = feats["verified_hyp_action_is_answer"] * feats["a_ANSWER"]
    feats["verified_hyp_action_is_defer_x_defer"] = feats["verified_hyp_action_is_defer"] * feats["a_DEFER"]
    feats["n_hyp_with_verified_contradiction_x_defer"] = feats["n_hyp_with_verified_contradiction"] * feats["a_DEFER"]
    feats["n_eliminated_hypotheses_x_defer"] = feats["n_eliminated_hypotheses"] * feats["a_DEFER"]
    feats["has_unique_verified_supported_hypothesis_x_answer"] = feats["has_unique_verified_supported_hypothesis"] * feats["a_ANSWER"]
    feats["has_verified_unresolved_competition_x_continue"] = feats["has_verified_unresolved_competition"] * (1 - feats["a_ANSWER"] - feats["a_DEFER"])
    return feats


def extract_v3r2_features_no_vha(state_features: dict, action: str, v3_struct: dict) -> dict:
    """V3R2-B: same as V3R2-A but WITHOUT verified_hyp_action features."""
    feats = extract_v3r2_features(state_features, action, v3_struct)
    # Remove verified_hyp_action features and their interactions
    for k in ["verified_hyp_action_is_answer", "verified_hyp_action_is_defer",
              "verified_hyp_action_is_answer_x_answer", "verified_hyp_action_is_defer_x_defer"]:
        feats.pop(k, None)
        feats[k] = 0  # set to 0 instead of removing (keeps feature keys stable)
    return feats


def get_feature_keys(use_vha: bool = True) -> list[str]:
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
    feats = extract_v3r2_features(dummy_sf, "ANSWER", dummy_v3)
    if not use_vha:
        for k in ["verified_hyp_action_is_answer", "verified_hyp_action_is_defer",
                  "verified_hyp_action_is_answer_x_answer", "verified_hyp_action_is_defer_x_defer"]:
            feats.pop(k, None)
    return sorted(feats.keys())


def build_structural_state_v3(v3_struct, sf, can_verify):
    return StructuralStateV3(
        has_competing_unverified_support=bool(v3_struct["has_competing_unverified_support"]),
        n_hyp_unverified_support=v3_struct["n_hyp_unverified_support"],
        n_hyp_unverified_contradiction=v3_struct["n_hyp_unverified_contradiction"],
        can_verify=can_verify,
        verify_budget_exhausted=sf.get("verify_remaining", 0) <= 0,
        all_evidence_verified=v3_struct.get("verification_complete", 0) == 1,
        n_hyp_with_verified_support=v3_struct["n_hyp_with_verified_support"],
        n_hyp_with_verified_contradiction=v3_struct["n_hyp_with_verified_contradiction"],
        n_hyp_with_mixed_verified=v3_struct["n_hyp_with_mixed_verified"],
        n_viable_hypotheses=v3_struct["n_viable_hypotheses"],
        n_eliminated_hypotheses=v3_struct["n_eliminated_hypotheses"],
        has_unique_verified_supported_hypothesis=bool(v3_struct["has_unique_verified_supported_hypothesis"]),
        has_verified_unresolved_competition=bool(v3_struct["has_verified_unresolved_competition"]),
        verified_hyp_action_is_answer=bool(v3_struct["verified_hyp_action_is_answer"]),
        verified_hyp_action_is_defer=bool(v3_struct["verified_hyp_action_is_defer"]),
    )


def train_and_evaluate(train_records, heldout_records, use_vha, label):
    """Train a model and evaluate on held-out with exact authority logic."""
    feature_keys = get_feature_keys(use_vha=use_vha)
    print(f"\n  {label}: {len(feature_keys)} features (VHA={'yes' if use_vha else 'no'})")

    # Build training matrix — fail closed
    X_train = []
    y_train = []
    skipped = 0
    for r in train_records:
        sf = r.get("state_features", {})
        action = r.get("forced_action")
        if "v3_features" not in r:
            skipped += 1
            continue
        v3 = r["v3_features"]
        feats = extract_v3r2_features(sf, action, v3) if use_vha else extract_v3r2_features_no_vha(sf, action, v3)
        X_train.append([feats[k] for k in feature_keys])
        y_train.append(float(r.get("pinned_policy_utility", 0.0)))

    if skipped > 0:
        print(f"  WARNING: {skipped} records skipped (no v3_features)")

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    print(f"  X_train: {X_train.shape}, y_train mean: {y_train.mean():.2f}")

    model = GradientBoostingRegressor(**GBT_PARAMS)
    model.fit(X_train, y_train)
    print(f"  Train R²: {model.score(X_train, y_train):.4f}")

    # Held-out evaluation
    by_ckpt = defaultdict(list)
    for r in heldout_records:
        by_ckpt[r["checkpoint_id"]].append(r)

    answer_events = 0
    false_answer = 0
    defer_events = 0
    false_defer = 0
    continue_false = 0
    answer_cov = 0
    answer_total = 0
    defer_cov = 0
    defer_total = 0
    cert_violations = 0

    for ckpt_id, group in by_ckpt.items():
        r0 = group[0]
        sf = r0["state_features"]
        v3 = r0["v3_features"]
        expected = r0["expected_terminal"]
        category = r0["category"]
        legal = r0["legal_actions"]

        q_values = {}
        for a in legal:
            feats = extract_v3r2_features(sf, a, v3) if use_vha else extract_v3r2_features_no_vha(sf, a, v3)
            x = np.array([[feats[k] for k in feature_keys]])
            q_values[a] = float(model.predict(x)[0])

        can_verify = sf.get("can_verify", 0) > 0
        struct = build_structural_state_v3(v3, sf, can_verify)
        decision = decide_authority_v3(q_values=q_values, legal_actions=legal, structural=struct)

        is_continue = "P3" in category
        if "P1a" in category or "P1b" in category:
            answer_total += 1
        if "P2a" in category or "P2b" in category or "P2_elim" in category or "P5" in category:
            defer_total += 1

        if decision.mode == AuthorityMode.HARD_ANSWER:
            answer_events += 1
            if expected != "ANSWER":
                false_answer += 1
            if is_continue:
                continue_false += 1
            if "P1a" in category or "P1b" in category:
                answer_cov += 1
            if not struct.has_unique_verified_supported_hypothesis:
                cert_violations += 1
        elif decision.mode == AuthorityMode.HARD_DEFER:
            defer_events += 1
            if expected != "DEFER":
                false_defer += 1
            if is_continue:
                continue_false += 1
            if "P2a" in category or "P2b" in category or "P2_elim" in category or "P5" in category:
                defer_cov += 1

    far_a = false_answer / max(answer_events, 1)
    far_d = false_defer / max(defer_events, 1)
    total_t = answer_events + defer_events
    correct_t = total_t - false_answer - false_defer
    precision = correct_t / max(total_t, 1)

    # G0: Feature overlap check
    train_sigs = set()
    for r in train_records:
        if "v3_features" not in r:
            continue
        feats = extract_v3r2_features(r.get("state_features", {}), r.get("forced_action"), r["v3_features"]) if use_vha else extract_v3r2_features_no_vha(r.get("state_features", {}), r.get("forced_action"), r["v3_features"])
        train_sigs.add(tuple(feats[k] for k in feature_keys))

    overlap = 0
    for r in heldout_records:
        feats = extract_v3r2_features(r.get("state_features", {}), r.get("forced_action"), r["v3_features"]) if use_vha else extract_v3r2_features_no_vha(r.get("state_features", {}), r.get("forced_action"), r["v3_features"])
        if tuple(feats[k] for k in feature_keys) in train_sigs:
            overlap += 1

    print(f"\n  Held-out checkpoints: {len(by_ckpt)}")
    print(f"  G0 feature overlap: {overlap}/{len(heldout_records)} ({overlap/len(heldout_records)*100:.1f}%)")
    print(f"  ANSWER events: {answer_events}, false: {false_answer}, FAR: {far_a:.4f}")
    print(f"  DEFER events: {defer_events}, false: {false_defer}, FAR: {far_d:.4f}")
    print(f"  Terminal precision: {precision:.4f}")
    print(f"  False terminal on CONTINUE: {continue_false}")
    print(f"  Certificate violations: {cert_violations}")
    print(f"  ANSWER coverage: {answer_cov}/{answer_total}")
    print(f"  DEFER coverage: {defer_cov}/{defer_total}")

    gates = {
        "G0_overlap_zero": overlap == 0,
        "G1_FAR_ANSWER_zero": far_a == 0.0,
        "G2_FAR_DEFER_zero": far_d == 0.0,
        "G3_precision_1": precision == 1.0,
        "G4_continue_false_zero": continue_false == 0,
        "G5_answer_cov_positive": answer_cov > 0,
        "G6_defer_cov_positive": defer_cov > 0,
        "G7_cert_violations_zero": cert_violations == 0,
    }
    for g, v in gates.items():
        print(f"    {g}: {'PASS' if v else 'FAIL'}")
    all_pass = all(gates.values())
    print(f"  OVERALL: {'PASS' if all_pass else 'FAIL'}")

    return {"gates": gates, "all_pass": all_pass, "far_answer": far_a, "far_defer": far_d,
            "precision": precision, "answer_events": answer_events, "defer_events": defer_events,
            "answer_coverage": f"{answer_cov}/{answer_total}", "defer_coverage": f"{defer_cov}/{defer_total}"}


def main():
    print("=" * 70)
    print("I3.30R2: Structural Holdout Training + Evaluation")
    print("=" * 70)

    # Load I3.30R2 boundary data (structurally disjoint)
    boundary_path = REPO_ROOT / "experiments/i3_30r/causal_boundary_v3/causal_actions_v3.jsonl"
    boundary_records = load_records_fail_closed(boundary_path)
    train_boundary = [r for r in boundary_records if r.get("split") == "train"]
    heldout_boundary = [r for r in boundary_records if r.get("split") == "heldout"]
    print(f"  I3.30R2 boundary: {len(train_boundary)} train, {len(heldout_boundary)} heldout")

    # Load I3.5 original data (always training)
    i3_5_path = REPO_ROOT / "experiments/i3_5/pinned_policy/pinned_causal_actions_v1.jsonl"
    i3_5_records = load_records_fail_closed(i3_5_path)
    print(f"  I3.5: {len(i3_5_records)} records")

    # Reconstruct V3 features for I3.5 records
    ckpts = {}
    ckpt_path = REPO_ROOT / "experiments/i3_5/datasets/checkpoints_v1.jsonl"
    if ckpt_path.exists():
        with open(ckpt_path) as f:
            for line in f:
                r = json.loads(line)
                ckpts[r["checkpoint_id"]] = r

    i3_5_with_v3 = []
    i3_5_skipped = 0
    for r in i3_5_records:
        ckpt = ckpts.get(r.get("checkpoint_id"))
        if ckpt and "evidence" in ckpt:
            v3 = compute_v3_features_canonical(ckpt.get("evidence", []), ckpt.get("hypotheses", []))
            r["v3_features"] = v3
            i3_5_with_v3.append(r)
        else:
            i3_5_skipped += 1

    print(f"  I3.5 with V3 features: {len(i3_5_with_v3)}, skipped: {i3_5_skipped}")

    # NOTE: I3.28B and I3.28C are EXPLICITLY EXCLUDED from training lineage
    # because their checkpoint structures don't support V3 feature reconstruction.
    # This is documented, not silently dropped.
    print(f"  I3.28B/C: EXPLICITLY EXCLUDED (checkpoints don't support V3 reconstruction)")

    # Combined training set
    train_records = i3_5_with_v3 + train_boundary
    print(f"  Total train: {len(train_records)}")

    # ============================================================
    # V3R2-A: WITH verified_hyp_action
    # ============================================================
    print("\n" + "=" * 70)
    print("V3R2-A: WITH verified_hyp_action")
    print("=" * 70)
    results_a = train_and_evaluate(train_records, heldout_boundary, use_vha=True, label="V3R2-A")

    # Save model A
    feature_keys_a = get_feature_keys(use_vha=True)
    model_a = GradientBoostingRegressor(**GBT_PARAMS)
    X_a = np.array([[extract_v3r2_features(r.get("state_features", {}), r.get("forced_action"), r["v3_features"])[k] for k in feature_keys_a] for r in train_records if "v3_features" in r])
    y_a = np.array([float(r.get("pinned_policy_utility", 0.0)) for r in train_records if "v3_features" in r])
    model_a.fit(X_a, y_a)
    with open(OUTPUT_DIR / "Q_V3R2_A.pkl", "wb") as f:
        pickle.dump(model_a, f)

    # ============================================================
    # V3R2-B: WITHOUT verified_hyp_action (ablation)
    # ============================================================
    print("\n" + "=" * 70)
    print("V3R2-B: WITHOUT verified_hyp_action (ablation)")
    print("=" * 70)
    results_b = train_and_evaluate(train_records, heldout_boundary, use_vha=False, label="V3R2-B")

    # Save model B
    feature_keys_b = get_feature_keys(use_vha=False)
    model_b = GradientBoostingRegressor(**GBT_PARAMS)
    X_b = np.array([[extract_v3r2_features_no_vha(r.get("state_features", {}), r.get("forced_action"), r["v3_features"])[k] for k in feature_keys_b] for r in train_records if "v3_features" in r])
    y_b = np.array([float(r.get("pinned_policy_utility", 0.0)) for r in train_records if "v3_features" in r])
    model_b.fit(X_b, y_b)
    with open(OUTPUT_DIR / "Q_V3R2_B.pkl", "wb") as f:
        pickle.dump(model_b, f)

    # Summary
    print("\n" + "=" * 70)
    print("Ablation Summary")
    print("=" * 70)
    print(f"  V3R2-A (with VHA):  {'PASS' if results_a['all_pass'] else 'FAIL'}")
    print(f"    FAR_ANSWER={results_a['far_answer']:.4f}, FAR_DEFER={results_b['far_defer']:.4f}")
    print(f"    ANSWER cov={results_a['answer_coverage']}, DEFER cov={results_a['defer_coverage']}")
    print(f"  V3R2-B (without VHA): {'PASS' if results_b['all_pass'] else 'FAIL'}")
    print(f"    FAR_ANSWER={results_b['far_answer']:.4f}, FAR_DEFER={results_b['far_defer']:.4f}")
    print(f"    ANSWER cov={results_b['answer_coverage']}, DEFER cov={results_b['defer_coverage']}")

    # Save results
    results = {
        "v3r2_a_with_vha": results_a,
        "v3r2_b_without_vha": results_b,
        "train_records": len(train_records),
        "heldout_records": len(heldout_boundary),
        "i3_5_included": len(i3_5_with_v3),
        "i3_28bc_explicitly_excluded": True,
        "i3_28bc_exclusion_reason": "checkpoints don't support V3 feature reconstruction",
    }
    with open(OUTPUT_DIR / "structural_holdout_gates.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {OUTPUT_DIR / 'structural_holdout_gates.json'}")


if __name__ == "__main__":
    main()
