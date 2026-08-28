#!/usr/bin/env python3
"""I3.30R: Train Q_V3R2 and evaluate on held-out with exact authority logic.

Uses:
  - Corrected I3.30R boundary data (canonical semantics)
  - Original I3.5/I3.28B/I3.28C causal data (train only)
  - Held-out split by domain family (never in training)
  - Exact decide_authority_v3() for offline safety evaluation
  - Same frozen GBT hyperparameters

Gates (held-out):
  G1: FAR_ANSWER = 0
  G2: FAR_DEFER = 0
  G3: TerminalAuthorityPrecision = 1.0
  G4: false terminal authority on CONTINUE states = 0
  G5: ANSWER coverage > 0
  G6: DEFER coverage > 0
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
from daph.authority.policy_v3 import (
    StructuralStateV3, decide_authority_v3,
)
from daph.epistemic.v3_features import compute_v3_features_canonical
from run_i3_28_rep_repair import extract_v1_features

OUTPUT_DIR = REPO_ROOT / "experiments/i3_30r"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GBT_PARAMS = dict(n_estimators=200, max_depth=4, random_state=42)

# Training data: original + I3.30R corrected
TRAIN_DATA_PATHS = [
    REPO_ROOT / "experiments/i3_5/pinned_policy/pinned_causal_actions_v1.jsonl",
    REPO_ROOT / "experiments/i3_28b/boundary_causal_actions_v1.jsonl",
    REPO_ROOT / "experiments/i3_28c/strata_causal_actions_v1.jsonl",
    REPO_ROOT / "experiments/i3_30r/causal_boundary_v2/causal_actions_v2.jsonl",
]

CKPT_PATH = REPO_ROOT / "experiments/i3_5/datasets/checkpoints_v1.jsonl"


def load_checkpoints():
    ckpts = {}
    if CKPT_PATH.exists():
        with open(CKPT_PATH) as f:
            for line in f:
                r = json.loads(line)
                ckpts[r["checkpoint_id"]] = r
    return ckpts


def extract_v3r2_features(state_features: dict, action: str, v3_struct: dict) -> dict:
    """V3R2 feature extraction: V1 + V3 canonical + interactions."""
    feats = extract_v1_features(state_features, action)

    # V3 canonical features (from topology, correct FALSIFIED semantics)
    feats["n_hyp_with_verified_support"] = v3_struct["n_hyp_with_verified_support"]
    feats["n_hyp_with_verified_contradiction"] = v3_struct["n_hyp_with_verified_contradiction"]
    feats["n_hyp_with_mixed_verified"] = v3_struct["n_hyp_with_mixed_verified"]
    feats["n_viable_hypotheses"] = v3_struct["n_viable_hypotheses"]
    feats["n_eliminated_hypotheses"] = v3_struct["n_eliminated_hypotheses"]
    feats["has_unique_verified_supported_hypothesis"] = v3_struct["has_unique_verified_supported_hypothesis"]
    feats["has_verified_unresolved_competition"] = v3_struct["has_verified_unresolved_competition"]
    feats["verified_hyp_action_is_answer"] = v3_struct["verified_hyp_action_is_answer"]
    feats["verified_hyp_action_is_defer"] = v3_struct["verified_hyp_action_is_defer"]

    # V2R backward-compatible features
    feats["n_hyp_unverified_support"] = v3_struct["n_hyp_unverified_support"]
    feats["n_hyp_unverified_contradiction"] = v3_struct["n_hyp_unverified_contradiction"]
    feats["has_competing_unverified_support"] = v3_struct["has_competing_unverified_support"]

    # Interactions
    feats["verified_hyp_action_is_answer_x_answer"] = feats["verified_hyp_action_is_answer"] * feats["a_ANSWER"]
    feats["verified_hyp_action_is_defer_x_defer"] = feats["verified_hyp_action_is_defer"] * feats["a_DEFER"]
    feats["n_hyp_with_verified_contradiction_x_defer"] = feats["n_hyp_with_verified_contradiction"] * feats["a_DEFER"]
    feats["n_eliminated_hypotheses_x_defer"] = feats["n_eliminated_hypotheses"] * feats["a_DEFER"]
    feats["has_unique_verified_supported_hypothesis_x_answer"] = feats["has_unique_verified_supported_hypothesis"] * feats["a_ANSWER"]
    feats["has_verified_unresolved_competition_x_continue"] = feats["has_verified_unresolved_competition"] * (1 - feats["a_ANSWER"] - feats["a_DEFER"])

    return feats


def get_v3r2_feature_keys() -> list[str]:
    dummy_sf = {k: 0 for k in [
        "n_live", "n_eliminated", "n_untested", "n_total_hypotheses",
        "n_visible_evidence", "n_verified", "n_supporting", "n_contradicting",
        "n_stale", "retrieval_remaining", "search_remaining", "verify_remaining",
        "steps_remaining", "can_retrieve", "can_search", "can_verify",
        "searched", "reasoning_complete", "same_action_run_length",
        "retrieval_count", "search_count", "verify_count",
    ]}
    dummy_v3 = {
        "n_hyp_with_verified_support": 0, "n_hyp_with_verified_contradiction": 0,
        "n_hyp_with_mixed_verified": 0, "n_viable_hypotheses": 0,
        "n_eliminated_hypotheses": 0,
        "has_unique_verified_supported_hypothesis": 0,
        "has_verified_unresolved_competition": 0,
        "verified_hyp_action_is_answer": 0, "verified_hyp_action_is_defer": 0,
        "n_hyp_unverified_support": 0, "n_hyp_unverified_contradiction": 0,
        "has_competing_unverified_support": 0,
    }
    feats = extract_v3r2_features(dummy_sf, "ANSWER", dummy_v3)
    return sorted(feats.keys())


def build_structural_state_v3(v3_struct: dict, sf: dict, can_verify: bool) -> StructuralStateV3:
    """Build StructuralStateV3 from V3 features and state features."""
    verify_remaining = sf.get("verify_remaining", 0)
    verify_budget_exhausted = verify_remaining <= 0
    all_evidence_verified = v3_struct.get("verification_complete", 0) == 1

    return StructuralStateV3(
        has_competing_unverified_support=bool(v3_struct["has_competing_unverified_support"]),
        n_hyp_unverified_support=v3_struct["n_hyp_unverified_support"],
        n_hyp_unverified_contradiction=v3_struct["n_hyp_unverified_contradiction"],
        can_verify=can_verify,
        verify_budget_exhausted=verify_budget_exhausted,
        all_evidence_verified=all_evidence_verified,
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


def main():
    print("=" * 70)
    print("I3.30R: Train Q_V3R2 + Held-Out Offline Gates")
    print("=" * 70)

    ckpts = load_checkpoints()
    print(f"  Checkpoints: {len(ckpts)}")

    # Load training records (exclude held-out)
    train_records = []
    for path in TRAIN_DATA_PATHS:
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if r.get("split", "train") == "train":
                    train_records.append(r)

    # Load held-out records
    heldout_records = []
    heldout_path = REPO_ROOT / "experiments/i3_30r/causal_boundary_v2/causal_actions_v2.jsonl"
    with open(heldout_path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("split") == "heldout":
                heldout_records.append(r)

    print(f"  Train records: {len(train_records)}")
    print(f"  Held-out records: {len(heldout_records)}")

    feature_keys = get_v3r2_feature_keys()
    print(f"  V3R2 feature count: {len(feature_keys)}")

    # Build training matrix
    X_train = []
    y_train = []
    for r in train_records:
        sf = r.get("state_features", {})
        action = r.get("forced_action")
        utility = r.get("pinned_policy_utility", 0.0)

        if "v3_features" in r:
            v3 = r["v3_features"]
        else:
            ckpt = ckpts.get(r.get("checkpoint_id"))
            if ckpt and "evidence" in ckpt:
                v3 = compute_v3_features_canonical(
                    ckpt.get("evidence", []), ckpt.get("hypotheses", []))
            else:
                continue

        feats = extract_v3r2_features(sf, action, v3)
        X_train.append([feats[k] for k in feature_keys])
        y_train.append(float(utility))

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    print(f"  X_train shape: {X_train.shape}")
    print(f"  y_train mean: {y_train.mean():.2f}, std: {y_train.std():.2f}")

    # Train
    print("\n  Training Q_V3R2...")
    model = GradientBoostingRegressor(**GBT_PARAMS)
    model.fit(X_train, y_train)
    print(f"  Train R²: {model.score(X_train, y_train):.4f}")

    # Save model
    model_path = OUTPUT_DIR / "Q_V3R2.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    schema_path = OUTPUT_DIR / "v3r2_feature_schema.json"
    with open(schema_path, "w") as f:
        json.dump({"featurekeys": feature_keys, "count": len(feature_keys)}, f, indent=2)
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    print(f"  Model SHA-256: {model_hash}")

    # ============================================================
    # Held-out evaluation with EXACT authority logic
    # ============================================================
    print("\n" + "=" * 70)
    print("Held-Out Evaluation (exact decide_authority_v3)")
    print("=" * 70)

    # Group held-out by checkpoint (one Q prediction per checkpoint, then check authority)
    by_ckpt = defaultdict(list)
    for r in heldout_records:
        by_ckpt[r["checkpoint_id"]].append(r)

    answer_authority_events = 0
    false_answer_authority = 0
    defer_authority_events = 0
    false_defer_authority = 0
    continue_false_terminal = 0  # false terminal authority on CONTINUE states
    answer_coverage = 0  # P1a/P1b where ANSWER authority fires
    answer_coverage_total = 0
    defer_coverage = 0  # P2a/P2b/P2_elim/P5 where DEFER authority fires
    defer_coverage_total = 0
    certificate_violations = 0

    for ckpt_id, group in by_ckpt.items():
        r0 = group[0]
        sf = r0["state_features"]
        v3 = r0["v3_features"]
        expected = r0["expected_terminal"]
        category = r0["category"]
        legal = r0["legal_actions"]

        # Predict Q for all legal actions
        q_values = {}
        for a in legal:
            feats = extract_v3r2_features(sf, a, v3)
            x = np.array([[feats[k] for k in feature_keys]])
            q_values[a] = float(model.predict(x)[0])

        # Build structural state for authority
        can_verify = sf.get("can_verify", 0) > 0
        struct_v3 = build_structural_state_v3(v3, sf, can_verify)

        # Use EXACT authority policy
        decision = decide_authority_v3(
            q_values=q_values,
            legal_actions=legal,
            structural=struct_v3,
        )

        is_answer_correct_state = expected == "ANSWER"
        is_defer_correct_state = expected == "DEFER"
        is_continue_state = "P3" in category  # CONTINUE-correct states

        # Track coverage
        if "P1a" in category or "P1b" in category:
            answer_coverage_total += 1
        if "P2a" in category or "P2b" in category or "P2_elim" in category or "P5" in category:
            defer_coverage_total += 1

        if decision.mode == AuthorityMode.HARD_ANSWER:
            answer_authority_events += 1
            if not is_answer_correct_state:
                false_answer_authority += 1
            if is_continue_state:
                continue_false_terminal += 1
            if "P1a" in category or "P1b" in category:
                answer_coverage += 1
            # Check certificate
            if not struct_v3.has_unique_verified_supported_hypothesis:
                certificate_violations += 1

        elif decision.mode == AuthorityMode.HARD_DEFER:
            defer_authority_events += 1
            if not is_defer_correct_state:
                false_defer_authority += 1
            if is_continue_state:
                continue_false_terminal += 1
            if "P2a" in category or "P2b" in category or "P2_elim" in category or "P5" in category:
                defer_coverage += 1
            # Check certificate
            if not (struct_v3.has_unique_verified_supported_hypothesis or
                    struct_v3.n_eliminated_hypotheses > 0 or
                    (struct_v3.verify_budget_exhausted and
                     struct_v3.n_hyp_with_verified_support == 0)):
                certificate_violations += 1

    # Compute metrics
    far_answer = false_answer_authority / max(answer_authority_events, 1)
    far_defer = false_defer_authority / max(defer_authority_events, 1)
    total_terminal = answer_authority_events + defer_authority_events
    correct_terminal = total_terminal - false_answer_authority - false_defer_authority
    precision = correct_terminal / max(total_terminal, 1)

    # Print results
    print(f"\n  Held-out checkpoints: {len(by_ckpt)}")
    print(f"  ANSWER authority events: {answer_authority_events}")
    print(f"  False ANSWER authority: {false_answer_authority}")
    print(f"  FAR_ANSWER: {far_answer:.4f}")
    print(f"  DEFER authority events: {defer_authority_events}")
    print(f"  False DEFER authority: {false_defer_authority}")
    print(f"  FAR_DEFER: {far_defer:.4f}")
    print(f"  Terminal authority precision: {precision:.4f}")
    print(f"  False terminal on CONTINUE states: {continue_false_terminal}")
    print(f"  Certificate violations: {certificate_violations}")
    print(f"  ANSWER coverage: {answer_coverage}/{answer_coverage_total}")
    print(f"  DEFER coverage: {defer_coverage}/{defer_coverage_total}")

    # Confidence intervals (rule of three for zero failures)
    n_answer = answer_authority_events
    n_defer = defer_authority_events
    ci_answer = 3.0 / max(n_answer, 1) if false_answer_authority == 0 else 1.0
    ci_defer = 3.0 / max(n_defer, 1) if false_defer_authority == 0 else 1.0
    print(f"\n  95% upper bound (rule of 3):")
    print(f"    FAR_ANSWER < {ci_answer:.4f}" if false_answer_authority == 0 else f"    FAR_ANSWER = {far_answer:.4f}")
    print(f"    FAR_DEFER < {ci_defer:.4f}" if false_defer_authority == 0 else f"    FAR_DEFER = {far_defer:.4f}")

    # Gates
    gates = {
        "G1_FAR_ANSWER_zero": far_answer == 0.0,
        "G2_FAR_DEFER_zero": far_defer == 0.0,
        "G3_TerminalAuthorityPrecision_1": precision == 1.0,
        "G4_false_terminal_on_CONTINUE_zero": continue_false_terminal == 0,
        "G5_ANSWER_coverage_positive": answer_coverage > 0,
        "G6_DEFER_coverage_positive": defer_coverage > 0,
        "G7_certificate_violations_zero": certificate_violations == 0,
    }

    print(f"\n  Gates:")
    for g, v in gates.items():
        print(f"    {g}: {'PASS' if v else 'FAIL'}")

    all_pass = all(gates.values())
    print(f"\n  OVERALL: {'PASS' if all_pass else 'FAIL'}")

    # Save
    results = {
        "model_sha256": model_hash,
        "feature_count": len(feature_keys),
        "train_records": len(train_records),
        "heldout_records": len(heldout_records),
        "heldout_checkpoints": len(by_ckpt),
        "gates": {k: bool(v) for k, v in gates.items()},
        "all_pass": bool(all_pass),
        "far_answer": far_answer,
        "far_defer": far_defer,
        "terminal_authority_precision": precision,
        "false_terminal_on_continue": continue_false_terminal,
        "certificate_violations": certificate_violations,
        "answer_coverage": f"{answer_coverage}/{answer_coverage_total}",
        "defer_coverage": f"{defer_coverage}/{defer_coverage_total}",
        "ci_upper_far_answer": ci_answer if false_answer_authority == 0 else None,
        "ci_upper_far_defer": ci_defer if false_defer_authority == 0 else None,
    }
    with open(OUTPUT_DIR / "heldout_gates.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {OUTPUT_DIR / 'heldout_gates.json'}")


if __name__ == "__main__":
    main()
