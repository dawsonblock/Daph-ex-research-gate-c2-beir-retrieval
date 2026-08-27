#!/usr/bin/env python3
"""I3.28: Authority-State Sufficiency — Representation Repair for DEFER.

This experiment tests whether adding 3 minimal structural features to the Q-state
representation enables safe DEFER authority.

Design:
  - Q_V1:  old features, same causal dataset, same GBT, same hyperparameters (control)
  - Q_V2R: old features + 3 structural features, same everything else

The 3 new features (Q_STATE_SCHEMA_V2):
  n_hyp_unverified_support:         # hypotheses with >=1 unverified supporting evidence
  n_hyp_unverified_contradiction:   # hypotheses with >=1 unverified contradicting evidence
  has_competing_unverified_support: binary: n_hyp_unverified_support > 1

These are OBSERVABLE — they use only visible evidence supports/contradicts fields.
They do NOT use verify_result (oracle), do NOT inspect hidden evidence, and are
NOT changed by future outcomes.

Pipeline:
  1. Reconstruct 3 new features for all 1056 causal records from checkpoint evidence.
  2. Run 4 leakage tests per feature.
  3. Train Q_V1 (control) and Q_V2R on same data, same GBT, same hyperparameters.
  4. Offline separation audit at problematic defer/contradiction states.
     Gate: min Q_V2R(DEFER|defer-correct) - max Q_V2R(DEFER|contra) > 5
  5. Preservation check on 220 causal checkpoints:
     - mean regret no worse than V1
     - near-optimal coverage no worse
     - ANSWER authority cases remain correctly separated
     - no new false high-confidence DEFER regions
     - clear-choice accuracy does not regress materially
  6. Offline authority test: would_force_ANSWER, would_force_DEFER, correct causal best.
     Require FalseAuthorityRate ~ 0 for both.

Output:
  experiments/i3_28/Q_STATE_SCHEMA_V2.json
  experiments/i3_28/Q_V1_control.pkl
  experiments/i3_28/Q_V2R_repaired.pkl
  experiments/i3_28/leakage_tests.json
  experiments/i3_28/separation_audit.json
  experiments/i3_28/preservation_check.json
  experiments/i3_28/offline_authority_test.json
  experiments/i3_28/full_results.json
"""
from __future__ import annotations

import hashlib
import inspect
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

# ============================================================
# Constants
# ============================================================
OUTPUT_DIR = REPO_ROOT / "experiments/i3_28"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CAUSAL_DATA_PATH = REPO_ROOT / "experiments/i3_5/pinned_policy/pinned_causal_actions_v1.jsonl"
CHECKPOINTS_PATH = REPO_ROOT / "experiments/i3_5/datasets/checkpoints_v1.jsonl"
V1_FEATURE_SCHEMA_PATH = REPO_ROOT / "experiments/i3_5/pinned_policy/frozen_estimators/feature_schema.json"
V1_QCAUSAL_PATH = REPO_ROOT / "experiments/i3_5/pinned_policy/frozen_estimators/QCAUSAL_gbt.pkl"

GBT_PARAMS = dict(n_estimators=200, max_depth=4, random_state=42)
AUTHORITY_THRESHOLD = 5.0
SEPARATION_MARGIN = 5.0  # preregistered

NEW_FEATURES = [
    "n_hyp_unverified_support",
    "n_hyp_unverified_contradiction",
    "has_competing_unverified_support",
]

# ============================================================
# Feature extraction
# ============================================================

def extract_v1_features(state_features: dict, action: str) -> dict:
    """V1 feature extraction (same as freeze_i3_5_estimators.py)."""
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


def compute_structural_features(evidence_items: list[dict]) -> dict:
    """Compute the 3 new structural features from checkpoint evidence.

    These features are OBSERVABLE:
      - Uses only visible evidence (retrieved=True)
      - Uses only supports/contradicts fields
      - Does NOT use verify_result (oracle)
      - Does NOT inspect hidden evidence
      - Is NOT changed by future outcomes
    """
    hyps_with_unverified_support = set()
    hyps_with_unverified_contradiction = set()

    for ev in evidence_items:
        # Only visible (retrieved) evidence
        if not ev.get("retrieved", False):
            continue
        # Only unverified evidence
        if ev.get("verification_state") != "UNVERIFIED":
            continue
        for h_id in ev.get("supports", []):
            hyps_with_unverified_support.add(h_id)
        for h_id in ev.get("contradicts", []):
            hyps_with_unverified_contradiction.add(h_id)

    return {
        "n_hyp_unverified_support": len(hyps_with_unverified_support),
        "n_hyp_unverified_contradiction": len(hyps_with_unverified_contradiction),
        "has_competing_unverified_support": int(len(hyps_with_unverified_support) > 1),
    }


def extract_v2r_features(state_features: dict, action: str, structural: dict) -> dict:
    """V2R feature extraction: V1 features + 3 structural features + interactions."""
    feats = extract_v1_features(state_features, action)
    feats["n_hyp_unverified_support"] = structural["n_hyp_unverified_support"]
    feats["n_hyp_unverified_contradiction"] = structural["n_hyp_unverified_contradiction"]
    feats["has_competing_unverified_support"] = structural["has_competing_unverified_support"]
    # Interactions with DEFER (the action we're trying to make safe)
    feats["has_competing_x_defer"] = feats["has_competing_unverified_support"] * feats["a_DEFER"]
    feats["n_hyp_unverified_support_x_defer"] = feats["n_hyp_unverified_support"] * feats["a_DEFER"]
    feats["n_hyp_unverified_contradiction_x_defer"] = feats["n_hyp_unverified_contradiction"] * feats["a_DEFER"]
    return feats


def get_v1_feature_keys() -> list[str]:
    dummy_sf = {k: 0 for k in [
        "n_live", "n_eliminated", "n_untested", "n_total_hypotheses",
        "n_visible_evidence", "n_verified", "n_supporting", "n_contradicting",
        "n_stale", "retrieval_remaining", "search_remaining", "verify_remaining",
        "steps_remaining", "can_retrieve", "can_search", "can_verify",
        "searched", "reasoning_complete", "same_action_run_length",
        "retrieval_count", "search_count", "verify_count",
    ]}
    feats = extract_v1_features(dummy_sf, "ANSWER")
    return sorted(feats.keys())


def get_v2r_feature_keys() -> list[str]:
    dummy_sf = {k: 0 for k in [
        "n_live", "n_eliminated", "n_untested", "n_total_hypotheses",
        "n_visible_evidence", "n_verified", "n_supporting", "n_contradicting",
        "n_stale", "retrieval_remaining", "search_remaining", "verify_remaining",
        "steps_remaining", "can_retrieve", "can_search", "can_verify",
        "searched", "reasoning_complete", "same_action_run_length",
        "retrieval_count", "search_count", "verify_count",
    ]}
    dummy_struct = {"n_hyp_unverified_support": 0, "n_hyp_unverified_contradiction": 0, "has_competing_unverified_support": 0}
    feats = extract_v2r_features(dummy_sf, "ANSWER", dummy_struct)
    return sorted(feats.keys())


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ============================================================
# Step 1: Load data and reconstruct structural features
# ============================================================

def load_data():
    """Load causal records and checkpoints, reconstruct structural features."""
    print("\n=== Step 1: Load data and reconstruct structural features ===")

    # Load checkpoints
    checkpoints = {}
    with open(CHECKPOINTS_PATH) as f:
        for line in f:
            r = json.loads(line)
            checkpoints[r["checkpoint_id"]] = r
    print(f"  Checkpoints: {len(checkpoints)}")

    # Load causal records
    causal_records = []
    with open(CAUSAL_DATA_PATH) as f:
        for line in f:
            causal_records.append(json.loads(line))
    print(f"  Causal records: {len(causal_records)}")

    # Reconstruct structural features for each causal record
    matched = 0
    for r in causal_records:
        ckpt = checkpoints.get(r["checkpoint_id"])
        if ckpt is None:
            r["structural_features"] = {"n_hyp_unverified_support": 0, "n_hyp_unverified_contradiction": 0, "has_competing_unverified_support": 0}
            continue
        r["structural_features"] = compute_structural_features(ckpt["evidence"])
        matched += 1
    print(f"  Reconstructed structural features: {matched}/{len(causal_records)}")

    return causal_records, checkpoints


# ============================================================
# Step 2: Leakage tests
# ============================================================

def run_leakage_tests(causal_records, checkpoints):
    """Run 4 leakage tests per feature."""
    print("\n=== Step 2: Leakage tests ===")

    tests = {}

    for feat_name in NEW_FEATURES:
        tests[feat_name] = {
            "test1_observable_from_visible_snapshot": None,
            "test2_no_verify_result": None,
            "test3_no_hidden_evidence": None,
            "test4_unchanged_by_future_outcomes": None,
        }

    # Test 1: Observable from visible snapshot
    # Verify that the feature can be computed from visible evidence alone
    all_ok = True
    for r in causal_records[:100]:  # sample
        ckpt = checkpoints[r["checkpoint_id"]]
        visible_ev = [ev for ev in ckpt["evidence"] if ev.get("retrieved", False)]
        computed = compute_structural_features(visible_ev)
        if computed != r["structural_features"]:
            all_ok = False
            break
    tests["n_hyp_unverified_support"]["test1_observable_from_visible_snapshot"] = all_ok
    tests["n_hyp_unverified_contradiction"]["test1_observable_from_visible_snapshot"] = all_ok
    tests["has_competing_unverified_support"]["test1_observable_from_visible_snapshot"] = all_ok
    print(f"  Test 1 (observable from visible snapshot): {'PASS' if all_ok else 'FAIL'}")

    # Test 2: No verify_result usage
    # The compute_structural_features function must not access verify_result.
    # We check the AST for any attribute access or key lookup of "verify_result",
    # not just the string (which would false-positive on comments/docstrings).
    import ast
    src = inspect.getsource(compute_structural_features)
    tree = ast.parse(src)
    uses_verify_result = False
    for node in ast.walk(tree):
        # Check subscript: ev["verify_result"]
        if isinstance(node, ast.Subscript):
            slc = node.slice
            if isinstance(slc, ast.Constant) and isinstance(slc.value, str) and slc.value == "verify_result":
                uses_verify_result = True
                break
        # Check attribute: ev.verify_result
        if isinstance(node, ast.Attribute) and node.attr == "verify_result":
            uses_verify_result = True
            break
    no_verify_result = not uses_verify_result
    for feat_name in NEW_FEATURES:
        tests[feat_name]["test2_no_verify_result"] = no_verify_result
    print(f"  Test 2 (no verify_result): {'PASS' if no_verify_result else 'FAIL'}")

    # Test 3: No hidden evidence
    # The function filters to retrieved=True (visible only)
    no_hidden = "retrieved" in src and "if not ev.get" in src
    for feat_name in NEW_FEATURES:
        tests[feat_name]["test3_no_hidden_evidence"] = no_hidden
    print(f"  Test 3 (no hidden evidence): {'PASS' if no_hidden else 'FAIL'}")

    # Test 4: Unchanged by future outcomes
    # The feature depends only on current evidence state, not on what
    # verification WILL find. Verify by checking that the feature doesn't
    # change if we only modify verify_result on unverified evidence.
    all_ok = True
    for r in causal_records[:100]:
        ckpt = checkpoints[r["checkpoint_id"]]
        # Modify verify_result on unverified evidence (simulating future outcome)
        modified_ev = []
        for ev in ckpt["evidence"]:
            ev_copy = dict(ev)
            if ev_copy.get("verification_state") == "UNVERIFIED":
                ev_copy["verify_result"] = "SUFFICIENT"  # simulate future
            modified_ev.append(ev_copy)
        original = compute_structural_features(ckpt["evidence"])
        modified = compute_structural_features(modified_ev)
        if original != modified:
            all_ok = False
            break
    for feat_name in NEW_FEATURES:
        tests[feat_name]["test4_unchanged_by_future_outcomes"] = all_ok
    print(f"  Test 4 (unchanged by future outcomes): {'PASS' if all_ok else 'FAIL'}")

    # Save
    leakage_path = OUTPUT_DIR / "leakage_tests.json"
    with open(leakage_path, "w") as f:
        json.dump({
            "tests": tests,
            "all_pass": all(all(t.values()) for t in tests.values()),
            "description": "4 leakage tests per feature: observable, no verify_result, no hidden, unchanged by future",
        }, f, indent=2)
    print(f"  Saved: {leakage_path}")

    return all(all(t.values()) for t in tests.values())


# ============================================================
# Step 3: Train Q_V1 (control) and Q_V2R (repaired)
# ============================================================

def train_models(causal_records):
    """Train Q_V1 and Q_V2R on same data, same GBT, same hyperparameters."""
    print("\n=== Step 3: Train Q_V1 (control) and Q_V2R (repaired) ===")

    v1_keys = get_v1_feature_keys()
    v2r_keys = get_v2r_feature_keys()
    print(f"  V1 features: {len(v1_keys)}")
    print(f"  V2R features: {len(v2r_keys)} (+{len(v2r_keys) - len(v1_keys)} new)")

    # Build feature matrices
    X_v1 = []
    X_v2r = []
    y = []
    for r in causal_records:
        sf = r["state_features"]
        action = r["forced_action"]
        structural = r["structural_features"]

        v1_feats = extract_v1_features(sf, action)
        v2r_feats = extract_v2r_features(sf, action, structural)

        X_v1.append([v1_feats[k] for k in v1_keys])
        X_v2r.append([v2r_feats[k] for k in v2r_keys])
        y.append(r["pinned_policy_utility"])

    X_v1 = np.array(X_v1)
    X_v2r = np.array(X_v2r)
    y = np.array(y)

    print(f"  Training data: {len(y)} records, target range [{y.min():.2f}, {y.max():.2f}]")

    # Train Q_V1 (control) — same GBT, same hyperparameters
    print("  Training Q_V1 (control)...")
    q_v1 = GradientBoostingRegressor(**GBT_PARAMS)
    q_v1.fit(X_v1, y)
    v1_pkl = pickle.dumps(q_v1)
    v1_sha = sha256_bytes(v1_pkl)
    (OUTPUT_DIR / "Q_V1_control.pkl").write_bytes(v1_pkl)
    print(f"    Q_V1 SHA: {v1_sha[:16]}...")

    # Train Q_V2R (repaired) — same GBT, same hyperparameters
    print("  Training Q_V2R (repaired)...")
    q_v2r = GradientBoostingRegressor(**GBT_PARAMS)
    q_v2r.fit(X_v2r, y)
    v2r_pkl = pickle.dumps(q_v2r)
    v2r_sha = sha256_bytes(v2r_pkl)
    (OUTPUT_DIR / "Q_V2R_repaired.pkl").write_bytes(v2r_pkl)
    print(f"    Q_V2R SHA: {v2r_sha[:16]}...")

    # Save schema
    schema = {
        "name": "Q_STATE_SCHEMA_V2",
        "version": "V2R",
        "v1_frozen_permanently": True,
        "new_features": NEW_FEATURES,
        "new_feature_definitions": {
            "n_hyp_unverified_support": "Number of hypotheses with >=1 unverified supporting evidence (visible only)",
            "n_hyp_unverified_contradiction": "Number of hypotheses with >=1 unverified contradicting evidence (visible only)",
            "has_competing_unverified_support": "Binary: n_hyp_unverified_support > 1 (contradiction signal)",
        },
        "new_feature_interactions": [
            "has_competing_x_defer",
            "n_hyp_unverified_support_x_defer",
            "n_hyp_unverified_contradiction_x_defer",
        ],
        "v1_feature_keys": v1_keys,
        "v2r_feature_keys": v2r_keys,
        "n_v1_features": len(v1_keys),
        "n_v2r_features": len(v2r_keys),
        "gbt_hyperparameters": GBT_PARAMS,
        "training_data": str(CAUSAL_DATA_PATH),
        "n_training_records": len(y),
        "q_v1_sha256": v1_sha,
        "q_v2r_sha256": v2r_sha,
        "authority_threshold": AUTHORITY_THRESHOLD,
        "separation_margin": SEPARATION_MARGIN,
    }
    schema_path = OUTPUT_DIR / "Q_STATE_SCHEMA_V2.json"
    with open(schema_path, "w") as f:
        json.dump(schema, f, indent=2)
    print(f"  Schema saved: {schema_path}")

    return q_v1, q_v2r, v1_keys, v2r_keys, v1_sha, v2r_sha


# ============================================================
# Step 4: Offline separation audit at problematic states
# ============================================================

def predict_q(model, feature_keys, state_features, action, structural=None):
    """Predict Q(s, a) for a single action."""
    if structural is not None:
        feats = extract_v2r_features(state_features, action, structural)
    else:
        feats = extract_v1_features(state_features, action)
    x = np.array([[feats[k] for k in feature_keys]])
    return float(model.predict(x)[0])


def predict_all_q(model, feature_keys, state_features, legal_actions, structural=None):
    """Predict Q(s, a) for all legal actions."""
    q_vals = {}
    for a in legal_actions:
        q_vals[a] = predict_q(model, feature_keys, state_features, a, structural)
    return q_vals


def separation_audit(causal_records, checkpoints, q_v1, q_v2r, v1_keys, v2r_keys):
    """Offline separation audit at problematic defer/contradiction states."""
    print("\n=== Step 4: Offline separation audit ===")

    # Find defer-correct and contradiction-correct checkpoints
    defer_ckpts = {cid: c for cid, c in checkpoints.items() if c["category"] in ("ol_defer", "tl_defer")}
    contra_ckpts = {cid: c for cid, c in checkpoints.items() if c["category"] in ("ol_contradiction", "tl_contradiction", "contradiction")}

    # Also check the I3.26 development benchmark states
    # We need to generate those states too
    sys.path.insert(0, str(REPO_ROOT))
    from hrm_adaptive_memory.cognitive_control.core import DecisionAction
    from hrm_adaptive_memory.executive.evidence_benchmark import initial_evidence_runtime
    from hrm_adaptive_memory.executive.evidence_benchmark.executor import EvidenceExecutor
    from hrm_adaptive_memory.executive.evidence_benchmark.i3_26_development_generator import generate_development_benchmark
    from hrm_adaptive_memory.executive.evidence_benchmark.i3_5_confirmation_generator import CONFIRMATION_BUDGET_PROFILES
    from hrm_adaptive_memory.executive.resources import ResourceBudget, ResourceState
    from daph.intervention.checkpoint import compute_state_features

    executor = EvidenceExecutor()
    dev_tasks = generate_development_benchmark(seed=7719)
    dev_defer = [t for t in dev_tasks if t.category == "defer"]
    dev_contra = [t for t in dev_tasks if t.category == "contradiction"]

    def get_budget(task):
        p = CONFIRMATION_BUDGET_PROFILES[task.budget_profile]
        return ResourceBudget(
            max_executive_steps=p["max_executive_steps"],
            max_retrieval_calls=p["max_retrieval_calls"],
            max_verification_calls=p["max_verification_calls"],
            max_search_calls=p["max_search_calls"],
            max_reasoning_tokens=p.get("max_reasoning_tokens", 256),
            max_elapsed_ms=p.get("max_elapsed_ms", 10_000),
        )

    def get_dev_state_and_q(task, prior_actions=()):
        budget = get_budget(task)
        runtime = initial_evidence_runtime(task, ResourceState(budget=budget))
        for action_str in prior_actions:
            action = DecisionAction(action_str)
            try:
                res = executor.execute(runtime, action)
                runtime = res.runtime
                if res.terminal:
                    return None, None, None, None
            except:
                return None, None, None, None
        sf = compute_state_features(runtime, tuple(prior_actions))
        # Compute structural features from visible evidence
        visible_ev = []
        for ev in runtime.visible_evidence:
            visible_ev.append({
                "evidence_id": ev.evidence_id,
                "supports": list(ev.supports),
                "contradicts": list(ev.contradicts),
                "verification_state": ev.verification_state.name,
                "retrieved": ev.retrieved,
            })
        structural = compute_structural_features(visible_ev)
        legal = ["ANSWER", "DEFER", "RETRIEVE", "VERIFY", "SEARCH_MORE", "REASON_MORE", "STOP"]
        legal = [a for a in legal if runtime.resources.can_execute(DecisionAction(a))]
        return sf, structural, legal, runtime

    # Collect defer-correct and contra-correct states
    defer_states = []
    contra_states = []

    # From I3.26 dev benchmark
    for task in dev_defer:
        sf, structural, legal, _ = get_dev_state_and_q(task)
        if sf is not None:
            defer_states.append({"source": "dev_defer_s0", "task_id": task.task_id, "sf": sf, "structural": structural, "legal": legal})

    for task in dev_contra:
        # Step 0 (DEFER would be wrong)
        sf, structural, legal, _ = get_dev_state_and_q(task)
        if sf is not None:
            contra_states.append({"source": "dev_contra_s0", "task_id": task.task_id, "sf": sf, "structural": structural, "legal": legal})
        # Step 1 (after SEARCH_MORE — the exact aliasing state)
        sf, structural, legal, _ = get_dev_state_and_q(task, ["SEARCH_MORE"])
        if sf is not None:
            contra_states.append({"source": "dev_contra_s1_after_search", "task_id": task.task_id, "sf": sf, "structural": structural, "legal": legal})

    # From I3.5 checkpoints
    for cid, ckpt in defer_ckpts.items():
        sf = ckpt["state_features"]
        structural = compute_structural_features(ckpt["evidence"])
        legal = ckpt.get("legal_actions", ["ANSWER", "DEFER", "RETRIEVE", "VERIFY", "SEARCH_MORE", "REASON_MORE", "STOP"])
        defer_states.append({"source": "i3_5_defer", "task_id": ckpt["task_id"], "sf": sf, "structural": structural, "legal": legal})

    for cid, ckpt in contra_ckpts.items():
        sf = ckpt["state_features"]
        structural = compute_structural_features(ckpt["evidence"])
        legal = ckpt.get("legal_actions", ["ANSWER", "DEFER", "RETRIEVE", "VERIFY", "SEARCH_MORE", "REASON_MORE", "STOP"])
        contra_states.append({"source": "i3_5_contra", "task_id": ckpt["task_id"], "sf": sf, "structural": structural, "legal": legal})

    print(f"  Defer-correct states: {len(defer_states)}")
    print(f"  Contra-correct states: {len(contra_states)}")

    # Compute Q(DEFER) for both models at all states
    results = {"defer_correct": [], "contra_correct": []}

    for s in defer_states:
        q_v1_vals = predict_all_q(q_v1, v1_keys, s["sf"], s["legal"])
        q_v2r_vals = predict_all_q(q_v2r, v2r_keys, s["sf"], s["legal"], s["structural"])
        results["defer_correct"].append({
            "source": s["source"], "task_id": s["task_id"],
            "q_v1_DEFER": q_v1_vals.get("DEFER", 0),
            "q_v2r_DEFER": q_v2r_vals.get("DEFER", 0),
            "q_v1_all": q_v1_vals,
            "q_v2r_all": q_v2r_vals,
            "structural": s["structural"],
        })

    for s in contra_states:
        q_v1_vals = predict_all_q(q_v1, v1_keys, s["sf"], s["legal"])
        q_v2r_vals = predict_all_q(q_v2r, v2r_keys, s["sf"], s["legal"], s["structural"])
        results["contra_correct"].append({
            "source": s["source"], "task_id": s["task_id"],
            "q_v1_DEFER": q_v1_vals.get("DEFER", 0),
            "q_v2r_DEFER": q_v2r_vals.get("DEFER", 0),
            "q_v1_all": q_v1_vals,
            "q_v2r_all": q_v2r_vals,
            "structural": s["structural"],
        })

    # Print summary
    print("\n  Q(DEFER) comparison:")
    v1_defer = [r["q_v1_DEFER"] for r in results["defer_correct"]]
    v1_contra = [r["q_v1_DEFER"] for r in results["contra_correct"]]
    v2r_defer = [r["q_v2r_DEFER"] for r in results["defer_correct"]]
    v2r_contra = [r["q_v2r_DEFER"] for r in results["contra_correct"]]

    print(f"    V1  defer-correct: mean={np.mean(v1_defer):.2f}, range=[{np.min(v1_defer):.2f}, {np.max(v1_defer):.2f}]")
    print(f"    V1  contra-correct: mean={np.mean(v1_contra):.2f}, range=[{np.min(v1_contra):.2f}, {np.max(v1_contra):.2f}]")
    print(f"    V2R defer-correct: mean={np.mean(v2r_defer):.2f}, range=[{np.min(v2r_defer):.2f}, {np.max(v2r_defer):.2f}]")
    print(f"    V2R contra-correct: mean={np.mean(v2r_contra):.2f}, range=[{np.min(v2r_contra):.2f}, {np.max(v2r_contra):.2f}]")

    # Gate: min Q_V2R(DEFER|defer-correct) - max Q_V2R(DEFER|contra) > SEPARATION_MARGIN
    min_defer = min(v2r_defer)
    max_contra = max(v2r_contra)
    margin = min_defer - max_contra
    gate_pass = margin > SEPARATION_MARGIN

    print(f"\n  Gate: min Q_V2R(DEFER|defer) - max Q_V2R(DEFER|contra) = {min_defer:.2f} - {max_contra:.2f} = {margin:.2f}")
    print(f"  Required: > {SEPARATION_MARGIN}")
    print(f"  Result: {'PASS' if gate_pass else 'FAIL'}")

    # Also check: does V2R appropriately value VERIFY/SEARCH in contra states?
    print("\n  Q(VERIFY) and Q(SEARCH_MORE) in contra states (should be higher than DEFER):")
    for r in results["contra_correct"][:5]:
        q = r["q_v2r_all"]
        print(f"    {r['source']} {r['task_id']}: Q(DEFER)={q.get('DEFER',0):.2f}, Q(VERIFY)={q.get('VERIFY',0):.2f}, Q(SEARCH)={q.get('SEARCH_MORE',0):.2f}, Q(ANSWER)={q.get('ANSWER',0):.2f}")

    audit = {
        "n_defer_states": len(defer_states),
        "n_contra_states": len(contra_states),
        "v1_defer_q_defer": {"mean": float(np.mean(v1_defer)), "min": float(np.min(v1_defer)), "max": float(np.max(v1_defer))},
        "v1_contra_q_defer": {"mean": float(np.mean(v1_contra)), "min": float(np.min(v1_contra)), "max": float(np.max(v1_contra))},
        "v2r_defer_q_defer": {"mean": float(np.mean(v2r_defer)), "min": float(np.min(v2r_defer)), "max": float(np.max(v2r_defer))},
        "v2r_contra_q_defer": {"mean": float(np.mean(v2r_contra)), "min": float(np.min(v2r_contra)), "max": float(np.max(v2r_contra))},
        "separation_margin": float(margin),
        "required_margin": SEPARATION_MARGIN,
        "gate_pass": gate_pass,
        "defer_states_detail": results["defer_correct"],
        "contra_states_detail": results["contra_correct"],
    }

    audit_path = OUTPUT_DIR / "separation_audit.json"
    with open(audit_path, "w") as f:
        json.dump(audit, f, indent=2, default=str)
    print(f"  Saved: {audit_path}")

    return gate_pass, audit


# ============================================================
# Step 5: Preservation check on 220 causal checkpoints
# ============================================================

def preservation_check(causal_records, checkpoints, q_v1, q_v2r, v1_keys, v2r_keys):
    """Check that V2R preserves what V1 already knew."""
    print("\n=== Step 5: Preservation check on 220 causal checkpoints ===")

    # For each unique checkpoint, compute Q for all legal actions
    # and compare V1 vs V2R
    unique_ckpts = {}
    for r in causal_records:
        cid = r["checkpoint_id"]
        if cid not in unique_ckpts:
            unique_ckpts[cid] = {
                "state_features": r["state_features"],
                "structural": r["structural_features"],
                "correct_first_action": r["correct_first_action"],
                "category": r["category"],
                "task_id": r["task_id"],
            }

    print(f"  Unique checkpoints: {len(unique_ckpts)}")

    # Compute Q values and regret for each checkpoint
    results = []
    v1_regrets = []
    v2r_regrets = []
    v1_near_optimal = 0
    v2r_near_optimal = 0
    v1_correct_best = 0
    v2r_correct_best = 0
    answer_cases_v1_correct = 0
    answer_cases_v2r_correct = 0
    answer_cases_total = 0
    new_false_defer_v2r = 0

    ACTIONS = ["ANSWER", "DEFER", "RETRIEVE", "VERIFY", "SEARCH_MORE", "REASON_MORE", "STOP"]

    for cid, info in unique_ckpts.items():
        sf = info["state_features"]
        structural = info["structural"]
        correct = info["correct_first_action"]
        legal = ACTIONS  # use all for comparison

        q_v1_vals = predict_all_q(q_v1, v1_keys, sf, legal)
        q_v2r_vals = predict_all_q(q_v2r, v2r_keys, sf, legal, structural)

        # Best action
        v1_best = max(q_v1_vals, key=q_v1_vals.get)
        v2r_best = max(q_v2r_vals, key=q_v2r_vals.get)

        # Near-optimal: correct action within epsilon=3.0 of best
        v1_q_correct = q_v1_vals.get(correct, -999)
        v2r_q_correct = q_v2r_vals.get(correct, -999)
        v1_q_max = max(q_v1_vals.values())
        v2r_q_max = max(q_v2r_vals.values())

        v1_regret = v1_q_max - v1_q_correct
        v2r_regret = v2r_q_max - v2r_q_correct
        v1_regrets.append(v1_regret)
        v2r_regrets.append(v2r_regret)

        if v1_regret <= 3.0:
            v1_near_optimal += 1
        if v2r_regret <= 3.0:
            v2r_near_optimal += 1

        if v1_best == correct:
            v1_correct_best += 1
        if v2r_best == correct:
            v2r_correct_best += 1

        # ANSWER authority cases
        if correct == "ANSWER":
            answer_cases_total += 1
            if v1_best == "ANSWER":
                answer_cases_v1_correct += 1
            if v2r_best == "ANSWER":
                answer_cases_v2r_correct += 1

        # New false high-confidence DEFER: V2R recommends DEFER with gap > 5
        # but correct action is not DEFER
        if v2r_best == "DEFER" and correct != "DEFER":
            v2r_q_second = sorted(q_v2r_vals.values(), reverse=True)[1]
            v2r_gap = v2r_q_max - v2r_q_second
            if v2r_gap > AUTHORITY_THRESHOLD:
                new_false_defer_v2r += 1

        results.append({
            "checkpoint_id": cid[:16],
            "category": info["category"],
            "correct": correct,
            "v1_best": v1_best,
            "v2r_best": v2r_best,
            "v1_regret": float(v1_regret),
            "v2r_regret": float(v2r_regret),
            "v1_q_correct": float(v1_q_correct),
            "v2r_q_correct": float(v2r_q_correct),
        })

    # Summary
    v1_mean_regret = float(np.mean(v1_regrets))
    v2r_mean_regret = float(np.mean(v2r_regrets))
    n = len(unique_ckpts)

    print(f"  Mean regret: V1={v1_mean_regret:.4f}, V2R={v2r_mean_regret:.4f}")
    print(f"  Near-optimal (eps=3): V1={v1_near_optimal}/{n}, V2R={v2r_near_optimal}/{n}")
    print(f"  Correct best action: V1={v1_correct_best}/{n}, V2R={v2r_correct_best}/{n}")
    print(f"  ANSWER cases: V1={answer_cases_v1_correct}/{answer_cases_total}, V2R={answer_cases_v2r_correct}/{answer_cases_total}")
    print(f"  New false high-conf DEFER (V2R): {new_false_defer_v2r}")

    # Gates
    gates = {
        "regret_no_worse": v2r_mean_regret <= v1_mean_regret * 1.05,  # 5% tolerance
        "near_optimal_no_worse": v2r_near_optimal >= v1_near_optimal,
        "answer_cases_preserved": answer_cases_v2r_correct >= answer_cases_v1_correct,
        "no_new_false_high_conf_defer": new_false_defer_v2r == 0,
        "correct_best_no_regression": v2r_correct_best >= v1_correct_best * 0.95,  # 5% tolerance
    }

    print(f"\n  Gates:")
    for g, v in gates.items():
        print(f"    {g}: {'PASS' if v else 'FAIL'}")

    all_pass = all(gates.values())

    check = {
        "n_checkpoints": n,
        "v1_mean_regret": v1_mean_regret,
        "v2r_mean_regret": v2r_mean_regret,
        "v1_near_optimal": v1_near_optimal,
        "v2r_near_optimal": v2r_near_optimal,
        "v1_correct_best": v1_correct_best,
        "v2r_correct_best": v2r_correct_best,
        "answer_cases_total": answer_cases_total,
        "answer_cases_v1_correct": answer_cases_v1_correct,
        "answer_cases_v2r_correct": answer_cases_v2r_correct,
        "new_false_high_conf_defer_v2r": new_false_defer_v2r,
        "gates": gates,
        "all_gates_pass": all_pass,
        "detail": results,
    }

    check_path = OUTPUT_DIR / "preservation_check.json"
    with open(check_path, "w") as f:
        json.dump(check, f, indent=2, default=str)
    print(f"  Saved: {check_path}")

    return all_pass, check


# ============================================================
# Step 6: Offline authority test
# ============================================================

def offline_authority_test(causal_records, checkpoints, q_v1, q_v2r, v1_keys, v2r_keys):
    """Test authority rules offline: would_force_ANSWER, would_force_DEFER, correct causal best."""
    print("\n=== Step 6: Offline authority test ===")

    unique_ckpts = {}
    for r in causal_records:
        cid = r["checkpoint_id"]
        if cid not in unique_ckpts:
            unique_ckpts[cid] = {
                "state_features": r["state_features"],
                "structural": r["structural_features"],
                "correct_first_action": r["correct_first_action"],
                "category": r["category"],
                "task_id": r["task_id"],
            }

    ACTIONS = ["ANSWER", "DEFER", "RETRIEVE", "VERIFY", "SEARCH_MORE", "REASON_MORE", "STOP"]

    # Test both A2A (ANSWER authority) and A2AD (ANSWER+DEFER authority with V2R)
    a2a_results = []  # V1 + ANSWER authority
    a2ad_results = []  # V2R + ANSWER/DEFER authority

    a2a_false_authority = 0
    a2a_triggers = 0
    a2ad_false_authority_answer = 0
    a2ad_false_authority_defer = 0
    a2ad_answer_triggers = 0
    a2ad_defer_triggers = 0

    for cid, info in unique_ckpts.items():
        sf = info["state_features"]
        structural = info["structural"]
        correct = info["correct_first_action"]

        # V1 Q values (for A2A)
        q_v1_vals = predict_all_q(q_v1, v1_keys, sf, ACTIONS)
        v1_sorted = sorted(q_v1_vals.items(), key=lambda x: -x[1])
        v1_best = v1_sorted[0][0]
        v1_q_max = v1_sorted[0][1]
        v1_q_second = v1_sorted[1][1] if len(v1_sorted) > 1 else 0
        v1_gap = v1_q_max - v1_q_second

        # V2R Q values (for A2AD)
        q_v2r_vals = predict_all_q(q_v2r, v2r_keys, sf, ACTIONS, structural)
        v2r_sorted = sorted(q_v2r_vals.items(), key=lambda x: -x[1])
        v2r_best = v2r_sorted[0][0]
        v2r_q_max = v2r_sorted[0][1]
        v2r_q_second = v2r_sorted[1][1] if len(v2r_sorted) > 1 else 0
        v2r_gap = v2r_q_max - v2r_q_second

        # A2A: V1 + ANSWER authority
        # Force ANSWER if: v1_best == ANSWER, gap > 5, and ANSWER is sole near-optimal
        v1_near_opt = [a for a, q in q_v1_vals.items() if q >= v1_q_max - 3.0]
        a2a_would_force = (v1_best == "ANSWER" and v1_gap > AUTHORITY_THRESHOLD
                          and len(v1_near_opt) == 1 and v1_near_opt[0] == "ANSWER")
        a2a_correct_force = (a2a_would_force and correct == "ANSWER")
        a2a_false_force = (a2a_would_force and correct != "ANSWER")

        if a2a_would_force:
            a2a_triggers += 1
        if a2a_false_force:
            a2a_false_authority += 1

        a2a_results.append({
            "checkpoint_id": cid[:16],
            "correct": correct,
            "v1_best": v1_best,
            "v1_gap": float(v1_gap),
            "would_force_answer": a2a_would_force,
            "correct_force": a2a_correct_force,
            "false_force": a2a_false_force,
        })

        # A2AD: V2R + ANSWER/DEFER authority with structural safety predicate
        v2r_near_opt = [a for a, q in q_v2r_vals.items() if q >= v2r_q_max - 3.0]

        # ANSWER authority (same rule, V2R Q)
        a2ad_would_force_answer = (v2r_best == "ANSWER" and v2r_gap > AUTHORITY_THRESHOLD
                                   and len(v2r_near_opt) == 1 and v2r_near_opt[0] == "ANSWER")
        a2ad_answer_correct = (a2ad_would_force_answer and correct == "ANSWER")
        a2ad_answer_false = (a2ad_would_force_answer and correct != "ANSWER")

        # DEFER authority with structural safety predicate:
        # Force DEFER if:
        #   Q(DEFER) - Q_second >= 5
        #   AND NOT has_competing_unverified_support
        #   AND DEFER is sole near-optimal
        #   AND n_hyp_unverified_contradiction > 0 (contradiction exists, not just absence)
        q_v2r_defer = q_v2r_vals.get("DEFER", 0)
        v2r_q_sorted = sorted(q_v2r_vals.values(), reverse=True)
        v2r_q_max_val = v2r_q_sorted[0]
        v2r_q_second_val = v2r_q_sorted[1] if len(v2r_q_sorted) > 1 else 0

        defer_is_best = (v2r_best == "DEFER")
        defer_gap = q_v2r_defer - v2r_q_second_val if defer_is_best else 0
        defer_sole_near_opt = (defer_is_best and len(v2r_near_opt) == 1 and v2r_near_opt[0] == "DEFER")
        no_competing_support = structural["has_competing_unverified_support"] == 0
        has_contradiction = structural["n_hyp_unverified_contradiction"] > 0

        a2ad_would_force_defer = (defer_is_best and defer_gap >= AUTHORITY_THRESHOLD
                                  and defer_sole_near_opt
                                  and no_competing_support
                                  and has_contradiction)
        a2ad_defer_correct = (a2ad_would_force_defer and correct == "DEFER")
        a2ad_defer_false = (a2ad_would_force_defer and correct != "DEFER")

        if a2ad_would_force_answer:
            a2ad_answer_triggers += 1
        if a2ad_would_force_defer:
            a2ad_defer_triggers += 1
        if a2ad_answer_false:
            a2ad_false_authority_answer += 1
        if a2ad_defer_false:
            a2ad_false_authority_defer += 1

        a2ad_results.append({
            "checkpoint_id": cid[:16],
            "correct": correct,
            "v2r_best": v2r_best,
            "v2r_gap": float(v2r_gap),
            "structural": structural,
            "would_force_answer": a2ad_would_force_answer,
            "would_force_defer": a2ad_would_force_defer,
            "answer_correct": a2ad_answer_correct,
            "answer_false": a2ad_answer_false,
            "defer_correct": a2ad_defer_correct,
            "defer_false": a2ad_defer_false,
        })

    n = len(unique_ckpts)
    print(f"  Checkpoints: {n}")
    print(f"  A2A (V1 + ANSWER): triggers={a2a_triggers}, false_authority={a2a_false_authority}")
    print(f"  A2AD (V2R + ANSWER): triggers={a2ad_answer_triggers}, false_authority={a2ad_false_authority_answer}")
    print(f"  A2AD (V2R + DEFER):  triggers={a2ad_defer_triggers}, false_authority={a2ad_false_authority_defer}")

    # Gates
    gates = {
        "a2a_false_authority_rate": a2a_false_authority / a2a_triggers if a2a_triggers > 0 else 0.0,
        "a2ad_answer_false_authority_rate": a2ad_false_authority_answer / a2ad_answer_triggers if a2ad_answer_triggers > 0 else 0.0,
        "a2ad_defer_false_authority_rate": a2ad_false_authority_defer / a2ad_defer_triggers if a2ad_defer_triggers > 0 else 0.0,
    }

    print(f"\n  False authority rates:")
    for g, v in gates.items():
        print(f"    {g}: {v:.4f}")

    # Require essentially zero
    all_pass = (gates["a2a_false_authority_rate"] <= 0.01
                and gates["a2ad_answer_false_authority_rate"] <= 0.01
                and gates["a2ad_defer_false_authority_rate"] <= 0.01)

    test = {
        "n_checkpoints": n,
        "a2a_triggers": a2a_triggers,
        "a2a_false_authority": a2a_false_authority,
        "a2ad_answer_triggers": a2ad_answer_triggers,
        "a2ad_answer_false_authority": a2ad_false_authority_answer,
        "a2ad_defer_triggers": a2ad_defer_triggers,
        "a2ad_defer_false_authority": a2ad_false_authority_defer,
        "false_authority_rates": gates,
        "all_pass": all_pass,
        "a2a_detail": a2a_results,
        "a2ad_detail": a2ad_results,
    }

    test_path = OUTPUT_DIR / "offline_authority_test.json"
    with open(test_path, "w") as f:
        json.dump(test, f, indent=2, default=str)
    print(f"  Saved: {test_path}")

    return all_pass, test


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("I3.28: Authority-State Sufficiency — Representation Repair for DEFER")
    print("=" * 70)

    # Step 1: Load data
    causal_records, checkpoints = load_data()

    # Step 2: Leakage tests
    leakage_pass = run_leakage_tests(causal_records, checkpoints)
    if not leakage_pass:
        print("\n*** LEAKAGE TESTS FAILED — STOP ***")
        return

    # Step 3: Train models
    q_v1, q_v2r, v1_keys, v2r_keys, v1_sha, v2r_sha = train_models(causal_records)

    # Step 4: Separation audit
    sep_pass, sep_audit = separation_audit(causal_records, checkpoints, q_v1, q_v2r, v1_keys, v2r_keys)

    # Step 5: Preservation check
    pres_pass, pres_check = preservation_check(causal_records, checkpoints, q_v1, q_v2r, v1_keys, v2r_keys)

    # Step 6: Offline authority test
    auth_pass, auth_test = offline_authority_test(causal_records, checkpoints, q_v1, q_v2r, v1_keys, v2r_keys)

    # Summary
    print("\n" + "=" * 70)
    print("I3.28 SUMMARY")
    print("=" * 70)
    print(f"  Leakage tests:        {'PASS' if leakage_pass else 'FAIL'}")
    print(f"  Separation audit:     {'PASS' if sep_pass else 'FAIL'} (margin={sep_audit['separation_margin']:.2f}, required>{SEPARATION_MARGIN})")
    print(f"  Preservation check:   {'PASS' if pres_pass else 'FAIL'}")
    print(f"  Offline authority:    {'PASS' if auth_pass else 'FAIL'}")
    print()

    all_pass = leakage_pass and sep_pass and pres_pass and auth_pass
    print(f"  OVERALL: {'PASS — proceed to live validation' if all_pass else 'FAIL — do not proceed to live'}")

    # Save full results
    full_results = {
        "experiment": "I3.28: Authority-State Sufficiency",
        "date": "2026-08-26",
        "leakage_pass": leakage_pass,
        "separation_pass": sep_pass,
        "separation_margin": sep_audit["separation_margin"],
        "required_margin": SEPARATION_MARGIN,
        "preservation_pass": pres_pass,
        "preservation_gates": pres_check["gates"],
        "authority_pass": auth_pass,
        "authority_false_rates": auth_test["false_authority_rates"],
        "all_pass": all_pass,
        "q_v1_sha256": v1_sha,
        "q_v2r_sha256": v2r_sha,
        "gbt_params": GBT_PARAMS,
        "n_training_records": len(causal_records),
        "n_checkpoints": len(checkpoints),
    }
    full_path = OUTPUT_DIR / "full_results.json"
    with open(full_path, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"\n  Full results: {full_path}")

    if all_pass:
        print("\n  RECOMMENDATION: Freeze Q_V2R and proceed to live validation sequence:")
        print("    1. Rescue audit on known DEFER failures")
        print("    2. Negative controls (contradiction variants)")
        print("    3. Targeted live safety run")
        print("    4. Fresh untouched confirmation")
    else:
        print("\n  RECOMMENDATION: Do NOT proceed to live DEFER authority.")
        print("    Identify which gate failed and address the root cause.")


if __name__ == "__main__":
    main()
