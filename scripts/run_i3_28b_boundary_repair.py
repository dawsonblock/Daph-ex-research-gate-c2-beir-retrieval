#!/usr/bin/env python3
"""I3.28B: DEFER Boundary Causal Repair.

Generate matched state pairs at the exact aliasing boundary:
  n_verified = 0
  A (safe defer):    has_competing_unverified_support = 0  → DEFER is correct
  B (unsafe defer):  has_competing_unverified_support = 1  → DEFER is wrong

Force ALL legal actions at each state, compute realized utility using
the same MetareasoningUtility as the original causal data collection.

Append to original causal dataset and retrain with same GBT, same hyperparameters.
Then rerun separation and authority gates.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from hrm_adaptive_memory.cognitive_control.core import DecisionAction
from hrm_adaptive_memory.cognitive_control.state import VerificationState, TemporalStatus
from hrm_adaptive_memory.executive.evidence_benchmark.schema import (
    EvidenceTask, EvidenceHypothesis, EvidenceItem,
)
from hrm_adaptive_memory.executive.evidence_benchmark import initial_evidence_runtime
from hrm_adaptive_memory.executive.evidence_benchmark.executor import (
    EvidenceExecutor, valid_verify_targets,
)
from hrm_adaptive_memory.executive.evidence_benchmark.i3_5_confirmation_generator import (
    CONFIRMATION_BUDGET_PROFILES,
)
from hrm_adaptive_memory.executive.resources import ResourceBudget, ResourceState
from hrm_adaptive_memory.executive.metareasoning_utility import MetareasoningUtility
from daph.intervention.checkpoint import compute_state_features

OUTPUT_DIR = REPO_ROOT / "experiments/i3_28b"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CAUSAL_DATA_PATH = REPO_ROOT / "experiments/i3_5/pinned_policy/pinned_causal_actions_v1.jsonl"
CHECKPOINTS_PATH = REPO_ROOT / "experiments/i3_5/datasets/checkpoints_v1.jsonl"
UTILITY_CONFIG_PATH = REPO_ROOT / "configs" / "v2b_i3_1_utility_v1.json"

GBT_PARAMS = dict(n_estimators=200, max_depth=4, random_state=42)
AUTHORITY_THRESHOLD = 5.0
SEPARATION_MARGIN = 5.0
N_BOUNDARY_PAIRS = 40

V2B_ACTION_NAMES = ["ANSWER", "DEFER", "RETRIEVE", "VERIFY", "SEARCH_MORE", "REASON_MORE", "STOP"]
V2B_ACTIONS = [DecisionAction(a) for a in V2B_ACTION_NAMES]

from run_i3_28_rep_repair import (
    extract_v1_features,
    compute_structural_features,
    extract_v2r_features,
    get_v1_feature_keys,
    get_v2r_feature_keys,
    predict_q,
    predict_all_q,
    sha256_bytes,
)

UTILITY = MetareasoningUtility.from_file(UTILITY_CONFIG_PATH)

DOMAIN_TEMPLATES = [
    ("stroke", "ischemic stroke", "hemorrhagic stroke",
     "CT scan shows no hemorrhage", "CT scan shows acute hemorrhage",
     "Is the patient experiencing an ischemic or hemorrhagic stroke?"),
    ("sepsis", "gram-negative sepsis", "gram-positive sepsis",
     "Blood culture grows gram-negative rods", "Blood culture grows gram-positive cocci",
     "Is the patient infected with gram-negative or gram-positive sepsis?"),
    ("diabetes", "type 1 diabetes", "type 2 diabetes",
     "C-peptide levels are undetectable", "C-peptide levels are within normal range",
     "Does the patient have type 1 or type 2 diabetes?"),
    ("cardiac", "NSTEMI", "STEMI",
     "ECG shows ST depression without elevation", "ECG shows ST elevation in leads V1-V4",
     "Is the patient experiencing NSTEMI or STEMI?"),
    ("infection", "viral pneumonia", "bacterial pneumonia",
     "Sputum culture shows no bacterial growth", "Sputum culture grows Streptococcus pneumoniae",
     "Does the patient have viral or bacterial pneumonia?"),
    ("anemia", "iron deficiency anemia", "thalassemia trait",
     "Serum ferritin is low", "Serum ferritin is normal with microcytosis",
     "Does the patient have iron deficiency anemia or thalassemia trait?"),
    ("thyroid", "hyperthyroidism", "hypothyroidism",
     "TSH is suppressed with high T4", "TSH is elevated with low T4",
     "Does the patient have hyperthyroidism or hypothyroidism?"),
    ("renal", "acute kidney injury", "chronic kidney disease",
     "Creatinine rose 2x in 48 hours", "Creatinine stable at 3.5 for months",
     "Does the patient have AKI or CKD?"),
    ("liver", "viral hepatitis", "autoimmune hepatitis",
     "Viral serology is positive for HAV", "ANA and smooth muscle antibody are positive",
     "Does the patient have viral or autoimmune hepatitis?"),
    ("cancer", "small cell lung cancer", "non-small cell lung cancer",
     "Biopsy shows small blue cells", "Biopsy shows adenocarcinoma pattern",
     "Does the patient have small cell or non-small cell lung cancer?"),
]


def make_boundary_task(task_id, domain, safe_defer, seed):
    rng = random.Random(seed)
    _, h1_prop, h2_prop, e1_prop, e2_prop, summary = domain

    h1 = EvidenceHypothesis("H1", h1_prop, DecisionAction.ANSWER, f"ANSWER:H1:{h1_prop}")
    h2 = EvidenceHypothesis("H2", h2_prop, DecisionAction.DEFER, f"DEFER:H2:{h2_prop}")

    if safe_defer:
        e1 = EvidenceItem("E1", e1_prop, "initial", ("H1",), (),
                          VerificationState.UNVERIFIED, TemporalStatus.CURRENT, True, "SUFFICIENT")
        e2 = EvidenceItem("E2", f"Evidence contradicting {h2_prop}", "initial", (), ("H2",),
                          VerificationState.UNVERIFIED, TemporalStatus.CURRENT, True, "FALSIFIED")
        oracle = ("DEFER",)
        expected_terminal = DecisionAction.DEFER
        correct_hyp = "H2"
    else:
        e1 = EvidenceItem("E1", e1_prop, "initial", ("H1",), (),
                          VerificationState.UNVERIFIED, TemporalStatus.CURRENT, True, "SUFFICIENT")
        e2 = EvidenceItem("E2", e2_prop, "initial", ("H2",), (),
                          VerificationState.UNVERIFIED, TemporalStatus.CURRENT, True, "FALSIFIED")
        oracle = ("VERIFY:E1", "VERIFY:E2", "ANSWER")
        expected_terminal = DecisionAction.ANSWER
        correct_hyp = "H1"

    return EvidenceTask(
        task_id=task_id, split="i3_28b_boundary",
        category="boundary_safe_defer" if safe_defer else "boundary_unsafe_defer",
        task_summary=summary, high_stakes=True,
        budget_profile="TIGHT_NO_RETRIEVE_NO_SEARCH",
        hypotheses=(h1, h2), evidence_items=(e1, e2),
        retrieve_exposes=(), search_exposes=(),
        oracle_resolution_path=oracle,
        expected_terminal=expected_terminal,
        correct_hypothesis_id=correct_hyp,
    )


def generate_boundary_pairs(n_pairs, seed=42):
    pairs = []
    for i in range(n_pairs):
        domain = DOMAIN_TEMPLATES[i % len(DOMAIN_TEMPLATES)]
        safe = make_boundary_task(f"i3_28b_safe_{i:04d}", domain, True, seed + i)
        unsafe = make_boundary_task(f"i3_28b_unsafe_{i:04d}", domain, False, seed + i + 1000)
        pairs.append((safe, unsafe))
    return pairs


def compute_realized_utility(task, forced_action, runtime_before, exec_result, downstream_runtime=None):
    """Compute realized utility using MetareasoningUtility.

    For terminal forced actions: terminal_reward(action, success) - action_cost
    For non-terminal: follow oracle, sum step costs, then terminal_reward
    """
    resources_before = runtime_before.resources
    resources_after = exec_result.runtime.resources if not exec_result.terminal else resources_before

    if exec_result.terminal:
        tr = UTILITY.terminal_reward(forced_action, bool(exec_result.task_success))
        step_cost = UTILITY.action_cost(resources_before, exec_result.runtime.resources)
        return float(tr - step_cost), bool(exec_result.task_success)

    # Non-terminal: continue with oracle policy
    executor = EvidenceExecutor()
    current = exec_result.runtime
    total_utility = 0.0
    step_cost = UTILITY.action_cost(resources_before, current.resources)
    total_utility -= step_cost

    oracle_path = list(task.oracle_resolution_path)
    # Follow remaining oracle steps
    for step_spec in oracle_path:
        parts = step_spec.split(":")
        action = DecisionAction(parts[0])
        target = parts[1] if len(parts) > 1 else None

        if action == forced_action:
            continue  # already done

        if action == DecisionAction.VERIFY:
            valid = valid_verify_targets(current)
            if valid:
                target = target or valid[0]
            # Skip if already verified
            if target:
                for ev in current.visible_evidence:
                    if ev.evidence_id == target and ev.verification_state != VerificationState.UNVERIFIED:
                        action = None
                        break
                if action is None:
                    continue

        try:
            res_before = current.resources
            res = executor.execute(current, action, target_evidence_id=target)
            current = res.runtime
            total_utility -= UTILITY.action_cost(res_before, current.resources)

            if res.terminal:
                tr = UTILITY.terminal_reward(action, bool(res.task_success))
                total_utility += tr
                return float(total_utility), bool(res.task_success)
        except:
            break

    # If not terminal after oracle, try expected terminal
    try:
        res = executor.execute(current, task.expected_terminal)
        total_utility -= UTILITY.action_cost(current.resources, res.runtime.resources)
        tr = UTILITY.terminal_reward(task.expected_terminal, bool(res.task_success))
        total_utility += tr
        return float(total_utility), bool(res.task_success)
    except:
        return float(total_utility - 30), False


def force_and_collect(task, forced_action):
    """Force an action at initial state and collect causal outcome."""
    bp = CONFIRMATION_BUDGET_PROFILES[task.budget_profile]
    budget = ResourceBudget(
        max_executive_steps=bp["max_executive_steps"],
        max_retrieval_calls=bp["max_retrieval_calls"],
        max_verification_calls=bp["max_verification_calls"],
        max_search_calls=bp["max_search_calls"],
        max_reasoning_tokens=bp.get("max_reasoning_tokens", 256),
        max_elapsed_ms=bp.get("max_elapsed_ms", 10_000),
    )
    runtime = initial_evidence_runtime(task, ResourceState(budget=budget))
    executor = EvidenceExecutor()

    sf = compute_state_features(runtime, ())
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
    legal = [a.value for a in V2B_ACTIONS if runtime.resources.can_execute(a)]

    target = None
    if forced_action == DecisionAction.VERIFY:
        valid = valid_verify_targets(runtime)
        if valid:
            target = valid[0]

    try:
        res = executor.execute(runtime, forced_action, target_evidence_id=target)
        utility, success = compute_realized_utility(task, forced_action, runtime, res)
        return {
            "forced_action": forced_action.value,
            "pinned_policy_utility": round(utility, 4),
            "pinned_policy_success": success,
            "state_features": sf,
            "structural_features": structural,
            "legal_actions": legal,
        }
    except Exception as e:
        return {
            "forced_action": forced_action.value,
            "pinned_policy_utility": -30.0,
            "pinned_policy_success": False,
            "state_features": sf,
            "structural_features": structural,
            "legal_actions": legal,
            "error": str(e),
        }


def collect_boundary_data(pairs):
    print("\n=== Collecting boundary causal data ===")
    records = []
    actions_to_force = [
        DecisionAction.DEFER,
        DecisionAction.VERIFY,
        DecisionAction.ANSWER,
        DecisionAction.REASON_MORE,
        DecisionAction.STOP,
    ]

    for i, (safe_task, unsafe_task) in enumerate(pairs):
        for task, label in [(safe_task, "safe"), (unsafe_task, "unsafe")]:
            for action in actions_to_force:
                bp = CONFIRMATION_BUDGET_PROFILES[task.budget_profile]
                budget = ResourceBudget(
                    max_executive_steps=bp["max_executive_steps"],
                    max_retrieval_calls=bp["max_retrieval_calls"],
                    max_verification_calls=bp["max_verification_calls"],
                    max_search_calls=bp["max_search_calls"],
                    max_reasoning_tokens=bp.get("max_reasoning_tokens", 256),
                    max_elapsed_ms=bp.get("max_elapsed_ms", 10_000),
                )
                runtime = initial_evidence_runtime(task, ResourceState(budget=budget))
                if not runtime.resources.can_execute(action):
                    continue

                result = force_and_collect(task, action)
                result["task_id"] = task.task_id
                result["category"] = task.category
                result["boundary_type"] = label
                result["correct_first_action"] = task.oracle_resolution_path[0].split(":")[0]
                result["expected_terminal"] = task.expected_terminal.value
                result["oracle_path"] = list(task.oracle_resolution_path)
                records.append(result)

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(pairs)} pairs, {len(records)} records")

    print(f"  Total boundary records: {len(records)}")
    return records


def boundary_to_causal_format(records):
    causal = []
    for r in records:
        ckpt_id = hashlib.sha256(
            f"{r['task_id']}|{r['forced_action']}|{json.dumps(r['state_features'], sort_keys=True)}".encode()
        ).hexdigest()
        causal.append({
            "checkpoint_id": ckpt_id,
            "task_id": r["task_id"],
            "category": r["category"],
            "forced_action": r["forced_action"],
            "state_features": r["state_features"],
            "structural_features": r["structural_features"],
            "pinned_policy_utility": r["pinned_policy_utility"],
            "pinned_policy_success": r["pinned_policy_success"],
            "correct_first_action": r["correct_first_action"],
            "expected_terminal": r["expected_terminal"],
            "boundary_type": r.get("boundary_type", "unknown"),
            "legal_actions": r["legal_actions"],
            "source": "i3_28b_boundary",
        })
    return causal


def main():
    print("=" * 70)
    print("I3.28B: DEFER Boundary Causal Repair")
    print("=" * 70)

    # Step 1: Generate boundary pairs
    print("\n=== Step 1: Generate matched boundary state pairs ===")
    pairs = generate_boundary_pairs(N_BOUNDARY_PAIRS, seed=42)
    print(f"  Generated {len(pairs)} matched pairs ({len(pairs) * 2} states)")

    # Step 2: Collect boundary data
    boundary_records = collect_boundary_data(pairs)
    boundary_causal = boundary_to_causal_format(boundary_records)
    boundary_path = OUTPUT_DIR / "boundary_causal_actions_v1.jsonl"
    with open(boundary_path, "w") as f:
        for r in boundary_causal:
            f.write(json.dumps(r, default=str) + "\n")
    print(f"  Saved: {boundary_path}")

    # Analyze
    print("\n=== Boundary data analysis ===")
    safe = [r for r in boundary_causal if r["boundary_type"] == "safe"]
    unsafe = [r for r in boundary_causal if r["boundary_type"] == "unsafe"]
    for label, recs in [("SAFE (has_competing=0)", safe), ("UNSAFE (has_competing=1)", unsafe)]:
        print(f"\n  {label}:")
        by_action = defaultdict(list)
        for r in recs:
            by_action[r["forced_action"]].append(r["pinned_policy_utility"])
        for action, utils in sorted(by_action.items()):
            print(f"    {action}: n={len(utils)}, mean={np.mean(utils):.2f}, "
                  f"range=[{np.min(utils):.2f}, {np.max(utils):.2f}]")

    # Step 3: Load original + append boundary
    print("\n=== Step 3: Append to original causal dataset ===")
    original = []
    with open(CAUSAL_DATA_PATH) as f:
        for line in f:
            original.append(json.loads(line))

    checkpoints = {}
    with open(CHECKPOINTS_PATH) as f:
        for line in f:
            r = json.loads(line)
            checkpoints[r["checkpoint_id"]] = r

    for r in original:
        ckpt = checkpoints.get(r["checkpoint_id"])
        if ckpt:
            r["structural_features"] = compute_structural_features(ckpt["evidence"])
        else:
            r["structural_features"] = {"n_hyp_unverified_support": 0, "n_hyp_unverified_contradiction": 0, "has_competing_unverified_support": 0}

    combined = original + boundary_causal
    print(f"  Original: {len(original)}, Boundary: {len(boundary_causal)}, Combined: {len(combined)}")

    # Step 4: Train
    print("\n=== Step 4: Train Q_V2R on combined data ===")
    v1_keys = get_v1_feature_keys()
    v2r_keys = get_v2r_feature_keys()

    X_v2r, X_v1, y = [], [], []
    for r in combined:
        sf = r["state_features"]
        action = r["forced_action"]
        structural = r.get("structural_features", {"n_hyp_unverified_support": 0, "n_hyp_unverified_contradiction": 0, "has_competing_unverified_support": 0})
        v2r_feats = extract_v2r_features(sf, action, structural)
        v1_feats = extract_v1_features(sf, action)
        X_v2r.append([v2r_feats[k] for k in v2r_keys])
        X_v1.append([v1_feats[k] for k in v1_keys])
        y.append(r["pinned_policy_utility"])

    X_v2r, X_v1, y = np.array(X_v2r), np.array(X_v1), np.array(y)
    print(f"  Training: {len(y)} records, target range [{y.min():.2f}, {y.max():.2f}]")

    q_v2r = GradientBoostingRegressor(**GBT_PARAMS)
    q_v2r.fit(X_v2r, y)
    v2r_pkl = pickle.dumps(q_v2r)
    v2r_sha = sha256_bytes(v2r_pkl)
    (OUTPUT_DIR / "Q_V2R_boundary_repaired.pkl").write_bytes(v2r_pkl)
    print(f"  Q_V2R SHA: {v2r_sha[:16]}...")

    q_v1 = GradientBoostingRegressor(**GBT_PARAMS)
    q_v1.fit(X_v1, y)
    v1_pkl = pickle.dumps(q_v1)
    v1_sha = sha256_bytes(v1_pkl)
    (OUTPUT_DIR / "Q_V1_boundary_control.pkl").write_bytes(v1_pkl)
    print(f"  Q_V1 SHA: {v1_sha[:16]}...")

    # Step 5: Separation audit
    print("\n=== Step 5: Offline separation audit ===")

    # In-sample (boundary states)
    safe_defer_q, unsafe_defer_q = [], []
    for r in boundary_causal:
        sf = r["state_features"]
        structural = r["structural_features"]
        legal = r["legal_actions"]
        q_vals = predict_all_q(q_v2r, v2r_keys, sf, legal, structural)
        qd = q_vals.get("DEFER", 0)
        if r["boundary_type"] == "safe":
            safe_defer_q.append(qd)
        else:
            unsafe_defer_q.append(qd)

    # Out-of-sample (I3.26 dev)
    from hrm_adaptive_memory.executive.evidence_benchmark.i3_26_development_generator import generate_development_benchmark
    dev_tasks = generate_development_benchmark(seed=7719)
    dev_defer = [t for t in dev_tasks if t.category == "defer"]
    dev_contra = [t for t in dev_tasks if t.category == "contradiction"]

    def get_dev_state(task, prior=()):
        bp = CONFIRMATION_BUDGET_PROFILES[task.budget_profile]
        budget = ResourceBudget(
            max_executive_steps=bp["max_executive_steps"],
            max_retrieval_calls=bp["max_retrieval_calls"],
            max_verification_calls=bp["max_verification_calls"],
            max_search_calls=bp["max_search_calls"],
            max_reasoning_tokens=bp.get("max_reasoning_tokens", 256),
            max_elapsed_ms=bp.get("max_elapsed_ms", 10_000),
        )
        runtime = initial_evidence_runtime(task, ResourceState(budget=budget))
        for a in prior:
            try:
                res = EvidenceExecutor().execute(runtime, DecisionAction(a))
                runtime = res.runtime
                if res.terminal:
                    return None, None, None
            except:
                return None, None, None
        sf = compute_state_features(runtime, tuple(prior))
        visible_ev = [{"evidence_id": ev.evidence_id, "supports": list(ev.supports),
                       "contradicts": list(ev.contradicts),
                       "verification_state": ev.verification_state.name,
                       "retrieved": ev.retrieved} for ev in runtime.visible_evidence]
        structural = compute_structural_features(visible_ev)
        legal = [a.value for a in V2B_ACTIONS if runtime.resources.can_execute(a)]
        return sf, structural, legal

    dev_safe_q, dev_unsafe_q = [], []
    for task in dev_defer:
        sf, st, legal = get_dev_state(task)
        if sf:
            dev_safe_q.append(predict_all_q(q_v2r, v2r_keys, sf, legal, st).get("DEFER", 0))
    for task in dev_contra:
        sf, st, legal = get_dev_state(task)
        if sf:
            dev_unsafe_q.append(predict_all_q(q_v2r, v2r_keys, sf, legal, st).get("DEFER", 0))
        sf, st, legal = get_dev_state(task, ["SEARCH_MORE"])
        if sf:
            dev_unsafe_q.append(predict_all_q(q_v2r, v2r_keys, sf, legal, st).get("DEFER", 0))

    print(f"\n  Boundary (in-sample):")
    print(f"    Safe Q(DEFER): mean={np.mean(safe_defer_q):.2f}, range=[{np.min(safe_defer_q):.2f}, {np.max(safe_defer_q):.2f}]")
    print(f"    Unsafe Q(DEFER): mean={np.mean(unsafe_defer_q):.2f}, range=[{np.min(unsafe_defer_q):.2f}, {np.max(unsafe_defer_q):.2f}]")
    margin_b = min(safe_defer_q) - max(unsafe_defer_q) if safe_defer_q and unsafe_defer_q else -999
    print(f"    Margin: {margin_b:.2f} (required >{SEPARATION_MARGIN})")

    print(f"\n  I3.26 dev (out-of-sample):")
    print(f"    Safe Q(DEFER): mean={np.mean(dev_safe_q):.2f}, range=[{np.min(dev_safe_q):.2f}, {np.max(dev_safe_q):.2f}]")
    print(f"    Unsafe Q(DEFER): mean={np.mean(dev_unsafe_q):.2f}, range=[{np.min(dev_unsafe_q):.2f}, {np.max(dev_unsafe_q):.2f}]")
    margin_d = min(dev_safe_q) - max(dev_unsafe_q) if dev_safe_q and dev_unsafe_q else -999
    print(f"    Margin: {margin_d:.2f} (required >{SEPARATION_MARGIN})")

    # Step 6: Authority coverage
    print("\n=== Step 6: Authority coverage ===")
    safe_auth, safe_tot, unsafe_auth, unsafe_tot, unsafe_false = 0, 0, 0, 0, 0
    for r in boundary_causal:
        sf = r["state_features"]
        structural = r["structural_features"]
        legal = r["legal_actions"]
        q_vals = predict_all_q(q_v2r, v2r_keys, sf, legal, structural)
        q_sorted = sorted(q_vals.items(), key=lambda x: -x[1])
        q_best, q_second = q_sorted[0], q_sorted[1] if len(q_sorted) > 1 else ("", 0)
        gap = q_best[1] - q_second[1]
        near_opt = [a for a, q in q_vals.items() if q >= q_best[1] - 3.0]
        no_competing = structural["has_competing_unverified_support"] == 0
        has_contradiction = structural["n_hyp_unverified_contradiction"] > 0
        would_force = (q_best[0] == "DEFER" and gap >= AUTHORITY_THRESHOLD
                      and len(near_opt) == 1 and near_opt[0] == "DEFER"
                      and no_competing and has_contradiction)
        if r["boundary_type"] == "safe":
            safe_tot += 1
            if would_force: safe_auth += 1
        else:
            unsafe_tot += 1
            if would_force:
                unsafe_auth += 1
                unsafe_false += 1

    print(f"  Safe: {safe_auth}/{safe_tot} trigger DEFER authority")
    print(f"  Unsafe: {unsafe_auth}/{unsafe_tot} trigger DEFER authority (all false)")
    false_rate = unsafe_false / max(unsafe_tot, 1)

    # Step 7: Preservation on 220 checkpoints
    print("\n=== Step 7: Preservation on 220 checkpoints ===")
    unique_ckpts = {}
    for r in original:
        cid = r["checkpoint_id"]
        if cid not in unique_ckpts:
            unique_ckpts[cid] = {
                "state_features": r["state_features"],
                "structural": r["structural_features"],
                "correct_first_action": r["correct_first_action"],
                "category": r["category"],
            }

    ACTIONS = V2B_ACTION_NAMES
    answer_correct, answer_total = 0, 0
    v1_reg, v2r_reg, v1_no, v2r_no, v1_cb, v2r_cb = [], [], 0, 0, 0, 0
    n = len(unique_ckpts)

    for cid, info in unique_ckpts.items():
        sf = info["state_features"]
        structural = info["structural"]
        correct = info["correct_first_action"]
        q_v1_vals = predict_all_q(q_v1, v1_keys, sf, ACTIONS)
        q_v2r_vals = predict_all_q(q_v2r, v2r_keys, sf, ACTIONS, structural)
        v1_best = max(q_v1_vals, key=q_v1_vals.get)
        v2r_best = max(q_v2r_vals, key=q_v2r_vals.get)
        v1_qm = max(q_v1_vals.values())
        v2r_qm = max(q_v2r_vals.values())
        r1 = v1_qm - q_v1_vals.get(correct, -999)
        r2 = v2r_qm - q_v2r_vals.get(correct, -999)
        v1_reg.append(r1)
        v2r_reg.append(r2)
        if r1 <= 3.0: v1_no += 1
        if r2 <= 3.0: v2r_no += 1
        if v1_best == correct: v1_cb += 1
        if v2r_best == correct: v2r_cb += 1
        if correct == "ANSWER":
            answer_total += 1
            # Preservation check: V2R must still rank ANSWER as best
            # (gap threshold is not the right metric on I3.5 checkpoints
            # where n_verified=2 makes both ANSWER and VERIFY high-value)
            v1_best_for_answer = max(q_v1_vals, key=q_v1_vals.get)
            v2r_best_for_answer = max(q_v2r_vals, key=q_v2r_vals.get)
            if v2r_best_for_answer == "ANSWER":
                answer_correct += 1

    print(f"  Mean regret: V1={np.mean(v1_reg):.4f}, V2R={np.mean(v2r_reg):.4f}")
    print(f"  Near-optimal: V1={v1_no}/{n}, V2R={v2r_no}/{n}")
    print(f"  Correct best: V1={v1_cb}/{n}, V2R={v2r_cb}/{n}")
    print(f"  ANSWER authority: {answer_correct}/{answer_total}")

    # Gates
    gates = {
        "separation_boundary": margin_b > SEPARATION_MARGIN,
        "separation_dev": margin_d > SEPARATION_MARGIN,
        "false_defer_authority_zero": false_rate == 0,
        "answer_authority_preserved": answer_correct >= answer_total * 0.95,
        "regret_no_worse": np.mean(v2r_reg) <= np.mean(v1_reg) * 1.05,
        "near_optimal_no_worse": v2r_no >= v1_no,
        "correct_best_no_regression": v2r_cb >= int(v1_cb * 0.95),
    }

    print("\n" + "=" * 70)
    print("I3.28B SUMMARY")
    print("=" * 70)
    for g, v in gates.items():
        print(f"  {g}: {'PASS' if v else 'FAIL'}")
    all_pass = all(gates.values())
    print(f"\n  OVERALL: {'PASS' if all_pass else 'FAIL'}")

    results = {
        "experiment": "I3.28B: DEFER Boundary Causal Repair",
        "n_boundary_pairs": N_BOUNDARY_PAIRS,
        "n_boundary_records": len(boundary_causal),
        "n_combined_records": len(combined),
        "q_v1_sha256": v1_sha,
        "q_v2r_sha256": v2r_sha,
        "separation_margin_boundary": float(margin_b),
        "separation_margin_dev": float(margin_d),
        "required_margin": SEPARATION_MARGIN,
        "false_defer_authority_rate": float(false_rate),
        "answer_authority": f"{answer_correct}/{answer_total}",
        "preservation": {
            "v1_mean_regret": float(np.mean(v1_reg)),
            "v2r_mean_regret": float(np.mean(v2r_reg)),
            "v1_near_optimal": int(v1_no),
            "v2r_near_optimal": int(v2r_no),
            "v1_correct_best": int(v1_cb),
            "v2r_correct_best": int(v2r_cb),
        },
        "gates": {k: bool(v) for k, v in gates.items()},
        "all_pass": bool(all_pass),
    }
    with open(OUTPUT_DIR / "full_results.json", "w") as f:
        json.dump(results, f, indent=2)

    if all_pass:
        print("\n  RECOMMENDATION: Proceed to live validation sequence.")
    else:
        print("\n  RECOMMENDATION: Do NOT proceed to live DEFER authority.")


if __name__ == "__main__":
    main()
