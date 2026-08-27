#!/usr/bin/env python3
"""I3.28C: Targeted DEFER Coverage Audit.

Test whether DEFER hard authority can have meaningful coverage on safe states
where continuation actions are structurally dominated, while remaining blocked
on unsafe states.

Three frozen strata:
  D1: safe DEFER, VERIFY unavailable (budget has max_verification_calls=0)
  D2: safe DEFER, verification completed (replay VERIFY first, then evaluate)
  D3: unsafe contradiction/competing-support (DEFER must never get authority)

Structural variation within each stratum:
  - 2-3 hypotheses
  - 2-4 visible evidence items
  - varied remaining budgets (steps, reasoning tokens)
  - 10 domain templates
  - varied evidence support/contradiction patterns
  - varied which hypothesis is correct

The model must learn "DEFER is clearly best under this epistemic/resource
condition," not "remaining_steps=1 means DEFER."

Same repaired representation (Q_STATE_SCHEMA_V2), same GBT, same hyperparameters,
same threshold 5.0, same asymmetric structural predicate.

Metrics:
  Coverage_DEFER = P(force DEFER | D1 ∪ D2)   target: materially > 0
  FalseAuthorityRate_DEFER = P(force DEFER | D3)  target: 0
  ANSWER authority preserved
  Regret, near-optimal, correct-best non-inferior
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

OUTPUT_DIR = REPO_ROOT / "experiments/i3_28c"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CAUSAL_DATA_PATH = REPO_ROOT / "experiments/i3_5/pinned_policy/pinned_causal_actions_v1.jsonl"
BOUNDARY_DATA_PATH = REPO_ROOT / "experiments/i3_28b/boundary_causal_actions_v1.jsonl"
CHECKPOINTS_PATH = REPO_ROOT / "experiments/i3_5/datasets/checkpoints_v1.jsonl"
UTILITY_CONFIG_PATH = REPO_ROOT / "configs" / "v2b_i3_1_utility_v1.json"

GBT_PARAMS = dict(n_estimators=200, max_depth=4, random_state=42)
AUTHORITY_THRESHOLD = 5.0
SEPARATION_MARGIN = 5.0

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

# ============================================================
# Domain templates for structural variation
# ============================================================

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

# ============================================================
# Budget configurations for structural variation
# ============================================================

# D1 budgets: VERIFY unavailable, varied steps/reasoning
D1_BUDGETS = [
    # (max_steps, max_verify, max_reasoning, label)
    (1, 0, 0, "D1_1step_noreason"),
    (1, 0, 128, "D1_1step_reason"),
    (2, 0, 0, "D1_2step_noreason"),
    (2, 0, 128, "D1_2step_reason"),
    (3, 0, 0, "D1_3step_noreason"),
    (3, 0, 128, "D1_3step_reason"),
]

# D2 budgets: verification available but we replay VERIFY first
# After VERIFY, remaining steps vary
D2_BUDGETS = [
    # (max_steps, max_verify, max_reasoning, label)
    # After 1 VERIFY step, remaining = max_steps - 1
    (2, 1, 0, "D2_1step_left_noreason"),     # 1 step left after verify
    (2, 1, 128, "D2_1step_left_reason"),
    (3, 1, 0, "D2_2step_left_noreason"),     # 2 steps left after verify
    (3, 1, 128, "D2_2step_left_reason"),
    (3, 2, 0, "D2_2verify_1step_noreason"),  # 1 step left, 2 verify available
    (4, 2, 128, "D2_2verify_2step_reason"),  # 2 steps left after 1 verify
]

# D3 budgets: same variety as D1/D2 but unsafe states
D4_BUDGETS = [
    # ANSWER-correct states with varied budgets
    (1, 1, 0, "D4_1step"),
    (1, 1, 128, "D4_1step_reason"),
    (2, 1, 0, "D4_2step"),
    (2, 1, 128, "D4_2step_reason"),
    (3, 1, 0, "D4_3step"),
    (3, 1, 128, "D4_3step_reason"),
]

D3_BUDGETS = [
    (1, 0, 0, "D3_1step_noreason"),
    (1, 0, 128, "D3_1step_reason"),
    (2, 1, 128, "D3_2step_verify"),
    (3, 1, 128, "D3_3step_verify"),
    (3, 0, 0, "D3_3step_noverify"),
    (2, 0, 128, "D3_2step_noverify"),
]


# ============================================================
# Task generation with structural variation
# ============================================================

def make_safe_defer_task(task_id, domain, n_hypotheses, n_evidence, seed,
                         correct_hyp_idx=1):
    """Create a safe-defer task (has_competing=0).

    Evidence pattern: some evidence supports H_correct, some contradicts H_wrong.
    No competing support chains.
    correct_hyp_idx: which hypothesis is correct (0=H1, 1=H2, 2=H3)
    """
    rng = random.Random(seed)
    _, *hyps_and_evidence, summary = domain

    # Build hypotheses
    hyps = []
    for i in range(n_hypotheses):
        h_prop = hyps_and_evidence[i]
        if i == correct_hyp_idx:
            action = DecisionAction.DEFER
            payload = f"DEFER:H{i+1}:{h_prop}"
        else:
            action = DecisionAction.ANSWER
            payload = f"ANSWER:H{i+1}:{h_prop}"
        hyps.append(EvidenceHypothesis(f"H{i+1}", h_prop, action, payload))

    # Build evidence: support correct hyp, contradict wrong hyps
    # No competing support (only one hyp gets support)
    evidence = []
    correct_hyp = f"H{correct_hyp_idx+1}"
    wrong_hyps = [f"H{i+1}" for i in range(n_hypotheses) if i != correct_hyp_idx]

    # Assign evidence items
    e_idx = 0
    # Some evidence supports correct hyp
    n_support = rng.randint(1, max(1, n_evidence // 2))
    for _ in range(n_support):
        if e_idx >= n_evidence:
            break
        evidence.append(EvidenceItem(
            f"E{e_idx+1}",
            f"Evidence supporting {correct_hyp}",
            "initial",
            (correct_hyp,), (),
            VerificationState.UNVERIFIED, TemporalStatus.CURRENT, True,
            "SUFFICIENT",  # oracle
        ))
        e_idx += 1

    # Some evidence contradicts wrong hyps
    for _ in range(n_evidence - e_idx):
        if e_idx >= n_evidence:
            break
        wrong = rng.choice(wrong_hyps) if wrong_hyps else None
        if wrong:
            evidence.append(EvidenceItem(
                f"E{e_idx+1}",
                f"Evidence contradicting {wrong}",
                "initial",
                (), (wrong,),
                VerificationState.UNVERIFIED, TemporalStatus.CURRENT, True,
                "FALSIFIED",  # oracle
            ))
        else:
            evidence.append(EvidenceItem(
                f"E{e_idx+1}",
                f"Evidence supporting {correct_hyp}",
                "initial",
                (correct_hyp,), (),
                VerificationState.UNVERIFIED, TemporalStatus.CURRENT, True,
                "SUFFICIENT",
            ))
        e_idx += 1

    return EvidenceTask(
        task_id=task_id, split="i3_28c",
        category="D1_safe_defer",
        task_summary=summary, high_stakes=True,
        budget_profile="TIGHT_NO_RETRIEVE_NO_SEARCH",  # will be overridden
        hypotheses=tuple(hyps), evidence_items=tuple(evidence),
        retrieve_exposes=(), search_exposes=(),
        oracle_resolution_path=("DEFER",),
        expected_terminal=DecisionAction.DEFER,
        correct_hypothesis_id=correct_hyp,
    )


def make_unsafe_contradiction_task(task_id, domain, n_hypotheses, n_evidence, seed,
                                    correct_hyp_idx=0):
    """Create an unsafe contradiction task (has_competing=1).

    Evidence pattern: competing support for multiple hypotheses.
    DEFER is wrong; VERIFY then ANSWER is correct.
    """
    rng = random.Random(seed)
    _, *hyps_and_evidence, summary = domain

    hyps = []
    for i in range(n_hypotheses):
        h_prop = hyps_and_evidence[i] if i < len(hyps_and_evidence) else f"Condition {i+1}"
        if i == correct_hyp_idx:
            action = DecisionAction.ANSWER
            payload = f"ANSWER:H{i+1}:{h_prop}"
        else:
            action = DecisionAction.DEFER
            payload = f"DEFER:H{i+1}:{h_prop}"
        hyps.append(EvidenceHypothesis(f"H{i+1}", h_prop, action, payload))

    # Build evidence: competing support for H1 and H2
    evidence = []
    correct_hyp = f"H{correct_hyp_idx+1}"
    competing_hyp = f"H{(correct_hyp_idx + 1) % n_hypotheses + 1}"

    e_idx = 0
    # Evidence supporting correct hyp (will verify as SUFFICIENT)
    n_support_correct = rng.randint(1, max(1, n_evidence // 2))
    for _ in range(n_support_correct):
        if e_idx >= n_evidence:
            break
        evidence.append(EvidenceItem(
            f"E{e_idx+1}",
            f"Evidence supporting {correct_hyp}",
            "initial",
            (correct_hyp,), (),
            VerificationState.UNVERIFIED, TemporalStatus.CURRENT, True,
            "SUFFICIENT",
        ))
        e_idx += 1

    # Evidence supporting competing hyp (will verify as FALSIFIED)
    for _ in range(n_evidence - e_idx):
        if e_idx >= n_evidence:
            break
        evidence.append(EvidenceItem(
            f"E{e_idx+1}",
            f"Evidence supporting {competing_hyp}",
            "initial",
            (competing_hyp,), (),
            VerificationState.UNVERIFIED, TemporalStatus.CURRENT, True,
            "FALSIFIED",
        ))
        e_idx += 1

    # Build oracle: verify correct evidence, verify competing evidence, answer
    oracle = []
    for ev in evidence:
        if correct_hyp in ev.supports:
            oracle.append(f"VERIFY:{ev.evidence_id}")
            break
    for ev in evidence:
        if competing_hyp in ev.supports:
            oracle.append(f"VERIFY:{ev.evidence_id}")
            break
    oracle.append("ANSWER")

    return EvidenceTask(
        task_id=task_id, split="i3_28c",
        category="D3_unsafe_contradiction",
        task_summary=summary, high_stakes=True,
        budget_profile="TIGHT_NO_RETRIEVE_NO_SEARCH",
        hypotheses=tuple(hyps), evidence_items=tuple(evidence),
        retrieve_exposes=(), search_exposes=(),
        oracle_resolution_path=tuple(oracle),
        expected_terminal=DecisionAction.ANSWER,
        correct_hypothesis_id=correct_hyp,
    )


def make_answer_correct_task(task_id, domain, n_evidence, seed):
    """Create an ANSWER-correct task with n_verified=2.

    Both evidence items are already verified as SUFFICIENT for H1.
    ANSWER is correct. VERIFY is dominated (wastes a step).
    This provides contrast so the GBT doesn't overestimate VERIFY
    at n_verified=2 due to the high-utility VERIFY records in D3.
    """
    rng = random.Random(seed)
    _, h1_prop, h2_prop, e1_prop, e2_prop, summary = domain

    h1 = EvidenceHypothesis("H1", h1_prop, DecisionAction.ANSWER, f"ANSWER:H1:{h1_prop}")
    h2 = EvidenceHypothesis("H2", h2_prop, DecisionAction.DEFER, f"DEFER:H2:{h2_prop}")

    # Both verified as SUFFICIENT for H1
    evidence = []
    for i in range(n_evidence):
        evidence.append(EvidenceItem(
            f"E{i+1}",
            f"Verified evidence supporting H1" if i == 0 else f"Additional verified evidence {i+1}",
            "initial",
            ("H1",), (),
            VerificationState.SUFFICIENT, TemporalStatus.CURRENT, True,
            "SUFFICIENT",
        ))

    return EvidenceTask(
        task_id=task_id, split="i3_28c",
        category="D4_answer_correct",
        task_summary=summary, high_stakes=True,
        budget_profile="TIGHT_NO_RETRIEVE_NO_SEARCH",
        hypotheses=(h1, h2), evidence_items=tuple(evidence),
        retrieve_exposes=(), search_exposes=(),
        oracle_resolution_path=("ANSWER",),
        expected_terminal=DecisionAction.ANSWER,
        correct_hypothesis_id="H1",
    )


def make_budget(max_steps, max_verify, max_reasoning):
    return ResourceBudget(
        max_executive_steps=max_steps,
        max_retrieval_calls=0,
        max_verification_calls=max_verify,
        max_search_calls=0,
        max_reasoning_tokens=max_reasoning,
        max_elapsed_ms=10000,
    )


# ============================================================
# Generate strata
# ============================================================

def generate_strata(n_per_stratum=30, seed=42):
    """Generate D1, D2, D3 strata with structural variation."""
    rng = random.Random(seed)
    strata = {"D1": [], "D2": [], "D3": [], "D4": []}

    for i in range(n_per_stratum):
        domain = DOMAIN_TEMPLATES[i % len(DOMAIN_TEMPLATES)]
        n_hyps = rng.choice([2, 2, 3])  # mostly 2, sometimes 3
        n_ev = rng.choice([2, 2, 3, 4])  # varied evidence counts
        correct_idx = rng.randint(0, n_hyps - 1)

        # D1: safe defer, VERIFY unavailable
        d1_budget = D1_BUDGETS[i % len(D1_BUDGETS)]
        d1_task = make_safe_defer_task(
            f"i3_28c_d1_{i:04d}", domain, n_hyps, n_ev, seed + i, correct_idx)
        strata["D1"].append({"task": d1_task, "budget": d1_budget, "stratum": "D1"})

        # D2: safe defer, verification completed (replay VERIFY first)
        d2_budget = D2_BUDGETS[i % len(D2_BUDGETS)]
        d2_task = make_safe_defer_task(
            f"i3_28c_d2_{i:04d}", domain, n_hyps, n_ev, seed + i + 1000, correct_idx)
        strata["D2"].append({"task": d2_task, "budget": d2_budget, "stratum": "D2"})

        # D3: unsafe contradiction
        d3_budget = D3_BUDGETS[i % len(D3_BUDGETS)]
        d3_task = make_unsafe_contradiction_task(
            f"i3_28c_d3_{i:04d}", domain, n_hyps, n_ev, seed + i + 2000, 0)
        strata["D3"].append({"task": d3_task, "budget": d3_budget, "stratum": "D3"})

        # D4: ANSWER-correct, n_verified=2 (contrast for VERIFY overestimation)
        d4_budget = D4_BUDGETS[i % len(D4_BUDGETS)]
        d4_n_ev = rng.choice([2, 2, 3])  # varied evidence count
        d4_task = make_answer_correct_task(
            f"i3_28c_d4_{i:04d}", domain, d4_n_ev, seed + i + 3000)
        strata["D4"].append({"task": d4_task, "budget": d4_budget, "stratum": "D4"})

    return strata


# ============================================================
# Force actions and collect outcomes
# ============================================================

def get_state_and_collect(task, budget_config, pre_actions=()):
    """Get the state at a given point (after pre_actions), force all legal actions."""
    max_steps, max_verify, max_reasoning, budget_label = budget_config
    budget = make_budget(max_steps, max_verify, max_reasoning)
    runtime = initial_evidence_runtime(task, ResourceState(budget=budget))
    executor = EvidenceExecutor()

    # Execute pre-actions (for D2: VERIFY first)
    for action_name, target_id in pre_actions:
        action = DecisionAction(action_name)
        try:
            res = executor.execute(runtime, action, target_evidence_id=target_id)
            runtime = res.runtime
            if res.terminal:
                return None
        except:
            return None

    # Capture state
    sf = compute_state_features(runtime, tuple(a[0] for a in pre_actions))
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

    return {
        "state_features": sf,
        "structural_features": structural,
        "legal_actions": legal,
        "runtime": runtime,
        "task": task,
        "budget_label": budget_label,
        "pre_actions": [a[0] for a in pre_actions],
    }


def compute_forced_utility(task, runtime_before, forced_action, pre_actions=()):
    """Force an action and compute realized utility."""
    executor = EvidenceExecutor()
    resources_before = runtime_before.resources

    target = None
    if forced_action == DecisionAction.VERIFY:
        valid = valid_verify_targets(runtime_before)
        if valid:
            target = valid[0]

    try:
        res = executor.execute(runtime_before, forced_action, target_evidence_id=target)
        if res.terminal:
            tr = UTILITY.terminal_reward(forced_action, bool(res.task_success))
            cost = UTILITY.action_cost(resources_before, res.runtime.resources)
            return float(tr - cost), bool(res.task_success)

        # Non-terminal: follow oracle from current position
        current = res.runtime
        total = -UTILITY.action_cost(resources_before, current.resources)

        # Follow oracle
        oracle = list(task.oracle_resolution_path)
        # Skip oracle steps already done by pre_actions + forced_action
        done = list(pre_actions) + [(forced_action.value, target)]
        for step_spec in oracle:
            parts = step_spec.split(":")
            action_name = parts[0]
            target_id = parts[1] if len(parts) > 1 else None

            # Skip if already done
            if (action_name, target_id) in done or action_name == forced_action.value:
                continue

            action = DecisionAction(action_name)
            if action == DecisionAction.VERIFY:
                valid = valid_verify_targets(current)
                if valid:
                    target_id = target_id or valid[0]
                else:
                    continue

            try:
                res_before = current.resources
                res2 = executor.execute(current, action, target_evidence_id=target_id)
                current = res2.runtime
                total -= UTILITY.action_cost(res_before, current.resources)
                done.append((action_name, target_id))
                if res2.terminal:
                    tr = UTILITY.terminal_reward(action, bool(res2.task_success))
                    total += tr
                    return float(total), bool(res2.task_success)
            except:
                break

        # Try expected terminal
        try:
            res_before = current.resources
            res3 = executor.execute(current, task.expected_terminal)
            total -= UTILITY.action_cost(res_before, res3.runtime.resources)
            tr = UTILITY.terminal_reward(task.expected_terminal, bool(res3.task_success))
            total += tr
            return float(total), bool(res3.task_success)
        except:
            return float(total - 30), False
    except Exception as e:
        return -30.0, False


def collect_stratum_data(stratum_items):
    """Collect causal data for a stratum."""
    records = []

    for item in stratum_items:
        task = item["task"]
        budget_config = item["budget"]
        stratum = item["stratum"]

        # D2: replay VERIFY first
        pre_actions = ()
        if stratum == "D2":
            # Find first VERIFY target in oracle
            budget = make_budget(*budget_config[:3])
            runtime = initial_evidence_runtime(task, ResourceState(budget=budget))
            valid = valid_verify_targets(runtime)
            if valid:
                pre_actions = [("VERIFY", valid[0])]
            else:
                # No verify possible, skip this D2 item
                continue

        state_info = get_state_and_collect(task, budget_config, pre_actions)
        if state_info is None:
            continue

        sf = state_info["state_features"]
        structural = state_info["structural_features"]
        legal = state_info["legal_actions"]
        runtime = state_info["runtime"]

        for action_name in legal:
            action = DecisionAction(action_name)
            # Recreate runtime for each action (fresh from pre_actions)
            rt_fresh = initial_evidence_runtime(task, ResourceState(
                budget=make_budget(*budget_config[:3])))
            executor = EvidenceExecutor()
            for a_name, t_id in pre_actions:
                try:
                    res = executor.execute(rt_fresh, DecisionAction(a_name), target_evidence_id=t_id)
                    rt_fresh = res.runtime
                except:
                    break

            util, success = compute_forced_utility(task, rt_fresh, action, pre_actions)

            ckpt_id = hashlib.sha256(
                f"{task.task_id}|{action_name}|{json.dumps(sf, sort_keys=True)}".encode()
            ).hexdigest()

            records.append({
                "checkpoint_id": ckpt_id,
                "task_id": task.task_id,
                "category": task.category,
                "forced_action": action_name,
                "state_features": sf,
                "structural_features": structural,
                "pinned_policy_utility": round(util, 4),
                "pinned_policy_success": success,
                "correct_first_action": task.oracle_resolution_path[0].split(":")[0],
                "expected_terminal": task.expected_terminal.value,
                "stratum": stratum,
                "budget_label": budget_config[3],
                "legal_actions": legal,
                "source": "i3_28c",
            })

    return records


# ============================================================
# Main experiment
# ============================================================

def main():
    print("=" * 70)
    print("I3.28C: Targeted DEFER Coverage Audit")
    print("=" * 70)

    # Step 1: Generate strata
    print("\n=== Step 1: Generate D1/D2/D3 strata ===")
    strata = generate_strata(n_per_stratum=30, seed=42)
    for s, items in strata.items():
        print(f"  {s}: {len(items)} states")

    # Step 2: Collect causal data
    print("\n=== Step 2: Collect causal data ===")
    all_records = []
    for s, items in strata.items():
        recs = collect_stratum_data(items)
        print(f"  {s}: {len(recs)} records")
        all_records.extend(recs)
    print(f"  Total: {len(all_records)} records")

    # Save
    strata_path = OUTPUT_DIR / "strata_causal_actions_v1.jsonl"
    with open(strata_path, "w") as f:
        for r in all_records:
            f.write(json.dumps(r, default=str) + "\n")
    print(f"  Saved: {strata_path}")

    # Analyze strata data
    print("\n=== Strata data analysis ===")
    for stratum in ["D1", "D2", "D3", "D4"]:
        recs = [r for r in all_records if r["stratum"] == stratum]
        print(f"\n  {stratum}:")
        by_action = defaultdict(list)
        for r in recs:
            by_action[r["forced_action"]].append(r["pinned_policy_utility"])
        for action, utils in sorted(by_action.items()):
            print(f"    {action}: n={len(utils)}, mean={np.mean(utils):.2f}, "
                  f"range=[{np.min(utils):.2f}, {np.max(utils):.2f}]")

    # Step 3: Load all training data (original + I3.28B boundary + I3.28C strata)
    print("\n=== Step 3: Load and combine training data ===")
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

    boundary = []
    with open(BOUNDARY_DATA_PATH) as f:
        for line in f:
            boundary.append(json.loads(line))

    combined = original + boundary + all_records
    print(f"  Original: {len(original)}")
    print(f"  I3.28B boundary: {len(boundary)}")
    print(f"  I3.28C strata: {len(all_records)}")
    print(f"  Combined: {len(combined)}")

    # Step 4: Train Q_V2R on combined data
    print("\n=== Step 4: Train Q_V2R ===")
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
    (OUTPUT_DIR / "Q_V2R_coverage_repaired.pkl").write_bytes(v2r_pkl)
    print(f"  Q_V2R SHA: {v2r_sha[:16]}...")

    q_v1 = GradientBoostingRegressor(**GBT_PARAMS)
    q_v1.fit(X_v1, y)
    v1_pkl = pickle.dumps(q_v1)
    v1_sha = sha256_bytes(v1_pkl)
    (OUTPUT_DIR / "Q_V1_coverage_control.pkl").write_bytes(v1_pkl)

    # Step 5: Coverage and false authority audit
    print("\n=== Step 5: DEFER coverage and false authority audit ===")

    def check_defer_authority(q_vals, structural):
        """Check if DEFER hard authority would fire."""
        q_sorted = sorted(q_vals.items(), key=lambda x: -x[1])
        q_best = q_sorted[0]
        q_second = q_sorted[1] if len(q_sorted) > 1 else ("", 0)
        gap = q_best[1] - q_second[1]
        near_opt = [a for a, q in q_vals.items() if q >= q_best[1] - 3.0]
        no_competing = structural["has_competing_unverified_support"] == 0
        has_contradiction = structural["n_hyp_unverified_contradiction"] > 0
        would_force = (q_best[0] == "DEFER" and gap >= AUTHORITY_THRESHOLD
                      and len(near_opt) == 1 and near_opt[0] == "DEFER"
                      and no_competing and has_contradiction)
        return would_force, gap, q_best[0]

    # Evaluate on unique states in each stratum
    safe_authority = 0
    safe_total = 0
    unsafe_authority = 0
    unsafe_total = 0
    unsafe_false_authority = 0

    # Collect unique states per stratum
    stratum_states = defaultdict(list)
    for r in all_records:
        # Use state_features + structural as unique key
        key = json.dumps(r["state_features"], sort_keys=True) + json.dumps(r["structural_features"], sort_keys=True)
        if key not in [s["key"] for s in stratum_states[r["stratum"]]]:
            stratum_states[r["stratum"]].append({
                "key": key,
                "state_features": r["state_features"],
                "structural_features": r["structural_features"],
                "legal_actions": r["legal_actions"],
            })

    for stratum in ["D1", "D2", "D3", "D4"]:
        states = stratum_states[stratum]
        authority_count = 0
        for s in states:
            q_vals = predict_all_q(q_v2r, v2r_keys, s["state_features"], s["legal_actions"], s["structural_features"])
            would_force, gap, best = check_defer_authority(q_vals, s["structural_features"])
            if would_force:
                authority_count += 1

        total = len(states)
        if stratum in ["D1", "D2"]:
            safe_authority += authority_count
            safe_total += total
        elif stratum == "D3":
            unsafe_authority += authority_count
            unsafe_total += total
            unsafe_false_authority += authority_count
        # D4 is ANSWER-correct, not counted in DEFER coverage/false authority

        print(f"  {stratum}: {authority_count}/{total} trigger DEFER authority")

    coverage = safe_authority / max(safe_total, 1)
    false_rate = unsafe_false_authority / max(unsafe_total, 1)

    print(f"\n  Coverage_DEFER (D1∪D2): {safe_authority}/{safe_total} = {coverage:.4f}")
    print(f"  FalseAuthorityRate_DEFER (D3): {unsafe_false_authority}/{unsafe_total} = {false_rate:.4f}")

    # Show Q values at sample states
    print("\n  Sample Q values:")
    for stratum in ["D1", "D2", "D3", "D4"]:
        states = stratum_states[stratum][:3]
        for s in states:
            q_vals = predict_all_q(q_v2r, v2r_keys, s["state_features"], s["legal_actions"], s["structural_features"])
            q_sorted = sorted(q_vals.items(), key=lambda x: -x[1])
            gap = q_sorted[0][1] - (q_sorted[1][1] if len(q_sorted) > 1 else 0)
            st = s["structural_features"]
            print(f"    {stratum}: best={q_sorted[0][0]}({q_sorted[0][1]:.1f}) "
                  f"2nd={q_sorted[1][0]}({q_sorted[1][1]:.1f}) gap={gap:.2f} "
                  f"competing={st['has_competing_unverified_support']} "
                  f"legal={s['legal_actions']}")

    # Step 6: Separation audit (I3.28B boundary + I3.26 dev)
    print("\n=== Step 6: Separation audit ===")

    # I3.28B boundary states
    safe_defer_q = []
    unsafe_defer_q = []
    for r in boundary:
        sf = r["state_features"]
        structural = r["structural_features"]
        legal = r["legal_actions"]
        q_vals = predict_all_q(q_v2r, v2r_keys, sf, legal, structural)
        qd = q_vals.get("DEFER", 0)
        if r["boundary_type"] == "safe":
            safe_defer_q.append(qd)
        else:
            unsafe_defer_q.append(qd)

    margin_b = min(safe_defer_q) - max(unsafe_defer_q) if safe_defer_q and unsafe_defer_q else -999
    print(f"  I3.28B boundary: margin={margin_b:.2f} (required >{SEPARATION_MARGIN})")

    # I3.26 dev states
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

    margin_d = min(dev_safe_q) - max(dev_unsafe_q) if dev_safe_q and dev_unsafe_q else -999
    print(f"  I3.26 dev: margin={margin_d:.2f} (required >{SEPARATION_MARGIN})")

    # Step 7: Preservation on 220 original checkpoints
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
            if v2r_best == "ANSWER":
                answer_correct += 1

    # Also check D4 stratum states for ANSWER authority preservation
    d4_states = stratum_states.get("D4", [])
    d4_answer_correct = 0
    d4_answer_total = len(d4_states)
    for s in d4_states:
        q_vals = predict_all_q(q_v2r, v2r_keys, s["state_features"], s["legal_actions"], s["structural_features"])
        v2r_best_d4 = max(q_vals, key=q_vals.get)
        if v2r_best_d4 == "ANSWER":
            d4_answer_correct += 1
    print(f"  D4 ANSWER ranked best: {d4_answer_correct}/{d4_answer_total}")

    print(f"  Mean regret: V1={np.mean(v1_reg):.4f}, V2R={np.mean(v2r_reg):.4f}")
    print(f"  Near-optimal: V1={v1_no}/{n}, V2R={v2r_no}/{n}")
    print(f"  Correct best: V1={v1_cb}/{n}, V2R={v2r_cb}/{n}")
    print(f"  ANSWER ranked best: {answer_correct}/{answer_total}")

    # Gates
    gates = {
        "coverage_defer_safe": coverage > 0.0,
        "false_authority_defer_zero": false_rate == 0.0,
        "separation_boundary": margin_b > SEPARATION_MARGIN,
        "separation_dev": margin_d > SEPARATION_MARGIN,
        "answer_authority_preserved": answer_correct >= answer_total * 0.95 and d4_answer_correct >= d4_answer_total * 0.95,
        "regret_no_worse": np.mean(v2r_reg) <= np.mean(v1_reg) * 1.05,
        "near_optimal_no_worse": v2r_no >= v1_no,
        "correct_best_no_regression": v2r_cb >= int(v1_cb * 0.95),
    }

    print("\n" + "=" * 70)
    print("I3.28C SUMMARY")
    print("=" * 70)
    for g, v in gates.items():
        print(f"  {g}: {'PASS' if v else 'FAIL'}")
    all_pass = all(gates.values())
    print(f"\n  Coverage_DEFER: {coverage:.4f} ({safe_authority}/{safe_total})")
    print(f"  FalseAuthorityRate_DEFER: {false_rate:.4f} ({unsafe_false_authority}/{unsafe_total})")
    print(f"\n  OVERALL: {'PASS' if all_pass else 'FAIL'}")

    results = {
        "experiment": "I3.28C: Targeted DEFER Coverage Audit",
        "n_strata_records": len(all_records),
        "n_combined_records": len(combined),
        "q_v1_sha256": v1_sha,
        "q_v2r_sha256": v2r_sha,
        "coverage_defer": float(coverage),
        "coverage_defer_count": f"{safe_authority}/{safe_total}",
        "false_authority_defer": float(false_rate),
        "false_authority_defer_count": f"{unsafe_false_authority}/{unsafe_total}",
        "separation_margin_boundary": float(margin_b),
        "separation_margin_dev": float(margin_d),
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
        print("\n  RECOMMENDATION: Freeze DAPH_ADAPTIVE_AUTHORITY_EXECUTIVE_V2.")
        print("  Proceed to targeted live D1/D2/D3 safety run.")
    else:
        print("\n  RECOMMENDATION: Do NOT freeze V2. Identify failed gates.")


if __name__ == "__main__":
    main()
