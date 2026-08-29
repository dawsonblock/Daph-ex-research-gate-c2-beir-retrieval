#!/usr/bin/env python3
"""I3.30R2: Structural holdout boundary collection.

The previous I3.30R held-out split only changed domain text, which never
enters Q features. 93.1% of held-out feature vectors were exact duplicates
of training vectors.

This script creates a STRUCTURALLY DISJOINT held-out split by holding out
combinations of:
  - hypothesis count (n_hyps)
  - evidence topology layout
  - budget regime
  - verification depth
  - number of competing supports
  - resource exhaustion pattern

Train and held-out share NO exact model input feature vector.

G0: exact train/heldout feature overlap = 0
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from hrm_adaptive_memory.cognitive_control.core import DecisionAction
from hrm_adaptive_memory.cognitive_control.state import VerificationState, TemporalStatus
from hrm_adaptive_memory.executive.evidence_benchmark import initial_evidence_runtime
from hrm_adaptive_memory.executive.evidence_benchmark.executor import (
    EvidenceExecutor, valid_verify_targets,
)
from hrm_adaptive_memory.executive.evidence_benchmark.schema import (
    EvidenceTask, EvidenceHypothesis, EvidenceItem,
)
from hrm_adaptive_memory.executive.resources import ResourceBudget, ResourceState
from hrm_adaptive_memory.executive.metareasoning_utility import MetareasoningUtility
from daph.intervention.checkpoint import compute_state_features
from daph.epistemic.v3_features import compute_v3_features_canonical
from run_i3_28_rep_repair import extract_v1_features

OUTPUT_DIR = REPO_ROOT / "experiments/i3_30r/causal_boundary_v3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

UTILITY = MetareasoningUtility.from_file(
    REPO_ROOT / "configs" / "v2b_i3_1_utility_v1.json")

V2B_ACTION_NAMES = ["ANSWER", "DEFER", "RETRIEVE", "VERIFY", "SEARCH_MORE", "REASON_MORE", "STOP"]
V2B_ACTIONS = [DecisionAction(a) for a in V2B_ACTION_NAMES]

DOMAINS = [
    ("stroke", "ischemic stroke", "hemorrhagic stroke",
     "CT shows no acute hemorrhage", "CT shows acute intraparenchymal hemorrhage",
     "Is the patient experiencing an ischemic or hemorrhagic stroke?"),
    ("sepsis", "gram-negative sepsis", "gram-positive sepsis",
     "Blood culture grows E. coli", "Blood culture grows S. aureus",
     "Is the patient infected with gram-negative or gram-positive sepsis?"),
    ("pneumonia", "viral pneumonia", "bacterial pneumonia",
     "Respiratory viral panel is positive for influenza", "Sputum culture grows S. pneumoniae",
     "Does the patient have viral or bacterial pneumonia?"),
    ("cardiac", "NSTEMI", "STEMI",
     "ECG shows ST depression in lateral leads", "ECG shows ST elevation in leads V1-V4",
     "Is the patient experiencing NSTEMI or STEMI?"),
    ("thyroid", "Graves disease", "toxic multinodular goiter",
     "TSI is positive", "TSI is negative with hot nodules on scan",
     "Does the patient have Graves disease or toxic multinodular goiter?"),
    ("anemia", "iron deficiency anemia", "anemia of chronic disease",
     "Serum ferritin is low (<15)", "Serum ferritin is high (>100) with low TIBC",
     "Does the patient have iron deficiency anemia or anemia of chronic disease?"),
    ("hepatitis", "viral hepatitis B", "autoimmune hepatitis",
     "HBsAg is positive", "ANA and anti-smooth muscle antibody are positive",
     "Does the patient have viral hepatitis B or autoimmune hepatitis?"),
    ("renal", "acute tubular necrosis", "prerenal azotemia",
     "Urine sodium is high (>40)", "Urine sodium is low (<20)",
     "Does the patient have ATN or prerenal azotemia?"),
    ("cancer", "small cell lung cancer", "non-small cell lung cancer",
     "Biopsy shows small blue cells with scant cytoplasm", "Biopsy shows adenocarcinoma",
     "Does the patient have small cell or non-small cell lung cancer?"),
    ("diabetes", "type 1 diabetes", "type 2 diabetes",
     "C-peptide is undetectable", "C-peptide is within normal range",
     "Does the patient have type 1 or type 2 diabetes?"),
    ("seizure", "focal seizure", "generalized seizure",
     "EEG shows temporal lobe spikes", "EEG shows generalized 3Hz spike-wave",
     "Is the patient experiencing focal or generalized seizures?"),
    ("colitis", "Crohn disease", "ulcerative colitis",
     "Biopsy shows transmural granulomatous inflammation", "Biopsy shows mucosal continuous inflammation",
     "Does the patient have Crohn disease or ulcerative colitis?"),
    ("asthma", "allergic asthma", "non-allergic asthma",
     "Skin prick test is positive for dust mites", "Skin prick test is negative",
     "Does the patient have allergic or non-allergic asthma?"),
    ("arthritis", "rheumatoid arthritis", "osteoarthritis",
     "RF and anti-CCP are positive", "RF and anti-CCP are negative",
     "Does the patient have rheumatoid arthritis or osteoarthritis?"),
    ("encephalitis", "viral encephalitis", "bacterial encephalitis",
     "CSF PCR is positive for HSV", "CSF culture grows Listeria",
     "Does the patient have viral or bacterial encephalitis?"),
    ("lymphoma", "Hodgkin lymphoma", "non-Hodgkin lymphoma",
     "Biopsy shows Reed-Sternberg cells", "Biopsy shows no Reed-Sternberg cells",
     "Does the patient have Hodgkin or non-Hodgkin lymphoma?"),
]


def make_budget(max_steps, max_verify, max_reasoning, max_retrieve=2, max_search=2):
    return ResourceBudget(
        max_executive_steps=max_steps,
        max_retrieval_calls=max_retrieve,
        max_verification_calls=max_verify,
        max_search_calls=max_search,
        max_reasoning_tokens=max_reasoning,
        max_elapsed_ms=10000,
    )


def make_hypotheses(domain, n_hyps, correct_idx, correct_action):
    _, *props, summary = domain
    hyps = []
    for i in range(n_hyps):
        prop = props[i] if i < len(props) else f"Condition {i+1}"
        if i == correct_idx:
            action = correct_action
            payload = f"{action.value}:H{i+1}:{prop}"
        else:
            action = DecisionAction.DEFER if correct_action == DecisionAction.ANSWER else DecisionAction.ANSWER
            payload = f"{action.value}:H{i+1}:{prop}"
        hyps.append(EvidenceHypothesis(f"H{i+1}", prop, action, payload))
    return tuple(hyps)


def _v(ev):
    return {
        "evidence_id": ev.evidence_id,
        "supports": list(ev.supports),
        "contradicts": list(ev.contradicts),
        "verification_state": ev.verification_state.value,
        "temporal_status": ev.temporal_status.value,
        "retrieved": ev.retrieved,
    }

def _h(h):
    return {"hypothesis_id": h.hypothesis_id, "answer_action": h.answer_action.value}


# ============================================================
# Structural split configuration
# ============================================================
# The key insight: the model sees numerical structural features, not domain text.
# To create a structural holdout, we must hold out combinations of:
#   - n_hypotheses
#   - n_evidence_items
#   - budget configuration (steps, verify, retrieve, search)
#   - verification depth (how many items are pre-verified)
#   - topology pattern (which hypotheses have support/contradiction)
#
# Train: n_hyps in {2, 3}, specific budget combos
# Heldout: n_hyps in {4, 5}, different budget combos, different topology patterns

# TRAIN structural configurations
TRAIN_STRUCT = {
    "n_hyps_pool": [2, 2, 3, 3],  # weighted toward 2-3
    "n_ev_pool": [2, 2, 3, 3],
    "budgets": {
        "normal": [
            (3, 2, 256, 2, 2, "train_normal_3s_2v"),
            (4, 2, 256, 2, 2, "train_normal_4s_2v"),
            (5, 3, 256, 2, 2, "train_normal_5s_3v"),
            (3, 1, 128, 1, 1, "train_normal_3s_1v_tight"),
            (4, 1, 128, 1, 1, "train_normal_4s_1v_tight"),
        ],
        "exhausted": [
            (1, 0, 0, 0, 0, "train_exhausted_1s_nor_nov"),
            (1, 0, 128, 0, 0, "train_exhausted_1s_r_nov"),
            (2, 0, 0, 0, 0, "train_exhausted_2s_nor_nov"),
            (2, 0, 128, 0, 0, "train_exhausted_2s_r_nov"),
            (1, 1, 0, 0, 0, "train_exhausted_1s_nor_1v"),
            (2, 1, 0, 0, 0, "train_exhausted_2s_nor_1v"),
            (3, 0, 0, 0, 0, "train_exhausted_3s_nor_nov"),
        ],
        "p3": [
            (3, 2, 256, 2, 2, "train_p3_3s_2v"),
            (4, 2, 256, 2, 2, "train_p3_4s_2v"),
            (5, 2, 256, 2, 2, "train_p3_5s_2v"),
            (3, 1, 128, 1, 1, "train_p3_3s_1v_tight"),
            (4, 1, 128, 1, 1, "train_p3_4s_1v_tight"),
        ],
        "p5": [
            (1, 0, 0, 0, 0, "train_p5_1s_nor_nov"),
            (1, 0, 128, 0, 0, "train_p5_1s_r_nov"),
            (2, 0, 0, 0, 0, "train_p5_2s_nor_nov"),
            (3, 0, 0, 0, 0, "train_p5_3s_nor_nov"),
        ],
    },
}

# HELD-OUT structural configurations — DIFFERENT n_hyps, DIFFERENT budgets
HELDOUT_STRUCT = {
    "n_hyps_pool": [4, 4, 5, 5],  # never seen in training
    "n_ev_pool": [4, 5, 5, 6],    # different evidence counts
    "budgets": {
        "normal": [
            (4, 3, 256, 3, 3, "heldout_normal_4s_3v"),      # more verify, more retrieve
            (5, 3, 256, 3, 3, "heldout_normal_5s_3v"),
            (6, 4, 256, 3, 3, "heldout_normal_6s_4v"),      # never seen step count
            (4, 2, 256, 0, 0, "heldout_normal_4s_2v_noret"), # no retrieve/search
            (5, 2, 128, 1, 1, "heldout_normal_5s_2v_tight"),
        ],
        "exhausted": [
            (2, 0, 256, 0, 0, "heldout_exhausted_2s_r_nov"),    # different from train
            (3, 0, 128, 0, 0, "heldout_exhausted_3s_r_nov"),
            (2, 1, 128, 0, 0, "heldout_exhausted_2s_1v_r"),
            (3, 1, 0, 1, 0, "heldout_exhausted_3s_1v_ret"),     # has retrieve
            (1, 0, 0, 1, 0, "heldout_exhausted_1s_nov_ret"),   # has retrieve, no verify
        ],
        "p3": [
            (4, 3, 256, 3, 3, "heldout_p3_4s_3v"),     # more verify budget
            (5, 3, 256, 2, 2, "heldout_p3_5s_3v"),
            (6, 4, 256, 3, 3, "heldout_p3_6s_4v"),     # never seen
            (4, 2, 256, 1, 1, "heldout_p3_4s_2v_tight"),
        ],
        "p5": [
            (2, 0, 256, 0, 0, "heldout_p5_2s_r_nov"),  # has reasoning, no verify
            (3, 0, 128, 1, 0, "heldout_p5_3s_r_ret"),  # has retrieve
            (1, 0, 0, 1, 1, "heldout_p5_1s_ret_srch"), # has retrieve AND search
        ],
    },
}


# ============================================================
# Task builders (same as before, parameterized by struct config)
# ============================================================

def make_p1a_task(task_id, domain, n_hyps, n_ev, seed):
    correct_hyp = "H1"
    hyps = make_hypotheses(domain, n_hyps, 0, DecisionAction.ANSWER)
    evidence = [
        EvidenceItem("E1", f"Verified evidence supporting {correct_hyp}",
                     "initial", (correct_hyp,), (),
                     VerificationState.SUFFICIENT, TemporalStatus.CURRENT, True, "SUFFICIENT"),
    ]
    for i in range(1, n_ev):
        h_id = f"H{(i % n_hyps) + 1}"
        if h_id == correct_hyp:
            h_id = f"H{(i % (n_hyps-1)) + 2}"
        evidence.append(EvidenceItem(
            f"E{i+1}", f"Unverified evidence supporting {h_id}",
            "initial", (h_id,), (),
            VerificationState.UNVERIFIED, TemporalStatus.CURRENT, True, None))
    return EvidenceTask(
        task_id=task_id, split="i3_30r2", category="P1a_answer_supported_only",
        task_summary=domain[-1], high_stakes=True, budget_profile="STANDARD",
        hypotheses=hyps, evidence_items=tuple(evidence),
        retrieve_exposes=(), search_exposes=(),
        oracle_resolution_path=("ANSWER",),
        expected_terminal=DecisionAction.ANSWER,
        correct_hypothesis_id=correct_hyp)


def make_p1b_task(task_id, domain, n_hyps, n_ev, seed):
    correct_hyp = "H1"
    hyps = make_hypotheses(domain, n_hyps, 0, DecisionAction.ANSWER)
    evidence = [
        EvidenceItem("E1", f"Verified evidence supporting {correct_hyp}",
                     "initial", (correct_hyp,), (),
                     VerificationState.SUFFICIENT, TemporalStatus.CURRENT, True, "SUFFICIENT"),
        EvidenceItem("E2", f"Verified evidence contradicting H2",
                     "initial", (), ("H2",),
                     VerificationState.SUFFICIENT, TemporalStatus.CURRENT, True, "SUFFICIENT"),
    ]
    for i in range(2, n_ev):
        h_id = f"H{(i % n_hyps) + 1}"
        if h_id in (correct_hyp, "H2"):
            h_id = f"H{(i % (n_hyps-2)) + 3}" if n_hyps > 2 else "H1"
        evidence.append(EvidenceItem(
            f"E{i+1}", f"Unverified evidence supporting {h_id}",
            "initial", (h_id,), (),
            VerificationState.UNVERIFIED, TemporalStatus.CURRENT, True, None))
    return EvidenceTask(
        task_id=task_id, split="i3_30r2", category="P1b_answer_mixed",
        task_summary=domain[-1], high_stakes=True, budget_profile="STANDARD",
        hypotheses=hyps, evidence_items=tuple(evidence),
        retrieve_exposes=(), search_exposes=(),
        oracle_resolution_path=("ANSWER",),
        expected_terminal=DecisionAction.ANSWER,
        correct_hypothesis_id=correct_hyp)


def make_p2a_task(task_id, domain, n_hyps, n_ev, seed):
    correct_hyp = "H1"
    hyps = make_hypotheses(domain, n_hyps, 0, DecisionAction.DEFER)
    evidence = [
        EvidenceItem("E1", f"Verified evidence supporting {correct_hyp}",
                     "initial", (correct_hyp,), (),
                     VerificationState.SUFFICIENT, TemporalStatus.CURRENT, True, "SUFFICIENT"),
    ]
    for i in range(1, n_ev):
        h_id = f"H{(i % n_hyps) + 1}"
        if h_id == correct_hyp:
            h_id = f"H{(i % (n_hyps-1)) + 2}"
        evidence.append(EvidenceItem(
            f"E{i+1}", f"Unverified evidence supporting {h_id}",
            "initial", (h_id,), (),
            VerificationState.UNVERIFIED, TemporalStatus.CURRENT, True, None))
    return EvidenceTask(
        task_id=task_id, split="i3_30r2", category="P2a_defer_supported_only",
        task_summary=domain[-1], high_stakes=True, budget_profile="STANDARD",
        hypotheses=hyps, evidence_items=tuple(evidence),
        retrieve_exposes=(), search_exposes=(),
        oracle_resolution_path=("DEFER",),
        expected_terminal=DecisionAction.DEFER,
        correct_hypothesis_id=correct_hyp)


def make_p2b_task(task_id, domain, n_hyps, n_ev, seed):
    correct_hyp = "H1"
    hyps = make_hypotheses(domain, n_hyps, 0, DecisionAction.DEFER)
    evidence = [
        EvidenceItem("E1", f"Verified evidence supporting {correct_hyp}",
                     "initial", (correct_hyp,), (),
                     VerificationState.SUFFICIENT, TemporalStatus.CURRENT, True, "SUFFICIENT"),
        EvidenceItem("E2", f"Verified evidence contradicting H2",
                     "initial", (), ("H2",),
                     VerificationState.SUFFICIENT, TemporalStatus.CURRENT, True, "SUFFICIENT"),
    ]
    for i in range(2, n_ev):
        h_id = f"H{(i % n_hyps) + 1}"
        if h_id in (correct_hyp, "H2"):
            h_id = f"H{(i % (n_hyps-2)) + 3}" if n_hyps > 2 else "H1"
        evidence.append(EvidenceItem(
            f"E{i+1}", f"Unverified evidence supporting {h_id}",
            "initial", (h_id,), (),
            VerificationState.UNVERIFIED, TemporalStatus.CURRENT, True, None))
    return EvidenceTask(
        task_id=task_id, split="i3_30r2", category="P2b_defer_mixed",
        task_summary=domain[-1], high_stakes=True, budget_profile="STANDARD",
        hypotheses=hyps, evidence_items=tuple(evidence),
        retrieve_exposes=(), search_exposes=(),
        oracle_resolution_path=("DEFER",),
        expected_terminal=DecisionAction.DEFER,
        correct_hypothesis_id=correct_hyp)


def make_p2_elim_task(task_id, domain, n_hyps, n_ev, seed):
    """P2_elim: all hypotheses eliminated via SUFFICIENT contradiction."""
    correct_hyp = "H1"
    hyps = make_hypotheses(domain, n_hyps, 0, DecisionAction.DEFER)
    evidence = []
    for i in range(min(n_hyps, n_ev)):
        h_id = f"H{i+1}"
        evidence.append(EvidenceItem(
            f"E{i+1}", f"Verified evidence contradicting {h_id}",
            "initial", (), (h_id,),
            VerificationState.SUFFICIENT, TemporalStatus.CURRENT, True, "SUFFICIENT"))
    # Add extra unverified evidence if n_ev > n_hyps
    for i in range(n_hyps, n_ev):
        evidence.append(EvidenceItem(
            f"E{i+1}", f"Unverified evidence",
            "initial", (), (),
            VerificationState.UNVERIFIED, TemporalStatus.CURRENT, True, None))
    return EvidenceTask(
        task_id=task_id, split="i3_30r2", category="P2_elim_defer_correct",
        task_summary=domain[-1], high_stakes=True, budget_profile="STANDARD",
        hypotheses=hyps, evidence_items=tuple(evidence),
        retrieve_exposes=(), search_exposes=(),
        oracle_resolution_path=("DEFER",),
        expected_terminal=DecisionAction.DEFER,
        correct_hypothesis_id=correct_hyp)


def make_p3_task(task_id, domain, n_hyps, n_ev, seed):
    """P3: competing verified support + unverified discriminator."""
    correct_hyp = "H1"
    hyps = make_hypotheses(domain, n_hyps, 0, DecisionAction.ANSWER)
    evidence = [
        EvidenceItem("E1", f"Verified evidence supporting H1",
                     "initial", ("H1",), (),
                     VerificationState.SUFFICIENT, TemporalStatus.CURRENT, True, "SUFFICIENT"),
        EvidenceItem("E2", f"Verified evidence supporting H2",
                     "initial", ("H2",), (),
                     VerificationState.SUFFICIENT, TemporalStatus.CURRENT, True, "SUFFICIENT"),
        EvidenceItem("E3", f"Unverified evidence contradicting H2",
                     "initial", (), ("H2",),
                     VerificationState.UNVERIFIED, TemporalStatus.CURRENT, True, None),
    ]
    # Add extra evidence for larger n_ev
    for i in range(3, n_ev):
        h_id = f"H{(i % n_hyps) + 1}"
        evidence.append(EvidenceItem(
            f"E{i+1}", f"Unverified evidence supporting {h_id}",
            "initial", (h_id,), (),
            VerificationState.UNVERIFIED, TemporalStatus.CURRENT, True, None))
    return EvidenceTask(
        task_id=task_id, split="i3_30r2", category="P3_continue_competing",
        task_summary=domain[-1], high_stakes=True, budget_profile="STANDARD",
        hypotheses=hyps, evidence_items=tuple(evidence),
        retrieve_exposes=(), search_exposes=(),
        oracle_resolution_path=("VERIFY:E3", "ANSWER"),
        expected_terminal=DecisionAction.ANSWER,
        correct_hypothesis_id=correct_hyp)


def make_p5_task(task_id, domain, n_hyps, n_ev, seed):
    """P5: resource exhausted, no verified evidence, DEFER correct."""
    correct_hyp = "H1"
    hyps = make_hypotheses(domain, n_hyps, 0, DecisionAction.DEFER)
    evidence = []
    for i in range(n_ev):
        h_id = f"H{(i % n_hyps) + 1}"
        evidence.append(EvidenceItem(
            f"E{i+1}", f"Unverified evidence supporting {h_id}",
            "initial", (h_id,), (),
            VerificationState.UNVERIFIED, TemporalStatus.CURRENT, True, None))
    return EvidenceTask(
        task_id=task_id, split="i3_30r2", category="P5_defer_exhausted",
        task_summary=domain[-1], high_stakes=True, budget_profile="STANDARD",
        hypotheses=hyps, evidence_items=tuple(evidence),
        retrieve_exposes=(), search_exposes=(),
        oracle_resolution_path=("DEFER",),
        expected_terminal=DecisionAction.DEFER,
        correct_hypothesis_id=correct_hyp)


REGIME_BUILDERS = {
    "P1a": (make_p1a_task, "normal+exhausted"),
    "P1b": (make_p1b_task, "normal+exhausted"),
    "P2a": (make_p2a_task, "normal+exhausted"),
    "P2b": (make_p2b_task, "normal+exhausted"),
    "P2_elim": (make_p2_elim_task, "normal+exhausted"),
    "P3": (make_p3_task, "p3"),
    "P5": (make_p5_task, "p5"),
}


def get_budgets(struct_config, regime_name):
    builder, budget_key = REGIME_BUILDERS[regime_name]
    if "+" in budget_key:
        parts = budget_key.split("+")
        result = []
        for p in parts:
            result.extend(struct_config["budgets"][p])
        return result
    return struct_config["budgets"][budget_key]


def compute_forced_utility(task, runtime_before, forced_action, pre_actions=()):
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
        current = res.runtime
        total = -UTILITY.action_cost(resources_before, current.resources)
        oracle = list(task.oracle_resolution_path)
        done = list(pre_actions) + [(forced_action.value, target)]
        for step_spec in oracle:
            parts = step_spec.split(":")
            action_name = parts[0]
            target_id = parts[1] if len(parts) > 1 else None
            if (action_name, target_id) in done or action_name == forced_action.value:
                continue
            action = DecisionAction(action_name)
            if action == DecisionAction.VERIFY:
                valid = valid_verify_targets(current)
                if valid:
                    target = valid[0]
                else:
                    continue
            else:
                target = None
            if not current.resources.can_execute(action):
                continue
            try:
                res2 = executor.execute(current, action, target_evidence_id=target)
                cost2 = UTILITY.action_cost(current.resources, res2.runtime.resources)
                total -= cost2
                current = res2.runtime
                done.append((action_name, target_id))
                if res2.terminal:
                    tr2 = UTILITY.terminal_reward(action, bool(res2.task_success))
                    total += tr2
                    return float(total), bool(res2.task_success)
            except:
                continue
        return float(total - 0.5), False
    except Exception:
        return -200.0, False


def collect_boundary_data(task, budget, split, pre_actions=()):
    runtime = initial_evidence_runtime(task, ResourceState(budget=budget))
    executor = EvidenceExecutor()
    for action_name, target_id in pre_actions:
        action = DecisionAction(action_name)
        try:
            res = executor.execute(runtime, action, target_evidence_id=target_id)
            runtime = res.runtime
            if res.terminal:
                return None
        except:
            return None

    sf = compute_state_features(runtime, tuple(a[0] for a in pre_actions))
    visible_ev = [_v(ev) for ev in runtime.visible_evidence]
    hyps_list = [_h(h) for h in task.hypotheses]
    v3 = compute_v3_features_canonical(visible_ev, hyps_list)
    legal = [a.value for a in V2B_ACTIONS if runtime.resources.can_execute(a)]

    records = []
    for action_name in legal:
        action = DecisionAction(action_name)
        utility_val, success = compute_forced_utility(task, runtime, action, pre_actions)
        record = {
            "checkpoint_id": f"i3_30r2_{task.task_id}_{hashlib.sha256(str(pre_actions).encode()).hexdigest()[:8]}",
            "task_id": task.task_id,
            "category": task.category,
            "forced_action": action_name,
            "state_features": sf,
            "v3_features": v3,
            "pinned_policy_utility": utility_val,
            "pinned_policy_success": success,
            "expected_terminal": task.expected_terminal.value,
            "legal_actions": legal,
            "source": "i3_30r2",
            "split": split,
            "pre_actions": [a[0] for a in pre_actions],
        }
        records.append(record)

    if records:
        best = max(records, key=lambda r: r["pinned_policy_utility"])
        for r in records:
            r["correct_first_action"] = best["forced_action"]
    return records


def main():
    print("=" * 70)
    print("I3.30R2: Structural Holdout Boundary Collection")
    print("=" * 70)

    seed = 3026
    rng = random.Random(seed)

    all_records = []

    # Training: n_hyps in {2,3}, train budgets, all domains
    # Held-out: n_hyps in {4,5}, heldout budgets, all domains (same text is fine since text doesn't enter features)

    train_n = 50  # per regime
    heldout_n = 30  # per regime

    for regime_name in REGIME_BUILDERS:
        builder, _ = REGIME_BUILDERS[regime_name]

        # Training
        train_budgets = get_budgets(TRAIN_STRUCT, regime_name)
        print(f"\n  Regime {regime_name} (train):")
        regime_records = []
        for i in range(train_n):
            domain = DOMAINS[(i + seed) % len(DOMAINS)]
            n_hyps = rng.choice(TRAIN_STRUCT["n_hyps_pool"])
            n_ev = rng.choice(TRAIN_STRUCT["n_ev_pool"])
            task_id = f"i3_30r2_{regime_name.lower()}_train_{i:04d}"
            task = builder(task_id, domain, n_hyps, n_ev, seed + i)
            budget_cfg = train_budgets[i % len(train_budgets)]
            budget = make_budget(*budget_cfg[:5])
            records = collect_boundary_data(task, budget, "train")
            if records:
                regime_records.extend(records)
        print(f"    {len(regime_records)} records")
        all_records.extend(regime_records)

        # Held-out (structurally different)
        heldout_budgets = get_budgets(HELDOUT_STRUCT, regime_name)
        print(f"  Regime {regime_name} (heldout):")
        regime_records = []
        for i in range(heldout_n):
            domain = DOMAINS[(i + seed + 100) % len(DOMAINS)]
            n_hyps = rng.choice(HELDOUT_STRUCT["n_hyps_pool"])
            n_ev = rng.choice(HELDOUT_STRUCT["n_ev_pool"])
            task_id = f"i3_30r2_{regime_name.lower()}_heldout_{i:04d}"
            task = builder(task_id, domain, n_hyps, n_ev, seed + i + 1000)
            budget_cfg = heldout_budgets[i % len(heldout_budgets)]
            budget = make_budget(*budget_cfg[:5])
            records = collect_boundary_data(task, budget, "heldout")
            if records:
                regime_records.extend(records)
        print(f"    {len(regime_records)} records")
        all_records.extend(regime_records)

    print(f"\n  Total records: {len(all_records)}")

    # Save
    output_path = OUTPUT_DIR / "causal_actions_v3.jsonl"
    with open(output_path, "w") as f:
        for r in all_records:
            f.write(json.dumps(r, default=str) + "\n")
    print(f"  Saved to {output_path}")

    data_hash = hashlib.sha256(output_path.read_bytes()).hexdigest()
    print(f"  Data SHA-256: {data_hash}")

    # G0: Verify NO feature overlap
    print("\n" + "=" * 70)
    print("G0: Structural disjointness check")
    print("=" * 70)

    from run_i3_30r_train_v3r2 import extract_v3r2_features, get_v3r2_feature_keys
    feature_keys = get_v3r2_feature_keys()

    train_recs = [r for r in all_records if r["split"] == "train"]
    heldout_recs = [r for r in all_records if r["split"] == "heldout"]

    train_sigs = set()
    for r in train_recs:
        sf = r.get("state_features", {})
        v3 = r.get("v3_features", {})
        action = r.get("forced_action")
        feats = extract_v3r2_features(sf, action, v3)
        sig = tuple(feats[k] for k in feature_keys)
        train_sigs.add(sig)

    overlap = 0
    for r in heldout_recs:
        sf = r.get("state_features", {})
        v3 = r.get("v3_features", {})
        action = r.get("forced_action")
        feats = extract_v3r2_features(sf, action, v3)
        sig = tuple(feats[k] for k in feature_keys)
        if sig in train_sigs:
            overlap += 1

    total_heldout = len(heldout_recs)
    pct = overlap / total_heldout * 100 if total_heldout > 0 else 0
    print(f"  Train records: {len(train_recs)}")
    print(f"  Held-out records: {total_heldout}")
    print(f"  Unique train signatures: {len(train_sigs)}")
    print(f"  Exact overlap: {overlap}/{total_heldout} ({pct:.1f}%)")
    print(f"  G0 (overlap = 0): {'PASS' if overlap == 0 else 'FAIL'}")

    # P3 causal validation
    p3_train = [r for r in train_recs if "P3" in r["category"]]
    p3_heldout = [r for r in heldout_recs if "P3" in r["category"]]
    for label, p3_recs in [("train", p3_train), ("heldout", p3_heldout)]:
        by_ckpt = defaultdict(list)
        for r in p3_recs:
            by_ckpt[r["checkpoint_id"]].append(r)
        verify_best = sum(1 for group in by_ckpt.values()
                         if max(group, key=lambda r: r["pinned_policy_utility"])["forced_action"] == "VERIFY")
        answer_best = sum(1 for group in by_ckpt.values()
                         if max(group, key=lambda r: r["pinned_policy_utility"])["forced_action"] == "ANSWER")
        print(f"  P3 {label}: VERIFY best={verify_best}/{len(by_ckpt)}, ANSWER best={answer_best}/{len(by_ckpt)}")


if __name__ == "__main__":
    main()
