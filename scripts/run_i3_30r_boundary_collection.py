#!/usr/bin/env python3
"""I3.30R: Rebuild causal boundary data with corrected epistemic semantics.

Fixes:
  - P2_elim: Use SUFFICIENT+contradicts(H) instead of FALSIFIED+contradicts(H)
  - P3: Add unverified discriminating evidence so CONTINUE is actually correct
  - D5: Add unverified discriminating evidence so CONTINUE is actually correct
  - Use canonical V3 features from daph.epistemic.v3_features

Regimes:
  P1a: ANSWER-correct, unique verified support, vha=ANSWER, no competition
  P1b: ANSWER-correct, unique verified support + eliminated, vha=ANSWER
  P2a: DEFER-correct, unique verified support, vha=DEFER, no competition
  P2b: DEFER-correct, unique verified support + eliminated, vha=DEFER
  P2_elim: DEFER-correct, all hypotheses eliminated (SUFFICIENT contradiction)
  P3: CONTINUE-correct, competing verified support + unverified discriminator
  P5: DEFER-correct, resource exhausted, no verified evidence

Output: experiments/i3_30r/causal_boundary_v2/causal_actions_v2.jsonl
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

OUTPUT_DIR = REPO_ROOT / "experiments/i3_30r/causal_boundary_v2"
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

# Domains for held-out split (J-L templates)
HELDOUT_DOMAINS = [
    ("migraine", "migraine with aura", "migraine without aura",
     "Visual aura precedes headache", "No visual aura",
     "Does the patient have migraine with or without aura?"),
    ("copd", "COPD exacerbation", "asthma exacerbation",
     "FEV1/FVC < 0.70 with smoking history", "FEV1/FVC > 0.70 with reversible bronchospasm",
     "Is the patient having a COPD or asthma exacerbation?"),
    ("pancreatitis", "acute pancreatitis", "chronic pancreatitis",
     "Lipase is 3x upper limit of normal", "CT shows calcifications and ductal changes",
     "Does the patient have acute or chronic pancreatitis?"),
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
    """Convert EvidenceItem to dict for feature computation."""
    return {
        "evidence_id": ev.evidence_id,
        "supports": list(ev.supports),
        "contradicts": list(ev.contradicts),
        "verification_state": ev.verification_state.value,
        "temporal_status": ev.temporal_status.value,
        "retrieved": ev.retrieved,
    }


def _h(h):
    """Convert EvidenceHypothesis to dict for feature computation."""
    return {
        "hypothesis_id": h.hypothesis_id,
        "answer_action": h.answer_action.value,
    }


# ============================================================
# Regime builders
# ============================================================

def make_p1a_task(task_id, domain, n_hyps, seed):
    """P1a: ANSWER-correct, unique verified support, vha=ANSWER, no competition.
    H1 has SUFFICIENT support (answer_action=ANSWER). No other verified support.
    Other hypotheses have unverified evidence only.
    """
    rng = random.Random(seed)
    correct_hyp = "H1"
    hyps = make_hypotheses(domain, n_hyps, 0, DecisionAction.ANSWER)
    evidence = [
        EvidenceItem("E1", f"Verified evidence supporting {correct_hyp}",
                     "initial", (correct_hyp,), (),
                     VerificationState.SUFFICIENT, TemporalStatus.CURRENT, True, "SUFFICIENT"),
    ]
    for i in range(1, n_hyps):
        evidence.append(EvidenceItem(
            f"E{i+1}", f"Unverified evidence supporting H{i+1}",
            "initial", (f"H{i+1}",), (),
            VerificationState.UNVERIFIED, TemporalStatus.CURRENT, True, None))
    return EvidenceTask(
        task_id=task_id, split="i3_30r", category="P1a_answer_supported_only",
        task_summary=domain[-1], high_stakes=True, budget_profile="STANDARD",
        hypotheses=hyps, evidence_items=tuple(evidence),
        retrieve_exposes=(), search_exposes=(),
        oracle_resolution_path=("ANSWER",),
        expected_terminal=DecisionAction.ANSWER,
        correct_hypothesis_id=correct_hyp)


def make_p1b_task(task_id, domain, n_hyps, seed):
    """P1b: ANSWER-correct, unique verified support + eliminated, vha=ANSWER.
    H1 has SUFFICIENT support. H2 has SUFFICIENT contradiction (eliminated).
    """
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
    if n_hyps > 2:
        evidence.append(EvidenceItem("E3", f"Unverified evidence supporting H3",
                                     "initial", ("H3",), (),
                                     VerificationState.UNVERIFIED, TemporalStatus.CURRENT, True, None))
    return EvidenceTask(
        task_id=task_id, split="i3_30r", category="P1b_answer_mixed",
        task_summary=domain[-1], high_stakes=True, budget_profile="STANDARD",
        hypotheses=hyps, evidence_items=tuple(evidence),
        retrieve_exposes=(), search_exposes=(),
        oracle_resolution_path=("ANSWER",),
        expected_terminal=DecisionAction.ANSWER,
        correct_hypothesis_id=correct_hyp)


def make_p2a_task(task_id, domain, n_hyps, seed):
    """P2a: DEFER-correct, unique verified support, vha=DEFER, no competition.
    H1 has SUFFICIENT support (answer_action=DEFER). ANSWER is wrong.
    """
    correct_hyp = "H1"
    hyps = make_hypotheses(domain, n_hyps, 0, DecisionAction.DEFER)
    evidence = [
        EvidenceItem("E1", f"Verified evidence supporting {correct_hyp}",
                     "initial", (correct_hyp,), (),
                     VerificationState.SUFFICIENT, TemporalStatus.CURRENT, True, "SUFFICIENT"),
    ]
    for i in range(1, n_hyps):
        evidence.append(EvidenceItem(
            f"E{i+1}", f"Unverified evidence supporting H{i+1}",
            "initial", (f"H{i+1}",), (),
            VerificationState.UNVERIFIED, TemporalStatus.CURRENT, True, None))
    return EvidenceTask(
        task_id=task_id, split="i3_30r", category="P2a_defer_supported_only",
        task_summary=domain[-1], high_stakes=True, budget_profile="STANDARD",
        hypotheses=hyps, evidence_items=tuple(evidence),
        retrieve_exposes=(), search_exposes=(),
        oracle_resolution_path=("DEFER",),
        expected_terminal=DecisionAction.DEFER,
        correct_hypothesis_id=correct_hyp)


def make_p2b_task(task_id, domain, n_hyps, seed):
    """P2b: DEFER-correct, unique verified support + eliminated, vha=DEFER.
    H1 has SUFFICIENT support (DEFER). H2 has SUFFICIENT contradiction (eliminated).
    """
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
    if n_hyps > 2:
        evidence.append(EvidenceItem("E3", f"Unverified evidence supporting H3",
                                     "initial", ("H3",), (),
                                     VerificationState.UNVERIFIED, TemporalStatus.CURRENT, True, None))
    return EvidenceTask(
        task_id=task_id, split="i3_30r", category="P2b_defer_mixed",
        task_summary=domain[-1], high_stakes=True, budget_profile="STANDARD",
        hypotheses=hyps, evidence_items=tuple(evidence),
        retrieve_exposes=(), search_exposes=(),
        oracle_resolution_path=("DEFER",),
        expected_terminal=DecisionAction.DEFER,
        correct_hypothesis_id=correct_hyp)


def make_p2_elim_task(task_id, domain, n_hyps, seed):
    """P2_elim: DEFER-correct, all hypotheses eliminated.

    FIXED: Uses SUFFICIENT+contradicts(H) to eliminate hypotheses,
    NOT FALSIFIED+contradicts(H) which was semantically wrong.

    Per EPISTEMIC_SEMANTICS_V1.md §3.2:
      SUFFICIENT + contradicts(H) → verified contradiction against H (eliminated)
      FALSIFIED + contradicts(H) → no effect (contradiction claim failed)
    """
    rng = random.Random(seed)
    correct_hyp = "H1"
    hyps = make_hypotheses(domain, n_hyps, 0, DecisionAction.DEFER)
    evidence = []
    for i in range(n_hyps):
        h_id = f"H{i+1}"
        evidence.append(EvidenceItem(
            f"E{i+1}", f"Verified evidence contradicting {h_id}",
            "initial", (), (h_id,),
            VerificationState.SUFFICIENT, TemporalStatus.CURRENT, True, "SUFFICIENT"))
    return EvidenceTask(
        task_id=task_id, split="i3_30r", category="P2_elim_defer_correct",
        task_summary=domain[-1], high_stakes=True, budget_profile="STANDARD",
        hypotheses=hyps, evidence_items=tuple(evidence),
        retrieve_exposes=(), search_exposes=(),
        oracle_resolution_path=("DEFER",),
        expected_terminal=DecisionAction.DEFER,
        correct_hypothesis_id=correct_hyp)


def make_p3_task(task_id, domain, n_hyps, seed):
    """P3: CONTINUE-correct, competing verified support + unverified discriminator.

    FIXED: H1 and H2 both have SUFFICIENT support (competing).
    E3 has UNVERIFIED contradiction against H2 — verifying E3 would
    eliminate H2, making H1 uniquely supported → ANSWER_READY.

    Under the fixed executor, ANSWER at t0 fails because there are 2
    supported hypotheses (not unique). VERIFY E3 transitions to a state
    where H1 is uniquely supported → ANSWER succeeds.
    """
    rng = random.Random(seed)
    correct_hyp = "H1"
    hyps = make_hypotheses(domain, n_hyps, 0, DecisionAction.ANSWER)
    evidence = [
        EvidenceItem("E1", f"Verified evidence supporting H1",
                     "initial", ("H1",), (),
                     VerificationState.SUFFICIENT, TemporalStatus.CURRENT, True, "SUFFICIENT"),
        EvidenceItem("E2", f"Verified evidence supporting H2",
                     "initial", ("H2",), (),
                     VerificationState.SUFFICIENT, TemporalStatus.CURRENT, True, "SUFFICIENT"),
        # Unverified discriminator: contradicting H2
        # When verified SUFFICIENT, H2 gets verified contradiction → eliminated
        EvidenceItem("E3", f"Unverified evidence contradicting H2",
                     "initial", (), ("H2",),
                     VerificationState.UNVERIFIED, TemporalStatus.CURRENT, True, None),
    ]
    return EvidenceTask(
        task_id=task_id, split="i3_30r", category="P3_continue_competing",
        task_summary=domain[-1], high_stakes=True, budget_profile="STANDARD",
        hypotheses=hyps, evidence_items=tuple(evidence),
        retrieve_exposes=(), search_exposes=(),
        oracle_resolution_path=("VERIFY:E3", "ANSWER"),
        expected_terminal=DecisionAction.ANSWER,
        correct_hypothesis_id=correct_hyp)


def make_p5_task(task_id, domain, n_hyps, seed):
    """P5: DEFER-correct, resource exhausted, no verified evidence.
    No verification possible. All evidence unverified. DEFER is correct.
    """
    rng = random.Random(seed)
    correct_hyp = "H1"
    hyps = make_hypotheses(domain, n_hyps, 0, DecisionAction.DEFER)
    evidence = []
    for i in range(n_hyps):
        h_id = f"H{i+1}"
        evidence.append(EvidenceItem(
            f"E{i+1}", f"Unverified evidence supporting {h_id}",
            "initial", (h_id,), (),
            VerificationState.UNVERIFIED, TemporalStatus.CURRENT, True, None))
    return EvidenceTask(
        task_id=task_id, split="i3_30r", category="P5_defer_exhausted",
        task_summary=domain[-1], high_stakes=True, budget_profile="STANDARD",
        hypotheses=hyps, evidence_items=tuple(evidence),
        retrieve_exposes=(), search_exposes=(),
        oracle_resolution_path=("DEFER",),
        expected_terminal=DecisionAction.DEFER,
        correct_hypothesis_id=correct_hyp)


# ============================================================
# Utility computation
# ============================================================

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


def collect_boundary_data(task, budget, pre_actions=()):
    """Collect causal data for all legal actions at a state."""
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
            "checkpoint_id": f"i3_30r_{task.task_id}_{hashlib.sha256(str(pre_actions).encode()).hexdigest()[:8]}",
            "task_id": task.task_id,
            "category": task.category,
            "forced_action": action_name,
            "state_features": sf,
            "v3_features": v3,
            "pinned_policy_utility": utility_val,
            "pinned_policy_success": success,
            "expected_terminal": task.expected_terminal.value,
            "legal_actions": legal,
            "source": "i3_30r",
            "split": "train",
            "pre_actions": [a[0] for a in pre_actions],
        }
        records.append(record)

    if records:
        best = max(records, key=lambda r: r["pinned_policy_utility"])
        for r in records:
            r["correct_first_action"] = best["forced_action"]

    return records


# ============================================================
# Main
# ============================================================

# Budget configurations
NORMAL_BUDGETS = [
    (3, 2, 256, 2, 2, "normal_3s_2v"),
    (4, 2, 256, 2, 2, "normal_4s_2v"),
    (5, 3, 256, 2, 2, "normal_5s_3v"),
    (3, 1, 128, 1, 1, "normal_3s_1v_tight"),
    (4, 1, 128, 1, 1, "normal_4s_1v_tight"),
]

EXHAUSTED_BUDGETS = [
    (1, 0, 0, 0, 0, "exhausted_1s_nor_nov_noret_nosrch"),
    (1, 0, 128, 0, 0, "exhausted_1s_r_nov_noret_nosrch"),
    (2, 0, 0, 0, 0, "exhausted_2s_nor_nov_noret_nosrch"),
    (2, 0, 128, 0, 0, "exhausted_2s_r_nov_noret_nosrch"),
    (1, 1, 0, 0, 0, "exhausted_1s_nor_1v_noret_nosrch"),
    (2, 1, 0, 0, 0, "exhausted_2s_nor_1v_noret_nosrch"),
    (3, 0, 0, 0, 0, "exhausted_3s_nor_nov_noret_nosrch"),
]

# P3 needs verify budget to make CONTINUE work
P3_BUDGETS = [
    (3, 2, 256, 2, 2, "p3_3s_2v"),
    (4, 2, 256, 2, 2, "p3_4s_2v"),
    (5, 2, 256, 2, 2, "p3_5s_2v"),
    (3, 1, 128, 1, 1, "p3_3s_1v_tight"),
    (4, 1, 128, 1, 1, "p3_4s_1v_tight"),
]

# P5 (resource exhausted) needs tight budgets
P5_BUDGETS = [
    (1, 0, 0, 0, 0, "p5_1s_nor_nov"),
    (1, 0, 128, 0, 0, "p5_1s_r_nov"),
    (2, 0, 0, 0, 0, "p5_2s_nor_nov"),
    (3, 0, 0, 0, 0, "p5_3s_nor_nov"),
]


def main():
    print("=" * 70)
    print("I3.30R: Rebuild Causal Boundary Data (Corrected Semantics)")
    print("=" * 70)

    seed = 3026
    rng = random.Random(seed)

    # Regime configurations: (name, builder, budget_list, n_tasks, split)
    # split: "train" for training domains, "heldout" for held-out domains
    regimes = [
        ("P1a", make_p1a_task, EXHAUSTED_BUDGETS + NORMAL_BUDGETS, 50, "train"),
        ("P1b", make_p1b_task, EXHAUSTED_BUDGETS + NORMAL_BUDGETS, 50, "train"),
        ("P2a", make_p2a_task, EXHAUSTED_BUDGETS + NORMAL_BUDGETS, 50, "train"),
        ("P2b", make_p2b_task, EXHAUSTED_BUDGETS + NORMAL_BUDGETS, 50, "train"),
        ("P2_elim", make_p2_elim_task, NORMAL_BUDGETS + EXHAUSTED_BUDGETS, 50, "train"),
        ("P3", make_p3_task, P3_BUDGETS, 50, "train"),
        ("P5", make_p5_task, P5_BUDGETS, 50, "train"),
        # Held-out regimes using different domains
        ("P1a_heldout", make_p1a_task, EXHAUSTED_BUDGETS + NORMAL_BUDGETS, 20, "heldout"),
        ("P2a_heldout", make_p2a_task, EXHAUSTED_BUDGETS + NORMAL_BUDGETS, 20, "heldout"),
        ("P2_elim_heldout", make_p2_elim_task, NORMAL_BUDGETS + EXHAUSTED_BUDGETS, 20, "heldout"),
        ("P3_heldout", make_p3_task, P3_BUDGETS, 20, "heldout"),
        ("P5_heldout", make_p5_task, P5_BUDGETS, 20, "heldout"),
    ]

    all_records = []
    for regime_name, make_func, budget_list, n_tasks, split in regimes:
        print(f"\n  Regime {regime_name} ({split}):")
        regime_records = []
        domain_pool = HELDOUT_DOMAINS if split == "heldout" else DOMAINS

        for i in range(n_tasks):
            domain = domain_pool[(i + seed) % len(domain_pool)]
            n_hyps = rng.choice([2, 2, 3, 3, 4])
            task_id = f"i3_30r_{regime_name.lower()}_{i:04d}"
            task = make_func(task_id, domain, n_hyps, seed + i)
            budget_cfg = budget_list[i % len(budget_list)]
            budget = make_budget(*budget_cfg[:5])
            records = collect_boundary_data(task, budget)
            if records:
                for r in records:
                    r["split"] = split
                regime_records.extend(records)

        print(f"    {len(regime_records)} causal records")
        all_records.extend(regime_records)

    print(f"\n  Total causal records: {len(all_records)}")

    # Save
    output_path = OUTPUT_DIR / "causal_actions_v2.jsonl"
    with open(output_path, "w") as f:
        for r in all_records:
            f.write(json.dumps(r, default=str) + "\n")
    print(f"  Saved to {output_path}")

    data_hash = hashlib.sha256(output_path.read_bytes()).hexdigest()
    print(f"  Data SHA-256: {data_hash}")

    # Coverage check
    print("\n" + "=" * 70)
    print("Coverage check (corrected semantics)")
    print("=" * 70)

    train_records = [r for r in all_records if r["split"] == "train"]
    heldout_records = [r for r in all_records if r["split"] == "heldout"]
    print(f"  Train records: {len(train_records)}")
    print(f"  Held-out records: {len(heldout_records)}")

    # Check P3: VERIFY should be correct (not ANSWER)
    p3_records = [r for r in all_records if "P3" in r["category"] and "heldout" not in r["category"]]
    if p3_records:
        # Group by checkpoint
        by_ckpt = defaultdict(list)
        for r in p3_records:
            by_ckpt[r["checkpoint_id"]].append(r)
        verify_best = 0
        answer_best = 0
        total_ckpts = len(by_ckpt)
        for ckpt_id, group in by_ckpt.items():
            best = max(group, key=lambda r: r["pinned_policy_utility"])
            if best["forced_action"] == "VERIFY":
                verify_best += 1
            elif best["forced_action"] == "ANSWER":
                answer_best += 1
        print(f"\n  P3 (CONTINUE-correct) causal validation:")
        print(f"    Total checkpoints: {total_ckpts}")
        print(f"    VERIFY is best action: {verify_best}")
        print(f"    ANSWER is best action: {answer_best}")
        if answer_best > 0:
            print(f"    WARNING: {answer_best} P3 states still have ANSWER as best!")

    # Check P2_elim: all hypotheses should be CONTRADICTED (SUFFICIENT contradiction)
    p2_elim_records = [r for r in all_records if "P2_elim" in r["category"] and "heldout" not in r["category"]]
    if p2_elim_records:
        r0 = p2_elim_records[0]
        v3 = r0["v3_features"]
        print(f"\n  P2_elim semantics check:")
        print(f"    n_hyp_with_verified_contradiction: {v3['n_hyp_with_verified_contradiction']}")
        print(f"    n_eliminated_hypotheses: {v3['n_eliminated_hypotheses']}")
        print(f"    n_hyp_with_verified_support: {v3['n_hyp_with_verified_support']}")


if __name__ == "__main__":
    main()
