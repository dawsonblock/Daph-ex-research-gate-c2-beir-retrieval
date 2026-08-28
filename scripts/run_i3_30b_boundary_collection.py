#!/usr/bin/env python3
"""I3.30B: Post-Verification Causal Boundary Collection.

Collect matched causal states for three post-verification regimes:
  P1: ANSWER-correct (unique verified support, vha=ANSWER)
  P2: DEFER-correct (unique verified support, vha=DEFER)
  P3: CONTINUE-correct (competing verified support or mixed)

Force every legal relevant action at those states:
  ANSWER, DEFER, VERIFY, RETRIEVE, SEARCH_MORE, REASON_MORE

This fills the zero-support cells identified by the V3 coverage matrix.

Output:
  experiments/i3_30b/post_verify_causal_actions_v1.jsonl
  experiments/i3_30b/coverage_check.json
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from hrm_adaptive_memory.cognitive_control.core import DecisionAction
V2B_ACTION_NAMES = ["ANSWER", "DEFER", "RETRIEVE", "VERIFY", "SEARCH_MORE", "REASON_MORE", "STOP"]
V2B_ACTIONS = [DecisionAction(a) for a in V2B_ACTION_NAMES]
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
from run_i3_28_rep_repair import compute_structural_features
from run_i3_30_v3_coverage import compute_v3_features

OUTPUT_DIR = REPO_ROOT / "experiments/i3_30b"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

UTILITY = MetareasoningUtility.from_file(
    REPO_ROOT / "configs" / "v2b_i3_1_utility_v1.json")

# Domain templates (same 12 as I3.29, plus 4 new ones for diversity)
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
    # 4 new domains for diversity
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
    """Build n_hypotheses with the correct one having correct_action."""
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


def make_p1_task(task_id, domain, n_hyps, seed):
    """P1: ANSWER-correct, unique verified support, vha=ANSWER.

    Two variants:
      - P1a: complete_supported (only SUFFICIENT, no FALSIFIED)
      - P1b: complete_mixed (SUFFICIENT + FALSIFIED)

    Alternates based on task index.
    """
    rng = random.Random(seed)
    correct_idx = 0
    correct_hyp = "H1"
    hyps = make_hypotheses(domain, n_hyps, correct_idx, DecisionAction.ANSWER)

    # Alternate between supported-only and mixed topologies
    variant = seed % 2

    if variant == 0:
        # P1a: only SUFFICIENT (complete_supported, unique_supported)
        evidence = [
            EvidenceItem("E1", f"Verified evidence supporting {correct_hyp}",
                         "initial", (correct_hyp,), (),
                         VerificationState.SUFFICIENT, TemporalStatus.CURRENT, True, "SUFFICIENT"),
        ]
        # Add unverified evidence for other hypotheses
        for i in range(1, n_hyps):
            evidence.append(EvidenceItem(
                f"E{i+1}", f"Unverified evidence supporting H{i+1}",
                "initial", (f"H{i+1}",), (),
                VerificationState.UNVERIFIED, TemporalStatus.CURRENT, True, None))
        category = "P1a_answer_supported_only"
    else:
        # P1b: SUFFICIENT + FALSIFIED (complete_mixed, unique_supported_with_elim)
        evidence = [
            EvidenceItem("E1", f"Verified evidence supporting {correct_hyp}",
                         "initial", (correct_hyp,), (),
                         VerificationState.SUFFICIENT, TemporalStatus.CURRENT, True, "SUFFICIENT"),
            EvidenceItem("E2", f"Verified evidence contradicting H2",
                         "initial", (), ("H2",),
                         VerificationState.FALSIFIED, TemporalStatus.CURRENT, True, "FALSIFIED"),
        ]
        if n_hyps > 2:
            evidence.append(EvidenceItem("E3", f"Unverified evidence supporting H3",
                                         "initial", ("H3",), (),
                                         VerificationState.UNVERIFIED, TemporalStatus.CURRENT, True, None))
        category = "P1b_answer_mixed"

    return EvidenceTask(
        task_id=task_id, split="i3_30b", category=category,
        task_summary=domain[-1], high_stakes=True,
        budget_profile="STANDARD",
        hypotheses=hyps, evidence_items=tuple(evidence),
        retrieve_exposes=(), search_exposes=(),
        oracle_resolution_path=("ANSWER",),
        expected_terminal=DecisionAction.ANSWER,
        correct_hypothesis_id=correct_hyp,
    )


def make_p2_task(task_id, domain, n_hyps, seed):
    """P2: DEFER-correct, unique verified support, vha=DEFER.

    Two variants:
      - P2a: complete_supported (only SUFFICIENT, no FALSIFIED)
      - P2b: complete_mixed (SUFFICIENT + FALSIFIED)
    """
    rng = random.Random(seed)
    correct_idx = 0
    correct_hyp = "H1"
    hyps = make_hypotheses(domain, n_hyps, correct_idx, DecisionAction.DEFER)

    variant = seed % 2

    if variant == 0:
        # P2a: only SUFFICIENT (complete_supported, unique_supported, vha=DEFER)
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
        category = "P2a_defer_supported_only"
    else:
        # P2b: SUFFICIENT + FALSIFIED (complete_mixed, unique_supported_with_elim, vha=DEFER)
        evidence = [
            EvidenceItem("E1", f"Verified evidence supporting {correct_hyp}",
                         "initial", (correct_hyp,), (),
                         VerificationState.SUFFICIENT, TemporalStatus.CURRENT, True, "SUFFICIENT"),
            EvidenceItem("E2", f"Verified evidence contradicting H2",
                         "initial", (), ("H2",),
                         VerificationState.FALSIFIED, TemporalStatus.CURRENT, True, "FALSIFIED"),
        ]
        if n_hyps > 2:
            evidence.append(EvidenceItem("E3", f"Unverified evidence supporting H3",
                                         "initial", ("H3",), (),
                                         VerificationState.UNVERIFIED, TemporalStatus.CURRENT, True, None))
        category = "P2b_defer_mixed"

    return EvidenceTask(
        task_id=task_id, split="i3_30b", category=category,
        task_summary=domain[-1], high_stakes=True,
        budget_profile="STANDARD",
        hypotheses=hyps, evidence_items=tuple(evidence),
        retrieve_exposes=(), search_exposes=(),
        oracle_resolution_path=("DEFER",),
        expected_terminal=DecisionAction.DEFER,
        correct_hypothesis_id=correct_hyp,
    )


def make_p3_task(task_id, domain, n_hyps, seed):
    """P3: CONTINUE-correct, competing verified support.
    
    Both H1 and H2 have verified support (SUFFICIENT). Need more verification
    to resolve. The correct action is to continue (VERIFY, REASON_MORE, etc.).
    """
    rng = random.Random(seed)
    correct_idx = 0
    correct_hyp = "H1"
    # P3: ANSWER is eventually correct after more verification
    hyps = make_hypotheses(domain, n_hyps, correct_idx, DecisionAction.ANSWER)
    
    # Both H1 and H2 have verified support — competing
    evidence = [
        EvidenceItem("E1", f"Verified evidence supporting H1",
                     "initial", ("H1",), (),
                     VerificationState.SUFFICIENT, TemporalStatus.CURRENT, True, "SUFFICIENT"),
        EvidenceItem("E2", f"Verified evidence supporting H2",
                     "initial", ("H2",), (),
                     VerificationState.SUFFICIENT, TemporalStatus.CURRENT, True, "SUFFICIENT"),
    ]
    
    # Oracle: need to verify more to resolve, then answer
    # Since both are SUFFICIENT, we need more evidence to break the tie
    # In practice, the executor will handle this
    
    return EvidenceTask(
        task_id=task_id, split="i3_30b", category="P3_continue_correct",
        task_summary=domain[-1], high_stakes=True,
        budget_profile="STANDARD",
        hypotheses=hyps, evidence_items=tuple(evidence),
        retrieve_exposes=(), search_exposes=(),
        oracle_resolution_path=("REASON_MORE", "ANSWER"),
        expected_terminal=DecisionAction.ANSWER,
        correct_hypothesis_id=correct_hyp,
    )


def make_p2_elim_task(task_id, domain, n_hyps, seed):
    """P2-elim: DEFER-correct, only eliminated (no verified support).
    
    All hypotheses have verified contradiction (FALSIFIED). No verified support.
    DEFER is correct because no hypothesis can be confirmed.
    """
    rng = random.Random(seed)
    correct_idx = 0
    correct_hyp = "H1"
    hyps = make_hypotheses(domain, n_hyps, correct_idx, DecisionAction.DEFER)
    
    evidence = [
        EvidenceItem("E1", f"Verified evidence contradicting H1",
                     "initial", (), ("H1",),
                     VerificationState.FALSIFIED, TemporalStatus.CURRENT, True, "FALSIFIED"),
        EvidenceItem("E2", f"Verified evidence contradicting H2",
                     "initial", (), ("H2",),
                     VerificationState.FALSIFIED, TemporalStatus.CURRENT, True, "FALSIFIED"),
    ]
    if n_hyps > 2:
        evidence.append(EvidenceItem("E3", f"Verified evidence contradicting H3",
                                     "initial", (), ("H3",),
                                     VerificationState.FALSIFIED, TemporalStatus.CURRENT, True, "FALSIFIED"))
    
    return EvidenceTask(
        task_id=task_id, split="i3_30b", category="P2_elim_defer_correct",
        task_summary=domain[-1], high_stakes=True,
        budget_profile="STANDARD",
        hypotheses=hyps, evidence_items=tuple(evidence),
        retrieve_exposes=(), search_exposes=(),
        oracle_resolution_path=("DEFER",),
        expected_terminal=DecisionAction.DEFER,
        correct_hypothesis_id=correct_hyp,
    )


def compute_forced_utility(task, runtime_before, forced_action, pre_actions=()):
    """Force an action and compute realized utility (same as I3.28C)."""
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

        # Non-terminal: follow oracle
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

        return float(total - 0.5), False  # step limit penalty

    except Exception as e:
        return -200.0, False


def collect_boundary_data(task, budget, pre_actions=()):
    """Collect causal data for all legal actions at a state."""
    runtime = initial_evidence_runtime(task, ResourceState(budget=budget))
    executor = EvidenceExecutor()

    # Execute pre-actions
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
    
    # Compute V3 features
    hyps_list = [{"hypothesis_id": h.hypothesis_id, "answer_action": h.answer_action.value}
                 for h in task.hypotheses]
    v3 = compute_v3_features(visible_ev, hyps_list)
    
    legal = [a.value for a in V2B_ACTIONS if runtime.resources.can_execute(a)]

    records = []
    for action_name in legal:
        action = DecisionAction(action_name)
        utility_val, success = compute_forced_utility(task, runtime, action, pre_actions)

        record = {
            "checkpoint_id": f"i3_30b_{task.task_id}_{hashlib.sha256(str(pre_actions).encode()).hexdigest()[:8]}",
            "task_id": task.task_id,
            "category": task.category,
            "forced_action": action_name,
            "state_features": sf,
            "structural_features": structural,
            "v3_features": v3,
            "pinned_policy_utility": utility_val,
            "pinned_policy_success": success,
            "correct_first_action": None,  # will be filled
            "expected_terminal": task.expected_terminal.value,
            "legal_actions": legal,
            "source": "i3_30b",
            "pre_actions": [a[0] for a in pre_actions],
        }
        records.append(record)

    # Determine correct first action (highest utility)
    if records:
        best = max(records, key=lambda r: r["pinned_policy_utility"])
        for r in records:
            r["correct_first_action"] = best["forced_action"]

    return records


def main():
    print("=" * 70)
    print("I3.30B: Post-Verification Causal Boundary Collection")
    print("=" * 70)

    seed = 3026  # fresh seed
    rng = random.Random(seed)

    all_records = []
    
    # Budget configurations for post-verification states
    # Need enough steps/budget to allow all actions
    BUDGETS = [
        (3, 2, 256, 2, 2, "post_verify_3s_2v"),
        (4, 2, 256, 2, 2, "post_verify_4s_2v"),
        (5, 3, 256, 2, 2, "post_verify_5s_3v"),
        (3, 1, 128, 1, 1, "post_verify_3s_1v_tight"),
        (4, 1, 128, 1, 1, "post_verify_4s_1v_tight"),
    ]

    n_per_regime = 40  # 40 tasks per regime × 16 domains ≈ 640 boundary states

    for regime, make_func in [
        ("P1", make_p1_task),
        ("P2", make_p2_task),
        ("P3", make_p3_task),
        ("P2_elim", make_p2_elim_task),
    ]:
        print(f"\n  Regime {regime}:")
        regime_records = []
        
        for i in range(n_per_regime):
            domain = DOMAINS[(i + seed) % len(DOMAINS)]
            n_hyps = rng.choice([2, 2, 3, 3, 4])
            task_id = f"i3_30b_{regime.lower()}_{i:04d}"
            
            task = make_func(task_id, domain, n_hyps, seed + i)
            budget_cfg = BUDGETS[i % len(BUDGETS)]
            budget = make_budget(*budget_cfg[:5])
            
            records = collect_boundary_data(task, budget)
            if records:
                regime_records.extend(records)
        
        print(f"    {len(regime_records)} causal records")
        all_records.extend(regime_records)

    print(f"\n  Total causal records: {len(all_records)}")

    # Save records
    output_path = OUTPUT_DIR / "post_verify_causal_actions_v1.jsonl"
    with open(output_path, "w") as f:
        for r in all_records:
            f.write(json.dumps(r, default=str) + "\n")
    print(f"  Saved to {output_path}")

    # Compute data hash
    data_hash = hashlib.sha256(output_path.read_bytes()).hexdigest()
    print(f"  Data SHA-256: {data_hash}")

    # Coverage check
    print("\n" + "=" * 70)
    print("Coverage check")
    print("=" * 70)

    # Group by V3 cell
    from run_i3_30_v3_coverage import classify_v3_cell
    coverage = defaultdict(int)
    for r in all_records:
        cell = classify_v3_cell(r["v3_features"], r["expected_terminal"], r["forced_action"])
        coverage[cell] += 1

    # Check critical cells
    critical_cells = [
        ("complete_supported", "unique_supported", "ANSWER", "ANSWER", "D4: ANSWER authority"),
        ("complete_supported", "unique_supported", "DEFER", "DEFER", "D2: DEFER authority"),
        ("complete_supported", "unique_supported", "ANSWER", "DEFER", "D2 false: NOT ANSWER"),
        ("complete_supported", "unique_supported", "DEFER", "ANSWER", "D3 false: NOT DEFER"),
        ("complete_eliminated", "only_eliminated", None, "DEFER", "D2 elim: DEFER authority"),
        ("complete_mixed", "unique_supported_with_elim", "ANSWER", "ANSWER", "D3 post-verify: ANSWER"),
        ("complete_mixed", "unique_supported_with_elim", "DEFER", "DEFER", "D2 elim: DEFER"),
        ("complete_supported", "competing_support", None, "ANSWER", "D3 competing: NOT terminal"),
    ]

    print(f"\n{'VState':<22} {'Topology':<25} {'VHA':<8} {'Expected':<10} {'Count':<6} {'Description'}")
    print("-" * 120)

    for vstate, topo, vha, et, desc in critical_cells:
        count = sum(v for k, v in coverage.items()
                    if k[0] == vstate and k[1] == topo and k[2] == vha and k[3] == et)
        marker = " *** ZERO ***" if count == 0 else (" * TINY *" if count < 10 else "")
        print(f"{vstate:<22} {topo:<25} {str(vha):<8} {et:<10} {count:<6} {desc}{marker}")

    # Save coverage check
    coverage_check = {
        "total_records": len(all_records),
        "data_sha256": data_hash,
        "coverage": {str(k): v for k, v in coverage.items()},
    }
    with open(OUTPUT_DIR / "coverage_check.json", "w") as f:
        json.dump(coverage_check, f, indent=2)


if __name__ == "__main__":
    main()
