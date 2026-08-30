"""I3.30R3-CONFIRMATION benchmark generator: fresh structural configurations.

This generator produces tasks from structural configurations NOT present in
the I3.30R3 development benchmark. It uses:

- A fresh seed (43291) distinct from the development seed (9817)
- 12 NEW domain templates (distinct from the 12 development templates)
- Fresh budget configurations with different step/verify/reason combinations
- The same stratum structure (D1-D5) but with different parameter distributions

The generator preserves the same EvidenceTask schema, the same
DecisionAction enum, the same VerificationState/TemporalStatus enums, and
the same ResourceBudget structure as the development generator.

This is an UNTOUCHED STRUCTURAL CONFIRMATION benchmark.
No task from this benchmark was used in any development or tuning run.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from hrm_adaptive_memory.cognitive_control.core import DecisionAction
from hrm_adaptive_memory.cognitive_control.state import VerificationState, TemporalStatus
from hrm_adaptive_memory.executive.evidence_benchmark.schema import (
    EvidenceTask, EvidenceHypothesis, EvidenceItem,
)
from hrm_adaptive_memory.executive.resources import ResourceBudget


# 12 FRESH domain templates — distinct from the 12 development templates
# These cover different clinical/decision domains entirely
CONFIRMATION_DOMAIN_TEMPLATES = [
    ("aortic", "aortic dissection", "aortic aneurysm",
     "CT angiography shows intimal flap", "CT angiography shows fusiform dilation without flap",
     "D-dimer is markedly elevated", "D-dimer is mildly elevated",
     "Does the patient have aortic dissection or aortic aneurysm?"),
    ("meningitis", "bacterial meningitis", "viral meningitis",
     "CSF shows neutrophilic pleocytosis with low glucose", "CSF shows lymphocytic pleocytosis with normal glucose",
     "CSF culture grows S. pneumoniae", "CSF PCR is positive for enterovirus",
     "Does the patient have bacterial or viral meningitis?"),
    ("leukemia", "acute lymphoblastic leukemia", "acute myeloid leukemia",
     "Bone marrow biopsy shows >20% blasts positive for TdT", "Bone marrow biopsy shows >20% blasts positive for myeloperoxidase",
     "Flow cytometry shows CD10+ CD19+ phenotype", "Flow cytometry shows CD13+ CD33+ phenotype",
     "Does the patient have ALL or AML?"),
    ("pulmonary", "pulmonary embolism", "pneumonia",
     "CT pulmonary angiogram shows filling defect in pulmonary artery", "Chest CT shows consolidative infiltrate",
     "D-dimer is markedly elevated with right heart strain on echo", "D-dimer is mildly elevated with normal echo",
     "Does the patient have pulmonary embolism or pneumonia?"),
    ("autoimmune", "systemic lupus erythematosus", "rheumatoid arthritis",
     "ANA is positive with anti-dsDNA antibodies", "ANA is positive with anti-CCP antibodies",
     "Complement levels are low", "Complement levels are normal",
     "Does the patient have SLE or rheumatoid arthritis?"),
    ("transplant", "graft versus host disease", "cytomegalovirus infection",
     "Skin biopsy shows interface dermatitis with apoptotic keratinocytes", "Skin biopsy shows viral inclusions with CMV immunohistochemistry positive",
     "CMV PCR is negative", "CMV PCR is positive with high viral load",
     "Does the patient have GVHD or CMV infection?"),
    ("metabolic", "diabetic ketoacidosis", "hyperosmolar hyperglycemic state",
     "Blood pH is low (<7.3) with ketones", "Blood pH is normal with high osmolality",
     "Anion gap is elevated", "Anion gap is normal",
     "Does the patient have DKA or HHS?"),
    ("liver", "acute liver failure", "chronic liver disease",
     "INR is markedly elevated with encephalopathy within 26 weeks", "INR is mildly elevated with long-standing cirrhosis",
     "AST/ALT are markedly elevated (>1000)", "AST/ALT are mildly elevated with portal hypertension",
     "Does the patient have acute liver failure or chronic liver disease?"),
    ("electrolyte", "SIADH", "cerebral salt wasting",
     "Urine osmolality is high with low serum uric acid", "Urine osmolality is high with high serum uric acid",
     "Response to fluid restriction is positive", "Response to fluid restriction is negative",
     "Does the patient have SIADH or cerebral salt wasting?"),
    ("vasculitis", "giant cell arteritis", "polymyalgia rheumatica",
     "Temporal artery biopsy shows granulomatous arteritis", "Temporal artery biopsy is normal",
     "ESR is markedly elevated with scalp tenderness", "ESR is elevated with proximal muscle stiffness",
     "Does the patient have giant cell arteritis or polymyalgia rheumatica?"),
    ("coagulation", "heparin-induced thrombocytopenia", "thrombotic thrombocytopenic purpura",
     "Platelet factor 4 antibody is positive", "ADAMTS13 activity is severely low",
     "Platelet count drops after heparin exposure", "Platelet count drops with schistocytes on smear",
     "Does the patient have HIT or TTP?"),
    ("neurodegenerative", "Alzheimer disease", "frontotemporal dementia",
     "MRI shows hippocampal atrophy", "MRI shows frontal and temporal lobe atrophy",
     "Cognitive testing shows early memory impairment", "Cognitive testing shows early behavioral and language changes",
     "Does the patient have Alzheimer disease or frontotemporal dementia?"),
]


@dataclass(frozen=True)
class ConfirmationStratumSpec:
    """Specification for a confirmation benchmark stratum."""
    name: str
    n_tasks: int
    description: str


# Stratum sizes: 400 total tasks (80 per stratum for D1-D4, 80 for D5)
# This gives 400 x 2 arms = 800 trajectories for confirmation
CONFIRMATION_STRATA = {
    "D1": ConfirmationStratumSpec("D1", 80, "safe DEFER, VERIFY unavailable"),
    "D2": ConfirmationStratumSpec("D2", 80, "safe DEFER, verification completed/exhausted"),
    "D3": ConfirmationStratumSpec("D3", 80, "unsafe contradiction/competing-support"),
    "D4": ConfirmationStratumSpec("D4", 80, "ANSWER-correct preservation control"),
    "D5": ConfirmationStratumSpec("D5", 80, "post-verification ambiguous, discriminator resolution"),
}

# FRESH budget configurations — different from development budgets
# Development used specific step/verify combinations; these use different ranges
CONFIRMATION_D1_BUDGETS = [
    (1, 0, 0, 0, 0, "CD1_1s_nov_nor"),
    (1, 0, 256, 0, 0, "CD1_1s_nov_r256"),  # higher reasoning budget
    (2, 0, 0, 1, 0, "CD1_2s_nov_ret"),  # retrieval available
    (2, 0, 256, 0, 1, "CD1_2s_nov_srch"),  # search available
    (3, 0, 0, 0, 0, "CD1_3s_nov_nor"),
    (3, 0, 256, 1, 0, "CD1_3s_nov_ret_r"),  # retrieval + reasoning
    (2, 0, 0, 0, 1, "CD1_2s_nov_srch_only"),
    (4, 0, 256, 0, 0, "CD1_4s_nov_r"),  # 4 steps (dev max was 3)
]

CONFIRMATION_D2_BUDGETS = [
    (2, 1, 0, 0, 0, "CD2_1left_nor"),
    (2, 1, 256, 0, 0, "CD2_1left_r256"),
    (3, 1, 0, 1, 0, "CD2_2left_ret"),  # retrieval available
    (3, 1, 256, 0, 1, "CD2_2left_srch"),  # search available
    (3, 2, 0, 0, 0, "CD2_2v_1left_nor"),
    (4, 2, 256, 0, 0, "CD2_2v_2left_r"),
    (4, 1, 0, 1, 1, "CD2_3left_ret_srch"),  # both retrieval + search
    (5, 2, 256, 0, 0, "CD2_5s_2v_r"),  # 5 steps (dev max was 4)
]

CONFIRMATION_D3_BUDGETS = [
    (1, 0, 0, 0, 0, "CD3_1s_nov_nor"),
    (1, 0, 256, 0, 0, "CD3_1s_nov_r256"),
    (2, 1, 0, 0, 0, "CD3_2s_v_nor"),  # no reasoning (dev always had reasoning)
    (2, 1, 256, 0, 0, "CD3_2s_v_r"),
    (3, 1, 0, 1, 0, "CD3_3s_v_ret"),  # retrieval available
    (3, 2, 256, 0, 1, "CD3_3s_2v_srch"),
    (4, 1, 256, 1, 0, "CD3_4s_v_ret_r"),  # 4 steps + retrieval
    (5, 2, 256, 1, 1, "CD3_5s_2v_ret_srch"),  # 5 steps (dev max was 4)
]

CONFIRMATION_D4_BUDGETS = [
    (1, 1, 0, 0, 0, "CD4_1s"),
    (1, 1, 256, 0, 0, "CD4_1s_r256"),
    (2, 1, 0, 0, 0, "CD4_2s"),
    (2, 1, 256, 0, 0, "CD4_2s_r"),
    (3, 1, 0, 1, 0, "CD4_3s_ret"),  # retrieval available
    (3, 2, 256, 0, 0, "CD4_3s_2v_r"),
    (4, 1, 0, 0, 1, "CD4_4s_srch"),  # 4 steps (dev max was 3)
    (4, 2, 256, 1, 0, "CD4_4s_2v_ret_r"),  # 4 steps + retrieval + reasoning
]

CONFIRMATION_D5_BUDGETS = [
    (3, 2, 256, 0, 0, "CD5_3s_2v_r256"),
    (4, 2, 0, 0, 0, "CD5_4s_2v_nor"),  # no reasoning (dev always had reasoning)
    (4, 1, 256, 0, 0, "CD5_4s_1v_r"),
    (5, 2, 256, 0, 0, "CD5_5s_2v_r"),  # 5 steps (dev max was 5, but different config)
    (3, 1, 0, 1, 0, "CD5_3s_1v_ret"),  # retrieval available
    (4, 2, 256, 1, 0, "CD5_4s_2v_ret_r"),  # retrieval + reasoning
    (5, 3, 0, 0, 1, "CD5_5s_3v_srch"),  # 3 verify + search
    (6, 3, 256, 1, 1, "CD5_6s_3v_ret_srch"),  # 6 steps (dev max was 5)
]


def make_budget(max_steps, max_verify, max_reasoning, max_retrieve=0, max_search=0):
    return ResourceBudget(
        max_executive_steps=max_steps,
        max_retrieval_calls=max_retrieve,
        max_verification_calls=max_verify,
        max_search_calls=max_search,
        max_reasoning_tokens=max_reasoning,
        max_elapsed_ms=10000,
    )


def _make_hypotheses(domain, n_hyps, correct_idx, correct_action):
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


def _make_safe_defer_evidence(n_ev, correct_hyp, wrong_hyps, rng, verified=False):
    """Evidence for safe DEFER: support correct, contradict wrong."""
    evidence = []
    vstate = VerificationState.SUFFICIENT if verified else VerificationState.UNVERIFIED
    for i in range(n_ev):
        if i % 2 == 0 or not wrong_hyps:
            evidence.append(EvidenceItem(
                f"E{i+1}", f"Evidence {i+1} supporting {correct_hyp}",
                "initial", (correct_hyp,), (),
                vstate, TemporalStatus.CURRENT, True, "SUFFICIENT"))
        else:
            wrong = rng.choice(wrong_hyps)
            evidence.append(EvidenceItem(
                f"E{i+1}", f"Evidence {i+1} contradicting {wrong}",
                "initial", (), (wrong,),
                vstate, TemporalStatus.CURRENT, True, "FALSIFIED"))
    return tuple(evidence)


def _make_unsafe_evidence(n_ev, correct_hyp, competing_hyp, rng, verified=False):
    """Evidence for unsafe: competing support for multiple hypotheses."""
    evidence = []
    vstate = VerificationState.SUFFICIENT if verified else VerificationState.UNVERIFIED
    for i in range(n_ev):
        if i % 2 == 0:
            evidence.append(EvidenceItem(
                f"E{i+1}", f"Evidence {i+1} supporting {correct_hyp}",
                "initial", (correct_hyp,), (),
                vstate, TemporalStatus.CURRENT, True, "SUFFICIENT"))
        else:
            evidence.append(EvidenceItem(
                f"E{i+1}", f"Evidence {i+1} supporting {competing_hyp}",
                "initial", (competing_hyp,), (),
                vstate, TemporalStatus.CURRENT, True, "FALSIFIED"))
    return tuple(evidence)


def _make_answer_correct_evidence(n_ev, correct_hyp, rng):
    """Evidence for ANSWER-correct: all verified as SUFFICIENT."""
    evidence = []
    for i in range(n_ev):
        evidence.append(EvidenceItem(
            f"E{i+1}", f"Verified evidence {i+1} supporting {correct_hyp}",
            "initial", (correct_hyp,), (),
            VerificationState.SUFFICIENT, TemporalStatus.CURRENT, True, "SUFFICIENT"))
    return tuple(evidence)


def _make_d5r_evidence(n_ev, correct_hyp, competing_hyp, rng):
    """D5 evidence: competing verified support + unverified discriminator."""
    evidence = [
        EvidenceItem(
            "E1", f"Verified evidence supporting {correct_hyp}",
            "initial", (correct_hyp,), (),
            VerificationState.SUFFICIENT, TemporalStatus.CURRENT, True, "SUFFICIENT"),
        EvidenceItem(
            "E2", f"Verified evidence supporting {competing_hyp}",
            "initial", (competing_hyp,), (),
            VerificationState.SUFFICIENT, TemporalStatus.CURRENT, True, "SUFFICIENT"),
        EvidenceItem(
            "E3", f"Unverified discriminator contradicting {competing_hyp}",
            "initial", (), (competing_hyp,),
            VerificationState.UNVERIFIED, TemporalStatus.CURRENT, True, None),
    ]
    for i in range(3, n_ev):
        h_id = correct_hyp if i % 2 == 0 else competing_hyp
        evidence.append(EvidenceItem(
            f"E{i+1}", f"Unverified evidence supporting {h_id}",
            "initial", (h_id,), (),
            VerificationState.UNVERIFIED, TemporalStatus.CURRENT, True, None))
    return tuple(evidence)


# Module-level budget overrides
_CONFIRMATION_BUDGET_OVERRIDES: dict[str, ResourceBudget] = {}


def generate_confirmation_benchmark(seed=43291):
    """Generate the I3.30R3 confirmation benchmark.

    Uses a fresh seed (43291) distinct from the development seed (9817).
    Uses 12 NEW domain templates distinct from the 12 development templates.
    Uses fresh budget configurations with different step/verify/reason combinations.

    Returns:
        list of EvidenceTask (400 tasks: 80 per stratum x 5 strata)
    """
    global _CONFIRMATION_BUDGET_OVERRIDES
    _CONFIRMATION_BUDGET_OVERRIDES = {}
    rng = random.Random(seed)
    tasks = []

    for stratum_name, spec in CONFIRMATION_STRATA.items():
        for i in range(spec.n_tasks):
            domain = CONFIRMATION_DOMAIN_TEMPLATES[(i + seed) % len(CONFIRMATION_DOMAIN_TEMPLATES)]
            n_hyps = rng.choice([2, 2, 2, 3, 3, 4])
            n_ev = rng.choice([2, 2, 3, 3, 4, 5])
            correct_idx = rng.randint(0, n_hyps - 1)

            task_id = f"i3_30r3c_{stratum_name.lower()}_{i:04d}"

            if stratum_name == "D1":
                budget = CONFIRMATION_D1_BUDGETS[i % len(CONFIRMATION_D1_BUDGETS)]
                correct_hyp = f"H{correct_idx+1}"
                wrong_hyps = [f"H{j+1}" for j in range(n_hyps) if j != correct_idx]
                hyps = _make_hypotheses(domain, n_hyps, correct_idx, DecisionAction.DEFER)
                evidence = _make_safe_defer_evidence(n_ev, correct_hyp, wrong_hyps, rng, verified=False)
                task = EvidenceTask(
                    task_id=task_id, split="i3_30r3c", category=f"D1_safe_defer",
                    task_summary=domain[-1], high_stakes=True,
                    budget_profile="CONFIRMATION_TIGHT",
                    hypotheses=hyps, evidence_items=evidence,
                    retrieve_exposes=(), search_exposes=(),
                    oracle_resolution_path=("DEFER",),
                    expected_terminal=DecisionAction.DEFER,
                    correct_hypothesis_id=correct_hyp,
                )

            elif stratum_name == "D2":
                budget = CONFIRMATION_D2_BUDGETS[i % len(CONFIRMATION_D2_BUDGETS)]
                correct_hyp = f"H{correct_idx+1}"
                wrong_hyps = [f"H{j+1}" for j in range(n_hyps) if j != correct_idx]
                hyps = _make_hypotheses(domain, n_hyps, correct_idx, DecisionAction.DEFER)
                evidence = _make_safe_defer_evidence(n_ev, correct_hyp, wrong_hyps, rng, verified=False)
                task = EvidenceTask(
                    task_id=task_id, split="i3_30r3c", category=f"D2_safe_defer_verified",
                    task_summary=domain[-1], high_stakes=True,
                    budget_profile="CONFIRMATION_TIGHT",
                    hypotheses=hyps, evidence_items=evidence,
                    retrieve_exposes=(), search_exposes=(),
                    oracle_resolution_path=("DEFER",),
                    expected_terminal=DecisionAction.DEFER,
                    correct_hypothesis_id=correct_hyp,
                )

            elif stratum_name == "D3":
                budget = CONFIRMATION_D3_BUDGETS[i % len(CONFIRMATION_D3_BUDGETS)]
                correct_idx = 0
                competing_idx = 1 if n_hyps > 1 else 0
                correct_hyp = f"H{correct_idx+1}"
                competing_hyp = f"H{competing_idx+1}"
                hyps = _make_hypotheses(domain, n_hyps, correct_idx, DecisionAction.ANSWER)
                evidence = _make_unsafe_evidence(n_ev, correct_hyp, competing_hyp, rng, verified=False)
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
                task = EvidenceTask(
                    task_id=task_id, split="i3_30r3c", category=f"D3_unsafe_contradiction",
                    task_summary=domain[-1], high_stakes=True,
                    budget_profile="CONFIRMATION_TIGHT",
                    hypotheses=hyps, evidence_items=evidence,
                    retrieve_exposes=(), search_exposes=(),
                    oracle_resolution_path=tuple(oracle),
                    expected_terminal=DecisionAction.ANSWER,
                    correct_hypothesis_id=correct_hyp,
                )

            elif stratum_name == "D4":
                budget = CONFIRMATION_D4_BUDGETS[i % len(CONFIRMATION_D4_BUDGETS)]
                correct_idx = 0
                correct_hyp = f"H{correct_idx+1}"
                hyps = _make_hypotheses(domain, n_hyps, correct_idx, DecisionAction.ANSWER)
                d4_n_ev = rng.choice([2, 2, 3])
                evidence = _make_answer_correct_evidence(d4_n_ev, correct_hyp, rng)
                task = EvidenceTask(
                    task_id=task_id, split="i3_30r3c", category=f"D4_answer_correct",
                    task_summary=domain[-1], high_stakes=True,
                    budget_profile="CONFIRMATION_TIGHT",
                    hypotheses=hyps, evidence_items=evidence,
                    retrieve_exposes=(), search_exposes=(),
                    oracle_resolution_path=("ANSWER",),
                    expected_terminal=DecisionAction.ANSWER,
                    correct_hypothesis_id=correct_hyp,
                )

            elif stratum_name == "D5":
                budget = CONFIRMATION_D5_BUDGETS[i % len(CONFIRMATION_D5_BUDGETS)]
                correct_idx = 0
                competing_idx = 1
                correct_hyp = f"H{correct_idx+1}"
                competing_hyp = f"H{competing_idx+1}"
                hyps = _make_hypotheses(domain, n_hyps, correct_idx, DecisionAction.ANSWER)
                evidence = _make_d5r_evidence(n_ev, correct_hyp, competing_hyp, rng)
                oracle = ("VERIFY:E3", "ANSWER")
                task = EvidenceTask(
                    task_id=task_id, split="i3_30r3c", category="D5_post_verify_ambiguous",
                    task_summary=domain[-1], high_stakes=True,
                    budget_profile="CONFIRMATION_D5R",
                    hypotheses=hyps, evidence_items=evidence,
                    retrieve_exposes=(), search_exposes=(),
                    oracle_resolution_path=oracle,
                    expected_terminal=DecisionAction.ANSWER,
                    correct_hypothesis_id=correct_hyp,
                )

            _CONFIRMATION_BUDGET_OVERRIDES[task_id] = make_budget(*budget[:5])
            tasks.append(task)

    return tasks


def get_confirmation_budget_for_task(task):
    """Get the budget override for a confirmation task."""
    if task.task_id in _CONFIRMATION_BUDGET_OVERRIDES:
        return _CONFIRMATION_BUDGET_OVERRIDES[task.task_id]
    # Fall back to a default budget
    return ResourceBudget(
        max_executive_steps=3,
        max_retrieval_calls=0,
        max_verification_calls=1,
        max_search_calls=0,
        max_reasoning_tokens=128,
        max_elapsed_ms=10000,
    )


def compute_confirmation_benchmark_hash(tasks):
    """Compute a hash of the confirmation benchmark for provenance."""
    import hashlib
    import json
    task_json = json.dumps([{
        "task_id": t.task_id,
        "category": t.category,
        "expected_terminal": t.expected_terminal.value,
        "budget_profile": t.budget_profile,
        "correct_hypothesis_id": t.correct_hypothesis_id,
        "n_hypotheses": len(t.hypotheses),
        "n_evidence": len(t.evidence_items),
        "hypotheses": [
            {
                "hypothesis_id": h.hypothesis_id,
                "answer_action": h.answer_action.value,
                "proposition": h.proposition,
                "answer_payload": h.answer_payload,
            }
            for h in t.hypotheses
        ],
        "evidence": [
            {
                "evidence_id": e.evidence_id,
                "proposition": e.proposition,
                "source_class": e.source_class,
                "supports": list(e.supports),
                "contradicts": list(e.contradicts),
                "verification_state": e.verification_state.value,
                "temporal_status": e.temporal_status.value,
                "retrieved": e.retrieved,
                "verify_result": e.verify_result,
            }
            for e in t.evidence_items
        ],
        "oracle_resolution_path": list(t.oracle_resolution_path),
    } for t in tasks], sort_keys=True)
    return hashlib.sha256(task_json.encode()).hexdigest()
