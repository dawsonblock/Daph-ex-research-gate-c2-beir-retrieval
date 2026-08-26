"""I3.5-PQ Phase 25: Hostile confirmation benchmark generator.

Generates an untouched benchmark with tight resource budgets, traps,
and anti-heuristic balance. Designed to falsify VP, not optimize it.

Key design principles:
1. Tight budgets: wasted actions cause resource exhaustion and failure
2. Anti-heuristic balance: no single action prior can dominate
3. Premature-ANSWER traps: ANSWER looks good but is wrong
4. Premature-DEFER traps: DEFER looks safe but is wrong
5. Repeated-action traps: redundant RETRIEVE/VERIFY/SEARCH waste budget
6. Retrieval-required: must RETRIEVE before ANSWER
7. Verification-required: must VERIFY before ANSWER
8. Search-required: must SEARCH_MORE before ANSWER
9. Contradictory evidence: must verify to resolve
10. Multi-step chains: retrieve → verify → search → verify → answer

Budget profiles are deliberately tight:
  TIGHT: 6 steps, 2 retrievals, 2 verifications, 1 search
  TIGHT_NO_RETRIEVE: 4 steps, 0 retrievals, 2 verifications, 1 search
  TIGHT_NO_SEARCH: 4 steps, 2 retrievals, 2 verifications, 0 searches
  TIGHT_NO_RETRIEVE_NO_SEARCH: 3 steps, 0 retrievals, 1 verification, 0 searches

Uses seed 4287 (different from development seed 9137).
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path

from hrm_adaptive_memory.cognitive_control.core import DecisionAction
from hrm_adaptive_memory.cognitive_control.state import (
    VerificationState, TemporalStatus,
)
from hrm_adaptive_memory.executive.evidence_benchmark.schema import (
    EvidenceTask, EvidenceHypothesis, EvidenceItem,
)


# ============================================================
# Domain content (different from development set)
# ============================================================

_DOMAINS = [
    {"name": "cardiology", "context": "emergency cardiac assessment",
     "h1": "acute myocardial infarction", "h2": "stable angina",
     "e1_initial": "patient reports chest pain and shortness of breath",
     "e1_supports": "H1", "e1_contradicts": "",
     "e2_initial": "patient has history of hypertension",
     "e2_supports": "", "e2_contradicts": "H2",
     "e3_hidden": "troponin levels are elevated above diagnostic threshold",
     "e3_supports": "H1", "e3_contradicts": "",
     "e4_chain": "ECG shows ST-segment elevation in anterior leads",
     "e4_supports": "H1", "e4_contradicts": "",
     "e5_search": "cardiac catheterization confirms coronary artery occlusion",
     "e5_supports": "H1", "e5_contradicts": ""},
    {"name": "oncology", "context": "tumor classification",
     "h1": "malignant neoplasm", "h2": "benign growth",
     "e1_initial": "biopsy shows abnormal cell proliferation",
     "e1_supports": "H1", "e1_contradicts": "",
     "e2_initial": "tumor size is within benign range",
     "e2_supports": "", "e2_contradicts": "H2",
     "e3_hidden": "histological analysis reveals invasive margins",
     "e3_supports": "H1", "e3_contradicts": "",
     "e4_chain": "molecular markers indicate aggressive phenotype",
     "e4_supports": "H1", "e4_contradicts": "",
     "e5_search": "genomic sequencing confirms cancer-driving mutations",
     "e5_supports": "H1", "e5_contradicts": ""},
    {"name": "cybersecurity", "context": "incident severity assessment",
     "h1": "active data breach", "h2": "false positive alert",
     "e1_initial": "unusual outbound traffic detected on port 443",
     "e1_supports": "H1", "e1_contradicts": "",
     "e2_initial": "alert was triggered by known benign process",
     "e2_supports": "", "e2_contradicts": "H2",
     "e3_hidden": "forensic log analysis shows data exfiltration pattern",
     "e3_supports": "H1", "e3_contradicts": "",
     "e4_chain": "compromised credentials traced to external IP",
     "e4_supports": "H1", "e4_contradicts": "",
     "e5_search": "threat intelligence confirms APT group signature",
     "e5_supports": "H1", "e5_contradicts": ""},
    {"name": "aviation", "context": "mechanical failure diagnosis",
     "h1": "hydraulic system failure", "h2": "sensor malfunction",
     "e1_initial": "pressure reading shows anomalous fluctuation",
     "e1_supports": "H1", "e1_contradicts": "",
     "e2_initial": "sensor was recently calibrated without issues",
     "e2_supports": "", "e2_contradicts": "H2",
     "e3_hidden": "maintenance log reveals recurring hydraulic fluid loss",
     "e3_supports": "H1", "e3_contradicts": "",
     "e4_chain": "physical inspection finds damaged hydraulic line",
     "e4_supports": "H1", "e4_contradicts": "",
     "e5_search": "metallurgical analysis confirms stress fracture in line",
     "e5_supports": "H1", "e5_contradicts": ""},
    {"name": "pharmacology", "context": "adverse drug reaction assessment",
     "h1": "severe allergic reaction", "h2": "mild side effect",
     "e1_initial": "patient developed rash after medication",
     "e1_supports": "H1", "e1_contradicts": "",
     "e2_initial": "listed side effects include mild skin irritation",
     "e2_supports": "", "e2_contradicts": "H2",
     "e3_hidden": "blood test shows elevated IgE and eosinophil count",
     "e3_supports": "H1", "e3_contradicts": "",
     "e4_chain": "skin prick test confirms drug-specific allergy",
     "e4_supports": "H1", "e4_contradicts": "",
     "e5_search": "lymphocyte transformation test strongly positive",
     "e5_supports": "H1", "e5_contradicts": ""},
    {"name": "geology", "context": "seismic event classification",
     "h1": "tectonic earthquake", "h2": "induced seismic event",
     "e1_initial": "seismograph shows unusual waveform pattern",
     "e1_supports": "H1", "e1_contradicts": "",
     "e2_initial": "event occurred near known mining operations",
     "e2_supports": "", "e2_contradicts": "H2",
     "e3_hidden": "depth analysis places hypocenter at tectonic boundary",
     "e3_supports": "H1", "e3_contradicts": "",
     "e4_chain": "focal mechanism solution confirms strike-slip fault",
     "e4_supports": "H1", "e4_contradicts": "",
     "e5_search": "historical seismicity shows tectonic cluster at this depth",
     "e5_supports": "H1", "e5_contradicts": ""},
    {"name": "materials", "context": "structural failure analysis",
     "h1": "metal fatigue failure", "h2": "design defect",
     "e1_initial": "fracture surface shows crack propagation pattern",
     "e1_supports": "H1", "e1_contradicts": "",
     "e2_initial": "component was within design specifications",
     "e2_supports": "", "e2_contradicts": "H2",
     "e3_hidden": "microscopy reveals beach marks characteristic of fatigue",
     "e3_supports": "H1", "e3_contradicts": "",
     "e4_chain": "stress analysis confirms cyclic loading exceeded fatigue limit",
     "e4_supports": "H1", "e4_contradicts": "",
     "e5_search": "SEM imaging shows striation spacing consistent with fatigue",
     "e5_supports": "H1", "e5_contradicts": ""},
    {"name": "ecology", "context": "species population assessment",
     "h1": "invasive species establishment", "h2": "transient population",
     "e1_initial": "non-native species sightings reported in area",
     "e1_supports": "H1", "e1_contradicts": "",
     "e2_initial": "sightings could be temporary migration",
     "e2_supports": "", "e2_contradicts": "H2",
     "e3_hidden": "environmental DNA sampling confirms breeding population",
     "e3_supports": "H1", "e3_contradicts": "",
     "e4_chain": "nest sites found with confirmed reproductive activity",
     "e4_supports": "H1", "e4_contradicts": "",
     "e5_search": "genetic analysis shows founder effect consistent with establishment",
     "e5_supports": "H1", "e5_contradicts": ""},
]


def _seeded_rng(task_id: str) -> random.Random:
    h = hashlib.sha256(task_id.encode()).hexdigest()
    return random.Random(int(h[:16], 16))


def _make_hypotheses(domain: dict, rng: random.Random) -> tuple[EvidenceHypothesis, ...]:
    h1 = EvidenceHypothesis(
        hypothesis_id="H1",
        proposition=f"{domain['h1']}",
        answer_action=DecisionAction.ANSWER,
        answer_payload=f"ANSWER: The assessment is {domain['h1']}",
    )
    h2 = EvidenceHypothesis(
        hypothesis_id="H2",
        proposition=f"{domain['h2']}",
        answer_action=DecisionAction.DEFER,
        answer_payload=f"DEFER: Cannot determine, possibly {domain['h2']}",
    )
    return (h1, h2)


def _make_evidence(
    eid: str, proposition: str, source_class: str,
    supports: tuple[str, ...], contradicts: tuple[str, ...],
    verification_state: VerificationState = VerificationState.UNVERIFIED,
    temporal_status: TemporalStatus = TemporalStatus.CURRENT,
    retrieved: bool = False,
    verify_result: str | None = None,
) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=eid,
        proposition=proposition,
        source_class=source_class,
        supports=supports,
        contradicts=contradicts,
        verification_state=verification_state,
        temporal_status=temporal_status,
        retrieved=retrieved,
        verify_result=verify_result,
    )


# ============================================================
# Hostile task generators
# ============================================================

def _gen_tight_answer(domain: dict, idx: int) -> EvidenceTask:
    """ANSWER-optimal with tight budget — premature ANSWER trap.

    Evidence looks conclusive but requires verification.
    If you ANSWER without VERIFY, you might succeed, but if you
    waste steps on RETRIEVE first, you'll run out of budget.
    """
    tid = f"i3_5c_answer_{idx:04d}"
    rng = _seeded_rng(tid)
    hyps = _make_hypotheses(domain, rng)

    e1 = _make_evidence("E1", domain["e1_initial"], "initial",
                        ("H1",), (),
                        VerificationState.SUFFICIENT, TemporalStatus.CURRENT,
                        retrieved=True, verify_result="SUFFICIENT")
    e2 = _make_evidence("E2", domain["e2_initial"], "initial",
                        (), ("H2",),
                        VerificationState.FALSIFIED, TemporalStatus.CURRENT,
                        retrieved=True, verify_result="FALSIFIED")

    return EvidenceTask(
        task_id=tid, split="confirmation", category="answer",
        task_summary=f"In {domain['context']}, determine if {domain['h1']} or {domain['h2']}.",
        high_stakes=True, budget_profile="TIGHT",
        hypotheses=hyps,
        evidence_items=(e1, e2),
        retrieve_exposes=(), search_exposes=(),
        oracle_resolution_path=("VERIFY:E1", "ANSWER"),
        expected_terminal=DecisionAction.ANSWER,
        correct_hypothesis_id="H1",
    )


def _gen_tight_defer(domain: dict, idx: int) -> EvidenceTask:
    """DEFER-optimal with no budget — premature ANSWER trap.

    Evidence is insufficient. ANSWER is tempting but wrong.
    No retrieval or search budget. Must DEFER.
    """
    tid = f"i3_5c_defer_{idx:04d}"
    rng = _seeded_rng(tid)
    hyps = _make_hypotheses(domain, rng)

    e1 = _make_evidence("E1", domain["e1_initial"], "initial",
                        ("H1",), (),
                        VerificationState.UNVERIFIED, TemporalStatus.CURRENT,
                        retrieved=True, verify_result="MISSING")
    e2 = _make_evidence("E2", domain["e2_initial"], "initial",
                        (), ("H2",),
                        VerificationState.UNVERIFIED, TemporalStatus.CURRENT,
                        retrieved=True, verify_result="MISSING")

    return EvidenceTask(
        task_id=tid, split="confirmation", category="defer",
        task_summary=f"In {domain['context']}, determine if {domain['h1']} or {domain['h2']}.",
        high_stakes=True, budget_profile="TIGHT_NO_RETRIEVE_NO_SEARCH",
        hypotheses=hyps,
        evidence_items=(e1, e2),
        retrieve_exposes=(), search_exposes=(),
        oracle_resolution_path=("DEFER",),
        expected_terminal=DecisionAction.DEFER,
        correct_hypothesis_id="H2",
    )


def _gen_tight_retrieve(domain: dict, idx: int) -> EvidenceTask:
    """RETRIEVE-optimal with tight budget — repeated-action trap.

    Must RETRIEVE to get key evidence, then VERIFY, then ANSWER.
    Budget allows exactly 1 RETRIEVE, 1 VERIFY, 1 ANSWER.
    A wasted RETRIEVE means failure.
    """
    tid = f"i3_5c_retrieve_{idx:04d}"
    rng = _seeded_rng(tid)
    hyps = _make_hypotheses(domain, rng)

    e1 = _make_evidence("E1", domain["e1_initial"], "initial",
                        ("H1",), (),
                        VerificationState.UNVERIFIED, TemporalStatus.CURRENT,
                        retrieved=True, verify_result="MISSING")
    e2 = _make_evidence("E2", domain["e2_initial"], "initial",
                        (), ("H2",),
                        VerificationState.FALSIFIED, TemporalStatus.CURRENT,
                        retrieved=True, verify_result="FALSIFIED")
    e3 = _make_evidence("E3", domain["e3_hidden"], "primary",
                        ("H1",), (),
                        VerificationState.UNVERIFIED, TemporalStatus.CURRENT,
                        retrieved=False, verify_result="SUFFICIENT")

    return EvidenceTask(
        task_id=tid, split="confirmation", category="retrieve",
        task_summary=f"In {domain['context']}, determine if {domain['h1']} or {domain['h2']}.",
        high_stakes=True, budget_profile="TIGHT",
        hypotheses=hyps,
        evidence_items=(e1, e2, e3),
        retrieve_exposes=("E3",), search_exposes=(),
        oracle_resolution_path=("RETRIEVE:E3", "VERIFY:E3", "ANSWER"),
        expected_terminal=DecisionAction.ANSWER,
        correct_hypothesis_id="H1",
    )


def _gen_tight_verify(domain: dict, idx: int) -> EvidenceTask:
    """VERIFY-optimal with tight budget — must verify before answer.

    Evidence is visible but unverified. Must VERIFY to confirm.
    No retrieval budget. Tight step budget.
    """
    tid = f"i3_5c_verify_{idx:04d}"
    rng = _seeded_rng(tid)
    hyps = _make_hypotheses(domain, rng)

    e1 = _make_evidence("E1", domain["e1_initial"], "initial",
                        ("H1",), (),
                        VerificationState.UNVERIFIED, TemporalStatus.CURRENT,
                        retrieved=True, verify_result="SUFFICIENT")
    e2 = _make_evidence("E2", domain["e2_initial"], "initial",
                        (), ("H2",),
                        VerificationState.FALSIFIED, TemporalStatus.CURRENT,
                        retrieved=True, verify_result="FALSIFIED")

    return EvidenceTask(
        task_id=tid, split="confirmation", category="verify",
        task_summary=f"In {domain['context']}, determine if {domain['h1']} or {domain['h2']}.",
        high_stakes=True, budget_profile="TIGHT_NO_RETRIEVE",
        hypotheses=hyps,
        evidence_items=(e1, e2),
        retrieve_exposes=(), search_exposes=(),
        oracle_resolution_path=("VERIFY:E1", "ANSWER"),
        expected_terminal=DecisionAction.ANSWER,
        correct_hypothesis_id="H1",
    )


def _gen_tight_search(domain: dict, idx: int) -> EvidenceTask:
    """SEARCH-optimal with tight budget — must search to find key evidence.

    Initial evidence is stale. Must SEARCH_MORE to get current evidence.
    No retrieval budget. Only 1 search allowed.
    """
    tid = f"i3_5c_search_{idx:04d}"
    rng = _seeded_rng(tid)
    hyps = _make_hypotheses(domain, rng)

    e1 = _make_evidence("E1", domain["e1_initial"], "initial",
                        ("H1",), (),
                        VerificationState.STALE, TemporalStatus.STALE,
                        retrieved=True, verify_result="STALE")
    e2 = _make_evidence("E2", domain["e2_initial"], "initial",
                        (), ("H2",),
                        VerificationState.FALSIFIED, TemporalStatus.CURRENT,
                        retrieved=True, verify_result="FALSIFIED")
    e3 = _make_evidence("E3", domain["e5_search"], "search",
                        ("H1",), (),
                        VerificationState.UNVERIFIED, TemporalStatus.CURRENT,
                        retrieved=False, verify_result="SUFFICIENT")

    return EvidenceTask(
        task_id=tid, split="confirmation", category="search",
        task_summary=f"In {domain['context']}, determine if {domain['h1']} or {domain['h2']}.",
        high_stakes=True, budget_profile="TIGHT_NO_RETRIEVE",
        hypotheses=hyps,
        evidence_items=(e1, e2, e3),
        retrieve_exposes=(), search_exposes=("E3",),
        oracle_resolution_path=("SEARCH_MORE:E3", "VERIFY:E3", "ANSWER"),
        expected_terminal=DecisionAction.ANSWER,
        correct_hypothesis_id="H1",
    )


def _gen_contradiction(domain: dict, idx: int) -> EvidenceTask:
    """Contradictory evidence — must verify to resolve.

    Both hypotheses have supporting evidence, but one is falsified
    upon verification. Must VERIFY to distinguish.
    """
    tid = f"i3_5c_contradiction_{idx:04d}"
    rng = _seeded_rng(tid)
    hyps = _make_hypotheses(domain, rng)

    e1 = _make_evidence("E1", domain["e1_initial"], "initial",
                        ("H1",), (),
                        VerificationState.UNVERIFIED, TemporalStatus.CURRENT,
                        retrieved=True, verify_result="SUFFICIENT")
    e2 = _make_evidence("E2", domain["e2_initial"], "initial",
                        ("H2",), (),
                        VerificationState.UNVERIFIED, TemporalStatus.CURRENT,
                        retrieved=True, verify_result="FALSIFIED")

    return EvidenceTask(
        task_id=tid, split="confirmation", category="contradiction",
        task_summary=f"In {domain['context']}, determine if {domain['h1']} or {domain['h2']}. "
                     f"Evidence appears contradictory.",
        high_stakes=True, budget_profile="TIGHT_NO_RETRIEVE",
        hypotheses=hyps,
        evidence_items=(e1, e2),
        retrieve_exposes=(), search_exposes=(),
        oracle_resolution_path=("VERIFY:E1", "VERIFY:E2", "ANSWER"),
        expected_terminal=DecisionAction.ANSWER,
        correct_hypothesis_id="H1",
    )


def _gen_chain(domain: dict, idx: int) -> EvidenceTask:
    """Multi-step evidence chain — retrieve → verify → search → verify → answer.

    Requires multiple steps in sequence. Tight budget means
    any wasted action causes failure.
    """
    tid = f"i3_5c_chain_{idx:04d}"
    rng = _seeded_rng(tid)
    hyps = _make_hypotheses(domain, rng)

    e1 = _make_evidence("E1", domain["e1_initial"], "initial",
                        ("H1",), (),
                        VerificationState.UNVERIFIED, TemporalStatus.CURRENT,
                        retrieved=True, verify_result="MISSING")
    e2 = _make_evidence("E2", domain["e2_initial"], "initial",
                        (), ("H2",),
                        VerificationState.FALSIFIED, TemporalStatus.CURRENT,
                        retrieved=True, verify_result="FALSIFIED")
    e3 = _make_evidence("E3", domain["e3_hidden"], "primary",
                        ("H1",), (),
                        VerificationState.UNVERIFIED, TemporalStatus.CURRENT,
                        retrieved=False, verify_result="SUFFICIENT")
    e4 = _make_evidence("E4", domain["e4_chain"], "primary",
                        ("H1",), (),
                        VerificationState.UNVERIFIED, TemporalStatus.CURRENT,
                        retrieved=False, verify_result="SUFFICIENT")
    e5 = _make_evidence("E5", domain["e5_search"], "search",
                        ("H1",), (),
                        VerificationState.UNVERIFIED, TemporalStatus.CURRENT,
                        retrieved=False, verify_result="SUFFICIENT")

    return EvidenceTask(
        task_id=tid, split="confirmation", category="chain",
        task_summary=f"In {domain['context']}, determine if {domain['h1']} or {domain['h2']}. "
                     f"Requires multi-step investigation.",
        high_stakes=True, budget_profile="TIGHT_CHAIN",
        hypotheses=hyps,
        evidence_items=(e1, e2, e3, e4, e5),
        retrieve_exposes=("E3", "E4"), search_exposes=("E5",),
        oracle_resolution_path=("RETRIEVE:E3", "VERIFY:E3", "SEARCH_MORE:E5", "VERIFY:E5", "ANSWER"),
        expected_terminal=DecisionAction.ANSWER,
        correct_hypothesis_id="H1",
    )


def _gen_two_live_retrieve(domain: dict, idx: int) -> EvidenceTask:
    """Two-live with retrieval — must retrieve to break the tie.

    Both hypotheses have unverified evidence. Must RETRIEVE to get
    the discriminating evidence. Tight budget.
    """
    tid = f"i3_5c_tl_retrieve_{idx:04d}"
    rng = _seeded_rng(tid)
    hyps = _make_hypotheses(domain, rng)

    e1 = _make_evidence("E1", domain["e1_initial"], "initial",
                        ("H1",), (),
                        VerificationState.UNVERIFIED, TemporalStatus.CURRENT,
                        retrieved=True, verify_result="MISSING")
    e2 = _make_evidence("E2", domain["e2_initial"], "initial",
                        ("H2",), (),
                        VerificationState.UNVERIFIED, TemporalStatus.CURRENT,
                        retrieved=True, verify_result="FALSIFIED")
    e3 = _make_evidence("E3", domain["e3_hidden"], "primary",
                        ("H1",), (),
                        VerificationState.UNVERIFIED, TemporalStatus.CURRENT,
                        retrieved=False, verify_result="SUFFICIENT")

    return EvidenceTask(
        task_id=tid, split="confirmation", category="tl_retrieve",
        task_summary=f"In {domain['context']}, determine if {domain['h1']} or {domain['h2']}. "
                     f"Both hypotheses have supporting evidence.",
        high_stakes=True, budget_profile="TIGHT",
        hypotheses=hyps,
        evidence_items=(e1, e2, e3),
        retrieve_exposes=("E3",), search_exposes=(),
        oracle_resolution_path=("RETRIEVE:E3", "VERIFY:E3", "ANSWER"),
        expected_terminal=DecisionAction.ANSWER,
        correct_hypothesis_id="H1",
    )


def _gen_two_live_verify(domain: dict, idx: int) -> EvidenceTask:
    """Two-live with verification — must verify to break the tie.

    Both hypotheses have visible evidence. Must VERIFY to distinguish.
    No retrieval budget. Tight step budget.
    """
    tid = f"i3_5c_tl_verify_{idx:04d}"
    rng = _seeded_rng(tid)
    hyps = _make_hypotheses(domain, rng)

    e1 = _make_evidence("E1", domain["e1_initial"], "initial",
                        ("H1",), (),
                        VerificationState.UNVERIFIED, TemporalStatus.CURRENT,
                        retrieved=True, verify_result="SUFFICIENT")
    e2 = _make_evidence("E2", domain["e2_initial"], "initial",
                        ("H2",), (),
                        VerificationState.UNVERIFIED, TemporalStatus.CURRENT,
                        retrieved=True, verify_result="FALSIFIED")

    return EvidenceTask(
        task_id=tid, split="confirmation", category="tl_verify",
        task_summary=f"In {domain['context']}, determine if {domain['h1']} or {domain['h2']}. "
                     f"Both hypotheses have supporting evidence.",
        high_stakes=True, budget_profile="TIGHT_NO_RETRIEVE",
        hypotheses=hyps,
        evidence_items=(e1, e2),
        retrieve_exposes=(), search_exposes=(),
        oracle_resolution_path=("VERIFY:E1", "VERIFY:E2", "ANSWER"),
        expected_terminal=DecisionAction.ANSWER,
        correct_hypothesis_id="H1",
    )


def _gen_two_live_search(domain: dict, idx: int) -> EvidenceTask:
    """Two-live with search — must search to break the tie.

    Both hypotheses have stale evidence. Must SEARCH_MORE to get
    current discriminating evidence. No retrieval budget.
    """
    tid = f"i3_5c_tl_search_{idx:04d}"
    rng = _seeded_rng(tid)
    hyps = _make_hypotheses(domain, rng)

    e1 = _make_evidence("E1", domain["e1_initial"], "initial",
                        ("H1",), (),
                        VerificationState.STALE, TemporalStatus.STALE,
                        retrieved=True, verify_result="STALE")
    e2 = _make_evidence("E2", domain["e2_initial"], "initial",
                        ("H2",), (),
                        VerificationState.STALE, TemporalStatus.STALE,
                        retrieved=True, verify_result="STALE")
    e3 = _make_evidence("E3", domain["e5_search"], "search",
                        ("H1",), (),
                        VerificationState.UNVERIFIED, TemporalStatus.CURRENT,
                        retrieved=False, verify_result="SUFFICIENT")

    return EvidenceTask(
        task_id=tid, split="confirmation", category="tl_search",
        task_summary=f"In {domain['context']}, determine if {domain['h1']} or {domain['h2']}. "
                     f"Both hypotheses have stale evidence.",
        high_stakes=True, budget_profile="TIGHT_NO_RETRIEVE",
        hypotheses=hyps,
        evidence_items=(e1, e2, e3),
        retrieve_exposes=(), search_exposes=("E3",),
        oracle_resolution_path=("SEARCH_MORE:E3", "VERIFY:E3", "ANSWER"),
        expected_terminal=DecisionAction.ANSWER,
        correct_hypothesis_id="H1",
    )


def _gen_two_live_defer(domain: dict, idx: int) -> EvidenceTask:
    """Two-live with no budget — must DEFER.

    Both hypotheses have unverified evidence. No budget to discriminate.
    Must DEFER. Premature ANSWER is the trap.
    """
    tid = f"i3_5c_tl_defer_{idx:04d}"
    rng = _seeded_rng(tid)
    hyps = _make_hypotheses(domain, rng)

    e1 = _make_evidence("E1", domain["e1_initial"], "initial",
                        ("H1",), (),
                        VerificationState.UNVERIFIED, TemporalStatus.CURRENT,
                        retrieved=True, verify_result="MISSING")
    e2 = _make_evidence("E2", domain["e2_initial"], "initial",
                        ("H2",), (),
                        VerificationState.UNVERIFIED, TemporalStatus.CURRENT,
                        retrieved=True, verify_result="MISSING")

    return EvidenceTask(
        task_id=tid, split="confirmation", category="tl_defer",
        task_summary=f"In {domain['context']}, determine if {domain['h1']} or {domain['h2']}. "
                     f"Both hypotheses have unverified evidence.",
        high_stakes=True, budget_profile="TIGHT_NO_RETRIEVE_NO_SEARCH",
        hypotheses=hyps,
        evidence_items=(e1, e2),
        retrieve_exposes=(), search_exposes=(),
        oracle_resolution_path=("DEFER",),
        expected_terminal=DecisionAction.DEFER,
        correct_hypothesis_id="H2",
    )


def _gen_premature_answer_trap(domain: dict, idx: int) -> EvidenceTask:
    """Premature ANSWER trap — ANSWER looks good but is wrong.

    E1 appears to support H1, but verification would show it's MISSING.
    E2 eliminates H2. So it looks like H1 is the answer.
    But H1's evidence is actually insufficient — must DEFER.
    Correct answer is H2 (DEFER) because H1 cannot be confirmed.
    """
    tid = f"i3_5c_prem_answer_{idx:04d}"
    rng = _seeded_rng(tid)
    hyps = _make_hypotheses(domain, rng)

    e1 = _make_evidence("E1", domain["e1_initial"], "initial",
                        ("H1",), (),
                        VerificationState.UNVERIFIED, TemporalStatus.CURRENT,
                        retrieved=True, verify_result="MISSING")
    e2 = _make_evidence("E2", domain["e2_initial"], "initial",
                        (), ("H2",),
                        VerificationState.FALSIFIED, TemporalStatus.CURRENT,
                        retrieved=True, verify_result="FALSIFIED")

    return EvidenceTask(
        task_id=tid, split="confirmation", category="prem_answer",
        task_summary=f"In {domain['context']}, determine if {domain['h1']} or {domain['h2']}. "
                     f"Evidence appears to support one hypothesis but may be insufficient.",
        high_stakes=True, budget_profile="TIGHT_NO_RETRIEVE_NO_SEARCH",
        hypotheses=hyps,
        evidence_items=(e1, e2),
        retrieve_exposes=(), search_exposes=(),
        oracle_resolution_path=("VERIFY:E1", "DEFER"),
        expected_terminal=DecisionAction.DEFER,
        correct_hypothesis_id="H2",
    )


def _gen_premature_defer_trap(domain: dict, idx: int) -> EvidenceTask:
    """Premature DEFER trap — DEFER looks safe but evidence is sufficient.

    Evidence is verified and sufficient for H1. But the task looks
    ambiguous on the surface. Must ANSWER, not DEFER.
    """
    tid = f"i3_5c_prem_defer_{idx:04d}"
    rng = _seeded_rng(tid)
    hyps = _make_hypotheses(domain, rng)

    e1 = _make_evidence("E1", domain["e1_initial"], "initial",
                        ("H1",), (),
                        VerificationState.SUFFICIENT, TemporalStatus.CURRENT,
                        retrieved=True, verify_result="SUFFICIENT")
    e2 = _make_evidence("E2", domain["e2_initial"], "initial",
                        (), ("H2",),
                        VerificationState.FALSIFIED, TemporalStatus.CURRENT,
                        retrieved=True, verify_result="FALSIFIED")

    return EvidenceTask(
        task_id=tid, split="confirmation", category="prem_defer",
        task_summary=f"In {domain['context']}, determine if {domain['h1']} or {domain['h2']}. "
                     f"Evidence may appear ambiguous but is actually conclusive.",
        high_stakes=True, budget_profile="TIGHT",
        hypotheses=hyps,
        evidence_items=(e1, e2),
        retrieve_exposes=(), search_exposes=(),
        oracle_resolution_path=("ANSWER",),
        expected_terminal=DecisionAction.ANSWER,
        correct_hypothesis_id="H1",
    )


def _gen_retrieval_waste_trap(domain: dict, idx: int) -> EvidenceTask:
    """Retrieval waste trap — RETRIEVE is tempting but wasteful.

    Evidence is already visible and verified. RETRIEVE would waste
    budget with nothing to find. Must VERIFY or ANSWER directly.
    Tight budget means wasted RETRIEVE causes failure.
    """
    tid = f"i3_5c_retr_waste_{idx:04d}"
    rng = _seeded_rng(tid)
    hyps = _make_hypotheses(domain, rng)

    e1 = _make_evidence("E1", domain["e1_initial"], "initial",
                        ("H1",), (),
                        VerificationState.UNVERIFIED, TemporalStatus.CURRENT,
                        retrieved=True, verify_result="SUFFICIENT")
    e2 = _make_evidence("E2", domain["e2_initial"], "initial",
                        (), ("H2",),
                        VerificationState.FALSIFIED, TemporalStatus.CURRENT,
                        retrieved=True, verify_result="FALSIFIED")

    return EvidenceTask(
        task_id=tid, split="confirmation", category="retr_waste",
        task_summary=f"In {domain['context']}, determine if {domain['h1']} or {domain['h2']}. "
                     f"All evidence is already available.",
        high_stakes=True, budget_profile="TIGHT",
        hypotheses=hyps,
        evidence_items=(e1, e2),
        retrieve_exposes=(), search_exposes=(),
        oracle_resolution_path=("VERIFY:E1", "ANSWER"),
        expected_terminal=DecisionAction.ANSWER,
        correct_hypothesis_id="H1",
    )


def _gen_stale_defer(domain: dict, idx: int) -> EvidenceTask:
    """Stale evidence with no budget — must DEFER.

    All evidence is stale. No budget to search for current evidence.
    Must DEFER. ANSWER on stale evidence would be wrong.
    """
    tid = f"i3_5c_stale_defer_{idx:04d}"
    rng = _seeded_rng(tid)
    hyps = _make_hypotheses(domain, rng)

    e1 = _make_evidence("E1", domain["e1_initial"], "initial",
                        ("H1",), (),
                        VerificationState.STALE, TemporalStatus.STALE,
                        retrieved=True, verify_result="STALE")
    e2 = _make_evidence("E2", domain["e2_initial"], "initial",
                        ("H2",), (),
                        VerificationState.STALE, TemporalStatus.STALE,
                        retrieved=True, verify_result="STALE")

    return EvidenceTask(
        task_id=tid, split="confirmation", category="stale_defer",
        task_summary=f"In {domain['context']}, determine if {domain['h1']} or {domain['h2']}. "
                     f"All evidence is stale and no budget for new evidence.",
        high_stakes=True, budget_profile="TIGHT_NO_RETRIEVE_NO_SEARCH",
        hypotheses=hyps,
        evidence_items=(e1, e2),
        retrieve_exposes=(), search_exposes=(),
        oracle_resolution_path=("DEFER",),
        expected_terminal=DecisionAction.DEFER,
        correct_hypothesis_id="H2",
    )


# ============================================================
# Main generator
# ============================================================

def generate_confirmation_benchmark(
    n_per_subtype: int = 12,
    seed: int = 4287,
) -> tuple[EvidenceTask, ...]:
    """Generate the hostile confirmation benchmark.

    Args:
        n_per_subtype: Number of tasks per subtype (default 12)
        seed: Random seed for domain assignment (default 4287, different
              from development seed 9137)

    Returns:
        Tuple of EvidenceTask objects. 12 subtypes × n_per_subtype tasks.
    """
    rng = random.Random(seed)
    domains = list(_DOMAINS)

    generators = [
        _gen_tight_answer,
        _gen_tight_defer,
        _gen_tight_retrieve,
        _gen_tight_verify,
        _gen_tight_search,
        _gen_contradiction,
        _gen_chain,
        _gen_two_live_retrieve,
        _gen_two_live_verify,
        _gen_two_live_search,
        _gen_two_live_defer,
        _gen_premature_answer_trap,
        _gen_premature_defer_trap,
        _gen_retrieval_waste_trap,
        _gen_stale_defer,
    ]

    tasks = []
    for gen_func in generators:
        for i in range(n_per_subtype):
            domain = domains[i % len(domains)]
            task = gen_func(domain, i)
            tasks.append(task)

    return tuple(tasks)


# Budget profiles for confirmation benchmark
CONFIRMATION_BUDGET_PROFILES = {
    "TIGHT": {
        "max_executive_steps": 6,
        "max_retrieval_calls": 2,
        "max_verification_calls": 2,
        "max_search_calls": 1,
        "max_reasoning_tokens": 256,
        "max_elapsed_ms": 10_000,
    },
    "TIGHT_NO_RETRIEVE": {
        "max_executive_steps": 4,
        "max_retrieval_calls": 0,
        "max_verification_calls": 2,
        "max_search_calls": 1,
        "max_reasoning_tokens": 256,
        "max_elapsed_ms": 10_000,
    },
    "TIGHT_NO_SEARCH": {
        "max_executive_steps": 4,
        "max_retrieval_calls": 2,
        "max_verification_calls": 2,
        "max_search_calls": 0,
        "max_reasoning_tokens": 256,
        "max_elapsed_ms": 10_000,
    },
    "TIGHT_NO_RETRIEVE_NO_SEARCH": {
        "max_executive_steps": 3,
        "max_retrieval_calls": 0,
        "max_verification_calls": 1,
        "max_search_calls": 0,
        "max_reasoning_tokens": 128,
        "max_elapsed_ms": 10_000,
    },
    "TIGHT_CHAIN": {
        "max_executive_steps": 8,
        "max_retrieval_calls": 2,
        "max_verification_calls": 3,
        "max_search_calls": 1,
        "max_reasoning_tokens": 256,
        "max_elapsed_ms": 10_000,
    },
}


if __name__ == "__main__":
    import json
    tasks = generate_confirmation_benchmark(n_per_subtype=12, seed=4287)
    print(f"Generated {len(tasks)} confirmation tasks")

    # Compute hash
    task_json = json.dumps([{
        "task_id": t.task_id,
        "category": t.category,
        "expected_terminal": t.expected_terminal.value,
        "budget_profile": t.budget_profile,
        "correct_hypothesis_id": t.correct_hypothesis_id,
    } for t in tasks], sort_keys=True)
    bench_hash = hashlib.sha256(task_json.encode()).hexdigest()
    print(f"Benchmark hash: {bench_hash}")

    # Print distribution
    from collections import Counter
    cats = Counter(t.category for t in tasks)
    print(f"\nCategory distribution:")
    for cat, n in sorted(cats.items()):
        print(f"  {cat}: {n}")

    terminals = Counter(t.expected_terminal.value for t in tasks)
    print(f"\nTerminal distribution:")
    for term, n in sorted(terminals.items()):
        print(f"  {term}: {n}")

    budgets = Counter(t.budget_profile for t in tasks)
    print(f"\nBudget distribution:")
    for budget, n in sorted(budgets.items()):
        print(f"  {budget}: {n}")
