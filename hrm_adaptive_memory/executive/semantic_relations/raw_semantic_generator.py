"""I3.12b: Controlled raw-semantic task generator.

Generates tasks where the support/contradiction relation is
recoverable from proposition text but not explicitly annotated
for the controller.

The propositions use controlled semantic language:
  - "confirms", "validates", "establishes" -> SUPPORT
  - "refutes", "denies", "contradicts" -> CONTRADICT
  - "mentions in passing", "tangential" -> NEUTRAL
  - "stale", "outdated" -> temporal mismatch
  - "current", "recent", "updated" -> temporal match

Gold relations are stored in a separate field for evaluation.
The EvidenceItem.supports/contradicts fields still exist (the
executor needs them), but in S1 condition they are replaced by
inferred relations at runtime.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from hrm_adaptive_memory.cognitive_control.core import DecisionAction
from hrm_adaptive_memory.cognitive_control.state import (
    TemporalStatus, VerificationState,
)
from hrm_adaptive_memory.executive.evidence_benchmark import (
    EvidenceItem, EvidenceTask, EvidenceHypothesis,
)


# ---------------------------------------------------------------------------
# Semantic templates for controlled language
# ---------------------------------------------------------------------------

SUBJECTS = [
    "the API endpoint status",
    "the database migration outcome",
    "the service deployment state",
    "the configuration change validity",
    "the incident resolution status",
    " the feature rollout progress",
    "the security patch application",
    "the backup completion state",
    "the cache invalidation result",
    "the cluster health status",
    "the certificate renewal status",
    "the queue processing outcome",
    "the replication lag status",
    "the schema evolution result",
    "the load balancer configuration",
    "the monitoring alert state",
    "the pipeline execution status",
    "the rollback operation outcome",
    "the scaling event result",
    "the failover procedure status",
]

# Hypothesis templates: H1 = ANSWER (current/confirmed), H2 = DEFER (stale/unconfirmed)
H1_PROPOSITION_TEMPLATE = (
    "The system should ANSWER because {subject} is current and confirmed."
)
H2_PROPOSITION_TEMPLATE = (
    "The system should DEFER because {subject} is stale or unconfirmed."
)

# Evidence proposition templates that semantically encode relations
SUPPORT_H1_TEMPLATES = [
    "Source {source} confirms that {subject} is current and operational.",
    "The primary documentation explicitly validates that {subject} is currently active.",
    "A recent update establishes that {subject} is confirmed and current.",
    "Source {source} demonstrates that {subject} is currently operational.",
    "The current monitoring data shows that {subject} is active and confirmed.",
    "Source {source} verifies that {subject} is operational and current.",
]

CONTRADICT_H1_TEMPLATES = [
    "Source {source} refutes the claim that {subject} is current.",
    "Source {source} denies that {subject} is currently operational.",
    "Source {source} contradicts the claim that {subject} is confirmed.",
    "Source {source} confirms that {subject} is stale and outdated.",
    "Source {source} disputes that {subject} is current or confirmed.",
]

SUPPORT_H2_TEMPLATES = [
    "Source {source} confirms that {subject} is stale and outdated.",
    "Source {source} establishes that {subject} is outdated and expired.",
    "The only available report is from 2022 and confirms that {subject} is stale and outdated.",
    "Source {source} verifies that {subject} is unconfirmed and stale.",
    "Source {source} documents that {subject} has expired and is outdated.",
]

CONTRADICT_H2_TEMPLATES = [
    "Source {source} refutes that {subject} is stale.",
    "Source {source} contradicts the claim that {subject} is outdated.",
    "Source {source} denies that {subject} is unconfirmed.",
    "Source {source} disputes that {subject} has expired.",
]

NEUTRAL_TEMPLATES = [
    "A tangential reference mentions {subject} in passing.",
    "Source {source} is silent on {subject}.",
    "An unrelated note references {subject} without confirming or denying it.",
    "Source {source} discusses a different topic entirely.",
    "A record mentions {subject} without providing status details.",
]

STALE_TEMPLATES = [
    "An outdated report from 2021 mentions {subject}.",
    "A stale reference states that {subject}.",
    "An old document references {subject} but may no longer be accurate.",
]


SOURCES = ["A", "B", "C", "D", "E", "F"]


def _pick(rng: random.Random, templates: list[str]) -> str:
    return rng.choice(templates)


def _fill(template: str, subject: str, source: str) -> str:
    return template.format(subject=subject, source=source)


# ---------------------------------------------------------------------------
# Gold relation storage
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GoldRelation:
    """Evaluator-side gold relation (never enters controller packets)."""
    evidence_id: str
    hypothesis_id: str
    relation: str  # "SUPPORT" or "CONTRADICT"


@dataclass
class SemanticTask:
    """A task with both gold relations and the EvidenceTask."""
    evidence_task: EvidenceTask
    gold_relations: tuple[GoldRelation, ...]

    @property
    def task_id(self) -> str:
        return self.evidence_task.task_id

    @property
    def category(self) -> str:
        return self.evidence_task.category


# ---------------------------------------------------------------------------
# Evidence item builders (with gold relations)
# ---------------------------------------------------------------------------

def _make_hyps(n: int, subject: str) -> tuple[EvidenceHypothesis, ...]:
    h1 = EvidenceHypothesis(
        hypothesis_id="H1",
        proposition=H1_PROPOSITION_TEMPLATE.format(subject=subject),
        answer_action=DecisionAction.ANSWER,
        answer_payload=f"confirmed: {subject}",
    )
    h2 = EvidenceHypothesis(
        hypothesis_id="H2",
        proposition=H2_PROPOSITION_TEMPLATE.format(subject=subject),
        answer_action=DecisionAction.DEFER,
        answer_payload=f"insufficient evidence: {subject}",
    )
    hyps = [h1, h2]
    for i in range(3, n + 1):
        hyps.append(EvidenceHypothesis(
            hypothesis_id=f"H{i}",
            proposition=f"The system should DEFER because {subject} is ambiguous (hypothesis {i}).",
            answer_action=DecisionAction.DEFER,
            answer_payload=f"insufficient evidence (hypothesis {i})",
        ))
    return tuple(hyps)


def _suff(eid: str, prop: str, supports: tuple, contradicts: tuple,
          retrieved: bool = True) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=eid, proposition=prop,
        source_class="primary" if retrieved else "search",
        supports=supports, contradicts=contradicts,
        verification_state=VerificationState.UNVERIFIED,
        temporal_status=TemporalStatus.CURRENT,
        retrieved=retrieved, verify_result="SUFFICIENT",
    )


def _falsified(eid: str, prop: str, supports: tuple, contradicts: tuple,
               retrieved: bool = True) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=eid, proposition=prop,
        source_class="initial",
        supports=supports, contradicts=contradicts,
        verification_state=VerificationState.UNVERIFIED,
        temporal_status=TemporalStatus.CURRENT,
        retrieved=retrieved, verify_result="FALSIFIED",
    )


def _noise(eid: str, subject: str, rng: random.Random, retrieved: bool = False) -> EvidenceItem:
    template = _pick(rng, NEUTRAL_TEMPLATES)
    source = rng.choice(SOURCES)
    prop = _fill(template, subject, source)
    return EvidenceItem(
        evidence_id=eid, proposition=prop,
        source_class="search",
        supports=(), contradicts=(),
        verification_state=VerificationState.UNVERIFIED,
        temporal_status=TemporalStatus.CURRENT,
        retrieved=retrieved, verify_result="MISSING",
    )


# ---------------------------------------------------------------------------
# Generators (mirror I3.11d topology but with semantically encoded text)
# ---------------------------------------------------------------------------

def gen_bilateral_conflict_h0(task_id, subject, rng):
    """2 hyps, bilateral SUFFICIENT conflict. T2 at step 2."""
    h = _make_hyps(2, subject)
    sa, sb = rng.choice(SOURCES), rng.choice(SOURCES)
    e1_prop = _fill(_pick(rng, SUPPORT_H1_TEMPLATES), subject, sa)
    e2_prop = _fill(_pick(rng, SUPPORT_H2_TEMPLATES), subject, sb)
    ev = (
        _suff("E1", e1_prop, ("H1",), ("H2",)),
        _suff("E2", e2_prop, ("H2",), ("H1",)),
    )
    gold = (
        GoldRelation("E1", "H1", "SUPPORT"),
        GoldRelation("E1", "H2", "CONTRADICT"),
        GoldRelation("E2", "H2", "SUPPORT"),
        GoldRelation("E2", "H1", "CONTRADICT"),
    )
    et = EvidenceTask(task_id=task_id, split="i3_12_s1",
        category="bilateral_conflict_h0", task_summary=f"Determine {subject}.",
        high_stakes=True, budget_profile="STANDARD", hypotheses=h,
        evidence_items=ev, retrieve_exposes=(), search_exposes=(),
        oracle_resolution_path=("VERIFY:E1", "VERIFY:E2", "DEFER"),
        expected_terminal=DecisionAction.DEFER, correct_hypothesis_id="H2")
    return SemanticTask(et, gold)


def gen_bilateral_conflict_h1_noise(task_id, subject, rng):
    """2 hyps, bilateral conflict + 1 hidden noise."""
    h = _make_hyps(2, subject)
    sa, sb = rng.choice(SOURCES), rng.choice(SOURCES)
    e1_prop = _fill(_pick(rng, SUPPORT_H1_TEMPLATES), subject, sa)
    e2_prop = _fill(_pick(rng, SUPPORT_H2_TEMPLATES), subject, sb)
    ev = (
        _suff("E1", e1_prop, ("H1",), ("H2",)),
        _suff("E2", e2_prop, ("H2",), ("H1",)),
        _noise("E3", subject, rng, retrieved=False),
    )
    gold = (
        GoldRelation("E1", "H1", "SUPPORT"),
        GoldRelation("E1", "H2", "CONTRADICT"),
        GoldRelation("E2", "H2", "SUPPORT"),
        GoldRelation("E2", "H1", "CONTRADICT"),
        GoldRelation("E3", "H1", "NEUTRAL"),
        GoldRelation("E3", "H2", "NEUTRAL"),
    )
    et = EvidenceTask(task_id=task_id, split="i3_12_s1",
        category="bilateral_conflict_h1_noise", task_summary=f"Determine {subject}.",
        high_stakes=True, budget_profile="STANDARD", hypotheses=h,
        evidence_items=ev, retrieve_exposes=(), search_exposes=("E3",),
        oracle_resolution_path=("VERIFY:E1", "VERIFY:E2", "DEFER"),
        expected_terminal=DecisionAction.DEFER, correct_hypothesis_id="H2")
    return SemanticTask(et, gold)


def gen_triple_all_eliminated(task_id, subject, rng):
    """3 hyps, all eliminated by verified contradictions. T2 should fire."""
    h = _make_hyps(3, subject)
    sa, sb, sc = rng.sample(SOURCES, 3)
    e1_prop = _fill(_pick(rng, SUPPORT_H1_TEMPLATES), subject, sa)
    e2_prop = _fill(_pick(rng, SUPPORT_H2_TEMPLATES), subject, sb)
    e3_prop = _fill(_pick(rng, CONTRADICT_H1_TEMPLATES), subject, sc)
    ev = (
        _suff("E1", e1_prop, ("H1",), ("H2",)),
        _suff("E2", e2_prop, ("H2",), ("H1",)),
        _suff("E3", e3_prop, (), ("H1",)),
    )
    # E3 contradicts H1; H3 is eliminated by being unaddressed + conflict
    # Actually need E3 to contradict H3 for T2 to fire on all 3
    e3_prop = f"Source {sc} contradicts the claim that {subject} is ambiguous (hypothesis 3)."
    ev = (
        _suff("E1", e1_prop, ("H1",), ("H2",)),
        _suff("E2", e2_prop, ("H2",), ("H1",)),
        _suff("E3", e3_prop, (), ("H3",)),
    )
    gold = (
        GoldRelation("E1", "H1", "SUPPORT"),
        GoldRelation("E1", "H2", "CONTRADICT"),
        GoldRelation("E2", "H2", "SUPPORT"),
        GoldRelation("E2", "H1", "CONTRADICT"),
        GoldRelation("E3", "H3", "CONTRADICT"),
        GoldRelation("E3", "H1", "NEUTRAL"),
        GoldRelation("E3", "H2", "NEUTRAL"),
    )
    et = EvidenceTask(task_id=task_id, split="i3_12_s1",
        category="triple_all_eliminated", task_summary=f"Determine {subject}.",
        high_stakes=True, budget_profile="STANDARD", hypotheses=h,
        evidence_items=ev, retrieve_exposes=(), search_exposes=(),
        oracle_resolution_path=("VERIFY:E1", "VERIFY:E2", "VERIFY:E3", "DEFER"),
        expected_terminal=DecisionAction.DEFER, correct_hypothesis_id="H2")
    return SemanticTask(et, gold)


def gen_quad_all_eliminated(task_id, subject, rng):
    """4 hyps, all eliminated. T2 should fire with 4 hypotheses."""
    h = _make_hyps(4, subject)
    sa, sb, sc, sd = rng.sample(SOURCES, 4)
    e1_prop = _fill(_pick(rng, SUPPORT_H1_TEMPLATES), subject, sa)
    e2_prop = _fill(_pick(rng, SUPPORT_H2_TEMPLATES), subject, sb)
    e3_prop = f"Source {sc} contradicts the claim that {subject} is ambiguous (hypothesis 3)."
    e4_prop = f"Source {sd} contradicts the claim that {subject} is ambiguous (hypothesis 4)."
    ev = (
        _suff("E1", e1_prop, ("H1",), ("H2",)),
        _suff("E2", e2_prop, ("H2",), ("H1",)),
        _suff("E3", e3_prop, (), ("H3",)),
        _suff("E4", e4_prop, (), ("H4",)),
    )
    gold = (
        GoldRelation("E1", "H1", "SUPPORT"),
        GoldRelation("E1", "H2", "CONTRADICT"),
        GoldRelation("E2", "H2", "SUPPORT"),
        GoldRelation("E2", "H1", "CONTRADICT"),
        GoldRelation("E3", "H3", "CONTRADICT"),
        GoldRelation("E4", "H4", "CONTRADICT"),
    )
    et = EvidenceTask(task_id=task_id, split="i3_12_s1",
        category="quad_all_eliminated", task_summary=f"Determine {subject}.",
        high_stakes=True, budget_profile="STANDARD", hypotheses=h,
        evidence_items=ev, retrieve_exposes=(), search_exposes=(),
        oracle_resolution_path=("VERIFY:E1", "VERIFY:E2", "VERIFY:E3", "VERIFY:E4", "DEFER"),
        expected_terminal=DecisionAction.DEFER, correct_hypothesis_id="H2")
    return SemanticTask(et, gold)


def gen_late_conflict(task_id, subject, rng):
    """Conflict appears late. T2 fires later."""
    h = _make_hyps(2, subject)
    sa, sb, sc, sd = rng.sample(SOURCES, 4)
    e1_prop = _fill(_pick(rng, SUPPORT_H1_TEMPLATES), subject, sa)
    e2_prop = _fill(_pick(rng, SUPPORT_H2_TEMPLATES), subject, sb)
    e3_prop = _fill(_pick(rng, SUPPORT_H1_TEMPLATES), subject, sc)
    e4_prop = _fill(_pick(rng, SUPPORT_H2_TEMPLATES), subject, sd)
    ev = (
        _suff("E1", e1_prop, ("H1",), ("H2",)),
        _falsified("E2", e2_prop, ("H2",), ("H1",)),
        _suff("E3", e3_prop, ("H1",), ("H2",)),
        _suff("E4", e4_prop, ("H2",), ("H1",)),
    )
    gold = (
        GoldRelation("E1", "H1", "SUPPORT"),
        GoldRelation("E1", "H2", "CONTRADICT"),
        GoldRelation("E2", "H2", "SUPPORT"),
        GoldRelation("E2", "H1", "CONTRADICT"),
        GoldRelation("E3", "H1", "SUPPORT"),
        GoldRelation("E3", "H2", "CONTRADICT"),
        GoldRelation("E4", "H2", "SUPPORT"),
        GoldRelation("E4", "H1", "CONTRADICT"),
    )
    et = EvidenceTask(task_id=task_id, split="i3_12_s1",
        category="late_conflict", task_summary=f"Determine {subject}.",
        high_stakes=True, budget_profile="STANDARD", hypotheses=h,
        evidence_items=ev, retrieve_exposes=(), search_exposes=(),
        oracle_resolution_path=("VERIFY:E1", "VERIFY:E2", "VERIFY:E3", "VERIFY:E4", "DEFER"),
        expected_terminal=DecisionAction.DEFER, correct_hypothesis_id="H2")
    return SemanticTask(et, gold)


def gen_noise_before_conflict(task_id, subject, rng):
    """Irrelevant visible evidence before conflict evidence."""
    h = _make_hyps(2, subject)
    sc, sd = rng.sample(SOURCES, 2)
    e1 = _noise("E1", subject, rng, retrieved=True)
    e2 = _noise("E2", subject, rng, retrieved=True)
    e3_prop = _fill(_pick(rng, SUPPORT_H1_TEMPLATES), subject, sc)
    e4_prop = _fill(_pick(rng, SUPPORT_H2_TEMPLATES), subject, sd)
    ev = (
        e1,
        e2,
        _suff("E3", e3_prop, ("H1",), ("H2",)),
        _suff("E4", e4_prop, ("H2",), ("H1",)),
    )
    gold = (
        GoldRelation("E1", "H1", "NEUTRAL"),
        GoldRelation("E1", "H2", "NEUTRAL"),
        GoldRelation("E2", "H1", "NEUTRAL"),
        GoldRelation("E2", "H2", "NEUTRAL"),
        GoldRelation("E3", "H1", "SUPPORT"),
        GoldRelation("E3", "H2", "CONTRADICT"),
        GoldRelation("E4", "H2", "SUPPORT"),
        GoldRelation("E4", "H1", "CONTRADICT"),
    )
    et = EvidenceTask(task_id=task_id, split="i3_12_s1",
        category="noise_before_conflict", task_summary=f"Determine {subject}.",
        high_stakes=True, budget_profile="STANDARD", hypotheses=h,
        evidence_items=ev, retrieve_exposes=(), search_exposes=(),
        oracle_resolution_path=("VERIFY:E3", "VERIFY:E4", "DEFER"),
        expected_terminal=DecisionAction.DEFER, correct_hypothesis_id="H2")
    return SemanticTask(et, gold)


def gen_subset_eliminated(task_id, subject, rng):
    """3 hyps, only H2 and H3 eliminated. H1 remains viable. T2 should NOT fire."""
    h = _make_hyps(3, subject)
    sa, sb = rng.sample(SOURCES, 2)
    e1_prop = _fill(_pick(rng, SUPPORT_H1_TEMPLATES), subject, sa)
    e2_prop = f"Source {sb} contradicts the claim that {subject} is ambiguous (hypothesis 3)."
    ev = (
        _suff("E1", e1_prop, ("H1",), ("H2",)),
        _suff("E2", e2_prop, (), ("H3",)),
    )
    gold = (
        GoldRelation("E1", "H1", "SUPPORT"),
        GoldRelation("E1", "H2", "CONTRADICT"),
        GoldRelation("E1", "H3", "NEUTRAL"),
        GoldRelation("E2", "H3", "CONTRADICT"),
        GoldRelation("E2", "H1", "NEUTRAL"),
        GoldRelation("E2", "H2", "NEUTRAL"),
    )
    et = EvidenceTask(task_id=task_id, split="i3_12_s1",
        category="subset_eliminated", task_summary=f"Determine {subject}.",
        high_stakes=True, budget_profile="STANDARD", hypotheses=h,
        evidence_items=ev, retrieve_exposes=(), search_exposes=(),
        oracle_resolution_path=("VERIFY:E1", "VERIFY:E2", "ANSWER"),
        expected_terminal=DecisionAction.ANSWER, correct_hypothesis_id="H1")
    return SemanticTask(et, gold)


def gen_simple_answer(task_id, subject, rng):
    """2 hyps, H1 clearly supported. T2 should NOT fire. Correct = ANSWER."""
    h = _make_hyps(2, subject)
    sa = rng.choice(SOURCES)
    e1_prop = _fill(_pick(rng, SUPPORT_H1_TEMPLATES), subject, sa)
    e2_prop = _fill(_pick(rng, SUPPORT_H2_TEMPLATES), subject, rng.choice(SOURCES))
    ev = (
        _suff("E1", e1_prop, ("H1",), ("H2",)),
        _falsified("E2", e2_prop, ("H2",), ("H1",)),
    )
    gold = (
        GoldRelation("E1", "H1", "SUPPORT"),
        GoldRelation("E1", "H2", "CONTRADICT"),
        GoldRelation("E2", "H2", "SUPPORT"),
        GoldRelation("E2", "H1", "CONTRADICT"),
    )
    et = EvidenceTask(task_id=task_id, split="i3_12_s1",
        category="simple_answer", task_summary=f"Determine {subject}.",
        high_stakes=True, budget_profile="STANDARD", hypotheses=h,
        evidence_items=ev, retrieve_exposes=(), search_exposes=(),
        oracle_resolution_path=("VERIFY:E1", "ANSWER"),
        expected_terminal=DecisionAction.ANSWER, correct_hypothesis_id="H1")
    return SemanticTask(et, gold)


def gen_falsified_contradiction(task_id, subject, rng):
    """2 hyps, apparent contradiction is falsified on verification. H1 wins."""
    h = _make_hyps(2, subject)
    sa, sb = rng.sample(SOURCES, 2)
    e1_prop = _fill(_pick(rng, SUPPORT_H1_TEMPLATES), subject, sa)
    e2_prop = _fill(_pick(rng, CONTRADICT_H1_TEMPLATES), subject, sb)
    ev = (
        _suff("E1", e1_prop, ("H1",), ("H2",)),
        _falsified("E2", e2_prop, ("H2",), ("H1",)),
    )
    gold = (
        GoldRelation("E1", "H1", "SUPPORT"),
        GoldRelation("E1", "H2", "CONTRADICT"),
        GoldRelation("E2", "H2", "SUPPORT"),
        GoldRelation("E2", "H1", "CONTRADICT"),
    )
    et = EvidenceTask(task_id=task_id, split="i3_12_s1",
        category="falsified_contradiction", task_summary=f"Determine {subject}.",
        high_stakes=True, budget_profile="STANDARD", hypotheses=h,
        evidence_items=ev, retrieve_exposes=(), search_exposes=(),
        oracle_resolution_path=("VERIFY:E1", "VERIFY:E2", "ANSWER"),
        expected_terminal=DecisionAction.ANSWER, correct_hypothesis_id="H1")
    return SemanticTask(et, gold)


def gen_stale_then_current(task_id, subject, rng):
    """2 hyps, stale evidence first, then current. H1 wins after verification."""
    h = _make_hyps(2, subject)
    sa, sb = rng.sample(SOURCES, 2)
    e1_prop = _fill(_pick(rng, SUPPORT_H2_TEMPLATES), subject, sa)
    e2_prop = _fill(_pick(rng, SUPPORT_H1_TEMPLATES), subject, sb)
    ev = (
        EvidenceItem(
            evidence_id="E1", proposition=e1_prop,
            source_class="initial",
            supports=("H2",), contradicts=("H1",),
            verification_state=VerificationState.UNVERIFIED,
            temporal_status=TemporalStatus.STALE,
            retrieved=True, verify_result="STALE",
        ),
        _suff("E2", e2_prop, ("H1",), ("H2",)),
    )
    gold = (
        GoldRelation("E1", "H2", "SUPPORT"),
        GoldRelation("E1", "H1", "CONTRADICT"),
        GoldRelation("E2", "H1", "SUPPORT"),
        GoldRelation("E2", "H2", "CONTRADICT"),
    )
    et = EvidenceTask(task_id=task_id, split="i3_12_s1",
        category="stale_then_current", task_summary=f"Determine {subject}.",
        high_stakes=True, budget_profile="STANDARD", hypotheses=h,
        evidence_items=ev, retrieve_exposes=(), search_exposes=(),
        oracle_resolution_path=("VERIFY:E1", "VERIFY:E2", "ANSWER"),
        expected_terminal=DecisionAction.ANSWER, correct_hypothesis_id="H1")
    return SemanticTask(et, gold)


def gen_one_viable_among_eliminated(task_id, subject, rng):
    """4 hyps, H2/H3/H4 eliminated, H1 viable. T2 should NOT fire."""
    h = _make_hyps(4, subject)
    sa, sb, sc = rng.sample(SOURCES, 3)
    e1_prop = _fill(_pick(rng, SUPPORT_H1_TEMPLATES), subject, sa)
    e2_prop = f"Source {sb} contradicts the claim that {subject} is ambiguous (hypothesis 3)."
    e3_prop = f"Source {sc} contradicts the claim that {subject} is ambiguous (hypothesis 4)."
    ev = (
        _suff("E1", e1_prop, ("H1",), ("H2",)),
        _suff("E2", e2_prop, (), ("H3",)),
        _suff("E3", e3_prop, (), ("H4",)),
    )
    gold = (
        GoldRelation("E1", "H1", "SUPPORT"),
        GoldRelation("E1", "H2", "CONTRADICT"),
        GoldRelation("E2", "H3", "CONTRADICT"),
        GoldRelation("E3", "H4", "CONTRADICT"),
    )
    et = EvidenceTask(task_id=task_id, split="i3_12_s1",
        category="one_viable_among_eliminated", task_summary=f"Determine {subject}.",
        high_stakes=True, budget_profile="STANDARD", hypotheses=h,
        evidence_items=ev, retrieve_exposes=(), search_exposes=(),
        oracle_resolution_path=("VERIFY:E1", "VERIFY:E2", "VERIFY:E3", "ANSWER"),
        expected_terminal=DecisionAction.ANSWER, correct_hypothesis_id="H1")
    return SemanticTask(et, gold)


def gen_triple_verify_answer(task_id, subject, rng):
    """3 hyps, H1 supported, H2/H3 eliminated. T2 should NOT fire. ANSWER."""
    h = _make_hyps(3, subject)
    sa, sb = rng.sample(SOURCES, 2)
    e1_prop = _fill(_pick(rng, SUPPORT_H1_TEMPLATES), subject, sa)
    e2_prop = f"Source {sb} contradicts the claim that {subject} is ambiguous (hypothesis 3)."
    ev = (
        _suff("E1", e1_prop, ("H1",), ("H2",)),
        _suff("E2", e2_prop, (), ("H3",)),
    )
    gold = (
        GoldRelation("E1", "H1", "SUPPORT"),
        GoldRelation("E1", "H2", "CONTRADICT"),
        GoldRelation("E2", "H3", "CONTRADICT"),
    )
    et = EvidenceTask(task_id=task_id, split="i3_12_s1",
        category="triple_verify_answer", task_summary=f"Determine {subject}.",
        high_stakes=True, budget_profile="STANDARD", hypotheses=h,
        evidence_items=ev, retrieve_exposes=(), search_exposes=(),
        oracle_resolution_path=("VERIFY:E1", "VERIFY:E2", "ANSWER"),
        expected_terminal=DecisionAction.ANSWER, correct_hypothesis_id="H1")
    return SemanticTask(et, gold)


def gen_multi_viable(task_id, subject, rng):
    """2 hyps, both have supporting evidence. Neither eliminated. No T2.
    Expected: DEFER (insufficient discrimination)."""
    h = _make_hyps(2, subject)
    sa, sb = rng.sample(SOURCES, 2)
    e1_prop = _fill(_pick(rng, SUPPORT_H1_TEMPLATES), subject, sa)
    e2_prop = _fill(_pick(rng, CONTRADICT_H2_TEMPLATES), subject, sb)
    ev = (
        _suff("E1", e1_prop, ("H1",), ("H2",)),
        _suff("E2", e2_prop, ("H1",), ("H2",)),
    )
    gold = (
        GoldRelation("E1", "H1", "SUPPORT"),
        GoldRelation("E1", "H2", "CONTRADICT"),
        GoldRelation("E2", "H1", "SUPPORT"),
        GoldRelation("E2", "H2", "CONTRADICT"),
    )
    et = EvidenceTask(task_id=task_id, split="i3_12_s1",
        category="multi_viable", task_summary=f"Determine {subject}.",
        high_stakes=True, budget_profile="STANDARD", hypotheses=h,
        evidence_items=ev, retrieve_exposes=(), search_exposes=(),
        oracle_resolution_path=("VERIFY:E1", "VERIFY:E2", "DEFER"),
        expected_terminal=DecisionAction.DEFER, correct_hypothesis_id="H2")
    return SemanticTask(et, gold)


def gen_early_false_ready(task_id, subject, rng):
    """2 hyps, initial evidence looks sufficient for H1 but is actually falsified.
    Tests whether MDSG handles falsified evidence correctly."""
    h = _make_hyps(2, subject)
    sa, sb = rng.sample(SOURCES, 2)
    e1_prop = _fill(_pick(rng, SUPPORT_H1_TEMPLATES), subject, sa)
    e2_prop = _fill(_pick(rng, SUPPORT_H2_TEMPLATES), subject, sb)
    ev = (
        _falsified("E1", e1_prop, ("H1",), ("H2",)),
        _suff("E2", e2_prop, ("H2",), ("H1",)),
    )
    gold = (
        GoldRelation("E1", "H1", "SUPPORT"),
        GoldRelation("E1", "H2", "CONTRADICT"),
        GoldRelation("E2", "H2", "SUPPORT"),
        GoldRelation("E2", "H1", "CONTRADICT"),
    )
    et = EvidenceTask(task_id=task_id, split="i3_12_s1",
        category="early_false_ready", task_summary=f"Determine {subject}.",
        high_stakes=True, budget_profile="STANDARD", hypotheses=h,
        evidence_items=ev, retrieve_exposes=(), search_exposes=(),
        oracle_resolution_path=("VERIFY:E1", "VERIFY:E2", "DEFER"),
        expected_terminal=DecisionAction.DEFER, correct_hypothesis_id="H2")
    return SemanticTask(et, gold)


# ---------------------------------------------------------------------------
# Corpus generation
# ---------------------------------------------------------------------------

GENERATORS = {
    "bilateral_conflict_h0": gen_bilateral_conflict_h0,
    "bilateral_conflict_h1_noise": gen_bilateral_conflict_h1_noise,
    "triple_all_eliminated": gen_triple_all_eliminated,
    "quad_all_eliminated": gen_quad_all_eliminated,
    "late_conflict": gen_late_conflict,
    "noise_before_conflict": gen_noise_before_conflict,
    "subset_eliminated": gen_subset_eliminated,
    "simple_answer": gen_simple_answer,
    "falsified_contradiction": gen_falsified_contradiction,
    "stale_then_current": gen_stale_then_current,
    "one_viable_among_eliminated": gen_one_viable_among_eliminated,
    "triple_verify_answer": gen_triple_verify_answer,
    "multi_viable": gen_multi_viable,
    "early_false_ready": gen_early_false_ready,
}


def generate_i3_12_corpus(
    n_per_category: int = 20,
    seed: int = 42,
    split: str = "i3_12_s1",
) -> list[SemanticTask]:
    """Generate the I3.12 S1 controlled raw-semantic corpus.

    Args:
        n_per_category: number of tasks per category
        seed: random seed for reproducibility
        split: split name for task IDs

    Returns:
        List of SemanticTask with gold relations
    """
    rng = random.Random(seed)
    tasks: list[SemanticTask] = []
    task_counter = 0

    for cat_name, gen_func in GENERATORS.items():
        for _ in range(n_per_category):
            task_id = f"i3_12_s1_{task_counter:04d}"
            subject = rng.choice(SUBJECTS)
            task = gen_func(task_id, subject, rng)
            tasks.append(task)
            task_counter += 1

    return tasks
