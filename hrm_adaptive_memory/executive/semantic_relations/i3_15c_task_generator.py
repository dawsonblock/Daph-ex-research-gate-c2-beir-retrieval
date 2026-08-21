"""I3.15c: Phase-Eligible Balanced Benchmark (i3_15_t2_eligible).

Creates tasks across three causal strata:
  A. ANSWER_CONTROL: correct terminal = ANSWER, T2 must never fire
  B. DEFER_CONTROL: correct terminal = DEFER, T2 must never fire
     (evidence insufficient/falsified, but NOT all hypotheses eliminated)
  C. T2_CONFLICT: correct terminal = DEFER, T2 should fire after verification
     (all hypotheses become eliminated via SUFFICIENT contradicting evidence)

The key design requirement is:
  0 < P(T2=1) < 1

T2_CONFLICT tasks use genuine conflicting evidence where both hypotheses
have SUFFICIENT, CURRENT evidence contradicting them. After verification:
  H1 -> ELIMINATED (SUFFICIENT evidence contradicts H1)
  H2 -> ELIMINATED (SUFFICIENT evidence contradicts H2)
  therefore T2=1

The epistemic conclusion is NOT "pick one report" but rather:
  "the current hypothesis set cannot be resolved consistently from
   the verified evidence."

That is exactly the operating regime where M3 should provide value.

Design axes:
  RetrievalDifficulty: easy / hard
  EpistemicDifficulty: easy / medium / hard
  PhaseEligibility: T2-positive / T2-negative

For the first clean confirmation:
  25 T2-positive easy retrieval
  25 T2-positive hard retrieval
  25 matched DEFER_CONTROL easy retrieval
  25 matched DEFER_CONTROL hard retrieval
  25 ANSWER_CONTROL easy retrieval
  25 ANSWER_CONTROL hard retrieval
  = 150 tasks
"""
from __future__ import annotations

import random
from typing import Any

from hrm_adaptive_memory.cognitive_control.core import DecisionAction
from hrm_adaptive_memory.cognitive_control.state import (
    TemporalStatus, VerificationState,
)
from hrm_adaptive_memory.executive.evidence_benchmark import (
    EvidenceItem, EvidenceTask, EvidenceHypothesis,
)
from hrm_adaptive_memory.executive.semantic_relations.raw_semantic_generator import (
    SemanticTask, GoldRelation,
)
from hrm_adaptive_memory.executive.semantic_relations.i3_15_epistemic_corpus import (
    I3_15Passage, EASY_PASSAGES, MEDIUM_PASSAGES, HARD_PASSAGES,
    get_corpus,
)


# ---------------------------------------------------------------------------
# T2_CONFLICT evidence texts: genuine conflicting reports
# ---------------------------------------------------------------------------

# SUFFICIENT evidence that contradicts H1 (operational):
# authoritative reports saying the service is NOT operational.
CONFLICT_H1_EVIDENCE = {
    "api_gateway": "An authoritative monitoring probe confirms all API gateway requests fail before reaching the service. The API gateway is offline.",
    "database": "An authoritative integrity probe confirms the database is rejecting all queries with corruption errors. The database is offline.",
    "cdn": "An authoritative edge health check confirms all CDN locations are returning 5xx errors. The CDN is offline.",
    "kubernetes": "An authoritative cluster probe confirms all Kubernetes nodes are in NotReady state with container runtime failures. The Kubernetes cluster is offline.",
    "security": "An authoritative security audit confirms an active breach is exfiltrating data. The security posture is offline.",
    "deployment": "An authoritative deployment probe confirms the new version is crashing on every request. The deployment is offline.",
    "monitoring": "An authoritative health check confirms the monitoring system has lost all data ingestion pipelines. The monitoring system is offline.",
    "cache": "An authoritative cache probe confirms the Redis cluster is rejecting all connections with OOM errors. The Redis cache is offline.",
    "message_queue": "An authoritative queue probe confirms the message queue has exhausted all consumer connections. The message queue is offline.",
    "load_balancer": "An authoritative LB probe confirms the load balancer is returning 503 for all requests with no healthy backends. The load balancer is offline.",
}

# SUFFICIENT evidence that contradicts H2 (not operational):
# authoritative reports saying the service IS operational.
CONFLICT_H2_EVIDENCE = {
    "api_gateway": "An authoritative synthetic monitoring probe confirms the API gateway is accepting traffic successfully with all endpoints responding within normal parameters. The API gateway is definitively operational.",
    "database": "An authoritative connection test confirms the database is accepting queries with normal response times across all replicas. The database is definitively operational.",
    "cdn": "An authoritative edge test confirms all CDN locations are serving cached content with normal latency. The CDN is definitively operational.",
    "kubernetes": "An authoritative cluster status check confirms all Kubernetes nodes are Ready and pods are scheduled normally. The Kubernetes cluster is definitively operational.",
    "security": "An authoritative security scan confirms all vulnerabilities have been patched and no active threats are detected. The security posture is definitively confirmed.",
    "deployment": "An authoritative deployment verification confirms version 2.4.1 is serving traffic correctly with zero errors. The deployment is definitively operational.",
    "monitoring": "An authoritative pipeline check confirms all monitoring data ingestion is flowing normally with real-time updates. The monitoring system is definitively operational.",
    "cache": "An authoritative cache test confirms the Redis cluster is responding to all commands with normal memory usage. The Redis cache is definitively operational.",
    "message_queue": "An authoritative queue check confirms the message queue is processing messages with no backlog and all consumers are active. The message queue is definitively operational.",
    "load_balancer": "An authoritative LB test confirms the load balancer is distributing traffic evenly across all healthy backends. The load balancer is definitively operational.",
}


# ---------------------------------------------------------------------------
# Conflict corpus passages: added to the base corpus for T2_CONFLICT tasks
# ---------------------------------------------------------------------------

def _build_conflict_passages() -> list[I3_15Passage]:
    """Build corpus passages for T2_CONFLICT evidence.

    These passages are added to the base I3.15 corpus so that retrieval
    can find them.  Each domain gets two conflict passages:
      - one that contradicts H1 (service NOT operational)
      - one that contradicts H2 (service IS operational)
    Both are SUFFICIENT when verified, so both hypotheses get eliminated.
    """
    passages = []
    pid_counter = 0
    for domain, text_h1 in CONFLICT_H1_EVIDENCE.items():
        text_h2 = CONFLICT_H2_EVIDENCE[domain]
        passages.append(I3_15Passage(
            passage_id=f"CF{pid_counter:03d}",
            text=text_h1,
            source="authoritative_probe",
            domain=domain,
            timestamp="2024-06-10T09:00Z",
            gold_relation="CONTRADICT",  # contradicts H1
        ))
        pid_counter += 1
        passages.append(I3_15Passage(
            passage_id=f"CF{pid_counter:03d}",
            text=text_h2,
            source="authoritative_probe",
            domain=domain,
            timestamp="2024-06-10T09:00Z",
            gold_relation="SUPPORT",  # supports H1, contradicts H2
        ))
        pid_counter += 1
    return passages


CONFLICT_PASSAGES = _build_conflict_passages()


def get_i3_15c_corpus() -> list[I3_15Passage]:
    """Return the combined corpus: base I3.15 passages + conflict passages."""
    return get_corpus() + CONFLICT_PASSAGES


DOMAIN_SUBJECTS = {
    "api_gateway": "the API gateway",
    "database": "the database",
    "cdn": "the CDN",
    "kubernetes": "the Kubernetes cluster",
    "security": "the security posture",
    "deployment": "the deployment",
    "monitoring": "the monitoring system",
    "cache": "the Redis cache",
    "message_queue": "the message queue",
    "load_balancer": "the load balancer",
}

QUERY_EASY = "Determine the current status of {subject}."
QUERY_HARD = "Check whether {subject} is up and serving traffic right now."


def _make_hyps(subject: str) -> tuple[EvidenceHypothesis, ...]:
    h1 = EvidenceHypothesis(
        hypothesis_id="H1",
        proposition=f"{subject.capitalize()} is currently operational and confirmed.",
        answer_action=DecisionAction.ANSWER,
        answer_payload=f"confirmed: {subject}",
    )
    h2 = EvidenceHypothesis(
        hypothesis_id="H2",
        proposition=f"{subject.capitalize()} is currently unavailable or unconfirmed.",
        answer_action=DecisionAction.DEFER,
        answer_payload=f"insufficient evidence: {subject}",
    )
    return (h1, h2)


def _passage_to_evidence(
    passage: I3_15Passage,
    evidence_id: str,
    retrieved: bool = True,
) -> EvidenceItem:
    """Convert a passage to an EvidenceItem with gold relations."""
    supports = ()
    contradicts = ()
    vstate = VerificationState.MISSING
    vr = "MISSING"

    if passage.gold_relation == "SUPPORT":
        supports = ("H1",)
        contradicts = ("H2",)
        vstate = VerificationState.SUFFICIENT
        vr = "SUFFICIENT"
    elif passage.gold_relation == "CONTRADICT":
        supports = ("H2",)
        contradicts = ("H1",)
        vstate = VerificationState.FALSIFIED
        vr = "FALSIFIED"
    elif passage.gold_relation == "CONDITIONAL":
        vstate = VerificationState.MISSING
        vr = "MISSING"
    elif passage.gold_relation == "TEMPORAL":
        vstate = VerificationState.MISSING
        vr = "MISSING"

    return EvidenceItem(
        evidence_id=evidence_id,
        proposition=passage.text,
        source_class="primary" if retrieved else "search",
        supports=supports,
        contradicts=contradicts,
        verification_state=vstate,
        temporal_status=TemporalStatus.CURRENT,
        retrieved=retrieved,
        verify_result=vr,
    )


def _passage_gold_relations(
    passage: I3_15Passage,
    evidence_id: str,
) -> tuple[GoldRelation, ...]:
    if passage.gold_relation == "SUPPORT":
        return (
            GoldRelation(evidence_id, "H1", "SUPPORT"),
            GoldRelation(evidence_id, "H2", "CONTRADICT"),
        )
    elif passage.gold_relation == "CONTRADICT":
        return (
            GoldRelation(evidence_id, "H1", "CONTRADICT"),
            GoldRelation(evidence_id, "H2", "SUPPORT"),
        )
    else:
        return (
            GoldRelation(evidence_id, "H1", "NEUTRAL"),
            GoldRelation(evidence_id, "H2", "NEUTRAL"),
        )


# ---------------------------------------------------------------------------
# T2_CONFLICT: Genuine conflict where both hypotheses are eliminated
# ---------------------------------------------------------------------------


def _make_conflict_evidence(
    domain: str,
    evidence_id_1: str,
    evidence_id_2: str,
    retrieved: bool = True,
) -> tuple[EvidenceItem, EvidenceItem]:
    """Create two SUFFICIENT evidence items that contradict both hypotheses.

    Evidence 1: SUFFICIENT, contradicts H1 (says service is NOT operational)
    Evidence 2: SUFFICIENT, contradicts H2 (says service IS operational)

    After both are verified:
      H1 -> ELIMINATED (SUFFICIENT evidence contradicts H1)
      H2 -> ELIMINATED (SUFFICIENT evidence contradicts H2)
      therefore T2=1
    """
    text_1 = CONFLICT_H1_EVIDENCE.get(domain, CONFLICT_H1_EVIDENCE["api_gateway"])
    text_2 = CONFLICT_H2_EVIDENCE.get(domain, CONFLICT_H2_EVIDENCE["api_gateway"])

    ev1 = EvidenceItem(
        evidence_id=evidence_id_1,
        proposition=text_1,
        source_class="primary" if retrieved else "search",
        supports=("H2",),  # Supports H2 (not operational)
        contradicts=("H1",),  # Contradicts H1 (operational)
        verification_state=VerificationState.SUFFICIENT,
        temporal_status=TemporalStatus.CURRENT,
        retrieved=retrieved,
        verify_result="SUFFICIENT",
    )
    ev2 = EvidenceItem(
        evidence_id=evidence_id_2,
        proposition=text_2,
        source_class="primary" if retrieved else "search",
        supports=("H1",),  # Supports H1 (operational)
        contradicts=("H2",),  # Contradicts H2 (not operational)
        verification_state=VerificationState.SUFFICIENT,
        temporal_status=TemporalStatus.CURRENT,
        retrieved=retrieved,
        verify_result="SUFFICIENT",
    )
    return ev1, ev2


def _find_conflict_passages(domain: str) -> tuple[I3_15Passage, I3_15Passage]:
    """Find the two conflict passages for a domain from the conflict corpus."""
    domain_conflict = [p for p in CONFLICT_PASSAGES if p.domain == domain]
    # First CONTRADICT passage (contradicts H1), first SUPPORT passage (contradicts H2)
    p_h1 = next(p for p in domain_conflict if p.gold_relation == "CONTRADICT")
    p_h2 = next(p for p in domain_conflict if p.gold_relation == "SUPPORT")
    return p_h1, p_h2


def _gen_t2_conflict_task(
    task_id: str,
    domain: str,
    rng: random.Random,
    retrieval_hard: bool,
    task_index: int = 0,
) -> SemanticTask:
    """T2_CONFLICT: both hypotheses eliminated via SUFFICIENT contradicting evidence.

    Expected terminal: DEFER (the hypothesis set cannot be resolved consistently)
    T2 should fire after both evidence items are verified.
    """
    subject = DOMAIN_SUBJECTS.get(domain, domain)
    h = _make_hyps(subject)

    ev1, ev2 = _make_conflict_evidence(domain, "E1", "E2", retrieved=True)
    ev = (ev1, ev2)

    gold = (
        GoldRelation("E1", "H1", "CONTRADICT"),
        GoldRelation("E1", "H2", "SUPPORT"),
        GoldRelation("E2", "H1", "SUPPORT"),
        GoldRelation("E2", "H2", "CONTRADICT"),
    )

    query_template = QUERY_HARD if retrieval_hard else QUERY_EASY
    query = query_template.format(subject=subject)

    et = EvidenceTask(
        task_id=task_id, split="i3_15c",
        category=f"t2_conflict_immediate_{'retr_hard' if retrieval_hard else 'retr_easy'}",
        task_summary=query,
        high_stakes=True, budget_profile="STANDARD",
        hypotheses=h, evidence_items=ev,
        retrieve_exposes=(), search_exposes=(),
        oracle_resolution_path=("VERIFY:E1", "VERIFY:E2", "DEFER"),
        expected_terminal=DecisionAction.DEFER, correct_hypothesis_id="H2",
    )
    return SemanticTask(et, gold, tier="", semantic_class="real_evidence")


def _gen_t2_conflict_late_task(
    task_id: str,
    domain: str,
    rng: random.Random,
    retrieval_hard: bool,
    task_index: int = 0,
) -> SemanticTask:
    """T2_CONFLICT_LATE: T2 fires mid-trajectory after verification.

    One evidence item (E1) starts SUFFICIENT (eliminates H1 immediately).
    The other (E2) starts UNVERIFIED — it must be VERIFYed first.
    After VERIFY(E2), E2 becomes SUFFICIENT and eliminates H2.
    Then T2 fires.

    Trajectory:
      step 0: E1 SUFFICIENT (H1 eliminated), E2 UNVERIFIED (H2 not eliminated)
              T2 = False → A1
      step 1: VERIFY(E2)
      step 2: E2 becomes SUFFICIENT (H2 eliminated)
              T2 = True → A1→M3 switch
      step 3+: M3 representation

    Expected terminal: DEFER
    """
    subject = DOMAIN_SUBJECTS.get(domain, domain)
    h = _make_hyps(subject)

    # E1: SUFFICIENT from the start, contradicts H1
    text_1 = CONFLICT_H1_EVIDENCE.get(domain, CONFLICT_H1_EVIDENCE["api_gateway"])
    ev1 = EvidenceItem(
        evidence_id="E1",
        proposition=text_1,
        source_class="primary",
        supports=("H2",),
        contradicts=("H1",),
        verification_state=VerificationState.SUFFICIENT,
        temporal_status=TemporalStatus.CURRENT,
        retrieved=True,
        verify_result="SUFFICIENT",
    )

    # E2: UNVERIFIED at start, becomes SUFFICIENT after VERIFY, contradicts H2
    text_2 = CONFLICT_H2_EVIDENCE.get(domain, CONFLICT_H2_EVIDENCE["api_gateway"])
    ev2 = EvidenceItem(
        evidence_id="E2",
        proposition=text_2,
        source_class="primary",
        supports=("H1",),
        contradicts=("H2",),
        verification_state=VerificationState.UNVERIFIED,  # starts unverified
        temporal_status=TemporalStatus.CURRENT,
        retrieved=True,
        verify_result="SUFFICIENT",  # becomes SUFFICIENT after VERIFY
    )

    ev = (ev1, ev2)
    gold = (
        GoldRelation("E1", "H1", "CONTRADICT"),
        GoldRelation("E1", "H2", "SUPPORT"),
        GoldRelation("E2", "H1", "SUPPORT"),
        GoldRelation("E2", "H2", "CONTRADICT"),
    )

    query_template = QUERY_HARD if retrieval_hard else QUERY_EASY
    query = query_template.format(subject=subject)

    et = EvidenceTask(
        task_id=task_id, split="i3_15c",
        category=f"t2_conflict_late_1_{'retr_hard' if retrieval_hard else 'retr_easy'}",
        task_summary=query,
        high_stakes=True, budget_profile="STANDARD",
        hypotheses=h, evidence_items=ev,
        retrieve_exposes=(), search_exposes=(),
        oracle_resolution_path=("VERIFY:E2", "DEFER"),
        expected_terminal=DecisionAction.DEFER, correct_hypothesis_id="H2",
    )
    return SemanticTask(et, gold, tier="", semantic_class="real_evidence")


def _gen_t2_conflict_late_2_task(
    task_id: str,
    domain: str,
    rng: random.Random,
    retrieval_hard: bool,
    task_index: int = 0,
) -> SemanticTask:
    """T2_CONFLICT_LATE_2: T2 fires after TWO sequential verifications.

    Both evidence items start UNVERIFIED.
    E1 becomes SUFFICIENT after VERIFY (contradicts H1 → H1 eliminated).
    E2 becomes SUFFICIENT after VERIFY (contradicts H2 → H2 eliminated).
    T2 fires only after BOTH are verified.

    Trajectory:
      step 0: E1 UNVERIFIED, E2 UNVERIFIED → T2 = False → A1
      step 1: VERIFY(E1) → E1 SUFFICIENT, H1 eliminated. T2 still False (H2 alive)
      step 2: VERIFY(E2) → E2 SUFFICIENT, H2 eliminated. T2 = True → M3
      step 3+: M3 representation

    Expected terminal: DEFER
    """
    subject = DOMAIN_SUBJECTS.get(domain, domain)
    h = _make_hyps(subject)

    text_1 = CONFLICT_H1_EVIDENCE.get(domain, CONFLICT_H1_EVIDENCE["api_gateway"])
    text_2 = CONFLICT_H2_EVIDENCE.get(domain, CONFLICT_H2_EVIDENCE["api_gateway"])

    ev1 = EvidenceItem(
        evidence_id="E1",
        proposition=text_1,
        source_class="primary",
        supports=("H2",),
        contradicts=("H1",),
        verification_state=VerificationState.UNVERIFIED,
        temporal_status=TemporalStatus.CURRENT,
        retrieved=True,
        verify_result="SUFFICIENT",
    )
    ev2 = EvidenceItem(
        evidence_id="E2",
        proposition=text_2,
        source_class="primary",
        supports=("H1",),
        contradicts=("H2",),
        verification_state=VerificationState.UNVERIFIED,
        temporal_status=TemporalStatus.CURRENT,
        retrieved=True,
        verify_result="SUFFICIENT",
    )

    ev = (ev1, ev2)
    gold = (
        GoldRelation("E1", "H1", "CONTRADICT"),
        GoldRelation("E1", "H2", "SUPPORT"),
        GoldRelation("E2", "H1", "SUPPORT"),
        GoldRelation("E2", "H2", "CONTRADICT"),
    )

    query_template = QUERY_HARD if retrieval_hard else QUERY_EASY
    query = query_template.format(subject=subject)

    et = EvidenceTask(
        task_id=task_id, split="i3_15c",
        category=f"t2_conflict_late_2_{'retr_hard' if retrieval_hard else 'retr_easy'}",
        task_summary=query,
        high_stakes=True, budget_profile="STANDARD",
        hypotheses=h, evidence_items=ev,
        retrieve_exposes=(), search_exposes=(),
        oracle_resolution_path=("VERIFY:E1", "VERIFY:E2", "DEFER"),
        expected_terminal=DecisionAction.DEFER, correct_hypothesis_id="H2",
    )
    return SemanticTask(et, gold, tier="", semantic_class="real_evidence")


def _gen_t2_conflict_late_3_task(
    task_id: str,
    domain: str,
    rng: random.Random,
    retrieval_hard: bool,
    task_index: int = 0,
) -> SemanticTask:
    """T2_CONFLICT_LATE_3_PLUS: T2 fires after RETRIEVE + VERIFY.

    E1 starts SUFFICIENT (contradicts H1 → H1 eliminated at step 0).
    E2 is hidden — must be RETRIEVE'd first, then VERIFY'd.
    After RETRIEVE + VERIFY(E2), E2 becomes SUFFICIENT (contradicts H2).
    T2 fires after retrieval + verification transition.

    Trajectory:
      step 0: E1 SUFFICIENT (H1 eliminated), E2 hidden → T2 = False → A1
      step 1: RETRIEVE → E2 exposed as UNVERIFIED
      step 2: VERIFY(E2) → E2 SUFFICIENT, H2 eliminated → T2 = True → M3
      step 3+: M3 representation

    Expected terminal: DEFER
    """
    subject = DOMAIN_SUBJECTS.get(domain, domain)
    h = _make_hyps(subject)

    text_1 = CONFLICT_H1_EVIDENCE.get(domain, CONFLICT_H1_EVIDENCE["api_gateway"])
    text_2 = CONFLICT_H2_EVIDENCE.get(domain, CONFLICT_H2_EVIDENCE["api_gateway"])

    ev1 = EvidenceItem(
        evidence_id="E1",
        proposition=text_1,
        source_class="primary",
        supports=("H2",),
        contradicts=("H1",),
        verification_state=VerificationState.SUFFICIENT,
        temporal_status=TemporalStatus.CURRENT,
        retrieved=True,
        verify_result="SUFFICIENT",
    )
    # E2 is hidden — exposed by RETRIEVE, then UNVERIFIED
    ev2 = EvidenceItem(
        evidence_id="E2",
        proposition=text_2,
        source_class="primary",
        supports=("H1",),
        contradicts=("H2",),
        verification_state=VerificationState.UNVERIFIED,
        temporal_status=TemporalStatus.CURRENT,
        retrieved=False,  # hidden initially
        verify_result="SUFFICIENT",
    )

    ev = (ev1, ev2)
    gold = (
        GoldRelation("E1", "H1", "CONTRADICT"),
        GoldRelation("E1", "H2", "SUPPORT"),
        GoldRelation("E2", "H1", "SUPPORT"),
        GoldRelation("E2", "H2", "CONTRADICT"),
    )

    query_template = QUERY_HARD if retrieval_hard else QUERY_EASY
    query = query_template.format(subject=subject)

    et = EvidenceTask(
        task_id=task_id, split="i3_15c",
        category=f"t2_conflict_late_3_{'retr_hard' if retrieval_hard else 'retr_easy'}",
        task_summary=query,
        high_stakes=True, budget_profile="STANDARD",
        hypotheses=h, evidence_items=ev,
        retrieve_exposes=("E2",),  # RETRIEVE exposes E2
        search_exposes=(),
        oracle_resolution_path=("RETRIEVE", "VERIFY:E2", "DEFER"),
        expected_terminal=DecisionAction.DEFER, correct_hypothesis_id="H2",
    )
    return SemanticTask(et, gold, tier="", semantic_class="real_evidence")


def _gen_defer_control_task(
    task_id: str,
    domain: str,
    passages: list[I3_15Passage],
    rng: random.Random,
    retrieval_hard: bool,
    task_index: int = 0,
) -> SemanticTask:
    """DEFER_CONTROL: evidence insufficient/falsified, T2 must never fire.

    Uses a single CONTRADICT passage (FALSIFIED after verification).
    H2 is supported but H1 is not eliminated via SUFFICIENT contradiction.
    T2 never fires because not all hypotheses are eliminated.
    """
    subject = DOMAIN_SUBJECTS.get(domain, domain)
    h = _make_hyps(subject)

    domain_passages = [p for p in passages if p.domain == domain]
    contradict_passages = [p for p in domain_passages if p.gold_relation == "CONTRADICT"]
    if not contradict_passages:
        contradict_passages = [p for p in passages if p.gold_relation == "CONTRADICT"]
    if not contradict_passages:
        # Fallback: use any passage
        contradict_passages = domain_passages or passages

    p1 = rng.choice(contradict_passages)
    ev = (_passage_to_evidence(p1, "E1", retrieved=True),)
    gold = _passage_gold_relations(p1, "E1")

    query_template = QUERY_HARD if retrieval_hard else QUERY_EASY
    query = query_template.format(subject=subject)

    et = EvidenceTask(
        task_id=task_id, split="i3_15c",
        category=f"defer_control_{'retr_hard' if retrieval_hard else 'retr_easy'}",
        task_summary=query,
        high_stakes=True, budget_profile="STANDARD",
        hypotheses=h, evidence_items=ev,
        retrieve_exposes=(), search_exposes=(),
        oracle_resolution_path=("VERIFY:E1", "DEFER"),
        expected_terminal=DecisionAction.DEFER, correct_hypothesis_id="H2",
    )
    return SemanticTask(et, gold, tier="", semantic_class="real_evidence")


def _gen_answer_control_task(
    task_id: str,
    domain: str,
    passages: list[I3_15Passage],
    rng: random.Random,
    retrieval_hard: bool,
    task_index: int = 0,
) -> SemanticTask:
    """ANSWER_CONTROL: SUFFICIENT evidence supports H1, T2 must never fire.

    Uses a single SUPPORT passage (SUFFICIENT after verification).
    H1 is VIABLE, H2 is ELIMINATED. T2 never fires because not all
    hypotheses are eliminated (H1 is still viable).
    """
    subject = DOMAIN_SUBJECTS.get(domain, domain)
    h = _make_hyps(subject)

    domain_passages = [p for p in passages if p.domain == domain]
    support_passages = [p for p in domain_passages if p.gold_relation == "SUPPORT"]
    if not support_passages:
        support_passages = [p for p in passages if p.gold_relation == "SUPPORT"]
    if not support_passages:
        support_passages = domain_passages or passages

    p1 = rng.choice(support_passages)
    ev = (_passage_to_evidence(p1, "E1", retrieved=True),)
    gold = _passage_gold_relations(p1, "E1")

    query_template = QUERY_HARD if retrieval_hard else QUERY_EASY
    query = query_template.format(subject=subject)

    et = EvidenceTask(
        task_id=task_id, split="i3_15c",
        category=f"answer_control_{'retr_hard' if retrieval_hard else 'retr_easy'}",
        task_summary=query,
        high_stakes=True, budget_profile="STANDARD",
        hypotheses=h, evidence_items=ev,
        retrieve_exposes=(), search_exposes=(),
        oracle_resolution_path=("VERIFY:E1", "ANSWER"),
        expected_terminal=DecisionAction.ANSWER, correct_hypothesis_id="H1",
    )
    return SemanticTask(et, gold, tier="", semantic_class="real_evidence")


# ---------------------------------------------------------------------------
# Matched T2-negative controls (R11)
# ---------------------------------------------------------------------------

def _gen_matched_t2_negative_immediate(
    task_id: str,
    domain: str,
    rng: random.Random,
    retrieval_hard: bool,
    task_index: int = 0,
) -> SemanticTask:
    """Matched control for T2_CONFLICT_IMMEDIATE.

    Same domain, same query, same number of evidence items, same source count.
    Difference: E2 is UNVERIFIED (not SUFFICIENT), so H2 is NOT eliminated.
    T2 never fires because only H1 is eliminated.

    T2 POSITIVE:  E1 SUFFICIENT contradicts H1, E2 SUFFICIENT contradicts H2 → T2
    MATCHED CTRL: E1 SUFFICIENT contradicts H1, E2 UNVERIFIED → H2 alive → no T2
    """
    subject = DOMAIN_SUBJECTS.get(domain, domain)
    h = _make_hyps(subject)

    ev1, ev2_verified = _make_conflict_evidence(domain, "E1", "E2", retrieved=True)
    # Make E2 UNVERIFIED with neutral relations — H2 stays alive
    from dataclasses import replace
    ev2 = replace(ev2_verified,
                  verification_state=VerificationState.UNVERIFIED,
                  supports=(),
                  contradicts=())

    ev = (ev1, ev2)
    gold = (
        GoldRelation("E1", "H1", "CONTRADICT"),
        GoldRelation("E1", "H2", "SUPPORT"),
        GoldRelation("E2", "H1", "NEUTRAL"),
        GoldRelation("E2", "H2", "NEUTRAL"),
    )

    query_template = QUERY_HARD if retrieval_hard else QUERY_EASY
    query = query_template.format(subject=subject)

    et = EvidenceTask(
        task_id=task_id, split="i3_15c",
        category=f"matched_neg_immediate_{'retr_hard' if retrieval_hard else 'retr_easy'}",
        task_summary=query,
        high_stakes=True, budget_profile="STANDARD",
        hypotheses=h, evidence_items=ev,
        retrieve_exposes=(), search_exposes=(),
        oracle_resolution_path=("VERIFY:E2", "DEFER"),
        expected_terminal=DecisionAction.DEFER, correct_hypothesis_id="H2",
    )
    return SemanticTask(et, gold, tier="", semantic_class="real_evidence")


def _gen_matched_t2_negative_late(
    task_id: str,
    domain: str,
    rng: random.Random,
    retrieval_hard: bool,
    task_index: int = 0,
) -> SemanticTask:
    """Matched control for T2_CONFLICT_LATE_1.

    Same structure as late_1 but E2's verify_result is MISSING (not SUFFICIENT).
    After VERIFY(E2), E2 becomes MISSING — H2 is NOT eliminated.
    T2 never fires even after verification.

    T2 POSITIVE:  E1 SUFFICIENT, E2 UNVERIFIED→SUFFICIENT → both eliminated → T2
    MATCHED CTRL: E1 SUFFICIENT, E2 UNVERIFIED→MISSING → only H1 eliminated → no T2
    """
    subject = DOMAIN_SUBJECTS.get(domain, domain)
    h = _make_hyps(subject)

    text_1 = CONFLICT_H1_EVIDENCE.get(domain, CONFLICT_H1_EVIDENCE["api_gateway"])
    text_2 = CONFLICT_H2_EVIDENCE.get(domain, CONFLICT_H2_EVIDENCE["api_gateway"])

    ev1 = EvidenceItem(
        evidence_id="E1",
        proposition=text_1,
        source_class="primary",
        supports=("H2",),
        contradicts=("H1",),
        verification_state=VerificationState.SUFFICIENT,
        temporal_status=TemporalStatus.CURRENT,
        retrieved=True,
        verify_result="SUFFICIENT",
    )
    # E2: UNVERIFIED, verify_result is MISSING, and NEUTRAL relations
    # Key difference: no supports/contradicts → verifying E2 eliminates nothing
    ev2 = EvidenceItem(
        evidence_id="E2",
        proposition=text_2,
        source_class="primary",
        supports=(),  # No support — key difference
        contradicts=(),  # No contradiction — key difference
        verification_state=VerificationState.UNVERIFIED,
        temporal_status=TemporalStatus.CURRENT,
        retrieved=True,
        verify_result="MISSING",  # Not SUFFICIENT
    )

    ev = (ev1, ev2)
    gold = (
        GoldRelation("E1", "H1", "CONTRADICT"),
        GoldRelation("E1", "H2", "SUPPORT"),
        GoldRelation("E2", "H1", "NEUTRAL"),  # Neutral, not SUPPORT
        GoldRelation("E2", "H2", "NEUTRAL"),  # Neutral, not CONTRADICT
    )

    query_template = QUERY_HARD if retrieval_hard else QUERY_EASY
    query = query_template.format(subject=subject)

    et = EvidenceTask(
        task_id=task_id, split="i3_15c",
        category=f"matched_neg_late_{'retr_hard' if retrieval_hard else 'retr_easy'}",
        task_summary=query,
        high_stakes=True, budget_profile="STANDARD",
        hypotheses=h, evidence_items=ev,
        retrieve_exposes=(), search_exposes=(),
        oracle_resolution_path=("VERIFY:E2", "DEFER"),
        expected_terminal=DecisionAction.DEFER, correct_hypothesis_id="H2",
    )
    return SemanticTask(et, gold, tier="", semantic_class="real_evidence")


# ---------------------------------------------------------------------------
# Structural validator
# ---------------------------------------------------------------------------

def _simulate_t2(et, i3_7e, initial_state: bool = False) -> bool:
    """Simulate T2 firing for a task.

    If initial_state=True, use the evidence as-is (some may be UNVERIFIED).
    If initial_state=False, simulate all evidence at gold verification state.
    """
    from hrm_adaptive_memory.executive.evidence_benchmark import EvidenceSnapshot
    from hrm_adaptive_memory.executive.resources import ResourceState, ResourceBudget
    from dataclasses import replace

    if initial_state:
        evidence = et.evidence_items
    else:
        # Simulate gold state: all UNVERIFIED evidence transitions to its verify_result
        evidence = []
        for ev in et.evidence_items:
            if ev.verification_state == VerificationState.UNVERIFIED and ev.verify_result:
                evidence.append(replace(
                    ev,
                    verification_state=VerificationState(ev.verify_result),
                ))
            else:
                evidence.append(ev)

    budget = ResourceBudget()
    snapshot = EvidenceSnapshot(
        task_id=et.task_id,
        task_summary=et.task_summary,
        visible_evidence=tuple(evidence),
        hidden_evidence_count=0,
        hypotheses=et.hypotheses,
        verified_count=len([
            e for e in evidence
            if e.verification_state != VerificationState.UNVERIFIED
        ]),
        supporting_count=0,
        contradicting_count=0,
        searched=False,
        reasoning_complete=False,
        resource_state=ResourceState(budget).as_dict(),
        prior_actions=(),
        prior_outcomes=(),
        can_retrieve=False, can_search=False, can_verify=False,
    )
    viability = i3_7e._classify_from_snapshot(snapshot)
    eliminated = [
        h_id for h_id, info in viability.items()
        if info["status"] == "ELIMINATED"
    ]
    n_hyp = len(et.hypotheses)
    return len(eliminated) == n_hyp and n_hyp > 0


def validate_t2_eligibility(tasks: list[SemanticTask]) -> dict[str, Any]:
    """Validate T2 eligibility across all 4 strata.

    This is an offline structural check that requires zero LLM calls.

    Checks:
      T2_CONFLICT_IMMEDIATE: T2 fires at initial state AND gold state
      T2_CONFLICT_LATE: T2 does NOT fire at initial state, DOES fire at gold state
      DEFER_CONTROL: T2 never fires at gold state
      ANSWER_CONTROL: T2 never fires at gold state
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "i3_7e", "scripts/run_i3_7e_compact_governor.py")
    i3_7e = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(i3_7e)

    results = {
        "total_tasks": len(tasks),
        "per_stratum": {},
        "t2_positive_expected": 0,
        "t2_negative_expected": 0,
        "t2_positive_reachable_gold": 0,
        "t2_negative_incorrectly_reachable_gold": 0,
        "late_t2_initial_false": 0,
        "late_t2_initial_incorrectly_true": 0,
        "immediate_t2_initial_true": 0,
        "immediate_t2_initial_incorrectly_false": 0,
    }

    for task in tasks:
        et = task.evidence_task
        category = et.category

        if category.startswith("t2_conflict_immediate"):
            stratum = "T2_CONFLICT_IMMEDIATE"
            results["t2_positive_expected"] += 1
        elif category.startswith("t2_conflict_late"):
            stratum = "T2_CONFLICT_LATE"
            results["t2_positive_expected"] += 1
        elif category.startswith("matched_neg"):
            stratum = "MATCHED_NEG"
            results["t2_negative_expected"] += 1
        elif category.startswith("defer_control"):
            stratum = "DEFER_CONTROL"
            results["t2_negative_expected"] += 1
        elif category.startswith("answer_control"):
            stratum = "ANSWER_CONTROL"
            results["t2_negative_expected"] += 1
        else:
            stratum = "UNKNOWN"

        t2_initial = _simulate_t2(et, i3_7e, initial_state=True)
        t2_gold = _simulate_t2(et, i3_7e, initial_state=False)

        if stratum not in results["per_stratum"]:
            results["per_stratum"][stratum] = {
                "n": 0,
                "t2_initial_true": 0,
                "t2_initial_false": 0,
                "t2_gold_true": 0,
                "t2_gold_false": 0,
            }
        s = results["per_stratum"][stratum]
        s["n"] += 1
        if t2_initial:
            s["t2_initial_true"] += 1
        else:
            s["t2_initial_false"] += 1
        if t2_gold:
            s["t2_gold_true"] += 1
        else:
            s["t2_gold_false"] += 1

        # Stratum-specific checks
        if stratum == "T2_CONFLICT_IMMEDIATE":
            if t2_gold:
                results["t2_positive_reachable_gold"] += 1
            if t2_initial:
                results["immediate_t2_initial_true"] += 1
            else:
                results["immediate_t2_initial_incorrectly_false"] += 1
        elif stratum == "T2_CONFLICT_LATE":
            if t2_gold:
                results["t2_positive_reachable_gold"] += 1
            if not t2_initial:
                results["late_t2_initial_false"] += 1
            else:
                results["late_t2_initial_incorrectly_true"] += 1
        else:
            if t2_gold:
                results["t2_negative_incorrectly_reachable_gold"] += 1

    results["passed"] = (
        results["t2_positive_expected"] > 0
        and results["t2_positive_reachable_gold"] == results["t2_positive_expected"]
        and results["t2_negative_incorrectly_reachable_gold"] == 0
        and results["late_t2_initial_incorrectly_true"] == 0
        and results["immediate_t2_initial_incorrectly_false"] == 0
    )
    return results


# ---------------------------------------------------------------------------
# Corpus generation
# ---------------------------------------------------------------------------

def generate_i3_15c_corpus(
    n_per_cell: int = 25,
    seed: int = 42,
) -> list[SemanticTask]:
    """Generate the I3.15c task corpus.

    4 strata x 2 retrieval levels x n_per_cell tasks
    = 8 cells x n_per_cell tasks

    Strata:
      T2_CONFLICT_IMMEDIATE: both evidence SUFFICIENT from start, T2 fires at step 0
      T2_CONFLICT_LATE: one evidence UNVERIFIED, T2 fires after VERIFY
      DEFER_CONTROL: evidence insufficient, T2 never fires, expected=DEFER
      ANSWER_CONTROL: evidence sufficient, T2 never fires, expected=ANSWER

    Retrieval:
      easy: domain-specific query terms
      hard: abstract query terms

    With n_per_cell=25: 4 x 2 x 25 = 200 unique tasks.
    With 3 retrieval systems (Q0/Q3/Q4) x 2 arms (A1/R1):
      200 x 3 x 2 = 1200 trajectories.
    """
    rng = random.Random(seed)
    corpus = get_corpus()
    domains = list(DOMAIN_SUBJECTS.keys())

    tasks: list[SemanticTask] = []
    task_idx = 0

    for stratum in ["t2_conflict_immediate", "t2_conflict_late_1",
                    "t2_conflict_late_2", "t2_conflict_late_3",
                    "matched_neg_immediate", "matched_neg_late",
                    "defer_control", "answer_control"]:
        for retrieval_hard in [False, True]:
            for i in range(n_per_cell):
                domain = domains[task_idx % len(domains)]
                task_id = f"i3_15c_{task_idx:04d}"
                if stratum == "t2_conflict_immediate":
                    task = _gen_t2_conflict_task(
                        task_id, domain, rng, retrieval_hard, task_index=i)
                elif stratum == "t2_conflict_late_1":
                    task = _gen_t2_conflict_late_task(
                        task_id, domain, rng, retrieval_hard, task_index=i)
                elif stratum == "t2_conflict_late_2":
                    task = _gen_t2_conflict_late_2_task(
                        task_id, domain, rng, retrieval_hard, task_index=i)
                elif stratum == "t2_conflict_late_3":
                    task = _gen_t2_conflict_late_3_task(
                        task_id, domain, rng, retrieval_hard, task_index=i)
                elif stratum == "matched_neg_immediate":
                    task = _gen_matched_t2_negative_immediate(
                        task_id, domain, rng, retrieval_hard, task_index=i)
                elif stratum == "matched_neg_late":
                    task = _gen_matched_t2_negative_late(
                        task_id, domain, rng, retrieval_hard, task_index=i)
                elif stratum == "defer_control":
                    task = _gen_defer_control_task(
                        task_id, domain, corpus, rng, retrieval_hard, task_index=i)
                else:
                    task = _gen_answer_control_task(
                        task_id, domain, corpus, rng, retrieval_hard, task_index=i)
                tasks.append(task)
                task_idx += 1

    return tasks


if __name__ == "__main__":
    tasks = generate_i3_15c_corpus(n_per_cell=25, seed=42)
    print(f"Generated {len(tasks)} tasks")

    from collections import Counter
    cats = Counter(t.evidence_task.category for t in tasks)
    for cat, count in sorted(cats.items()):
        print(f"  {cat}: {count}")

    # Structural validation
    print("\nStructural T2 eligibility validation:")
    validation = validate_t2_eligibility(tasks)
    print(f"  T2 positive expected: {validation['t2_positive_expected']}")
    print(f"  T2 positive reachable (gold): {validation['t2_positive_reachable_gold']}")
    print(f"  T2 negative expected: {validation['t2_negative_expected']}")
    print(f"  T2 negative incorrectly reachable (gold): {validation['t2_negative_incorrectly_reachable_gold']}")
    print(f"  Late T2 initial false (correct): {validation['late_t2_initial_false']}")
    print(f"  Late T2 initial incorrectly true: {validation['late_t2_initial_incorrectly_true']}")
    print(f"  Immediate T2 initial true (correct): {validation['immediate_t2_initial_true']}")
    print(f"  Immediate T2 initial incorrectly false: {validation['immediate_t2_initial_incorrectly_false']}")
    print(f"  PASSED: {validation['passed']}")
    print()
    for stratum, info in sorted(validation["per_stratum"].items()):
        print(f"  {stratum}: n={info['n']} "
              f"t2_initial_true={info['t2_initial_true']} "
              f"t2_initial_false={info['t2_initial_false']} "
              f"t2_gold_true={info['t2_gold_true']} "
              f"t2_gold_false={info['t2_gold_false']}")
