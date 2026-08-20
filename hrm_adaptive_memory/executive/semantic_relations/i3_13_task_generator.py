"""I3.13: Task generator for retrieved-evidence experiments.

Creates tasks with competing hypotheses where evidence comes from
real document passages. Each task has:
  - 2-4 competing hypotheses (current/stale orientation)
  - Evidence items drawn from the document corpus
  - Evaluator-side gold relations
  - Oracle resolution path
  - Required evidence IDs (for retrieval recall calculation)

The task structure is compatible with the existing EvidenceTask
schema, but the proposition text comes from real passages instead
of controlled templates.
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
from hrm_adaptive_memory.executive.semantic_relations.raw_semantic_generator import (
    GoldRelation, SemanticTask,
)
from hrm_adaptive_memory.executive.semantic_relations.i3_13_document_corpus import (
    DocumentPassage, get_corpus,
)


# ---------------------------------------------------------------------------
# Hypothesis templates for real-world tasks
# ---------------------------------------------------------------------------

H1_PROPOSITION_TEMPLATE = (
    "{subject} is currently operational and confirmed."
)
H2_PROPOSITION_TEMPLATE = (
    "{subject} is stale, outdated, or unconfirmed."
)

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
    "feature_flags": "the feature flag configuration",
    "infrastructure": "the infrastructure",
}


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


def _passage_to_evidence(
    passage: DocumentPassage,
    evidence_id: str,
    retrieved: bool = True,
    verify_result: str = "SUFFICIENT",
) -> EvidenceItem:
    """Convert a document passage to an EvidenceItem.

    The supports/contradicts fields are populated from the passage's
    gold relations. In the GOLD condition, these are used directly.
    In the INFERRED condition, they are replaced by the extractor's
    output at runtime.

    Evidence is pre-verified to isolate semantic extraction from
    verification behavior. The verification_state reflects the
    gold relation: SUFFICIENT for support, FALSIFIED for contradiction,
    MISSING for neutral.
    """
    supports = tuple(
        "H1" if orient == "current" else "H2"
        for orient, rel in passage.gold_relations
        if rel == "SUPPORT"
    )
    contradicts = tuple(
        "H1" if orient == "current" else "H2"
        for orient, rel in passage.gold_relations
        if rel == "CONTRADICT"
    )

    # Determine verification result based on gold relations
    has_current_support = any(
        rel == "SUPPORT" and orient == "current"
        for orient, rel in passage.gold_relations
    )
    has_current_contradict = any(
        rel == "CONTRADICT" and orient == "current"
        for orient, rel in passage.gold_relations
    )

    if verify_result == "SUFFICIENT":
        vr = "SUFFICIENT"
        vstate = VerificationState.SUFFICIENT
    elif verify_result == "FALSIFIED":
        vr = "FALSIFIED"
        vstate = VerificationState.FALSIFIED
    elif verify_result == "MISSING":
        vr = "MISSING"
        vstate = VerificationState.MISSING
    elif has_current_contradict and not has_current_support:
        vr = "FALSIFIED"
        vstate = VerificationState.FALSIFIED
    elif has_current_support:
        vr = "SUFFICIENT"
        vstate = VerificationState.SUFFICIENT
    else:
        vr = "MISSING"
        vstate = VerificationState.MISSING

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
    passage: DocumentPassage,
    evidence_id: str,
    n_hyps: int,
) -> tuple[GoldRelation, ...]:
    """Convert passage gold relations to GoldRelation objects.

    Each passage has gold_relations as (orientation, relation) pairs.
    orientation "current" maps to H1, "stale" maps to H2.
    The opposite hypothesis gets the inverse relation.
    Deduplicates to avoid double-counting when both orientations are present.
    """
    rel_map: dict[str, str] = {}  # hypothesis_id -> relation

    for orient, rel in passage.gold_relations:
        if orient == "current":
            h1_rel = rel
            h2_rel = {"SUPPORT": "CONTRADICT", "CONTRADICT": "SUPPORT", "NEUTRAL": "NEUTRAL"}[rel]
        elif orient == "stale":
            h2_rel = rel
            h1_rel = {"SUPPORT": "CONTRADICT", "CONTRADICT": "SUPPORT", "NEUTRAL": "NEUTRAL"}[rel]
        else:
            continue
        # Last write wins, but they should be consistent
        rel_map["H1"] = h1_rel
        rel_map["H2"] = h2_rel

    # Ensure all hypotheses have a relation
    for i in range(1, n_hyps + 1):
        hid = f"H{i}"
        if hid not in rel_map:
            rel_map[hid] = "NEUTRAL"

    return tuple(GoldRelation(evidence_id, hid, rel) for hid, rel in sorted(rel_map.items()))


# ---------------------------------------------------------------------------
# Task generators
# ---------------------------------------------------------------------------

def gen_clear_answer_task(
    task_id: str,
    domain: str,
    passages: list[DocumentPassage],
    rng: random.Random,
) -> SemanticTask:
    """Task with clear support for H1 (current). T2 should NOT fire."""
    subject = DOMAIN_SUBJECTS.get(domain, domain)
    h = _make_hyps(2, subject)

    # Find a passage that supports current
    support_passages = [p for p in passages if p.domain == domain
                        and any(r == "SUPPORT" and o == "current" for o, r in p.gold_relations)]
    if not support_passages:
        support_passages = [p for p in passages
                           if any(r == "SUPPORT" and o == "current" for o, r in p.gold_relations)]

    p1 = rng.choice(support_passages)
    ev = (_passage_to_evidence(p1, "E1", retrieved=True, verify_result="SUFFIFIED"),)
    ev = (_passage_to_evidence(p1, "E1", retrieved=True, verify_result="SUFFICIENT"),)

    gold = _passage_gold_relations(p1, "E1", 2)

    et = EvidenceTask(
        task_id=task_id, split="i3_13",
        category="clear_answer",
        task_summary=f"Determine the current status of {subject}.",
        high_stakes=True, budget_profile="STANDARD",
        hypotheses=h, evidence_items=ev,
        retrieve_exposes=(), search_exposes=(),
        oracle_resolution_path=("VERIFY:E1", "ANSWER"),
        expected_terminal=DecisionAction.ANSWER, correct_hypothesis_id="H1",
    )
    return SemanticTask(et, gold, tier="", semantic_class="real_evidence")


def gen_conflict_task(
    task_id: str,
    domain: str,
    passages: list[DocumentPassage],
    rng: random.Random,
) -> SemanticTask:
    """Task with bilateral conflict. T2 should fire."""
    subject = DOMAIN_SUBJECTS.get(domain, domain)
    h = _make_hyps(2, subject)

    # Find one passage supporting current and one supporting stale
    current_support = [p for p in passages
                       if any(r == "SUPPORT" and o == "current" for o, r in p.gold_relations)]
    stale_support = [p for p in passages
                     if any(r == "SUPPORT" and o == "stale" for o, r in p.gold_relations)]

    p1 = rng.choice(current_support)
    p2 = rng.choice(stale_support)

    ev = (
        _passage_to_evidence(p1, "E1", retrieved=True, verify_result="SUFFICIENT"),
        _passage_to_evidence(p2, "E2", retrieved=True, verify_result="SUFFICIENT"),
    )
    gold = _passage_gold_relations(p1, "E1", 2) + _passage_gold_relations(p2, "E2", 2)

    et = EvidenceTask(
        task_id=task_id, split="i3_13",
        category="bilateral_conflict",
        task_summary=f"Determine the current status of {subject}.",
        high_stakes=True, budget_profile="STANDARD",
        hypotheses=h, evidence_items=ev,
        retrieve_exposes=(), search_exposes=(),
        oracle_resolution_path=("VERIFY:E1", "VERIFY:E2", "DEFER"),
        expected_terminal=DecisionAction.DEFER, correct_hypothesis_id="H2",
    )
    return SemanticTask(et, gold, tier="", semantic_class="real_evidence")


def gen_degraded_task(
    task_id: str,
    domain: str,
    passages: list[DocumentPassage],
    rng: random.Random,
) -> SemanticTask:
    """Task where evidence contradicts H1 (current). H2 (stale/defer) wins."""
    subject = DOMAIN_SUBJECTS.get(domain, domain)
    h = _make_hyps(2, subject)

    contradict_passages = [p for p in passages
                          if any(r == "CONTRADICT" and o == "current" for o, r in p.gold_relations)]
    if not contradict_passages:
        contradict_passages = [p for p in passages
                              if any(r == "CONTRADICT" for o, r in p.gold_relations)]

    p1 = rng.choice(contradict_passages)
    ev = (_passage_to_evidence(p1, "E1", retrieved=True, verify_result="FALSIFIED"),)

    gold = _passage_gold_relations(p1, "E1", 2)

    et = EvidenceTask(
        task_id=task_id, split="i3_13",
        category="degraded_service",
        task_summary=f"Determine the current status of {subject}.",
        high_stakes=True, budget_profile="STANDARD",
        hypotheses=h, evidence_items=ev,
        retrieve_exposes=(), search_exposes=(),
        oracle_resolution_path=("VERIFY:E1", "DEFER"),
        expected_terminal=DecisionAction.DEFER, correct_hypothesis_id="H2",
    )
    return SemanticTask(et, gold, tier="", semantic_class="real_evidence")


def gen_noise_plus_signal_task(
    task_id: str,
    domain: str,
    passages: list[DocumentPassage],
    rng: random.Random,
) -> SemanticTask:
    """Task with noise passages + one signal passage. T2 should NOT fire."""
    subject = DOMAIN_SUBJECTS.get(domain, domain)
    h = _make_hyps(2, subject)

    neutral_passages = [p for p in passages
                        if all(r == "NEUTRAL" for _, r in p.gold_relations)]
    support_passages = [p for p in passages
                        if any(r == "SUPPORT" and o == "current" for o, r in p.gold_relations)]

    p_noise1 = rng.choice(neutral_passages)
    p_noise2 = rng.choice(neutral_passages)
    p_signal = rng.choice(support_passages)

    ev = (
        _passage_to_evidence(p_noise1, "E1", retrieved=True, verify_result="MISSING"),
        _passage_to_evidence(p_noise2, "E2", retrieved=True, verify_result="MISSING"),
        _passage_to_evidence(p_signal, "E3", retrieved=True, verify_result="SUFFICIENT"),
    )
    gold = (_passage_gold_relations(p_noise1, "E1", 2)
            + _passage_gold_relations(p_noise2, "E2", 2)
            + _passage_gold_relations(p_signal, "E3", 2))

    et = EvidenceTask(
        task_id=task_id, split="i3_13",
        category="noise_plus_signal",
        task_summary=f"Determine the current status of {subject}.",
        high_stakes=True, budget_profile="STANDARD",
        hypotheses=h, evidence_items=ev,
        retrieve_exposes=(), search_exposes=(),
        oracle_resolution_path=("VERIFY:E3", "ANSWER"),
        expected_terminal=DecisionAction.ANSWER, correct_hypothesis_id="H1",
    )
    return SemanticTask(et, gold, tier="", semantic_class="real_evidence")


def gen_conflict_with_noise_task(
    task_id: str,
    domain: str,
    passages: list[DocumentPassage],
    rng: random.Random,
) -> SemanticTask:
    """Task with bilateral conflict + hidden noise. T2 should fire."""
    subject = DOMAIN_SUBJECTS.get(domain, domain)
    h = _make_hyps(2, subject)

    current_support = [p for p in passages
                       if any(r == "SUPPORT" and o == "current" for o, r in p.gold_relations)]
    stale_support = [p for p in passages
                     if any(r == "SUPPORT" and o == "stale" for o, r in p.gold_relations)]
    neutral_passages = [p for p in passages
                        if all(r == "NEUTRAL" for _, r in p.gold_relations)]

    p1 = rng.choice(current_support)
    p2 = rng.choice(stale_support)
    p3 = rng.choice(neutral_passages)

    ev = (
        _passage_to_evidence(p1, "E1", retrieved=True, verify_result="SUFFICIENT"),
        _passage_to_evidence(p2, "E2", retrieved=True, verify_result="SUFFICIENT"),
        _passage_to_evidence(p3, "E3", retrieved=False, verify_result="MISSING"),
    )
    gold = (_passage_gold_relations(p1, "E1", 2)
            + _passage_gold_relations(p2, "E2", 2)
            + _passage_gold_relations(p3, "E3", 2))

    et = EvidenceTask(
        task_id=task_id, split="i3_13",
        category="conflict_with_noise",
        task_summary=f"Determine the current status of {subject}.",
        high_stakes=True, budget_profile="STANDARD",
        hypotheses=h, evidence_items=ev,
        retrieve_exposes=(), search_exposes=("E3",),
        oracle_resolution_path=("VERIFY:E1", "VERIFY:E2", "DEFER"),
        expected_terminal=DecisionAction.DEFER, correct_hypothesis_id="H2",
    )
    return SemanticTask(et, gold, tier="", semantic_class="real_evidence")


def gen_compositional_task(
    task_id: str,
    domain: str,
    passages: list[DocumentPassage],
    rng: random.Random,
) -> SemanticTask:
    """Task with compositional/complex passages. Tests real language understanding."""
    subject = DOMAIN_SUBJECTS.get(domain, domain)
    h = _make_hyps(2, subject)

    # Use complex passages (those with multiple sentences or compositional structure)
    complex_passages = [p for p in passages if len(p.text.split(".")) > 2
                        and any(r != "NEUTRAL" for _, r in p.gold_relations)]
    if not complex_passages:
        complex_passages = [p for p in passages if any(r != "NEUTRAL" for _, r in p.gold_relations)]

    p1 = rng.choice(complex_passages)
    # Determine if this is a support or contradict passage
    is_support = any(r == "SUPPORT" and o == "current" for o, r in p1.gold_relations)
    vr = "SUFFICIENT" if is_support else "FALSIFIED"
    expected = DecisionAction.ANSWER if is_support else DecisionAction.DEFER
    correct = "H1" if is_support else "H2"

    ev = (_passage_to_evidence(p1, "E1", retrieved=True, verify_result=vr),)
    gold = _passage_gold_relations(p1, "E1", 2)

    et = EvidenceTask(
        task_id=task_id, split="i3_13",
        category="compositional",
        task_summary=f"Determine the current status of {subject}.",
        high_stakes=True, budget_profile="STANDARD",
        hypotheses=h, evidence_items=ev,
        retrieve_exposes=(), search_exposes=(),
        oracle_resolution_path=("VERIFY:E1", expected.value),
        expected_terminal=expected, correct_hypothesis_id=correct,
    )
    return SemanticTask(et, gold, tier="", semantic_class="real_evidence")


# ---------------------------------------------------------------------------
# Corpus generation
# ---------------------------------------------------------------------------

GENERATORS = {
    "clear_answer": gen_clear_answer_task,
    "bilateral_conflict": gen_conflict_task,
    "degraded_service": gen_degraded_task,
    "noise_plus_signal": gen_noise_plus_signal_task,
    "conflict_with_noise": gen_conflict_with_noise_task,
    "compositional": gen_compositional_task,
}


def generate_i3_13_corpus(
    n_per_category: int = 25,
    seed: int = 42,
) -> list[SemanticTask]:
    """Generate the I3.13 retrieved-evidence corpus.

    Args:
        n_per_category: number of tasks per category
        seed: random seed for reproducibility

    Returns:
        List of SemanticTask with gold relations and real passage evidence
    """
    rng = random.Random(seed)
    passages = get_corpus()
    tasks: list[SemanticTask] = []
    task_counter = 0

    domains = list(DOMAIN_SUBJECTS.keys())

    for cat_name, gen_func in GENERATORS.items():
        for i in range(n_per_category):
            task_id = f"i3_13_{task_counter:04d}"
            domain = rng.choice(domains)
            task = gen_func(task_id, domain, passages, rng)
            tasks.append(task)
            task_counter += 1

    return tasks


if __name__ == "__main__":
    tasks = generate_i3_13_corpus(n_per_category=25, seed=42)
    print(f"Generated {len(tasks)} I3.13 tasks")
    by_cat = {}
    for t in tasks:
        by_cat.setdefault(t.category, 0)
        by_cat[t.category] += 1
    print(f"By category: {by_cat}")
