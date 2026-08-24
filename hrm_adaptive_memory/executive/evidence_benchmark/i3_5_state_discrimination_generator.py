"""I3.5 State-Discrimination Benchmark Generator.

Creates tasks where superficially similar epistemic states require different
optimal first actions. This destroys the "one_live => DEFER" shortcut that
PS05 exploited in I3.4e.

Five one_live subtypes (24 each = 120 tasks):

  OL-A — one live, answer ready
    1 live hypothesis, sufficient verified support, no contradiction.
    Correct first action: ANSWER

  OL-D — one live, must defer
    1 live hypothesis, insufficient evidence, no viable acquisition route.
    Correct first action: DEFER

  OL-R — one live, retrieve required
    1 live hypothesis, key supporting evidence absent (hidden, retrievable).
    Correct first action: RETRIEVE

  OL-V — one live, verification required
    1 live hypothesis, supporting evidence exists but unverified.
    Correct first action: VERIFY

  OL-S — one live, search required
    1 live hypothesis, local retrieval exhausted, external search viable.
    Correct first action: SEARCH_MORE

A trivial "always DEFER" policy scores ~20% (only OL-D).
A trivial "always ANSWER" policy scores ~20% (only OL-A).
This forces the controller to discriminate between states.

Each task also has a `correct_first_action` field for evaluation.
The `expected_terminal` remains ANSWER or DEFER for executor compatibility.
"""
from __future__ import annotations

import hashlib
import random
from typing import Any

from hrm_adaptive_memory.cognitive_control.core import DecisionAction
from hrm_adaptive_memory.cognitive_control.state import (
    TemporalStatus, VerificationState,
)

from hrm_adaptive_memory.executive.evidence_benchmark.schema import (
    EvidenceItem, EvidenceTask, EvidenceHypothesis,
)


# ---------------------------------------------------------------------------
# Domain templates (reused from i3_15c for consistency)
# ---------------------------------------------------------------------------

DOMAINS = [
    "api_gateway", "database", "cdn", "kubernetes", "security",
    "deployment", "monitoring", "cache", "message_queue", "load_balancer",
]

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


def _seeded_rng(task_id: str) -> random.Random:
    h = hashlib.sha256(task_id.encode()).hexdigest()
    return random.Random(int(h[:16], 16))


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


# ---------------------------------------------------------------------------
# One-live subtype generators
# ---------------------------------------------------------------------------

def _gen_ol_answer(task_id: str, domain: str, rng: random.Random) -> EvidenceTask:
    """OL-A: one live, answer ready. Correct first action: ANSWER.

    State: H1 is viable with SUFFICIENT verified support.
    H2 is eliminated (SUFFICIENT contradiction).
    The controller should ANSWER immediately.
    """
    subject = DOMAIN_SUBJECTS[domain]
    h1, h2 = _make_hyps(subject)

    evidence = (
        # E1: verified sufficient support for H1
        EvidenceItem(
            evidence_id="E1",
            proposition=f"An authoritative source confirms {subject} is operational with all checks passing.",
            source_class="initial",
            supports=("H1",),
            contradicts=("H2",),
            verification_state=VerificationState.SUFFICIENT,
            temporal_status=TemporalStatus.CURRENT,
            retrieved=True,
            verify_result="SUFFICIENT",
        ),
        # E2: verified sufficient contradiction for H2
        EvidenceItem(
            evidence_id="E2",
            proposition=f"An authoritative probe confirms {subject} is responding normally, contradicting the unavailability hypothesis.",
            source_class="initial",
            supports=("H1",),
            contradicts=("H2",),
            verification_state=VerificationState.SUFFICIENT,
            temporal_status=TemporalStatus.CURRENT,
            retrieved=True,
            verify_result="SUFFICIENT",
        ),
    )

    return EvidenceTask(
        task_id=task_id,
        split="i3_5_state_discrimination",
        category="ol_answer",
        task_summary=f"Determine {subject}.",
        high_stakes=rng.random() > 0.5,
        budget_profile="STANDARD",
        hypotheses=(h1, h2),
        evidence_items=evidence,
        retrieve_exposes=(),
        search_exposes=(),
        oracle_resolution_path=("ANSWER",),
        expected_terminal=DecisionAction.ANSWER,
        correct_hypothesis_id="H1",
    )


def _gen_ol_defer(task_id: str, domain: str, rng: random.Random) -> EvidenceTask:
    """OL-D: one live, must defer. Correct first action: DEFER.

    State: H1 has only unverified/missing evidence.
    H2 is eliminated (contradicted by E2, but E2 does NOT support H1).
    No retrieval or search available (budgets exhausted or no hidden evidence).
    The controller should DEFER because H1 cannot be confirmed.
    """
    subject = DOMAIN_SUBJECTS[domain]
    h1, h2 = _make_hyps(subject)

    evidence = (
        # E1: unverified, missing support for H1
        EvidenceItem(
            evidence_id="E1",
            proposition=f"An initial source vaguely mentions {subject}, but details are missing.",
            source_class="initial",
            supports=("H1",),
            contradicts=(),
            verification_state=VerificationState.UNVERIFIED,
            temporal_status=TemporalStatus.CURRENT,
            retrieved=True,
            verify_result="MISSING",
        ),
        # E2: verified contradiction for H2 (eliminates H2, does NOT support H1)
        EvidenceItem(
            evidence_id="E2",
            proposition=f"An authoritative probe confirms {subject} is not in an error state, contradicting the unavailability hypothesis.",
            source_class="initial",
            supports=(),  # does NOT support H1
            contradicts=("H2",),
            verification_state=VerificationState.SUFFICIENT,
            temporal_status=TemporalStatus.CURRENT,
            retrieved=True,
            verify_result="SUFFICIENT",
        ),
    )

    return EvidenceTask(
        task_id=task_id,
        split="i3_5_state_discrimination",
        category="ol_defer",
        task_summary=f"Determine {subject}.",
        high_stakes=rng.random() > 0.5,
        budget_profile="STANDARD_NO_RETRIEVE_NO_SEARCH",  # no retrieval/search budget
        hypotheses=(h1, h2),
        evidence_items=evidence,
        retrieve_exposes=(),  # nothing to retrieve
        search_exposes=(),    # nothing to search
        oracle_resolution_path=("DEFER",),
        expected_terminal=DecisionAction.DEFER,
        correct_hypothesis_id="H2",
    )


def _gen_ol_retrieve(task_id: str, domain: str, rng: random.Random) -> EvidenceTask:
    """OL-R: one live, retrieve required. Correct first action: RETRIEVE.

    State: H1 is live but lacks sufficient verified support.
    H2 is eliminated.
    Key supporting evidence is hidden but retrievable.
    The controller should RETRIEVE first, then VERIFY, then ANSWER.
    """
    subject = DOMAIN_SUBJECTS[domain]
    h1, h2 = _make_hyps(subject)

    evidence = (
        # E1: initial evidence, weak/unverified support for H1
        EvidenceItem(
            evidence_id="E1",
            proposition=f"An initial source mentions {subject} but the claim is unverified.",
            source_class="initial",
            supports=("H1",),
            contradicts=(),
            verification_state=VerificationState.UNVERIFIED,
            temporal_status=TemporalStatus.CURRENT,
            retrieved=True,
            verify_result="MISSING",  # verifying E1 shows it's insufficient
        ),
        # E2: verified contradiction for H2 (eliminates H2, does NOT support H1)
        EvidenceItem(
            evidence_id="E2",
            proposition=f"An authoritative probe confirms {subject} is not in an error state, contradicting the unavailability hypothesis.",
            source_class="initial",
            supports=(),  # does NOT support H1
            contradicts=("H2",),
            verification_state=VerificationState.SUFFICIENT,
            temporal_status=TemporalStatus.CURRENT,
            retrieved=True,
            verify_result="SUFFICIENT",
        ),
        # E3: hidden, retrievable — the key supporting evidence
        EvidenceItem(
            evidence_id="E3",
            proposition=f"The primary documentation explicitly confirms {subject} is operational with verified status.",
            source_class="primary",
            supports=("H1",),
            contradicts=("H2",),
            verification_state=VerificationState.UNVERIFIED,
            temporal_status=TemporalStatus.CURRENT,
            retrieved=False,
            verify_result="SUFFICIENT",
        ),
    )

    return EvidenceTask(
        task_id=task_id,
        split="i3_5_state_discrimination",
        category="ol_retrieve",
        task_summary=f"Determine {subject}.",
        high_stakes=rng.random() > 0.5,
        budget_profile="STANDARD",
        hypotheses=(h1, h2),
        evidence_items=evidence,
        retrieve_exposes=("E3",),
        search_exposes=(),
        oracle_resolution_path=("RETRIEVE:E3", "VERIFY:E3", "ANSWER"),
        expected_terminal=DecisionAction.ANSWER,
        correct_hypothesis_id="H1",
    )


def _gen_ol_verify(task_id: str, domain: str, rng: random.Random) -> EvidenceTask:
    """OL-V: one live, verification required. Correct first action: VERIFY.

    State: H1 is live with unverified supporting evidence.
    H2 is eliminated.
    The supporting evidence is already visible (retrieved) but unverified.
    The controller should VERIFY first, then ANSWER.
    No retrieval needed — evidence is already visible.
    """
    subject = DOMAIN_SUBJECTS[domain]
    h1, h2 = _make_hyps(subject)

    evidence = (
        # E1: visible but UNVERIFIED support for H1 — needs VERIFY
        EvidenceItem(
            evidence_id="E1",
            proposition=f"A source claims {subject} is operational, but this has not been verified.",
            source_class="initial",
            supports=("H1",),
            contradicts=(),
            verification_state=VerificationState.UNVERIFIED,
            temporal_status=TemporalStatus.CURRENT,
            retrieved=True,
            verify_result="SUFFICIENT",  # verifying will confirm it's sufficient
        ),
        # E2: verified contradiction for H2 (eliminates H2, does NOT support H1)
        EvidenceItem(
            evidence_id="E2",
            proposition=f"An authoritative probe confirms {subject} is not in an error state, contradicting the unavailability hypothesis.",
            source_class="initial",
            supports=(),  # does NOT support H1
            contradicts=("H2",),
            verification_state=VerificationState.SUFFICIENT,
            temporal_status=TemporalStatus.CURRENT,
            retrieved=True,
            verify_result="SUFFICIENT",
        ),
    )

    return EvidenceTask(
        task_id=task_id,
        split="i3_5_state_discrimination",
        category="ol_verify",
        task_summary=f"Determine {subject}.",
        high_stakes=rng.random() > 0.5,
        budget_profile="STANDARD_NO_RETRIEVE",  # no retrieval needed, but verify available
        hypotheses=(h1, h2),
        evidence_items=evidence,
        retrieve_exposes=(),  # nothing to retrieve — evidence already visible
        search_exposes=(),
        oracle_resolution_path=("VERIFY:E1", "ANSWER"),
        expected_terminal=DecisionAction.ANSWER,
        correct_hypothesis_id="H1",
    )


def _gen_ol_search(task_id: str, domain: str, rng: random.Random) -> EvidenceTask:
    """OL-S: one live, search required. Correct first action: SEARCH_MORE.

    State: H1 is live but local evidence is insufficient/stale.
    H2 is eliminated.
    Local retrieval is exhausted (no retrievable evidence).
    External search has the key evidence.
    The controller should SEARCH_MORE first, then VERIFY, then ANSWER.
    """
    subject = DOMAIN_SUBJECTS[domain]
    h1, h2 = _make_hyps(subject)

    evidence = (
        # E1: stale evidence — verifying shows it's stale
        EvidenceItem(
            evidence_id="E1",
            proposition=f"An older source claims {subject} is operational, but the source is outdated.",
            source_class="initial",
            supports=("H1",),
            contradicts=(),
            verification_state=VerificationState.UNVERIFIED,
            temporal_status=TemporalStatus.STALE,
            retrieved=True,
            verify_result="STALE",
        ),
        # E2: verified contradiction for H2 (eliminates H2, does NOT support H1)
        EvidenceItem(
            evidence_id="E2",
            proposition=f"An authoritative probe confirms {subject} is not in an error state, contradicting the unavailability hypothesis.",
            source_class="initial",
            supports=(),  # does NOT support H1
            contradicts=("H2",),
            verification_state=VerificationState.SUFFICIENT,
            temporal_status=TemporalStatus.CURRENT,
            retrieved=True,
            verify_result="SUFFICIENT",
        ),
        # E3: hidden, only accessible via SEARCH_MORE — the current evidence
        EvidenceItem(
            evidence_id="E3",
            proposition=f"A recent update confirms {subject} is currently operational with all systems green.",
            source_class="search",
            supports=("H1",),
            contradicts=("H2",),
            verification_state=VerificationState.UNVERIFIED,
            temporal_status=TemporalStatus.CURRENT,
            retrieved=False,
            verify_result="SUFFICIENT",
        ),
    )

    return EvidenceTask(
        task_id=task_id,
        split="i3_5_state_discrimination",
        category="ol_search",
        task_summary=f"Determine {subject}.",
        high_stakes=rng.random() > 0.5,
        budget_profile="STANDARD_NO_RETRIEVE",  # retrieval exhausted, but search available
        hypotheses=(h1, h2),
        evidence_items=evidence,
        retrieve_exposes=(),  # nothing retrievable — local retrieval exhausted
        search_exposes=("E3",),
        oracle_resolution_path=("SEARCH_MORE:E3", "VERIFY:E3", "ANSWER"),
        expected_terminal=DecisionAction.ANSWER,
        correct_hypothesis_id="H1",
    )


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

SUBTYPE_GENERATORS = {
    "ol_answer": _gen_ol_answer,
    "ol_defer": _gen_ol_defer,
    "ol_retrieve": _gen_ol_retrieve,
    "ol_verify": _gen_ol_verify,
    "ol_search": _gen_ol_search,
}

CORRECT_FIRST_ACTION = {
    "ol_answer": DecisionAction.ANSWER,
    "ol_defer": DecisionAction.DEFER,
    "ol_retrieve": DecisionAction.RETRIEVE,
    "ol_verify": DecisionAction.VERIFY,
    "ol_search": DecisionAction.SEARCH_MORE,
}


def generate_i3_5_state_discrimination_benchmark(
    n_per_subtype: int = 24,
    seed: int = 9137,
) -> tuple[EvidenceTask, ...]:
    """Generate the I3.5 state-discrimination benchmark.

    Args:
        n_per_subtype: number of tasks per one_live subtype (default 24)
        seed: random seed for domain assignment

    Returns:
        Tuple of EvidenceTask objects with correct_first_action metadata.
        Total: 5 * n_per_subtype tasks.
    """
    rng = random.Random(seed)
    tasks: list[EvidenceTask] = []

    subtypes = list(SUBTYPE_GENERATORS.keys())

    for subtype in subtypes:
        gen_func = SUBTYPE_GENERATORS[subtype]
        for i in range(n_per_subtype):
            task_id = f"i3_5_{subtype}_{i:04d}"
            domain = DOMAINS[rng.randrange(len(DOMAINS))]
            task_rng = _seeded_rng(task_id)
            task = gen_func(task_id, domain, task_rng)
            tasks.append(task)

    return tuple(tasks)


def benchmark_summary(tasks: tuple[EvidenceTask, ...]) -> dict:
    """Compute benchmark summary statistics."""
    from collections import Counter

    categories = Counter(t.category for t in tasks)
    terminals = Counter(t.expected_terminal.value for t in tasks)

    # Compute correct first action distribution
    first_actions = Counter(
        CORRECT_FIRST_ACTION.get(t.category, "UNKNOWN").value
        if hasattr(CORRECT_FIRST_ACTION.get(t.category, "UNKNOWN"), "value")
        else str(CORRECT_FIRST_ACTION.get(t.category, "UNKNOWN"))
        for t in tasks
    )

    subtype_keys = list(SUBTYPE_GENERATORS.keys())
    return {
        "total_tasks": len(tasks),
        "categories": dict(categories),
        "expected_terminals": dict(terminals),
        "correct_first_actions": dict(first_actions),
        "balanced": all(
            categories.get(st, 0) == categories.get(subtype_keys[0], 0)
            for st in subtype_keys
        ) if categories else False,
    }


def benchmark_sha256(tasks: tuple[EvidenceTask, ...]) -> str:
    """Compute deterministic SHA256 over the benchmark."""
    task_ids = sorted(t.task_id for t in tasks)
    return hashlib.sha256(
        "|".join(task_ids).encode()
    ).hexdigest()
