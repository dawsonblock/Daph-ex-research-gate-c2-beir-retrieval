"""Evidence-bearing task generator for I3.7.

Generates tasks with:
  - 2 competing hypotheses (H1: ANSWER, H2: DEFER)
  - 3-5 evidence items with proposition-level claims
  - Some evidence initially visible, some hidden
  - Evidence that supports or contradicts each hypothesis
  - A discriminating evidence item that must be found and verified
  - An oracle resolution path

Task patterns:
  Pattern A (answer task): H1 is correct, discriminating evidence supports H1
  Pattern B (defer task): H2 is correct, no sufficient evidence for H1
  Pattern C (stale evidence task): initial evidence is stale, must search for current
  Pattern D (conflict task): conflicting evidence, must verify to resolve
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Iterator

from hrm_adaptive_memory.cognitive_control.core import DecisionAction
from hrm_adaptive_memory.cognitive_control.state import (
    TemporalStatus, VerificationState,
)

from .schema import (
    EvidenceItem, EvidenceTask, EvidenceHypothesis,
)


# Deterministic RNG for reproducible task generation
def _seeded_rng(task_id: str) -> random.Random:
    h = hashlib.sha256(task_id.encode()).hexdigest()
    return random.Random(int(h[:16], 16))


# Task templates with realistic propositions
TASK_TEMPLATES = [
    {
        "subject": "whether the API documentation specifies a rate limit of 100 requests per minute",
        "h1_proposition": "the API documentation specifies a rate limit of 100 requests per minute, so the answer should be ANSWER",
        "h1_answer": DecisionAction.ANSWER,
        "h1_payload": "rate limit is 100 requests per minute",
        "h2_proposition": "the API documentation does not specify a rate limit, so the answer should be DEFER",
        "h2_answer": DecisionAction.DEFER,
        "h2_payload": "insufficient evidence to determine rate limit",
    },
    {
        "subject": "whether the database supports transactional rollback",
        "h1_proposition": "the database documentation confirms transactional rollback support, so the answer should be ANSWER",
        "h1_answer": DecisionAction.ANSWER,
        "h1_payload": "database supports transactional rollback",
        "h2_proposition": "the database documentation does not confirm transactional rollback, so the answer should be DEFER",
        "h2_answer": DecisionAction.DEFER,
        "h2_payload": "insufficient evidence to confirm rollback support",
    },
    {
        "subject": "whether the service endpoint requires authentication",
        "h1_proposition": "the service documentation confirms authentication is required, so the answer should be ANSWER",
        "h1_answer": DecisionAction.ANSWER,
        "h1_payload": "authentication is required",
        "h2_proposition": "the service documentation does not confirm authentication requirements, so the answer should be DEFER",
        "h2_answer": DecisionAction.DEFER,
        "h2_payload": "insufficient evidence to determine authentication requirements",
    },
    {
        "subject": "whether the configuration file supports environment variable substitution",
        "h1_proposition": "the configuration documentation confirms environment variable substitution, so the answer should be ANSWER",
        "h1_answer": DecisionAction.ANSWER,
        "h1_payload": "environment variable substitution is supported",
        "h2_proposition": "the configuration documentation does not confirm variable substitution, so the answer should be DEFER",
        "h2_answer": DecisionAction.DEFER,
        "h2_payload": "insufficient evidence to confirm variable substitution",
    },
    {
        "subject": "whether the caching layer supports TTL-based expiration",
        "h1_proposition": "the caching documentation confirms TTL-based expiration, so the answer should be ANSWER",
        "h1_answer": DecisionAction.ANSWER,
        "h1_payload": "TTL-based expiration is supported",
        "h2_proposition": "the caching documentation does not confirm TTL expiration, so the answer should be DEFER",
        "h2_answer": DecisionAction.DEFER,
        "h2_payload": "insufficient evidence to confirm TTL expiration",
    },
]


class EvidenceTaskGenerator:
    """Generates evidence-bearing tasks with deterministic variation."""

    def __init__(self, n_tasks: int = 50, split: str = "structure_dev_v2") -> None:
        self.n_tasks = n_tasks
        self.split = split

    def generate(self) -> tuple[EvidenceTask, ...]:
        """Generate n_tasks evidence-bearing tasks."""
        tasks: list[EvidenceTask] = []
        for i in range(self.n_tasks):
            task_id = f"i3_7_evidence_{self.split}_{i:04d}"
            rng = _seeded_rng(task_id)
            template = TASK_TEMPLATES[i % len(TASK_TEMPLATES)]
            pattern = rng.choice(["answer", "answer", "defer", "stale", "conflict"])
            task = self._generate_one(task_id, template, pattern, rng)
            tasks.append(task)
        return tuple(tasks)

    def _generate_one(
        self,
        task_id: str,
        template: dict,
        pattern: str,
        rng: random.Random,
    ) -> EvidenceTask:
        """Generate one task with the given pattern."""
        subject = template["subject"]

        h1 = EvidenceHypothesis(
            hypothesis_id="H1",
            proposition=template["h1_proposition"],
            answer_action=template["h1_answer"],
            answer_payload=template["h1_payload"],
        )
        h2 = EvidenceHypothesis(
            hypothesis_id="H2",
            proposition=template["h2_proposition"],
            answer_action=template["h2_answer"],
            answer_payload=template["h2_payload"],
        )

        if pattern == "answer":
            return self._gen_answer_task(task_id, subject, h1, h2, template, rng)
        elif pattern == "defer":
            return self._gen_defer_task(task_id, subject, h1, h2, template, rng)
        elif pattern == "stale":
            return self._gen_stale_task(task_id, subject, h1, h2, template, rng)
        elif pattern == "conflict":
            return self._gen_conflict_task(task_id, subject, h1, h2, template, rng)
        else:
            return self._gen_answer_task(task_id, subject, h1, h2, template, rng)

    def _gen_answer_task(
        self, task_id: str, subject: str,
        h1: EvidenceHypothesis, h2: EvidenceHypothesis,
        template: dict, rng: random.Random,
    ) -> EvidenceTask:
        """Pattern A: H1 (ANSWER) is correct. Discriminating evidence supports H1."""
        # E1: initial evidence, weak support for H1, unverified
        # E2: initial evidence, mentions H2 but unverified
        # E3: hidden evidence (RETRIEVE), strongly supports H1, verify -> SUFFICIENT
        # E4: hidden evidence (SEARCH_MORE), confirms H1 is current
        evidence = (
            EvidenceItem(
                evidence_id="E1",
                proposition=f"An initial source mentions that {subject}, but the claim is unverified.",
                source_class="initial",
                supports=("H1",),
                contradicts=(),
                verification_state=VerificationState.UNVERIFIED,
                temporal_status=TemporalStatus.CURRENT,
                retrieved=True,
                verify_result="SUFFICIENT",
            ),
            EvidenceItem(
                evidence_id="E2",
                proposition=f"Another source is silent on {subject}.",
                source_class="initial",
                supports=("H2",),
                contradicts=("H1",),
                verification_state=VerificationState.UNVERIFIED,
                temporal_status=TemporalStatus.CURRENT,
                retrieved=True,
                verify_result="FALSIFIED",  # verifying E2 shows it doesn't actually support H2
            ),
            EvidenceItem(
                evidence_id="E3",
                proposition=f"The primary documentation explicitly confirms that {subject}.",
                source_class="primary",
                supports=("H1",),
                contradicts=("H2",),
                verification_state=VerificationState.UNVERIFIED,
                temporal_status=TemporalStatus.CURRENT,
                retrieved=False,
                verify_result="SUFFICIENT",
            ),
            EvidenceItem(
                evidence_id="E4",
                proposition=f"A recent update confirms the current status of {subject}.",
                source_class="search",
                supports=("H1",),
                contradicts=(),
                verification_state=VerificationState.UNVERIFIED,
                temporal_status=TemporalStatus.CURRENT,
                retrieved=False,
                verify_result="SUFFICIENT",
            ),
        )
        return EvidenceTask(
            task_id=task_id, split=self.split, category="evidence_answer",
            task_summary=f"Determine {subject}.",
            high_stakes=rng.random() > 0.5,
            budget_profile="STANDARD",
            hypotheses=(h1, h2),
            evidence_items=evidence,
            retrieve_exposes=("E3",),
            search_exposes=("E4",),
            oracle_resolution_path=("RETRIEVE:E3", "VERIFY:E3", "ANSWER"),
            expected_terminal=DecisionAction.ANSWER,
            correct_hypothesis_id="H1",
        )

    def _gen_defer_task(
        self, task_id: str, subject: str,
        h1: EvidenceHypothesis, h2: EvidenceHypothesis,
        template: dict, rng: random.Random,
    ) -> EvidenceTask:
        """Pattern B: H2 (DEFER) is correct. No sufficient evidence for H1."""
        evidence = (
            EvidenceItem(
                evidence_id="E1",
                proposition=f"An initial source vaguely mentions {subject}, but details are missing.",
                source_class="initial",
                supports=("H1",),
                contradicts=(),
                verification_state=VerificationState.UNVERIFIED,
                temporal_status=TemporalStatus.CURRENT,
                retrieved=True,
                verify_result="MISSING",  # verifying shows it's insufficient
            ),
            EvidenceItem(
                evidence_id="E2",
                proposition=f"Another source does not address {subject}.",
                source_class="initial",
                supports=("H2",),
                contradicts=("H1",),
                verification_state=VerificationState.UNVERIFIED,
                temporal_status=TemporalStatus.CURRENT,
                retrieved=True,
                verify_result="SUFFICIENT",
            ),
            EvidenceItem(
                evidence_id="E3",
                proposition=f"A search for additional sources finds no confirmation of {subject}.",
                source_class="search",
                supports=("H2",),
                contradicts=("H1",),
                verification_state=VerificationState.UNVERIFIED,
                temporal_status=TemporalStatus.CURRENT,
                retrieved=False,
                verify_result="SUFFICIENT",
            ),
        )
        return EvidenceTask(
            task_id=task_id, split=self.split, category="evidence_defer",
            task_summary=f"Determine {subject}.",
            high_stakes=rng.random() > 0.5,
            budget_profile="STANDARD",
            hypotheses=(h1, h2),
            evidence_items=evidence,
            retrieve_exposes=(),  # no retrievable evidence
            search_exposes=("E3",),
            oracle_resolution_path=("SEARCH_MORE:E3", "VERIFY:E3", "DEFER"),
            expected_terminal=DecisionAction.DEFER,
            correct_hypothesis_id="H2",
        )

    def _gen_stale_task(
        self, task_id: str, subject: str,
        h1: EvidenceHypothesis, h2: EvidenceHypothesis,
        template: dict, rng: random.Random,
    ) -> EvidenceTask:
        """Pattern C: initial evidence is stale, must search for current."""
        evidence = (
            EvidenceItem(
                evidence_id="E1",
                proposition=f"An older source claims {subject}, but the source is outdated.",
                source_class="initial",
                supports=("H1",),
                contradicts=(),
                verification_state=VerificationState.UNVERIFIED,
                temporal_status=TemporalStatus.STALE,
                retrieved=True,
                verify_result="STALE",  # verifying shows it's stale
            ),
            EvidenceItem(
                evidence_id="E2",
                proposition=f"A current source confirms {subject}.",
                source_class="search",
                supports=("H1",),
                contradicts=("H2",),
                verification_state=VerificationState.UNVERIFIED,
                temporal_status=TemporalStatus.CURRENT,
                retrieved=False,
                verify_result="SUFFICIENT",
            ),
            EvidenceItem(
                evidence_id="E3",
                proposition=f"Another current source also confirms {subject}.",
                source_class="primary",
                supports=("H1",),
                contradicts=(),
                verification_state=VerificationState.UNVERIFIED,
                temporal_status=TemporalStatus.CURRENT,
                retrieved=False,
                verify_result="SUFFICIENT",
            ),
        )
        return EvidenceTask(
            task_id=task_id, split=self.split, category="evidence_stale",
            task_summary=f"Determine {subject}.",
            high_stakes=rng.random() > 0.5,
            budget_profile="STANDARD",
            hypotheses=(h1, h2),
            evidence_items=evidence,
            retrieve_exposes=("E3",),
            search_exposes=("E2",),
            oracle_resolution_path=("SEARCH_MORE:E2", "VERIFY:E2", "ANSWER"),
            expected_terminal=DecisionAction.ANSWER,
            correct_hypothesis_id="H1",
        )

    def _gen_conflict_task(
        self, task_id: str, subject: str,
        h1: EvidenceHypothesis, h2: EvidenceHypothesis,
        template: dict, rng: random.Random,
    ) -> EvidenceTask:
        """Pattern D: conflicting evidence, must verify to resolve."""
        evidence = (
            EvidenceItem(
                evidence_id="E1",
                proposition=f"Source A claims {subject}.",
                source_class="initial",
                supports=("H1",),
                contradicts=(),
                verification_state=VerificationState.UNVERIFIED,
                temporal_status=TemporalStatus.CURRENT,
                retrieved=True,
                verify_result="SUFFICIENT",
            ),
            EvidenceItem(
                evidence_id="E2",
                proposition=f"Source B contradicts, claiming not-{subject}.",
                source_class="initial",
                supports=("H2",),
                contradicts=("H1",),
                verification_state=VerificationState.UNVERIFIED,
                temporal_status=TemporalStatus.CURRENT,
                retrieved=True,
                verify_result="FALSIFIED",  # E2 is actually wrong
            ),
            EvidenceItem(
                evidence_id="E3",
                proposition=f"A definitive source confirms {subject}.",
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
            task_id=task_id, split=self.split, category="evidence_conflict",
            task_summary=f"Determine {subject}.",
            high_stakes=True,  # conflicts are always high-stakes
            budget_profile="STANDARD",
            hypotheses=(h1, h2),
            evidence_items=evidence,
            retrieve_exposes=(),
            search_exposes=("E3",),
            oracle_resolution_path=("SEARCH_MORE:E3", "VERIFY:E3", "ANSWER"),
            expected_terminal=DecisionAction.ANSWER,
            correct_hypothesis_id="H1",
        )


def generate_evidence_tasks(
    n_tasks: int = 50,
    split: str = "structure_dev_v2",
) -> tuple[EvidenceTask, ...]:
    """Generate a frozen set of evidence-bearing tasks."""
    gen = EvidenceTaskGenerator(n_tasks=n_tasks, split=split)
    return gen.generate()
