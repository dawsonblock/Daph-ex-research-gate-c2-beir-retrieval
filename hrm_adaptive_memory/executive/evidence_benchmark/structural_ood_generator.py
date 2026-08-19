"""Structural out-of-distribution task generator for I3.9.

Generates tasks that materially vary surface structure while testing the
same underlying stopping invariant:

    one uniquely viable supported hypothesis => stop further cognition

Structural variations vs. the I3.7 generator:
  - 3 hypotheses (not always 2)
  - 1, 3, or 4 VERIFY operations before READY_TO_ANSWER (not always 2)
  - Different visible/hidden splits (1+3, 3+1, 2+2, 4+0)
  - Irrelevant/noise evidence items
  - New subject-matter templates
  - Different resource budgets
  - Adversarial subgroups:
      EARLY_FALSE_READY: visible evidence makes wrong hypothesis
        look viable; hidden evidence needed to find correct answer
      LATE_RESOLUTION: requires 4+ operations before resolution
      MULTI_HYPOTHESIS_AMBIGUITY: 3 hypotheses with complex topology
      STALE_SUPPORT: visible support is stale, must search
      CONFLICT_UNRESOLVED: genuine unresolvable conflict, expected DEFER
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


def _seeded_rng(task_id: str) -> random.Random:
    h = hashlib.sha256(task_id.encode()).hexdigest()
    return random.Random(int(h[:16], 16))


# New templates with different subject matter
STRUCTURAL_TEMPLATES = [
    {
        "subject": "whether the encryption protocol uses AES-256",
        "h1_proposition": "the documentation confirms AES-256 encryption, so the answer should be ANSWER",
        "h1_answer": DecisionAction.ANSWER,
        "h1_payload": "uses AES-256 encryption",
        "h2_proposition": "the documentation does not confirm AES-256, so the answer should be DEFER",
        "h2_answer": DecisionAction.DEFER,
        "h2_payload": "insufficient evidence to confirm encryption standard",
    },
    {
        "subject": "whether the deployment supports blue-green releases",
        "h1_proposition": "the deployment documentation confirms blue-green release support, so the answer should be ANSWER",
        "h1_answer": DecisionAction.ANSWER,
        "h1_payload": "supports blue-green releases",
        "h2_proposition": "the deployment documentation does not confirm blue-green releases, so the answer should be DEFER",
        "h2_answer": DecisionAction.DEFER,
        "h2_payload": "insufficient evidence to confirm release strategy",
    },
    {
        "subject": "whether the message queue guarantees exactly-once delivery",
        "h1_proposition": "the queue documentation confirms exactly-once delivery, so the answer should be ANSWER",
        "h1_answer": DecisionAction.ANSWER,
        "h1_payload": "guarantees exactly-once delivery",
        "h2_proposition": "the queue documentation does not confirm exactly-once delivery, so the answer should be DEFER",
        "h2_answer": DecisionAction.DEFER,
        "h2_payload": "insufficient evidence to confirm delivery semantics",
    },
    {
        "subject": "whether the storage system supports ACID transactions",
        "h1_proposition": "the storage documentation confirms ACID transaction support, so the answer should be ANSWER",
        "h1_answer": DecisionAction.ANSWER,
        "h1_payload": "supports ACID transactions",
        "h2_proposition": "the storage documentation does not confirm ACID transactions, so the answer should be DEFER",
        "h2_answer": DecisionAction.DEFER,
        "h2_payload": "insufficient evidence to confirm transaction support",
    },
    {
        "subject": "whether the API gateway enforces request validation",
        "h1_proposition": "the gateway documentation confirms request validation, so the answer should be ANSWER",
        "h1_answer": DecisionAction.ANSWER,
        "h1_payload": "enforces request validation",
        "h2_proposition": "the gateway documentation does not confirm request validation, so the answer should be DEFER",
        "h2_answer": DecisionAction.DEFER,
        "h2_payload": "insufficient evidence to confirm validation",
    },
    {
        "subject": "whether the monitoring system supports distributed tracing",
        "h1_proposition": "the monitoring documentation confirms distributed tracing support, so the answer should be ANSWER",
        "h1_answer": DecisionAction.ANSWER,
        "h1_payload": "supports distributed tracing",
        "h2_proposition": "the monitoring documentation does not confirm distributed tracing, so the answer should be DEFER",
        "h2_answer": DecisionAction.DEFER,
        "h2_payload": "insufficient evidence to confirm tracing support",
    },
    {
        "subject": "whether the database supports point-in-time recovery",
        "h1_proposition": "the database documentation confirms point-in-time recovery, so the answer should be ANSWER",
        "h1_answer": DecisionAction.ANSWER,
        "h1_payload": "supports point-in-time recovery",
        "h2_proposition": "the database documentation does not confirm point-in-time recovery, so the answer should be DEFER",
        "h2_answer": DecisionAction.DEFER,
        "h2_payload": "insufficient evidence to confirm recovery capability",
    },
    {
        "subject": "whether the load balancer supports weighted routing",
        "h1_proposition": "the load balancer documentation confirms weighted routing, so the answer should be ANSWER",
        "h1_answer": DecisionAction.ANSWER,
        "h1_payload": "supports weighted routing",
        "h2_proposition": "the load balancer documentation does not confirm weighted routing, so the answer should be DEFER",
        "h2_answer": DecisionAction.DEFER,
        "h2_payload": "insufficient evidence to confirm routing capability",
    },
]


class StructuralOODGenerator:
    """Generates structurally varied tasks for I3.9 generalization testing."""

    def __init__(self, n_tasks: int = 300, split: str = "structural_ood_v1") -> None:
        self.n_tasks = n_tasks
        self.split = split

    def generate(self) -> tuple[EvidenceTask, ...]:
        """Generate n_tasks with structural variation."""
        tasks: list[EvidenceTask] = []

        # Pattern distribution: ensure all adversarial subgroups are represented
        # Plus structural variations of standard patterns
        patterns = [
            "single_verify_ready",      # 1 VERIFY then ready
            "triple_verify_ready",      # 3 VERIFYs then ready
            "noise_evidence",           # extra irrelevant evidence
            "three_hypothesis",         # 3 hypotheses
            "early_false_ready",        # adversarial: wrong hypothesis looks viable
            "late_resolution",          # 4+ operations
            "multi_hypothesis_ambiguity",  # 3 hypotheses, complex
            "stale_support",            # visible support is stale
            "conflict_unresolved",      # genuine unresolvable, DEFER
            "varying_visible_split",    # different visible/hidden ratios
        ]

        for i in range(self.n_tasks):
            task_id = f"i3_9_structural_{self.split}_{i:04d}"
            rng = _seeded_rng(task_id)
            template = STRUCTURAL_TEMPLATES[i % len(STRUCTURAL_TEMPLATES)]
            pattern = patterns[i % len(patterns)]
            task = self._generate_one(task_id, template, pattern, rng)
            tasks.append(task)

        return tuple(tasks)

    def _generate_one(
        self, task_id: str, template: dict, pattern: str, rng: random.Random,
    ) -> EvidenceTask:
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

        dispatch = {
            "single_verify_ready": self._gen_single_verify_ready,
            "triple_verify_ready": self._gen_triple_verify_ready,
            "noise_evidence": self._gen_noise_evidence,
            "three_hypothesis": self._gen_three_hypothesis,
            "early_false_ready": self._gen_early_false_ready,
            "late_resolution": self._gen_late_resolution,
            "multi_hypothesis_ambiguity": self._gen_multi_hypothesis_ambiguity,
            "stale_support": self._gen_stale_support,
            "conflict_unresolved": self._gen_conflict_unresolved,
            "varying_visible_split": self._gen_varying_visible_split,
        }

        handler = dispatch.get(pattern, self._gen_single_verify_ready)
        return handler(task_id, subject, h1, h2, template, rng)

    # -----------------------------------------------------------------------
    # Structural variation: 1-VERIFY ready
    # -----------------------------------------------------------------------

    def _gen_single_verify_ready(
        self, task_id: str, subject: str,
        h1: EvidenceHypothesis, h2: EvidenceHypothesis,
        template: dict, rng: random.Random,
    ) -> EvidenceTask:
        """Only 1 visible evidence item. After 1 VERIFY, ready to answer.
        Tests: does MDSG recognize readiness with minimal evidence?"""
        evidence = (
            EvidenceItem(
                evidence_id="E1",
                proposition=f"The primary documentation confirms that {subject}.",
                source_class="primary",
                supports=("H1",),
                contradicts=("H2",),
                verification_state=VerificationState.UNVERIFIED,
                temporal_status=TemporalStatus.CURRENT,
                retrieved=True,
                verify_result="SUFFICIENT",
            ),
            EvidenceItem(
                evidence_id="E2",
                proposition=f"A secondary source does not address {subject}.",
                source_class="search",
                supports=("H2",),
                contradicts=(),
                verification_state=VerificationState.UNVERIFIED,
                temporal_status=TemporalStatus.CURRENT,
                retrieved=False,
                verify_result="MISSING",
            ),
        )
        return EvidenceTask(
            task_id=task_id, split=self.split,
            category="single_verify_ready",
            task_summary=f"Determine {subject}.",
            high_stakes=rng.random() > 0.5,
            budget_profile="STANDARD",
            hypotheses=(h1, h2),
            evidence_items=evidence,
            retrieve_exposes=(),
            search_exposes=("E2",),
            oracle_resolution_path=("VERIFY:E1", "ANSWER"),
            expected_terminal=DecisionAction.ANSWER,
            correct_hypothesis_id="H1",
        )

    # -----------------------------------------------------------------------
    # Structural variation: 3-VERIFY ready
    # -----------------------------------------------------------------------

    def _gen_triple_verify_ready(
        self, task_id: str, subject: str,
        h1: EvidenceHypothesis, h2: EvidenceHypothesis,
        template: dict, rng: random.Random,
    ) -> EvidenceTask:
        """3 visible evidence items. After 3 VERIFYs, ready to answer.
        Tests: does MDSG handle longer verification sequences?"""
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
                proposition=f"Source B also claims {subject}.",
                source_class="initial",
                supports=("H1",),
                contradicts=("H2",),
                verification_state=VerificationState.UNVERIFIED,
                temporal_status=TemporalStatus.CURRENT,
                retrieved=True,
                verify_result="SUFFICIENT",
            ),
            EvidenceItem(
                evidence_id="E3",
                proposition=f"Source C contradicts, claiming not-{subject}.",
                source_class="initial",
                supports=("H2",),
                contradicts=("H1",),
                verification_state=VerificationState.UNVERIFIED,
                temporal_status=TemporalStatus.CURRENT,
                retrieved=True,
                verify_result="FALSIFIED",
            ),
        )
        return EvidenceTask(
            task_id=task_id, split=self.split,
            category="triple_verify_ready",
            task_summary=f"Determine {subject}.",
            high_stakes=True,
            budget_profile="STANDARD",
            hypotheses=(h1, h2),
            evidence_items=evidence,
            retrieve_exposes=(),
            search_exposes=(),
            oracle_resolution_path=("VERIFY:E1", "VERIFY:E2", "VERIFY:E3", "ANSWER"),
            expected_terminal=DecisionAction.ANSWER,
            correct_hypothesis_id="H1",
        )

    # -----------------------------------------------------------------------
    # Structural variation: noise/irrelevant evidence
    # -----------------------------------------------------------------------

    def _gen_noise_evidence(
        self, task_id: str, subject: str,
        h1: EvidenceHypothesis, h2: EvidenceHypothesis,
        template: dict, rng: random.Random,
    ) -> EvidenceTask:
        """Extra irrelevant evidence items that don't support or contradict
        any hypothesis. Tests: does MDSG ignore noise?"""
        evidence = (
            EvidenceItem(
                evidence_id="E1",
                proposition=f"The documentation confirms that {subject}.",
                source_class="primary",
                supports=("H1",),
                contradicts=("H2",),
                verification_state=VerificationState.UNVERIFIED,
                temporal_status=TemporalStatus.CURRENT,
                retrieved=True,
                verify_result="SUFFICIENT",
            ),
            EvidenceItem(
                evidence_id="E2",
                proposition=f"A source contradicts, claiming not-{subject}.",
                source_class="initial",
                supports=("H2",),
                contradicts=("H1",),
                verification_state=VerificationState.UNVERIFIED,
                temporal_status=TemporalStatus.CURRENT,
                retrieved=True,
                verify_result="FALSIFIED",
            ),
            # Noise items — irrelevant
            EvidenceItem(
                evidence_id="E3",
                proposition="The system uses 64-bit integers for internal IDs.",
                source_class="initial",
                supports=(),
                contradicts=(),
                verification_state=VerificationState.UNVERIFIED,
                temporal_status=TemporalStatus.CURRENT,
                retrieved=True,
                verify_result="SUFFICIENT",
            ),
            EvidenceItem(
                evidence_id="E4",
                proposition="The logging framework outputs JSON format.",
                source_class="initial",
                supports=(),
                contradicts=(),
                verification_state=VerificationState.UNVERIFIED,
                temporal_status=TemporalStatus.CURRENT,
                retrieved=True,
                verify_result="SUFFICIENT",
            ),
        )
        return EvidenceTask(
            task_id=task_id, split=self.split,
            category="noise_evidence",
            task_summary=f"Determine {subject}.",
            high_stakes=rng.random() > 0.5,
            budget_profile="STANDARD",
            hypotheses=(h1, h2),
            evidence_items=evidence,
            retrieve_exposes=(),
            search_exposes=(),
            oracle_resolution_path=("VERIFY:E1", "VERIFY:E2", "ANSWER"),
            expected_terminal=DecisionAction.ANSWER,
            correct_hypothesis_id="H1",
        )

    # -----------------------------------------------------------------------
    # Structural variation: 3 hypotheses
    # -----------------------------------------------------------------------

    def _gen_three_hypothesis(
        self, task_id: str, subject: str,
        h1: EvidenceHypothesis, h2: EvidenceHypothesis,
        template: dict, rng: random.Random,
    ) -> EvidenceTask:
        """3 competing hypotheses. Tests: does MDSG handle >2 hypotheses?"""
        h3 = EvidenceHypothesis(
            hypothesis_id="H3",
            proposition=f"the documentation is ambiguous about {subject}, so the answer should be DEFER",
            answer_action=DecisionAction.DEFER,
            answer_payload="documentation is ambiguous",
        )

        evidence = (
            EvidenceItem(
                evidence_id="E1",
                proposition=f"Source A claims {subject}.",
                source_class="initial",
                supports=("H1",),
                contradicts=("H2", "H3"),
                verification_state=VerificationState.UNVERIFIED,
                temporal_status=TemporalStatus.CURRENT,
                retrieved=True,
                verify_result="SUFFICIENT",
            ),
            EvidenceItem(
                evidence_id="E2",
                proposition=f"Source B claims not-{subject}.",
                source_class="initial",
                supports=("H2",),
                contradicts=("H1",),
                verification_state=VerificationState.UNVERIFIED,
                temporal_status=TemporalStatus.CURRENT,
                retrieved=True,
                verify_result="FALSIFIED",
            ),
            EvidenceItem(
                evidence_id="E3",
                proposition=f"Source C is ambiguous about {subject}.",
                source_class="initial",
                supports=("H3",),
                contradicts=(),
                verification_state=VerificationState.UNVERIFIED,
                temporal_status=TemporalStatus.CURRENT,
                retrieved=True,
                verify_result="MISSING",
            ),
        )
        return EvidenceTask(
            task_id=task_id, split=self.split,
            category="three_hypothesis",
            task_summary=f"Determine {subject}.",
            high_stakes=True,
            budget_profile="STANDARD",
            hypotheses=(h1, h2, h3),
            evidence_items=evidence,
            retrieve_exposes=(),
            search_exposes=(),
            oracle_resolution_path=("VERIFY:E1", "VERIFY:E2", "VERIFY:E3", "ANSWER"),
            expected_terminal=DecisionAction.ANSWER,
            correct_hypothesis_id="H1",
        )

    # -----------------------------------------------------------------------
    # Adversarial: EARLY_FALSE_READY
    # -----------------------------------------------------------------------

    def _gen_early_false_ready(
        self, task_id: str, subject: str,
        h1: EvidenceHypothesis, h2: EvidenceHypothesis,
        template: dict, rng: random.Random,
    ) -> EvidenceTask:
        """Adversarial: visible evidence makes H1 look viable, but H2 is
        actually correct. Hidden evidence is needed to find the right answer.

        After VERIFY E1, VERIFY E2:
          - E1 SUFFICIENT supports H1 → H1 looks VIABLE
          - E2 FALSIFIED → H1's contradiction eliminated
          - MDSG says READY_TO_ANSWER
          - Model ANSWERs → FAILS (correct is H2, no support for H2)

        Correct path: RETRIEVE E3, VERIFY E3, ANSWER
          - E3 SUFFICIENT supports H2, contradicts H1
          - Now H1 ELIMINATED, H2 VIABLE → READY_TO_ANSWER
          - Model ANSWERs → SUCCESS

        This tests whether MDSG prematurely signals readiness when hidden
        evidence could overturn the apparent conclusion.
        """
        # Both hypotheses use ANSWER with different payloads
        h1_answer = EvidenceHypothesis(
            hypothesis_id="H1",
            proposition=template["h1_proposition"],
            answer_action=DecisionAction.ANSWER,
            answer_payload=template["h1_payload"],
        )
        h2_answer = EvidenceHypothesis(
            hypothesis_id="H2",
            proposition=f"the documentation refutes the claim about {subject}, so the answer should be ANSWER with refutation",
            answer_action=DecisionAction.ANSWER,
            answer_payload=f"refuted: {template['h1_payload']}",
        )

        evidence = (
            # Visible: makes H1 look viable
            EvidenceItem(
                evidence_id="E1",
                proposition=f"An initial source claims {subject}.",
                source_class="initial",
                supports=("H1",),
                contradicts=("H2",),
                verification_state=VerificationState.UNVERIFIED,
                temporal_status=TemporalStatus.CURRENT,
                retrieved=True,
                verify_result="SUFFICIENT",
            ),
            EvidenceItem(
                evidence_id="E2",
                proposition=f"Another source contradicts, claiming not-{subject}.",
                source_class="initial",
                supports=("H2",),
                contradicts=("H1",),
                verification_state=VerificationState.UNVERIFIED,
                temporal_status=TemporalStatus.CURRENT,
                retrieved=True,
                verify_result="FALSIFIED",
            ),
            # Hidden: overturns H1, supports H2
            EvidenceItem(
                evidence_id="E3",
                proposition=f"A definitive source refutes the claim about {subject}.",
                source_class="primary",
                supports=("H2",),
                contradicts=("H1",),
                verification_state=VerificationState.UNVERIFIED,
                temporal_status=TemporalStatus.CURRENT,
                retrieved=False,
                verify_result="SUFFICIENT",
            ),
        )
        return EvidenceTask(
            task_id=task_id, split=self.split,
            category="early_false_ready",
            task_summary=f"Determine {subject}.",
            high_stakes=True,
            budget_profile="STANDARD",
            hypotheses=(h1_answer, h2_answer),
            evidence_items=evidence,
            retrieve_exposes=("E3",),
            search_exposes=(),
            oracle_resolution_path=("RETRIEVE:E3", "VERIFY:E3", "ANSWER"),
            expected_terminal=DecisionAction.ANSWER,
            correct_hypothesis_id="H2",
        )

    # -----------------------------------------------------------------------
    # Structural variation: late resolution (4+ operations)
    # -----------------------------------------------------------------------

    def _gen_late_resolution(
        self, task_id: str, subject: str,
        h1: EvidenceHypothesis, h2: EvidenceHypothesis,
        template: dict, rng: random.Random,
    ) -> EvidenceTask:
        """Requires 4+ operations: RETRIEVE, VERIFY, SEARCH_MORE, VERIFY, ANSWER.
        Tests: does MDSG maintain patience through longer sequences?"""
        evidence = (
            EvidenceItem(
                evidence_id="E1",
                proposition=f"An initial source vaguely mentions {subject}.",
                source_class="initial",
                supports=("H1",),
                contradicts=(),
                verification_state=VerificationState.UNVERIFIED,
                temporal_status=TemporalStatus.CURRENT,
                retrieved=True,
                verify_result="MISSING",
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
                verify_result="FALSIFIED",
            ),
            EvidenceItem(
                evidence_id="E3",
                proposition=f"A retrieved source provides partial confirmation of {subject}.",
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
                proposition=f"A searched source provides definitive confirmation of {subject}.",
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
            task_id=task_id, split=self.split,
            category="late_resolution",
            task_summary=f"Determine {subject}.",
            high_stakes=True,
            budget_profile="STANDARD",
            hypotheses=(h1, h2),
            evidence_items=evidence,
            retrieve_exposes=("E3",),
            search_exposes=("E4",),
            oracle_resolution_path=(
                "RETRIEVE:E3", "VERIFY:E3", "SEARCH_MORE:E4", "VERIFY:E4", "ANSWER",
            ),
            expected_terminal=DecisionAction.ANSWER,
            correct_hypothesis_id="H1",
        )

    # -----------------------------------------------------------------------
    # Adversarial: MULTI_HYPOTHESIS_AMBIGUITY
    # -----------------------------------------------------------------------

    def _gen_multi_hypothesis_ambiguity(
        self, task_id: str, subject: str,
        h1: EvidenceHypothesis, h2: EvidenceHypothesis,
        template: dict, rng: random.Random,
    ) -> EvidenceTask:
        """3 hypotheses with complex topology. Two look viable initially.
        Tests: does MDSG correctly say NEEDS_DISCRIMINATION when multiple
        hypotheses have support?"""
        h3 = EvidenceHypothesis(
            hypothesis_id="H3",
            proposition=f"the documentation is ambiguous about {subject}, so the answer should be DEFER",
            answer_action=DecisionAction.DEFER,
            answer_payload="documentation is ambiguous",
        )

        evidence = (
            # E1 supports H1
            EvidenceItem(
                evidence_id="E1",
                proposition=f"Source A claims {subject}.",
                source_class="initial",
                supports=("H1",),
                contradicts=("H2",),
                verification_state=VerificationState.UNVERIFIED,
                temporal_status=TemporalStatus.CURRENT,
                retrieved=True,
                verify_result="SUFFICIENT",
            ),
            # E2 supports H2 — so after verifying both, two viable hypotheses
            EvidenceItem(
                evidence_id="E2",
                proposition=f"Source B claims not-{subject}.",
                source_class="initial",
                supports=("H2",),
                contradicts=("H1",),
                verification_state=VerificationState.UNVERIFIED,
                temporal_status=TemporalStatus.CURRENT,
                retrieved=True,
                verify_result="SUFFICIENT",
            ),
            # E3 (hidden) breaks the tie — contradicts H2
            EvidenceItem(
                evidence_id="E3",
                proposition=f"A definitive source confirms {subject}.",
                source_class="search",
                supports=("H1",),
                contradicts=("H2", "H3"),
                verification_state=VerificationState.UNVERIFIED,
                temporal_status=TemporalStatus.CURRENT,
                retrieved=False,
                verify_result="SUFFICIENT",
            ),
        )
        return EvidenceTask(
            task_id=task_id, split=self.split,
            category="multi_hypothesis_ambiguity",
            task_summary=f"Determine {subject}.",
            high_stakes=True,
            budget_profile="STANDARD",
            hypotheses=(h1, h2, h3),
            evidence_items=evidence,
            retrieve_exposes=(),
            search_exposes=("E3",),
            oracle_resolution_path=(
                "VERIFY:E1", "VERIFY:E2", "SEARCH_MORE:E3", "VERIFY:E3", "ANSWER",
            ),
            expected_terminal=DecisionAction.ANSWER,
            correct_hypothesis_id="H1",
        )

    # -----------------------------------------------------------------------
    # Adversarial: STALE_SUPPORT
    # -----------------------------------------------------------------------

    def _gen_stale_support(
        self, task_id: str, subject: str,
        h1: EvidenceHypothesis, h2: EvidenceHypothesis,
        template: dict, rng: random.Random,
    ) -> EvidenceTask:
        """Visible support is STALE. After verification, E1 becomes STALE
        (not SUFFICIENT), so H1 is NOT viable. Must search for current evidence.
        Tests: does MDSG correctly handle STALE verification results?"""
        evidence = (
            EvidenceItem(
                evidence_id="E1",
                proposition=f"An older source claims {subject}, but may be outdated.",
                source_class="initial",
                supports=("H1",),
                contradicts=(),
                verification_state=VerificationState.UNVERIFIED,
                temporal_status=TemporalStatus.STALE,
                retrieved=True,
                verify_result="STALE",
            ),
            EvidenceItem(
                evidence_id="E2",
                proposition=f"A source is silent on {subject}.",
                source_class="initial",
                supports=("H2",),
                contradicts=("H1",),
                verification_state=VerificationState.UNVERIFIED,
                temporal_status=TemporalStatus.CURRENT,
                retrieved=True,
                verify_result="FALSIFIED",
            ),
            EvidenceItem(
                evidence_id="E3",
                proposition=f"A current source confirms {subject}.",
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
            task_id=task_id, split=self.split,
            category="stale_support",
            task_summary=f"Determine {subject}.",
            high_stakes=True,
            budget_profile="STANDARD",
            hypotheses=(h1, h2),
            evidence_items=evidence,
            retrieve_exposes=(),
            search_exposes=("E3",),
            oracle_resolution_path=(
                "VERIFY:E1", "SEARCH_MORE:E3", "VERIFY:E3", "ANSWER",
            ),
            expected_terminal=DecisionAction.ANSWER,
            correct_hypothesis_id="H1",
        )

    # -----------------------------------------------------------------------
    # Adversarial: CONFLICT_UNRESOLVED
    # -----------------------------------------------------------------------

    def _gen_conflict_unresolved(
        self, task_id: str, subject: str,
        h1: EvidenceHypothesis, h2: EvidenceHypothesis,
        template: dict, rng: random.Random,
    ) -> EvidenceTask:
        """Genuine unresolvable conflict. Both sides have SUFFICIENT support.
        No hidden evidence can break the tie. Expected terminal: DEFER.
        Tests: does MDSG correctly say INSUFFICIENT or NEEDS_EVIDENCE
        rather than READY_TO_ANSWER?"""
        evidence = (
            EvidenceItem(
                evidence_id="E1",
                proposition=f"Source A definitively confirms {subject}.",
                source_class="primary",
                supports=("H1",),
                contradicts=("H2",),
                verification_state=VerificationState.UNVERIFIED,
                temporal_status=TemporalStatus.CURRENT,
                retrieved=True,
                verify_result="SUFFICIENT",
            ),
            EvidenceItem(
                evidence_id="E2",
                proposition=f"Source B definitively refutes {subject}.",
                source_class="primary",
                supports=("H2",),
                contradicts=("H1",),
                verification_state=VerificationState.UNVERIFIED,
                temporal_status=TemporalStatus.CURRENT,
                retrieved=True,
                verify_result="SUFFICIENT",
            ),
        )
        return EvidenceTask(
            task_id=task_id, split=self.split,
            category="conflict_unresolved",
            task_summary=f"Determine {subject}.",
            high_stakes=True,
            budget_profile="STANDARD",
            hypotheses=(h1, h2),
            evidence_items=evidence,
            retrieve_exposes=(),
            search_exposes=(),
            oracle_resolution_path=("VERIFY:E1", "VERIFY:E2", "DEFER"),
            expected_terminal=DecisionAction.DEFER,
            correct_hypothesis_id="H2",
        )

    # -----------------------------------------------------------------------
    # Structural variation: varying visible/hidden split
    # -----------------------------------------------------------------------

    def _gen_varying_visible_split(
        self, task_id: str, subject: str,
        h1: EvidenceHypothesis, h2: EvidenceHypothesis,
        template: dict, rng: random.Random,
    ) -> EvidenceTask:
        """3 visible + 1 hidden evidence. More visible evidence than standard.
        Tests: does MDSG handle different visible/hidden ratios?"""
        evidence = (
            EvidenceItem(
                evidence_id="E1",
                proposition=f"Source A confirms {subject}.",
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
                proposition=f"Source B also confirms {subject}.",
                source_class="initial",
                supports=("H1",),
                contradicts=("H2",),
                verification_state=VerificationState.UNVERIFIED,
                temporal_status=TemporalStatus.CURRENT,
                retrieved=True,
                verify_result="SUFFICIENT",
            ),
            EvidenceItem(
                evidence_id="E3",
                proposition=f"Source C contradicts, claiming not-{subject}.",
                source_class="initial",
                supports=("H2",),
                contradicts=("H1",),
                verification_state=VerificationState.UNVERIFIED,
                temporal_status=TemporalStatus.CURRENT,
                retrieved=True,
                verify_result="FALSIFIED",
            ),
            EvidenceItem(
                evidence_id="E4",
                proposition=f"A hidden source provides additional confirmation of {subject}.",
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
            task_id=task_id, split=self.split,
            category="varying_visible_split",
            task_summary=f"Determine {subject}.",
            high_stakes=rng.random() > 0.5,
            budget_profile="STANDARD",
            hypotheses=(h1, h2),
            evidence_items=evidence,
            retrieve_exposes=(),
            search_exposes=("E4",),
            oracle_resolution_path=("VERIFY:E1", "VERIFY:E2", "VERIFY:E3", "ANSWER"),
            expected_terminal=DecisionAction.ANSWER,
            correct_hypothesis_id="H1",
        )


def generate_structural_ood_tasks(
    n_tasks: int = 300,
    split: str = "structural_ood_v1",
) -> tuple[EvidenceTask, ...]:
    """Generate a frozen set of structurally varied OOD tasks."""
    gen = StructuralOODGenerator(n_tasks=n_tasks, split=split)
    return gen.generate()
