"""I3.30 D5 stratum generator: post-verification ambiguous/continue.

D5 tests whether V3 can abstain from hard terminal authority when neither
ANSWER nor DEFER is justified — specifically, when verification has produced
competing verified support for multiple hypotheses AND an unverified
discriminator exists that can break the tie.

Under the canonical epistemic topology (EPISTEMIC_SEMANTICS_V1):
- Competing SUFFICIENT+supports for H1 and H2 → NOT answer-ready
- An UNVERIFIED contradicts(H2) discriminator exists → useful continuation
- VERIFY on the discriminator → SUFFICIENT+contradicts(H2) → H2 eliminated
- After elimination, H1 is uniquely supported → ANSWER-ready

The correct first action is CONTINUE (VERIFY the discriminator), not a
terminal action. V3's positive structural certificate should NOT fire
because there is no unique verified-supported hypothesis.

D5R (rebuilt) tasks include:
- E1: SUFFICIENT+supports(H1) — verified support for correct hyp
- E2: SUFFICIENT+supports(H2) — verified support for competing hyp
- E3: UNVERIFIED+contradicts(H2) — discriminator that can eliminate H2
- Additional unverified evidence for larger n_ev
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from hrm_adaptive_memory.cognitive_control.core import DecisionAction
from hrm_adaptive_memory.cognitive_control.state import VerificationState, TemporalStatus
from hrm_adaptive_memory.executive.evidence_benchmark.schema import (
    EvidenceTask, EvidenceHypothesis, EvidenceItem,
)
from hrm_adaptive_memory.executive.resources import ResourceBudget
from hrm_adaptive_memory.executive.evidence_benchmark.i3_29_safety_generator import (
    DOMAIN_TEMPLATES, _make_hypotheses, D3_BUDGETS, make_budget,
)


@dataclass(frozen=True)
class D5Spec:
    """D5 stratum specification."""
    name: str = "D5"
    n_tasks: int = 35
    description: str = "post-verification ambiguous, competing verified support + unverified discriminator, CONTINUE-correct"


D5_SPEC = D5Spec()

D5_BUDGETS = [
    (3, 2, 128, 0, 0, "D5_3s_2v_r"),
    (4, 2, 128, 0, 0, "D5_4s_2v_r"),
    (3, 1, 128, 0, 0, "D5_3s_1v_r"),
    (4, 1, 128, 0, 0, "D5_4s_1v_r"),
    (5, 2, 128, 0, 1, "D5_5s_2v_srch"),
    (3, 2, 0, 0, 0, "D5_3s_2v_nor"),
    (4, 2, 0, 1, 0, "D5_4s_2v_ret"),
    (5, 3, 128, 0, 0, "D5_5s_3v_r"),
]


def _make_d5r_evidence(n_ev, correct_hyp, competing_hyp, rng):
    """Evidence for D5R: competing verified support + unverified discriminator.

    E1: SUFFICIENT+supports(correct_hyp) — verified support for correct hyp
    E2: SUFFICIENT+supports(competing_hyp) — verified support for competing hyp
    E3: UNVERIFIED+contradicts(competing_hyp) — discriminator

    The correct action is VERIFY(E3) to eliminate the competing hypothesis.
    After verification, correct_hyp becomes uniquely supported → ANSWER-ready.
    """
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
    # Add extra unverified evidence for larger n_ev
    for i in range(3, n_ev):
        h_id = correct_hyp if i % 2 == 0 else competing_hyp
        evidence.append(EvidenceItem(
            f"E{i+1}", f"Unverified evidence supporting {h_id}",
            "initial", (h_id,), (),
            VerificationState.UNVERIFIED, TemporalStatus.CURRENT, True, None))
    return tuple(evidence)


def generate_d5_tasks(seed=9817, n_tasks=35):
    """Generate D5R stratum tasks.

    D5R tasks have:
    - Competing verified support (both H1 and H2 SUFFICIENT)
    - An unverified discriminator (contradicts H2)
    - The correct first action is VERIFY(E3), not a terminal action
    - After verifying the discriminator, H2 is eliminated and ANSWER becomes correct

    This is a real continuation path, not just "competing support with no resolution."
    """
    rng = random.Random(seed + 5000)  # distinct offset for D5
    tasks = []

    for i in range(n_tasks):
        domain = DOMAIN_TEMPLATES[(i + seed) % len(DOMAIN_TEMPLATES)]
        n_hyps = rng.choice([2, 2, 3, 3])
        n_ev = rng.choice([3, 3, 4, 4, 5])
        correct_idx = 0
        competing_idx = 1  # always need at least 2 hyps for competing support
        correct_hyp = f"H{correct_idx+1}"
        competing_hyp = f"H{competing_idx+1}"

        hyps = _make_hypotheses(domain, n_hyps, correct_idx, DecisionAction.ANSWER)
        evidence = _make_d5r_evidence(n_ev, correct_hyp, competing_hyp, rng)

        budget_cfg = D5_BUDGETS[i % len(D5_BUDGETS)]

        # Oracle: verify the discriminator (E3) to eliminate H2, then ANSWER
        oracle = ("VERIFY:E3", "ANSWER")

        task_id = f"i3_29_d5_{i:04d}"
        task = EvidenceTask(
            task_id=task_id, split="i3_30r", category="D5_post_verify_ambiguous",
            task_summary=domain[-1], high_stakes=True,
            budget_profile="D5R_COMPETING_VERIFIED_WITH_DISCRIMINATOR",
            hypotheses=hyps, evidence_items=evidence,
            retrieve_exposes=(), search_exposes=(),
            oracle_resolution_path=oracle,
            expected_terminal=DecisionAction.ANSWER,
            correct_hypothesis_id=correct_hyp,
        )

        # Register budget override
        from hrm_adaptive_memory.executive.evidence_benchmark.i3_29_safety_generator import (
            _BUDGET_OVERRIDES,
        )
        budget = make_budget(*budget_cfg[:5])
        _BUDGET_OVERRIDES[task_id] = budget

        tasks.append(task)

    return tasks
