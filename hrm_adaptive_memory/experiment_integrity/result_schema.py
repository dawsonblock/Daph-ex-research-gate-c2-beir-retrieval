"""The shared result schema this project has needed since Confirmation #1.

Two collapses have already caused real confusion in this project's own
history, both recorded in RESEARCH_STATUS.json:

    Confirmation #1: RUN_VALID=true, but the mechanism itself failed
                     (SCIENTIFIC_PASS=false, MECHANISM_PASS=false)
    G2-v1:           RUN_VALID=true, OUTCOME_C=construction deficiency,
                     PROMOTED=false

Both are legitimate, valid, informative runs. Neither is a "pass." A single
boolean VALID_RUN or PASS field cannot represent either case without losing
information a reader needs. This module makes the distinction a first-class,
enforced part of every gate's result, instead of relying on prose in a commit
message to carry it.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ScientificVerdict(str, Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    PARTIAL = "PARTIAL"
    NOT_COMPUTABLE = "NOT_COMPUTABLE"


class MechanismStatus(str, Enum):
    PROMOTED = "PROMOTED"
    NOT_PROMOTED = "NOT_PROMOTED"
    PENDING_FURTHER_EVIDENCE = "PENDING_FURTHER_EVIDENCE"


class FailureClass(str, Enum):
    NONE = "NONE"
    CONSTRUCTION_DEFICIENT = "CONSTRUCTION_DEFICIENT"
    TOPOLOGY_DEFICIENT = "TOPOLOGY_DEFICIENT"
    RECOGNIZER_INADEQUATE = "RECOGNIZER_INADEQUATE"
    SELECTOR_LIMITED = "SELECTOR_LIMITED"
    RETRIEVAL_LIMITED = "RETRIEVAL_LIMITED"
    SAFETY_BOUND_VIOLATED = "SAFETY_BOUND_VIOLATED"
    #: The mechanism under test is not at fault -- the DATASET cannot express
    #: the question being asked of it. First needed for the Executive
    #: Opportunity Study (evidence/gate_executive/opportunity_execute.json):
    #: ANSWER_NOW scored exactly 0/750 because b3_calibration_v1's facts are
    #: procedurally synthesized per corpus build and cannot exist in any
    #: pretrained model's weights, so the benchmark structurally cannot
    #: produce ANSWER_NOW-vs-MEMORY heterogeneity regardless of how good or
    #: bad the action-selection mechanism is. Distinct from every other
    #: FailureClass here, which all describe a deficiency IN the mechanism.
    BENCHMARK_HAS_NO_ACTION_HETEROGENEITY = "BENCHMARK_HAS_NO_ACTION_HETEROGENEITY"
    #: The opposite failure mode from BENCHMARK_HAS_NO_ACTION_HETEROGENEITY:
    #: real, strong heterogeneity exists WITHIN individual regimes/strata
    #: (e.g. EOB-v1's D1 shows 48/100 strict memory wins, D3 shows 64/100
    #: strict answer-now wins), but the frozen benchmark's regime/stratum MIX
    #: proportions dilute the POOLED diversity metric below its floor (e.g.
    #: D0's 100/100 ties pull memory's aggregate strict-win share to 14.75%,
    #: just under a 15% floor, even though memory clearly matters within D1).
    #: Not a mechanism deficiency and not "no heterogeneity" -- a mix-design
    #: property of THIS specific frozen task composition.
    DIVERSITY_FLOOR_NOT_CLEARED_IN_POOLED_MIX = "DIVERSITY_FLOOR_NOT_CLEARED_IN_POOLED_MIX"
    #: The probe signal DID separate MEMORY-strict-win from ANSWER-strict-win
    #: tasks (the frozen Cohen's d / bootstrap-CI stop-gate cleared, so
    #: training was not skipped) -- but the resulting fitted policy still
    #: failed to beat the better of the two trivial fixed policies
    #: (always-accept / always-escalate) on held-out eval. First needed for
    #: ANSWER_PROBE_GATE_V1 (evidence/gate_executive/exec_training_v1_execute.
    #: gate_result.json): Delta_U_gate=-0.1364, LCB=-0.3182, driven by severe
    #: label imbalance (escalation-helps was the minority class in a 100-row
    #: train split) that let the small logistic model learn a decision
    #: boundary worse than the trivial "always escalate" baseline. Distinct
    #: from BENCHMARK_HAS_NO_ACTION_HETEROGENEITY (no signal exists at all)
    #: and from every mechanism-deficiency class above (nothing about
    #: CERTIFIED_MEMORY_V1 itself is implicated) -- this describes the
    #: escalation POLICY specifically failing to exploit a signal that does
    #: exist, most plausibly due to sample size / class imbalance rather than
    #: the signal being valueless.
    LEARNED_POLICY_UNDERPERFORMS_FIXED_BASELINE = "LEARNED_POLICY_UNDERPERFORMS_FIXED_BASELINE"
    #: The promotion TEST itself lacked statistical power -- not that the
    #: policy failed. First needed for ANSWER_PROBE_GATE_V2 (evidence/
    #: gate_executive/exec_training_v2_execute.receipts.gate_result.json):
    #: after fixing a real retrieval-scope defect (see
    #: exec_training_v2_execute_ATTEMPT1_INVALID.md) that had crushed
    #: MEMORY_strict_win representation, the corrected run cleared that
    #: floor comfortably (53 in eval vs 40 required) but ANSWER_strict_win
    #: remained at only 5 in eval (10 required) -- the exact "honest
    #: limitation" flagged in configs/gate_answer_probe_v2_design.json
    #: PHASE_1 before any V2 data existed: ANSWER_strict_win's natural
    #: incidence is bottlenecked by hand-verified-fact curation, not
    #: compute, and further scaling that specific class was known in
    #: advance to be hard. Distinct from every other FailureClass here: no
    #: model was fit, no promotion/non-promotion claim is made, and this is
    #: NOT a second negative result -- per configs/gate_answer_probe_v2_
    #: design.json PHASE_5, a floor miss stops the pipeline before fitting.
    MINIMUM_CLASS_FLOOR_NOT_MET = "MINIMUM_CLASS_FLOOR_NOT_MET"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class SplitStatus(str, Enum):
    FRESH = "FRESH"
    CONSUMED = "CONSUMED"
    NOT_YET_RUN = "NOT_YET_RUN"


@dataclass(frozen=True)
class GateResult:
    """The shared shape every gate's result block should carry, on top of
    whatever gate-specific numbers it also reports. RUN_VALID (did the
    machinery execute correctly -- no leakage, no crashed arm, provenance
    intact) and MECHANISM_PASS/scientific_verdict (did the thing under test
    actually work) are independent axes and must never be collapsed into one
    field."""
    run_valid: bool
    scientific_verdict: ScientificVerdict
    mechanism_status: MechanismStatus
    failure_class: FailureClass
    split_status: SplitStatus

    def __post_init__(self):
        if not self.run_valid and self.mechanism_status == MechanismStatus.PROMOTED:
            raise ValueError(
                "run_valid=False cannot coexist with mechanism_status=PROMOTED -- "
                "an invalid run cannot promote anything, regardless of its numbers")
        if (self.scientific_verdict == ScientificVerdict.POSITIVE
                and self.mechanism_status == MechanismStatus.NOT_PROMOTED
                and self.failure_class == FailureClass.NONE):
            # Positive-but-not-promoted is legitimate (e.g. pending further
            # evidence) but should carry a reason; NONE alongside NOT_PROMOTED
            # on an otherwise-positive result is the one combination likely to
            # indicate a forgotten field rather than a real state.
            raise ValueError(
                "scientific_verdict=POSITIVE with mechanism_status=NOT_PROMOTED "
                "requires a failure_class other than NONE, or use "
                "PENDING_FURTHER_EVIDENCE instead of NOT_PROMOTED")

    def as_dict(self) -> dict[str, str | bool]:
        return {
            "run_valid": self.run_valid,
            "scientific_verdict": self.scientific_verdict.value,
            "mechanism_status": self.mechanism_status.value,
            "failure_class": self.failure_class.value,
            "split_status": self.split_status.value,
        }
