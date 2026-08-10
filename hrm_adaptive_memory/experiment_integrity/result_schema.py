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
