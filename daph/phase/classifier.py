"""DAPH I3.4 — Deterministic reference phase classifier.

I3.4a starts with a deterministic classifier derived from observable
MDSG state. Later, semantic uncertainty can produce calibrated confidence.

The classifier uses the actual MDSG semantics (decision_state, n_live,
n_eliminated, t2) rather than naïve counts alone.

Classification rules (checked in order, first match wins):

1. NO_VIABLE_HYPOTHESIS
   t2 == True OR (n_live == 0 AND n_total > 0)
   All hypotheses eliminated.

2. ANSWER_READY
   decision_state == "READY_TO_ANSWER"
   Evidence supports a viable answer.

3. DISCRIMINATION
   n_live >= 2
   Multiple viable hypotheses remain.

4. RESOLUTION
   n_live == 1 AND decision_state in ("SUPPORTED_BUT_UNRESOLVED", "INSUFFICIENT")
   Narrowed to one hypothesis but not yet terminal.

5. EVIDENCE_ACQUISITION
   default
   Evidence coverage not yet sufficient.
"""

from __future__ import annotations

from typing import Any, Mapping

from daph.phase.ontology import EpistemicPhase, Phase
from daph.phase.features import PhaseFeatures


def classify_phase(
    *,
    decision_state: str = "",
    n_live: int = 0,
    n_eliminated: int = 0,
    n_total: int = 0,
    t2: bool = False,
) -> EpistemicPhase:
    """Classify the epistemic phase from observable state.

    Args:
        decision_state: the exposed decision state label
        n_live: number of live (non-eliminated) hypotheses
        n_eliminated: number of eliminated hypotheses
        n_total: total hypotheses (n_live + n_eliminated)
        t2: whether T2 has fired (all hypotheses eliminated)

    Returns:
        EpistemicPhase with confidence=1.0 (deterministic)
    """
    # Rule 1: NO_VIABLE_HYPOTHESIS
    if t2 or (n_live == 0 and n_total > 0):
        return EpistemicPhase(
            phase=Phase.NO_VIABLE_HYPOTHESIS,
            confidence=1.0,
            evidence_basis=("t2", "n_live", "n_total"),
            ambiguous=False,
        )

    # Rule 2: ANSWER_READY
    if decision_state == "READY_TO_ANSWER":
        return EpistemicPhase(
            phase=Phase.ANSWER_READY,
            confidence=1.0,
            evidence_basis=("decision_state",),
            ambiguous=False,
        )

    # Rule 3: DISCRIMINATION
    if n_live >= 2:
        return EpistemicPhase(
            phase=Phase.DISCRIMINATION,
            confidence=1.0,
            evidence_basis=("n_live",),
            ambiguous=False,
        )

    # Rule 4: RESOLUTION
    if n_live == 1:
        return EpistemicPhase(
            phase=Phase.RESOLUTION,
            confidence=1.0,
            evidence_basis=("n_live", "decision_state"),
            ambiguous=False,
        )

    # Rule 5: EVIDENCE_ACQUISITION (default)
    return EpistemicPhase(
        phase=Phase.EVIDENCE_ACQUISITION,
        confidence=1.0,
        evidence_basis=("default",),
        ambiguous=False,
    )


def classify_from_receipt(receipt: Mapping[str, Any]) -> EpistemicPhase:
    """Classify phase from a mechanism receipt."""
    return classify_phase(
        decision_state=receipt.get("decision_state_exposed", ""),
        n_live=int(receipt.get("n_live_hypotheses", 0)),
        n_eliminated=int(receipt.get("n_eliminated_hypotheses", 0)),
        n_total=int(receipt.get("n_live_hypotheses", 0)) + int(receipt.get("n_eliminated_hypotheses", 0)),
        t2=bool(receipt.get("t2", False)),
    )


def classify_from_features(features: PhaseFeatures) -> EpistemicPhase:
    """Classify phase from a PhaseFeatures vector."""
    return classify_phase(
        decision_state=features.decision_state,
        n_live=features.n_live,
        n_eliminated=features.n_eliminated,
        n_total=features.n_total,
        t2=features.t2,
    )
