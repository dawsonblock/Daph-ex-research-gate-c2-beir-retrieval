"""DAPH Epistemic Semantics — canonical hypothesis topology derivation.

This module implements the single normative interpretation of evidence,
hypothesis viability, and terminal readiness defined in
EPISTEMIC_SEMANTICS_V1.md.

All consumers (MDSG classifier, Q feature extraction, authority
certificates, executor success criteria, benchmark analysis) MUST derive
epistemic state from this module, not from independent re-implementations.
"""
from daph.epistemic.types import (
    HypothesisState,
    HypothesisTopology,
    TerminalReadiness,
)
from daph.epistemic.topology import (
    derive_hypothesis_topology,
    classify_terminal_readiness,
    is_answer_ready,
    is_defer_ready,
    is_continue_required,
)

__all__ = [
    "HypothesisState",
    "HypothesisTopology",
    "TerminalReadiness",
    "derive_hypothesis_topology",
    "classify_terminal_readiness",
    "is_answer_ready",
    "is_defer_ready",
    "is_continue_required",
]
