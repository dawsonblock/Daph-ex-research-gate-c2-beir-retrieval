from __future__ import annotations

from collections.abc import Iterable

from .base import ComparisonOutcome, ComparisonResult, RelationSchema
from ._helpers import canonical_string


class SetMembershipComparator:
    def compare(self, schema: RelationSchema, claim_value: object,
                evidence_value: object) -> ComparisonResult:
        try:
            if isinstance(evidence_value, (str, bytes)) or not isinstance(evidence_value, Iterable):
                raise ValueError("evidence value must be a set-like collection")
            claim = canonical_string(claim_value)
            evidence = tuple(sorted({canonical_string(value) for value in evidence_value}))
            matched = claim in evidence
            return ComparisonResult(ComparisonOutcome.MATCH if matched else ComparisonOutcome.MISMATCH,
                                    claim, evidence,
                                    "SET_MEMBER" if matched else "SET_NOT_MEMBER")
        except ValueError:
            return ComparisonResult(ComparisonOutcome.INCONCLUSIVE, None, None,
                                    "SET_MEMBERSHIP_PARSE_ERROR")
