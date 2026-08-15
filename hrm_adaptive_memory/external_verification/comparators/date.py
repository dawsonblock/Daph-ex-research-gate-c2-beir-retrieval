from __future__ import annotations

from datetime import date

from .base import ComparisonOutcome, ComparisonResult, RelationSchema


class DateComparator:
    def compare(self, schema: RelationSchema, claim_value: object,
                evidence_value: object) -> ComparisonResult:
        try:
            claim, evidence = date.fromisoformat(str(claim_value)), date.fromisoformat(str(evidence_value))
            matched = claim == evidence
            return ComparisonResult(ComparisonOutcome.MATCH if matched else ComparisonOutcome.MISMATCH,
                                    claim.isoformat(), evidence.isoformat(),
                                    "DATE_MATCH" if matched else "DATE_MISMATCH")
        except ValueError:
            return ComparisonResult(ComparisonOutcome.INCONCLUSIVE, None, None, "DATE_PARSE_ERROR")
