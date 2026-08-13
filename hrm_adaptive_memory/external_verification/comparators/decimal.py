from __future__ import annotations

from decimal import Decimal

from .base import ComparisonKind, ComparisonOutcome, ComparisonResult, RelationSchema
from ._helpers import decimal_value


class DecimalComparator:
    def compare(self, schema: RelationSchema, claim_value: object,
                evidence_value: object) -> ComparisonResult:
        try:
            claim, evidence = decimal_value(claim_value), decimal_value(evidence_value)
            if schema.comparator is ComparisonKind.TOLERANCE:
                absolute = decimal_value(schema.absolute_tolerance or "0")
                relative = decimal_value(schema.relative_tolerance or "0")
                matched = abs(claim - evidence) <= max(absolute, abs(claim) * relative)
            else:
                matched = claim == evidence
            return ComparisonResult(ComparisonOutcome.MATCH if matched else ComparisonOutcome.MISMATCH,
                                    str(claim), str(evidence),
                                    "DECIMAL_MATCH" if matched else "DECIMAL_MISMATCH")
        except ValueError:
            return ComparisonResult(ComparisonOutcome.INCONCLUSIVE, None, None, "DECIMAL_PARSE_ERROR")
