from __future__ import annotations

from datetime import datetime, timezone

from .base import ComparisonOutcome, ComparisonResult, RelationSchema


def _parse(value: object) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("datetime timezone is required")
    return parsed.astimezone(timezone.utc)


class DateTimeComparator:
    def compare(self, schema: RelationSchema, claim_value: object,
                evidence_value: object) -> ComparisonResult:
        try:
            claim, evidence = _parse(claim_value), _parse(evidence_value)
            matched = claim == evidence
            return ComparisonResult(ComparisonOutcome.MATCH if matched else ComparisonOutcome.MISMATCH,
                                    claim.isoformat(), evidence.isoformat(),
                                    "DATETIME_MATCH" if matched else "DATETIME_MISMATCH")
        except ValueError:
            return ComparisonResult(ComparisonOutcome.INCONCLUSIVE, None, None, "DATETIME_PARSE_ERROR")
