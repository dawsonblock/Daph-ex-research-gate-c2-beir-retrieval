from __future__ import annotations

import re

from .base import ComparisonOutcome, ComparisonResult, RelationSchema


_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$")


class IdentifierComparator:
    def compare(self, schema: RelationSchema, claim_value: object,
                evidence_value: object) -> ComparisonResult:
        claim, evidence = str(claim_value).strip(), str(evidence_value).strip()
        if not _IDENTIFIER.fullmatch(claim) or not _IDENTIFIER.fullmatch(evidence):
            return ComparisonResult(ComparisonOutcome.INCONCLUSIVE, None, None, "IDENTIFIER_PARSE_ERROR")
        matched = claim == evidence
        return ComparisonResult(ComparisonOutcome.MATCH if matched else ComparisonOutcome.MISMATCH,
                                claim, evidence, "IDENTIFIER_MATCH" if matched else "IDENTIFIER_MISMATCH")
