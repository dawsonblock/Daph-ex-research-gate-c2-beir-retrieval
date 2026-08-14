from __future__ import annotations

from decimal import Decimal
import re
from typing import Any

from .base import (ComparisonKind, ComparisonOutcome, ComparisonResult,
                   RelationSchema, within_tolerance)
from ._helpers import decimal_value


_UNITS: dict[str, tuple[str, Decimal]] = {
    "kg": ("mass", Decimal("1")), "g": ("mass", Decimal("0.001")),
    "mg": ("mass", Decimal("0.000001")),
    "m": ("length", Decimal("1")), "cm": ("length", Decimal("0.01")),
    "mm": ("length", Decimal("0.001")), "km": ("length", Decimal("1000")),
}
_QUANTITY = re.compile(r"^\s*([^\s]+)\s+([A-Za-z]+)\s*$")


def _quantity(value: Any, canonical_unit: str) -> Decimal:
    if isinstance(value, dict):
        magnitude, unit = value.get("value"), value.get("unit")
    elif isinstance(value, str):
        match = _QUANTITY.match(value)
        if not match:
            raise ValueError("quantity must contain a magnitude and unit")
        magnitude, unit = match.groups()
    else:
        raise ValueError("quantity must be an object or string")
    if not isinstance(unit, str) or unit not in _UNITS or canonical_unit not in _UNITS:
        raise ValueError("unsupported quantity unit")
    dimension, factor = _UNITS[unit]
    target_dimension, target_factor = _UNITS[canonical_unit]
    if dimension != target_dimension:
        raise ValueError("quantity dimension mismatch")
    return decimal_value(magnitude) * factor / target_factor


class QuantityComparator:
    def compare(self, schema: RelationSchema, claim_value: Any,
                evidence_value: Any) -> ComparisonResult:
        try:
            claim = _quantity(claim_value, schema.canonical_unit or "")
            evidence = _quantity(evidence_value, schema.canonical_unit or "")
            if schema.comparator is ComparisonKind.TOLERANCE:
                matched = within_tolerance(claim, evidence, schema)
            else:
                matched = claim == evidence
            return ComparisonResult(ComparisonOutcome.MATCH if matched else ComparisonOutcome.MISMATCH,
                                    f"{claim} {schema.canonical_unit}", f"{evidence} {schema.canonical_unit}",
                                    "QUANTITY_MATCH" if matched else "QUANTITY_MISMATCH")
        except ValueError:
            return ComparisonResult(ComparisonOutcome.INCONCLUSIVE, None, None, "QUANTITY_PARSE_ERROR")
