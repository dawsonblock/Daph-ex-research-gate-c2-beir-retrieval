"""Relation-bound parsers and comparison contracts for V2B verification."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any, Protocol


class ValueType(str, Enum):
    INTEGER = "INTEGER"
    DECIMAL = "DECIMAL"
    QUANTITY = "QUANTITY"
    DATE = "DATE"
    DATETIME = "DATETIME"
    ENUM = "ENUM"
    IDENTIFIER = "IDENTIFIER"
    CANONICAL_STRING = "CANONICAL_STRING"
    SET_MEMBERSHIP = "SET_MEMBERSHIP"


class ComparisonKind(str, Enum):
    EXACT = "EXACT"
    TOLERANCE = "TOLERANCE"
    MEMBER_OF = "MEMBER_OF"


class ComparisonOutcome(str, Enum):
    MATCH = "MATCH"
    MISMATCH = "MISMATCH"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass(frozen=True)
class ComparisonResult:
    outcome: ComparisonOutcome
    claim_value: Any | None
    evidence_value: Any | None
    reason_code: str


@dataclass(frozen=True)
class RelationSchema:
    relation: str
    value_type: ValueType
    comparator: ComparisonKind = ComparisonKind.EXACT
    canonical_unit: str | None = None
    absolute_tolerance: str | None = None
    relative_tolerance: str | None = None
    allowed_values: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.relation:
            raise ValueError("relation is required")
        if not isinstance(self.value_type, ValueType) or not isinstance(self.comparator, ComparisonKind):
            raise ValueError("value_type and comparator must use the frozen comparison enums")
        allowed = {
            ValueType.INTEGER: {ComparisonKind.EXACT},
            ValueType.DECIMAL: {ComparisonKind.EXACT, ComparisonKind.TOLERANCE},
            ValueType.QUANTITY: {ComparisonKind.EXACT, ComparisonKind.TOLERANCE},
            ValueType.DATE: {ComparisonKind.EXACT},
            ValueType.DATETIME: {ComparisonKind.EXACT},
            ValueType.ENUM: {ComparisonKind.EXACT},
            ValueType.IDENTIFIER: {ComparisonKind.EXACT},
            ValueType.CANONICAL_STRING: {ComparisonKind.EXACT},
            ValueType.SET_MEMBERSHIP: {ComparisonKind.MEMBER_OF},
        }
        if self.comparator not in allowed[self.value_type]:
            raise ValueError(
                f"{self.value_type.value} does not support {self.comparator.value} comparison")
        if self.value_type is ValueType.QUANTITY and not self.canonical_unit:
            raise ValueError("quantity schemas require a canonical unit")
        if self.value_type is not ValueType.QUANTITY and self.canonical_unit is not None:
            raise ValueError("only quantity schemas may declare a canonical unit")
        if self.comparator is ComparisonKind.TOLERANCE and not (
                self.absolute_tolerance or self.relative_tolerance):
            raise ValueError("tolerance comparator requires a tolerance")
        if self.value_type is ValueType.ENUM and not self.allowed_values:
            raise ValueError("enum schemas require allowed values")
        if self.value_type is not ValueType.ENUM and self.allowed_values:
            raise ValueError("only enum schemas may declare allowed values")
        if self.comparator is not ComparisonKind.TOLERANCE and (
                self.absolute_tolerance is not None or self.relative_tolerance is not None):
            raise ValueError("only tolerance comparators may declare tolerances")
        for name, raw in (("absolute_tolerance", self.absolute_tolerance),
                          ("relative_tolerance", self.relative_tolerance)):
            if raw is None:
                continue
            try:
                value = Decimal(raw)
            except (InvalidOperation, ValueError) as error:
                raise ValueError(f"{name} must be a finite nonnegative decimal") from error
            if not value.is_finite() or value < 0:
                raise ValueError(f"{name} must be a finite nonnegative decimal")


def within_tolerance(claim: Decimal, evidence: Decimal, schema: RelationSchema) -> bool:
    """Use symmetric relative tolerance so comparison direction cannot change the result."""
    absolute = Decimal(schema.absolute_tolerance or "0")
    relative = Decimal(schema.relative_tolerance or "0")
    threshold = max(absolute, max(abs(claim), abs(evidence)) * relative)
    return abs(claim - evidence) <= threshold


class TypedComparator(Protocol):
    def compare(self, schema: RelationSchema, claim_value: Any,
                evidence_value: Any) -> ComparisonResult: ...


class ComparatorRegistry:
    def __init__(self, comparators: dict[ValueType, TypedComparator]):
        self._comparators = dict(comparators)

    def compare(self, schema: RelationSchema, claim_value: Any,
                evidence_value: Any) -> ComparisonResult:
        comparator = self._comparators.get(schema.value_type)
        if comparator is None:
            return ComparisonResult(ComparisonOutcome.INCONCLUSIVE, None, None,
                                    "UNSUPPORTED_VALUE_TYPE")
        return comparator.compare(schema, claim_value, evidence_value)
