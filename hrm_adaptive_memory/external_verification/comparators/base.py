"""Relation-bound parsers and comparison contracts for V2B verification."""
from __future__ import annotations

from dataclasses import dataclass
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
        if self.value_type is ValueType.QUANTITY and not self.canonical_unit:
            raise ValueError("quantity schemas require a canonical unit")
        if self.comparator is ComparisonKind.TOLERANCE and not (
                self.absolute_tolerance or self.relative_tolerance):
            raise ValueError("tolerance comparator requires a tolerance")
        if self.value_type is ValueType.ENUM and not self.allowed_values:
            raise ValueError("enum schemas require allowed values")


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
