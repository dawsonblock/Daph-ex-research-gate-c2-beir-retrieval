"""Small frozen typed-comparison vocabulary for relation-bound verification."""
from .base import (ComparatorRegistry, ComparisonKind, ComparisonOutcome,
                   ComparisonResult, RelationSchema, ValueType)
from .canonical_string import CanonicalStringComparator
from .date import DateComparator
from .datetime import DateTimeComparator
from .decimal import DecimalComparator
from .enum import EnumComparator
from .identifier import IdentifierComparator
from .integer import IntegerComparator
from .quantity import QuantityComparator
from .set_membership import SetMembershipComparator


def default_comparator_registry() -> ComparatorRegistry:
    return ComparatorRegistry({
        ValueType.INTEGER: IntegerComparator(),
        ValueType.DECIMAL: DecimalComparator(),
        ValueType.QUANTITY: QuantityComparator(),
        ValueType.DATE: DateComparator(),
        ValueType.DATETIME: DateTimeComparator(),
        ValueType.ENUM: EnumComparator(),
        ValueType.IDENTIFIER: IdentifierComparator(),
        ValueType.CANONICAL_STRING: CanonicalStringComparator(),
        ValueType.SET_MEMBERSHIP: SetMembershipComparator(),
    })


__all__ = [
    "ComparatorRegistry", "ComparisonKind", "ComparisonOutcome", "ComparisonResult",
    "RelationSchema", "ValueType", "default_comparator_registry",
]
