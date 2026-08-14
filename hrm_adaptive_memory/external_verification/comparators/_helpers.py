from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any


def canonical_string(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("value must be a string")
    return " ".join(value.strip().casefold().split())


def decimal_value(value: Any) -> Decimal:
    if isinstance(value, bool):
        raise ValueError("boolean is not numeric")
    try:
        result = Decimal(str(value).strip())
    except (InvalidOperation, ValueError) as error:
        raise ValueError("value is not a decimal") from error
    if not result.is_finite():
        raise ValueError("decimal must be finite")
    return result
