"""Strict finite-numeric validation for every reported metric.

The bug this generalizes: scripts/certify_c4_run.py reads a family's
``mean_delta`` out of a JSON report and compares it directly against a
regression threshold (``d["mean_delta"] < threshold``). JSON happily
round-trips ``NaN``/``Infinity`` (Python's json module accepts them by
default), and a ``bool`` passes any ``isinstance(x, (int, float))`` check
because ``bool`` is an ``int`` subclass. None of those values should ever
silently satisfy or fail a numeric gate -- they should hard-fail the
certification with a specific, diagnosable reason.

Type validation ("is this a number") is not the same guarantee as this
module's ("is this a NUMBER a human could act on"). Use these helpers for
every reported metric: Q, CES, family deltas, bootstrap bounds, role
availability, retention, latency, cost, and any future executive utility
estimate.
"""
from __future__ import annotations

import math
import re
from typing import Any


class MetricValidationError(ValueError):
    """A reported metric failed strict finite-numeric validation. Callers in
    fail-closed contexts (certifiers, gates) should treat this as a hard
    failure of the run, not a warning."""


def require_finite_number(value: Any, *, field: str) -> float:
    """The single generic check every numeric metric should pass through
    before being compared, thresholded, or reported. Rejects None, NaN,
    +/-Inf, bool, strings, and anything else that isn't a real float/int."""
    if value is None:
        raise MetricValidationError(f"{field}: value is missing (None)")
    if isinstance(value, bool):
        raise MetricValidationError(f"{field}: value is a bool ({value!r}), not a number")
    if not isinstance(value, (int, float)):
        raise MetricValidationError(f"{field}: value {value!r} is not numeric "
                                    f"(got {type(value).__name__})")
    as_float = float(value)
    if math.isnan(as_float):
        raise MetricValidationError(f"{field}: value is NaN")
    if math.isinf(as_float):
        raise MetricValidationError(f"{field}: value is {'+' if as_float > 0 else '-'}Inf")
    return as_float


def require_probability(value: Any, *, field: str) -> float:
    """A finite number in [0, 1] -- for rates, retention, availability,
    conditional probabilities. Values fractionally outside [0,1] due to
    floating-point round trip (e.g. 1.0000000002) are tolerated up to a tight
    epsilon; anything further out is a real defect, not noise."""
    x = require_finite_number(value, field=field)
    epsilon = 1e-9
    if x < -epsilon or x > 1 + epsilon:
        raise MetricValidationError(f"{field}: {x} is not a probability in [0, 1]")
    return min(1.0, max(0.0, x))


def require_nonneg_int(value: Any, *, field: str) -> int:
    """A finite non-negative integer -- for counts (paths_enumerated,
    working_set_size, records_examined, etc)."""
    if isinstance(value, bool):
        raise MetricValidationError(f"{field}: value is a bool ({value!r}), not a count")
    x = require_finite_number(value, field=field)
    if x != int(x):
        raise MetricValidationError(f"{field}: {x} is not an integer count")
    if x < 0:
        raise MetricValidationError(f"{field}: {x} is negative, counts cannot be negative")
    return int(x)


_HEX_HASH = re.compile(r"^[0-9a-f]+$")


def require_hash_format(value: Any, *, field: str, expected_length: int | None = None) -> str:
    """A lowercase hex digest string, optionally of an exact expected length
    (e.g. 16 for the truncated hashes used throughout this repo's provenance
    fields, 64 for a full SHA-256 hex digest)."""
    if not isinstance(value, str) or not value:
        raise MetricValidationError(f"{field}: {value!r} is not a non-empty hash string")
    if not _HEX_HASH.match(value):
        raise MetricValidationError(f"{field}: {value!r} is not lowercase hex")
    if expected_length is not None and len(value) != expected_length:
        raise MetricValidationError(
            f"{field}: hash length {len(value)} != expected {expected_length}")
    return value


#: Result of a rate/ratio computation where the denominator may legitimately
#: be zero. Distinguishing NOT_COMPUTABLE from a real 0.0 or 1.0 is the whole
#: point -- see require_nonempty_rate below.
NOT_COMPUTABLE = "NOT_COMPUTABLE"


def require_nonempty_rate(hits: Any, total: Any, *, field: str) -> float | str:
    """A hits/total rate that fails closed on an empty denominator instead of
    raising ZeroDivisionError OR silently returning a misleading 0.0.

    Returns NOT_COMPUTABLE (a string sentinel, never a float) when total==0,
    so a caller cannot accidentally compare NOT_COMPUTABLE against a numeric
    threshold and have it evaluate as if it were data. Callers that need a
    gate to FAIL (not skip) on an empty denominator should check for this
    sentinel explicitly and fail the gate, per configs' 'empty denominator'
    reporting rule -- this function only refuses to lie about the number, it
    does not decide whether an empty comparison should pass or fail a given
    protocol's gate."""
    total_n = require_nonneg_int(total, field=f"{field}.total")
    hits_n = require_nonneg_int(hits, field=f"{field}.hits")
    if hits_n > total_n:
        raise MetricValidationError(f"{field}: hits ({hits_n}) > total ({total_n})")
    if total_n == 0:
        return NOT_COMPUTABLE
    return round(hits_n / total_n, 6)
