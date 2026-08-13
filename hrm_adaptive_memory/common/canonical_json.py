"""Strict RFC-8259 JSON for hash-bearing DAPH artifacts.

Python's standard encoder accepts NaN and infinities by default.  Those tokens
are not JSON and make frozen evidence parser-dependent, so this module is the
only writer used by the I3.3.2 benchmark and oracle-cache pipeline.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-RFC JSON numeric constant: {value}")


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def pretty_text(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False,
    ) + "\n"


def write_json(path: str | Path, value: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(pretty_text(value), encoding="utf-8", newline="\n")


def loads_strict(raw: str | bytes | bytearray) -> Any:
    return json.loads(raw, parse_constant=_reject_constant)


def load_strict(path: str | Path) -> Any:
    return loads_strict(Path(path).read_bytes())


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()
