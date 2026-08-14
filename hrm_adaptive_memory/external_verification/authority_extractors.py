"""Deterministic extractors whose source files are pinned by the authority registry."""
from __future__ import annotations

import json
from typing import Any, Mapping


def world_bank_country_v1(raw: bytes, content_type: str) -> Mapping[str, Any]:
    """Extract one country record from the World Bank country endpoint JSON."""
    if content_type != "application/json":
        raise ValueError("world bank extractor requires application/json")
    payload = json.loads(raw.decode("utf-8"))
    records = payload[1] if isinstance(payload, list) and len(payload) > 1 else payload
    if not isinstance(records, list) or len(records) != 1 or not isinstance(records[0], dict):
        raise ValueError("world bank country response must contain exactly one country record")
    record = records[0]
    entity, capital_city = record.get("name"), record.get("capitalCity")
    if not isinstance(entity, str) or not entity or not isinstance(capital_city, str) or not capital_city:
        raise ValueError("world bank country response lacks name or capitalCity")
    return {"entity": entity, "capital_city": capital_city}
