"""Receipts for semantic relation extraction.

Each extraction call produces a receipt for provenance and
replayability. Receipts never store API secrets.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RelationExtractionReceipt:
    """Receipt for one relation extraction call.

    Records what was extracted, from what text, by what extractor.
    Never stores API keys or oracle-side data.
    """
    receipt_id: str
    task_id: str
    evidence_id: str
    hypothesis_id: str
    relation: str
    reason_code: str
    evidence_sha256: str
    hypothesis_sha256: str
    extractor_identity_sha256: str
    confidence: float | None
    latency_ms: int | None
    timestamp_utc: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "task_id": self.task_id,
            "evidence_id": self.evidence_id,
            "hypothesis_id": self.hypothesis_id,
            "relation": self.relation,
            "reason_code": self.reason_code,
            "evidence_sha256": self.evidence_sha256,
            "hypothesis_sha256": self.hypothesis_sha256,
            "extractor_identity_sha256": self.extractor_identity_sha256,
            "confidence": self.confidence,
            "latency_ms": self.latency_ms,
            "timestamp_utc": self.timestamp_utc,
        }
