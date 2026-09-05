"""R13 checkpoint receipt: immutable frozen state with hashing."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
import hashlib
import json

from daph_x.operators.types import RuntimeState, Candidate, TrajectoryPoint


@dataclass(frozen=True)
class ReasoningCheckpoint:
    """Immutable checkpoint with receipts."""
    checkpoint_id: str
    runtime_state: RuntimeState
    dataset_id: str
    corpus_sha256: str
    selector_version: str
    feature_version: str

    def canonical_bytes(self) -> bytes:
        data = {
            "checkpoint_id": self.checkpoint_id,
            "runtime_state": json.loads(self.runtime_state.canonical_bytes()),
            "dataset_id": self.dataset_id,
            "corpus_sha256": self.corpus_sha256,
            "selector_version": self.selector_version,
            "feature_version": self.feature_version,
        }
        return json.dumps(data, sort_keys=True, default=str).encode("utf-8")

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()
