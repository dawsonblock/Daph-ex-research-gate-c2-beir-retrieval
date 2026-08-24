"""Provenance receipts for interventions.

Every intervention must produce a receipt with:
  - checkpoint_id
  - action
  - intervention_type
  - result
  - state hash before and after
  - timestamp
  - backend identity
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class InterventionReceipt:
    """Provenance receipt for a single intervention.

    Attributes:
        receipt_id: SHA256 hash of the receipt content
        checkpoint_id: The checkpoint intervened from
        action: The forced action
        intervention_type: CAUSAL_DETERMINISTIC or FORCED_ACTION_ROLLOUT
        result: The ForcedActionResult as a dict
        state_sha_before: State hash before the intervention
        state_sha_after: State hash after the intervention (if non-terminal)
        backend_identity_sha256: Backend identity for provenance
        timestamp: ISO timestamp
    """
    receipt_id: str
    checkpoint_id: str
    action: str
    intervention_type: str
    result: dict
    state_sha_before: str
    state_sha_after: str | None
    backend_identity_sha256: str
    timestamp: str

    def as_dict(self) -> dict:
        return {
            "receipt_id": self.receipt_id,
            "checkpoint_id": self.checkpoint_id,
            "action": self.action,
            "intervention_type": self.intervention_type,
            "result": self.result,
            "state_sha_before": self.state_sha_before,
            "state_sha_after": self.state_sha_after,
            "backend_identity_sha256": self.backend_identity_sha256,
            "timestamp": self.timestamp,
        }


def create_receipt(
    checkpoint_id: str,
    action: str,
    intervention_type: str,
    result: dict,
    state_sha_before: str,
    state_sha_after: str | None,
    backend_identity_sha256: str,
    timestamp: str,
) -> InterventionReceipt:
    """Create a provenance receipt for an intervention."""
    content = json.dumps({
        "checkpoint_id": checkpoint_id,
        "action": action,
        "intervention_type": intervention_type,
        "result": result,
        "state_sha_before": state_sha_before,
        "state_sha_after": state_sha_after,
        "backend_identity_sha256": backend_identity_sha256,
        "timestamp": timestamp,
    }, sort_keys=True)
    receipt_id = hashlib.sha256(content.encode()).hexdigest()

    return InterventionReceipt(
        receipt_id=receipt_id,
        checkpoint_id=checkpoint_id,
        action=action,
        intervention_type=intervention_type,
        result=result,
        state_sha_before=state_sha_before,
        state_sha_after=state_sha_after,
        backend_identity_sha256=backend_identity_sha256,
        timestamp=timestamp,
    )


def save_receipts(receipts: list[InterventionReceipt], path: str) -> None:
    """Save receipts to JSONL."""
    from pathlib import Path
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        for r in receipts:
            f.write(json.dumps(r.as_dict(), sort_keys=True) + "\n")
