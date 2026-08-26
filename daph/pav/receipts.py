"""PAV provenance receipts."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PAVReceipt:
    """Provenance receipt for a PAV scoring call."""
    receipt_id: str
    checkpoint_id: str
    scorer_type: str
    config_sha: str
    model_sha: str
    actions: tuple[str, ...]
    predictions: tuple[dict, ...]
    selected: tuple[str, ...]
    abstained: bool
    timing_ms: float

    def as_dict(self) -> dict:
        return {
            "receipt_id": self.receipt_id,
            "checkpoint_id": self.checkpoint_id,
            "scorer_type": self.scorer_type,
            "config_sha": self.config_sha,
            "model_sha": self.model_sha,
            "actions": list(self.actions),
            "predictions": list(self.predictions),
            "selected": list(self.selected),
            "abstained": self.abstained,
            "timing_ms": self.timing_ms,
        }


def create_pav_receipt(
    checkpoint_id: str,
    scorer_type: str,
    config_sha: str,
    model_sha: str,
    actions: tuple[str, ...],
    predictions: tuple[dict, ...],
    selected: tuple[str, ...],
    abstained: bool,
    timing_ms: float,
) -> PAVReceipt:
    content = json.dumps({
        "checkpoint_id": checkpoint_id,
        "scorer_type": scorer_type,
        "config_sha": config_sha,
        "model_sha": model_sha,
        "actions": list(actions),
        "predictions": list(predictions),
        "selected": list(selected),
        "abstained": abstained,
        "timing_ms": timing_ms,
    }, sort_keys=True)
    receipt_id = hashlib.sha256(content.encode()).hexdigest()
    return PAVReceipt(
        receipt_id=receipt_id,
        checkpoint_id=checkpoint_id,
        scorer_type=scorer_type,
        config_sha=config_sha,
        model_sha=model_sha,
        actions=actions,
        predictions=predictions,
        selected=selected,
        abstained=abstained,
        timing_ms=timing_ms,
    )
