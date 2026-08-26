"""Search provenance receipts."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SearchReceipt:
    """Provenance receipt for a search invocation."""
    receipt_id: str
    root_state_sha: str
    trigger_reasons: tuple[str, ...]
    config_sha: str
    nodes_expanded: int
    model_calls: int
    wall_ms: float
    winner: str | None
    abstained: bool
    fallback_reason: str | None
    branches: tuple[dict, ...]

    def as_dict(self) -> dict:
        return {
            "receipt_id": self.receipt_id,
            "root_state_sha": self.root_state_sha,
            "trigger_reasons": list(self.trigger_reasons),
            "config_sha": self.config_sha,
            "nodes_expanded": self.nodes_expanded,
            "model_calls": self.model_calls,
            "wall_ms": round(self.wall_ms, 2),
            "winner": self.winner,
            "abstained": self.abstained,
            "fallback_reason": self.fallback_reason,
            "branches": list(self.branches),
        }


def create_search_receipt(
    root_state_sha: str,
    trigger_reasons: tuple[str, ...],
    config_sha: str,
    nodes_expanded: int,
    model_calls: int,
    wall_ms: float,
    winner: str | None,
    abstained: bool,
    fallback_reason: str | None,
    branches: tuple[dict, ...],
) -> SearchReceipt:
    content = json.dumps({
        "root_state_sha": root_state_sha,
        "trigger_reasons": list(trigger_reasons),
        "config_sha": config_sha,
        "nodes_expanded": nodes_expanded,
        "model_calls": model_calls,
        "wall_ms": wall_ms,
        "winner": winner,
        "abstained": abstained,
        "fallback_reason": fallback_reason,
        "branches": list(branches),
    }, sort_keys=True)
    receipt_id = hashlib.sha256(content.encode()).hexdigest()
    return SearchReceipt(
        receipt_id=receipt_id,
        root_state_sha=root_state_sha,
        trigger_reasons=trigger_reasons,
        config_sha=config_sha,
        nodes_expanded=nodes_expanded,
        model_calls=model_calls,
        wall_ms=wall_ms,
        winner=winner,
        abstained=abstained,
        fallback_reason=fallback_reason,
        branches=branches,
    )
