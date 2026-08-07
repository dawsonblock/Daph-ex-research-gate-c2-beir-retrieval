"""C4 packet stage — compose the bounded evidence packet.

Uses the existing compose_evidence_prompt. The packet budget is frozen at
C4_PRIMARY_PACKET_BUDGET = 6 across all arms. No budget sweeps during C4.

For arms with deterministic packet ordering (C4_4, C4_3o), the frozen
c4_packet_ordering_v1 policy is applied. For other arms (C4_0–C4_3, C4_4m),
pool order from the selector is used as-is.
"""
from __future__ import annotations

import hashlib
from typing import Mapping, Sequence

from ..evidence.packing import compose_evidence_prompt
from .contracts import C4Arm, SelectionResult, PacketResult
from .packet_ordering import order_packet

# Arms that use the deterministic packet ordering policy
_DETERMINISTIC_ORDER_ARMS = {"C4_4", "C4_3o"}


def run_packet_stage(arm: C4Arm, question: str,
                     selection: SelectionResult,
                     texts: Mapping[str, str]) -> tuple[str, PacketResult]:
    """Compose the evidence packet from selected IDs.

    Returns (prompt, PacketResult).
    """
    # Apply deterministic ordering for arms that use it
    if arm.arm_id in _DETERMINISTIC_ORDER_ARMS:
        ordered_ids = order_packet(selection.selected_ids)
    else:
        ordered_ids = list(selection.selected_ids)

    contents = [texts.get(eid, "") for eid in ordered_ids if eid in texts]
    prompt = compose_evidence_prompt(question, contents)

    # Token count: simple whitespace split for now (HRM tokenizer may differ)
    token_count = sum(len(c.split()) for c in contents)
    packet_hash = hashlib.sha256(
        "|".join(ordered_ids).encode()).hexdigest()

    return prompt, PacketResult(
        packet_ids=tuple(ordered_ids),
        packet_contents=tuple(contents),
        packet_token_count=token_count,
        packet_hash=packet_hash,
        packet_budget=arm.packet_budget,
    )
