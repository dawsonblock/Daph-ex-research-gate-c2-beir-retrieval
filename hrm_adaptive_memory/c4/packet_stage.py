"""C4 packet stage — compose the bounded evidence packet.

Uses the existing compose_evidence_prompt. The packet budget is frozen at
C4_PRIMARY_PACKET_BUDGET = 6 across all arms. No budget sweeps during C4.
"""
from __future__ import annotations

import hashlib
from typing import Mapping, Sequence

from ..evidence.packing import compose_evidence_prompt
from .contracts import C4Arm, SelectionResult, PacketResult


def run_packet_stage(arm: C4Arm, question: str,
                     selection: SelectionResult,
                     texts: Mapping[str, str]) -> tuple[str, PacketResult]:
    """Compose the evidence packet from selected IDs.

    Returns (prompt, PacketResult).
    """
    contents = [texts.get(eid, "") for eid in selection.selected_ids if eid in texts]
    prompt = compose_evidence_prompt(question, contents)

    # Token count: simple whitespace split for now (HRM tokenizer may differ)
    token_count = sum(len(c.split()) for c in contents)
    packet_hash = hashlib.sha256(
        "|".join(selection.selected_ids).encode()).hexdigest()

    return prompt, PacketResult(
        packet_ids=selection.selected_ids,
        packet_contents=tuple(contents),
        packet_token_count=token_count,
        packet_hash=packet_hash,
        packet_budget=arm.packet_budget,
    )
