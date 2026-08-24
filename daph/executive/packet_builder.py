"""DAPH I3.4 — Packet builder for P0/P1/P2 arms.

P0: Base packet (identical to R2 C0, no phase or values)
P1: Base packet + epistemic_phase
P2: Base packet + epistemic_phase + action_value_estimates

All prompts are identical except the intervention fields. No explanatory
prose such as "VERIFY is probably the right thing to do." That would
confound representation with instruction.
"""

from __future__ import annotations

from typing import Any

from daph.phase.ontology import Phase
from daph.phase.classifier import classify_phase
from daph.value.empirical import PhaseActionTable


def build_packet(
    base_packet: dict,
    *,
    arm: str,
    phase: Phase,
    value_table: PhaseActionTable | None = None,
    legal_actions: list[str] | None = None,
    features: dict | None = None,
    normalize: bool = True,
    epsilon: float = 1e-8,
) -> dict:
    """Build a model packet for the given arm.

    Args:
        base_packet: The R2-style MDSG packet (from build_mdsg_state_with_affordances_packet)
        arm: "P0", "P1", or "P2"
        phase: The classified EpistemicPhase
        value_table: The B1 phase×action table (required for P2)
        legal_actions: Legal actions at this state
        features: Feature dict for value lookup
        normalize: Whether to normalize values to [0, 1]
        epsilon: Small constant for normalization

    Returns:
        Modified packet dict
    """
    packet = dict(base_packet)

    if arm == "P0":
        # No additions — identical to baseline
        return packet

    if arm == "P1":
        # Add phase only
        packet["epistemic_phase"] = phase.value
        return packet

    if arm == "P2":
        # Add phase + action value estimates
        packet["epistemic_phase"] = phase.value

        if value_table is not None and legal_actions:
            # Get raw values for each legal action
            raw_values = {}
            for action in legal_actions:
                raw_values[action] = value_table.predict(
                    phase.value, action, features or {}
                )

            if normalize:
                # Monotonic normalization to [0, 1]
                min_v = min(raw_values.values()) if raw_values else 0.0
                max_v = max(raw_values.values()) if raw_values else 0.0
                range_v = max_v - min_v + epsilon
                normalized = {
                    a: round((v - min_v) / range_v, 4)
                    for a, v in raw_values.items()
                }
            else:
                normalized = {a: round(v, 4) for a, v in raw_values.items()}

            # Build action value estimates (no explanatory prose)
            packet["action_value_estimates"] = {
                action: {
                    "normalized_value": normalized.get(action, 0.0),
                }
                for action in legal_actions
            }

            # Also include ranking (descending by normalized value)
            ranking = sorted(legal_actions, key=lambda a: -normalized.get(a, 0.0))
            packet["action_value_ranking"] = ranking

        return packet

    raise ValueError(f"Unknown arm: {arm}")


def get_phase_from_packet(packet: dict) -> Phase:
    """Extract the phase from a packet's decision state."""
    ds_summary = packet.get("decision_state_summary", {})
    decision_state = ds_summary.get("decision_state", "UNKNOWN")

    # Extract hypothesis info from the packet
    hypotheses = packet.get("hypotheses", [])
    n_live = sum(1 for h in hypotheses if h.get("status") in ("LIVE", "SUPPORTED", "UNRESOLVED"))
    n_eliminated = sum(1 for h in hypotheses if h.get("status") in ("ELIMINATED", "FALSIFIED"))
    n_total = n_live + n_eliminated

    # Check T2
    t2 = (n_live == 0 and n_total > 0)

    ep = classify_phase(
        decision_state=decision_state,
        n_live=n_live,
        n_eliminated=n_eliminated,
        n_total=n_total,
        t2=t2,
    )
    return ep.phase
