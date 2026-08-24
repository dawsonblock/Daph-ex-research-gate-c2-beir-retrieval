"""DAPH I3.4 — Packet builder for P0/P1/P2/PS arms.

P0: Base packet (identical to R2 C0, no phase or values)
P1: Base packet + epistemic_phase
P2: Base packet + epistemic_phase + action_value_estimates (correct B1 values)
PS: Base packet + epistemic_phase + action_value_estimates (shuffled B1 values)

PS is a causal control: it preserves the ranked-recommendation structure
(identical field count, identical numeric distribution, identical packet
size) but permutes the action→value association within each phase.
If P2 > PS, the correct values matter, not just the presence of a ranking.

The PS permutation is deterministically seeded via SHA-256(phase|seed)
so it is stable across process boundaries (Python's built-in hash() is
process-randomized and must NOT be used). A pre-computed frozen mapping
can also be loaded to guarantee identical shuffles across all runs.

All prompts are identical except the intervention fields. No explanatory
prose such as "VERIFY is probably the right thing to do." That would
confound representation with instruction.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

from daph.phase.ontology import Phase, ALL_PHASES
from daph.phase.classifier import classify_phase
from daph.value.empirical import PhaseActionTable


def stable_shuffle_seed(phase: str, shuffle_seed: int) -> int:
    """Compute a deterministic 64-bit seed from (phase, shuffle_seed).

    Uses SHA-256 so the seed is stable across process boundaries and
    Python hash-randomization settings. Python's built-in hash() is
    process-randomized (PYTHONHASHSEED) and must NOT be used for
    any computation that needs to be reproducible across runs.
    """
    payload = f"{phase}|{shuffle_seed}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big")


def _shuffle_values(
    raw_values: dict[str, float],
    phase: str,
    shuffle_seed: int,
) -> dict[str, float]:
    """Permute action→value association within a phase.

    Preserves the exact set of numeric values but randomly reassigns them
    to different actions. The permutation is deterministic per (phase, seed)
    via SHA-256 seeding, so the same phase always gets the same shuffle
    regardless of process or PYTHONHASHSEED.
    """
    actions = list(raw_values.keys())
    values = list(raw_values.values())
    seed = stable_shuffle_seed(phase, shuffle_seed)
    rng = random.Random(seed)
    rng.shuffle(values)
    return dict(zip(actions, values))


def generate_frozen_ps_mapping(
    value_table: PhaseActionTable,
    shuffle_seed: int = 42,
    actions: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Pre-compute the PS shuffle mapping for all phases.

    Returns a dict keyed by phase name, where each value is a dict
    mapping action→shuffled raw value. This can be serialized to JSON
    and loaded in subsequent runs to guarantee the exact same PS
    permutation across all processes and sessions.

    Args:
        value_table: The frozen B1 PhaseActionTable
        shuffle_seed: The shuffle seed (must match the experiment's shuffle_seed)
        actions: Full action vocabulary (defaults to standard R2 actions)

    Returns:
        {phase: {action: shuffled_raw_value}}
    """
    if actions is None:
        actions = ["ANSWER", "VERIFY", "DEFER", "SEARCH_MORE", "RETRIEVE"]

    mapping: dict[str, dict[str, float]] = {}
    for phase in ALL_PHASES:
        # Get raw values for all actions in this phase
        raw_values = {}
        for action in actions:
            raw_values[action] = value_table.predict(phase.value, action, {})

        # Shuffle deterministically
        shuffled = _shuffle_values(raw_values, phase.value, shuffle_seed)
        mapping[phase.value] = {a: round(v, 6) for a, v in shuffled.items()}

    return mapping


def save_frozen_ps_mapping(
    mapping: dict[str, dict[str, float]],
    path: Path,
) -> str:
    """Save the frozen PS mapping to JSON and return its SHA-256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(mapping, f, indent=2, sort_keys=True)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frozen_ps_mapping(path: Path) -> dict[str, dict[str, float]]:
    """Load a frozen PS mapping from JSON."""
    with open(path) as f:
        return json.load(f)


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
    shuffle_seed: int = 0,
    ps_frozen_mapping: dict[str, dict[str, float]] | None = None,
) -> dict:
    """Build a model packet for the given arm.

    Args:
        base_packet: The R2-style MDSG packet (from build_mdsg_state_with_affordances_packet)
        arm: "P0", "P1", "P2", or "PS"
        phase: The classified EpistemicPhase
        value_table: The B1 phase×action table (required for P2/PS)
        legal_actions: Legal actions at this state
        features: Feature dict for value lookup
        normalize: Whether to normalize values to [0, 1]
        epsilon: Small constant for normalization
        shuffle_seed: Seed for PS arm shuffling (deterministic per phase via SHA-256)
        ps_frozen_mapping: Pre-computed frozen PS mapping (phase→action→value).
            If provided for PS arm, uses the frozen values directly instead of
            shuffling dynamically. This guarantees the same permutation across
            all processes and sessions.

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

    if arm in ("P2", "PS"):
        # Add phase + action value estimates
        packet["epistemic_phase"] = phase.value

        if value_table is not None and legal_actions:
            # Get raw values for each legal action
            raw_values = {}
            for action in legal_actions:
                raw_values[action] = value_table.predict(
                    phase.value, action, features or {}
                )

            # PS: shuffle the action→value association
            if arm == "PS":
                if ps_frozen_mapping is not None:
                    # Use pre-computed frozen mapping (deterministic across processes)
                    phase_mapping = ps_frozen_mapping.get(phase.value, {})
                    raw_values = {
                        a: phase_mapping.get(a, raw_values[a])
                        for a in legal_actions
                    }
                else:
                    # Fallback: deterministic SHA-256 seeded shuffle
                    raw_values = _shuffle_values(raw_values, phase.value, shuffle_seed)

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
