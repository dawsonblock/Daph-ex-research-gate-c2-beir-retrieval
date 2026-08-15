"""S4: path-coherent packet composition. Frozen deterministic contract.

The division of responsibility this implements -- and the reason S2 is NOT
replaced:

    S2                 scores and ranks individual records (still useful signal)
    G2 graph           supplies path structure
    packet composer    enforces structural consistency across the six records

S3 established that S2's failure is packet INCOHERENCE, not record blindness:
of required records present in the working set but absent from the packet, ~64%
were lost to F3_WRONG_PATH_PREFERRED -- S2 assembled packets from fragments of
incompatible paths. Record utility plus structural consistency yields packet
utility; independent per-record selection cannot express the second term.

The contract, frozen before any S4 arm was scored
--------------------------------------------------
    1. take the complete paths G2 already produced (no new enumeration)
    2. rank them by the EXISTING frozen path order only
    3. pick the top-ranked path whose records fit the packet budget
    4. admit its supporting records ATOMICALLY -- all or nothing
    5. deduplicate
    6. fill remaining slots from the current S2 ordering, unchanged
    7. never exceed the packet budget

"Atomically" is the load-bearing word. A path either survives coherently or is
not admitted as a path at all; the composer never cherry-picks two records from
path A and two from path B. Cross-path mixing can still occur, but only in the
FILL stage, after one path has been preserved whole -- which is exactly the
behaviour S3 showed was missing.

Tie-breaking uses existing deterministic signals only (path tier, hop count,
retrieval tie-break, then path_id lexically). No new tuned score, no learned
parameter, no oracle field.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

#: Frozen path ordering key. Mirrors g2_paths.rank_and_select_paths' existing
#: order (tier, hop_count, -retrieval_score, record_ids) and appends path_id as
#: a final lexical tie-break so the choice is total and reproducible.
def path_order_key(path) -> tuple:
    return (path.tier, path.hop_count, -path.retrieval_score,
            path.record_ids, path.path_id)


@dataclass
class PacketComposition:
    """The packet plus the coherence evidence needed to judge it."""
    packet: list[str] = field(default_factory=list)
    anchor_path_id: str | None = None
    anchor_path_records: tuple[str, ...] = ()
    path_admitted_atomically: bool = False
    fill_records: list[str] = field(default_factory=list)
    complete_paths_available: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "packet": list(self.packet), "anchor_path_id": self.anchor_path_id,
            "anchor_path_records": list(self.anchor_path_records),
            "path_admitted_atomically": self.path_admitted_atomically,
            "fill_records": list(self.fill_records),
            "complete_paths_available": self.complete_paths_available,
        }


def compose_path_coherent_packet(
    *, complete_paths: Sequence[Any], s2_ordering: Sequence[str],
    working_set: Sequence[str], packet_budget: int,
) -> PacketComposition:
    """Build a packet around ONE coherent complete path, then fill.

    ``s2_ordering`` is the unchanged S2 record ranking over the working set --
    the composer consumes it rather than replacing it. ``complete_paths`` are
    the PathCandidates G2 already produced with ``complete=True``.
    """
    available = set(working_set)
    ordered_paths = sorted(complete_paths, key=path_order_key)
    result = PacketComposition(complete_paths_available=len(ordered_paths))

    # steps 2-4: first path in frozen order whose records fit, admitted whole
    for path in ordered_paths:
        records = tuple(dict.fromkeys(
            r for r in path.record_ids if r in available))
        if not records or len(records) > packet_budget:
            continue
        result.anchor_path_id = path.path_id
        result.anchor_path_records = records
        result.path_admitted_atomically = True
        result.packet = list(records)
        break

    # steps 5-7: fill from the unchanged S2 ordering, dedup, respect the budget
    chosen = set(result.packet)
    for record_id in s2_ordering:
        if len(result.packet) >= packet_budget:
            break
        if record_id in chosen or record_id not in available:
            continue
        result.packet.append(record_id)
        result.fill_records.append(record_id)
        chosen.add(record_id)
    return result


#: --- frozen packet-coherence metrics ---------------------------------------
NOT_COMPUTABLE = "NOT_COMPUTABLE"


def complete_path_packet(packet: Sequence[str], complete_paths: Sequence[Any]) -> int:
    """Strict binary: does the packet contain EVERY record of at least one
    complete path? This is the metric that distinguishes a usable evidence set
    from a pile of individually-plausible fragments."""
    chosen = set(packet)
    for path in complete_paths:
        records = set(path.record_ids)
        if records and records <= chosen:
            return 1
    return 0


def packet_coherence_ratio(packet: Sequence[str],
                           complete_paths: Sequence[Any]) -> float | str:
    """PCR = (records belonging to the single best-represented complete path)
    / (packet records belonging to ANY complete path).

    Returns NOT_COMPUTABLE, never 0.0, when no packet record belongs to any
    complete path -- an empty denominator is not a coherence score of zero.
    """
    chosen = set(packet)
    path_relevant = set()
    best = 0
    for path in complete_paths:
        overlap = chosen & set(path.record_ids)
        path_relevant |= overlap
        best = max(best, len(overlap))
    if not path_relevant:
        return NOT_COMPUTABLE
    return round(best / len(path_relevant), 6)


def cross_path_fragmentation(packet: Sequence[str],
                             complete_paths: Sequence[Any]) -> float | str:
    pcr = packet_coherence_ratio(packet, complete_paths)
    return pcr if pcr == NOT_COMPUTABLE else round(1.0 - pcr, 6)


def complete_paths_represented(packet: Sequence[str],
                               complete_paths: Sequence[Any]) -> int:
    chosen = set(packet)
    return sum(1 for p in complete_paths if chosen & set(p.record_ids))


#: --- HRM qualification hash-chain links (configs/gate_hrm_qualification_v1.json) --
#: candidate_pool_hash / membership_hash / order_hash / prompt_hash already
#: exist in packet_ordering.py and are reused verbatim, not duplicated here.

def graph_hash(graph) -> str:
    """Sha256 over every edge's (type, source, target, source_record_id),
    sorted for order-independence. Localizes graph-construction drift
    independent of which paths were later selected from it."""
    import hashlib
    rows = sorted(
        f"{e.edge_type}|{e.source}|{e.target}|{e.source_record_id}"
        for e in graph.edges)
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()[:16]


def path_set_hash(paths: Sequence[Any]) -> str:
    """Sha256 over the sorted path_id set of the (typically complete) paths
    considered for composition -- localizes path-enumeration drift separately
    from graph-construction drift."""
    import hashlib
    ids = sorted(p.path_id for p in paths)
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()[:16]


def composed_packet_hash(selected_ids: Sequence[str]) -> str:
    """Sha256 over the H-arm's selected ids, ORDER-SENSITIVE, taken BEFORE the
    C4_4 deterministic ordering policy is applied in run_packet_stage. Lets a
    receipt distinguish 'the composer chose different records/order' from 'the
    downstream ordering policy reordered the same records' -- the latter would
    show up as a differing order_hash with an UNCHANGED composed_packet_hash
    membership, which is the intended, harmless case."""
    import hashlib
    return hashlib.sha256("|".join(selected_ids).encode("utf-8")).hexdigest()[:16]


def generation_hash(output_text: str) -> str:
    """Sha256 of the HRM output text, for reproducibility auditing."""
    import hashlib
    return hashlib.sha256(output_text.encode("utf-8")).hexdigest()[:16]
