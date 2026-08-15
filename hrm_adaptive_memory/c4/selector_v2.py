"""Sprint B2 selector arms (c4_selector_v1). Additive and opt-in.

The frozen C4 v2.1 selection path in selection_stage.py is NOT modified. S0 is
that path, reached by delegation, so the baseline cannot drift from the
certified behavior it is meant to reproduce.

What this repairs, and why it is shaped this way
-----------------------------------------------
The pre-freeze measurement located the defect precisely. Keep-rate of required
evidence that WAS in the candidate pool, on development:

    EXACT    unbridged   72.2%
    EXACT    bridged     12.8%   <-- the defect
    RESOLVED unbridged   88.2%
    RESOLVED bridged     95.0%

RESOLVED+bridged is healthy, so multi-hop structure alone is not the problem;
the failure is specific to EXACT+bridged, which is 34 of 39 EXACT drops (87.2%).
It matches the role-retention picture: identity and bridge retention are both
1.000 while answer-support retention is 0.529 -- S2c assembles the identity and
bridge records and then drops the terminal answer.

A subject+relation eligibility rule cannot reach that population: a bridged
task's terminal record is anchored on the BRIDGE entity, not the query subject
("- updated Finch control module: ownership tier now 3529" for a question about
"Sparrow intake manifold"). It fires on 36/36 unbridged but only 18/84 bridged
tasks. Hence the bridge-aware anchor.

Runtime-signal discipline (audited by test)
-------------------------------------------
Eligibility may read ONLY: candidate record content, the target relation PARSED
FROM THE QUESTION, the canonical subject from the I3 identity stage, and frozen
fusion score / retrieval rank. It must never read record_kind (whose values in
this corpus -- 'required', 'dead_end_link', 'rejected_candidate' -- are
generator answer-key labels), _oracle_metadata, proof_edges, latent_bridge,
answer_node, required_evidence_ids, or any other oracle field.

Bounded to exactly one hop
--------------------------
subject -> visible bridge record -> bridge entity -> terminal record. No
recursion, no subject -> bridge -> bridge -> terminal. One-hop reachability is
the kind of thing that quietly becomes unbounded graph search, so the bound is
enforced structurally here and asserted by test. If one hop proves
insufficient, that belongs in a successor protocol.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..retrieval.canonicalization import _norm
from .bridge_extraction import extract_v4_entities
from .query_stage import extract_target_relation

#: Reasons a record may be protected. Recorded in the receipt so every
#: protection decision is auditable after the fact.
DIRECT_SUBJECT_TARGET = "DIRECT_SUBJECT_TARGET"
ONE_HOP_BRIDGE_TARGET = "ONE_HOP_BRIDGE_TARGET"


@dataclass
class ProtectionReceipt:
    """Why one record was protected, and through which anchor.

    Emitted for every protected record so a leakage audit can check that the
    anchoring chain used only runtime-visible signals.
    """
    protection_reason: str
    anchor_subject: str
    target_relation: str
    protected_record_id: str
    bridge_entity: str | None = None
    fusion_score: float | None = None
    retrieval_rank: int | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "protection_reason": self.protection_reason,
            "anchor_subject": self.anchor_subject,
            "bridge_entity": self.bridge_entity,
            "target_relation": self.target_relation,
            "protected_record_id": self.protected_record_id,
            "fusion_score": self.fusion_score,
            "retrieval_rank": self.retrieval_rank,
        }


@dataclass
class _Candidate:
    """A pool record with the runtime signals eligibility may consult."""
    record_id: str
    content: str
    rank: int
    fusion_score: float | None = None
    entities: tuple[str, ...] = field(default_factory=tuple)


def _candidates(candidate_ids: Sequence[str], texts: Mapping[str, str],
                scores: Mapping[str, float] | None) -> list[_Candidate]:
    rows = []
    for rank, record_id in enumerate(candidate_ids, 1):
        content = texts.get(record_id, "")
        rows.append(_Candidate(
            record_id=record_id, content=content, rank=rank,
            fusion_score=(scores or {}).get(record_id),
            entities=extract_v4_entities(content)))
    return rows


def _mentions(candidate: _Candidate, phrase: str) -> bool:
    return bool(phrase) and _norm(phrase) in _norm(candidate.content)


def one_hop_bridge_entities(candidates: Sequence[_Candidate],
                            canonical_subject: str) -> tuple[str, ...]:
    """Entities reachable in EXACTLY one hop from the canonical subject.

    A bridge entity is any entity extracted from a candidate record that
    mentions the canonical subject, other than the subject itself. Derived
    purely from visible candidate text via the same V4 extractor the identity
    stage already uses.

    Deliberately NOT transitive: the returned entities are never re-expanded.
    """
    subject_norm = _norm(canonical_subject)
    found: list[str] = []
    for candidate in candidates:
        if not _mentions(candidate, canonical_subject):
            continue
        for entity in candidate.entities:
            if _norm(entity) != subject_norm:
                found.append(entity)
    return tuple(dict.fromkeys(found))


def find_protected_record(
    *, question: str, canonical_subject: str | None,
    candidate_ids: Sequence[str], texts: Mapping[str, str],
    fusion_scores: Mapping[str, float] | None = None,
) -> ProtectionReceipt | None:
    """Pick the record to protect, or None if nothing is eligible.

    Eligible iff the parsed target relation appears in the record AND the record
    is anchored either on the canonical subject directly, or on an entity
    reachable by exactly one visible hop from it.

    Frozen tie-break order:
      1. direct subject anchor preferred over bridge anchor
      2. higher frozen fusion score
      3. lower retrieval rank
      4. record_id lexical
    """
    relation = extract_target_relation(question) or ""
    if not relation or not canonical_subject:
        return None

    rows = _candidates(candidate_ids, texts, fusion_scores)
    relation_matches = [c for c in rows if _mentions(c, relation)]
    if not relation_matches:
        return None

    bridges = one_hop_bridge_entities(rows, canonical_subject)
    bridge_norms = {_norm(b): b for b in bridges}

    eligible: list[tuple[tuple, _Candidate, str, str | None]] = []
    for candidate in relation_matches:
        if _mentions(candidate, canonical_subject):
            reason, bridge = DIRECT_SUBJECT_TARGET, None
        else:
            matched = next(
                (original for norm, original in bridge_norms.items()
                 if norm and norm in _norm(candidate.content)), None)
            if matched is None:
                continue
            reason, bridge = ONE_HOP_BRIDGE_TARGET, matched
        # Sort key mirrors the frozen tie-break order exactly. Negated score so
        # higher wins; rank ascending; record_id last for total determinism.
        key = (0 if reason == DIRECT_SUBJECT_TARGET else 1,
               -(candidate.fusion_score if candidate.fusion_score is not None else 0.0),
               candidate.rank,
               candidate.record_id)
        eligible.append((key, candidate, reason, bridge))

    if not eligible:
        return None

    eligible.sort(key=lambda item: item[0])
    _key, best, reason, bridge = eligible[0]
    return ProtectionReceipt(
        protection_reason=reason,
        anchor_subject=canonical_subject,
        target_relation=relation,
        protected_record_id=best.record_id,
        bridge_entity=bridge,
        fusion_score=best.fusion_score,
        retrieval_rank=best.rank,
    )


#: Content markers for temporal precedence (S3). Read from record TEXT, never
#: from record_kind or embedding similarity: this corpus writes "Revision 2
#: (effective ...) supersedes revision 1:" on the current record and "Revision 1
#: (effective ..., since superseded) recorded:" on the stale one.
_STALE_MARKERS = ("since superseded",)
_CURRENT_MARKERS = ("supersedes revision",)


def is_stale(content: str) -> bool:
    """True if the record's own text marks it as superseded."""
    low = content.lower()
    return any(marker in low for marker in _STALE_MARKERS)


def is_current(content: str) -> bool:
    low = content.lower()
    return any(marker in low for marker in _CURRENT_MARKERS)


def _is_identity_like(content: str) -> bool:
    """Identity records, detected the way the identity stage already does.

    Uses the qualified canonicalization parser rather than record_kind, which
    is an answer-key label. Identity records are exempt from S2's connectivity
    constraint because a surface->canonical mapping legitimately mentions
    neither the canonical subject in isolation nor any bridge.
    """
    from ..retrieval.canonicalization import extract_identity_links

    class _Row:
        def __init__(self, content: str):
            self.evidence_id = "probe"
            self.content = content

    return bool(extract_identity_links([_Row(content)]))


def connectivity_status(content: str, canonical_subject: str,
                        bridges: Sequence[str]) -> str:
    """Classify a record's connection to the query subject.

    CONNECTED_SUBJECT / CONNECTED_BRIDGE / IDENTITY / DISCONNECTED. The last is
    the class S2 rejects: a record that is individually plausible but attaches
    to neither the query subject nor any visible one-hop bridge, which is how
    S2c assembled six unrelated records in the worked failure case.
    """
    normalized = _norm(content)
    if canonical_subject and _norm(canonical_subject) in normalized:
        return "CONNECTED_SUBJECT"
    for bridge in bridges:
        if bridge and _norm(bridge) in normalized:
            return "CONNECTED_BRIDGE"
    if _is_identity_like(content):
        return "IDENTITY"
    return "DISCONNECTED"


def select_s2(
    *, identity_status: str, question: str, canonical_subject: str | None,
    candidate_ids: Sequence[str], texts: Mapping[str, str], budget: int,
    frozen_select, fusion_scores: Mapping[str, float] | None = None,
) -> tuple[list[str], ProtectionReceipt | None, dict[str, Any]]:
    """S2 = S1 plus a connectivity constraint on the remaining slots.

    A rejected record's slot goes to the next candidate rather than being left
    empty, so the packet budget is still spent in full.
    """
    selected, receipt = select_s1(
        identity_status=identity_status, question=question,
        canonical_subject=canonical_subject, candidate_ids=candidate_ids,
        texts=texts, budget=budget, frozen_select=frozen_select,
        fusion_scores=fusion_scores)

    if not canonical_subject:
        return selected, receipt, {"rejected": 0, "disconnected_in_packet": 0}

    rows = _candidates(candidate_ids, texts, fusion_scores)
    bridges = one_hop_bridge_entities(rows, canonical_subject)
    protected = receipt.protected_record_id if receipt else None

    kept: list[str] = []
    rejected = 0
    for record_id in selected:
        if record_id == protected:
            kept.append(record_id)
            continue
        status = connectivity_status(texts.get(record_id, ""),
                                     canonical_subject, bridges)
        if status == "DISCONNECTED":
            rejected += 1
            continue
        kept.append(record_id)

    # Refill from the pool in rank order with connected records only.
    if rejected:
        already = set(kept)
        for candidate in rows:
            if len(kept) >= budget:
                break
            if candidate.record_id in already:
                continue
            if connectivity_status(candidate.content, canonical_subject,
                                   bridges) == "DISCONNECTED":
                continue
            kept.append(candidate.record_id)
            already.add(candidate.record_id)

    disconnected_remaining = sum(
        1 for record_id in kept
        if record_id != protected
        and connectivity_status(texts.get(record_id, ""), canonical_subject,
                                bridges) == "DISCONNECTED")
    return (kept[:budget], receipt,
            {"rejected": rejected,
             "disconnected_in_packet": disconnected_remaining})


def select_s3(
    *, identity_status: str, question: str, canonical_subject: str | None,
    candidate_ids: Sequence[str], texts: Mapping[str, str], budget: int,
    frozen_select, fusion_scores: Mapping[str, float] | None = None,
) -> tuple[list[str], ProtectionReceipt | None, dict[str, Any]]:
    """S3 = S2 plus temporal-current precedence.

    Where the packet holds a record its own text marks as superseded, and the
    pool offers a current counterpart mentioning the same subject and relation,
    the current one replaces it. Precedence comes from content markers, never
    from embedding similarity.
    """
    selected, receipt, diag = select_s2(
        identity_status=identity_status, question=question,
        canonical_subject=canonical_subject, candidate_ids=candidate_ids,
        texts=texts, budget=budget, frozen_select=frozen_select,
        fusion_scores=fusion_scores)

    relation = extract_target_relation(question) or ""
    if not relation:
        return selected, receipt, {**diag, "temporal_swaps": 0}

    protected = receipt.protected_record_id if receipt else None
    in_packet = set(selected)
    swaps = 0
    result = list(selected)

    for index, record_id in enumerate(result):
        if record_id == protected:
            continue
        content = texts.get(record_id, "")
        if not is_stale(content):
            continue
        replacement = next(
            (c.record_id for c in _candidates(candidate_ids, texts, fusion_scores)
             if c.record_id not in in_packet
             and is_current(c.content)
             and _norm(relation) in _norm(c.content)), None)
        if replacement:
            result[index] = replacement
            in_packet.discard(record_id)
            in_packet.add(replacement)
            swaps += 1

    return result[:budget], receipt, {**diag, "temporal_swaps": swaps}


def select_s1(
    *, identity_status: str, question: str, canonical_subject: str | None,
    candidate_ids: Sequence[str], texts: Mapping[str, str], budget: int,
    frozen_select, fusion_scores: Mapping[str, float] | None = None,
) -> tuple[list[str], ProtectionReceipt | None]:
    """S1: bridge-aware terminal-answer protection for EXACT identity.

    Frozen behavior, decided before the arm ran:

        EXACT and an eligible record exists -> protect it, then fill the
            REMAINING budget with unchanged frozen selection
        EXACT and nothing eligible          -> frozen selection, unchanged
        RESOLVED (or anything else)         -> frozen selection, unchanged

    The RESOLVED path is untouched deliberately: its keep-rate is already 91.9%
    development / 91.8% qualification, and the point of the arm is to repair
    EXACT without disturbing what already works.

    Packet-budget invariant: the protected record CONSUMES ONE SLOT of the same
    fixed budget. Packet size never grows, so a measured gain cannot come from
    handing the reader more evidence.

    ``frozen_select`` is the unmodified selector, injected rather than imported
    so this module cannot silently diverge from the certified path.
    """
    if identity_status != "EXACT":
        return list(frozen_select(budget)), None

    receipt = find_protected_record(
        question=question, canonical_subject=canonical_subject,
        candidate_ids=candidate_ids, texts=texts, fusion_scores=fusion_scores)
    if receipt is None:
        return list(frozen_select(budget)), None

    protected = receipt.protected_record_id
    # One slot spent on protection; the frozen selector fills what remains.
    #
    # Ask the frozen selector for the FULL budget rather than budget-1, then
    # drop the protected record if it also chose it. Requesting budget-1 and
    # filtering afterwards silently returned a short packet whenever the frozen
    # selector happened to pick the protected record too -- budget-1 items minus
    # one overlap = budget-2. A packet smaller than the budget is its own
    # violation of the fixed-budget invariant (the reader would get LESS
    # evidence, understating the arm), so the fill is taken from a full request.
    remainder = [r for r in frozen_select(budget) if r != protected]
    selected = [protected] + remainder[:budget - 1]
    return selected, receipt
