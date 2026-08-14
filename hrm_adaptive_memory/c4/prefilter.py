"""B4 structural candidate compression. Additive and opt-in.

Division of labour, deliberately different from S2's
---------------------------------------------------
    prefilter : remove redundancy and obvious irrelevance, PRESERVE structural
                possibilities, stay conservative
    S2        : make the actual structural selection, resolve bridge/terminal
                competition, choose the final six

The prefilter must NOT decide "this is the correct bridge". It only says "this
record is structurally plausible enough to survive compression". That is why
structure is used as an ADMISSIBILITY CONSTRAINT and retrieval score as the
ranking key -- if the prefilter ranked by S2's one-hop connectivity heuristic it
would simply re-run the same under-specified rule at an earlier stage and
reproduce the B3 failure at smaller scale.

What B3 measured, and what this is for
--------------------------------------
Expanding k restored availability (candidate CES 0.713 -> 0.940 at cal_700) but
collapsed selection (selected CES given availability 0.692 -> 0.234), because
connected candidates grew 10.0 -> 53.2 against a fixed 6-record packet. At
cal_3000/k=300 there are 103.8 connected candidates and 23.6 plausible bridges
per task. So the goal here is pressure reduction WITHOUT losing role coverage.

Runtime safety
--------------
Permitted: canonical entity match, target relation match (parsed from the
question), runtime-visible one-hop connectivity, retrieval score and rank,
duplicate / near-duplicate similarity, repeated entity-relation patterns.

Forbidden: required_evidence_ids, record_kind, proof_edges, latent_bridge,
answer_node, the evaluator graph, any generator label. Enforced by a test that
strips docstrings and comments before scanning.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..retrieval.canonicalization import _norm, extract_identity_links
from .bridge_extraction import extract_v4_entities
from .query_stage import extract_target_relation


@dataclass
class PrefilterResult:
    """Compressed working set plus the evidence for why it is safe."""
    kept: list[str] = field(default_factory=list)
    working_set_size: int = 0
    input_size: int = 0
    preserved_count: int = 0
    duplicates_collapsed: int = 0
    unique_signatures_before: int = 0
    unique_signatures_after: int = 0
    signature_duplicates_before: int = 0
    signature_duplicates_after: int = 0

    def summary(self) -> dict[str, Any]:
        return {
            "working_set_size": self.working_set_size,
            "input_size": self.input_size,
            "preserved_count": self.preserved_count,
            "duplicates_collapsed": self.duplicates_collapsed,
            "unique_signatures_before": self.unique_signatures_before,
            "unique_signatures_after": self.unique_signatures_after,
            "signature_duplicates_before": self.signature_duplicates_before,
            "signature_duplicates_after": self.signature_duplicates_after,
        }


class _Row:
    """Shim so a record can feed the qualified canonicalization parser."""

    def __init__(self, record_id: str, content: str):
        self.evidence_id = record_id
        self.content = content


def structural_signature(content: str, canonical_subject: str | None,
                         relation: str) -> tuple:
    """Runtime-safe signature used for near-duplicate suppression.

    (canonical anchor present, relation present, frozenset of visible entities).
    Records sharing a signature are structurally equivalent as far as anything
    the runtime can observe, so keeping several of them adds competition without
    adding structural coverage -- which is exactly the pressure B3 diagnosed.
    """
    normalized = _norm(content)
    return (
        bool(canonical_subject) and _norm(canonical_subject) in normalized,
        bool(relation) and _norm(relation) in normalized,
        frozenset(_norm(entity) for entity in extract_v4_entities(content)),
    )


def is_identity_like(content: str) -> bool:
    return bool(extract_identity_links([_Row("probe", content)]))


def structural_prefilter(
    *, candidate_ids: Sequence[str], texts: Mapping[str, str], question: str,
    canonical_subject: str | None, working_set_size: int,
    fusion_scores: Mapping[str, float] | None = None,
) -> PrefilterResult:
    """Compress a candidate pool to ``working_set_size``, conservatively.

    Order of operations, frozen:
      1. collapse near-identical records by structural signature
      2. mark as PRESERVED anything that is an identity anchor, mentions the
         canonical subject, matches the target relation, or mentions an entity
         reachable in one visible hop from the subject
      3. rank preserved records first, then the rest -- within each group by
         retrieval score descending, then rank ascending, then record_id
      4. truncate

    No hard per-category quotas in this version. Category PRESERVATION before
    global ranking is the weaker, safer intervention; quotas would add tunable
    numbers before it is known whether they are needed at all.
    """
    scores = fusion_scores or {}
    relation = extract_target_relation(question) or ""
    ordered = list(candidate_ids)
    rank_of = {record_id: index for index, record_id in enumerate(ordered, 1)}

    # --- step 1: near-duplicate suppression -------------------------------
    signatures_before: list[tuple] = []
    best_per_signature: dict[tuple, str] = {}
    for record_id in ordered:
        signature = structural_signature(
            texts.get(record_id, ""), canonical_subject, relation)
        signatures_before.append(signature)
        # First occurrence wins: the pool is already in retrieval-rank order, so
        # the retained representative is the best-ranked of its equivalence class.
        if signature not in best_per_signature:
            best_per_signature[signature] = record_id
    survivors = [record_id for record_id in ordered
                 if best_per_signature.get(
                     structural_signature(texts.get(record_id, ""),
                                          canonical_subject, relation))
                 == record_id]
    collapsed = len(ordered) - len(survivors)

    # --- step 2: admissibility, not ranking -------------------------------
    bridge_entities: set[str] = set()
    if canonical_subject:
        subject_norm = _norm(canonical_subject)
        for record_id in survivors:
            content = texts.get(record_id, "")
            if subject_norm and subject_norm in _norm(content):
                for entity in extract_v4_entities(content):
                    if _norm(entity) != subject_norm:
                        bridge_entities.add(_norm(entity))

    def preserved(record_id: str) -> bool:
        content = texts.get(record_id, "")
        normalized = _norm(content)
        if canonical_subject and _norm(canonical_subject) in normalized:
            return True
        if relation and _norm(relation) in normalized:
            return True
        if any(entity and entity in normalized for entity in bridge_entities):
            return True
        return is_identity_like(content)

    flags = {record_id: preserved(record_id) for record_id in survivors}

    # --- step 3: deterministic order --------------------------------------
    def sort_key(record_id: str) -> tuple:
        return (
            0 if flags[record_id] else 1,
            -(scores.get(record_id) or 0.0),
            rank_of.get(record_id, 10 ** 9),
            record_id,
        )

    kept = sorted(survivors, key=sort_key)[:working_set_size]

    signatures_after = [
        structural_signature(texts.get(record_id, ""), canonical_subject, relation)
        for record_id in kept]
    return PrefilterResult(
        kept=kept, working_set_size=len(kept), input_size=len(ordered),
        preserved_count=sum(1 for record_id in kept if flags[record_id]),
        duplicates_collapsed=collapsed,
        unique_signatures_before=len(set(signatures_before)),
        unique_signatures_after=len(set(signatures_after)),
        signature_duplicates_before=len(signatures_before) - len(set(signatures_before)),
        signature_duplicates_after=len(signatures_after) - len(set(signatures_after)),
    )


def oracle_prefilter(
    *, candidate_ids: Sequence[str], required: Sequence[str],
    working_set_size: int,
) -> PrefilterResult:
    """P3 CEILING. Uses oracle labels to choose the subset -- never promotable.

    Answers only the narrow question: if a perfect compressor retained the best
    ``working_set_size`` candidates, could UNCHANGED S2 recover? It hands S2 a
    subset, not an answer: S2 still has to select six records from it, so a low
    P3 means compression itself is insufficient rather than that this particular
    compressor is weak.
    """
    pool = list(candidate_ids)
    in_pool = set(pool)
    protected = [record_id for record_id in dict.fromkeys(required)
                 if record_id in in_pool]
    kept = list(protected[:working_set_size])
    chosen = set(kept)
    for record_id in pool:
        if len(kept) >= working_set_size:
            break
        if record_id not in chosen:
            kept.append(record_id)
            chosen.add(record_id)
    return PrefilterResult(
        kept=kept, working_set_size=len(kept), input_size=len(pool),
        preserved_count=len(protected))
