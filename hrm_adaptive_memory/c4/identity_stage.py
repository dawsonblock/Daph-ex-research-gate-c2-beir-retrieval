"""C4 identity stage — reuse the qualified I3 parser, no reimplementation.

I3 is the qualified surface-normalization mechanism (Gate C3, 92% correct,
0% false resolution). This stage imports the exact parser and decision logic.
It does NOT reimplement identity parsing.
"""
from __future__ import annotations

from typing import Sequence, Mapping

from ..retrieval.canonicalization import (
    extract_identity_links, IdentityLink, _norm)
from .contracts import C4Arm, IdentityResolution, RetrievalResult
from .query_stage import extract_subject
from .bridge_extraction import extract_v4_entities


class _Rec:
    """Shim so pool rows can feed the qualified canonicalization parser."""
    def __init__(self, eid: str, content: str):
        self.evidence_id = eid
        self.content = content


def run_identity_stage(question: str, arm: C4Arm,
                       retrieval: RetrievalResult,
                       texts: Mapping[str, str]) -> IdentityResolution:
    """Run identity resolution on the retrieved candidate pool.

    For "none" policy (C4-0..C4-2): no resolution attempted.
    For "i3_explicit_identity" (C4-3+): read identity records from the
    candidate pool and extract explicit surface→canonical mappings.

    Safety rule: if multiple conflicting mappings exist, return AMBIGUOUS.
    Never guess.
    """
    subject_raw = extract_subject(question)
    surface_norm = _norm(subject_raw) if subject_raw else ""

    if arm.identity_policy == "none":
        return IdentityResolution(
            status="UNRESOLVED",
            surface=subject_raw or None,
            canonical=None,
            evidence_ids=(),
            candidate_mappings=(),
            resolution_needed=bool(subject_raw),
            resolution_attempted=False,
            resolution_changed_state=False,
        )

    if arm.identity_policy != "i3_explicit_identity":
        raise ValueError(f"Unknown identity_policy: {arm.identity_policy}")

    # I3: read identity records from the retrieved candidate pool
    candidate_recs = [
        _Rec(eid, texts.get(eid, ""))
        for eid in retrieval.candidate_ids
        if eid in texts
    ]
    identity_links = extract_identity_links(candidate_recs)

    # Also try without leading "the " (description surfaces)
    stripped = _norm(subject_raw.replace("the ", "", 1)) if subject_raw else ""

    # Find all matching mappings
    mappings: list[IdentityLink] = []
    for link in identity_links:
        link_surface = _norm(link.surface)
        if link_surface == surface_norm or link_surface == stripped:
            mappings.append(link)
        elif surface_norm and (surface_norm in link_surface or stripped in link_surface):
            mappings.append(link)

    # Deduplicate by (surface, canonical)
    unique: dict[tuple[str, str], IdentityLink] = {}
    for link in mappings:
        key = (_norm(link.surface), _norm(link.canonical))
        if key not in unique:
            unique[key] = link
    mappings = list(unique.values())

    resolution_needed = bool(subject_raw)
    resolution_attempted = True

    if not mappings:
        # No identity record maps this surface. Check if the subject is already
        # a canonical entity: if it appears as a parsed entity (not merely a
        # substring) in non-identity evidence records in the candidate pool,
        # it is a known canonical that needs no resolution.
        # This is the EXACT path required by the C4 protocol.
        #
        # Tightened from substring matching to parsed entity equality so an
        # incidental distractor mention cannot make a surface form appear
        # canonical. The subject must match a V4 entity extracted from the
        # evidence content.
        if subject_raw:
            subject_norm = _norm(subject_raw)
            for eid in retrieval.candidate_ids:
                if eid in texts and "/identity" not in eid:
                    entities = extract_v4_entities(texts[eid])
                    entity_norms = {_norm(e) for e in entities}
                    if subject_norm in entity_norms:
                        return IdentityResolution(
                            status="EXACT",
                            surface=subject_raw,
                            canonical=subject_raw,
                            evidence_ids=(),
                            candidate_mappings=(),
                            resolution_needed=False,
                            resolution_attempted=True,
                            resolution_changed_state=False,
                        )
        return IdentityResolution(
            status="UNRESOLVED",
            surface=subject_raw or None,
            canonical=None,
            evidence_ids=(),
            candidate_mappings=(),
            resolution_needed=resolution_needed,
            resolution_attempted=resolution_attempted,
            resolution_changed_state=False,
        )

    canonicals = {_norm(m.canonical) for m in mappings}
    if len(canonicals) > 1:
        # AMBIGUOUS: conflicting mappings, abstain
        return IdentityResolution(
            status="AMBIGUOUS",
            surface=subject_raw or None,
            canonical=None,
            evidence_ids=tuple(m.record_id for m in mappings),
            candidate_mappings=tuple(
                {"surface": m.surface, "canonical": m.canonical,
                 "source_evidence_id": m.record_id}
                for m in mappings),
            resolution_needed=resolution_needed,
            resolution_attempted=resolution_attempted,
            resolution_changed_state=False,
        )

    # Single unique mapping → RESOLVED
    chosen = mappings[0]
    return IdentityResolution(
        status="RESOLVED",
        surface=subject_raw or None,
        canonical=chosen.canonical,
        evidence_ids=(chosen.record_id,),
        candidate_mappings=(
            {"surface": chosen.surface, "canonical": chosen.canonical,
             "source_evidence_id": chosen.record_id},
        ),
        resolution_needed=resolution_needed,
        resolution_attempted=resolution_attempted,
        resolution_changed_state=True,
    )
