"""C4 runtime relational state — structured representation of first-pass evidence.

This module replaces the regex-based bridge extraction with a proper
relational state model. The key insight is that bridge discovery is not
about guessing capitalized phrases — it is about parsing relation edges
from evidence records and determining whether the target relation is
already bound or needs a follow-up retrieval.

Key data structures:
- RelationFact: a parsed (source, relation, target) edge from evidence
- RelationalState: accumulated knowledge from first-pass retrieval

The mechanism asks:
1. Is the requested target relation already bound by current evidence?
2. If not, which runtime-visible evidence edge introduces a connected
   entity that could bridge toward the requested relation?

This is the correct InformationState problem.
"""
from __future__ import annotations

import re
import json
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from .bridge_extraction import extract_v4_entities
from .query_stage import extract_target_relation


# --- Relation parsing -------------------------------------------------------

# V4 relation names that appear in link records
_RELATION_NAMES = {
    "registered asset", "active configuration", "ownership tier",
    "assigned category", "calibration class", "routing class",
    "spare enclosure", "mounted monitor",
}

# Template patterns for parsing relation edges from V4 link records.
# Each pattern captures (source_entity, relation, target_entity).
# All patterns capture groups in (source, relation, target) order.
# For patterns where the natural capture order differs, a reorder tuple
# specifies which captured groups map to (source, relation, target).
# None means groups are already in (source, relation, target) order.
_LINK_PATTERNS: list[tuple[re.Pattern, tuple[int, ...] | None]] = [
    # "subject=X; RELATION=Y" (semicolon-separated) → (source, relation, target)
    (re.compile(r"subject\s*=\s*(.+?)\s*;\s*(.+?)\s*=\s*(.+?)\s*$"), None),
    # '{"subject": "X", "RELATION": "Y"}' (JSON) → (source, relation, target)
    (re.compile(r'\{\s*"subject"\s*:\s*"(.+?)"\s*,\s*"(.+?)"\s*:\s*"(.+?)"\s*\}'), None),
    # "[RELATION] X -> Y" → reorder (2, 1, 3) = (X, RELATION, Y)
    (re.compile(r"\[(.+?)\]\s+(.+?)\s+->\s+(.+?)\.?\s*$"), (2, 1, 3)),
    # "- updated X: RELATION now Y" → (source, relation, target)
    (re.compile(r"^-\s*updated\s+(.+?):\s*(.+?)\s+now\s+(.+?)\.?\s*$"), None),
    # "X is assigned Y" / "X is allocated to Y" → (source, implicit, target) = 2 groups
    (re.compile(r"(.+?)\s+is\s+(?:assigned|allocated\s+to)\s+(.+?)\.?\s*$"), None),
    # "Note: the RELATION for X resolves to Y" → reorder (2, 1, 3) = (X, RELATION, Y)
    (re.compile(
        r"(?:note:\s*)?the\s+(.+?)\s+for\s+(.+?)\s+resolves\s+to\s+(.+?)\.?\s*$",
        re.IGNORECASE), (2, 1, 3)),
    # "Engineering notes indicate X uses Y as its RELATION" → reorder (1, 3, 2) = (X, RELATION, Y)
    (re.compile(
        r"engineering\s+notes\s+indicate\s+(.+?)\s+uses\s+(.+?)\s+as\s+its\s+(.+?)\.?\s*$",
        re.IGNORECASE), (1, 3, 2)),
    # "Changelog: RELATION for X set to Y" → reorder (2, 1, 3) = (X, RELATION, Y)
    (re.compile(
        r"(?:changelog:\s*)?(.+?)\s+for\s+(.+?)\s+set\s+to\s+(.+?)\.?\s*$",
        re.IGNORECASE), (2, 1, 3)),
    # "Changelog: RELATION for X now Y" → reorder (2, 1, 3) = (X, RELATION, Y)
    (re.compile(
        r"(?:changelog:\s*)?(.+?)\s+for\s+(.+?)\s+now\s+(.+?)\.?\s*$",
        re.IGNORECASE), (2, 1, 3)),
    # "Per the RELATION register, X is allocated to Y" → (source, implicit, target) = 2 groups
    (re.compile(
        r"per\s+the\s+.+?\s+register,\s+(.+?)\s+is\s+allocated\s+to\s+(.+?)\.?\s*$",
        re.IGNORECASE), None),
    # "Registry entry: X — RELATION — Y" → (source, relation, target)
    (re.compile(
        r"registry\s+entry:\s+(.+?)\s+—\s+(.+?)\s+—\s+(.+?)\.?\s*$",
        re.IGNORECASE), None),
    # "During setup, X was paired with Y for RELATION" → reorder (1, 3, 2) = (X, RELATION, Y)
    (re.compile(
        r"during\s+setup,\s+(.+?)\s+was\s+paired\s+with\s+(?:\d+\s+for\s+)?(.+?)(?:\s+for\s+(.+?))?\.",
        re.IGNORECASE), (1, 3, 2)),
    # "Revision applied — X RELATION changed to Y" → (source, relation, target)
    # Use V4 entity pattern for source to avoid splitting entity name
    (re.compile(
        r"revision\s+applied\s+—\s+([A-Z][a-z]+(?:\s+[a-z]+){1,3})\s+(.+?)\s+changed\s+to\s+(.+?)\.?\s*$"),
     None),
]


@dataclass(frozen=True)
class RelationFact:
    """A parsed relation edge from evidence."""
    source_entity: str
    relation: str
    target_entity: str
    evidence_id: str

    def __post_init__(self):
        # Normalize relation to lowercase
        object.__setattr__(self, 'relation', self.relation.lower().strip())
        object.__setattr__(self, 'source_entity', self.source_entity.strip())
        object.__setattr__(self, 'target_entity', self.target_entity.strip())


def _normalize_relation(rel: str) -> str:
    """Normalize a relation name to the canonical form."""
    rel = rel.lower().strip().rstrip(".;:")
    # Map common variations
    rel_map = {
        "registered asset": "registered asset",
        "active configuration": "active configuration",
        "ownership tier": "ownership tier",
        "assigned category": "assigned category",
        "calibration class": "calibration class",
        "routing class": "routing class",
        "spare enclosure": "spare enclosure",
        "mounted monitor": "mounted monitor",
    }
    return rel_map.get(rel, rel)


def parse_relation_edges(content: str, evidence_id: str) -> list[RelationFact]:
    """Parse relation edges from a V4 evidence record.

    Returns a list of RelationFact edges found in the content.
    """
    edges: list[RelationFact] = []
    content = content.strip()

    for pat, reorder in _LINK_PATTERNS:
        m = pat.search(content)
        if m:
            groups = m.groups()
            if reorder:
                # Reorder groups to (source, relation, target)
                if len(groups) >= max(reorder):
                    source = groups[reorder[0] - 1]
                    relation = groups[reorder[1] - 1] if reorder[1] <= len(groups) else ""
                    target = groups[reorder[2] - 1] if reorder[2] <= len(groups) else ""
                else:
                    continue
            elif len(groups) == 3:
                source, relation, target = groups
            elif len(groups) == 2:
                # Two-group patterns: source and target, relation is implicit
                source, target = groups
                # Infer relation from content
                if "registered asset" in content.lower():
                    relation = "registered asset"
                elif "active configuration" in content.lower():
                    relation = "active configuration"
                elif "assigned" in content.lower():
                    relation = "assigned category"
                else:
                    relation = "unknown"
            else:
                continue

            # Extract V4 entities from source and target
            source_ents = extract_v4_entities(source)
            target_ents = extract_v4_entities(target)

            if source_ents and target_ents:
                edges.append(RelationFact(
                    source_entity=source_ents[0],
                    relation=_normalize_relation(relation),
                    target_entity=target_ents[0],
                    evidence_id=evidence_id,
                ))
                break  # Successfully parsed, stop trying patterns
            # If entity extraction failed, try next pattern

    return edges


# --- Relational state -------------------------------------------------------

@dataclass(frozen=True)
class RelationalState:
    """Structured representation of what the first-pass retrieval found.

    This is the runtime-visible knowledge state used to decide:
    1. Is the target relation already bound?
    2. If not, what bridge entity should we follow up on?
    """
    original_subject: str
    canonical_subject: str | None
    target_relation: str
    known_entities: tuple[str, ...] = ()
    known_relations: tuple[RelationFact, ...] = ()
    candidate_bridges: tuple[str, ...] = ()
    target_bound: bool = False

    def with_facts(self, facts: Sequence[RelationFact],
                   entities: Sequence[str]) -> "RelationalState":
        """Return a new state with additional facts and entities."""
        all_facts = tuple(self.known_relations) + tuple(facts)
        all_entities = tuple(dict.fromkeys(
            list(self.known_entities) + list(entities)))
        return RelationalState(
            original_subject=self.original_subject,
            canonical_subject=self.canonical_subject,
            target_relation=self.target_relation,
            known_entities=all_entities,
            known_relations=all_facts,
            candidate_bridges=self.candidate_bridges,
            target_bound=self.target_bound,
        )

    def with_bridge(self, bridge: str) -> "RelationalState":
        """Return a new state with a candidate bridge added."""
        bridges = tuple(dict.fromkeys(list(self.candidate_bridges) + [bridge]))
        return RelationalState(
            original_subject=self.original_subject,
            canonical_subject=self.canonical_subject,
            target_relation=self.target_relation,
            known_entities=self.known_entities,
            known_relations=self.known_relations,
            candidate_bridges=bridges,
            target_bound=self.target_bound,
        )

    def with_target_bound(self, bound: bool = True) -> "RelationalState":
        """Return a new state with target_bound updated."""
        return RelationalState(
            original_subject=self.original_subject,
            canonical_subject=self.canonical_subject,
            target_relation=self.target_relation,
            known_entities=self.known_entities,
            known_relations=self.known_relations,
            candidate_bridges=self.candidate_bridges,
            target_bound=bound,
        )


def build_relational_state(
    subject: str,
    target_relation: str,
    candidate_ids: tuple[str, ...],
    texts: Mapping[str, str],
    canonical_subject: str | None = None,
    question: str | None = None,
) -> RelationalState:
    """Build a RelationalState from first-pass retrieval results.

    Parses all non-identity evidence records for relation edges, extracts
    entities, and determines whether the target relation is already bound.

    If question is provided, target_relation is extracted from it (overriding
    the explicit parameter if it is empty).
    """
    """Build a RelationalState from first-pass retrieval results.

    Parses all non-identity evidence records for relation edges, extracts
    entities, and determines whether the target relation is already bound.
    """
    # Extract target relation from question if provided and parameter is empty
    if question and not target_relation:
        target_relation = extract_target_relation(question) or ""

    state = RelationalState(
        original_subject=subject,
        canonical_subject=canonical_subject,
        target_relation=target_relation.lower().strip(),
    )

    subject_lower = subject.lower()
    canonical_lower = canonical_subject.lower() if canonical_subject else subject_lower
    all_entities: list[str] = []
    all_facts: list[RelationFact] = []

    for eid in candidate_ids:
        content = texts.get(eid, "")
        if not content:
            continue
        if "/identity" in eid:
            continue

        # Parse relation edges
        edges = parse_relation_edges(content, eid)
        all_facts.extend(edges)

        # Extract entities
        ents = extract_v4_entities(content)
        all_entities.extend(ents)

    # Add subject to known entities
    all_entities.insert(0, subject)

    state = state.with_facts(all_facts, all_entities)

    # Check if target relation is already bound for the subject
    # "Bound" means: there exists a fact (subject, target_relation, X)
    # where X is a value (numeric, enum, or entity)
    target_bound = False
    for fact in state.known_relations:
        if (fact.source_entity.lower() == subject_lower or
            fact.source_entity.lower() == canonical_lower):
            # Check if the fact's relation matches the target relation
            if _relation_matches(fact.relation, target_relation):
                target_bound = True
                break

    # Also check for direct answer records (fact records with numeric values)
    # that mention the subject and contain the target relation keyword
    if not target_bound and target_relation:
        target_rel_lower = target_relation.lower()
        for eid in candidate_ids:
            content = texts.get(eid, "")
            if not content or "/identity" in eid or "/link" in eid:
                continue
            if subject_lower in content.lower() or canonical_lower in content.lower():
                if target_rel_lower in content.lower():
                    # Check for numeric value (answer-bearing)
                    if re.search(r"(?<![\w-])[-+]?\d+(?:\.\d+)?(?![\w-])", content):
                        target_bound = True
                        break
                    # Check for enum/symbolic value
                    if any(kw in content.lower() for kw in
                           ("category", "tier", "class", "grade", "status")):
                        target_bound = True
                        break

    state = state.with_target_bound(target_bound)

    # If target not bound, identify candidate bridges
    if not target_bound:
        bridges = _identify_bridges(subject, canonical_subject, state, candidate_ids, texts)
        for b in bridges:
            state = state.with_bridge(b)

    return state


def _relation_matches(fact_relation: str, target_relation: str) -> bool:
    """Check if a fact's relation matches the target relation."""
    fact_rel = fact_relation.lower().strip()
    target_rel = target_relation.lower().strip()

    # Empty target relation never matches
    if not target_rel:
        return False

    # Direct match
    if fact_rel == target_rel:
        return True

    # Partial match (one contains the other)
    if target_rel in fact_rel or fact_rel in target_rel:
        return True

    # Common synonyms
    synonyms = {
        "assigned category": {"category", "assigned"},
        "ownership tier": {"tier", "ownership"},
        "calibration class": {"calibration", "class"},
        "routing class": {"routing", "class"},
    }
    target_syns = synonyms.get(target_rel, set())
    if any(syn in fact_rel for syn in target_syns):
        return True

    return False


def _identify_bridges(
    subject: str,
    canonical: str | None,
    state: RelationalState,
    candidate_ids: tuple[str, ...],
    texts: Mapping[str, str],
) -> list[str]:
    """Identify candidate bridge entities from the relational state.

    A bridge is an entity that:
    1. Appears in a relation edge with the subject (source or target)
    2. Is NOT the subject itself
    3. Does NOT already bind the target relation

    Connectivity is counted across the FULL candidate pool, not just
    subject-containing records.
    """
    subject_lower = subject.lower()
    canonical_lower = canonical.lower() if canonical else subject_lower
    bridges: list[str] = []

    # From relation edges: find entities connected to subject
    for fact in state.known_relations:
        if fact.source_entity.lower() in (subject_lower, canonical_lower):
            # Subject → X: X is a potential bridge
            if fact.target_entity.lower() not in (subject_lower, canonical_lower):
                bridges.append(fact.target_entity)
        elif fact.target_entity.lower() in (subject_lower, canonical_lower):
            # X → Subject: X is a potential bridge
            if fact.source_entity.lower() not in (subject_lower, canonical_lower):
                bridges.append(fact.source_entity)

    # Count global connectivity for each bridge candidate
    bridge_connectivity: dict[str, dict[str, int]] = {}
    for b in set(bridges):
        b_lower = b.lower()
        counts = {"global_occurrence": 0, "subject_linked": 0,
                  "distinct_records": 0, "relation_edges": 0}
        for eid in candidate_ids:
            content = texts.get(eid, "")
            if not content or "/identity" in eid:
                continue
            if b_lower in content.lower():
                counts["global_occurrence"] += 1
                counts["distinct_records"] += 1
                if subject_lower in content.lower() or canonical_lower in content.lower():
                    counts["subject_linked"] += 1
        for fact in state.known_relations:
            if fact.source_entity.lower() == b_lower or fact.target_entity.lower() == b_lower:
                counts["relation_edges"] += 1
        bridge_connectivity[b] = counts

    # Sort by connectivity (most connected first)
    bridges_sorted = sorted(
        set(bridges),
        key=lambda b: (
            bridge_connectivity.get(b, {}).get("relation_edges", 0),
            bridge_connectivity.get(b, {}).get("global_occurrence", 0),
        ),
        reverse=True,
    )

    return bridges_sorted


def is_target_bound(state: RelationalState) -> bool:
    """Check if the target relation is already bound by current evidence."""
    return state.target_bound


def get_bridge(state: RelationalState) -> str | None:
    """Get the best bridge candidate from the relational state.

    Returns None if no bridge is needed (target already bound) or if no
    bridge candidate exists.
    """
    if state.target_bound:
        return None
    if not state.candidate_bridges:
        return None
    return state.candidate_bridges[0]
