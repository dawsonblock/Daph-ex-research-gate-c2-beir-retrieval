"""G2-v2: entity-bound path-completion semantics. Runtime-only.

G2-v1 diagnosis (frozen in configs/gate_g1_runtime_graph_v1.json:G2_RESULT): a
record was treated as "completing" a path if the target relation's normalized
text appeared ANYWHERE in the record, with no requirement that the relation be
expressed ABOUT the specific bridge entity the path is walking through. That is
K0 below, kept only as the control. This module adds K1/K2, which bind a
relation to a specific entity before calling a path complete.

Three predicates, kept separate on purpose
-------------------------------------------
    relation_surface_present : the relation's text occurs somewhere in the record
    relation_expressed       : the record asserts SOME entity has this relation
    relation_bound_to_entity : the record asserts THIS entity has this relation

Only the third should ever mark a path complete. K0 conflates all three into
the first (weakest) one; that conflation is exactly what G2-v1's diagnosis
named as the construction defect.

Entity binding, not sentence-level matching
--------------------------------------------
K1/K2 use ``hrm_adaptive_memory.c4.relation_grammar.bindings_for_entity``,
which parses (subject_entity, relation, value) facts directly from the
corpus's own generator grammar (24 template forms transcribed from
``generalization_dataset_v4.py:_render`` -- Changelog:/Revision applied/
registry entry/JSON kv/table/message forms, all visible surface structure).

An earlier version of this module reused
``hrm_adaptive_memory.c4.relational_state.parse_relation_edges`` instead, but
that parser requires BOTH sides of a fact to look like a V4 entity name, and
most answer-bearing records here have a value that is a number, code, or JSON
fragment -- so it silently matched almost nothing on real corpus text (11/13
real samples bind correctly with relation_grammar; the old parser bound 0/4 on
equivalent samples). ``relation_grammar`` does not constrain the value's
shape, only the subject side, since that is the side a G2 path is binding to.

A record only completes a path for (entity, relation) if some parsed fact
binds THAT entity to a relation matching the target -- so a record like
"Registry entry: Finch control module — ownership tier — Falcon regulator."
cannot bind Falcon's ownership tier to Finch merely because both entities
appear in the text: Falcon only ever appears in the VALUE slot, never as a
subject, so ``bindings_for_entity(..., "Falcon regulator")`` returns nothing.

K1 vs K2
--------
K1 requires the parsed relation to normalize to an EXACT match of the target
relation. K2 additionally accepts ``relational_state._relation_matches``'s
existing partial/synonym comparison (assigned category<->category/assigned,
ownership tier<->tier/ownership, etc) -- an already-frozen, already-tested
table, reused verbatim. Neither K1 nor K2 adds a new alias in this sprint.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping

from .relation_grammar import bindings_for_entity
from .relational_state import _relation_matches
from ..retrieval.canonicalization import _norm

PARSER_VERSION = "g2v2-endpoint-recognition-v1"


@dataclass(frozen=True)
class EndpointRecognition:
    """Why a record was (or was not) accepted as completing a path. Persisted,
    never reduced to a bare bool, so a false closure can be audited later."""
    record_id: str
    entity_id: str
    requested_relation: str
    canonical_relation: str | None
    entity_bound: bool
    relation_surface_match: bool
    relation_family_match: bool
    completed: bool
    completion_reason: str
    parser_version: str = PARSER_VERSION
    source_span_hash: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "record_id": self.record_id, "entity_id": self.entity_id,
            "requested_relation": self.requested_relation,
            "canonical_relation": self.canonical_relation,
            "entity_bound": self.entity_bound,
            "relation_surface_match": self.relation_surface_match,
            "relation_family_match": self.relation_family_match,
            "completed": self.completed,
            "completion_reason": self.completion_reason,
            "parser_version": self.parser_version,
            "source_span_hash": self.source_span_hash,
        }


def _span_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def k0_literal_completion(
    *, record_id: str, entity: str, relation: str, texts: Mapping[str, str],
) -> EndpointRecognition:
    """CONTROL. Unchanged G2-v1 behavior: relation text anywhere in the record,
    no entity binding. Kept only so K1/K2 have a like-for-like baseline."""
    content = texts.get(record_id, "")
    relation_norm = _norm(relation) if relation else ""
    surface = bool(relation_norm) and relation_norm in _norm(content)
    return EndpointRecognition(
        record_id=record_id, entity_id=_norm(entity), requested_relation=relation_norm,
        canonical_relation=None, entity_bound=False,
        relation_surface_match=surface, relation_family_match=surface,
        completed=surface,
        completion_reason=("k0_literal_surface_match" if surface
                           else "k0_no_surface_match"),
        source_span_hash=_span_hash(content))


def _entity_bound_facts(record_id: str, entity: str, texts: Mapping[str, str]):
    content = texts.get(record_id, "")
    entity_norm = _norm(entity)
    facts = bindings_for_entity(content, record_id, entity)
    return content, entity_norm, facts


def k1_entity_bound_exact_completion(
    *, record_id: str, entity: str, relation: str, texts: Mapping[str, str],
) -> EndpointRecognition:
    """PRIMARY. A record completes the path only if a parsed relation fact
    binds THIS entity to a relation that normalizes to an EXACT match of the
    requested relation."""
    content, entity_norm, bound_facts = _entity_bound_facts(record_id, entity, texts)
    relation_norm = _norm(relation) if relation else ""
    entity_bound = bool(bound_facts)
    exact = next((f for f in bound_facts if _norm(f.relation) == relation_norm), None)
    completed = exact is not None
    return EndpointRecognition(
        record_id=record_id, entity_id=entity_norm, requested_relation=relation_norm,
        canonical_relation=(exact.relation if exact else None),
        entity_bound=entity_bound,
        relation_surface_match=completed, relation_family_match=completed,
        completed=completed,
        completion_reason=("k1_entity_bound_exact_match" if completed
                           else "k1_entity_bound_no_exact_relation_match" if entity_bound
                           else "k1_entity_not_bound_by_any_parsed_fact"),
        source_span_hash=_span_hash(content))


def k2_entity_bound_family_completion(
    *, record_id: str, entity: str, relation: str, texts: Mapping[str, str],
) -> EndpointRecognition:
    """DIAGNOSTIC/OPTIONAL. Same entity binding as K1, but relation comparison
    uses relational_state's existing frozen partial/synonym matcher instead of
    exact-string equality. No new alias table is introduced."""
    content, entity_norm, bound_facts = _entity_bound_facts(record_id, entity, texts)
    relation_norm = _norm(relation) if relation else ""
    entity_bound = bool(bound_facts)
    exact = next((f for f in bound_facts if _norm(f.relation) == relation_norm), None)
    family = exact or next(
        (f for f in bound_facts if _relation_matches(f.relation, relation_norm)), None)
    completed = family is not None
    return EndpointRecognition(
        record_id=record_id, entity_id=entity_norm, requested_relation=relation_norm,
        canonical_relation=(family.relation if family else None),
        entity_bound=entity_bound,
        relation_surface_match=(exact is not None),
        relation_family_match=completed,
        completed=completed,
        completion_reason=("k2_entity_bound_exact_match" if exact is not None
                           else "k2_entity_bound_family_match" if completed
                           else "k2_entity_bound_no_family_match" if entity_bound
                           else "k2_entity_not_bound_by_any_parsed_fact"),
        source_span_hash=_span_hash(content))


#: Mode name -> recognizer function, all sharing the exact same call signature
#: so g2_paths.py's traversal/ranking code never needs to know which is active.
COMPLETION_MODES = {
    "K0": k0_literal_completion,
    "K1": k1_entity_bound_exact_completion,
    "K2": k2_entity_bound_family_completion,
}
