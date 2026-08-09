"""Runtime-safe entity/relation binding, grounded in the corpus's own
generator grammar -- not a guess at plausible sentence forms.

Why this exists instead of reusing relational_state.py
--------------------------------------------------------
``relational_state.parse_relation_edges`` already parses (source, relation,
target) triples, but it requires BOTH the source AND the target to look like a
V4 entity name (``extract_v4_entities``, a capitalized-word-plus-lowercase-
words pattern). Most answer-bearing records in this corpus have a value that
is a code, a number, an enum string, or a JSON fragment -- none of which match
that shape -- so ``parse_relation_edges`` silently returns nothing for the
majority of records that actually assert an entity's relation value. That
silent gap is exactly what G2-v1's diagnosis named: completion recognition
that never fires because it demands too much shape from the wrong side of the
fact. This module fixes that by not constraining the value's shape at all --
only the subject side needs to look like an entity, since that is the side a
G2 path is trying to bind.

The templates below are transcribed directly from
``hrm_adaptive_memory/experiments/generalization_dataset_v4.py:_render`` (the
6 styles x 4 variants the corpus is actually generated from). Reading a
generator's deterministic SURFACE grammar is explicitly permitted -- it is
visible sentence structure, not a hidden oracle object -- and no proof edge,
answer node, or evidence-id field is read anywhere in this module or by
anything that calls it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .bridge_extraction import extract_v4_entities
from ..retrieval.canonicalization import _norm

#: (pattern, group order as (subject, relation, obj), needs_entity_split)
#: needs_entity_split=True means subject and relation are NOT separated by a
#: fixed delimiter in this template (e.g. "X relation changed to Y") and must
#: be disambiguated by finding where the V4 entity name ends.
_TEMPLATES: list[tuple[re.Pattern, bool]] = [
    # formal_registry
    (re.compile(r"^The (?P<relation>.+?) registry records that (?P<subject>.+?) is assigned (?P<obj>.+?)\.$"), False),
    (re.compile(r"^Registry entry: (?P<subject>.+?) — (?P<relation>.+?) — (?P<obj>.+?)\.$"), False),
    (re.compile(r"^Per the (?P<relation>.+?) register, (?P<obj>.+?) is allocated to (?P<subject>.+?)\.$"), False),
    (re.compile(r"^It is recorded in the (?P<relation>.+?) registry that (?P<subject>.+?) holds (?P<obj>.+?)\.$"), False),
    # technical_note
    (re.compile(r"^During setup, (?P<subject>.+?) was paired with (?P<obj>.+?) for (?P<relation>.+?)\.$"), False),
    (re.compile(r"^Note: the (?P<relation>.+?) for (?P<subject>.+?) resolves to (?P<obj>.+?)\.$"), False),
    (re.compile(r"^Engineering notes indicate (?P<subject>.+?) uses (?P<obj>.+?) as its (?P<relation>.+?)\.$"), False),
    (re.compile(r"^For (?P<subject>.+?), (?P<relation>.+?) requires (?P<obj>.+?)\.$"), False),
    # key_value_log
    (re.compile(r"^subject=(?P<subject>.+?); (?P<relation>.+?)=(?P<obj>.+?)$"), False),
    (re.compile(r"^\[(?P<relation>.+?)\] (?P<subject>.+?) -> (?P<obj>.+?)$"), False),
    (re.compile(r'^\{"subject":\s*"(?P<subject>.+?)",\s*"(?P<relation>.+?)":\s*"(?P<obj>.+?)"\}$'), False),
    (re.compile(r"^(?P<relation>.+?):\n  subject: (?P<subject>.+?)\n  value: (?P<obj>.+?)$"), False),
    # table_text
    (re.compile(r"^\| subject \| (?P<relation>.+?) \|\n\| (?P<subject>.+?) \| (?P<obj>.+?) \|$"), False),
    (re.compile(r"^Row: (?P<subject>.+?) \| (?P<relation>.+?) \| (?P<obj>.+?)$"), False),
    (re.compile(r"^Table (?P<relation>.+?): (?P<subject>.+?) maps to (?P<obj>.+?)\.$"), False),
    (re.compile(r"^(?P<subject>.+?)\t(?P<relation>.+?)\t(?P<obj>.+?)$"), False),
    # change_log
    (re.compile(r"^Changelog: (?P<relation>.+?) for (?P<subject>.+?) set to (?P<obj>.+?)\.$"), False),
    (re.compile(r"^- updated (?P<subject>.+?): (?P<relation>.+?) now (?P<obj>.+?)$"), False),
    (re.compile(r"^Revision applied — (?P<entity_relation>.+?) changed to (?P<obj>.+?)\.$"), True),
    (re.compile(r"^\[change\] (?P<subject>.+?) :: (?P<relation>.+?) := (?P<obj>.+?)$"), False),
    # message
    (re.compile(r"^Quick note — (?P<subject>.+?)'s (?P<relation>.+?) is (?P<obj>.+?), in case it comes up\.$"), False),
    (re.compile(r"^Hi, confirming that (?P<subject>.+?) has (?P<obj>.+?) as its (?P<relation>.+?)\.$"), False),
    (re.compile(r"^FYI: for (?P<subject>.+?) the (?P<relation>.+?) we settled on was (?P<obj>.+?)\.$"), False),
    (re.compile(r"^Following up: (?P<subject>.+?) → (?P<relation>.+?) → (?P<obj>.+?)\.$"), False),
]


@dataclass(frozen=True)
class RelationBinding:
    """An (entity, relation, value) fact read off one record's surface text.
    Unlike relational_state.RelationFact, ``value`` has no shape requirement --
    it is whatever text followed the relation slot in the matched template."""
    subject_entity: str
    relation: str
    value: str
    record_id: str
    template_index: int


def _split_entity_relation(chunk: str, entity_hint: str | None = None) -> tuple[str, str] | None:
    """"{subject} {relation}" with no delimiter between them (the 'Revision
    applied' template) -- the corpus's own grammar makes this ambiguous when
    the entity name and the relation name are both multi-word (blind
    extraction over-consumes into the relation, e.g. "Ibis sensor array
    assigned category" -> entity="Ibis sensor array assigned", relation
    ="category" when relation is really "assigned category").

    ``entity_hint`` resolves the ambiguity when the caller already has a
    specific entity to test binding against (bindings_for_entity does, since
    the whole point of entity-bound checking is that a candidate entity is
    already in hand from the graph topology) -- if the chunk starts with that
    exact entity text, split there instead of guessing. Falls back to blind
    V4-entity extraction only when no hint is given or the hint doesn't match,
    which is the correct behavior for the entity-agnostic top-level parse."""
    if entity_hint:
        hint_norm = _norm(entity_hint)
        chunk_norm = _norm(chunk)
        if hint_norm and chunk_norm.startswith(hint_norm):
            remainder = chunk[len(entity_hint):].strip() if chunk.lower().startswith(
                entity_hint.lower()) else chunk[len(hint_norm):].strip()
            if remainder:
                return (entity_hint, remainder)
    entities = extract_v4_entities(chunk)
    if not entities:
        return None
    subject = entities[0]
    if not chunk.startswith(subject):
        return None
    remainder = chunk[len(subject):].strip()
    return (subject, remainder) if remainder else None


def parse_relation_bindings(
    content: str, record_id: str, entity_hint: str | None = None,
) -> list[RelationBinding]:
    """Parse every (entity, relation, value) fact this record's surface text
    supports, using ONLY the corpus's own known template grammar. Returns []
    for records that are not relation-bearing (identity/abbreviation records,
    free text, etc) -- that is the correct, not a failure, outcome.

    ``entity_hint``, when given, resolves the one structurally ambiguous
    template (see _split_entity_relation); it does not change matching for
    any other template."""
    text = content.strip()
    out: list[RelationBinding] = []
    for index, (pattern, needs_split) in enumerate(_TEMPLATES):
        m = pattern.match(text)
        if not m:
            continue
        groups = m.groupdict()
        if needs_split:
            split = _split_entity_relation(groups["entity_relation"], entity_hint)
            if split is None:
                continue
            subject_raw, relation_raw = split
        else:
            subject_raw, relation_raw = groups["subject"], groups["relation"]
        subject_entities = extract_v4_entities(subject_raw) or (subject_raw,)
        out.append(RelationBinding(
            subject_entity=subject_entities[0].strip(),
            relation=relation_raw.strip().rstrip(".;:"),
            value=groups["obj"].strip(),
            record_id=record_id, template_index=index))
        break  # one matched template is enough; templates are mutually exclusive by construction
    return out


def bindings_for_entity(content: str, record_id: str, entity: str) -> list[RelationBinding]:
    """Bindings from this record whose subject matches ``entity`` (normalized,
    substring-tolerant to absorb extract_v4_entities' boundary quirks)."""
    entity_norm = _norm(entity)
    if not entity_norm:
        return []
    out = []
    for binding in parse_relation_bindings(content, record_id, entity_hint=entity):
        bn = _norm(binding.subject_entity)
        if bn == entity_norm or entity_norm in bn or bn in entity_norm:
            out.append(binding)
    return out
