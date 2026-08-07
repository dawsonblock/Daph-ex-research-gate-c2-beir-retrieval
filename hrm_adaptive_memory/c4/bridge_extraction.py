"""C4 bridge extraction — runtime-visible bridge entity discovery.

The qualified C2 mechanism was a two-pass iterative retrieval:

    question → first retrieval → discover bridge → subject + bridge + relation
    → second retrieval → merge candidates

C4-1 originally collapsed this into a single-pass subject+relation query, which
is not equivalent. This module extracts bridge entities from first-pass
retrieval results so the C4 harness can perform the correct two-pass retrieval.

Bridge extraction is runtime-only: it uses only evidence content visible in the
candidate pool, no oracle metadata.

A bridge entity is:
1. Introduced by the evidence (not in the question)
2. Co-occurring with the subject in some record
3. Appearing in fewer than 2 records (unresolved — if it appeared in 2+,
   it would be a resolved link, not a dangling bridge)
4. Only followed up when NO answer-bearing record exists (a record that
   mentions the subject AND contains a numeric value — that would mean the
   first pass already found a candidate answer)

For single-hop tasks, the first pass typically retrieves the fact record
(subject + number), so condition 4 prevents an unnecessary second pass.
For multi-hop tasks, the first pass retrieves the link record (subject +
bridge, no number), so the bridge is followed up.

This mirrors the logic in ``hrm_adaptive_memory.evidence.sufficiency.assess``
but uses a V4-compatible entity extractor for multi-word proper nouns.
"""
from __future__ import annotations

import re
from typing import Mapping

# V4 entity names: Capitalized word + 1-3 lowercase words.
# Excludes sentence-initial non-entities and relation verbs.
_V4_ENTITY = re.compile(r"\b([A-Z][a-z]+(?:\s+[a-z]+){1,3})\b")

_STOP_FIRST = frozenset({
    "The", "During", "It", "Per", "Revision", "Engineering", "As", "For",
    "This", "That", "A", "An", "Which", "What", "Identify",
})

_STOP_LAST = frozenset({
    "is", "was", "are", "were", "has", "have", "had", "to", "for", "by",
    "of", "the", "a", "an", "in", "on", "at", "now", "then", "and", "or",
    "but", "spare", "mounted", "registered", "changed", "updated",
    "allocated", "holds", "assigned", "paired", "applies",
})

_NUMBER = re.compile(r"(?<![\w-])[-+]?\d+(?:\.\d+)?(?![\w-])")

# Patterns that indicate a number is NOT an answer (dates, revision numbers)
_NON_ANSWER_NUMBER = re.compile(
    r"(?:Revision\s+\d+"
    r"|\d{4}-\d{2}-\d{2}"  # ISO dates
    r"|\d{4}\s+\("  # Year followed by parenthesis
    r"|effective\s+\d{4})",
    re.IGNORECASE,
)


def extract_v4_entities(text: str) -> tuple[str, ...]:
    """Extract V4-style entity names from text."""
    candidates = _V4_ENTITY.findall(text)
    result: list[str] = []
    for c in candidates:
        words = c.split()
        if words[0] in _STOP_FIRST:
            continue
        if words[-1] in _STOP_LAST:
            trimmed = " ".join(words[:-1])
            if len(trimmed.split()) >= 2:
                result.append(trimmed)
            continue
        result.append(c)
    return tuple(dict.fromkeys(result))


def _has_answer_numbers(text: str) -> bool:
    """Check if text contains numbers that could be answers (not dates/revisions)."""
    if not _NUMBER.search(text):
        return False
    # If the only numbers are dates/revisions, it's not answer-bearing
    if _NON_ANSWER_NUMBER.search(text):
        # Check if there are numbers OUTSIDE the non-answer patterns
        # Simple heuristic: remove non-answer patterns and check for remaining numbers
        cleaned = _NON_ANSWER_NUMBER.sub("", text)
        return bool(_NUMBER.search(cleaned))
    return True


# V4 link record templates: subject RELATION bridge
# These patterns extract the bridge entity from link records by matching
# the relation verb/phrase that connects subject to bridge.
_LINK_TEMPLATES = [
    # "X is assigned Y" / "X is allocated Y" / "X holds Y"
    re.compile(r"\b" + r"(\S+(?:\s+\S+){1,4}?)" + r"\s+is\s+(?:assigned|allocated\s+to|paired\s+with)\s+(.+?)\.?\s*$"),
    # "subject=X; RELATION=Y" (semicolon-separated)
    re.compile(r"subject\s*=\s*(.+?)\s*;\s*\w+\s*=\s*(.+?)\s*$"),
    # "X: RELATION now Y" / "X: RELATION Y"
    re.compile(r"^(?:-\s*updated\s+)?(.+?):\s*\w+(?:\s+now)?\s+(.+?)\s*$"),
    # "X -> Y" (arrow notation)
    re.compile(r"(.+?)\s+->\s+(.+?)\s*$"),
]


def _template_bridge(content: str, subject: str) -> str | None:
    """Extract bridge using V4 link record templates."""
    subject_lower = subject.lower()
    for pat in _LINK_TEMPLATES:
        m = pat.search(content)
        if m:
            # The bridge is the entity in a group that is NOT the subject
            for group in (m.group(2), m.group(1)):
                group = group.strip().rstrip(".;{}[]")
                ents = extract_v4_entities(group)
                for ent in ents:
                    if ent.lower() != subject_lower:
                        return ent
    return None


def extract_bridge(
    subject: str,
    question: str,
    candidate_ids: tuple[str, ...],
    texts: Mapping[str, str],
) -> str | None:
    """Extract a bridge entity from first-pass retrieval results.

    Returns the bridge entity to follow up on, or None if no follow-up is
    needed (single-hop tasks where the first pass already found an
    answer-bearing record).

    The bridge is chosen deterministically. Template-based extraction from
    link records is tried first (most reliable for V4), falling back to
    co-occurrence-based extraction.
    """
    subject_lower = subject.lower()

    # Check for answer-bearing records: records that mention the subject
    # AND contain a numeric value. If any exist, the first pass may have
    # already found a candidate answer — no bridge follow-up needed.
    has_answer_bearing = False
    for eid in candidate_ids:
        content = texts.get(eid, "")
        if not content:
            continue
        if subject_lower in content.lower() and _has_answer_numbers(content):
            has_answer_bearing = True
            break

    if has_answer_bearing:
        return None

    # Try template-based extraction from link records first (most reliable)
    template_bridges: dict[str, int] = {}
    for eid in candidate_ids:
        content = texts.get(eid, "")
        if not content:
            continue
        if "/identity" in eid:
            continue
        if subject_lower not in content.lower():
            continue
        bridge = _template_bridge(content, subject)
        if bridge:
            template_bridges[bridge] = template_bridges.get(bridge, 0) + 1

    if template_bridges:
        # Return the most frequent template-extracted bridge
        # DETERMINISTIC: sort keys before max to break ties by lexical order
        return max(sorted(template_bridges), key=lambda e: (template_bridges[e], -len(e)))

    # Fall back to co-occurrence-based extraction
    # Extract entities from the question to exclude them
    question_entities = set(e.lower() for e in extract_v4_entities(question))
    question_entities.add(subject_lower)

    # Find bridge candidates: entities co-occurring with subject, appearing
    # in < 2 records (unresolved)
    support: dict[str, int] = {}
    for eid in candidate_ids:
        content = texts.get(eid, "")
        if not content:
            continue
        # Skip identity records — bridges come from link/fact records
        if "/identity" in eid:
            continue
        # Only consider records that mention the subject
        if subject_lower not in content.lower():
            continue

        entities = extract_v4_entities(content)
        for ent in entities:
            ent_lower = ent.lower()
            if ent_lower in question_entities:
                continue
            # Skip abbreviations (handled by identity stage)
            if re.match(r"^[A-Z]+-\d+$", ent):
                continue
            support[ent] = support.get(ent, 0) + 1

    # Bridge entities are unresolved (appear in < 2 records)
    bridges = {ent for ent, count in support.items() if count < 2}

    if not bridges:
        return None

    # Return the most frequent bridge (deterministic tie-break by name)
    # DETERMINISTIC: sort bridges before max to break ties by lexical order
    return max(sorted(bridges), key=lambda e: (support[e], -len(e)))
