"""C4 query stage — reuse the qualified subject-preserving formulation.

The measured defect this addresses: the old follow-up discarded the subject
after finding a bridge, losing +0.400 complete-set@50. The frozen formulation
"subject_bridge_relation" is the development-qualified winner.

This stage does NOT retune query wording. It uses the exact formulation from
InformationState.formulate_followup.
"""
from __future__ import annotations

import hashlib
import re

from ..retrieval.information_state import (
    InformationState, formulate_followup, FOLLOWUP_FORMULATION)
from .contracts import C4Arm, QueryResult

# Subject extraction from question templates (generic English, covers V4 and C3)
_SUBJECT_PATTERNS = [
    re.compile(r"is held by\s+(.+?)\?$"),
    re.compile(r"does\s+(.+?)\s+carry\?$"),
    re.compile(r"applies to\s+(.+?)\?$"),
    re.compile(r"(?:as of\s+.+?,\s*)?which\s+\S.*?\s+applies to\s+(.+?)\?$", re.I),
    re.compile(r"^for\s+(.+?),\s*which\b", re.I),
    re.compile(r"(?:is the)\s+\S.*?\s+for\s+(.+?)\?$"),
    re.compile(r"recorded for\s+(.+?)\.$"),
    re.compile(r"attached to\s+(.+?)\.$"),
    re.compile(r"associated with\s+(.+?)\.$"),
]

# Target relation extraction (covers V4 and C3 templates)
_RELATION_PATTERNS = [
    re.compile(r"^(?:which|what)\s+([a-z]+(?:\s+[a-z]+){0,2}?)\s+(?:is|are|was|does|do)\b"),
    re.compile(r"^(?:which|what)\s+(?:is the\s+)?([a-z]+(?:\s+[a-z]+){0,2}?)\s+(?:applies|for)\b"),
    re.compile(r"^(?:which|what)\s+([a-z]+(?:\s+[a-z]+){0,2}?)\s+(?:is held by|does)\b"),
    re.compile(r"^for\s+.+?,\s*which\s+([a-z]+(?:\s+[a-z]+){0,2}?)\s+(?:is|are|was)\b"),
    re.compile(r"^(?:as of\s+.+?,\s*)?which\s+([a-z]+(?:\s+[a-z]+){0,2}?)\s+(?:applies|is|are|was)\b"),
    re.compile(r"^identify the\s+([a-z]+(?:\s+[a-z]+){0,2}?)\s+associated with\b"),
]


def extract_subject(question: str) -> str:
    """Extract the subject phrase from a generic question template."""
    for pat in _SUBJECT_PATTERNS:
        m = pat.search(question)
        if m:
            return m.group(1).strip().rstrip(".?")
    # Fallback: try "for X?" pattern
    m = re.search(r"\s+for\s+(.+?)\?$", question)
    if m:
        return m.group(1).strip().rstrip(".?")
    return ""


def extract_target_relation(question: str) -> str | None:
    """Extract the target relation from the question."""
    for pat in _RELATION_PATTERNS:
        m = pat.match(question.strip().lower())
        if m:
            return m.group(1).strip()
    return None


def run_query_stage(question: str, arm: C4Arm,
                    identity_canonical: str | None = None) -> tuple[InformationState, QueryResult]:
    """Build the InformationState and render the query.

    For "original" policy (C4-0): use the raw question as-is.
    For "subject_preserving" policy (C4-1+): build InformationState and
    use formulate_followup with the frozen formulation.

    If identity_canonical is provided (from a prior resolution), the state
    carries it, but the original subject is NEVER discarded.
    """
    subject = extract_subject(question)
    relation = extract_target_relation(question) or ""

    if arm.query_policy == "original":
        # C4-0: use the raw question directly
        state = InformationState(subject=subject or "unknown", target_relation=relation)
        rendered = question
    elif arm.query_policy == "subject_preserving":
        state = InformationState(subject=subject or "unknown", target_relation=relation)
        if identity_canonical:
            state = state.with_identity(subject, identity_canonical)
        rendered = formulate_followup(state, formulation=FOLLOWUP_FORMULATION)
    else:
        raise ValueError(f"Unknown query_policy: {arm.query_policy}")

    query_hash = hashlib.sha256(rendered.encode()).hexdigest()
    return state, QueryResult(
        original_question=question,
        rendered_query=rendered,
        query_hash=query_hash,
        query_policy=arm.query_policy,
        query_policy_version=FOLLOWUP_FORMULATION,
    )
