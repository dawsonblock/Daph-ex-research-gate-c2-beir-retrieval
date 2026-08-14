"""K3: the construction ceiling. EVALUATOR-ONLY -- never import this from
runtime code (endpoint_recognition.py, g2_paths.py, runtime_graph.py,
relation_grammar.py). Import direction is runtime <- evaluator, never the
reverse; tests/unit/test_g2_endpoint_recognition.py asserts this mechanically.

What K3 answers, and what it deliberately does not answer
-----------------------------------------------------------
K3 answers a narrow question: "given a record the runtime graph ALREADY
discovered while walking from the subject (hop 0, or hop 1 through some
bridge), is this record actually one of the task's required_evidence_ids?"

It does NOT answer "construct the evaluator's gold path" -- it never adds a
record, entity, or edge that K0/K1's topology pass did not already reach. That
restriction is what makes K3 a construction-semantics ceiling rather than a
full graph oracle: if K3 also fails to beat K0 by much, the defect is that the
runtime graph never REACHES the required record in the first place (topology),
not that it fails to recognize a reached record as complete (recognition).
"""
from __future__ import annotations

import hashlib
from typing import Callable, Mapping

from .endpoint_recognition import EndpointRecognition, PARSER_VERSION
from ..retrieval.canonicalization import _norm


def _span_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def make_oracle_completion_fn(
    *, required_evidence_ids: frozenset[str], topology_record_ids: frozenset[str],
) -> Callable[..., EndpointRecognition]:
    """Build a K3 completion function with the same call signature as
    K0/K1/K2 (record_id, entity, relation, texts) -> EndpointRecognition, so
    g2_paths.py's traversal/ranking code stays identical across all four modes
    and only the completion predicate differs.

    ``topology_record_ids`` must be the record ids the SHARED hop-0/hop-1 pass
    already discovered (subject_records union every bridge's bridge_records) --
    passing the full candidate pool here would let K3 answer a different,
    easier question ("is this record required at all") instead of the correct
    one ("did the runtime graph reach this required record").
    """
    def _oracle_completion(
        *, record_id: str, entity: str, relation: str, texts: Mapping[str, str],
    ) -> EndpointRecognition:
        content = texts.get(record_id, "")
        relation_norm = _norm(relation) if relation else ""
        reachable = record_id in topology_record_ids
        required = record_id in required_evidence_ids
        completed = reachable and required
        if not reachable:
            reason = "k3_oracle_record_not_graph_reachable"
        elif not required:
            reason = "k3_oracle_reachable_but_not_required"
        else:
            reason = "k3_oracle_reachable_and_required"
        return EndpointRecognition(
            record_id=record_id, entity_id=_norm(entity), requested_relation=relation_norm,
            canonical_relation=(relation_norm if completed else None),
            entity_bound=reachable, relation_surface_match=completed,
            relation_family_match=completed, completed=completed,
            completion_reason=reason, parser_version=PARSER_VERSION + "-K3-oracle",
            source_span_hash=_span_hash(content))
    return _oracle_completion
