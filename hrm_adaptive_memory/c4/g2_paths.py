"""G2: bounded runtime-graph path enumeration, competition, and ranking.

Distinguishing hypothesis vs G1_TYPED_PATH
-------------------------------------------
G1_TYPED_PATH walks ONE frozen template (subject -> closing bridge -> relation)
and represents the result as a per-record tier, then fills any remaining
capacity with retrieval-rank order. G2 instead:

  1. enumerates every distinct subject-anchored path up to the frozen 2-hop
     bound as an explicit, inspectable ``PathCandidate`` -- not a tier label;
  2. treats distinct bridge entities reaching the same relation as competing
     HYPOTHESES to be ranked against each other, not admitted wholesale;
  3. never pads. The working set is the union of records that ranked-and-
     selected paths actually justify, up to a hard ceiling M -- if 17 records
     are justified, 17 are returned, not M.

That third point is a direct correction: G1_TYPED_PATH filled 41-46 of 50 M50
slots with retrieval-rank filler, which is exactly how pressure re-entered the
working set after compression. G2 does not do that.

Ranking is lexicographic, not a weighted sum
---------------------------------------------
P2 (B4's coarse prefilter) failed by trusting a single scalar -- retrieval
score -- once structure had done its coarse filtering, and bridge rank is the
known weak signal. A dozen hand-tuned weights would be the same mistake with
more knobs. So ranking here is tiered: subject-anchored, reaches the target
relation, path completeness, hop count, path diversity (does this path add
records the working set doesn't already have), and retrieval rank ONLY as the
final tie-break. No learned parameters, no evaluator labels.

Runtime safety
--------------
Every function here takes ``candidate_ids``/``texts``/``canonical_subject``/
``relation`` -- never a task or evaluator object -- and reads only what the
runtime graph already exposes (visible entity mentions, visible relation
matches, retrieval score/rank). Enforced by the same leakage scan used around
selector_v2 and runtime_graph.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from ..retrieval.canonicalization import _norm
from .endpoint_recognition import EndpointRecognition, k0_literal_completion
from .runtime_graph import RuntimeGraph, build_runtime_graph

#: (record_id, entity, relation, texts) -> EndpointRecognition. Defaults to
#: k0_literal_completion so callers that don't pass one get exactly G2-v1's
#: frozen historical behavior -- see configs/gate_g2_v2_path_completion.json.
CompletionFn = Callable[..., EndpointRecognition]

#: Structural tiers, frozen. Lower sorts first (higher priority).
TIER_DIRECT_COMPLETE = 0     # subject record itself expresses the relation
TIER_BRIDGED_COMPLETE = 1    # subject -> bridge -> bridge record expresses relation
TIER_DIRECT_PARTIAL = 2      # subject record, relation not confirmed there
TIER_BRIDGED_PARTIAL = 3     # subject -> bridge, relation not confirmed at bridge

#: A path is dropped from the retained set if this fraction of its records are
#: already present in the working set -- i.e. it would add nothing new. This is
#: the diversity criterion, applied as a real admission test rather than a
#: ranking-score component, so a path either earns its place or it doesn't.
FULLY_REDUNDANT_OVERLAP = 0.999


def _path_id(anchor: str, terminal: str, entity_ids: Sequence[str]) -> str:
    payload = "|".join(("path", anchor, terminal, *entity_ids))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class PathCandidate:
    """One explicit, inspectable structural hypothesis.

    Kept as a full object -- not reduced to a single score -- so that a G2
    failure is attributable to construction (paths never formed) vs ranking
    (paths formed but the wrong ones were kept).
    """
    path_id: str
    anchor_entity: str
    terminal_entity: str
    target_relation: str
    hop_count: int
    entity_ids: tuple[str, ...]
    record_ids: tuple[str, ...]
    edge_ids: tuple[str, ...]
    complete: bool
    subject_anchored: bool
    tier: int
    retrieval_score: float
    provenance: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "path_id": self.path_id, "anchor_entity": self.anchor_entity,
            "terminal_entity": self.terminal_entity,
            "target_relation": self.target_relation,
            "hop_count": self.hop_count, "entity_ids": list(self.entity_ids),
            "record_ids": list(self.record_ids), "edge_ids": list(self.edge_ids),
            "complete": self.complete, "subject_anchored": self.subject_anchored,
            "tier": self.tier, "retrieval_score": self.retrieval_score,
            "provenance": self.provenance,
        }


def _edges_between(graph: RuntimeGraph, record_ids: Sequence[str],
                   entity_ids: Sequence[str]) -> tuple[str, ...]:
    """Provenance edges supporting this path: only those whose source record
    is part of the path AND whose entities are part of the path's chain."""
    wanted_records = set(record_ids)
    wanted_entities = set(entity_ids)
    out = []
    for edge in graph.edges:
        if edge.source_record_id not in wanted_records:
            continue
        if edge.edge_type == "RECORD_MENTIONS_ENTITY" and edge.target in wanted_entities:
            out.append(edge.edge_id)
        elif edge.edge_type == "RECORD_EXPRESSES_RELATION":
            out.append(edge.edge_id)
        elif edge.edge_type == "ENTITY_LINKED_TO_ENTITY" and \
                edge.source in wanted_entities and edge.target in wanted_entities:
            out.append(edge.edge_id)
    return tuple(sorted(set(out)))


def bridges_for_subject(graph: RuntimeGraph, subject_norm: str) -> set[str]:
    """Entities directly co-mentioned with the (already-normalized) subject in
    some record. Factored out so K3's oracle topology and K0/K1/K2's traversal
    are guaranteed byte-identical rather than two hand-written copies that
    could silently drift apart."""
    bridges: set[str] = set()
    for record_id in graph.records_by_entity.get(subject_norm, set()):
        for entity in graph.entities_by_record.get(record_id, frozenset()):
            if entity and entity != subject_norm:
                bridges.add(entity)
    return bridges


def topology_reachable_records(
    graph: RuntimeGraph, canonical_subject: str | None,
) -> frozenset[str]:
    """Every record id the shared hop-0/hop-1 topology walk can reach from the
    subject, regardless of completion -- subject records plus every bridge's
    records. K3 restricts oracle labeling to this set so it can only relabel
    completion for records the walk already discovers, never invent new
    topology (configs/gate_g2_v2_path_completion.json, K3 definition)."""
    subject = _norm(canonical_subject or "")
    if not subject:
        return frozenset()
    subject_records = set(graph.records_by_entity.get(subject, set()))
    bridges = bridges_for_subject(graph, subject)
    for alias in graph.alias_links.get(subject, set()):
        for record_id in graph.records_by_entity.get(alias, set()):
            for entity in graph.entities_by_record.get(record_id, frozenset()):
                if entity and entity != subject:
                    bridges.add(entity)
    reachable = set(subject_records)
    for bridge in bridges:
        reachable |= graph.records_by_entity.get(bridge, set())
    return frozenset(reachable)


def enumerate_paths(
    *, graph: RuntimeGraph, canonical_subject: str | None, relation: str,
    texts: Mapping[str, str] | None = None,
    fusion_scores: Mapping[str, float] | None = None,
    completion_fn: CompletionFn | None = None,
) -> tuple[list[PathCandidate], list[EndpointRecognition]]:
    """Enumerate every distinct subject-anchored path up to the 2-hop bound.

    Distinct bridge entities produce distinct paths -- they are NOT collapsed
    into one flat neighbourhood the way a coarse prefilter would. Instances
    sharing an (anchor, terminal, entity_chain) signature ARE merged, since
    those are the same structural hypothesis reached via different records.

    ``completion_fn`` is the ONLY thing G2-v2's K0/K1/K2/K3 arms vary -- the
    topology walk below (which records are subject records, which entities are
    bridges, which records connect a bridge to the subject) is identical across
    every arm, per configs/gate_g2_v2_path_completion.json's frozen-unchanged
    list. Returns (paths, endpoint_recognitions) -- the second list is every
    completion decision examined, kept for R_endpoint/R_path instrumentation
    even for records that did NOT complete a path.
    """
    scores = fusion_scores or {}
    texts = texts or {}
    completion_fn = completion_fn or k0_literal_completion
    subject = _norm(canonical_subject or "")
    relation_norm = _norm(relation) if relation else ""
    if not subject:
        return [], []

    subject_records = sorted(graph.records_by_entity.get(subject, set()))
    by_signature: dict[tuple, PathCandidate] = {}
    recognitions: list[EndpointRecognition] = []

    def recognize(record_id: str, entity: str) -> EndpointRecognition:
        recognition = completion_fn(record_id=record_id, entity=entity,
                                    relation=relation, texts=texts)
        recognitions.append(recognition)
        return recognition

    def best_score(record_ids: Sequence[str]) -> float:
        return max((scores.get(rid) or 0.0 for rid in record_ids), default=0.0)

    def upsert(anchor, terminal, entity_ids, record_ids, complete, hop, tier,
              discriminator=None):
        # ``discriminator`` keeps hop-0 records from merging into one blob: each
        # subject record is its own atomic evidence unit, unlike a bridge path
        # where several records may jointly support the SAME structural
        # hypothesis and should be combined.
        signature = (anchor, terminal, entity_ids, discriminator)
        record_ids = tuple(sorted(set(record_ids)))
        existing = by_signature.get(signature)
        if existing is not None:
            merged_records = tuple(sorted(set(existing.record_ids) | set(record_ids)))
            by_signature[signature] = PathCandidate(
                path_id=existing.path_id, anchor_entity=anchor,
                terminal_entity=terminal, target_relation=relation_norm,
                hop_count=hop, entity_ids=entity_ids, record_ids=merged_records,
                edge_ids=_edges_between(graph, merged_records, entity_ids),
                complete=existing.complete or complete,
                subject_anchored=True,
                tier=min(existing.tier, tier if complete else existing.tier),
                retrieval_score=max(existing.retrieval_score, best_score(record_ids)),
                provenance={"merged_instances": existing.provenance.get(
                    "merged_instances", 1) + 1})
            return
        by_signature[signature] = PathCandidate(
            path_id=_path_id(anchor, terminal,
                             entity_ids + ((discriminator,) if discriminator else ())),
            anchor_entity=anchor, terminal_entity=terminal,
            target_relation=relation_norm, hop_count=hop, entity_ids=entity_ids,
            record_ids=record_ids,
            edge_ids=_edges_between(graph, record_ids, entity_ids),
            complete=complete, subject_anchored=True,
            tier=tier if complete else tier, retrieval_score=best_score(record_ids),
            provenance={"merged_instances": 1})

    # --- hop 0: the subject's own records -----------------------------------
    # Each subject record is a distinct atomic path (discriminator=record_id):
    # a record that merely mentions the subject without expressing the
    # relation is a different, weaker hypothesis than one that does, and two
    # different relation-expressing subject records are themselves competing
    # alternatives, not the same path re-supported.
    for record_id in subject_records:
        complete = recognize(record_id, subject).completed
        upsert(subject, subject, (subject,), (record_id,), complete,
              hop=0, tier=TIER_DIRECT_COMPLETE if complete else TIER_DIRECT_PARTIAL,
              discriminator=record_id)

    # --- hop 1: entities visible alongside the subject ("bridges") ---------
    bridges = bridges_for_subject(graph, subject)
    for alias in graph.alias_links.get(subject, set()):
        for record_id in sorted(graph.records_by_entity.get(alias, set())):
            for entity in graph.entities_by_record.get(record_id, frozenset()):
                if entity and entity != subject:
                    bridges.add(entity)

    for bridge in sorted(bridges):
        bridge_records = graph.records_by_entity.get(bridge, set())
        if not bridge_records:
            continue
        # The connecting record (subject co-mentioned with this bridge) and the
        # endpoint record (bridge co-occurring with the relation) are two
        # different roles in the SAME coherent path. A complete path needs
        # both: the connection to the subject, and the relation it resolves to.
        link_records = [r for r in subject_records
                        if bridge in graph.entities_by_record.get(r, frozenset())]
        endpoint_records = sorted(
            r for r in bridge_records if recognize(r, bridge).completed)
        complete = bool(endpoint_records)
        chosen = sorted(set(link_records) | set(endpoint_records)) if complete \
            else sorted(set(link_records) or bridge_records)
        upsert(subject, bridge, (subject, bridge), chosen, complete,
              hop=1, tier=TIER_BRIDGED_COMPLETE if complete else TIER_BRIDGED_PARTIAL)

    return list(by_signature.values()), recognitions


@dataclass
class G2Result:
    """Working set plus every diagnostic needed to classify a G2 outcome."""
    kept: list[str] = field(default_factory=list)
    working_set_size: int = 0
    input_size: int = 0
    retained_paths: list[PathCandidate] = field(default_factory=list)
    all_paths: list[PathCandidate] = field(default_factory=list)
    endpoint_recognitions: list[EndpointRecognition] = field(default_factory=list)
    canonical_subject_found: bool = False
    target_relation_extracted: bool = False
    graph_stats: dict[str, int] = field(default_factory=dict)

    def diagnostics(self) -> dict[str, Any]:
        complete = [p for p in self.all_paths if p.complete]
        bridges = {p.terminal_entity for p in self.all_paths if p.hop_count == 1}
        retained_complete = [p for p in self.retained_paths if p.complete]
        overlap_skipped = len(complete) - len(retained_complete) \
            if len(complete) >= len(retained_complete) else 0
        return {
            "canonical_subject_found": int(self.canonical_subject_found),
            "target_relation_extracted": int(self.target_relation_extracted),
            "paths_enumerated": len(self.all_paths),
            "paths_complete": len(complete),
            "paths_target_relation_compatible": len(complete),
            "paths_subject_anchored": sum(1 for p in self.all_paths if p.subject_anchored),
            "paths_competing": len(complete),
            "paths_retained": len(self.retained_paths),
            "unique_bridge_entities": len(bridges),
            "records_per_retained_path": (
                round(sum(len(p.record_ids) for p in self.retained_paths)
                     / len(self.retained_paths), 3) if self.retained_paths else 0.0),
            "path_overlap_ratio": (round(overlap_skipped / len(complete), 3)
                                   if complete else 0.0),
            "working_set_size": self.working_set_size,
            "records_examined": self.input_size,
            "endpoints_inspected": len(self.endpoint_recognitions),
            "endpoints_entity_bound": sum(1 for r in self.endpoint_recognitions
                                          if r.entity_bound),
            "endpoints_recognized_complete": sum(1 for r in self.endpoint_recognitions
                                                 if r.completed),
            **{f"graph_{k}": v for k, v in self.graph_stats.items()},
        }


def rank_and_select_paths(
    paths: Sequence[PathCandidate], working_set_size: int,
) -> tuple[list[PathCandidate], list[str]]:
    """Frozen lexicographic ranking, then greedy non-redundant admission.

    Sort key, in order: tier, hop_count, retrieval score (desc), then a
    deterministic tie-break on the record-id tuple. Diversity is enforced as
    an admission test during the greedy walk, not a ranking term: a path is
    skipped (not retained) if it would add no record the working set doesn't
    already have. Paths are admitted as whole units -- a path that would
    overflow the ceiling is not split into partial records, since a partial
    path is not the coherent unit the hypothesis is about.
    """
    ordered = sorted(
        paths,
        key=lambda p: (p.tier, p.hop_count, -p.retrieval_score, p.record_ids))
    retained: list[PathCandidate] = []
    working: list[str] = []
    seen: set[str] = set()
    for path in ordered:
        if len(working) >= working_set_size:
            break
        new_records = [r for r in path.record_ids if r not in seen]
        overlap = 1.0 - (len(new_records) / len(path.record_ids)
                         if path.record_ids else 0.0)
        if overlap >= FULLY_REDUNDANT_OVERLAP:
            continue
        if len(working) + len(new_records) > working_set_size:
            continue
        retained.append(path)
        for record_id in new_records:
            working.append(record_id)
            seen.add(record_id)
    return retained, working


def g2_prefilter(
    *, candidate_ids: Sequence[str], texts: Mapping[str, str],
    canonical_subject: str | None, relation: str, working_set_size: int,
    fusion_scores: Mapping[str, float] | None = None,
    graph: RuntimeGraph | None = None,
    completion_fn: CompletionFn | None = None,
) -> G2Result:
    """Compress ``candidate_ids`` to at most ``working_set_size`` records.

    M is a CEILING, not a target: if fewer records are structurally justified,
    fewer are returned. No retrieval-rank filler is added to pad the set.

    ``completion_fn`` selects the G2-v2 arm (K0/K1/K2/K3); defaults to K0,
    G2-v1's frozen literal-match behavior, if omitted.
    """
    pool = list(candidate_ids)
    graph = graph or build_runtime_graph(
        record_ids=pool, texts=texts, relation=relation)
    all_paths, recognitions = enumerate_paths(
        graph=graph, canonical_subject=canonical_subject, relation=relation,
        texts=texts, fusion_scores=fusion_scores, completion_fn=completion_fn)
    retained, working = rank_and_select_paths(all_paths, working_set_size)
    return G2Result(
        kept=working, working_set_size=len(working), input_size=len(pool),
        retained_paths=retained, all_paths=all_paths,
        endpoint_recognitions=recognitions,
        canonical_subject_found=bool(canonical_subject),
        target_relation_extracted=bool(relation), graph_stats=graph.stats())
