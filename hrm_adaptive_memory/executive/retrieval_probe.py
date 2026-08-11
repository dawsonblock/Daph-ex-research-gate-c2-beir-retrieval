"""Cheap retrieval probe + probe-gated memory, for RETRIEVAL_PROBE_GATE_V1.

Per configs/gate_retrieval_probe_v1_design.json. This module splits the
previously-monolithic A1_USE_CERTIFIED_MEMORY path at exactly one boundary:

    CHEAP PROBE            | ESCALATION-ONLY WORK
    -----------------------+---------------------------------------------
    query rendering        | G2 runtime graph + path enumeration
    C2 retrieval (BM25 +   | S2 ordering
      dense + frozen RRF)  | path-coherent packet composition
    identity/binding stage | the second HRM generation over that packet

Everything left of the bar is what an ACCEPT still pays for; everything
right of it is what an ACCEPT avoids. That is the whole point of the split,
and it is why escalation must CONSUME the probe's retrieval rather than
recompute it -- otherwise the probe becomes a second retrieval treatment and
the study would no longer be testing the frozen memory mechanism.

``run_full_memory_legacy`` is the ORIGINAL monolithic path, ported verbatim
from scripts/run_exec_training_v2_collection.py as it stood at commit
a697cf5. It exists solely so tests can prove the refactor is behaviour-
preserving: probe + forced-escalate must produce a byte-identical packet and
prompt. It is not used by new code paths.
"""
from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from hrm_adaptive_memory.backends import CanonicalRetrievalMode
from hrm_adaptive_memory.c4.contracts import (
    C4_RRF_K, RetrievalResult, SelectionResult)
from hrm_adaptive_memory.c4.endpoint_recognition import k1_entity_bound_exact_completion
from hrm_adaptive_memory.c4.fusion import frozen_rrf
from hrm_adaptive_memory.c4.g2_paths import g2_prefilter
from hrm_adaptive_memory.c4.identity_stage import run_identity_stage
from hrm_adaptive_memory.c4.packet_composition import (
    compose_path_coherent_packet, composed_packet_hash, graph_hash, path_set_hash)
from hrm_adaptive_memory.c4.packet_stage import run_packet_stage
from hrm_adaptive_memory.c4.query_stage import extract_target_relation, run_query_stage
from hrm_adaptive_memory.c4.retrieval_stage import get_cached_backend
from hrm_adaptive_memory.c4.runtime_graph import build_runtime_graph
from hrm_adaptive_memory.c4.selector_v2 import select_s2
from hrm_adaptive_memory.executive.stage_timing import StageTimer
from hrm_adaptive_memory.retrieval_bench.selectors import s0_raw
from hrm_adaptive_memory.retrieval_bench.selectors.chain import s2c_chain_plus_relation


@dataclass(frozen=True)
class RetrievalProbeResult:
    """The cheap probe's output. Carries the EXACT retrieval objects that
    escalation must consume -- not copies rebuilt from them."""
    question: str
    rendered_query: str
    relation: str
    fused: tuple[tuple[str, float], ...]
    pool: tuple[str, ...]
    scores: Mapping[str, float]
    retrieval: RetrievalResult
    identity_status: str
    canonical_subject: str | None
    surface: str | None
    depth: int

    def handoff_hash(self) -> str:
        """Canonical hash of everything escalation inherits from the probe:
        candidate ids, their scores, their ORDER, the retrieval policy, the
        budget/rrf parameters, and the query actually issued.

        Used by the PHASE_0 reuse test to prove escalation consumes this
        exact retrieval rather than silently recomputing one. Scores are
        formatted at fixed precision so the hash is stable across platforms
        without being sensitive to float repr differences."""
        parts = [
            f"q={self.question}",
            f"rendered={self.rendered_query}",
            f"relation={self.relation}",
            f"policy={self.retrieval.retrieval_policy}",
            f"budget={self.retrieval.candidate_budget}",
            f"rrf_k={self.retrieval.rrf_k}",
            "candidates=" + "|".join(self.retrieval.candidate_ids),
            "fusion=" + "|".join(f"{eid}:{score:.12g}" for eid, score in self.retrieval.fusion_ranked),
        ]
        return hashlib.sha256("\n".join(parts).encode()).hexdigest()


@dataclass
class MemoryResult:
    """Output of the escalation-only work."""
    packet_ids: tuple[str, ...]
    prompt: Any
    packet: Any
    graph_hash: str
    path_set_hash: str
    composed_packet_hash: str
    identity_status: str
    consumed_handoff_hash: str = ""
    timings: dict = field(default_factory=dict)


def _retrieve(question: str, arm, records, texts, depth: int):
    """The exact retrieval logic from the pre-refactor A1 path."""
    if records:
        if extract_target_relation(question):
            _s, qr = run_query_stage(question, arm)
            rendered_query = qr.rendered_query
        else:
            rendered_query = question
        bm = get_cached_backend(CanonicalRetrievalMode.BM25, records)
        bg = get_cached_backend(CanonicalRetrievalMode.DENSE_BGE, records)
        a_ids = [e.evidence_id for e in asyncio.run(bm.search(rendered_query, k=depth)).evidence]
        b_ids = [e.evidence_id for e in asyncio.run(bg.search(rendered_query, k=depth)).evidence]
        fused = frozen_rrf([a_ids, b_ids], C4_RRF_K, depth)
        pool = [e for e, _ in fused[:depth]]
        scores = dict(fused[:depth])
    else:
        rendered_query = question
        fused, pool, scores = [], [], {}

    relation = extract_target_relation(question) or ""
    retrieval = RetrievalResult(
        candidate_ids=tuple(pool), candidate_budget=depth,
        retrieval_policy=arm.retrieval_policy, bm25_backend="bm25",
        bge_model_id="", bge_revision="", rrf_k=C4_RRF_K,
        bm25_ranked=(), bge_ranked=(), fusion_ranked=tuple(fused[:depth]))
    return rendered_query, relation, fused, pool, scores, retrieval


def run_retrieval_probe(question: str, arm, records, texts: Mapping[str, str],
                        depth: int, timer: StageTimer | None = None) -> RetrievalProbeResult:
    """The CHEAP stage: retrieval + identity binding. Runs NO G2, NO path
    enumeration, NO composition, NO generation.

    When a timer is supplied the two sub-stages are recorded SEPARATELY
    (T_probe_retrieval, T_probe_identity_binding) so the probe's own cost can
    be attributed rather than reported as one opaque number. Timing is
    strictly observational: the timed and untimed paths execute the same
    calls in the same order and return equal results (asserted by
    tests/unit/test_retrieval_probe.py::TestInstrumentationIsNotATreatment).
    """
    if timer is None:
        rendered_query, relation, fused, pool, scores, retrieval = _retrieve(
            question, arm, records, texts, depth)
        ident = run_identity_stage(question, arm, retrieval, texts)
    else:
        with timer.stage("T_probe_retrieval"):
            rendered_query, relation, fused, pool, scores, retrieval = _retrieve(
                question, arm, records, texts, depth)
        with timer.stage("T_probe_identity_binding"):
            ident = run_identity_stage(question, arm, retrieval, texts)

    return RetrievalProbeResult(
        question=question, rendered_query=rendered_query, relation=relation,
        fused=tuple(fused[:depth]), pool=tuple(pool), scores=scores,
        retrieval=retrieval, identity_status=ident.status,
        canonical_subject=ident.canonical, surface=ident.surface, depth=depth)


def _s2_order(working_set, question, scores, arm, texts):
    """Verbatim from the pre-refactor A1 path."""
    if not working_set:
        return []
    ident = run_identity_stage(question, arm, RetrievalResult(
        candidate_ids=tuple(working_set), candidate_budget=len(working_set),
        retrieval_policy=arm.retrieval_policy, bm25_backend="bm25",
        bge_model_id="", bge_revision="", rrf_k=C4_RRF_K,
        bm25_ranked=(), bge_ranked=(), fusion_ranked=()), texts)
    rq = question
    if ident.surface and ident.canonical:
        rq = rq.replace(ident.surface, ident.canonical)
    cds = [{"document_id": e} for e in working_set]
    allowed = set(working_set)

    def fz(bb):
        out = (s2c_chain_plus_relation(cds, budget=bb, question=rq, texts=texts)
               if ident.status in ("EXACT", "RESOLVED") and ident.canonical
               else s0_raw(cds, budget=bb))
        return [c for c in out
                if (c["document_id"] if isinstance(c, dict) else c) in allowed]

    selected, _r, _d = select_s2(
        identity_status=ident.status, question=question,
        canonical_subject=ident.canonical, candidate_ids=working_set,
        texts=texts, budget=len(working_set), frozen_select=fz,
        fusion_scores=scores)
    return [x["document_id"] if isinstance(x, dict) else x for x in selected]


def run_full_memory_from_probe(probe: RetrievalProbeResult, arm, texts: Mapping[str, str],
                               working_set_size: int, packet_budget: int,
                               selector_label: str = "retrieval_probe_v1",
                               timer: StageTimer | None = None) -> MemoryResult:
    """The ESCALATION-ONLY work, consuming the probe's retrieval directly.

    Every retrieval-derived input below comes from ``probe`` -- pool, scores,
    relation, canonical subject, and the RetrievalResult handed to
    run_packet_stage. Retrieval is NOT recomputed here; that invariant is
    what tests/unit/test_retrieval_probe.py asserts via handoff_hash().
    """
    pool = list(probe.pool)

    def _g2():
        g2r = g2_prefilter(candidate_ids=pool, texts=texts,
                           canonical_subject=probe.canonical_subject,
                           relation=probe.relation,
                           working_set_size=working_set_size,
                           fusion_scores=probe.scores,
                           completion_fn=k1_entity_bound_exact_completion)
        return g2r.kept, [p for p in g2r.all_paths if p.complete]

    if timer is not None:
        with timer.stage("T_G2"):
            g2_ws, complete_paths = _g2()
    else:
        g2_ws, complete_paths = _g2()

    def _compose():
        order = _s2_order(g2_ws, probe.question, probe.scores, arm, texts)
        packet_ids = compose_path_coherent_packet(
            complete_paths=complete_paths, s2_ordering=order,
            working_set=g2_ws, packet_budget=packet_budget).packet
        g2_graph = build_runtime_graph(record_ids=pool, texts=texts, relation=probe.relation)
        selection = SelectionResult(
            selector=selector_label, selected_ids=tuple(packet_ids),
            selector_policy="A1_USE_CERTIFIED_MEMORY", identity_status=probe.identity_status)
        prompt, packet = run_packet_stage(arm, probe.question, selection, texts, probe.retrieval)
        return packet_ids, prompt, packet, graph_hash(g2_graph), path_set_hash(complete_paths)

    if timer is not None:
        with timer.stage("T_composition"):
            packet_ids, prompt, packet, ghash, pshash = _compose()
    else:
        packet_ids, prompt, packet, ghash, pshash = _compose()

    return MemoryResult(
        packet_ids=tuple(packet_ids), prompt=prompt, packet=packet,
        graph_hash=ghash, path_set_hash=pshash,
        composed_packet_hash=composed_packet_hash(tuple(packet_ids)),
        identity_status=probe.identity_status,
        consumed_handoff_hash=probe.handoff_hash(),
        timings=timer.as_dict() if timer is not None else {})


def run_full_memory_legacy(question: str, arm, records, texts: Mapping[str, str],
                           depth: int, working_set_size: int, packet_budget: int,
                           selector_label: str = "retrieval_probe_v1") -> MemoryResult:
    """The ORIGINAL monolithic A1 path (pre-refactor, commit a697cf5), ported
    verbatim. Retained ONLY as the reference behaviour for the PHASE_0
    equivalence regression -- new code must not call this."""
    rendered_query, relation, fused, pool, scores, retrieval = _retrieve(
        question, arm, records, texts, depth)
    ident = run_identity_stage(question, arm, retrieval, texts)

    g2r = g2_prefilter(candidate_ids=pool, texts=texts,
                       canonical_subject=ident.canonical, relation=relation,
                       working_set_size=working_set_size, fusion_scores=scores,
                       completion_fn=k1_entity_bound_exact_completion)
    g2_ws = g2r.kept
    complete_paths = [p for p in g2r.all_paths if p.complete]

    order = _s2_order(g2_ws, question, scores, arm, texts)
    packet_ids = compose_path_coherent_packet(
        complete_paths=complete_paths, s2_ordering=order,
        working_set=g2_ws, packet_budget=packet_budget).packet
    g2_graph = build_runtime_graph(record_ids=pool, texts=texts, relation=relation)
    selection = SelectionResult(
        selector=selector_label, selected_ids=tuple(packet_ids),
        selector_policy="A1_USE_CERTIFIED_MEMORY", identity_status=ident.status)
    prompt, packet = run_packet_stage(arm, question, selection, texts, retrieval)

    return MemoryResult(
        packet_ids=tuple(packet_ids), prompt=prompt, packet=packet,
        graph_hash=graph_hash(g2_graph), path_set_hash=path_set_hash(complete_paths),
        composed_packet_hash=composed_packet_hash(tuple(packet_ids)),
        identity_status=ident.status)


# --- probe features -------------------------------------------------------

#: Feature names this probe emits. Every one must be registered in
#: hrm_adaptive_memory.experiment_integrity.executive_features.KNOWN_FEATURES
#: at AvailabilityStage.POST_CHEAP_RETRIEVAL_PROBE_PRE_MEMORY, and
#: tests/unit/test_retrieval_probe.py asserts that correspondence both ways.
PROBE_FEATURE_NAMES = (
    "probe_top1_retrieval_score",
    "probe_topk_mean_retrieval_score",
    "probe_retrieval_score_margin",
    "probe_candidate_count",
    "probe_identity_binding_status_code",
    "probe_relation_extracted",
)

_BINDING_CODE = {"UNRESOLVED": 0, "AMBIGUOUS": 1, "RESOLVED": 2, "EXACT": 3}


def retrieval_probe_features(probe: RetrievalProbeResult, topk: int = 5) -> dict[str, float]:
    """Extract the admissible decision-state features from the cheap probe.

    Computable strictly after retrieval+binding and strictly before G2 --
    nothing here touches paths, graph reachability, packet completeness,
    selected evidence, or any generation result.
    """
    ranked = list(probe.retrieval.fusion_ranked)
    scores = [s for _eid, s in ranked]
    top1 = float(scores[0]) if scores else 0.0
    top2 = float(scores[1]) if len(scores) > 1 else 0.0
    head = scores[:topk]
    return {
        "probe_top1_retrieval_score": top1,
        "probe_topk_mean_retrieval_score": float(sum(head) / len(head)) if head else 0.0,
        "probe_retrieval_score_margin": top1 - top2,
        "probe_candidate_count": float(len(probe.retrieval.candidate_ids)),
        "probe_identity_binding_status_code": float(_BINDING_CODE.get(probe.identity_status, 0)),
        "probe_relation_extracted": 1.0 if probe.relation else 0.0,
    }
