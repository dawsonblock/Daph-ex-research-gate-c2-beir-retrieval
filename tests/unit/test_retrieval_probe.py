"""PHASE_0 hard gates for RETRIEVAL_PROBE_GATE_V1.

Per configs/gate_retrieval_probe_v1_design.json and the research lead's five
implementation requirements. Every test here is a structural correctness
check that needs no GPU -- PHASE_0 is decidable entirely on CPU.

The load-bearing test is TestProbeEquivalence: it runs the ORIGINAL
monolithic memory path and the new probe -> forced-ESCALATE path over the
same real b3 corpus and asserts the resulting packet and prompt are
identical. That is what proves this change ADDED SENSING rather than
accidentally altering the certified reasoning mechanism.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import hrm_adaptive_memory.evaluation  # noqa: E402,F401  (cycle-breaker)

from hrm_adaptive_memory.c4.arms import ARMS  # noqa: E402
from hrm_adaptive_memory.executive.retrieval_probe import (  # noqa: E402
    PROBE_FEATURE_NAMES, retrieval_probe_features, run_full_memory_from_probe,
    run_full_memory_legacy, run_retrieval_probe)
from hrm_adaptive_memory.executive.stage_timing import StageTimer, timed  # noqa: E402
from hrm_adaptive_memory.experiment_integrity.executive_features import (  # noqa: E402
    KNOWN_FEATURES, STAGES_ADMISSIBLE_FOR_ANSWER_VS_MEMORY, AvailabilityStage,
    FeatureAvailabilityError, require_admissible_for_answer_vs_memory)
from scripts.run_gate_c4 import _to_index_records as to_index_records  # noqa: E402

SUITE = ROOT / "data/hrm/exec_training_v2/MEMORY_required/exec2_700"
M = 50
PACKET = 6
DEPTH = 105  # c2(700) = ceil(0.15 * 700)


def _load_corpus(limit_tasks: int = 6):
    tasks = [json.loads(l) for l in (SUITE / "oracle_tasks.jsonl").read_text().splitlines() if l.strip()]
    evidence = [json.loads(l) for l in (SUITE / "evidence.jsonl").read_text().splitlines() if l.strip()]
    texts = {r["evidence_id"]: r["content"] for r in evidence}
    return tasks[:limit_tasks], to_index_records(evidence), texts


pytestmark = pytest.mark.skipif(
    not (SUITE / "oracle_tasks.jsonl").is_file(),
    reason="exec_training_v2 suite not present")


class TestProbeEquivalence:
    """Gate: probe -> forced ESCALATE must reproduce the pre-refactor path
    exactly, apart from timing/provenance fields."""

    def test_packet_and_prompt_identical_to_legacy_path(self):
        tasks, records, texts = _load_corpus()
        arm = ARMS["C4_4"]
        assert tasks, "need at least one task"
        for task in tasks:
            q = task["question"]
            legacy = run_full_memory_legacy(q, arm, records, texts, DEPTH, M, PACKET)
            probe = run_retrieval_probe(q, arm, records, texts, DEPTH)
            new = run_full_memory_from_probe(probe, arm, texts, M, PACKET)

            assert new.packet_ids == legacy.packet_ids, f"packet diverged for {task['task_id']}"
            assert new.packet.prompt_hash == legacy.packet.prompt_hash, \
                f"prompt hash diverged for {task['task_id']}"
            assert new.packet.membership_hash == legacy.packet.membership_hash
            assert new.packet.order_hash == legacy.packet.order_hash
            assert new.composed_packet_hash == legacy.composed_packet_hash
            assert new.graph_hash == legacy.graph_hash
            assert new.path_set_hash == legacy.path_set_hash
            assert new.identity_status == legacy.identity_status
            assert new.prompt == legacy.prompt


class TestProbeReuseIsExact:
    """Gate: escalation must consume the probe's retrieval, not recompute it."""

    def test_escalation_records_the_probes_handoff_hash(self):
        tasks, records, texts = _load_corpus(limit_tasks=3)
        arm = ARMS["C4_4"]
        for task in tasks:
            probe = run_retrieval_probe(task["question"], arm, records, texts, DEPTH)
            result = run_full_memory_from_probe(probe, arm, texts, M, PACKET)
            assert result.consumed_handoff_hash == probe.handoff_hash()

    def test_handoff_hash_covers_ids_scores_order_policy_and_query(self):
        tasks, records, texts = _load_corpus(limit_tasks=1)
        arm = ARMS["C4_4"]
        probe = run_retrieval_probe(tasks[0]["question"], arm, records, texts, DEPTH)
        base = probe.handoff_hash()

        import dataclasses

        from hrm_adaptive_memory.c4.contracts import RetrievalResult

        r = probe.retrieval
        # reordering the SAME candidate ids must change the hash
        if len(r.candidate_ids) > 1:
            swapped = dataclasses.replace(
                r, candidate_ids=(r.candidate_ids[1], r.candidate_ids[0]) + r.candidate_ids[2:])
            assert dataclasses.replace(probe, retrieval=swapped).handoff_hash() != base
        # perturbing a score must change the hash
        if len(r.fusion_ranked) > 0:
            eid, score = r.fusion_ranked[0]
            bumped = dataclasses.replace(r, fusion_ranked=((eid, score + 1.0),) + r.fusion_ranked[1:])
            assert dataclasses.replace(probe, retrieval=bumped).handoff_hash() != base
        # changing the retrieval policy must change the hash
        other = dataclasses.replace(r, retrieval_policy=r.retrieval_policy + "_X")
        assert dataclasses.replace(probe, retrieval=other).handoff_hash() != base
        # changing the query must change the hash
        assert dataclasses.replace(probe, question=probe.question + " ?").handoff_hash() != base
        assert isinstance(RetrievalResult, type)

    def test_probe_is_deterministic(self):
        tasks, records, texts = _load_corpus(limit_tasks=2)
        arm = ARMS["C4_4"]
        for task in tasks:
            a = run_retrieval_probe(task["question"], arm, records, texts, DEPTH)
            b = run_retrieval_probe(task["question"], arm, records, texts, DEPTH)
            assert a.handoff_hash() == b.handoff_hash()

    def test_escalation_performs_no_second_retrieval(self, monkeypatch):
        """Gate: NO DUPLICATE RETRIEVAL. The probe+escalate flow must issue
        exactly the retrieval the probe issued and not one more -- otherwise
        the probe is a second retrieval treatment and the frozen memory
        mechanism is no longer what is under test."""
        tasks, records, texts = _load_corpus(limit_tasks=3)
        arm = ARMS["C4_4"]
        import hrm_adaptive_memory.executive.retrieval_probe as rp

        calls = {"n": 0}
        real = rp.get_cached_backend

        def counting(*a, **k):
            calls["n"] += 1
            return real(*a, **k)

        monkeypatch.setattr(rp, "get_cached_backend", counting)
        for task in tasks:
            calls["n"] = 0
            probe = run_retrieval_probe(task["question"], arm, records, texts, DEPTH)
            after_probe = calls["n"]
            assert after_probe > 0, "probe should have retrieved"
            run_full_memory_from_probe(probe, arm, texts, M, PACKET)
            assert calls["n"] == after_probe, (
                f"escalation issued {calls['n'] - after_probe} extra retrieval "
                "backend call(s) -- it must consume the probe's retrieval")

    def test_legacy_and_split_paths_issue_the_same_retrieval_count(self, monkeypatch):
        """The split path must not be cheaper OR costlier in retrieval terms
        than the original monolithic path -- same work, different seam."""
        tasks, records, texts = _load_corpus(limit_tasks=2)
        arm = ARMS["C4_4"]
        import hrm_adaptive_memory.executive.retrieval_probe as rp

        calls = {"n": 0}
        real = rp.get_cached_backend

        def counting(*a, **k):
            calls["n"] += 1
            return real(*a, **k)

        monkeypatch.setattr(rp, "get_cached_backend", counting)
        for task in tasks:
            calls["n"] = 0
            run_full_memory_legacy(task["question"], arm, records, texts, DEPTH, M, PACKET)
            legacy_calls = calls["n"]

            calls["n"] = 0
            probe = run_retrieval_probe(task["question"], arm, records, texts, DEPTH)
            run_full_memory_from_probe(probe, arm, texts, M, PACKET)
            assert calls["n"] == legacy_calls


class TestProbeDoesNotRunEscalationWork:
    """Gate: the probe must not touch G2, composition, or generation."""

    def test_probe_never_calls_escalation_stages(self, monkeypatch):
        tasks, records, texts = _load_corpus(limit_tasks=2)
        arm = ARMS["C4_4"]
        import hrm_adaptive_memory.executive.retrieval_probe as rp

        def boom(*_a, **_k):
            raise AssertionError("probe ran escalation-only work")

        monkeypatch.setattr(rp, "g2_prefilter", boom)
        monkeypatch.setattr(rp, "compose_path_coherent_packet", boom)
        monkeypatch.setattr(rp, "build_runtime_graph", boom)
        monkeypatch.setattr(rp, "run_packet_stage", boom)
        for task in tasks:
            run_retrieval_probe(task["question"], arm, records, texts, DEPTH)


class TestFeatureBoundaryEnforcedMechanically:
    def test_new_stage_is_admissible(self):
        assert (AvailabilityStage.POST_CHEAP_RETRIEVAL_PROBE_PRE_MEMORY
                in STAGES_ADMISSIBLE_FOR_ANSWER_VS_MEMORY)

    @pytest.mark.parametrize("stage", [
        AvailabilityStage.POST_RETRIEVAL,
        AvailabilityStage.POST_GRAPH,
        AvailabilityStage.POST_GENERATION,
    ])
    def test_escalation_derived_stages_stay_inadmissible(self, stage):
        assert stage not in STAGES_ADMISSIBLE_FOR_ANSWER_VS_MEMORY

    @pytest.mark.parametrize("name", [
        "graph_reachability", "working_set_size", "n_complete_paths",
        "path_competition_bucket", "structural_competition_ratio",
        "packet_coherence", "cost_already_spent",
        "identity_status", "retrieval_score_margin",
    ])
    def test_forbidden_features_raise(self, name):
        with pytest.raises(FeatureAvailabilityError):
            require_admissible_for_answer_vs_memory(KNOWN_FEATURES[name])

    @pytest.mark.parametrize("name", PROBE_FEATURE_NAMES)
    def test_every_probe_feature_is_registered_and_admissible(self, name):
        assert name in KNOWN_FEATURES, f"{name} emitted but not declared"
        spec = KNOWN_FEATURES[name]
        assert spec.availability_stage == AvailabilityStage.POST_CHEAP_RETRIEVAL_PROBE_PRE_MEMORY
        require_admissible_for_answer_vs_memory(spec)

    def test_no_orphan_probe_features_declared(self):
        declared = {n for n, s in KNOWN_FEATURES.items()
                    if s.availability_stage == AvailabilityStage.POST_CHEAP_RETRIEVAL_PROBE_PRE_MEMORY}
        assert declared == set(PROBE_FEATURE_NAMES), (
            "declared probe features and emitted probe features must match exactly; "
            f"declared-only={declared - set(PROBE_FEATURE_NAMES)} "
            f"emitted-only={set(PROBE_FEATURE_NAMES) - declared}")

    def test_extracted_features_match_declared_names(self):
        tasks, records, texts = _load_corpus(limit_tasks=1)
        arm = ARMS["C4_4"]
        probe = run_retrieval_probe(tasks[0]["question"], arm, records, texts, DEPTH)
        feats = retrieval_probe_features(probe)
        assert set(feats) == set(PROBE_FEATURE_NAMES)
        assert all(isinstance(v, float) for v in feats.values())


class TestStageTiming:
    def test_timed_records_a_positive_monotonic_duration(self):
        store: dict[str, float] = {}
        with timed(store, "T_x"):
            sum(range(10000))
        assert store["T_x"] > 0.0

    def test_stage_timer_totals_and_reports_cuda_flag(self):
        t = StageTimer()
        with t.stage("T_probe_retrieval"):
            sum(range(1000))
        with t.stage("T_G2"):
            sum(range(1000))
        d = t.as_dict()
        assert d["T_total"] == pytest.approx(d["T_probe_retrieval"] + d["T_G2"])
        assert "cuda_synchronized" in d

    def test_required_stage_names_are_declared(self):
        for name in ("T_A0_generation", "T_probe_retrieval", "T_probe_identity_binding",
                     "T_G2", "T_composition", "T_A1_generation"):
            assert name in StageTimer.REQUIRED_STAGES

    def test_probe_and_escalation_populate_their_own_stages(self):
        tasks, records, texts = _load_corpus(limit_tasks=1)
        arm = ARMS["C4_4"]
        timer = StageTimer()
        probe = run_retrieval_probe(tasks[0]["question"], arm, records, texts, DEPTH, timer=timer)
        run_full_memory_from_probe(probe, arm, texts, M, PACKET, timer=timer)
        assert timer.stages["T_probe_retrieval"] > 0.0
        assert timer.stages["T_probe_identity_binding"] >= 0.0
        assert timer.stages["T_G2"] >= 0.0
        assert timer.stages["T_composition"] >= 0.0

    def test_derived_costs_are_consistent_with_atomic_stages(self):
        t = StageTimer()
        t.stages.update({
            "T_A0_generation": 1.0, "T_probe_retrieval": 2.0,
            "T_probe_identity_binding": 0.5, "T_G2": 4.0,
            "T_composition": 8.0, "T_A1_generation": 16.0,
        })
        d = t.derived_costs()
        assert d["T_probe_total"] == pytest.approx(2.5)
        assert d["C_accept"] == pytest.approx(3.5)
        assert d["C_avoided"] == pytest.approx(28.0)
        assert d["C_escalate"] == pytest.approx(31.5)
        # the decision only controls the marginal escalation work
        assert d["C_escalate"] - d["C_accept"] == pytest.approx(d["C_avoided"])
        # true no-probe deployment baselines
        assert d["C_noprobe_answer_only"] == pytest.approx(1.0)
        assert d["C_noprobe_full_memory"] == pytest.approx(30.5)
        # the probe architecture's inherent overhead vs going straight to memory
        assert d["C_escalate"] - d["C_noprobe_full_memory"] == pytest.approx(1.0)


class TestInstrumentationIsNotATreatment:
    """Gate: adding timing must not change ANY output. Only timing and
    provenance fields may differ between the instrumented and uninstrumented
    paths."""

    def test_probe_identical_with_and_without_timer(self):
        tasks, records, texts = _load_corpus(limit_tasks=3)
        arm = ARMS["C4_4"]
        for task in tasks:
            bare = run_retrieval_probe(task["question"], arm, records, texts, DEPTH)
            timed_ = run_retrieval_probe(task["question"], arm, records, texts, DEPTH,
                                         timer=StageTimer())
            assert bare.handoff_hash() == timed_.handoff_hash()
            assert bare.pool == timed_.pool
            assert bare.identity_status == timed_.identity_status
            assert bare.canonical_subject == timed_.canonical_subject
            assert bare.relation == timed_.relation
            assert bare.retrieval.fusion_ranked == timed_.retrieval.fusion_ranked

    def test_escalation_identical_with_and_without_timer(self):
        tasks, records, texts = _load_corpus(limit_tasks=3)
        arm = ARMS["C4_4"]
        for task in tasks:
            probe = run_retrieval_probe(task["question"], arm, records, texts, DEPTH)
            bare = run_full_memory_from_probe(probe, arm, texts, M, PACKET)
            timed_ = run_full_memory_from_probe(probe, arm, texts, M, PACKET,
                                                timer=StageTimer())
            assert bare.packet_ids == timed_.packet_ids
            assert bare.packet.prompt_hash == timed_.packet.prompt_hash
            assert bare.packet.membership_hash == timed_.packet.membership_hash
            assert bare.packet.order_hash == timed_.packet.order_hash
            assert bare.composed_packet_hash == timed_.composed_packet_hash
            assert bare.graph_hash == timed_.graph_hash
            assert bare.path_set_hash == timed_.path_set_hash
            assert bare.prompt == timed_.prompt

    def test_timing_wrapper_returns_inner_result_unchanged(self):
        """The A0 generation timer wraps a call; it must pass the result
        through untouched. Verified with a stub so no model is required."""
        sentinel = object()
        timer = StageTimer()

        def inner():
            return sentinel

        with timer.stage("T_A0_generation"):
            got = inner()
        assert got is sentinel
        assert timer.stages["T_A0_generation"] >= 0.0
