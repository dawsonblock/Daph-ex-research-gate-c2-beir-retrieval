"""Pool-artifact persistence and the frozen B3 protocol.

Two things are pinned here.

First, the candidate pool must be persisted BY THE RUN. Confirmation #1's
failure decomposition needed per-task availability and the receipts did not
carry pools, so availability had to be reconstructed by replay -- which then
could not be validated across platforms. Persisting the pool inline makes every
later decomposition artifact analysis instead of retrieval replay.

Second, the two candidate hashes must stay distinct. Collapsing them is what let
benign cross-platform ordering churn look like a validity failure, and what made
"selections reproduced" look like sufficient evidence of pool equality when it
is not.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
B3_PROTOCOL = ROOT / "configs/gate_b3_candidate_budget_v1.json"
POOLS = ROOT / "evidence/gate_b3/pools/confirmation_candidate_pools.json"
RECEIPTS = (ROOT / "evidence/gate_c4/diagnosis/"
            "development_c5_Jladder_dry.receipts.jsonl")

_spec = importlib.util.spec_from_file_location(
    "_c5_ladder_pools", ROOT / "scripts/run_c5_integrated_ladder.py")
lad = importlib.util.module_from_spec(_spec)
sys.modules["_c5_ladder_pools"] = lad
_spec.loader.exec_module(lad)


class TestTheTwoHashesAreDistinct:
    def test_order_hash_is_order_sensitive(self):
        from hrm_adaptive_memory.c4.packet_ordering import (
            canonical_candidate_pool_hash as order_hash)
        assert order_hash(["a", "b", "c"]) != order_hash(["c", "b", "a"])

    def test_membership_hash_is_order_independent(self):
        from hrm_adaptive_memory.c4.packet_ordering import (
            canonical_candidate_membership_hash as membership_hash)
        assert membership_hash(["a", "b", "c"]) == membership_hash(["c", "b", "a"])

    def test_membership_hash_still_detects_a_real_membership_change(self):
        """Order-independence must not mean insensitivity."""
        from hrm_adaptive_memory.c4.packet_ordering import (
            canonical_candidate_membership_hash as membership_hash)
        assert membership_hash(["a", "b", "c"]) != membership_hash(["a", "b", "d"])

    def test_they_are_not_the_same_function(self):
        from hrm_adaptive_memory.c4.packet_ordering import (
            canonical_candidate_membership_hash, canonical_candidate_pool_hash)
        assert canonical_candidate_membership_hash is not canonical_candidate_pool_hash


class TestEnvironmentFingerprint:
    def test_records_platform_identity(self):
        fingerprint = lad.environment_fingerprint()
        assert fingerprint["python"] and fingerprint["platform"]

    def test_records_device_so_a_replay_knows_where_it_ran(self):
        """Cross-platform bit-reproducibility is NOT guaranteed, so a replay
        must be able to tell whether it is on the scoring platform."""
        fingerprint = lad.environment_fingerprint()
        assert "device_name" in fingerprint or "torch_error" in fingerprint


@pytest.mark.skipif(not RECEIPTS.is_file(), reason="ladder receipts absent")
class TestReceiptsPersistThePool:
    @pytest.fixture(scope="class")
    def receipt(self):
        rows = [json.loads(l) for l in RECEIPTS.read_text().splitlines() if l.strip()]
        return rows[0]

    def test_every_arm_carries_the_ordered_pool(self, receipt):
        for arm, entry in receipt["arms"].items():
            assert entry["candidate_ids_ordered"], arm
            assert len(entry["candidate_ids_ordered"]) > 1, arm

    def test_every_arm_carries_ranks_and_scores(self, receipt):
        """Rank and score are needed to diagnose WHY evidence missed the pool,
        not merely that it did."""
        for arm, entry in receipt["arms"].items():
            ranked = entry["candidate_ranked"]
            assert len(ranked) == len(entry["candidate_ids_ordered"]), arm
            assert ranked[0]["fusion_rank"] == 1, arm
            assert {"record_id", "fusion_rank", "fusion_score"} <= set(ranked[0]), arm

    def test_ranked_order_agrees_with_the_ordered_pool(self, receipt):
        for arm, entry in receipt["arms"].items():
            assert [r["record_id"] for r in entry["candidate_ranked"]] == \
                entry["candidate_ids_ordered"], arm

    def test_both_hashes_are_persisted_and_self_consistent(self, receipt):
        from hrm_adaptive_memory.c4.packet_ordering import (
            canonical_candidate_membership_hash, canonical_candidate_pool_hash)
        for arm, entry in receipt["arms"].items():
            pool = entry["candidate_ids_ordered"]
            assert entry["candidate_order_hash"] == canonical_candidate_pool_hash(pool), arm
            assert entry["candidate_membership_hash"] == \
                canonical_candidate_membership_hash(pool), arm

    def test_environment_fingerprint_is_persisted(self, receipt):
        assert receipt["environment_fingerprint"].get("platform")

    def test_availability_is_computable_without_replaying_retrieval(self, receipt):
        """The whole point: a diagnostic can answer 'was record X available'
        straight from the artifact."""
        pool = set(receipt["arms"]["J0"]["candidate_ids_ordered"])
        assert isinstance(pool, set) and pool


@pytest.mark.skipif(not POOLS.is_file(), reason="confirmation pool artifact absent")
class TestConfirmationPoolArtifact:
    @pytest.fixture(scope="class")
    def artifact(self):
        return json.loads(POOLS.read_text())

    def test_captured_on_the_scoring_platform_and_reproduces_exactly(self, artifact):
        assert artifact["reproduces_scored_run_exactly"] is True
        assert artifact["order_hash_mismatch_count"] == 0
        assert artifact["environment_fingerprint"]["cuda_available"] is True

    def test_every_membership_hash_matches_its_own_candidate_ids(self, artifact):
        from hrm_adaptive_memory.c4.packet_ordering import (
            canonical_candidate_membership_hash)
        for entry in artifact["pools"]:
            assert canonical_candidate_membership_hash(entry["candidate_ids"]) == \
                entry["candidate_membership_hash"], entry["task_id"]

    def test_covers_the_whole_split(self, artifact):
        assert artifact["task_count"] == 500


class TestB3ProtocolIsFrozenWithoutValues:
    @pytest.fixture(scope="class")
    def protocol(self):
        return json.loads(B3_PROTOCOL.read_text())

    def test_rho_is_not_chosen(self, protocol):
        """Choosing rho before multi-scale calibration exists would either be
        arbitrary or smuggle in confirmation #1's numbers."""
        policy = protocol["policy_form_only_no_values"]
        assert policy["rho_status"].startswith("NOT CHOSEN")
        assert "rho" not in {k.lower() for k in policy} or "rho_status" in policy
        assert "NOT CHOSEN" in json.dumps(policy)

    def test_success_thresholds_are_deferred(self, protocol):
        assert protocol["success_criteria_deferred"]["status"].startswith("delta_Q")

    def test_packet_budget_is_frozen_at_six(self, protocol):
        assert protocol["frozen_and_not_to_be_modified"]["packet_budget"] == 6

    def test_selector_is_frozen_unmodified(self, protocol):
        frozen = protocol["frozen_and_not_to_be_modified"]["selector"]
        assert "No modification" in frozen

    def test_bridge_retention_is_the_primary_safety_metric(self, protocol):
        signal = protocol["known_selector_safety_signal"]
        assert signal["value"] == -0.0435
        assert signal["frozen_boundary"] == -0.05
        assert "PRIMARY SAFETY METRIC" in signal["status"]

    def test_policy_inputs_exclude_every_evaluation_label(self, protocol):
        forbidden = protocol["policy_form_only_no_values"]["inputs_forbidden"]
        for label in ("record_kind", "answer availability", "oracle proof depth"):
            assert any(label in f for f in forbidden), label

    def test_three_budgets_are_kept_separate(self, protocol):
        budgets = protocol["three_budgets_are_separate"]
        assert budgets["hrm_packet_budget"].endswith("FIXED at 6")

    def test_gpu_is_gated_behind_two_cheaper_gates(self, protocol):
        gates = protocol["staged_gates_before_gpu"]
        assert "STOP, no GPU" in gates["gate_1_retrieval_only"]
        assert "before spending GPU" in gates["gate_2_selector_only"]

    def test_consumed_splits_are_marked(self, protocol):
        lineage = protocol["split_lineage"]
        assert "CONSUMED" in lineage["qualification_1"]
        assert "CONSUMED" in lineage["confirmation_1"]

    def test_stop_gate_is_recorded_as_frozen_not_provisional(self, protocol):
        gate = protocol["closed_stop_gate"]
        assert gate["OUTCOME"] == "OUTCOME_A_POSITIVE"
        assert "not provisional" in gate["status"]
