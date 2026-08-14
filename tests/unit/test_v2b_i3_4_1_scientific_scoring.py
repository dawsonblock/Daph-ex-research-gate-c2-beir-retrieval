"""Tests for the I3.4.1 scientific scoring module.

Verifies the IG/DG/TR decomposition, the TR = IG + DG identity,
non-negativity at the aggregate level, paired deltas, and that
individual contributions are NOT clamped.
"""
from __future__ import annotations

import pytest

from hrm_adaptive_memory.executive.i3_4_scientific_scoring import (
    IDENTITY_TOLERANCE, SCORING_SCHEMA, SCORING_VERSION,
    I34ScientificAggregate, I34ScientificTaskContribution, I34PairedDelta,
    compute_aggregate, compute_paired_deltas, compute_task_contribution,
    mean_delta_dg, mean_delta_ig, mean_delta_tr,
    scoring_module_sha256, verify_all_identities)


def _make_contribution(
    task_id="t1", condition="STATE_BLIND_CONTROLLER",
    v_latent=10.0, v_observable=7.0, v_controller=5.0,
    info_class_hash="abc", oracle_set_hash="def", latent_table_hash="ghi",
):
    return compute_task_contribution(
        task_id=task_id, condition=condition,
        latent_optimal_value=v_latent,
        observable_optimal_value=v_observable,
        controller_value=v_controller,
        information_class_hash=info_class_hash,
        observable_oracle_set_sha256=oracle_set_hash,
        latent_oracle_table_sha256=latent_table_hash)


# --- Schema ---

def test_scoring_schema_is_frozen():
    assert SCORING_SCHEMA == "DAPH_V2B_I3_4_SCIENTIFIC_SCORING_V1"
    assert SCORING_VERSION == 1


def test_scoring_module_has_sha256():
    h = scoring_module_sha256()
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


# --- TR = IG + DG identity ---

def test_tr_equals_ig_plus_dg_for_task_contribution():
    c = _make_contribution(v_latent=10.0, v_observable=7.0, v_controller=5.0)
    assert c.information_gap_contribution == 3.0  # 10 - 7
    assert c.decision_gap_contribution == 2.0     # 7 - 5
    assert c.total_regret_contribution == 5.0      # 10 - 5
    assert c.verify_identity()


def test_tr_equals_ig_plus_dg_with_negative_dg():
    """DG can be negative for individual tasks (observable oracle is an
    expectation over the information class, not a per-task optimum)."""
    c = _make_contribution(v_latent=10.0, v_observable=6.0, v_controller=8.0)
    assert c.information_gap_contribution == 4.0   # 10 - 6
    assert c.decision_gap_contribution == -2.0     # 6 - 8 (negative!)
    assert c.total_regret_contribution == 2.0       # 10 - 8
    assert c.verify_identity()  # TR = IG + DG still holds


def test_tr_equals_ig_plus_dg_with_negative_ig():
    """IG can also be negative for individual tasks."""
    c = _make_contribution(v_latent=5.0, v_observable=7.0, v_controller=3.0)
    assert c.information_gap_contribution == -2.0  # 5 - 7 (negative!)
    assert c.decision_gap_contribution == 4.0      # 7 - 3
    assert c.total_regret_contribution == 2.0       # 5 - 3
    assert c.verify_identity()


def test_tr_equals_ig_plus_dg_at_aggregate():
    contributions = [
        _make_contribution(task_id="t1", v_latent=10.0, v_observable=7.0, v_controller=5.0),
        _make_contribution(task_id="t2", v_latent=8.0, v_observable=6.0, v_controller=4.0),
        _make_contribution(task_id="t3", v_latent=5.0, v_observable=7.0, v_controller=3.0),
    ]
    agg = compute_aggregate(condition="STATE_BLIND_CONTROLLER", contributions=contributions)
    assert agg.verify_identity()
    # E[V_L] = (10+8+5)/3 = 7.667
    # E[V_O] = (7+6+7)/3 = 6.667
    # E[V_pi] = (5+4+3)/3 = 4.0
    # IG = (3+2+(-2))/3 = 1.0
    # DG = (2+2+4)/3 = 2.667
    # TR = (5+4+2)/3 = 3.667
    assert abs(agg.information_gap - 1.0) < 1e-9
    assert abs(agg.decision_gap - (8.0 / 3.0)) < 1e-9
    assert abs(agg.total_regret - (11.0 / 3.0)) < 1e-9


# --- Non-negativity at aggregate level ---

def test_aggregate_nonnegativity_with_uniform_prior():
    """IG and DG should be non-negative at the aggregate level when the
    observable oracle is the correct expectation over the information class."""
    contributions = [
        _make_contribution(task_id="t1", v_latent=10.0, v_observable=7.0, v_controller=5.0),
        _make_contribution(task_id="t2", v_latent=8.0, v_observable=6.0, v_controller=4.0),
    ]
    agg = compute_aggregate(condition="STATE_BLIND_CONTROLLER", contributions=contributions)
    assert agg.verify_nonnegativity()
    assert agg.information_gap >= 0
    assert agg.decision_gap >= 0


def test_aggregate_nonnegativity_can_fail_with_negative_individual():
    """If the observable oracle is wrong (not the correct expectation),
    aggregate IG can be negative.  This test verifies the check works."""
    contributions = [
        _make_contribution(task_id="t1", v_latent=5.0, v_observable=7.0, v_controller=3.0),
        _make_contribution(task_id="t2", v_latent=5.0, v_observable=7.0, v_controller=3.0),
    ]
    agg = compute_aggregate(condition="STATE_BLIND_CONTROLLER", contributions=contributions)
    # IG = (-2 + -2) / 2 = -2.0 (negative because V_O > V_L for all tasks)
    assert agg.information_gap < 0
    assert not agg.verify_nonnegativity()


# --- No clamping of individual contributions ---

def test_individual_contributions_not_clamped():
    """The old code used max(0, ...) which clamped TR.  The new scoring
    must NOT clamp IG or DG for individual tasks."""
    c = _make_contribution(v_latent=5.0, v_observable=7.0, v_controller=8.0)
    # DG = 7 - 8 = -1 (must not be clamped to 0)
    assert c.decision_gap_contribution == -1.0
    # IG = 5 - 7 = -2 (must not be clamped to 0)
    assert c.information_gap_contribution == -2.0
    # TR = 5 - 8 = -3 (must not be clamped to 0)
    assert c.total_regret_contribution == -3.0


# --- Paired deltas ---

def test_paired_deltas_compute_correctly():
    blind = {
        "t1": _make_contribution(task_id="t1", condition="STATE_BLIND_CONTROLLER",
                                 v_latent=10.0, v_observable=6.0, v_controller=4.0),
        "t2": _make_contribution(task_id="t2", condition="STATE_BLIND_CONTROLLER",
                                 v_latent=8.0, v_observable=5.0, v_controller=3.0),
    }
    aware = {
        "t1": _make_contribution(task_id="t1", condition="STATE_AWARE_CONTROLLER",
                                 v_latent=10.0, v_observable=9.0, v_controller=7.0),
        "t2": _make_contribution(task_id="t2", condition="STATE_AWARE_CONTROLLER",
                                 v_latent=8.0, v_observable=7.0, v_controller=5.0),
    }
    deltas = compute_paired_deltas(
        blind_contributions=blind, aware_contributions=aware)
    assert len(deltas) == 2
    # t1: ΔIG = (10-6) - (10-9) = 4 - 1 = 3
    #     ΔDG = (6-4) - (9-7) = 2 - 2 = 0
    #     ΔTR = (10-4) - (10-7) = 6 - 3 = 3
    d1 = next(d for d in deltas if d.task_id == "t1")
    assert d1.delta_ig == 3.0
    assert d1.delta_dg == 0.0
    assert d1.delta_tr == 3.0
    # t2: ΔIG = (8-5) - (8-7) = 3 - 1 = 2
    #     ΔDG = (5-3) - (7-5) = 2 - 2 = 0
    #     ΔTR = (8-3) - (8-5) = 5 - 3 = 2
    d2 = next(d for d in deltas if d.task_id == "t2")
    assert d2.delta_ig == 2.0
    assert d2.delta_dg == 0.0
    assert d2.delta_tr == 2.0


def test_paired_deltas_only_for_common_tasks():
    blind = {"t1": _make_contribution(task_id="t1")}
    aware = {"t2": _make_contribution(task_id="t2")}
    deltas = compute_paired_deltas(blind_contributions=blind, aware_contributions=aware)
    assert len(deltas) == 0


def test_mean_delta_dg():
    deltas = [
        I34PairedDelta(task_id="t1", delta_ig=3.0, delta_dg=2.0, delta_tr=5.0, delta_cost=0.0),
        I34PairedDelta(task_id="t2", delta_ig=1.0, delta_dg=4.0, delta_tr=5.0, delta_cost=0.0),
    ]
    assert mean_delta_dg(deltas) == 3.0  # (2+4)/2


def test_mean_delta_ig():
    deltas = [
        I34PairedDelta(task_id="t1", delta_ig=3.0, delta_dg=2.0, delta_tr=5.0, delta_cost=0.0),
        I34PairedDelta(task_id="t2", delta_ig=1.0, delta_dg=4.0, delta_tr=5.0, delta_cost=0.0),
    ]
    assert mean_delta_ig(deltas) == 2.0  # (3+1)/2


def test_mean_delta_tr():
    deltas = [
        I34PairedDelta(task_id="t1", delta_ig=3.0, delta_dg=2.0, delta_tr=5.0, delta_cost=0.0),
        I34PairedDelta(task_id="t2", delta_ig=1.0, delta_dg=4.0, delta_tr=5.0, delta_cost=0.0),
    ]
    assert mean_delta_tr(deltas) == 5.0  # (5+5)/2


# --- Verify all identities ---

def test_verify_all_identities_pass():
    contributions = [
        _make_contribution(task_id="t1", v_latent=10.0, v_observable=7.0, v_controller=5.0),
        _make_contribution(task_id="t2", v_latent=8.0, v_observable=6.0, v_controller=4.0),
    ]
    all_pass, failures = verify_all_identities(contributions)
    assert all_pass
    assert failures == []


def test_verify_all_identities_fails():
    """Manually construct a broken contribution to test the verifier."""
    c = I34ScientificTaskContribution(
        task_id="t1", condition="STATE_BLIND_CONTROLLER",
        latent_optimal_value=10.0, observable_optimal_value=7.0, controller_value=5.0,
        information_gap_contribution=3.0, decision_gap_contribution=2.0,
        total_regret_contribution=99.0,  # Wrong! Should be 5.0
        information_class_hash="abc",
        observable_oracle_set_sha256="def",
        latent_oracle_table_sha256="ghi",
    )
    all_pass, failures = verify_all_identities([c])
    assert not all_pass
    assert "t1" in failures


# --- Serialization ---

def test_task_contribution_as_dict():
    c = _make_contribution(v_latent=10.0, v_observable=7.0, v_controller=5.0)
    d = c.as_dict()
    assert d["task_id"] == "t1"
    assert d["condition"] == "STATE_BLIND_CONTROLLER"
    assert d["latent_optimal_value"] == 10.0
    assert d["observable_optimal_value"] == 7.0
    assert d["controller_value"] == 5.0
    assert d["information_gap_contribution"] == 3.0
    assert d["decision_gap_contribution"] == 2.0
    assert d["total_regret_contribution"] == 5.0


def test_aggregate_as_dict():
    contributions = [_make_contribution(v_latent=10.0, v_observable=7.0, v_controller=5.0)]
    agg = compute_aggregate(condition="STATE_BLIND_CONTROLLER", contributions=contributions)
    d = agg.as_dict()
    assert d["condition"] == "STATE_BLIND_CONTROLLER"
    assert d["task_count"] == 1
    assert d["prior"] == "TASK_UNIFORM"
    assert d["information_gap"] == 3.0
    assert d["decision_gap"] == 2.0
    assert d["total_regret"] == 5.0


def test_paired_delta_as_dict():
    d = I34PairedDelta(task_id="t1", delta_ig=3.0, delta_dg=2.0, delta_tr=5.0, delta_cost=1.0)
    dd = d.as_dict()
    assert dd["task_id"] == "t1"
    assert dd["delta_ig"] == 3.0
    assert dd["delta_dg"] == 2.0
    assert dd["delta_tr"] == 5.0
    assert dd["delta_cost"] == 1.0


# --- Empty edge cases ---

def test_empty_aggregate():
    agg = compute_aggregate(condition="STATE_BLIND_CONTROLLER", contributions=[])
    assert agg.task_count == 0
    assert agg.information_gap == 0.0
    assert agg.decision_gap == 0.0
    assert agg.total_regret == 0.0
    assert agg.verify_identity()


def test_empty_paired_deltas():
    deltas = compute_paired_deltas(blind_contributions={}, aware_contributions={})
    assert deltas == []
    assert mean_delta_dg(deltas) == 0.0


# --- Distinct claims ---

def test_information_without_exploitation_claim():
    """ΔIG > 0 but ΔDG ≈ 0: state has info but model doesn't exploit it."""
    blind = {
        "t1": _make_contribution(task_id="t1", condition="B",
                                 v_latent=10.0, v_observable=5.0, v_controller=4.0),
    }
    aware = {
        "t1": _make_contribution(task_id="t1", condition="A",
                                 v_latent=10.0, v_observable=8.0, v_controller=7.0),
    }
    deltas = compute_paired_deltas(blind_contributions=blind, aware_contributions=aware)
    # ΔIG = (10-5) - (10-8) = 5 - 2 = 3 > 0
    # ΔDG = (5-4) - (8-7) = 1 - 1 = 0
    assert mean_delta_ig(deltas) > 0
    assert abs(mean_delta_dg(deltas)) < 1e-9


def test_executive_exploitation_claim():
    """ΔDG > 0: model makes better decisions with structured state."""
    blind = {
        "t1": _make_contribution(task_id="t1", condition="B",
                                 v_latent=10.0, v_observable=8.0, v_controller=4.0),
    }
    aware = {
        "t1": _make_contribution(task_id="t1", condition="A",
                                 v_latent=10.0, v_observable=8.0, v_controller=7.0),
    }
    deltas = compute_paired_deltas(blind_contributions=blind, aware_contributions=aware)
    # ΔDG = (8-4) - (8-7) = 4 - 1 = 3 > 0
    assert mean_delta_dg(deltas) > 0


def test_no_substitution_of_tr_for_dg():
    """TR must not be substituted for DG.  They differ when IG differs."""
    blind = {
        "t1": _make_contribution(task_id="t1", condition="B",
                                 v_latent=10.0, v_observable=5.0, v_controller=4.0),
    }
    aware = {
        "t1": _make_contribution(task_id="t1", condition="A",
                                 v_latent=10.0, v_observable=8.0, v_controller=7.0),
    }
    deltas = compute_paired_deltas(blind_contributions=blind, aware_contributions=aware)
    # ΔDG = (5-4) - (8-7) = 1 - 1 = 0
    # ΔTR = (10-4) - (10-7) = 6 - 3 = 3
    # Using TR would falsely suggest improvement; DG correctly shows no exploitation
    assert abs(mean_delta_dg(deltas)) < 1e-9
    assert mean_delta_tr(deltas) > 0
