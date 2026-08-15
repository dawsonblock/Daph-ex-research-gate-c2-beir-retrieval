"""Tests for the I3.4.1 statistical analysis module."""
from __future__ import annotations

import pytest

from hrm_adaptive_memory.executive.i3_4_scientific_scoring import I34PairedDelta
from hrm_adaptive_memory.executive.i3_4_statistical_analysis import (
    STATS_SCHEMA, STATS_VERSION, bonferroni_correct,
    paired_bootstrap, topology_cluster_bootstrap, stats_module_sha256)


def _make_deltas(values):
    """Create paired deltas from a list of (delta_dg,) values."""
    return [
        I34PairedDelta(task_id=f"t{i}", delta_ig=0.0, delta_dg=v,
                       delta_tr=v, delta_cost=0.0)
        for i, v in enumerate(values)
    ]


def test_stats_schema_is_frozen():
    assert STATS_SCHEMA == "DAPH_V2B_I3_4_STATISTICAL_ANALYSIS_V1"
    assert STATS_VERSION == 1


def test_stats_module_has_sha256():
    h = stats_module_sha256()
    assert len(h) == 64


def test_paired_bootstrap_basic():
    deltas = _make_deltas([2.0, 3.0, 1.0, 4.0, 2.0])
    result = paired_bootstrap(deltas, iterations=1000, seed=42)
    assert result.method == "paired_bootstrap"
    assert result.resampling_unit == "task"
    assert result.n_units == 5
    assert result.point_estimate == pytest.approx(2.4)
    assert result.lower_bound < result.upper_bound
    assert result.iterations == 1000


def test_paired_bootstrap_positive_ci():
    """When all deltas are clearly positive, LCB should be > 0."""
    deltas = _make_deltas([5.0, 6.0, 4.0, 7.0, 5.0, 6.0, 4.0, 7.0])
    result = paired_bootstrap(deltas, iterations=5000, seed=42)
    assert result.ci_significant_positive
    assert result.lower_bound > 0.0


def test_paired_bootstrap_negative_ci():
    """When all deltas are clearly negative, UCB should be < 0."""
    deltas = _make_deltas([-5.0, -6.0, -4.0, -7.0, -5.0, -6.0, -4.0, -7.0])
    result = paired_bootstrap(deltas, iterations=5000, seed=42)
    assert result.ci_significant_negative
    assert result.upper_bound < 0.0


def test_paired_bootstrap_zero_ci():
    """When deltas straddle zero, CI should contain zero."""
    deltas = _make_deltas([1.0, -1.0, 0.5, -0.5, 1.0, -1.0, 0.5, -0.5])
    result = paired_bootstrap(deltas, iterations=5000, seed=42)
    assert result.lower_bound <= 0.0 <= result.upper_bound
    assert not result.ci_significant_positive


def test_paired_bootstrap_empty():
    result = paired_bootstrap([], iterations=1000, seed=42)
    assert result.n_units == 0
    assert result.point_estimate == 0.0


def test_paired_bootstrap_deterministic_with_seed():
    """Same seed should produce same result."""
    deltas = _make_deltas([2.0, 3.0, 1.0, 4.0, 2.0])
    r1 = paired_bootstrap(deltas, iterations=1000, seed=42)
    r2 = paired_bootstrap(deltas, iterations=1000, seed=42)
    assert r1.lower_bound == r2.lower_bound
    assert r2.upper_bound == r2.upper_bound


def test_topology_cluster_bootstrap_basic():
    deltas = _make_deltas([2.0, 5.0, 1.0, 4.0, 3.0, 6.0])
    # Assign 6 tasks to 3 clusters with different means
    cluster_map = {
        "t0": "topo_A", "t1": "topo_A",
        "t2": "topo_B", "t3": "topo_B",
        "t4": "topo_C", "t5": "topo_C",
    }
    result = topology_cluster_bootstrap(
        deltas, cluster_map=cluster_map, iterations=1000, seed=42)
    assert result.method == "topology_cluster_bootstrap"
    assert result.resampling_unit == "topology_cluster"
    assert result.n_units == 3  # 3 clusters
    assert result.lower_bound < result.upper_bound


def test_topology_cluster_bootstrap_positive():
    deltas = _make_deltas([5.0, 5.0, 6.0, 6.0, 4.0, 4.0, 7.0, 7.0])
    cluster_map = {
        "t0": "A", "t1": "A",
        "t2": "B", "t3": "B",
        "t4": "C", "t5": "C",
        "t6": "D", "t7": "D",
    }
    result = topology_cluster_bootstrap(
        deltas, cluster_map=cluster_map, iterations=5000, seed=42)
    assert result.ci_significant_positive


def test_topology_cluster_bootstrap_empty():
    result = topology_cluster_bootstrap(
        [], cluster_map={}, iterations=1000, seed=42)
    assert result.n_units == 0


def test_topology_cluster_bootstrap_single_cluster():
    """With one cluster, the CI should be wide (no resampling variation)."""
    deltas = _make_deltas([3.0, 3.0, 3.0])
    cluster_map = {"t0": "only", "t1": "only", "t2": "only"}
    result = topology_cluster_bootstrap(
        deltas, cluster_map=cluster_map, iterations=1000, seed=42)
    assert result.n_units == 1
    assert result.point_estimate == pytest.approx(3.0)


def test_bonferroni_correct():
    p_values = [0.01, 0.04, 0.03]
    # Bonferroni threshold = 0.05 / 3 ≈ 0.0167
    results = bonferroni_correct(p_values, alpha=0.05)
    assert results[0] is True   # 0.01 < 0.0167
    assert results[1] is False  # 0.04 > 0.0167
    assert results[2] is False  # 0.03 > 0.0167


def test_bonferroni_correct_empty():
    assert bonferroni_correct([]) == []


def test_bonferroni_correct_all_significant():
    p_values = [0.001, 0.002, 0.003]
    results = bonferroni_correct(p_values, alpha=0.05)
    assert all(results)
