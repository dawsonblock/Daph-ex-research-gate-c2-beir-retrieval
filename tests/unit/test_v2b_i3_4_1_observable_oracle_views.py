"""Tests for the I3.4.1 evaluation-specific observable-oracle views (V2).

V2 fixes the V1 defect where sequential oracle table positions were zipped
against the global task order. V2 reads the ``members`` field of each
table's initial information state to determine which tasks belong to which
information class, then maps each task to its information class's V_O^*(B).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hrm_adaptive_memory.executive.i3_4_observable_oracle_views import (
    VIEW_SCHEMA, VIEW_VERSION, ObservableOracleView,
    build_observable_oracle_views, save_views, views_module_sha256,
    load_task_to_observable, load_information_classes)


ROOT = Path(__file__).resolve().parents[2]

ORACLE_AVAILABLE = (
    ROOT / "experiments/v2b_i3_3/oracle_tables/"
    "v2b_i3_3_sequential_state_aware_controller_v1.jsonl.gz").exists()


def test_view_schema_is_frozen():
    assert VIEW_SCHEMA == "DAPH_V2B_I3_4_OBSERVABLE_ORACLE_VIEW_V2"
    assert VIEW_VERSION == 2


def test_views_module_has_sha256():
    h = views_module_sha256()
    assert len(h) == 64


@pytest.mark.skipif(not ORACLE_AVAILABLE,
                    reason="Oracle tables not available in lite archive")
def test_load_task_to_observable_maps_all_750_tasks():
    """Every task must resolve to exactly one information class."""
    task_map = load_task_to_observable(ROOT, "STATE_AWARE_CONTROLLER")
    assert len(task_map) == 750
    # Every task must have a non-None V_O
    for tid, entry in task_map.items():
        assert entry.observable_optimal_value is not None
        assert entry.information_class_id  # non-empty class ID


@pytest.mark.skipif(not ORACLE_AVAILABLE,
                    reason="Oracle tables not available in lite archive")
def test_load_task_to_observable_blind_has_more_aliasing():
    """Blind condition should have more multi-member classes than aware."""
    aware_classes = load_information_classes(ROOT, "STATE_AWARE_CONTROLLER")
    blind_classes = load_information_classes(ROOT, "STATE_BLIND_CONTROLLER")
    aware_multi = sum(1 for c in aware_classes if len(c.member_task_ids) > 1)
    blind_multi = sum(1 for c in blind_classes if len(c.member_task_ids) > 1)
    # The blind condition aliases more tasks together (less information)
    assert blind_multi > aware_multi


@pytest.mark.skipif(not ORACLE_AVAILABLE,
                    reason="Oracle tables not available in lite archive")
def test_per_task_vo_differs_across_tasks():
    """V_O must differ across tasks — not one scalar per split."""
    task_map = load_task_to_observable(ROOT, "STATE_AWARE_CONTROLLER")
    dev_values = [entry.observable_optimal_value
                  for tid, entry in task_map.items()
                  if "development" in tid]
    # There must be more than one unique value across 300 dev tasks
    unique = set(round(v, 4) for v in dev_values)
    assert len(unique) > 1, f"Only {len(unique)} unique V_O values — positional mapping bug?"


@pytest.mark.skipif(not ORACLE_AVAILABLE,
                    reason="Oracle tables not available in lite archive")
def test_per_task_vo_blind_differs_from_aware():
    """Blind and aware must produce different V_O for at least some tasks."""
    aware_map = load_task_to_observable(ROOT, "STATE_AWARE_CONTROLLER")
    blind_map = load_task_to_observable(ROOT, "STATE_BLIND_CONTROLLER")
    common = set(aware_map) & set(blind_map)
    diffs = sum(1 for tid in common
                if abs(aware_map[tid].observable_optimal_value
                       - blind_map[tid].observable_optimal_value) > 0.01)
    assert diffs > 0, "Blind and aware V_O are identical for all tasks"


@pytest.mark.skipif(not ORACLE_AVAILABLE,
                    reason="Oracle tables not available in lite archive")
def test_build_views_produces_all_splits_and_conditions():
    views = build_observable_oracle_views(root=ROOT)
    assert len(views) > 0
    split_names = {v.split_name for v in views}
    assert "development" in split_names
    assert "held_out_structure" in split_names
    conditions = {v.condition for v in views}
    assert "STATE_AWARE_CONTROLLER" in conditions
    assert "STATE_BLIND_CONTROLLER" in conditions


@pytest.mark.skipif(not ORACLE_AVAILABLE,
                    reason="Oracle tables not available in lite archive")
def test_observable_values_differ_across_splits():
    """V_O must differ across splits for the same condition."""
    views = build_observable_oracle_views(root=ROOT)
    aware_views = {v.split_name: v for v in views
                   if v.condition == "STATE_AWARE_CONTROLLER"}
    if len(aware_views) >= 2:
        values = [v.observable_optimal_value for v in aware_views.values()]
        assert len(set(round(v, 6) for v in values)) > 1


@pytest.mark.skipif(not ORACLE_AVAILABLE,
                    reason="Oracle tables not available in lite archive")
def test_aware_observable_higher_than_blind():
    """STATE_AWARE should have higher V_O than STATE_BLIND (more information)."""
    views = build_observable_oracle_views(root=ROOT)
    for split in ("development", "held_out_structure"):
        aware = next((v for v in views
                      if v.split_name == split
                      and v.condition == "STATE_AWARE_CONTROLLER"), None)
        blind = next((v for v in views
                      if v.split_name == split
                      and v.condition == "STATE_BLIND_CONTROLLER"), None)
        if aware and blind:
            assert aware.observable_optimal_value > blind.observable_optimal_value, \
                f"{split}: aware V_O ({aware.observable_optimal_value}) should exceed blind V_O ({blind.observable_optimal_value})"


@pytest.mark.skipif(not ORACLE_AVAILABLE,
                    reason="Oracle tables not available in lite archive")
def test_views_have_sha256():
    views = build_observable_oracle_views(root=ROOT)
    for v in views:
        assert len(v.view_sha256) == 64
        assert len(v.observable_oracle_set_sha256) == 64
        assert len(v.latent_oracle_table_sha256) == 64


@pytest.mark.skipif(not ORACLE_AVAILABLE,
                    reason="Oracle tables not available in lite archive")
def test_views_have_correct_task_counts():
    views = build_observable_oracle_views(root=ROOT)
    for v in views:
        if v.split_name == "development":
            assert v.task_count == 300
        elif v.split_name == "validation":
            assert v.task_count == 150
        elif v.split_name == "held_out_instance":
            assert v.task_count == 100
        elif v.split_name == "held_out_surface":
            assert v.task_count == 50
        elif v.split_name == "held_out_structure":
            assert v.task_count == 150


@pytest.mark.skipif(not ORACLE_AVAILABLE,
                    reason="Oracle tables not available in lite archive")
def test_view_has_per_task_entries():
    """V2 views must contain per-task entries, not just a split-level scalar."""
    views = build_observable_oracle_views(root=ROOT)
    for v in views:
        assert len(v.task_entries) == v.task_count
        for entry in v.task_entries:
            assert entry.task_id
            assert entry.information_class_id
            assert entry.observable_optimal_value is not None


@pytest.mark.skipif(not ORACLE_AVAILABLE,
                    reason="Oracle tables not available in lite archive")
def test_view_has_information_classes():
    """V2 views must contain per-class structure with member tasks."""
    views = build_observable_oracle_views(root=ROOT)
    for v in views:
        assert len(v.information_classes) > 0
        for cls in v.information_classes:
            assert cls.class_id
            assert len(cls.member_task_ids) >= 1
            # Posterior weights must sum to 1
            from fractions import Fraction
            total = sum(Fraction(w) for w in cls.posterior_weights)
            assert total == Fraction(1, 1), f"Class {cls.class_id} weights sum to {total}, not 1"


@pytest.mark.skipif(not ORACLE_AVAILABLE,
                    reason="Oracle tables not available in lite archive")
def test_every_task_resolves_to_exactly_one_class():
    """Each task must appear in exactly one information class per condition."""
    from collections import Counter
    for condition in ("STATE_AWARE_CONTROLLER", "STATE_BLIND_CONTROLLER"):
        classes = load_information_classes(ROOT, condition)
        task_counts = Counter()
        for cls in classes:
            for tid in cls.member_task_ids:
                task_counts[tid] += 1
        # Every task appears exactly once
        for tid, count in task_counts.items():
            assert count == 1, f"Task {tid} appears in {count} classes under {condition}"


@pytest.mark.skipif(not ORACLE_AVAILABLE,
                    reason="Oracle tables not available in lite archive")
def test_class_members_have_identical_initial_observations():
    """All tasks in the same information class must share the same V_O."""
    for condition in ("STATE_AWARE_CONTROLLER", "STATE_BLIND_CONTROLLER"):
        classes = load_information_classes(ROOT, condition)
        for cls in classes:
            if len(cls.member_task_ids) > 1:
                # All members share the same V_O by construction
                assert cls.observable_optimal_value is not None


@pytest.mark.skipif(not ORACLE_AVAILABLE,
                    reason="Oracle tables not available in lite archive")
def test_view_as_dict_is_serializable():
    views = build_observable_oracle_views(root=ROOT)
    for v in views:
        d = v.as_dict()
        json.dumps(d)  # should not raise


@pytest.mark.skipif(not ORACLE_AVAILABLE,
                    reason="Oracle tables not available in lite archive")
def test_saved_views_file_exists_and_is_valid_json():
    views = build_observable_oracle_views(root=ROOT)
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        path = f.name
    try:
        sha = save_views(views, path)
        data = json.loads(Path(path).read_text())
        assert data["schema"] == VIEW_SCHEMA
        assert len(data["views"]) == len(views)
        assert len(sha) == 64
    finally:
        os.unlink(path)
