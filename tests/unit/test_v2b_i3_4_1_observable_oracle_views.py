"""Tests for the I3.4.1 evaluation-specific observable-oracle views."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hrm_adaptive_memory.executive.i3_4_observable_oracle_views import (
    VIEW_SCHEMA, VIEW_VERSION, ObservableOracleView,
    build_observable_oracle_views, save_views, views_module_sha256)


ROOT = Path(__file__).resolve().parents[2]


def test_view_schema_is_frozen():
    assert VIEW_SCHEMA == "DAPH_V2B_I3_4_OBSERVABLE_ORACLE_VIEW_V1"
    assert VIEW_VERSION == 1


def test_views_module_has_sha256():
    h = views_module_sha256()
    assert len(h) == 64


@pytest.mark.skipif(
    not (ROOT / "experiments/v2b_i3_3/oracle_tables/v2b_i3_3_sequential_state_aware_controller_v1.jsonl.gz").exists(),
    reason="Oracle tables not available in lite archive")
def test_build_views_produces_all_splits_and_conditions():
    views = build_observable_oracle_views(root=ROOT)
    # 5 splits × 7 conditions = 35 max (some may be missing if no oracle)
    assert len(views) > 0
    split_names = {v.split_name for v in views}
    assert "development" in split_names
    assert "held_out_structure" in split_names
    conditions = {v.condition for v in views}
    assert "STATE_AWARE_CONTROLLER" in conditions
    assert "STATE_BLIND_CONTROLLER" in conditions


@pytest.mark.skipif(
    not (ROOT / "experiments/v2b_i3_3/oracle_tables/v2b_i3_3_sequential_state_aware_controller_v1.jsonl.gz").exists(),
    reason="Oracle tables not available in lite archive")
def test_observable_values_differ_across_splits():
    """The key test: V_O must differ across splits for the same condition."""
    views = build_observable_oracle_views(root=ROOT)
    aware_views = {v.split_name: v for v in views
                   if v.condition == "STATE_AWARE_CONTROLLER"}
    if len(aware_views) >= 2:
        values = [v.observable_optimal_value for v in aware_views.values()]
        # Not all values should be identical
        assert len(set(round(v, 6) for v in values)) > 1


@pytest.mark.skipif(
    not (ROOT / "experiments/v2b_i3_3/oracle_tables/v2b_i3_3_sequential_state_aware_controller_v1.jsonl.gz").exists(),
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


@pytest.mark.skipif(
    not (ROOT / "experiments/v2b_i3_3/oracle_tables/v2b_i3_3_sequential_state_aware_controller_v1.jsonl.gz").exists(),
    reason="Oracle tables not available in lite archive")
def test_views_have_sha256():
    views = build_observable_oracle_views(root=ROOT)
    for v in views:
        assert len(v.view_sha256) == 64
        assert len(v.observable_oracle_set_sha256) == 64
        assert len(v.latent_oracle_table_sha256) == 64
        assert len(v.information_class_hash) == 64


@pytest.mark.skipif(
    not (ROOT / "experiments/v2b_i3_3/oracle_tables/v2b_i3_3_sequential_state_aware_controller_v1.jsonl.gz").exists(),
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


@pytest.mark.skipif(
    not (ROOT / "experiments/v2b_i3_3/oracle_tables/v2b_i3_3_sequential_state_aware_controller_v1.jsonl.gz").exists(),
    reason="Oracle tables not available in lite archive")
def test_view_as_dict_is_serializable():
    views = build_observable_oracle_views(root=ROOT)
    for v in views:
        d = v.as_dict()
        json.dumps(d)  # should not raise


@pytest.mark.skipif(
    not (ROOT / "experiments/v2b_i3_3/oracle_tables/v2b_i3_3_sequential_state_aware_controller_v1.jsonl.gz").exists(),
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
