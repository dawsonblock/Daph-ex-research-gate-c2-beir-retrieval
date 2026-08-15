"""Tests for V2 information-class split purity and observable oracle views.

Verifies that:
1. Every information class in a V2 split-specific view is split-pure
   (no class contains members from multiple splits)
2. Posterior weights in every information class sum to 1
3. V2 observable oracle views load correctly
4. Held-out topology isolation is maintained
"""
import gzip
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]

V2_ORACLE_DIR = ROOT / "experiments/v2b_i3_5/oracle_tables"
V2_SPLITS_PATH = ROOT / "experiments/v2b_i3_5/splits/v2b_i3_5_splits_v2.json"
V2_VIEWS_PATH = V2_ORACLE_DIR / "v2b_i3_5_observable_oracle_views_v1.json"
V2_DIFF_PATH = V2_ORACLE_DIR / "v2b_i3_5_difficulty_report_v1.json"

ORACLE_AVAILABLE = (V2_ORACLE_DIR / "v2b_i3_5_sequential_state_blind_controller_v1.jsonl.gz").exists()


@pytest.mark.skipif(not ORACLE_AVAILABLE, reason="V2 oracle tables not available")
class TestV2SplitPurity:
    """Information-class split purity for V2."""

    def test_blind_classes_are_split_pure(self):
        """Every blind information class must be split-pure."""
        task_to_split = self._load_task_splits()
        classes = self._load_classes("STATE_BLIND_CONTROLLER")
        for cls in classes:
            member_splits = set(
                task_to_split.get(tid, "UNKNOWN")
                for tid in cls["member_task_ids"]
            )
            assert len(member_splits) == 1, (
                f"Cross-split blind class {cls['class_id'][:16]}... "
                f"has members from {member_splits}"
            )

    def test_aware_classes_are_split_pure(self):
        """Every aware information class must be split-pure."""
        task_to_split = self._load_task_splits()
        classes = self._load_classes("STATE_AWARE_CONTROLLER")
        for cls in classes:
            member_splits = set(
                task_to_split.get(tid, "UNKNOWN")
                for tid in cls["member_task_ids"]
            )
            assert len(member_splits) == 1, (
                f"Cross-split aware class {cls['class_id'][:16]}... "
                f"has members from {member_splits}"
            )

    def test_posterior_weights_sum_to_one_blind(self):
        """Blind posterior weights must sum to 1."""
        from fractions import Fraction
        classes = self._load_classes("STATE_BLIND_CONTROLLER")
        for cls in classes:
            total = sum(Fraction(w) for w in cls["posterior_weights"])
            assert total == Fraction(1, 1), (
                f"Blind class {cls['class_id'][:16]}...: weights sum to {total}"
            )

    def test_posterior_weights_sum_to_one_aware(self):
        """Aware posterior weights must sum to 1."""
        from fractions import Fraction
        classes = self._load_classes("STATE_AWARE_CONTROLLER")
        for cls in classes:
            total = sum(Fraction(w) for w in cls["posterior_weights"])
            assert total == Fraction(1, 1), (
                f"Aware class {cls['class_id'][:16]}...: weights sum to {total}"
            )

    def _load_task_splits(self) -> dict[str, str]:
        data = json.loads(V2_SPLITS_PATH.read_text())
        task_to_split = {}
        for split_name, entries in data["splits"].items():
            for entry in entries:
                task_to_split[entry["task_id"]] = split_name
        return task_to_split

    def _load_classes(self, condition: str) -> list[dict]:
        path = V2_ORACLE_DIR / f"v2b_i3_5_sequential_{condition.lower()}_v1.jsonl.gz"
        classes = []
        with gzip.open(path, "rt") as f:
            for line in f:
                entry = json.loads(line)
                table = entry["table"]
                init_id = entry["initial_information_state_id"]
                init_state = table["information_states"].get(init_id, {})
                members = init_state.get("members", [])
                if not members:
                    continue
                member_task_ids = sorted(m["task_id"] for m in members)
                posterior_weights = [m["posterior_weight"] for m in sorted(members, key=lambda x: x["task_id"])]
                classes.append({
                    "class_id": init_id,
                    "member_task_ids": member_task_ids,
                    "posterior_weights": posterior_weights,
                })
        return classes


@pytest.mark.skipif(not V2_VIEWS_PATH.exists(), reason="V2 views not available")
class TestV2ObservableOracleViews:
    """V2 observable oracle view correctness."""

    def test_views_load(self):
        """V2 views load correctly."""
        data = json.loads(V2_VIEWS_PATH.read_text())
        assert data["schema"] == "DAPH_V2B_I3_5_OBSERVABLE_ORACLE_VIEW_V2"
        assert len(data["views"]) == 6  # 3 splits × 2 conditions

    def test_aware_vo_higher_than_blind(self):
        """Aware V_O should be >= blind V_O (information advantage)."""
        data = json.loads(V2_VIEWS_PATH.read_text())
        for split in ("structure_dev_v2", "structure_validation_v2", "structure_held_out_v2"):
            aware = next(v for v in data["views"]
                         if v["split_name"] == split and v["condition"] == "STATE_AWARE_CONTROLLER")
            blind = next(v for v in data["views"]
                         if v["split_name"] == split and v["condition"] == "STATE_BLIND_CONTROLLER")
            assert aware["observable_optimal_value"] >= blind["observable_optimal_value"], (
                f"{split}: aware V_O ({aware['observable_optimal_value']}) "
                f"< blind V_O ({blind['observable_optimal_value']})"
            )

    def test_all_tasks_have_entries(self):
        """Every task in a split should have an observable oracle entry."""
        data = json.loads(V2_VIEWS_PATH.read_text())
        for view in data["views"]:
            assert view["task_count"] > 0
            assert len(view["task_entries"]) == view["task_count"]


@pytest.mark.skipif(not V2_DIFF_PATH.exists(), reason="V2 difficulty report not available")
class TestV2TopologyIsolation:
    """V2 held-out topology isolation."""

    def test_held_out_isolated_from_dev(self):
        """T_H ∩ T_D = ∅."""
        from collections import defaultdict
        report = json.loads(V2_DIFF_PATH.read_text())
        split_topos = defaultdict(set)
        for t in report["tasks"]:
            split_topos[t["split"]].add(t["transition_topology_sha256"])
        dev = split_topos["structure_dev_v2"]
        held = split_topos["structure_held_out_v2"]
        assert len(held & dev) == 0, f"Held-out overlaps dev by {len(held & dev)}"

    def test_held_out_isolated_from_validation(self):
        """T_H ∩ T_V = ∅."""
        from collections import defaultdict
        report = json.loads(V2_DIFF_PATH.read_text())
        split_topos = defaultdict(set)
        for t in report["tasks"]:
            split_topos[t["split"]].add(t["transition_topology_sha256"])
        val = split_topos["structure_validation_v2"]
        held = split_topos["structure_held_out_v2"]
        assert len(held & val) == 0, f"Held-out overlaps validation by {len(held & val)}"

    def test_v2_isolated_from_i3_4(self):
        """V2 topologies ∩ I3.4 topologies = ∅."""
        from collections import defaultdict
        v2_report = json.loads(V2_DIFF_PATH.read_text())
        i3_4_path = ROOT / "experiments/v2b_i3_3/oracle_tables/v2b_i3_3_difficulty_report_v1.json"
        if not i3_4_path.exists():
            pytest.skip("I3.4 difficulty report not available")
        i3_4_report = json.loads(i3_4_path.read_text())
        v2_topos = {t["transition_topology_sha256"] for t in v2_report["tasks"]}
        i3_4_topos = {t["transition_topology_sha256"] for t in i3_4_report["tasks"]}
        overlap = v2_topos & i3_4_topos
        assert len(overlap) == 0, f"V2 overlaps I3.4 by {len(overlap)} topologies"
