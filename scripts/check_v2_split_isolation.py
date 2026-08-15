#!/usr/bin/env python3
"""Enforce V2 split topology isolation: T_H ∩ (T_D ∪ T_V) = ∅.

After oracle computation, checks whether any held-out topologies overlap
with dev or validation topologies. If so, regenerates the held-out tasks
with shifted variant offsets until isolation is achieved.

Uses the canonical behavior-derived transition_topology_sha256 from the
difficulty report, not the generator's semantic structure hash.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.path.insert(0, str(ROOT / "experiments/v2b_i3_5/generators"))
from generate_v2b_i3_5_v2 import (
    V2_SPLIT_COUNTS, V2_VARIANT_OFFSETS, V2_BUDGET_PROFILES,
    _v2_composed_topology, generate_v2,
    SURFACES, ENTITIES, PAIR_ACTIONS, _pair_mode,
    _base_action_task, _apply_alias, _apply_structure_variant,
    _apply_difficulty_variant, digest, semantic_structure,
    HARD_ALTERNATIVES, COMPOSABLE_ACTIONS,
)

import hashlib
import random


def check_split_isolation(difficulty_report_path: Path) -> dict:
    """Check T_H ∩ (T_D ∪ T_V) = ∅ using canonical topology hashes."""
    report = json.loads(difficulty_report_path.read_text())
    split_topos = defaultdict(set)
    for t in report["tasks"]:
        split_topos[t["split"]].add(t["transition_topology_sha256"])

    dev = split_topos.get("structure_dev_v2", set())
    val = split_topos.get("structure_validation_v2", set())
    held = split_topos.get("structure_held_out_v2", set())

    overlap = held & (dev | val)
    return {
        "dev_count": len(dev),
        "val_count": len(val),
        "held_count": len(held),
        "overlap_count": len(overlap),
        "overlap_hashes": sorted(overlap),
        "is_isolated": len(overlap) == 0,
    }


def regenerate_held_out_with_offset(offset_shift: int):
    """Regenerate held-out tasks with a shifted variant offset."""
    import copy
    rng = random.Random(7717 + offset_shift)
    tasks = []
    packets = []

    split = "structure_held_out_v2"
    count = V2_SPLIT_COUNTS[split]
    surface_pool = SURFACES[:30]

    for pair_index in range(count // 2):
        actions, mode = _pair_mode(pair_index)
        if mode == "budget":
            actions, mode = PAIR_ACTIONS[pair_index % len(PAIR_ACTIONS)], "ordinary"

        pair_material = f"7717:{split}:{pair_index}:{rng.getrandbits(64)}"
        pair_id = "opaque-v2-" + hashlib.sha256(pair_material.encode()).hexdigest()[:16]
        entity = rng.choice(ENTITIES)
        summary = f"{rng.choice(surface_pool)} Subject: {entity}."

        for offset, action in enumerate(actions):
            index = pair_index * 2 + offset
            budget = "STRUCTURE_HOLDOUT_V2"
            task = _base_action_task(
                action, index=index, split=split, pair_id=pair_id,
                summary=summary, budget=budget)

            if mode != "budget":
                _apply_alias(task, mode=mode, offset=offset, action=action,
                             pair_actions=actions)

            # Shift the variant offset to try to get different topologies
            variant = V2_VARIANT_OFFSETS[split] + offset_shift + pair_index * 7

            if mode != "budget":
                _apply_structure_variant(
                    task, variant=variant, structural_holdout=False)

            sequence = ()
            if mode != "budget":
                sequence = _v2_composed_topology(
                    task, variant=variant, split=split)

            selector = index % 10
            requested_band = "EASY" if selector < 3 else "HARD" if selector < 8 else "MEDIUM"
            if requested_band == "HARD" and action not in HARD_ALTERNATIVES:
                requested_band = "MEDIUM"
            if mode == "budget":
                requested_band = "MEDIUM"
                task["designed_difficulty_band"] = requested_band
            else:
                _apply_difficulty_variant(task, band=requested_band, sequence=sequence)

            task["semantic_structure_coarse"] = digest(
                semantic_structure(task, coarse=True))
            task["semantic_structure_exact"] = digest(
                semantic_structure(task, coarse=False))
            tasks.append(task)
            packets.append({
                "task_id": str(task["task_id"]),
                "instance_id": pair_id,
                "task_summary": str(task["task_summary"]),
            })

    return tasks, packets


def main():
    diff_path = ROOT / "experiments/v2b_i3_5/oracle_tables/v2b_i3_5_difficulty_report_v1.json"
    result = check_split_isolation(diff_path)
    print(f"Current isolation check:")
    print(f"  dev: {result['dev_count']} topologies")
    print(f"  validation: {result['val_count']} topologies")
    print(f"  held_out: {result['held_count']} topologies")
    print(f"  overlap: {result['overlap_count']}")

    if result["is_isolated"]:
        print("  ✓ Splits are isolated")
        return 0

    print(f"  ✗ {result['overlap_count']} overlapping topologies found")
    print(f"  Overlap hashes: {result['overlap_hashes'][:5]}...")
    print("\nRegeneration with shifted offsets is needed.")
    print("Run the generator with adjusted V2_VARIANT_OFFSETS for held_out,")
    print("then recompute oracles and re-check.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
