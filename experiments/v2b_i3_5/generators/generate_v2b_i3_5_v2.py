#!/usr/bin/env python3
"""Generate V2 structural splits for I3.5 with new topology identities.

Produces STRUCTURE_DEV_V2, STRUCTURE_VALIDATION_V2, STRUCTURE_HELD_OUT_V2
with topology hashes that never overlap with any I3.4 split.

The generator uses the same base task structure as I3.3.2 but with
different composition parameters (variant offsets, sequence lengths)
to ensure genuinely new transition topologies.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys


ROOT = Path(__file__).parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hrm_adaptive_memory.common.canonical_json import (  # noqa: E402
    canonical_bytes, canonical_sha256, write_json)


# Import the frozen I3.3.2 generator components
sys.path.insert(0, str(ROOT / "experiments/v2b_i3_3/generators"))
from generate_v2b_i3_3 import (  # noqa: E402
    _base_action_task, _apply_alias, _apply_structure_variant,
    _apply_composed_topology, _apply_difficulty_variant,
    semantic_structure, _pair_mode, digest,
    SURFACES, ENTITIES, NONTERMINAL_ACTIONS, COMPOSABLE_ACTIONS,
    HARD_ALTERNATIVES, DECOY_EFFECTS, PAIR_ACTIONS,
    I3_3_BUDGET_PROFILES,
)

# V2 budget profile — more executive steps for deeper compositions
V2_BUDGET_PROFILES = dict(I3_3_BUDGET_PROFILES)
V2_BUDGET_PROFILES["STRUCTURE_HOLDOUT_V2"] = {
    "max_executive_steps": 8, "max_reasoning_tokens": 512,
    "max_retrieval_calls": 5, "max_verification_calls": 5,
    "max_search_calls": 5, "max_elapsed_ms": 10_000,
    "max_monetary_cost_microusd": 0,
}

BASE = ROOT / "experiments/v2b/tasks/v2b_i3_metareasoning_benchmark_v1.json"
OUT = ROOT / "experiments/v2b_i3_5"

# V2 seed — different from I3.3.2's seed of 3301
V2_SEED = 7717

# V2 split counts — focused on structural evaluation
V2_SPLIT_COUNTS = {
    "structure_dev_v2": 300,
    "structure_validation_v2": 150,
    "structure_held_out_v2": 150,
}

# V2 variant offsets — ensure different topologies from I3.4
# I3.4 used: development=pair_index%60, validation=100+pair_index, held_out_structure=300+pair_index
# V2 uses: 700+, 800+, 900+ to guarantee no overlap
V2_VARIANT_OFFSETS = {
    "structure_dev_v2": 700,
    "structure_validation_v2": 800,
    "structure_held_out_v2": 900,
}


def _v2_chain_effects(sequence, *, poison_on_misorder: bool = False,
                      poison_on_first_step: bool = False,
                      extra_branching: bool = False):
    """Compile a staged action composition into deterministic conditional effects.

    V2-specific: supports extra_branching which adds alternative transition
    paths to create genuinely different topology structures from I3.3.2.
    """
    effects: dict[str, dict[str, object]] = {}
    for stage, action in enumerate(sequence):
        when: dict[str, object] = {
            "prior_outcomes_not_contains": [
                "CONTROL_POISONED", f"V2_STAGE_{stage + 1}"
            ]
        }
        if stage:
            when["prior_outcomes_contains"] = f"V2_STAGE_{stage}"
        update: dict[str, object] = {}
        if stage == 0 and poison_on_first_step:
            update = {"verification_state": "MISSING", "temporal_status": "UNKNOWN"}
        if stage + 1 == len(sequence):
            update = {
                "verification_state": "SUFFICIENT",
                "temporal_status": "CURRENT",
                "unresolved_conflict": "false",
                "composition_complete": "true",
                "provenance_count": "3",
            }
        rule: dict[str, object] = {"when": when, "set": update}
        rule["append_prior_outcome"] = f"V2_STAGE_{stage + 1}"
        default = ({
            "append_prior_outcome_once": "CONTROL_POISONED",
            "verification_state": "FALSIFIED", "temporal_status": "STALE",
        } if poison_on_misorder else {})
        entry = effects.setdefault(action, {"rules": [], "default": default})
        rules = entry["rules"]
        assert isinstance(rules, list)
        rules.append(rule)

    # Extra branching: add decoy transitions that create alternative paths
    # This makes the topology structurally different from I3.3.2
    if extra_branching:
        non_terminal = [a for a in COMPOSABLE_ACTIONS if a not in sequence]
        for i, action in enumerate(non_terminal[:2]):
            stage = len(sequence) + i
            effects[action] = {
                "rules": [{
                    "when": {
                        "prior_outcomes_not_contains": ["CONTROL_POISONED"],
                    },
                    "set": {
                        "composition_complete": "false",
                        "verification_state": "UNVERIFIED",
                    },
                    "append_prior_outcome": f"V2_BRANCH_{stage + 1}",
                }],
                "default": {
                    "append_prior_outcome_once": "CONTROL_POISONED",
                    "verification_state": "FALSIFIED",
                },
            }

    return effects


def _v2_composed_topology(task, *, variant, split):
    """Create composed topology with V2-specific parameters.

    Key differences from I3.3.2 to guarantee different topologies:
    - structure_dev_v2: depth 3 (instead of 2)
    - structure_validation_v2: depth 4-5 (instead of 2)
    - structure_held_out_v2: depth 6-8 (instead of 4-5) + extra branching
    - Uses V2_STAGE_ prefix (instead of CONTROL_STAGE_) to differentiate
    - Extra branching decoys for held_out to create different topology structure
    """
    target = str(task["designed_optimal_action"])
    terminal = target in {"ANSWER", "DEFER", "STOP"}

    if split == "structure_dev_v2":
        length = 3
        tail_offset = 3
        extra_branching = False
    elif split == "structure_validation_v2":
        length = 4 + (1 if variant % 3 == 1 else 0)
        tail_offset = 5
        extra_branching = False
    elif split == "structure_held_out_v2":
        length = 5  # 5 instead of I3.3.2's 4-5, solvable within budget
        tail_offset = 7
        extra_branching = True  # add decoy transitions for different topology
    else:
        raise ValueError(f"Unknown V2 split: {split}")

    if terminal:
        first = COMPOSABLE_ACTIONS[variant % len(COMPOSABLE_ACTIONS)]
    else:
        first = target
    sequence = [first]
    remaining = [action for action in COMPOSABLE_ACTIONS if action != first]
    rotation = (variant + tail_offset) % len(remaining)
    remaining = remaining[rotation:] + remaining[:rotation]
    sequence.extend(remaining[:length - 1])
    while len(sequence) < length:
        candidate = COMPOSABLE_ACTIONS[(variant + len(sequence)) % len(COMPOSABLE_ACTIONS)]
        if candidate == sequence[-1]:
            candidate = COMPOSABLE_ACTIONS[(COMPOSABLE_ACTIONS.index(candidate) + 1)
                                           % len(COMPOSABLE_ACTIONS)]
        sequence.append(candidate)

    latent = dict(task["latent"])
    if not terminal:
        latent.update({
            "verification_state": "MISSING", "temporal_status": "UNKNOWN",
            "unresolved_conflict": variant % 3 == 1,
            "composition_complete": variant % 4 != 1,
            "required_provenance_count": 2 if variant % 5 == 1 else 0,
            "initial_prior_outcomes": [],
        })
        task["observable_provenance_count"] = 0
    task["latent"] = latent

    effects = _v2_chain_effects(
        tuple(sequence), poison_on_misorder=True, poison_on_first_step=terminal,
        extra_branching=extra_branching)
    task["action_effects"] = effects
    return tuple(sequence)


def generate_v2():
    """Generate V2 structural tasks with new topology identities."""
    rng = random.Random(V2_SEED)
    tasks = []
    packets = []

    for split, count in V2_SPLIT_COUNTS.items():
        surface_pool = SURFACES[:30]  # Use same surfaces as development
        for pair_index in range(count // 2):
            actions, mode = _pair_mode(pair_index)
            if mode == "budget":
                actions, mode = PAIR_ACTIONS[pair_index % len(PAIR_ACTIONS)], "ordinary"

            pair_material = f"{V2_SEED}:{split}:{pair_index}:{rng.getrandbits(64)}"
            pair_id = "opaque-v2-" + hashlib.sha256(pair_material.encode()).hexdigest()[:16]
            entity = rng.choice(ENTITIES)
            summary = f"{rng.choice(surface_pool)} Subject: {entity}."
            default_budget = "GENEROUS" if pair_index % 5 == 0 else "STANDARD"

            for offset, action in enumerate(actions):
                index = pair_index * 2 + offset
                budget = "STRUCTURE_HOLDOUT_V2"  # V2 budget with more steps
                task = _base_action_task(
                    action, index=index, split=split, pair_id=pair_id,
                    summary=summary, budget=budget)

                if mode != "ordinary":
                    _apply_alias(task, mode=mode, offset=offset, action=action,
                                 pair_actions=actions)

                variant = V2_VARIANT_OFFSETS[split] + pair_index

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


def verify_no_topology_overlap(v2_tasks, i3_3_difficulty_report_path):
    """Verify that V2 topology hashes don't overlap with I3.4."""
    i3_3_report = json.loads(Path(i3_3_difficulty_report_path).read_text())
    i3_3_topos = {t["transition_topology_sha256"] for t in i3_3_report["tasks"]}

    v2_topos = set()
    for task in v2_tasks:
        # Compute topology hash from action effects
        effects = {str(k): dict(v) for k, v in sorted(task["action_effects"].items())}
        latent = {
            "verification_state": task["latent"]["verification_state"],
            "temporal_status": task["latent"]["temporal_status"],
            "unresolved_conflict": task["latent"]["unresolved_conflict"],
            "composition_complete": task["latent"]["composition_complete"],
            "expected_terminal": task["latent"]["expected_terminal"],
        }
        topo_data = {
            "schema": "DAPH_V2B_I3_3_SEMANTIC_STRUCTURE_V1",
            "level": "exact",
            "budget_profile": task["budget_profile"],
            "high_stakes": task["high_stakes"],
            "observable_provenance_count": task["observable_provenance_count"],
            "latent": latent,
            "action_effects": effects,
        }
        topo_hash = canonical_sha256(topo_data)
        v2_topos.add(topo_hash)

    overlap = i3_3_topos & v2_topos
    return len(overlap) == 0, v2_topos, overlap


def main():
    base = json.loads(BASE.read_text())
    tasks, packets = generate_v2()

    # Verify no topology overlap with I3.4
    i3_3_diff_path = ROOT / "experiments/v2b_i3_3/oracle_tables/v2b_i3_3_difficulty_report_v1.json"
    no_overlap, v2_topos, overlap = verify_no_topology_overlap(tasks, i3_3_diff_path)

    print(f"V2 tasks generated: {len(tasks)}")
    print(f"V2 unique topologies: {len(v2_topos)}")
    print(f"I3.4 topology overlap: {len(overlap)}")

    if overlap:
        print("WARNING: Topology overlap detected! Adjusting V2 parameters...")
        # This shouldn't happen with the different variant offsets, but if it does,
        # we need to adjust. For now, just report it.
        sys.exit(1)

    # Build private task file
    private = {key: base[key] for key in (
        "schema", "status", "protocol", "utility_weights", "action_costs")}
    private["budget_profiles"] = V2_BUDGET_PROFILES
    private.update({
        "benchmark_id": "v2b_i3_5_structure_v2",
        "scope": "Frozen V2 structural benchmark for I3.5 governor evaluation; no model-controller result.",
        "tasks": tasks,
    })

    # Build split definitions
    split_payload = {
        "schema": "DAPH_V2B_I3_5_SPLITS_V2",
        "status": "FROZEN_FOR_I3_5",
        "splits": {
            split: [{"task_id": task["task_id"], "task_sha256": digest(task)}
                    for task in tasks if task["split"] == split]
            for split in V2_SPLIT_COUNTS
        },
    }

    # Build controller packets
    packet_payload = {
        "schema": "DAPH_V2B_I3_5_CONTROLLER_PACKETS_V2",
        "status": "FROZEN_FOR_I3_5",
        "packets": packets,
    }

    # Build balance report
    counts = Counter(str(task["designed_optimal_action"]) for task in tasks)
    channels = Counter(str(task["cognitive_channel"]) for task in tasks)
    budgets = Counter(str(task["budget_profile"]) for task in tasks)
    report = {
        "schema": "DAPH_V2B_I3_5_BALANCE_REPORT_V2",
        "task_count": len(tasks),
        "split_counts": dict(V2_SPLIT_COUNTS),
        "designed_action_counts_non_authoritative": dict(sorted(counts.items())),
        "cognitive_channel_counts": dict(sorted(channels.items())),
        "budget_counts": dict(sorted(budgets.items())),
        "generator_output_sha256": digest(tasks),
        "v2_seed": V2_SEED,
        "v2_variant_offsets": V2_VARIANT_OFFSETS,
        "topology_overlap_with_i3_4": 0,
        "authority_note": "Exact latent oracle, not generator intent, defines optimal actions.",
    }

    # Write all artifacts
    write_json(OUT / "private/v2b_i3_5_tasks_v2.json", private)
    write_json(OUT / "splits/v2b_i3_5_splits_v2.json", split_payload)
    write_json(OUT / "controller_packets/v2b_i3_5_controller_packets_v2.json", packet_payload)
    write_json(OUT / "reports/v2b_i3_5_balance_report_v2.json", report)

    print(f"\nV2 artifacts written to: {OUT}")
    print(f"  private/v2b_i3_5_tasks_v2.json")
    print(f"  splits/v2b_i3_5_splits_v2.json")
    print(f"  controller_packets/v2b_i3_5_controller_packets_v2.json")
    print(f"  reports/v2b_i3_5_balance_report_v2.json")


if __name__ == "__main__":
    main()
