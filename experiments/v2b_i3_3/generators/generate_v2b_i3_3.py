#!/usr/bin/env python3
"""Deterministically generate the frozen V2B-I3.3.1 benchmark corpus."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Mapping


ROOT = Path(__file__).parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hrm_adaptive_memory.common.canonical_json import (  # noqa: E402
    canonical_bytes, canonical_sha256, write_json)


BASE = ROOT / "experiments/v2b/tasks/v2b_i3_metareasoning_benchmark_v1.json"
OUT = ROOT / "experiments/v2b_i3_3"
SEED = 3301
SPLIT_COUNTS = {
    "development": 300,
    "validation": 150,
    "held_out_instance": 100,
    "held_out_surface": 50,
    "held_out_structure": 150,
}
I3_3_BUDGET_PROFILES = {
    "TIGHT": {
        "max_executive_steps": 3, "max_reasoning_tokens": 128,
        "max_retrieval_calls": 0, "max_verification_calls": 0,
        "max_search_calls": 0, "max_elapsed_ms": 3_000,
        "max_monetary_cost_microusd": 0,
    },
    "STANDARD": {
        "max_executive_steps": 5, "max_reasoning_tokens": 256,
        "max_retrieval_calls": 2, "max_verification_calls": 2,
        "max_search_calls": 2, "max_elapsed_ms": 6_000,
        "max_monetary_cost_microusd": 0,
    },
    "GENEROUS": {
        "max_executive_steps": 7, "max_reasoning_tokens": 384,
        "max_retrieval_calls": 3, "max_verification_calls": 3,
        "max_search_calls": 3, "max_elapsed_ms": 9_000,
        "max_monetary_cost_microusd": 0,
    },
}
PAIR_ACTIONS = (
    ("ANSWER", "RETRIEVE"),
    ("VERIFY", "SEARCH_MORE"),
    ("REASON_MORE", "STOP"),
    ("ANSWER", "SEARCH_MORE"),
    ("RETRIEVE", "REASON_MORE"),
    ("DEFER", "STOP"),
    ("REASON_MORE", "STOP"),
)
SURFACE_STEMS = (
    "Choose", "Select", "Identify", "Determine", "Commit to",
    "Return", "Nominate", "Resolve", "Evaluate", "Issue",
)
SURFACE_TAILS = (
    "the next bounded cognitive action using the public evidence packet.",
    "one permitted control action under the remaining resource budget.",
    "the next auditable operation without access to private task state.",
    "one bounded action from the frozen seven-action vocabulary.",
)
SURFACES = tuple(f"{stem} {tail}" for stem in SURFACE_STEMS for tail in SURFACE_TAILS)
ENTITIES = tuple(f"case-{letter}{number:02d}" for letter in "ABCDEFGH" for number in range(10))
NONTERMINAL_ACTIONS = ("RETRIEVE", "VERIFY", "SEARCH_MORE", "REASON_MORE")
DECOY_EFFECTS = (
    {"composition_complete": "false"},
    {"verification_state": "UNVERIFIED"},
    {"temporal_status": "STALE"},
    {"unresolved_conflict": "true"},
    {"provenance_count": "0"},
)


def digest(value: object) -> str:
    return canonical_sha256(value)


def _base_action_task(action: str, *, index: int, split: str, pair_id: str,
                      summary: str, budget: str) -> dict[str, object]:
    latent: dict[str, object] = {
        "verification_state": "SUFFICIENT", "temporal_status": "CURRENT",
        "unresolved_conflict": False, "composition_complete": True,
        "expected_terminal": "ANSWER", "required_provenance_count": 0,
        "conflict_resolvable": False, "initial_prior_outcomes": [],
    }
    effects: dict[str, dict[str, str]] = {}
    provenance = 2
    channel = "verification"
    if action == "ANSWER":
        channel = "state_irrelevant"
    elif action == "RETRIEVE":
        latent.update(verification_state="MISSING", temporal_status="UNKNOWN")
        effects["RETRIEVE"] = {
            "verification_state": "SUFFICIENT", "temporal_status": "CURRENT"}
        provenance = 0
        channel = "history" if index % 3 == 0 else "verification"
    elif action == "VERIFY":
        mode = index % 3
        if mode == 0:
            latent["verification_state"] = "UNVERIFIED"
            effects["VERIFY"] = {"verification_state": "SUFFICIENT"}
            channel = "verification"
        elif mode == 1:
            latent["temporal_status"] = "STALE"
            effects["VERIFY"] = {"temporal_status": "CURRENT"}
            channel = "temporal"
        else:
            latent["required_provenance_count"] = 3
            provenance = 1
            effects["VERIFY"] = {"provenance_count": "3"}
            channel = "provenance"
    elif action == "SEARCH_MORE":
        provenance = 0
        if (index // 2) % 2:
            latent.update(unresolved_conflict=True, conflict_resolvable=True)
            effects["SEARCH_MORE"] = {"unresolved_conflict": "false"}
            channel = "conflict"
        else:
            latent.update(
                verification_state="MISSING", temporal_status="UNKNOWN",
                initial_prior_outcomes=["RETRIEVE_FAILED", "RETRIEVE_FAILED"])
            effects["RETRIEVE"] = {}
            effects["SEARCH_MORE"] = {
                "verification_state": "SUFFICIENT", "temporal_status": "CURRENT"}
            channel = "history"
    elif action == "REASON_MORE":
        latent["composition_complete"] = False
        effects["REASON_MORE"] = {"composition_complete": "true"}
        channel = "composition"
    elif action == "DEFER":
        latent.update(
            verification_state="MISSING", temporal_status="UNKNOWN",
            expected_terminal="DEFER")
        provenance = 0
        if index % 4 == 1:
            latent.update(
                verification_state="SUFFICIENT", temporal_status="CURRENT",
                unresolved_conflict=True)
            channel = "conflict"
        else:
            channel = "irreducible"
    elif action == "STOP":
        latent["expected_terminal"] = "STOP"
        channel = "state_irrelevant"
        summary += " NO_FINAL_ASSERTION_REQUESTED; terminate without a final assertion."
    else:
        raise ValueError(action)
    return {
        "task_id": f"i3_3_{split}_{index:04d}", "split": split,
        "category": f"{channel}_{action.lower()}", "task_summary": summary,
        "high_stakes": False, "budget_profile": budget,
        "observable_provenance_count": provenance,
        "latent": latent, "action_effects": effects,
        "designed_optimal_action": action, "cognitive_channel": channel,
        "generator_pair_id": pair_id,
    }


def _apply_alias(task: dict[str, object], *, mode: str, offset: int, action: str) -> None:
    if mode == "budget":
        task["latent"] = {
            "verification_state": "UNVERIFIED", "temporal_status": "CURRENT",
            "unresolved_conflict": False, "composition_complete": True,
            "expected_terminal": "ANSWER", "required_provenance_count": 0,
            "conflict_resolvable": False, "initial_prior_outcomes": [],
        }
        task["observable_provenance_count"] = 1
        task["action_effects"] = {"VERIFY": {"verification_state": "SUFFICIENT"}}
        task["cognitive_channel"] = "verification_x_budget"
    elif mode == "irreducible":
        task["latent"] = {
            "verification_state": "MISSING", "temporal_status": "UNKNOWN",
            "unresolved_conflict": False, "composition_complete": True,
            "expected_terminal": "ANSWER", "required_provenance_count": 0,
            "conflict_resolvable": False, "initial_prior_outcomes": [],
        }
        task["observable_provenance_count"] = 0
        task["action_effects"] = {
            action: {"verification_state": "SUFFICIENT", "temporal_status": "CURRENT"}}
        task["cognitive_channel"] = "irreducible_alias"
    elif mode == "conflict":
        task["latent"] = {
            "verification_state": "SUFFICIENT", "temporal_status": "CURRENT",
            "unresolved_conflict": True, "composition_complete": True,
            "expected_terminal": "ANSWER" if offset == 0 else "DEFER",
            "required_provenance_count": 0, "conflict_resolvable": offset == 0,
            "initial_prior_outcomes": [],
        }
        task["observable_provenance_count"] = 2
        task["action_effects"] = (
            {"SEARCH_MORE": {"unresolved_conflict": "false"}} if offset == 0 else {})
        task["cognitive_channel"] = "conflict_alias"
    elif mode == "temporal":
        task["latent"] = {
            "verification_state": "SUFFICIENT",
            "temporal_status": "CURRENT" if offset == 0 else "STALE",
            "unresolved_conflict": False, "composition_complete": True,
            "expected_terminal": "ANSWER", "required_provenance_count": 0,
            "conflict_resolvable": False, "initial_prior_outcomes": [],
        }
        task["observable_provenance_count"] = 2
        task["action_effects"] = (
            {} if offset == 0 else {"VERIFY": {"temporal_status": "CURRENT"}})
        task["cognitive_channel"] = "temporal_alias"
    elif mode == "verification":
        task["latent"] = {
            "verification_state": "SUFFICIENT" if offset == 0 else "UNVERIFIED",
            "temporal_status": "CURRENT", "unresolved_conflict": False,
            "composition_complete": True, "expected_terminal": "ANSWER",
            "required_provenance_count": 0, "conflict_resolvable": False,
            "initial_prior_outcomes": [],
        }
        task["observable_provenance_count"] = 2
        task["action_effects"] = (
            {} if offset == 0 else {"VERIFY": {"verification_state": "SUFFICIENT"}})
        task["cognitive_channel"] = "verification_alias"
    elif mode == "provenance":
        task["latent"] = {
            "verification_state": "SUFFICIENT", "temporal_status": "CURRENT",
            "unresolved_conflict": False, "composition_complete": True,
            "expected_terminal": "ANSWER",
            "required_provenance_count": 2 if offset == 0 else 3,
            "conflict_resolvable": False, "initial_prior_outcomes": [],
        }
        task["observable_provenance_count"] = 3 if offset == 0 else 1
        task["action_effects"] = (
            {} if offset == 0 else {"VERIFY": {"provenance_count": "3"}})
        task["cognitive_channel"] = "provenance_alias"
    elif mode == "history":
        task["latent"] = {
            "verification_state": "MISSING", "temporal_status": "UNKNOWN",
            "unresolved_conflict": False, "composition_complete": True,
            "expected_terminal": "ANSWER", "required_provenance_count": 0,
            "conflict_resolvable": False,
            "initial_prior_outcomes": ([] if offset == 0 else
                                       ["RETRIEVE_FAILED", "RETRIEVE_FAILED"]),
        }
        task["observable_provenance_count"] = 0
        task["action_effects"] = {
            action: {"verification_state": "SUFFICIENT", "temporal_status": "CURRENT"}}
        task["cognitive_channel"] = "history_alias"
    task["category"] = f"{task['cognitive_channel']}_{action.lower()}"


def _apply_structure_variant(task: dict[str, object], *, variant: int,
                             structural_holdout: bool) -> None:
    """Alter real transition semantics without changing the intended solution."""
    target = str(task["designed_optimal_action"])
    effects = {str(key): dict(value) for key, value in
               dict(task["action_effects"]).items()}
    if task["cognitive_channel"] == "verification_x_budget":
        effects["RETRIEVE"] = {"composition_complete": "false"}
        if structural_holdout:
            effects["SEARCH_MORE"] = {"temporal_status": "STALE"}
        task["action_effects"] = effects
        return
    candidates = [action for action in NONTERMINAL_ACTIONS if action != target]
    count = 2 if structural_holdout else 1
    for position in range(count):
        decoy = candidates[(variant + position) % len(candidates)]
        # Never overwrite a task-defining transition.  Pick the next decoy if
        # an alias already uses this action.
        for _ in candidates:
            if decoy not in effects:
                break
            decoy = candidates[(candidates.index(decoy) + 1) % len(candidates)]
        if decoy not in effects:
            effects[decoy] = dict(DECOY_EFFECTS[(variant + position) % len(DECOY_EFFECTS)])
    task["action_effects"] = effects
    # This is observable executive history, not a hidden nonce.  It changes
    # the Markov observation and is therefore part of the exact semantics.
    latent = dict(task["latent"])
    history = list(latent.get("initial_prior_outcomes", []))
    if structural_holdout and variant % 3:
        history.extend(["BOUNDED_ACTION_NO_GAIN"] * (1 + variant % 2))
    latent["initial_prior_outcomes"] = history
    task["latent"] = latent


def semantic_structure(task: Mapping[str, object], *, coarse: bool) -> dict[str, object]:
    latent = dict(task["latent"])
    effects = {str(action): dict(effect)
               for action, effect in sorted(dict(task["action_effects"]).items())}
    if coarse:
        effects = {action: sorted(effect) for action, effect in effects.items()}
        latent = {
            "verification_state": latent["verification_state"],
            "temporal_status": latent["temporal_status"],
            "unresolved_conflict": latent["unresolved_conflict"],
            "composition_complete": latent["composition_complete"],
            "expected_terminal": latent["expected_terminal"],
        }
    return {
        "schema": "DAPH_V2B_I3_3_SEMANTIC_STRUCTURE_V1",
        "level": "coarse" if coarse else "exact",
        "budget_profile": task["budget_profile"],
        "high_stakes": task["high_stakes"],
        "cognitive_channel": task["cognitive_channel"],
        "observable_provenance_count": task["observable_provenance_count"],
        "latent": latent,
        "action_effects": effects,
    }


def _pair_mode(pair_index: int) -> tuple[tuple[str, str], str]:
    budget = pair_index % 13 in {0, 5}
    modes = (
        (17, "irreducible", ("RETRIEVE", "VERIFY")),
        (19, "conflict", ("SEARCH_MORE", "DEFER")),
        (23, "temporal", ("ANSWER", "VERIFY")),
        (29, "verification", ("ANSWER", "VERIFY")),
        (31, "provenance", ("ANSWER", "VERIFY")),
        (37, "history", ("RETRIEVE", "SEARCH_MORE")),
    )
    if budget:
        return ("DEFER", "VERIFY"), "budget"
    for divisor, mode, actions in modes:
        if pair_index % divisor == 0:
            return actions, mode
    return PAIR_ACTIONS[pair_index % len(PAIR_ACTIONS)], "ordinary"


def generate() -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    rng = random.Random(SEED)
    tasks: list[dict[str, object]] = []
    packets: list[dict[str, str]] = []
    for split, count in SPLIT_COUNTS.items():
        surface_pool = SURFACES[30:] if split == "held_out_surface" else SURFACES[:30]
        for pair_index in range(count // 2):
            actions, mode = _pair_mode(pair_index)
            pair_material = f"{SEED}:{split}:{pair_index}:{rng.getrandbits(64)}"
            pair_id = "opaque-" + hashlib.sha256(pair_material.encode()).hexdigest()[:16]
            entity = rng.choice(ENTITIES)
            summary = f"{rng.choice(surface_pool)} Subject: {entity}."
            default_budget = "GENEROUS" if pair_index % 5 == 0 else "STANDARD"
            for offset, action in enumerate(actions):
                index = pair_index * 2 + offset
                budget = ("TIGHT" if mode == "budget" and offset == 0 else
                          "GENEROUS" if mode == "budget" else default_budget)
                task = _base_action_task(
                    action, index=index, split=split, pair_id=pair_id,
                    summary=summary, budget=budget)
                if mode != "ordinary":
                    _apply_alias(task, mode=mode, offset=offset, action=action)
                variant = (pair_index % 60 if split in {
                    "development", "held_out_instance", "held_out_surface"}
                           else 100 + pair_index if split == "validation"
                           else 300 + pair_index)
                _apply_structure_variant(
                    task, variant=variant,
                    structural_holdout=split in {"validation", "held_out_structure"})
                task["semantic_structure_coarse"] = digest(
                    semantic_structure(task, coarse=True))
                task["semantic_structure_exact"] = digest(
                    semantic_structure(task, coarse=False))
                tasks.append(task)
                packets.append({
                    "task_id": str(task["task_id"]), "instance_id": pair_id,
                    "task_summary": str(task["task_summary"]),
                })
    return tasks, packets


def _structure_report(tasks: list[dict[str, object]]) -> dict[str, object]:
    exact_by_split: dict[str, set[str]] = defaultdict(set)
    coarse_by_split: dict[str, set[str]] = defaultdict(set)
    for task in tasks:
        split = str(task["split"])
        exact_by_split[split].add(str(task["semantic_structure_exact"]))
        coarse_by_split[split].add(str(task["semantic_structure_coarse"]))
    development_exact = exact_by_split["development"]
    development_coarse = coarse_by_split["development"]
    novelty = {
        split: {
            "exact_unseen_from_development": len(values - development_exact),
            "coarse_unseen_from_development": len(coarse_by_split[split] - development_coarse),
        }
        for split, values in sorted(exact_by_split.items()) if split != "development"
    }
    overlap = {
        left: {right: len(exact_by_split[left] & exact_by_split[right])
               for right in SPLIT_COUNTS}
        for left in SPLIT_COUNTS
    }
    return {
        "schema": "DAPH_V2B_I3_3_1_STRUCTURAL_DIVERSITY_REPORT_V1",
        "status": "FROZEN_BENCHMARK_NOT_EXECUTIVE_RESULT",
        "unique_exact_structures": {
            split: len(values) for split, values in sorted(exact_by_split.items())},
        "unique_coarse_structures": {
            split: len(values) for split, values in sorted(coarse_by_split.items())},
        "novelty_against_development": novelty,
        "exact_structure_overlap_matrix": overlap,
    }


def main() -> None:
    base = json.loads(BASE.read_text())
    tasks, packets = generate()
    private = {key: base[key] for key in (
        "schema", "status", "protocol", "utility_weights", "action_costs")}
    private["budget_profiles"] = I3_3_BUDGET_PROFILES
    private.update({
        "benchmark_id": "v2b_i3_3_1_benchmark_integrity_v1",
        "scope": "Frozen deterministic I3.3.1 benchmark; no model-controller result.",
        "tasks": tasks,
    })
    packet_payload = {
        "schema": "DAPH_V2B_I3_3_CONTROLLER_PACKETS_V1",
        "status": "FROZEN_FOR_DEVELOPMENT", "packets": packets,
    }
    families = {
        "schema": "DAPH_V2B_I3_3_TASK_FAMILIES_V1",
        "status": "FROZEN_FOR_DEVELOPMENT",
        "generator_revision": "v2b-i3.3.1-generator-v1",
        "operational_seed": SEED,
        "seed_controls": ["surface_template", "entity", "opaque_instance_id"],
        "pair_action_cycle": PAIR_ACTIONS,
        "channels": sorted({str(task["cognitive_channel"]) for task in tasks}),
    }
    surfaces = {
        "schema": "DAPH_V2B_I3_3_SURFACE_TEMPLATES_V1",
        "status": "FROZEN_FOR_DEVELOPMENT",
        "development_templates": SURFACES[:30],
        "held_out_surface_templates": SURFACES[30:],
    }
    split_payload = {
        "schema": "DAPH_V2B_I3_3_SPLITS_V1", "status": "FROZEN_FOR_DEVELOPMENT",
        "splits": {
            split: [{"task_id": task["task_id"], "task_sha256": digest(task)}
                    for task in tasks if task["split"] == split]
            for split in SPLIT_COUNTS
        },
    }
    counts = Counter(str(task["designed_optimal_action"]) for task in tasks)
    channels = Counter(str(task["cognitive_channel"]) for task in tasks)
    budgets = Counter(str(task["budget_profile"]) for task in tasks)
    report = {
        "schema": "DAPH_V2B_I3_3_BALANCE_REPORT_V1", "task_count": len(tasks),
        "split_counts": dict(SPLIT_COUNTS),
        "designed_action_counts_non_authoritative": dict(sorted(counts.items())),
        "cognitive_channel_counts": dict(sorted(channels.items())),
        "budget_counts": dict(sorted(budgets.items())),
        "budget_sensitive_pair_count": sum(
            1 for task in tasks if task["cognitive_channel"] == "verification_x_budget") // 2,
        "generator_output_sha256": digest(tasks),
        "authority_note": "Exact latent oracle, not generator intent, defines optimal actions.",
    }
    structure_report = _structure_report(tasks)
    write_json(OUT / "private/v2b_i3_3_tasks_v1.json", private)
    write_json(OUT / "controller_packets/v2b_i3_3_controller_packets_v1.json", packet_payload)
    write_json(OUT / "task_families/v2b_i3_3_families_v1.json", families)
    write_json(OUT / "surface_templates/v2b_i3_3_surface_templates_v1.json", surfaces)
    write_json(OUT / "splits/v2b_i3_3_splits_v1.json", split_payload)
    write_json(OUT / "reports/v2b_i3_3_balance_report_v1.json", report)
    write_json(OUT / "reports/v2b_i3_3_1_structural_diversity_report_v1.json",
               structure_report)
    manifest = {
        "schema": "DAPH_V2B_I3_3_BENCHMARK_MANIFEST_V1",
        "status": "FROZEN_FOR_BENCHMARK_QUALIFICATION",
        "benchmark_id": "v2b_i3_3_1_benchmark_integrity_v1",
        "integrity_revision": "V2B-I3.3.1",
        "private_environment_path": "../private/v2b_i3_3_tasks_v1.json",
        "controller_packets_path": "../controller_packets/v2b_i3_3_controller_packets_v1.json",
        "task_families_path": "../task_families/v2b_i3_3_families_v1.json",
        "split_definitions_path": "../splits/v2b_i3_3_splits_v1.json",
        "surface_templates_path": "../surface_templates/v2b_i3_3_surface_templates_v1.json",
        "protocol_path": "../../../configs/v2b_i3_2_2_protocol_v1.json",
        "balance_report_path": "../reports/v2b_i3_3_balance_report_v1.json",
        "structural_diversity_report_path": "../reports/v2b_i3_3_1_structural_diversity_report_v1.json",
        "oracle_cache_manifest_path": "../oracle_tables/v2b_i3_3_oracle_cache_manifest_v1.json",
        "claim_boundary": "Frozen benchmark integrity evidence only; no executive result.",
    }
    write_json(OUT / "manifests/v2b_i3_3_benchmark_manifest_v1.json", manifest)
    print(canonical_bytes(report).decode())


if __name__ == "__main__":
    main()
