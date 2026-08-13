#!/usr/bin/env python3
"""Deterministically generate the frozen V2B-I3.3.2 benchmark corpus."""
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
    "STRUCTURE_HOLDOUT": {
        "max_executive_steps": 6, "max_reasoning_tokens": 512,
        "max_retrieval_calls": 4, "max_verification_calls": 4,
        "max_search_calls": 4, "max_elapsed_ms": 8_000,
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
COMPOSABLE_ACTIONS = ("RETRIEVE", "VERIFY", "SEARCH_MORE", "REASON_MORE")
HARD_ALTERNATIVES = {
    "RETRIEVE": "SEARCH_MORE",
    "SEARCH_MORE": "VERIFY",
    "REASON_MORE": "RETRIEVE",
}


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


def _apply_alias(task: dict[str, object], *, mode: str, offset: int, action: str,
                 pair_actions: tuple[str, str]) -> None:
    if mode == "budget":
        target = pair_actions[1]
        reasoning = target == "REASON_MORE"
        task["latent"] = {
            "verification_state": "SUFFICIENT" if reasoning else "MISSING",
            "temporal_status": "CURRENT" if reasoning else "UNKNOWN",
            "unresolved_conflict": False, "composition_complete": True,
            "expected_terminal": "ANSWER", "required_provenance_count": 0,
            "conflict_resolvable": False, "initial_prior_outcomes": [],
        }
        if reasoning:
            task["latent"]["composition_complete"] = False
        task["observable_provenance_count"] = 1
        task["action_effects"] = {target: (
            {"composition_complete": "true"} if reasoning else
            {"verification_state": "SUFFICIENT", "temporal_status": "CURRENT"})}
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
        decoy = next(action for action in NONTERMINAL_ACTIONS if action != target)
        effects[decoy] = {"composition_complete": "false"}
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


def _chain_effects(sequence: tuple[str, ...], *, poison_on_misorder: bool = False,
                   poison_on_first_step: bool = False) -> dict[str, dict[str, object]]:
    """Compile a staged action composition into deterministic conditional effects."""
    effects: dict[str, dict[str, object]] = {}
    for stage, action in enumerate(sequence):
        when: dict[str, object] = {
            "prior_outcomes_not_contains": [
                "CONTROL_POISONED", f"CONTROL_STAGE_{stage + 1}"
            ]
        }
        if stage:
            when["prior_outcomes_contains"] = f"CONTROL_STAGE_{stage}"
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
        rule["append_prior_outcome"] = f"CONTROL_STAGE_{stage + 1}"
        default = ({
            "append_prior_outcome_once": "CONTROL_POISONED",
            "verification_state": "FALSIFIED", "temporal_status": "STALE",
        } if poison_on_misorder else {})
        entry = effects.setdefault(action, {"rules": [], "default": default})
        rules = entry["rules"]
        assert isinstance(rules, list)
        rules.append(rule)
    return effects


def _apply_composed_topology(task: dict[str, object], *, variant: int,
                             split: str) -> tuple[str, ...]:
    """Create real multi-step compositions reserved by scientific split."""
    target = str(task["designed_optimal_action"])
    terminal = target in {"ANSWER", "DEFER", "STOP"}
    # Validation owns depth-2 operation programs. Final structure-held-out
    # owns depth-3/4 programs, so it cannot reuse a validation topology merely
    # through a different channel label or decoy parameter.
    if split == "validation":
        length = 2
        tail_offset = 1
    elif split == "held_out_structure":
        length = 4 + (1 if variant % 3 == 0 else 0)
        tail_offset = 2
    else:
        raise ValueError("composed topology used outside structural split")
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
            "unresolved_conflict": variant % 3 == 0,
            "composition_complete": variant % 4 != 0,
            "required_provenance_count": 2 if variant % 5 == 0 else 0,
            "initial_prior_outcomes": [],
        })
        task["observable_provenance_count"] = 0
    task["latent"] = latent
    effects = _chain_effects(
        tuple(sequence), poison_on_misorder=True, poison_on_first_step=terminal)
    # Terminal tasks remain immediately solvable, while their counterfactual
    # control graph contains the withheld composition. The composed program
    # replaces older generator effects so no legacy decoy can break a stage.
    task["action_effects"] = effects
    return tuple(sequence)


def _apply_difficulty_variant(task: dict[str, object], *, band: str,
                              sequence: tuple[str, ...]) -> None:
    """Encode decision difficulty in utility-relevant dynamics, never wording."""
    target = str(task["designed_optimal_action"])
    effects = {str(key): dict(value) for key, value in
               dict(task["action_effects"]).items()}
    if band == "HARD" and target in HARD_ALTERNATIVES:
        competitor = HARD_ALTERNATIVES[target]
        # Both actions make the same first decision-relevant improvement; the
        # frozen resource cost difference supplies the small non-zero margin.
        if competitor not in effects:
            effects[competitor] = json.loads(json.dumps(effects.get(target, {})))
    elif band == "EASY":
        required = set(sequence)
        for action in COMPOSABLE_ACTIONS:
            if action == target or action in required:
                continue
            effects[action] = {
                "append_prior_outcome_once": "CONTROL_POISONED",
                "verification_state": "FALSIFIED",
                "temporal_status": "STALE",
                "unresolved_conflict": "true",
                "composition_complete": "false",
            }
    task["action_effects"] = effects
    task["designed_difficulty_band"] = band


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
        "observable_provenance_count": task["observable_provenance_count"],
        "latent": latent,
        "action_effects": effects,
    }


def _pair_mode(pair_index: int) -> tuple[tuple[str, str], str]:
    budget = pair_index % 13 in {0, 2, 5, 8}
    modes = (
        (17, "irreducible", ("RETRIEVE", "VERIFY")),
        (19, "conflict", ("SEARCH_MORE", "DEFER")),
        (23, "temporal", ("ANSWER", "VERIFY")),
        (29, "verification", ("ANSWER", "VERIFY")),
        (31, "provenance", ("ANSWER", "VERIFY")),
        (37, "history", ("RETRIEVE", "SEARCH_MORE")),
    )
    if budget:
        target = NONTERMINAL_ACTIONS[(pair_index // 3) % 3]
        return ("DEFER", target), "budget"
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
            if split == "held_out_structure" and mode == "budget":
                actions, mode = PAIR_ACTIONS[pair_index % len(PAIR_ACTIONS)], "ordinary"
            pair_material = f"{SEED}:{split}:{pair_index}:{rng.getrandbits(64)}"
            pair_id = "opaque-" + hashlib.sha256(pair_material.encode()).hexdigest()[:16]
            entity = rng.choice(ENTITIES)
            summary = f"{rng.choice(surface_pool)} Subject: {entity}."
            default_budget = "GENEROUS" if pair_index % 5 == 0 else "STANDARD"
            for offset, action in enumerate(actions):
                index = pair_index * 2 + offset
                budget = ("TIGHT" if mode == "budget" and offset == 0 else
                          "GENEROUS" if mode == "budget" else default_budget)
                if split == "held_out_structure":
                    budget = "STRUCTURE_HOLDOUT"
                task = _base_action_task(
                    action, index=index, split=split, pair_id=pair_id,
                    summary=summary, budget=budget)
                if mode != "ordinary":
                    _apply_alias(task, mode=mode, offset=offset, action=action,
                                 pair_actions=actions)
                variant = (pair_index % 60 if split in {
                    "development", "held_out_instance", "held_out_surface"}
                           else 100 + pair_index if split == "validation"
                           else 300 + pair_index)
                if mode != "budget":
                    _apply_structure_variant(
                        task, variant=variant, structural_holdout=False)
                sequence: tuple[str, ...] = ()
                if mode != "budget" and split in {"validation", "held_out_structure"}:
                    sequence = _apply_composed_topology(
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
        "benchmark_id": "v2b_i3_3_2_scientific_split_v1",
        "scope": "Frozen deterministic I3.3.2 scientific benchmark; no model-controller result.",
        "tasks": tasks,
    })
    packet_payload = {
        "schema": "DAPH_V2B_I3_3_CONTROLLER_PACKETS_V1",
        "status": "FROZEN_FOR_DEVELOPMENT", "packets": packets,
    }
    families = {
        "schema": "DAPH_V2B_I3_3_TASK_FAMILIES_V1",
        "status": "FROZEN_FOR_DEVELOPMENT",
        "generator_revision": "v2b-i3.3.2-generator-v1",
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
        "benchmark_id": "v2b_i3_3_2_scientific_split_v1",
        "integrity_revision": "V2B-I3.3.2",
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
