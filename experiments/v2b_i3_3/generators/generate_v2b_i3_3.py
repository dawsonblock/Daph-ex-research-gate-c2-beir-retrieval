#!/usr/bin/env python3
"""Deterministically generate and freeze the V2B-I3.3 development corpus."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).parents[3]
BASE = ROOT / "experiments/v2b/tasks/v2b_i3_metareasoning_benchmark_v1.json"
OUT = ROOT / "experiments/v2b_i3_3"
SPLIT_COUNTS = {"development": 300, "validation": 150, "held_out": 300}
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
    ("REASON_MORE", "DEFER"),
    ("STOP", "STOP"),
    ("ANSWER", "VERIFY"),
    ("RETRIEVE", "SEARCH_MORE"),
    ("REASON_MORE", "DEFER"),
)
SURFACES = (
    "Choose the next bounded cognitive action for the available evidence state.",
    "Select one permitted control action under the current evidence and resource budget.",
    "Decide whether another bounded operation is warranted before termination.",
    "Choose the next auditable action using only the public control packet.",
)


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def action_task(action: str, *, index: int, split: str, pair_id: str,
                summary: str, budget: str) -> dict[str, object]:
    latent = {"verification_state": "SUFFICIENT", "temporal_status": "CURRENT",
              "unresolved_conflict": False, "composition_complete": True,
              "expected_terminal": "ANSWER", "required_provenance_count": 0}
    effects: dict[str, dict[str, str]] = {}
    provenance = 2
    channel = "verification"
    if action == "ANSWER":
        channel = "state_irrelevant"
    elif action == "RETRIEVE":
        latent.update(verification_state="MISSING", temporal_status="UNKNOWN")
        effects["RETRIEVE"] = {"verification_state": "SUFFICIENT", "temporal_status": "CURRENT"}
        provenance = 0; channel = "history" if index % 3 == 0 else "verification"
        if split == "held_out" and index % 4 == 0:
            effects["RETRIEVE"] = {"verification_state": "UNVERIFIED", "temporal_status": "CURRENT"}
            effects["VERIFY"] = {"verification_state": "SUFFICIENT", "temporal_status": "CURRENT"}
            channel = "verification_x_history"
    elif action == "VERIFY":
        mode = index % 3
        if mode == 0:
            latent.update(verification_state="UNVERIFIED")
            effects["VERIFY"] = {"verification_state": "SUFFICIENT"}
            channel = "verification"
        elif mode == 1:
            latent.update(temporal_status="STALE")
            effects["VERIFY"] = {"temporal_status": "CURRENT"}
            channel = "temporal"
        else:
            latent["required_provenance_count"] = 3
            provenance = 1
            effects["VERIFY"] = {"provenance_count": "3"}
            channel = "provenance"
    elif action == "SEARCH_MORE":
        provenance = 0
        # The frozen shared policy requires DEFER under an active conflict, so
        # SEARCH_MORE challenge cases model a prior failed retrieval rather
        # than bypassing that safety rule.
        latent.update(verification_state="MISSING", temporal_status="UNKNOWN")
        effects["RETRIEVE"] = {}
        effects["SEARCH_MORE"] = {"verification_state": "SUFFICIENT", "temporal_status": "CURRENT"}
        channel = "history"
    elif action == "REASON_MORE":
        latent["composition_complete"] = False
        effects["REASON_MORE"] = {"composition_complete": "true"}
        channel = "composition"
    elif action == "DEFER":
        latent.update(verification_state="MISSING", temporal_status="UNKNOWN",
                      expected_terminal="DEFER")
        provenance = 0
        if index % 4 == 1:
            latent.update(verification_state="SUFFICIENT", temporal_status="CURRENT",
                          unresolved_conflict=True)
            channel = "conflict"
        else:
            channel = "irreducible"
    elif action == "STOP":
        latent["expected_terminal"] = "STOP"
        channel = "state_irrelevant"
        summary = "NO_FINAL_ASSERTION_REQUESTED. Terminate the bounded internal control process."
    else:
        raise ValueError(action)
    return {
        "task_id": f"i3_3_{split}_{index:04d}", "split": split,
        "category": f"{channel}_{action.lower()}", "task_summary": summary,
        "high_stakes": action == "VERIFY" and index % 2 == 0,
        "budget_profile": budget, "observable_provenance_count": provenance,
        "latent": latent, "action_effects": effects,
        "designed_optimal_action": action, "cognitive_channel": channel,
        "generator_pair_id": pair_id,
    }


def generate() -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    tasks: list[dict[str, object]] = []; packets: list[dict[str, str]] = []
    for split, count in SPLIT_COUNTS.items():
        for pair_index in range(count // 2):
            budget_sensitive = pair_index % 11 == 0
            actions = (("DEFER", "VERIFY") if budget_sensitive
                       else PAIR_ACTIONS[pair_index % len(PAIR_ACTIONS)])
            pair_id = f"opaque-{split[:3]}-{pair_index:04d}"
            summary = SURFACES[(pair_index + len(split)) % len(SURFACES)]
            budget = ("TIGHT" if actions in {("REASON_MORE", "DEFER"), ("STOP", "STOP")}
                      and pair_index % 3 == 0 else
                      ("GENEROUS" if pair_index % 5 == 0 else "STANDARD"))
            for offset, action in enumerate(actions):
                index = pair_index * 2 + offset
                task = action_task(action, index=index, split=split, pair_id=pair_id,
                                   summary=summary,
                                   budget=("TIGHT" if budget_sensitive and offset == 0 else
                                           "GENEROUS" if budget_sensitive else budget))
                if budget_sensitive:
                    # Same latent epistemic problem, transition, and public surface;
                    # only the budget differs. With no verification credit DEFER is
                    # least-bad, while GENEROUS can VERIFY then ANSWER.
                    task["latent"] = {
                        "verification_state": "UNVERIFIED", "temporal_status": "CURRENT",
                        "unresolved_conflict": False, "composition_complete": True,
                        "expected_terminal": "ANSWER", "required_provenance_count": 0,
                    }
                    task["observable_provenance_count"] = 1
                    task["action_effects"] = {"VERIFY": {"verification_state": "SUFFICIENT"}}
                    task["cognitive_channel"] = "verification_x_budget"
                    task["category"] = f"verification_x_budget_{action.lower()}"
                tasks.append(task)
                packets.append({"task_id": str(task["task_id"]), "instance_id": pair_id,
                                "task_summary": str(task["task_summary"])})
    return tasks, packets


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def main() -> None:
    base = json.loads(BASE.read_text()); tasks, packets = generate()
    private = {key: base[key] for key in ("schema", "status", "protocol",
                                           "utility_weights", "action_costs")}
    private["budget_profiles"] = I3_3_BUDGET_PROFILES
    private.update({"benchmark_id": "v2b_i3_3_scaled_frozen_metareasoning_v1",
                    "scope": "Frozen deterministic I3.3 development benchmark; no model-controller result.",
                    "tasks": tasks})
    packet_payload = {"schema": "DAPH_V2B_I3_3_CONTROLLER_PACKETS_V1",
                      "status": "FROZEN_FOR_DEVELOPMENT", "packets": packets}
    families = {"schema": "DAPH_V2B_I3_3_TASK_FAMILIES_V1", "status": "FROZEN_FOR_DEVELOPMENT",
                "generator_revision": "v2b-i3.3-generator-v1", "seed": 3301,
                "pair_action_cycle": PAIR_ACTIONS,
                "channels": ["verification", "temporal", "provenance", "conflict", "history",
                             "composition", "irreducible", "state_irrelevant",
                             "verification_x_budget", "verification_x_history"]}
    surfaces = {"schema": "DAPH_V2B_I3_3_SURFACE_TEMPLATES_V1",
                "status": "FROZEN_FOR_DEVELOPMENT", "templates": SURFACES}
    split_payload = {"schema": "DAPH_V2B_I3_3_SPLITS_V1", "status": "FROZEN_FOR_DEVELOPMENT",
                     "splits": {split: [{"task_id": task["task_id"], "task_sha256": digest(task)}
                                        for task in tasks if task["split"] == split]
                                for split in SPLIT_COUNTS}}
    counts = Counter(str(task["designed_optimal_action"]) for task in tasks)
    channels = Counter(str(task["cognitive_channel"]) for task in tasks)
    budgets = Counter(str(task["budget_profile"]) for task in tasks)
    report = {"schema": "DAPH_V2B_I3_3_BALANCE_REPORT_V1", "task_count": len(tasks),
              "split_counts": dict(SPLIT_COUNTS), "designed_optimal_action_counts": dict(sorted(counts.items())),
              "cognitive_channel_counts": dict(sorted(channels.items())),
              "budget_counts": dict(sorted(budgets.items())),
              "budget_sensitive_pair_count": sum(1 for task in tasks
                                                   if task["cognitive_channel"] == "verification_x_budget") // 2,
              "minimum_action_count": min(counts.values()), "generator_output_sha256": digest(tasks)}
    write_json(OUT / "private/v2b_i3_3_tasks_v1.json", private)
    write_json(OUT / "controller_packets/v2b_i3_3_controller_packets_v1.json", packet_payload)
    write_json(OUT / "task_families/v2b_i3_3_families_v1.json", families)
    write_json(OUT / "surface_templates/v2b_i3_3_surface_templates_v1.json", surfaces)
    write_json(OUT / "splits/v2b_i3_3_splits_v1.json", split_payload)
    write_json(OUT / "reports/v2b_i3_3_balance_report_v1.json", report)
    manifest = {
        "schema": "DAPH_V2B_I3_3_BENCHMARK_MANIFEST_V1",
        "status": "FROZEN_FOR_BENCHMARK_QUALIFICATION",
        "benchmark_id": "v2b_i3_3_scaled_frozen_metareasoning_v1",
        "private_environment_path": "../private/v2b_i3_3_tasks_v1.json",
        "controller_packets_path": "../controller_packets/v2b_i3_3_controller_packets_v1.json",
        "task_families_path": "../task_families/v2b_i3_3_families_v1.json",
        "split_definitions_path": "../splits/v2b_i3_3_splits_v1.json",
        "surface_templates_path": "../surface_templates/v2b_i3_3_surface_templates_v1.json",
        "protocol_path": "../../../configs/v2b_i3_2_2_protocol_v1.json",
        "balance_report_path": "../reports/v2b_i3_3_balance_report_v1.json",
        "oracle_cache_manifest_path": "../oracle_tables/v2b_i3_3_oracle_cache_manifest_v1.json",
        "claim_boundary": "Scaled frozen development benchmark only; no model or V2B verdict."
    }
    write_json(OUT / "manifests/v2b_i3_3_benchmark_manifest_v1.json", manifest)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
