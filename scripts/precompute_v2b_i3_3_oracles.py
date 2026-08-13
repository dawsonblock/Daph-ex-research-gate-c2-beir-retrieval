#!/usr/bin/env python3
"""Precompute deterministic latent and sequential I3.3 oracle ground truth."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
from time import perf_counter


ROOT = Path(__file__).parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hrm_adaptive_memory.executive.metareasoning_benchmark import load_metareasoning_benchmark
from hrm_adaptive_memory.executive.metareasoning_artifacts import (
    artifact_graph_sha256, resolve_benchmark_artifact_graph)
from hrm_adaptive_memory.executive.metareasoning_controller import load_observation_masks
from hrm_adaptive_memory.executive.metareasoning_executor import initial_i3_runtime
from hrm_adaptive_memory.executive.metareasoning_sequential_oracle import (
    build_sequential_observable_oracle)
from hrm_adaptive_memory.executive.metareasoning_transition_table import OracleTableCache
from hrm_adaptive_memory.executive.metareasoning_utility import MetareasoningUtility
from hrm_adaptive_memory.executive.policy import load_frozen_policy
from hrm_adaptive_memory.executive.resources import ResourceState
from hrm_adaptive_memory.common.canonical_json import (  # noqa: E402
    canonical_bytes, write_json)


CONFIG = ROOT / "experiments/v2b_i3_3/configs/v2b_i3_3_benchmark_freeze_v1.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def combined_hash(values: object) -> str:
    return hashlib.sha256(canonical_bytes(values)).hexdigest()


def semantic_table(table) -> dict[str, object]:
    """Serialize oracle semantics without nondeterministic timing/RSS telemetry."""
    payload = table.serializable()
    payload.pop("build_metrics", None)
    return payload


def write_gzip_jsonl(path: Path, rows: list[object]) -> None:
    raw = b"".join(canonical_bytes(row) + b"\n" for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(raw, compresslevel=9, mtime=0))


def q_margin(table, state_id: str) -> float:
    values = sorted({round(value, 12) for (origin, _), value in table.q_values.items()
                     if origin == state_id}, reverse=True)
    return 0.0 if len(values) < 2 else values[0] - values[1]


def margin_band(value: float, tied: bool) -> str:
    if tied:
        return "TIE"
    if value < 1.0:
        return "HARD"
    if value < 5.0:
        return "MEDIUM"
    return "EASY"


def log_experiment(manifest: dict[str, object], characterization: dict[str, object],
                   manifest_path: Path) -> None:
    """Record every full cache-generation run in the required LitLogger workspace."""
    import litlogger
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    source_tree = subprocess.check_output(
        ["git", "rev-parse", "HEAD^{tree}"], cwd=ROOT, text=True).strip()
    experiment = litlogger.init(
        name=f"v2b-i3.3.1-oracle-regeneration-{source_commit[:7]}",
        teamspace="deep-gpu-acceleration-project",
        metadata={
            "protocol": "DAPH_V2B_I3_3_1_BENCHMARK_INTEGRITY",
            "status": "DEVELOPMENT_NOT_QUALIFIED",
            "source_commit": source_commit,
            "source_tree_hash": source_tree,
            "manifest_sha256": sha256(manifest_path),
            "location": "US", "altitude": "1334",
        },
        print_url=True,
    )
    latent = manifest["latent_oracles"]
    experiment["latent_tables"].append(latent["table_count"])
    experiment["latent_states"].append(latent["reachable_states"])
    experiment["latent_transitions"].append(latent["reachable_transitions"])
    experiment["latent_build_seconds"].append(characterization["latent_build_seconds"])
    for condition, record in manifest["sequential_observable_oracles"].items():
        prefix = str(condition).lower()
        experiment[f"{prefix}_states"].append(record["information_states"])
        experiment[f"{prefix}_transitions"].append(record["information_transitions"])
        experiment[f"{prefix}_task_uniform_information_gap"].append(
            record["task_uniform_information_gap"])
    experiment["run_valid"] = "true"
    experiment["qualification"] = "NOT_QUALIFIED_DEVELOPMENT_ONLY"
    experiment.finalize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", nargs="*", default=None)
    parser.add_argument("--skip-litlogger", action="store_true")
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    benchmark_path = ROOT / config["benchmark_manifest_path"]
    # Cache rebuilding necessarily starts while the previous cache no longer
    # matches new benchmark bytes. Exact cache validation resumes after the
    # atomic semantic artifacts and manifest have been regenerated.
    benchmark = load_metareasoning_benchmark(benchmark_path, verify_oracle_cache=False)
    benchmark_manifest = json.loads(benchmark_path.read_text())
    private_path = (benchmark_path.parent / benchmark_manifest["private_environment_path"]).resolve()
    private_payload = json.loads(private_path.read_text())
    private_by_id = {str(task["task_id"]): task for task in private_payload["tasks"]}
    policy = load_frozen_policy(ROOT / config["policy_path"])
    masks = load_observation_masks(ROOT / config["observation_masks_path"])
    utility = MetareasoningUtility.from_file(ROOT / config["utility_path"])
    selected = tuple(args.conditions or config["required_conditions"])
    if set(selected) - set(masks):
        raise RuntimeError("unknown I3.3 observation-mask condition")
    if set(selected) != set(config["required_conditions"]):
        raise RuntimeError("frozen I3.3 cache generation requires all observation conditions")
    limits = config["oracle_limits"]
    output = ROOT / "experiments/v2b_i3_3/oracle_tables"
    output.mkdir(parents=True, exist_ok=True)

    started = perf_counter(); cache = OracleTableCache()
    runtimes = {task.task_id: initial_i3_runtime(task, ResourceState(benchmark.budget_for(task)))
                for task in benchmark.tasks}
    latent = {}
    latent_rows = []
    action_counts: Counter[str] = Counter()
    singleton_counts: Counter[str] = Counter()
    tie_counts: Counter[str] = Counter()
    margin_counts: Counter[str] = Counter()
    singleton_by_split: dict[str, Counter[str]] = defaultdict(Counter)
    ties_by_split: dict[str, Counter[str]] = defaultdict(Counter)
    margins_by_split: dict[str, Counter[str]] = defaultdict(Counter)
    disagreements: list[dict[str, object]] = []
    difficulty_rows = []
    for task_id in sorted(runtimes):
        table = cache.get_or_build(initial_runtime=runtimes[task_id], policy=policy,
                                   utility=utility, include_policy_feedback=True)
        if len(table.states) > limits["max_latent_states_per_task"]:
            raise RuntimeError("I3.3 latent oracle state limit exceeded")
        if len(table.transitions) > limits["max_latent_transitions_per_task"]:
            raise RuntimeError("I3.3 latent oracle transition limit exceeded")
        latent[task_id] = table
        root = table.initial_state_id
        actions = [item.value for item in table.optimal_actions[root]]
        action_counts.update(actions)
        if len(actions) == 1:
            singleton_counts[actions[0]] += 1
            singleton_by_split[str(private_by_id[task_id]["split"])][actions[0]] += 1
        else:
            tie_counts["|".join(actions)] += 1
            ties_by_split[str(private_by_id[task_id]["split"])]["|".join(actions)] += 1
        designed = str(private_by_id[task_id]["designed_optimal_action"])
        if designed not in actions:
            disagreements.append({
                "task_id": task_id, "designed_action": designed,
                "oracle_optimal_actions": actions,
                "q_values": {action.value: value for (origin, action), value
                             in table.q_values.items() if origin == root},
            })
        margin = q_margin(table, root)
        margin_counts[margin_band(margin, len(actions) > 1)] += 1
        margins_by_split[str(private_by_id[task_id]["split"])][
            margin_band(margin, len(actions) > 1)] += 1
        minimum_cost = table.minimum_remaining_cost[root]
        successful_path_exists = minimum_cost != float("inf")
        latent_rows.append({"task_id": task_id, "table": semantic_table(table)})
        difficulty_rows.append({
            "task_id": task_id, "latent_value": table.state_values[root],
            "latent_optimal_actions": actions, "optimal_q_margin": q_margin(table, root),
            "successful_path_exists": successful_path_exists,
            "minimum_remaining_cost": minimum_cost if successful_path_exists else None,
            "semantic_structure_coarse": private_by_id[task_id]["semantic_structure_coarse"],
            "semantic_structure_exact": private_by_id[task_id]["semantic_structure_exact"],
            "split": private_by_id[task_id]["split"],
            "reachable_states": len(table.states),
            "reachable_transitions": len(table.transitions),
        })
    latent_seconds = perf_counter() - started
    latent_path = output / "v2b_i3_3_latent_oracles_v1.jsonl.gz"
    write_gzip_jsonl(latent_path, latent_rows)

    condition_records = {}
    benchmark_manifest_sha256 = sha256(benchmark_path)
    manifest_payload = benchmark_manifest
    # Oracle inputs form a closed graph that deliberately excludes the oracle
    # cache edge itself, avoiding a self-referential hash cycle.
    manifest_inputs = dict(manifest_payload)
    manifest_inputs.pop("oracle_cache_manifest_path", None)
    protocol_path = ROOT / config["protocol_path"]
    protocol_payload = json.loads(protocol_path.read_text())
    oracle_input_graph = resolve_benchmark_artifact_graph(
        manifest_path=benchmark_path.relative_to(ROOT).as_posix(), manifest=manifest_inputs,
        protocol_path=protocol_path.relative_to(ROOT).as_posix(), protocol=protocol_payload)
    benchmark_hash = artifact_graph_sha256(ROOT, oracle_input_graph)
    if set(selected) == set(config["required_conditions"]):
        selected = tuple(config["required_conditions"])
    for condition in selected:
        condition_started = perf_counter()
        oracle_set = build_sequential_observable_oracle(
            runtime_tables=((runtimes[key], latent[key]) for key in sorted(latent)),
            mask=masks[condition], policy=policy, utility=utility,
            benchmark_hash=benchmark_hash,
            max_information_states=limits["max_information_states_per_condition"],
            max_information_transitions=limits["max_information_transitions_per_condition"],
            max_members_per_belief=limits["max_members_per_belief"])
        rows = [{"initial_information_state_id": key, "table": semantic_table(table)}
                for key, table in sorted(oracle_set.tables.items())]
        target = output / f"v2b_i3_3_sequential_{condition.lower()}_v1.jsonl.gz"
        write_gzip_jsonl(target, rows)
        states = sum(len(table.information_states) for table in oracle_set.tables.values())
        transitions = sum(len(table.transitions) for table in oracle_set.tables.values())
        if states > limits["max_information_states_per_condition"]:
            raise RuntimeError("I3.3 information-state limit exceeded")
        if transitions > limits["max_information_transitions_per_condition"]:
            raise RuntimeError("I3.3 information-transition limit exceeded")
        information_gaps = []
        for table in oracle_set.tables.values():
            root = table.initial_information_state_id
            information_gaps.append((
                len(table.information_states[root].members),
                table.expected_latent_values[root] - table.belief_values[root]))
        task_mass = sum(count for count, _ in information_gaps)
        condition_records[condition] = {
            "path": target.relative_to(ROOT).as_posix(), "sha256": sha256(target),
            "set_sha256": oracle_set.table_sha256,
            "table_count": len(oracle_set.tables), "information_states": states,
            "information_transitions": transitions,
            "task_uniform_information_gap": sum(count * gap for count, gap in information_gaps) / task_mass,
            "class_uniform_information_gap": sum(gap for _, gap in information_gaps) / len(information_gaps),
        }
        condition_records[condition]["_build_seconds"] = perf_counter() - condition_started
        print(json.dumps({"condition": condition, **condition_records[condition]}, sort_keys=True),
              flush=True)

    difficulty_path = output / "v2b_i3_3_difficulty_report_v1.json"
    write_json(difficulty_path, {
        "schema": "DAPH_V2B_I3_3_DIFFICULTY_REPORT_V1",
        "status": "FROZEN_BENCHMARK_NOT_A_SCIENTIFIC_RESULT",
        "tasks": difficulty_rows,
    })
    oracle_balance = {
        "schema": "DAPH_V2B_I3_3_1_ORACLE_BALANCE_REPORT_V1",
        "status": "FROZEN_BENCHMARK_NOT_EXECUTIVE_RESULT",
        "task_count": len(difficulty_rows),
        "singleton_optimal_action_counts": dict(sorted(singleton_counts.items())),
        "tied_optimal_action_sets": dict(sorted(tie_counts.items())),
        "multi_optimal_task_count": sum(tie_counts.values()),
        "q_margin_bands": dict(sorted(margin_counts.items())),
        "by_split": {
            split: {
                "singleton_optimal_action_counts": dict(sorted(singleton_by_split[split].items())),
                "tied_optimal_action_sets": dict(sorted(ties_by_split[split].items())),
                "q_margin_bands": dict(sorted(margins_by_split[split].items())),
            }
            for split in sorted({str(task["split"]) for task in private_by_id.values()})
        },
        "designed_oracle_agreement_count": len(difficulty_rows) - len(disagreements),
        "designed_oracle_disagreements": disagreements,
        "tie_semantics": "Any action in latent_optimal_actions has zero action regret.",
    }
    oracle_balance_path = output.parent / "reports/v2b_i3_3_1_oracle_balance_report_v1.json"
    write_json(oracle_balance_path, oracle_balance)
    characterization = {
        "schema": "DAPH_V2B_I3_3_ORACLE_CHARACTERIZATION_V1",
        "status": "MEASURED_DEVELOPMENT_TELEMETRY_NOT_ORACLE_IDENTITY",
        "latent_build_seconds": latent_seconds,
        "condition_build_seconds": {
            name: record.pop("_build_seconds") for name, record in condition_records.items()},
        "peak_resident_memory_native_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "platform_note": "resource.getrusage(RUSAGE_SELF).ru_maxrss native platform units",
    }
    characterization_path = output / "v2b_i3_3_oracle_characterization_v1.json"
    write_json(characterization_path, characterization)
    manifest = {
        "schema": "DAPH_V2B_I3_3_ORACLE_CACHE_MANIFEST_V1",
        "status": "FROZEN_BENCHMARK_NOT_A_SCIENTIFIC_RESULT",
        "benchmark_manifest_sha256": benchmark_manifest_sha256,
        "benchmark_closure_sha256": benchmark_hash,
        "benchmark_closure_artifacts": dict(sorted(oracle_input_graph.items())),
        "latent_oracles": {
            "path": latent_path.relative_to(ROOT).as_posix(), "sha256": sha256(latent_path),
            "table_set_sha256": combined_hash({key: table.table_sha256
                                                 for key, table in sorted(latent.items())}),
            "table_count": len(latent),
            "reachable_states": sum(len(table.states) for table in latent.values()),
            "reachable_transitions": sum(len(table.transitions) for table in latent.values()),
        },
        "sequential_observable_oracles": condition_records,
        "difficulty_report": {"path": difficulty_path.relative_to(ROOT).as_posix(),
                              "sha256": sha256(difficulty_path)},
        "oracle_balance_report": {
            "path": oracle_balance_path.relative_to(ROOT).as_posix(),
            "sha256": sha256(oracle_balance_path),
        },
        "latent_optimal_action_counts_with_ties": dict(sorted(action_counts.items())),
        "limits": limits,
        "claim_boundary": "Precomputed deterministic oracle ground truth only; no executive result.",
    }
    manifest_path = output / "v2b_i3_3_oracle_cache_manifest_v1.json"
    write_json(manifest_path, manifest)
    if not args.skip_litlogger:
        log_experiment(manifest, characterization, manifest_path)
    print(manifest_path)


if __name__ == "__main__":
    main()
