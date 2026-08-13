"""Frozen V2B-I3 metareasoning benchmark with explicit latent/observable layers."""
from __future__ import annotations

import json
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from hrm_adaptive_memory.cognitive_control.actions import validate_v2b_action
from hrm_adaptive_memory.cognitive_control.core import DecisionAction
from hrm_adaptive_memory.cognitive_control.state import TemporalStatus, VerificationState

from .resources import DEFAULT_ACTION_COSTS, ResourceBudget
from .metareasoning_artifacts import oracle_cache_artifacts, resolve_benchmark_artifact_graph


BENCHMARK_SCHEMA = "DAPH_V2B_I3_METAREASONING_BENCHMARK_V1"
BENCHMARK_MANIFEST_SCHEMA = "DAPH_V2B_I3_BENCHMARK_MANIFEST_V1"
CONTROLLER_PACKETS_SCHEMA = "DAPH_V2B_I3_CONTROLLER_PACKETS_V1"
I3_1_CONTROLLER_PACKETS_SCHEMA = "DAPH_V2B_I3_1_CONTROLLER_PACKETS_V1"
I3_2_BENCHMARK_MANIFEST_SCHEMA = "DAPH_V2B_I3_2_BENCHMARK_MANIFEST_V1"
I3_2_TASK_EXTENSION_SCHEMA = "DAPH_V2B_I3_2_TASK_EXTENSION_V1"
I3_2_CONTROLLER_PACKETS_SCHEMA = "DAPH_V2B_I3_2_CONTROLLER_PACKETS_V1"
I3_3_BENCHMARK_MANIFEST_SCHEMA = "DAPH_V2B_I3_3_BENCHMARK_MANIFEST_V1"
I3_3_CONTROLLER_PACKETS_SCHEMA = "DAPH_V2B_I3_3_CONTROLLER_PACKETS_V1"
FROZEN_DEVELOPMENT_STATUS = "FROZEN_FOR_DEVELOPMENT"
FROZEN_BENCHMARK_STATUS = "FROZEN_FOR_BENCHMARK_QUALIFICATION"


@dataclass(frozen=True)
class LatentTaskState:
    """Environment state used for dynamics/scoring and never passed to a controller."""

    verification_state: VerificationState
    temporal_status: TemporalStatus
    unresolved_conflict: bool
    composition_complete: bool
    expected_terminal: DecisionAction
    required_provenance_count: int = 0
    conflict_resolvable: bool = False
    initial_prior_outcomes: tuple[str, ...] = ()


@dataclass(frozen=True)
class I3BenchmarkTask:
    task_id: str
    split: str
    category: str
    task_summary: str
    high_stakes: bool
    budget_profile: str
    latent: LatentTaskState
    observable_provenance_count: int
    action_effects: Mapping[DecisionAction, Mapping[str, str]]
    # An opaque public identifier.  The private task id is used only by the
    # environment, policy, and receipts; this value is what a controller sees.
    controller_instance_id: str = ""
    semantic_structure_coarse: str = ""
    semantic_structure_exact: str = ""

    def __post_init__(self) -> None:
        if self.latent.required_provenance_count < 0:
            raise ValueError("required provenance count must be nonnegative")
        if (not self.task_id or self.task_id != self.task_id.lower()
                or not self.category or not self.task_summary):
            raise ValueError("I3 tasks require lowercase ids, a category, and a summary")
        if self.split not in {"development", "validation", "held_out",
                              "held_out_instance", "held_out_surface",
                              "held_out_structure"}:
            raise ValueError("I3 task uses an unsupported benchmark split")
        if self.latent.expected_terminal not in {
                DecisionAction.ANSWER, DecisionAction.DEFER, DecisionAction.STOP}:
            raise ValueError("I3 tasks require a terminal action")
        if self.observable_provenance_count < 0:
            raise ValueError("observable provenance count must be nonnegative")
        for action, effect in self.action_effects.items():
            validate_v2b_action(action)
            if not isinstance(effect, Mapping):
                raise ValueError("I3 action effects must be mappings")


@dataclass(frozen=True)
class MetareasoningBenchmark:
    benchmark_id: str
    tasks: tuple[I3BenchmarkTask, ...]
    budget_profiles: Mapping[str, ResourceBudget]
    utility_weights: Mapping[str, float]
    metadata: Mapping[str, Any]
    artifact_hashes: Mapping[str, str]

    def budget_for(self, task: I3BenchmarkTask) -> ResourceBudget:
        try:
            return self.budget_profiles[task.budget_profile]
        except KeyError as error:
            raise ValueError(f"unknown I3 budget profile: {task.budget_profile}") from error

    def for_split(self, split: str) -> "MetareasoningBenchmark":
        tasks = tuple(task for task in self.tasks if task.split == split)
        if not tasks:
            raise ValueError(f"I3 benchmark split is empty: {split}")
        return MetareasoningBenchmark(
            self.benchmark_id, tasks, self.budget_profiles, self.utility_weights,
            self.metadata, self.artifact_hashes)


def _load_budget_profiles(raw: object) -> Mapping[str, ResourceBudget]:
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("I3 benchmark needs nonempty named budget profiles")
    profiles: dict[str, ResourceBudget] = {}
    for name, values in raw.items():
        if (not isinstance(name, str) or not name.isupper() or not isinstance(values, Mapping)):
            raise ValueError("I3 budget profiles require uppercase names and mapping values")
        profiles[name] = ResourceBudget(**dict(values))
    return profiles


def _load_utility_weights(raw: object) -> Mapping[str, float]:
    required = {"success_reward", "failure_penalty", "executive_step", "retrieval",
                "verification", "search", "reasoning_128_tokens", "logical_ms"}
    if not isinstance(raw, Mapping) or set(raw) != required:
        raise ValueError("I3 benchmark has an invalid frozen utility-weight set")
    weights = {str(name): float(value) for name, value in raw.items()}
    if any(value < 0 for value in weights.values()):
        raise ValueError("I3 utility weights must be nonnegative")
    if weights["success_reward"] == 0 or weights["failure_penalty"] == 0:
        raise ValueError("I3 terminal utility weights must be positive")
    return weights


def _validate_frozen_action_costs(raw: object) -> None:
    """Bind the benchmark's declared costs to the executor's actual cost table."""
    expected = {action.value: asdict(cost) for action, cost in DEFAULT_ACTION_COSTS.items()}
    if raw != expected:
        raise ValueError("I3 frozen action costs do not match the executor cost table")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_payloads(path: Path, *, verify_oracle_cache: bool = True
                   ) -> tuple[Mapping[str, Any], Mapping[str, Mapping[str, str]], Mapping[str, str]]:
    """Load private task dynamics and separately stored controller packets."""
    payload = json.loads(path.read_text())
    if payload.get("schema") == BENCHMARK_SCHEMA:
        # Compatibility for the frozen I3 development baseline. New I3 runs use
        # the manifest path below so controller packets are physically separate.
        return payload, {}, {"private_environment": _sha256(path)}
    if payload.get("schema") not in {BENCHMARK_MANIFEST_SCHEMA, I3_2_BENCHMARK_MANIFEST_SCHEMA,
                                     I3_3_BENCHMARK_MANIFEST_SCHEMA}:
        raise ValueError("unsupported V2B-I3 metareasoning benchmark schema")
    if payload.get("status") not in {FROZEN_DEVELOPMENT_STATUS, FROZEN_BENCHMARK_STATUS}:
        raise ValueError("V2B-I3 benchmark manifest must be frozen for development")
    root = next((parent for parent in path.parents if (parent / ".git").exists()), None)
    if root is None:
        # Extracted qualification archives intentionally omit .git; derive the
        # root from the stable experiments/ segment. Standalone test manifests
        # resolve relative to their own directory.
        experiments = next((parent for parent in path.parents if parent.name == "experiments"), None)
        root = experiments.parent if experiments is not None else path.parent
    manifest_relative = path.relative_to(root).as_posix()
    graph = resolve_benchmark_artifact_graph(
        manifest_path=manifest_relative, manifest=payload,
        json_loader=lambda relative: json.loads((root / relative).read_text()))
    if verify_oracle_cache and "oracle_cache_manifest" in graph:
        cache = json.loads((root / graph["oracle_cache_manifest"]).read_text())
        for role, (_, expected_sha256) in oracle_cache_artifacts(cache).items():
            if _sha256(root / graph[role]) != expected_sha256:
                raise ValueError(f"V2B-I3 oracle cache hash mismatch: {role}")
    private_path = (root / graph["private_environment"]).resolve()
    packets_path = (root / graph["controller_packets"]).resolve()
    private_payload = json.loads(private_path.read_text())
    packet_payload = json.loads(packets_path.read_text())
    if private_payload.get("schema") != BENCHMARK_SCHEMA:
        raise ValueError("V2B-I3 private environment has an unsupported schema")
    packet_schema = packet_payload.get("schema")
    if packet_schema not in {CONTROLLER_PACKETS_SCHEMA, I3_1_CONTROLLER_PACKETS_SCHEMA,
                             I3_2_CONTROLLER_PACKETS_SCHEMA, I3_3_CONTROLLER_PACKETS_SCHEMA}:
        raise ValueError("V2B-I3 controller packets have an unsupported schema")
    if packet_payload.get("status") != FROZEN_DEVELOPMENT_STATUS:
        raise ValueError("V2B-I3 controller packets must be frozen for development")
    packets = packet_payload.get("packets")
    extension_path = None
    if payload.get("schema") == I3_2_BENCHMARK_MANIFEST_SCHEMA:
        extension_path = (root / graph["task_extension"]).resolve()
        extension_payload = json.loads(extension_path.read_text())
        if (extension_payload.get("schema") != I3_2_TASK_EXTENSION_SCHEMA
                or extension_payload.get("status") != FROZEN_DEVELOPMENT_STATUS
                or not isinstance(extension_payload.get("tasks"), list)):
            raise ValueError("V2B-I3.2 task extension must be frozen and nonempty")
        private_payload = dict(private_payload)
        private_payload["tasks"] = list(private_payload.get("tasks", ())) + extension_payload["tasks"]
        private_payload["benchmark_id"] = str(payload.get("benchmark_id", private_payload.get("benchmark_id", "")))
        packets_extension_path = (root / graph["controller_packets_extension"]).resolve()
        packets_extension = json.loads(packets_extension_path.read_text())
        if (packets_extension.get("schema") != I3_2_CONTROLLER_PACKETS_SCHEMA
                or packets_extension.get("status") != FROZEN_DEVELOPMENT_STATUS
                or not isinstance(packets_extension.get("packets"), list)):
            raise ValueError("V2B-I3.2 controller extension must be frozen and nonempty")
        packets = list(packets) + packets_extension["packets"]
    if not isinstance(packets, list):
        raise ValueError("V2B-I3 controller packet artifact must contain packets")
    by_task: dict[str, Mapping[str, str]] = {}
    forbidden = {"reasoning_required", "correct_action", "optimal_action", "evidence_sufficient",
                 "expected_terminal_action", "oracle_value", "ground_truth_transition"}
    for packet in packets:
        if not isinstance(packet, Mapping) or not isinstance(packet.get("task_id"), str):
            raise ValueError("V2B-I3 controller packets require task ids")
        if forbidden.intersection(packet):
            raise ValueError("V2B-I3 controller packet contains a forbidden latent/oracle field")
        expected_fields = ({"task_id", "task_summary"} if packet_schema == CONTROLLER_PACKETS_SCHEMA
                           else {"task_id", "instance_id", "task_summary"})
        if set(packet) != expected_fields or not isinstance(packet["task_summary"], str):
            raise ValueError("V2B-I3 controller packet fields do not match the frozen schema")
        if (packet_schema in {I3_1_CONTROLLER_PACKETS_SCHEMA, I3_2_CONTROLLER_PACKETS_SCHEMA,
                              I3_3_CONTROLLER_PACKETS_SCHEMA}
                and (not isinstance(packet["instance_id"], str) or not packet["instance_id"])):
            raise ValueError("V2B-I3 controller packets require opaque instance ids")
        if any(token in packet["task_summary"].lower() for token in forbidden):
            raise ValueError("V2B-I3 controller packet contains a forbidden latent/oracle value")
        by_task[packet["task_id"]] = {
            "task_summary": packet["task_summary"],
            "instance_id": str(packet.get("instance_id", packet["task_id"])),
        }
    return private_payload, by_task, {
        role: _sha256(root / relative) for role, relative in graph.items()
    }


def load_metareasoning_benchmark(path: str | Path, *, verify_oracle_cache: bool = True
                                 ) -> MetareasoningBenchmark:
    path = Path(path).resolve()
    payload, packets, artifact_hashes = _load_payloads(path, verify_oracle_cache=verify_oracle_cache)
    if payload.get("status") not in {FROZEN_DEVELOPMENT_STATUS, FROZEN_BENCHMARK_STATUS}:
        raise ValueError("V2B-I3 benchmark must be frozen for development")
    profiles = _load_budget_profiles(payload.get("budget_profiles"))
    _validate_frozen_action_costs(payload.get("action_costs"))
    raw_tasks = payload.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ValueError("V2B-I3 frozen benchmark must contain tasks")
    tasks: list[I3BenchmarkTask] = []
    for raw in raw_tasks:
        if not isinstance(raw, Mapping):
            raise ValueError("I3 tasks must be mappings")
        latent = raw.get("latent")
        if not isinstance(latent, Mapping):
            raise ValueError("I3 task needs a latent environment state")
        effects = {
            validate_v2b_action(action): dict(effect)
            for action, effect in dict(raw.get("action_effects", {})).items()
        }
        task_id = str(raw["task_id"])
        public = packets.get(task_id, {})
        if packets and task_id not in packets:
            raise ValueError(f"I3 private task has no controller packet: {task_id}")
        task = I3BenchmarkTask(
            task_id=task_id, category=str(raw["category"]),
            split=str(raw.get("split", "development")),
            task_summary=str(public.get("task_summary", raw["task_summary"])), high_stakes=bool(raw["high_stakes"]),
            budget_profile=str(raw["budget_profile"]),
            latent=LatentTaskState(
                verification_state=VerificationState(latent["verification_state"]),
                temporal_status=TemporalStatus(latent["temporal_status"]),
                unresolved_conflict=bool(latent["unresolved_conflict"]),
                composition_complete=bool(latent["composition_complete"]),
                expected_terminal=validate_v2b_action(latent["expected_terminal"]),
                required_provenance_count=int(latent.get("required_provenance_count", 0)),
                conflict_resolvable=bool(latent.get("conflict_resolvable", False)),
                initial_prior_outcomes=tuple(str(value) for value in
                                             latent.get("initial_prior_outcomes", ())),
            ),
            observable_provenance_count=int(raw.get("observable_provenance_count", 0)),
            action_effects=effects,
            controller_instance_id=str(public.get("instance_id", task_id)),
            semantic_structure_coarse=str(raw.get("semantic_structure_coarse", "")),
            semantic_structure_exact=str(raw.get("semantic_structure_exact", "")),
        )
        if task.budget_profile not in profiles:
            raise ValueError(f"task {task.task_id} references an unknown budget profile")
        tasks.append(task)
    if len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError("V2B-I3 task ids must be unique")
    if packets and set(packets) != {task.task_id for task in tasks}:
        raise ValueError("V2B-I3 controller packet ids must exactly match private task ids")
    return MetareasoningBenchmark(
        benchmark_id=str(payload.get("benchmark_id", "")), tasks=tuple(tasks),
        budget_profiles=profiles, utility_weights=_load_utility_weights(payload.get("utility_weights")),
        metadata={key: value for key, value in payload.items()
                  if key not in {"tasks", "budget_profiles", "utility_weights"}},
        artifact_hashes=artifact_hashes,
    )
