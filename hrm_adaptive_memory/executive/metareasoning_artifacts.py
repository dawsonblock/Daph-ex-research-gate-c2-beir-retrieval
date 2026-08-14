"""Canonical artifact-graph resolution for V2B metareasoning benchmarks."""
from __future__ import annotations

from pathlib import PurePosixPath
from pathlib import Path
import hashlib
from typing import Callable, Mapping


BENCHMARK_ARTIFACT_FIELDS = {
    "private_environment": "private_environment_path",
    "controller_packets": "controller_packets_path",
    "task_extension": "task_extension_path",
    "controller_packets_extension": "controller_packets_extension_path",
    "task_families": "task_families_path",
    "split_definitions": "split_definitions_path",
    "surface_templates": "surface_templates_path",
    "balance_report": "balance_report_path",
    "structural_diversity_report": "structural_diversity_report_path",
    "oracle_cache_manifest": "oracle_cache_manifest_path",
}
PROTOCOL_ARTIFACT_FIELDS = {
    "observation_masks": "observation_masks_path",
    "policy": "policy_path",
    "utility": "utility_path",
    "resource_profiles": "resource_profiles_path",
}


def repository_relative(base_path: str, referenced_path: object) -> str:
    """Resolve one repository-relative edge and reject absolute/escaping paths."""
    if not isinstance(referenced_path, str) or not referenced_path:
        raise RuntimeError("metareasoning artifact graph has an empty path")
    if PurePosixPath(base_path).is_absolute() or PurePosixPath(referenced_path).is_absolute():
        raise RuntimeError("metareasoning artifact paths must be repository-relative")
    candidate = PurePosixPath(base_path).parent.joinpath(referenced_path)
    parts: list[str] = []
    for part in candidate.parts:
        if part in {"", "."}:
            continue
        if part == "..":
            if not parts:
                raise RuntimeError("metareasoning artifact escapes the repository")
            parts.pop()
        else:
            parts.append(part)
    if not parts:
        raise RuntimeError("metareasoning artifact resolves to the repository root")
    return PurePosixPath(*parts).as_posix()


def resolve_benchmark_artifact_graph(*, manifest_path: str,
                                     manifest: Mapping[str, object],
                                     protocol_path: str | None = None,
                                     protocol: Mapping[str, object] | None = None,
                                     json_loader: Callable[[str], Mapping[str, object]] | None = None,
                                     ) -> dict[str, str]:
    """Return the complete canonical graph affecting observations or scoring."""
    graph = {"benchmark_manifest": manifest_path}
    for name, field in BENCHMARK_ARTIFACT_FIELDS.items():
        if field in manifest:
            graph[name] = repository_relative(manifest_path, manifest[field])
    if protocol_path is not None:
        if protocol is None:
            raise RuntimeError("metareasoning protocol payload is required for closure")
        graph["protocol"] = protocol_path
        for name, field in PROTOCOL_ARTIFACT_FIELDS.items():
            if field not in protocol:
                raise RuntimeError(f"metareasoning protocol lacks {field}")
            # Protocol dependency paths are repository-root relative.
            graph[name] = repository_relative("protocol-root.json", protocol[field])
    if "oracle_cache_manifest" in graph and json_loader is not None:
        cache = json_loader(graph["oracle_cache_manifest"])
        graph.update({role: path for role, (path, _) in oracle_cache_artifacts(cache).items()})
    return graph


def oracle_cache_artifacts(cache: Mapping[str, object]) -> dict[str, tuple[str, str]]:
    """Resolve every table/report named by a frozen oracle-cache manifest."""
    if cache.get("schema") != "DAPH_V2B_I3_3_ORACLE_CACHE_MANIFEST_V1":
        raise RuntimeError("I3.3 oracle cache has an unsupported schema")
    result: dict[str, tuple[str, str]] = {}
    latent = cache.get("latent_oracles")
    difficulty = cache.get("difficulty_report")
    sequential = cache.get("sequential_observable_oracles")
    if not isinstance(latent, Mapping) or not isinstance(difficulty, Mapping) or not isinstance(sequential, Mapping):
        raise RuntimeError("I3.3 oracle cache is incomplete")
    entries = {"oracle_latent_tables": latent, "oracle_difficulty_report": difficulty}
    oracle_balance = cache.get("oracle_balance_report")
    if oracle_balance is not None:
        entries["oracle_balance_report"] = oracle_balance
    topology_allocation = cache.get("topology_allocation")
    if topology_allocation is not None:
        entries["topology_allocation"] = topology_allocation
    topology_report = cache.get("topology_diversity_report")
    if topology_report is not None:
        entries["topology_diversity_report"] = topology_report
    entries.update({f"oracle_sequential_{str(name).lower()}": value
                    for name, value in sequential.items()})
    for role, entry in entries.items():
        if (not isinstance(entry, Mapping) or not isinstance(entry.get("path"), str)
                or not isinstance(entry.get("sha256"), str)
                or len(str(entry["sha256"])) != 64):
            raise RuntimeError("I3.3 oracle cache entry is malformed")
        result[role] = (repository_relative("repository-root.json", entry["path"]),
                        str(entry["sha256"]))
    return result


def artifact_graph_sha256(root: str | Path, graph: Mapping[str, str]) -> str:
    """Commit role, path, and exact bytes for a closed artifact graph."""
    root = Path(root).resolve()
    hasher = hashlib.sha256()
    for role, relative in sorted(graph.items()):
        raw = (root / relative).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        hasher.update(role.encode() + b"\0" + relative.encode() + b"\0" + digest.encode() + b"\n")
    return hasher.hexdigest()
