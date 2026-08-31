"""Deterministic checkpoint serializer for DAPH-X.

Serializes the complete runtime state for exact fork/restore.
Serialization is deterministic: all dicts sorted, all sequences
canonicalized, all floats rounded to fixed precision.

The checkpoint hash uniquely identifies the state:
  h_s = SHA256(serialize(s))

Two checkpoints with the same hash must produce identical trajectories
under the same action and downstream policy.
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from daph_x.graph.epistemic_graph import EpistemicGraph, GraphNode, GraphEdge, NodeType
from daph_x.belief.belief_engine import BeliefState


@dataclass(frozen=True)
class Checkpoint:
    """A deterministic checkpoint of the complete runtime state.

    Contains everything needed to exactly reproduce the state:
    graph, belief state, resources, task state, RNG seed, policy config.
    """
    # Core state
    graph: EpistemicGraph
    belief: BeliefState

    # Task state
    task_id: str
    correct_hypothesis_id: str
    expected_terminal: str
    oracle_path: tuple[str, ...]

    # RNG state
    seed: int

    # Policy configuration
    downstream_policy_id: str
    executive_version: str
    model_version: str

    # World model state (for now, just the graph)
    # In future: learned world model parameters

    # Hash (computed, not stored in serialized form)
    checkpoint_hash: str = ""

    def __post_init__(self):
        if not self.checkpoint_hash:
            # Compute hash from serialized form
            object.__setattr__(self, 'checkpoint_hash', self.compute_hash())

    def compute_hash(self) -> str:
        """Compute deterministic hash of the checkpoint."""
        data = self._serialize_for_hash()
        return hashlib.sha256(data).hexdigest()

    def _serialize_for_hash(self) -> bytes:
        """Serialize to deterministic bytes for hashing."""
        data = {
            "graph": _serialize_graph(self.graph),
            "belief": _serialize_belief(self.belief),
            "task_id": self.task_id,
            "correct_hypothesis_id": self.correct_hypothesis_id,
            "expected_terminal": self.expected_terminal,
            "oracle_path": list(self.oracle_path),
            "seed": self.seed,
            "downstream_policy_id": self.downstream_policy_id,
            "executive_version": self.executive_version,
            "model_version": self.model_version,
        }
        return json.dumps(data, sort_keys=True, default=str).encode()

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return {
            "graph": _serialize_graph(self.graph),
            "belief": _serialize_belief(self.belief),
            "task_id": self.task_id,
            "correct_hypothesis_id": self.correct_hypothesis_id,
            "expected_terminal": self.expected_terminal,
            "oracle_path": list(self.oracle_path),
            "seed": self.seed,
            "downstream_policy_id": self.downstream_policy_id,
            "executive_version": self.executive_version,
            "model_version": self.model_version,
            "checkpoint_hash": self.checkpoint_hash,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Checkpoint:
        """Deserialize from a dict."""
        graph = _deserialize_graph(data["graph"])
        belief = _deserialize_belief(data["belief"])
        return cls(
            graph=graph,
            belief=belief,
            task_id=data["task_id"],
            correct_hypothesis_id=data["correct_hypothesis_id"],
            expected_terminal=data["expected_terminal"],
            oracle_path=tuple(data["oracle_path"]),
            seed=data["seed"],
            downstream_policy_id=data["downstream_policy_id"],
            executive_version=data["executive_version"],
            model_version=data["model_version"],
            checkpoint_hash=data.get("checkpoint_hash", ""),
        )


def _serialize_graph(graph: EpistemicGraph) -> dict:
    """Serialize graph deterministically."""
    return {
        "nodes": {
            k: {
                "node_type": v.node_type.value,
                "label": v.label,
                "verification_state": v.verification_state,
                "temporal_status": v.temporal_status,
                "answer_action": v.answer_action,
                "source_id": v.source_id,
                "derived_from": list(v.derived_from) if v.derived_from else [],
            }
            for k, v in sorted(graph.nodes.items())
        },
        "edges": sorted(
            (e.source_id, e.target_id, e.edge_type.value)
            for e in graph.edges
        ),
        "steps_remaining": graph.steps_remaining,
        "verify_remaining": graph.verify_remaining,
        "retrieve_remaining": graph.retrieve_remaining,
        "search_remaining": graph.search_remaining,
        "reasoning_tokens_remaining": graph.reasoning_tokens_remaining,
        "elapsed_ms": graph.elapsed_ms,
        "max_elapsed_ms": graph.max_elapsed_ms,
    }


def _deserialize_graph(data: dict) -> EpistemicGraph:
    """Deserialize graph from dict."""
    nodes = {}
    for k, v in data["nodes"].items():
        nodes[k] = GraphNode(
            node_id=k,
            node_type=NodeType(v["node_type"]),
            label=v.get("label", ""),
            verification_state=v.get("verification_state", "UNVERIFIED"),
            temporal_status=v.get("temporal_status", "CURRENT"),
            answer_action=v.get("answer_action", ""),
            source_id=v.get("source_id"),
            derived_from=tuple(v.get("derived_from", [])),
        )

    edges = []
    for src, tgt, etype in data["edges"]:
        from daph_x.graph.epistemic_graph import EdgeType
        edges.append(GraphEdge(
            source_id=src,
            target_id=tgt,
            edge_type=EdgeType(etype),
        ))

    return EpistemicGraph(
        nodes=nodes,
        edges=tuple(edges),
        steps_remaining=data.get("steps_remaining", 10),
        verify_remaining=data.get("verify_remaining", 5),
        retrieve_remaining=data.get("retrieve_remaining", 3),
        search_remaining=data.get("search_remaining", 3),
        reasoning_tokens_remaining=data.get("reasoning_tokens_remaining", 256),
        elapsed_ms=data.get("elapsed_ms", 0),
        max_elapsed_ms=data.get("max_elapsed_ms", 30000),
    )


def _serialize_belief(belief: BeliefState) -> dict:
    """Serialize belief state deterministically."""
    return {
        "probabilities": dict(sorted(belief.probabilities.items())),
        "entropy": round(belief.entropy, 10),
        "unique_supported": belief.unique_supported,
        "readiness": belief.readiness.value,
        "n_supported": belief.n_supported,
        "n_contradicted": belief.n_contradicted,
        "n_weakened": belief.n_weakened,
        "n_untested": belief.n_untested,
    }


def _deserialize_belief(data: dict) -> BeliefState:
    """Deserialize belief state from dict."""
    from daph.epistemic.types import TerminalReadiness
    return BeliefState(
        probabilities=data["probabilities"],
        entropy=data["entropy"],
        unique_supported=data["unique_supported"],
        readiness=TerminalReadiness(data["readiness"]),
        n_supported=data["n_supported"],
        n_contradicted=data["n_contradicted"],
        n_weakened=data["n_weakened"],
        n_untested=data["n_untested"],
    )


def checkpoint_from_task_and_runtime(task, runtime, seed: int = 42) -> Checkpoint:
    """Create a checkpoint from a legacy task and runtime."""
    from daph_x.graph.epistemic_graph import build_graph_from_evidence_task
    from daph_x.belief.belief_engine import compute_belief_state

    graph = build_graph_from_evidence_task(task)
    belief = compute_belief_state(graph)

    return Checkpoint(
        graph=graph,
        belief=belief,
        task_id=task.task_id,
        correct_hypothesis_id=task.correct_hypothesis_id,
        expected_terminal=task.expected_terminal.value,
        oracle_path=task.oracle_resolution_path,
        seed=seed,
        downstream_policy_id="v3r2_confirmed",
        executive_version="daph_x_0.1.0",
        model_version="Qwen2.5-7B-Instruct-Q4_K_M",
    )
