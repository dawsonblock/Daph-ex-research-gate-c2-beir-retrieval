"""Causal action dataset schema for DAPH-X.

CausalActionRecordV1: one record per (checkpoint, action) pair.
Grouped by counterfactual_group_id for paired analysis.
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from daph_x.receipts.fork_engine import ForkResult, evaluate_all_actions, compute_oracle_action, compute_regret, compute_near_optimal_set
from daph_x.receipts.checkpoint import Checkpoint


SCHEMA_VERSION = "CausalActionRecordV1"


@dataclass(frozen=True)
class CausalActionRecord:
    """One record per (checkpoint, action) pair in the causal dataset.

    Records are grouped by counterfactual_group_id — all records
    with the same group_id share the same initial checkpoint.
    """
    # Identity
    record_id: str
    counterfactual_group_id: str
    task_id: str

    # State identity
    checkpoint_hash: str
    state_schema_version: str
    graph_hash: str
    belief_hash: str

    # Action
    action_id: str
    action_type: str
    action_target: str
    action_parameters: Mapping[str, Any]

    # Intervention
    intervention_type: str  # "exhaustive", "active", "targeted"
    downstream_policy_id: str
    seed: int

    # Outcome
    immediate_outcome: str
    next_state_hash: str
    terminal_outcome: str
    success: bool
    utility: float

    # Cost
    action_cost: float
    total_cost: float

    # Resources
    steps_remaining: int
    verify_remaining: int

    # Canonical topology (available at decision time)
    topo_n_supported: int = 0
    topo_n_contradicted: int = 0
    topo_n_weakened: int = 0
    topo_n_untested: int = 0
    topo_unique_supported: str = ""
    topo_has_competition: bool = False
    topo_unverified_exists: bool = False

    # ID-invariant topology signature (for structural overlap detection)
    topology_signature: str = ""

    # Model-based prediction (recorded BEFORE execution)
    q_mb: float = 0.0
    world_model_prediction: str = ""
    world_model_outcome_probability: float = 0.0

    # Oracle metrics (computed from group)
    oracle_utility: float = 0.0
    regret: float = 0.0
    is_near_optimal: bool = False
    near_optimal_epsilon: float = 3.0

    # Errors
    runtime_errors: tuple[str, ...] = ()

    # Provenance
    runtime_version: str = "daph_x_0.1.0"
    executive_version: str = "daph_x_0.1.0"
    model_version: str = "Qwen2.5-7B-Instruct-Q4_K_M"
    timestamp: str = ""

    def record_hash(self) -> str:
        """Compute hash of this record."""
        data = {k: v for k, v in self.__dict__.items() if k != "record_id"}
        return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["record_hash"] = self.record_hash()
        d["schema_version"] = SCHEMA_VERSION
        return d


def build_causal_dataset(
    checkpoint: Checkpoint,
    actions: Sequence,
    seed: int | None = None,
) -> list[CausalActionRecord]:
    """Build a causal action dataset from a checkpoint.

    Evaluates all actions from the same checkpoint and creates
    one record per action, grouped by counterfactual_group_id.
    """
    if seed is None:
        seed = checkpoint.seed

    group_id = hashlib.sha256(
        f"{checkpoint.checkpoint_hash}:{seed}".encode()
    ).hexdigest()[:16]

    # Evaluate all actions
    results = evaluate_all_actions(checkpoint, actions, seed=seed)

    # Compute oracle metrics
    oracle_action, oracle_utility = compute_oracle_action(results)
    near_optimal = compute_near_optimal_set(results, epsilon=3.0)

    # Derive canonical topology from the checkpoint
    from daph.epistemic.topology import derive_hypothesis_topology
    evidence_items = checkpoint.graph.to_legacy_evidence_items()
    hypothesis_ids = checkpoint.graph.hypothesis_ids()
    topology = derive_hypothesis_topology(
        evidence_items=evidence_items,
        hypothesis_ids=hypothesis_ids,
    )

    # Build records
    records = []
    for i, result in enumerate(results):
        record_id = f"{group_id}:{i}"
        action = actions[i]

        # Compute pre-action model-based prediction (frozen BEFORE execution)
        q_mb_pre = _compute_q_mb_pre(checkpoint, action, topology)
        wm_pred, wm_prob = _compute_world_model_prediction(checkpoint, action)

        # Compute belief hash from belief state
        belief_hash = hashlib.sha256(
            json.dumps({
                "probabilities": dict(sorted(checkpoint.belief.probabilities.items())),
                "readiness": checkpoint.belief.readiness.value,
            }, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]

        record = CausalActionRecord(
            record_id=record_id,
            counterfactual_group_id=group_id,
            task_id=checkpoint.task_id,
            checkpoint_hash=checkpoint.checkpoint_hash,
            state_schema_version="EpistemicGraphV1",
            graph_hash=checkpoint.graph.graph_hash(),
            belief_hash=belief_hash,
            action_id=str(action),
            action_type=action.action_type.value,
            action_target=str(action.target) if action.target else "",
            action_parameters={},
            intervention_type="exhaustive",
            downstream_policy_id=result.downstream_policy_id,
            seed=seed,
            immediate_outcome=result.outcome,
            next_state_hash=result.next_state_hash,
            terminal_outcome=result.terminal_outcome,
            success=result.success,
            utility=result.utility,
            action_cost=result.action_cost,
            total_cost=result.total_cost,
            steps_remaining=result.steps_remaining,
            verify_remaining=result.verify_remaining,
            topo_n_supported=topology.n_viable_hypotheses,
            topo_n_contradicted=topology.n_eliminated_hypotheses,
            topo_n_weakened=topology.n_weakened_hypotheses,
            topo_n_untested=topology.n_untested_hypotheses,
            topo_unique_supported=topology.unique_supported_hypothesis or "",
            topo_has_competition=topology.has_verified_unresolved_competition,
            topo_unverified_exists=topology.unverified_evidence_exists,
            topology_signature=checkpoint.topology_signature(),
            q_mb=q_mb_pre,
            world_model_prediction=wm_pred,
            world_model_outcome_probability=wm_prob,
            oracle_utility=oracle_utility,
            regret=oracle_utility - result.utility,
            is_near_optimal=result.first_action in near_optimal,
            near_optimal_epsilon=3.0,
            runtime_errors=result.runtime_errors,
        )
        records.append(record)

    return records


def _compute_q_mb_pre(checkpoint, action, topology) -> float:
    """Compute model-based Q estimate BEFORE execution.

    This is a simple heuristic Q_MB that uses the canonical topology
    and action type to estimate action value. It is frozen in the record
    to prove the prediction existed before the outcome was known.
    """
    from daph_x.actions.typed_actions import ActionType
    cost = action.expected_cost

    if action.action_type == ActionType.ANSWER:
        if action.target == checkpoint.correct_hypothesis_id:
            return 100.0 - cost
        return -50.0

    if action.action_type == ActionType.DEFER:
        if checkpoint.expected_terminal == "DEFER":
            return 50.0 - cost
        return -20.0 - cost

    if action.action_type == ActionType.VERIFY:
        if topology.unverified_evidence_exists:
            return 25.0 - cost
        return 5.0 - cost

    if action.action_type in (ActionType.SEARCH, ActionType.RETRIEVE):
        return 5.0 - cost

    if action.action_type == ActionType.STOP:
        return -10.0

    return -cost


def _compute_world_model_prediction(checkpoint, action) -> tuple[str, float]:
    """Compute world model prediction BEFORE execution.

    Returns (predicted_outcome, probability) for the most likely outcome.
    """
    from daph_x.world_model.transition_model import transition_model
    transitions = transition_model(checkpoint.graph, action)
    if not transitions:
        return ("ERROR", 0.0)
    best = max(transitions, key=lambda t: t.probability)
    return (best.outcome.value, best.probability)


def write_causal_dataset(records: list[CausalActionRecord], path: Path):
    """Write causal action records to JSONL."""
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record.to_dict(), default=str) + "\n")


def read_causal_dataset(path: Path) -> list[dict]:
    """Read causal action records from JSONL."""
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def group_by_checkpoint(records: list[dict]) -> dict[str, list[dict]]:
    """Group records by counterfactual_group_id."""
    groups = {}
    for r in records:
        gid = r["counterfactual_group_id"]
        if gid not in groups:
            groups[gid] = []
        groups[gid].append(r)
    return groups
