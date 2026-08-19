"""Loader and saver for evidence-bearing benchmark tasks."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from hrm_adaptive_memory.cognitive_control.core import DecisionAction
from hrm_adaptive_memory.cognitive_control.state import (
    TemporalStatus, VerificationState,
)
from hrm_adaptive_memory.executive.resources import ResourceBudget

from .schema import (
    EvidenceItem, EvidenceTask, EvidenceHypothesis,
    EVIDENCE_SCHEMA, EVIDENCE_VERSION,
)


@dataclass(frozen=True)
class EvidenceBenchmark:
    """A frozen evidence-bearing benchmark."""
    benchmark_id: str
    tasks: tuple[EvidenceTask, ...]
    budget_profiles: Mapping[str, ResourceBudget]

    def for_split(self, split: str) -> "EvidenceBenchmark":
        tasks = tuple(t for t in self.tasks if t.split == split)
        if not tasks:
            raise ValueError(f"evidence benchmark split is empty: {split}")
        return EvidenceBenchmark(self.benchmark_id, tasks, self.budget_profiles)

    def budget_for(self, task: EvidenceTask) -> ResourceBudget:
        return self.budget_profiles[task.budget_profile]


def save_evidence_benchmark(
    benchmark: EvidenceBenchmark,
    path: str | Path,
) -> None:
    """Save an evidence benchmark to a JSON file."""
    payload = {
        "schema": EVIDENCE_SCHEMA,
        "version": EVIDENCE_VERSION,
        "benchmark_id": benchmark.benchmark_id,
        "budget_profiles": {
            name: {
                "max_executive_steps": b.max_executive_steps,
                "max_reasoning_tokens": b.max_reasoning_tokens,
                "max_retrieval_calls": b.max_retrieval_calls,
                "max_verification_calls": b.max_verification_calls,
                "max_search_calls": b.max_search_calls,
                "max_elapsed_ms": b.max_elapsed_ms,
            }
            for name, b in benchmark.budget_profiles.items()
        },
        "tasks": [t.as_dict() for t in benchmark.tasks],
    }
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_evidence_benchmark(path: str | Path) -> EvidenceBenchmark:
    """Load an evidence benchmark from a JSON file."""
    payload = json.loads(Path(path).read_text())
    if payload.get("schema") != EVIDENCE_SCHEMA:
        raise ValueError(f"unsupported evidence benchmark schema: {payload.get('schema')}")
    if payload.get("version") != EVIDENCE_VERSION:
        raise ValueError(f"unsupported evidence benchmark version: {payload.get('version')}")

    # Load budget profiles
    profiles: dict[str, ResourceBudget] = {}
    for name, values in payload.get("budget_profiles", {}).items():
        profiles[name] = ResourceBudget(**dict(values))

    # Load tasks
    tasks: list[EvidenceTask] = []
    for raw in payload.get("tasks", []):
        hypotheses = tuple(
            EvidenceHypothesis(
                hypothesis_id=h["hypothesis_id"],
                proposition=h["proposition"],
                answer_action=DecisionAction(h["answer_action"]),
                answer_payload=h["answer_payload"],
            )
            for h in raw["hypotheses"]
        )
        evidence = tuple(
            EvidenceItem(
                evidence_id=e["evidence_id"],
                proposition=e["proposition"],
                source_class=e["source_class"],
                supports=tuple(e.get("supports", ())),
                contradicts=tuple(e.get("contradicts", ())),
                verification_state=VerificationState(e["verification_state"]),
                temporal_status=TemporalStatus(e["temporal_status"]),
                retrieved=e.get("retrieved", False),
                verify_result=e.get("verify_result"),
            )
            for e in raw["evidence_items"]
        )
        task = EvidenceTask(
            task_id=raw["task_id"],
            split=raw["split"],
            category=raw["category"],
            task_summary=raw["task_summary"],
            high_stakes=raw["high_stakes"],
            budget_profile=raw["budget_profile"],
            hypotheses=hypotheses,
            evidence_items=evidence,
            retrieve_exposes=tuple(raw.get("retrieve_exposes", ())),
            search_exposes=tuple(raw.get("search_exposes", ())),
            oracle_resolution_path=tuple(raw.get("oracle_resolution_path", ())),
            expected_terminal=DecisionAction(raw["expected_terminal"]),
            correct_hypothesis_id=raw["correct_hypothesis_id"],
        )
        tasks.append(task)

    if len({t.task_id for t in tasks}) != len(tasks):
        raise ValueError("evidence benchmark task ids must be unique")

    return EvidenceBenchmark(
        benchmark_id=payload.get("benchmark_id", ""),
        tasks=tuple(tasks),
        budget_profiles=profiles,
    )
