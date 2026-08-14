"""One frozen utility implementation shared by I3.1 runtime and oracle paths."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

from hrm_adaptive_memory.cognitive_control.core import DecisionAction

from .resources import DEFAULT_ACTION_COSTS, ResourceState


UTILITY_SCHEMA = "DAPH_V2B_I3_1_UTILITY_V1"
UTILITY_REVISION = "v2b-i3.1-utility-v1"


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def frozen_action_cost_hash() -> str:
    return _canonical_hash({action.value: asdict(cost) for action, cost in sorted(
        DEFAULT_ACTION_COSTS.items(), key=lambda item: item[0].value)})


@dataclass(frozen=True)
class MetareasoningUtility:
    """Frozen reward and resource-cost semantics for a deterministic I3.1 run."""

    correct_answer: float
    incorrect_answer: float
    correct_defer: float
    incorrect_defer: float
    correct_stop: float
    incorrect_stop: float
    executive_step: float
    retrieval: float
    verification: float
    search: float
    reasoning_128_tokens: float
    logical_ms: float
    source_sha256: str
    revision: str = UTILITY_REVISION

    @classmethod
    def from_file(cls, path: str | Path) -> "MetareasoningUtility":
        file = Path(path)
        payload = json.loads(file.read_text())
        expected = {
            "schema", "status", "utility_id", "terminal_rewards", "resource_weights",
            "action_cost_sha256", "revision",
        }
        if set(payload) != expected or payload["schema"] != UTILITY_SCHEMA:
            raise ValueError("I3.1 utility configuration has an unsupported schema")
        if payload["status"] != "FROZEN_FOR_DEVELOPMENT":
            raise ValueError("I3.1 utility configuration must be frozen for development")
        if payload["action_cost_sha256"] != frozen_action_cost_hash():
            raise ValueError("I3.1 utility is not bound to the executor action-cost table")
        terminal = payload["terminal_rewards"]
        weights = payload["resource_weights"]
        terminal_fields = {"correct_answer", "incorrect_answer", "correct_defer", "incorrect_defer",
                           "correct_stop", "incorrect_stop"}
        weight_fields = {"executive_step", "retrieval", "verification", "search",
                         "reasoning_128_tokens", "logical_ms"}
        if not isinstance(terminal, Mapping) or set(terminal) != terminal_fields:
            raise ValueError("I3.1 utility has an invalid terminal reward set")
        if not isinstance(weights, Mapping) or set(weights) != weight_fields:
            raise ValueError("I3.1 utility has an invalid resource weight set")
        if any(float(value) < 0 for value in weights.values()):
            raise ValueError("I3.1 resource weights must be nonnegative")
        return cls(**{key: float(terminal[key]) for key in terminal_fields},
                   **{key: float(weights[key]) for key in weight_fields},
                   source_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
                   revision=str(payload["revision"]))

    @classmethod
    def from_i3_weights(cls, weights: Mapping[str, float]) -> "MetareasoningUtility":
        """Compatibility adapter for frozen I3 development utilities."""
        expected = {"success_reward", "failure_penalty", "executive_step", "retrieval",
                    "verification", "search", "reasoning_128_tokens", "logical_ms"}
        if set(weights) != expected:
            raise ValueError("unsupported legacy I3 utility weights")
        source = _canonical_hash(dict(sorted(weights.items())))
        return cls(
            correct_answer=float(weights["success_reward"]),
            incorrect_answer=-float(weights["failure_penalty"]),
            correct_defer=float(weights["success_reward"]),
            incorrect_defer=-float(weights["failure_penalty"]),
            correct_stop=float(weights["success_reward"]),
            incorrect_stop=-float(weights["failure_penalty"]),
            executive_step=float(weights["executive_step"]), retrieval=float(weights["retrieval"]),
            verification=float(weights["verification"]), search=float(weights["search"]),
            reasoning_128_tokens=float(weights["reasoning_128_tokens"]),
            logical_ms=float(weights["logical_ms"]), source_sha256=source,
            revision="legacy-i3-utility-adapter-v1")

    @property
    def sha256(self) -> str:
        return _canonical_hash({
            "revision": self.revision, "source_sha256": self.source_sha256,
            "action_cost_sha256": frozen_action_cost_hash(),
            "terminal": {key: getattr(self, key) for key in (
                "correct_answer", "incorrect_answer", "correct_defer", "incorrect_defer",
                "correct_stop", "incorrect_stop")},
            "resource": {key: getattr(self, key) for key in (
                "executive_step", "retrieval", "verification", "search",
                "reasoning_128_tokens", "logical_ms")},
        })

    def action_utility(self, before: ResourceState, after: ResourceState) -> float:
        """Compatibility net utility for a cost-only nonterminal transition."""
        return -self.action_cost(before, after)

    def action_cost(self, before: ResourceState, after: ResourceState) -> float:
        """Return a positive, explicitly named resource cost."""
        return (
            self.executive_step * (after.executive_steps_used - before.executive_steps_used)
            + self.retrieval * (after.retrieval_calls_used - before.retrieval_calls_used)
            + self.verification * (after.verification_calls_used - before.verification_calls_used)
            + self.search * (after.search_calls_used - before.search_calls_used)
            + self.reasoning_128_tokens *
            ((after.reasoning_tokens_used - before.reasoning_tokens_used) / 128)
            + self.logical_ms * (after.elapsed_ms - before.elapsed_ms)
        )

    @staticmethod
    def immediate_reward(*, before: ResourceState, after: ResourceState) -> float:
        """Frozen I3.2.2 shaping reward; kept separate from resource cost."""
        del before, after
        return 0.0

    def net_step_utility(self, before: ResourceState, after: ResourceState) -> float:
        return self.immediate_reward(before=before, after=after) - self.action_cost(before, after)

    def terminal_reward(self, action: DecisionAction, success: bool) -> float:
        if action is DecisionAction.ANSWER:
            return self.correct_answer if success else self.incorrect_answer
        if action is DecisionAction.DEFER:
            return self.correct_defer if success else self.incorrect_defer
        if action is DecisionAction.STOP:
            return self.correct_stop if success else self.incorrect_stop
        # RESOURCE_EXHAUSTED/nonterminal dead-ends are terminal failures.
        return self.incorrect_answer
