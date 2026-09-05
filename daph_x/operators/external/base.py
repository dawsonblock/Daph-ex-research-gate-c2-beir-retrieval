"""Generic External Cognitive Operator ABI.

Defines the contract for external reasoning operators (ThinkBooster, OptiLLM,
DeepConf, MUR, PaCoRe, Plan-and-Budget) and native reference baselines.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

from daph_x.executive.budget import BudgetEnvelope
from daph_x.operators.types import Candidate, Observation, RuntimeState


class StateMode(str, Enum):
    """How the operator interacts with existing state."""
    FRESH_SOLVE = "FRESH_SOLVE"              # Solves from raw problem prompt alone
    STATE_CONDITIONED = "STATE_CONDITIONED"  # Uses prompt + observable state summary/candidates
    CANDIDATE_RERANK = "CANDIDATE_RERANK"    # Reranks/selects from existing candidates
    CONTINUATION = "CONTINUATION"            # Generates additional candidates appending to state
    VERIFICATION = "VERIFICATION"            # Audits/verifies current answer and optionally repairs
    PARALLEL_SEARCH = "PARALLEL_SEARCH"      # Runs multi-path parallel search/tree exploration


@dataclass(frozen=True)
class CostVector:
    """Multidimensional cost measurement.

    Rule: Unmeasured dimensions are None (NOT_MEASURED), never zero.
    Only genuinely free operations (e.g. STOP) record explicit 0.

    gateway_calls: HTTP requests made by DAPH-X to the external service.
    underlying_model_calls: LLM inference calls made by the external service
        internally (e.g. Best-of-N with N=8 → 8 underlying calls). None if
        the service does not report this.
    """
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    gateway_calls: int | None = None
    underlying_model_calls: int | None = None
    wall_ms: float | None = None
    gpu_ms: float | None = None
    gpu_memory_peak_mb: float | None = None
    api_cost_usd: float | None = None
    estimated_flops: float | None = None

    def effective_total_tokens(self) -> int | None:
        """Return total tokens if measurable, else None.

        None means 'not measured' — this is NOT the same as zero.
        Budget checks must treat None as 'unknown, cannot verify'.
        """
        if self.total_tokens is not None:
            return self.total_tokens
        if self.prompt_tokens is not None and self.completion_tokens is not None:
            return self.prompt_tokens + self.completion_tokens
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "effective_tokens": self.effective_total_tokens(),
            "gateway_calls": self.gateway_calls,
            "underlying_model_calls": self.underlying_model_calls,
            "wall_ms": self.wall_ms,
            "gpu_ms": self.gpu_ms,
            "gpu_memory_peak_mb": self.gpu_memory_peak_mb,
            "api_cost_usd": self.api_cost_usd,
            "estimated_flops": self.estimated_flops,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CostVector:
        return cls(
            prompt_tokens=data.get("prompt_tokens"),
            completion_tokens=data.get("completion_tokens"),
            total_tokens=data.get("total_tokens"),
            gateway_calls=data.get("gateway_calls"),
            underlying_model_calls=data.get("underlying_model_calls"),
            wall_ms=data.get("wall_ms"),
            gpu_ms=data.get("gpu_ms"),
            gpu_memory_peak_mb=data.get("gpu_memory_peak_mb"),
            api_cost_usd=data.get("api_cost_usd"),
            estimated_flops=data.get("estimated_flops"),
        )


@dataclass(frozen=True)
class OperatorSpec:
    """Declarative specification and capability requirements of an operator."""
    operator_id: str
    operator_version: str
    provider: str                      # "daph_native", "thinkbooster", "optillm", "deepconf", etc.
    strategy: str                      # "stop", "verify", "best_of_n", "deepconf", "mur", etc.
    strategy_version: str
    state_mode: StateMode
    tier: int                          # 0 (free/stop), 1 (cheap/fast), 2 (moderate), 3 (expensive)
    backend_requirements: tuple[str, ...] = ("openai_compatible",)
    supports_budget: bool = True
    supports_seed: bool = True
    requires_logprobs: bool = False
    requires_whitebox: bool = False
    max_context_tokens: int = 4096
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "operator_id": self.operator_id,
            "operator_version": self.operator_version,
            "provider": self.provider,
            "strategy": self.strategy,
            "strategy_version": self.strategy_version,
            "state_mode": self.state_mode.value,
            "tier": self.tier,
            "backend_requirements": list(self.backend_requirements),
            "supports_budget": self.supports_budget,
            "supports_seed": self.supports_seed,
            "requires_logprobs": self.requires_logprobs,
            "requires_whitebox": self.requires_whitebox,
            "max_context_tokens": self.max_context_tokens,
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True)
class OperatorResult:
    """Standardized output of any cognitive operator execution."""
    terminal_answer: str
    candidates: tuple[Candidate, ...] = field(default_factory=tuple)
    reasoning_artifacts: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    status: str = "SUCCESS"            # "SUCCESS", "FAILURE", "TIMEOUT", "NOT_SUPPORTED", "BUDGET_EXCEEDED"
    error_code: str | None = None
    error_message: str | None = None
    cost: CostVector = field(default_factory=CostVector)
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        return self.status == "SUCCESS"

    def to_observation(self, operator_id: str, operator_version: str) -> Observation:
        """Convert to legacy Observation for backward compatibility with R12/R13 receipts."""
        confidence = float(self.diagnostics.get("confidence", 0.0) or 0.0)
        verification_score = float(self.diagnostics.get("verification_score", 0.0) or 0.0)
        return Observation(
            operator_id=operator_id,
            operator_version=operator_version,
            candidate_answer=self.terminal_answer,
            reasoning_trace=str(self.reasoning_artifacts.get("reasoning_trace", "")),
            confidence=confidence,
            verification_score=verification_score,
            evidence=dict(self.reasoning_artifacts),
            success=self.is_success,
            failure_reason=self.error_message or "",
            cost=self.cost.to_dict(),
            metadata=dict(self.diagnostics),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "terminal_answer": self.terminal_answer,
            "candidates": [
                {
                    "candidate_id": c.candidate_id,
                    "answer": c.answer,
                    "reasoning_trace": c.reasoning_trace,
                    "temperature": c.temperature,
                    "seed": c.seed,
                    "generation_index": c.generation_index,
                    "metadata": dict(c.metadata),
                }
                for c in self.candidates
            ],
            "reasoning_artifacts": dict(self.reasoning_artifacts),
            "diagnostics": dict(self.diagnostics),
            "status": self.status,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "cost": self.cost.to_dict(),
            "provenance": dict(self.provenance),
        }


class CognitiveOperator(ABC):
    """Abstract base class for all cognitive operators."""

    @property
    @abstractmethod
    def spec(self) -> OperatorSpec:
        """Return the immutable specification of this operator."""
        ...

    @abstractmethod
    def estimate_cost(
        self,
        state: RuntimeState,
        budget: BudgetEnvelope | None = None,
    ) -> CostVector:
        """Estimate the resource cost before execution."""
        ...

    @abstractmethod
    def execute(
        self,
        state: RuntimeState,
        backend: Any,
        budget: BudgetEnvelope | None = None,
        replicate_id: int = 42,
    ) -> OperatorResult:
        """Execute the cognitive operation on the given state."""
        ...

    def is_admissible(
        self,
        state: RuntimeState,
        capabilities: set[str] | Sequence[str] | None = None,
        budget: BudgetEnvelope | None = None,
    ) -> bool:
        """Check admissibility given state, hardware/service capabilities, and budget.

        Capabilities are properties of the execution provider, not necessarily
        DAPH-X itself. For example, a ThinkBooster service that owns a vLLM
        backend advertises 'thinkbooster.whitebox' and 'thinkbooster.mur' as
        service capabilities, even if DAPH-X's client machine has no white-box
        access.
        """
        if capabilities is not None:
            caps_set = set(capabilities)
            # Check declared backend requirements
            for req in self.spec.backend_requirements:
                if req not in caps_set:
                    return False
            # Enforce white-box requirement
            if self.spec.requires_whitebox and "whitebox" not in caps_set:
                # Allow service-level whitebox capability (e.g. thinkbooster.whitebox)
                if not any(c.endswith(".whitebox") for c in caps_set):
                    return False
            # Enforce logprobs requirement
            if self.spec.requires_logprobs and "logprobs" not in caps_set:
                if not any("logprobs" in c for c in caps_set):
                    return False
        if budget is not None:
            est = self.estimate_cost(state, budget)
            # Only check budget dimensions that are actually measured (not None)
            tokens = est.effective_total_tokens()
            if tokens is not None and budget.max_tokens is not None and tokens > budget.max_tokens:
                return False
            if est.gateway_calls is not None and budget.max_calls is not None and est.gateway_calls > budget.max_calls:
                return False
            if est.wall_ms is not None and budget.max_wall_ms is not None and est.wall_ms > budget.max_wall_ms:
                return False
            if est.gpu_ms is not None and budget.max_gpu_ms is not None and est.gpu_ms > budget.max_gpu_ms:
                return False
            if est.api_cost_usd is not None and budget.max_cost_usd is not None and est.api_cost_usd > budget.max_cost_usd:
                return False
        return True
