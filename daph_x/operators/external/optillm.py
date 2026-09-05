"""OptiLLM external operator adapter.

OptiLLM exposes 20+ inference-time approaches through an OpenAI-compatible
proxy. The approach is selected by prepending a slug to the model name:

    model = "{slug}-{base_model}"

or via the `optillm_approach` field in `extra_body`.

Supported approaches (partial list from OptiLLM README):
  - cot_reflection: CoT with reflection
  - plansearch: search over candidate plans
  - re2: rereading for improved reasoning
  - self_consistency: advanced self-consistency
  - z3: Z3 theorem prover for logical reasoning
  - bon: best-of-N
  - moa: mixture of agents
  - mcts: Monte Carlo tree search
  - rstar: RSTAR agent
  - rto: round-trip optimization
  - leap: leap-of-thought
  - executive: executive function approach

Note: When used with llama-server, only a subset of approaches are available
due to llama-server's sampling limitations:
  cot_reflection, leap, plansearch, rstar, rto, self_consistency, re2, z3
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from daph_x.backends.openai_compat import (
    ChatMessage,
    ExternalGenerationRequest,
    OpenAICompatibleBackend,
)
from daph_x.executive.budget import BudgetEnvelope
from daph_x.operators.external.base import (
    CognitiveOperator,
    CostVector,
    OperatorResult,
    OperatorSpec,
    StateMode,
)
from daph_x.operators.types import Candidate, RuntimeState
from daph_x.evaluation.answer_extractor import extract_answer


# ---------------------------------------------------------------------------
# Strategy metadata
# ---------------------------------------------------------------------------

# Map OptiLLM slugs to DAPH-X StateMode
_SLUG_STATE_MODE: dict[str, StateMode] = {
    "cot_reflection": StateMode.FRESH_SOLVE,
    "plansearch": StateMode.PARALLEL_SEARCH,
    "re2": StateMode.FRESH_SOLVE,
    "self_consistency": StateMode.FRESH_SOLVE,
    "z3": StateMode.FRESH_SOLVE,
    "bon": StateMode.FRESH_SOLVE,
    "moa": StateMode.FRESH_SOLVE,
    "mcts": StateMode.PARALLEL_SEARCH,
    "rstar": StateMode.PARALLEL_SEARCH,
    "rto": StateMode.FRESH_SOLVE,
    "leap": StateMode.FRESH_SOLVE,
    "executive": StateMode.FRESH_SOLVE,
}

# Map slugs to tier
_SLUG_TIER: dict[str, int] = {
    "cot_reflection": 1,
    "re2": 1,
    "leap": 1,
    "z3": 1,
    "self_consistency": 2,
    "bon": 2,
    "rto": 2,
    "plansearch": 2,
    "moa": 2,
    "mcts": 3,
    "rstar": 3,
    "executive": 2,
}

# Approaches compatible with llama-server (no multi-sample required)
_LLAMACPP_COMPATIBLE = {"cot_reflection", "leap", "plansearch", "rstar", "rto", "self_consistency", "re2", "z3"}

# Approaches requiring multi-response sampling (not supported by llama-server)
_REQUIRES_MULTI_SAMPLE = {"bon", "moa", "mcts"}


def is_slug_compatible_with_capabilities(slug: str, capabilities: set[str]) -> bool:
    """Check if an OptiLLM slug is compatible with the available capabilities.

    llama-server and Ollama do not support multi-response sampling, which
    limits the available approaches. If the backend advertises
    'multi_sample' capability, all slugs are allowed. Otherwise, only
    llama-server-compatible slugs are admissible.
    """
    if "multi_sample" in capabilities:
        return True
    if slug in _LLAMACPP_COMPATIBLE:
        return True
    if slug in _REQUIRES_MULTI_SAMPLE:
        return False
    # Unknown slugs: allow by default, will be caught by live conformance
    return True


# ---------------------------------------------------------------------------
# Discrete operator profiles
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OptiLLMProfile:
    """Frozen profile for an OptiLLM approach."""
    profile_id: str
    slug: str
    strategy_params: dict[str, Any] = field(default_factory=dict)


# Frozen profiles per R14_PROTOCOL.md §6
PROFILES: dict[str, OptiLLMProfile] = {
    "OPT_PLANSEARCH_LOW": OptiLLMProfile(
        profile_id="OPT_PLANSEARCH_LOW",
        slug="plansearch",
        strategy_params={"n": 3},
    ),
    "OPT_MOA_LOW": OptiLLMProfile(
        profile_id="OPT_MOA_LOW",
        slug="moa",
        strategy_params={"layers": 2, "temperature": 0.7},
    ),
    "OPT_SC_LOW": OptiLLMProfile(
        profile_id="OPT_SC_LOW",
        slug="self_consistency",
        strategy_params={"n": 4, "temperature": 0.7},
    ),
    "OPT_COT_REFLECT": OptiLLMProfile(
        profile_id="OPT_COT_REFLECT",
        slug="cot_reflection",
        strategy_params={},
    ),
    "OPT_RE2": OptiLLMProfile(
        profile_id="OPT_RE2",
        slug="re2",
        strategy_params={},
    ),
    "OPT_BON_LOW": OptiLLMProfile(
        profile_id="OPT_BON_LOW",
        slug="bon",
        strategy_params={"n": 4},
    ),
}


# ---------------------------------------------------------------------------
# OptiLLM adapter operator
# ---------------------------------------------------------------------------

class OptiLLMOperator(CognitiveOperator):
    """A single OptiLLM approach exposed as a DAPH-X operator."""

    def __init__(
        self,
        profile: OptiLLMProfile,
        backend: OpenAICompatibleBackend,
        operator_version: str = "1",
    ):
        self._profile = profile
        self._backend = backend
        self._operator_version = operator_version

        state_mode = _SLUG_STATE_MODE.get(profile.slug, StateMode.FRESH_SOLVE)
        tier = _SLUG_TIER.get(profile.slug, 2)

        self._spec = OperatorSpec(
            operator_id=profile.profile_id,
            operator_version=operator_version,
            provider="optillm",
            strategy=profile.slug,
            strategy_version="1",
            state_mode=state_mode,
            tier=tier,
            backend_requirements=("openai_compatible",),
            supports_budget=True,
            supports_seed=True,
            requires_logprobs=False,
            requires_whitebox=False,
            max_context_tokens=4096,
            provenance={
                "optillm_slug": profile.slug,
                "optillm_params": dict(profile.strategy_params),
                "backend_url": backend.base_url,
            },
        )

    @property
    def spec(self) -> OperatorSpec:
        return self._spec

    def is_admissible(
        self,
        state: RuntimeState,
        capabilities: set[str] | Sequence[str] | None = None,
        budget: BudgetEnvelope | None = None,
    ) -> bool:
        """Check admissibility including OptiLLM slug/backend compatibility."""
        if capabilities is not None:
            caps = set(capabilities)
            if not is_slug_compatible_with_capabilities(self._profile.slug, caps):
                return False
        return super().is_admissible(state, capabilities=capabilities, budget=budget)

    def estimate_cost(
        self,
        state: RuntimeState,
        budget: BudgetEnvelope | None = None,
    ) -> CostVector:
        params = self._profile.strategy_params
        n = params.get("n", 1)
        layers = params.get("layers", 1)
        effective_calls = max(n, layers, 1)

        return CostVector(
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            gateway_calls=1,
            underlying_model_calls=effective_calls,
            wall_ms=None,
        )

    def execute(
        self,
        state: RuntimeState,
        backend: Any | None = None,
        budget: BudgetEnvelope | None = None,
        replicate_id: int = 42,
    ) -> OperatorResult:
        """Execute the OptiLLM approach via the OpenAI-compatible proxy."""
        active_backend = backend if backend is not None else self._backend
        if not isinstance(active_backend, OpenAICompatibleBackend):
            return OperatorResult(
                terminal_answer="",
                status="FAILURE",
                error_code="BACKEND_TYPE_ERROR",
                error_message=f"Expected OpenAICompatibleBackend, got {type(active_backend).__name__}",
            )

        messages = (
            ChatMessage(role="system", content="You are a helpful reasoning assistant. Provide a clear final answer."),
            ChatMessage(role="user", content=state.task_prompt),
        )

        # OptiLLM uses the slug prefix on the model name to select the approach
        slug_model = f"{self._profile.slug}-{active_backend.model}"

        # Pass strategy params via extra_body
        extra_params: dict[str, Any] = {}
        for k, v in self._profile.strategy_params.items():
            extra_params[k] = v

        # Critical: override the server's default approach. When OptiLLM's
        # default approach is 'none' (the factory default), it prepends
        # 'none-' to the model name, producing 'none-cot_reflection-qwen'.
        # parse_combined_approach then sees approaches=['none','cot_reflection']
        # and approaches[0]=='none' triggers a pass-through, silently skipping
        # the requested strategy. Setting optillm_approach='auto' prevents
        # the prepend so the slug prefix is the only approach.
        extra_params["optillm_approach"] = "auto"

        max_tokens = 4096
        if budget is not None and budget.max_tokens is not None:
            max_tokens = min(max_tokens, budget.max_tokens)

        request = ExternalGenerationRequest(
            messages=messages,
            model=slug_model,
            temperature=self._profile.strategy_params.get("temperature", 0.0),
            max_tokens=max_tokens,
            seed=replicate_id,
            extra_params=extra_params,
        )

        result = active_backend.generate(request)

        if not result.is_success:
            return OperatorResult(
                terminal_answer="",
                status="FAILURE" if result.error_code != "TIMEOUT" else "TIMEOUT",
                error_code=result.error_code,
                error_message=result.error_message,
                cost=CostVector(
                    prompt_tokens=result.prompt_tokens,
                    completion_tokens=result.completion_tokens,
                    total_tokens=result.total_tokens,
                    gateway_calls=1,
                    underlying_model_calls=None,
                    wall_ms=result.latency_ms,
                ),
                provenance={
                    "optillm_slug": self._profile.slug,
                    "request_hash": result.request_hash,
                    **active_backend.service_identity.to_dict(),
                },
            )

        # Use canonical answer extractor for terminal_answer, full text for trace
        terminal = extract_answer(result.text, state.answer_type)
        candidate = Candidate(
            candidate_id=f"opt_{self._profile.profile_id}_{replicate_id}",
            answer=terminal,
            reasoning_trace=result.text,
            temperature=self._profile.strategy_params.get("temperature", 0.0),
            seed=replicate_id,
            generation_index=0,
            metadata={
                "slug": self._profile.slug,
                "finish_reason": result.finish_reason,
                "raw_answer": result.text.strip()[:200],
            },
        )

        return OperatorResult(
            terminal_answer=terminal,
            candidates=(candidate,),
            reasoning_artifacts={
                "raw_text": result.text,
                "finish_reason": result.finish_reason,
            },
            diagnostics={
                "slug": self._profile.slug,
                "profile_id": self._profile.profile_id,
            },
            status="SUCCESS",
            cost=CostVector(
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                total_tokens=result.total_tokens,
                # gateway_calls: DAPH-X made 1 HTTP request to OptiLLM proxy
                gateway_calls=1,
                # underlying_model_calls: unknown until OptiLLM reports it
                underlying_model_calls=None,
                wall_ms=result.latency_ms,
            ),
            provenance={
                "optillm_slug": self._profile.slug,
                "optillm_params": dict(self._profile.strategy_params),
                "backend_url": active_backend.base_url,
                "endpoint_url": active_backend.chat_completions_url,
                "model_id": result.model_id,
                "request_hash": result.request_hash,
                "response_hash": result.response_hash,
                **active_backend.service_identity.to_dict(),
            },
        )


def make_operator(
    profile_id: str,
    backend: OpenAICompatibleBackend,
    operator_version: str = "1",
) -> OptiLLMOperator:
    """Create an OptiLLM operator from a frozen profile ID."""
    if profile_id not in PROFILES:
        raise ValueError(f"Unknown OptiLLM profile: {profile_id}. Available: {list(PROFILES.keys())}")
    return OptiLLMOperator(
        profile=PROFILES[profile_id],
        backend=backend,
        operator_version=operator_version,
    )
