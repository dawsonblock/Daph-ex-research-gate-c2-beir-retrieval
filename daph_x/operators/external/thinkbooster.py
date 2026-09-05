"""ThinkBooster external operator adapter.

ThinkBooster exposes its strategies through an OpenAI-compatible gateway where
the strategy and scorer are encoded in the URL path:

    http://<host>:<port>/v1/{strategy}/{scorer}

This adapter wraps that gateway as a set of discrete CognitiveOperator profiles
per R14_PROTOCOL.md §6.

Strategies (from ThinkBooster README):
  - baseline (direct generation)
  - majority_voting
  - best_of_n
  - beam_search
  - extended_thinking
  - mur
  - deepconf_online
  - deepconf_offline
  - phi_decoding
  - uncertainty_cot

Scorers:
  - prm (process reward model)
  - entropy
  - probability
  - llm_judge
  - reprobe

Not all strategy/scorer combinations are valid. See ThinkBooster docs for
the compatibility matrix.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping

from daph_x.backends.openai_compat import (
    ChatMessage,
    ExternalGenerationRequest,
    ExternalGenerationResult,
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
# Strategy/scorer metadata
# ---------------------------------------------------------------------------

# Map ThinkBooster strategy names to DAPH-X StateMode
_STRATEGY_STATE_MODE: dict[str, StateMode] = {
    "baseline": StateMode.FRESH_SOLVE,
    "majority_voting": StateMode.FRESH_SOLVE,
    "best_of_n": StateMode.FRESH_SOLVE,
    "beam_search": StateMode.PARALLEL_SEARCH,
    "extended_thinking": StateMode.FRESH_SOLVE,
    "mur": StateMode.FRESH_SOLVE,
    "deepconf_online": StateMode.FRESH_SOLVE,
    "deepconf_offline": StateMode.CANDIDATE_RERANK,
    "phi_decoding": StateMode.FRESH_SOLVE,
    "uncertainty_cot": StateMode.FRESH_SOLVE,
}

# Map ThinkBooster strategy to tier (0=free, 1=cheap, 2=moderate, 3=expensive)
_STRATEGY_TIER: dict[str, int] = {
    "baseline": 0,
    "majority_voting": 1,
    "best_of_n": 1,
    "beam_search": 2,
    "extended_thinking": 1,
    "mur": 2,
    "deepconf_online": 2,
    "deepconf_offline": 1,
    "phi_decoding": 2,
    "uncertainty_cot": 1,
}

# White-box strategies (require logprob/internal model access)
_WHITEBOX_STRATEGIES = {"mur", "deepconf_online", "deepconf_offline", "phi_decoding", "uncertainty_cot"}


# ---------------------------------------------------------------------------
# Discrete operator profiles
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ThinkBoosterProfile:
    """Frozen profile for a ThinkBooster strategy/scorer combination."""
    profile_id: str
    strategy: str
    scorer: str
    strategy_params: dict[str, Any] = field(default_factory=dict)

    def url_path(self) -> str:
        """URL path component for this profile (relative to base_url).

        If base_url is http://localhost:8001/v1, the full endpoint becomes:
            http://localhost:8001/v1/{strategy}/{scorer}/chat/completions
        """
        return f"/{self.strategy}/{self.scorer}"


# Frozen profiles per R14_PROTOCOL.md §6
PROFILES: dict[str, ThinkBoosterProfile] = {
    "TB_BON_LOW": ThinkBoosterProfile(
        profile_id="TB_BON_LOW",
        strategy="best_of_n",
        scorer="prm",
        strategy_params={"tts_n_samples": 4, "temperature": 0.7},
    ),
    "TB_BON_MED": ThinkBoosterProfile(
        profile_id="TB_BON_MED",
        strategy="best_of_n",
        scorer="prm",
        strategy_params={"tts_n_samples": 8, "temperature": 0.7},
    ),
    "TB_DEEPCONF_LOW": ThinkBoosterProfile(
        profile_id="TB_DEEPCONF_LOW",
        strategy="deepconf_online",
        scorer="probability",
        strategy_params={"tts_confidence_threshold": 0.6},
    ),
    "TB_MUR_LOW": ThinkBoosterProfile(
        profile_id="TB_MUR_LOW",
        strategy="mur",
        scorer="entropy",
        strategy_params={"tts_uncertainty_threshold": 0.5},
    ),
    "TB_BEAM_LOW": ThinkBoosterProfile(
        profile_id="TB_BEAM_LOW",
        strategy="beam_search",
        scorer="prm",
        strategy_params={"tts_beam_size": 3, "tts_max_depth": 3},
    ),
    "TB_SC_LOW": ThinkBoosterProfile(
        profile_id="TB_SC_LOW",
        strategy="majority_voting",
        scorer="entropy",
        strategy_params={"tts_n_samples": 4, "temperature": 0.7},
    ),
}


# ---------------------------------------------------------------------------
# ThinkBooster adapter operator
# ---------------------------------------------------------------------------

class ThinkBoosterOperator(CognitiveOperator):
    """A single ThinkBooster strategy/scorer profile exposed as a DAPH-X operator."""

    def __init__(
        self,
        profile: ThinkBoosterProfile,
        backend: OpenAICompatibleBackend,
        operator_version: str = "1",
    ):
        self._profile = profile
        self._backend = backend
        self._operator_version = operator_version

        state_mode = _STRATEGY_STATE_MODE.get(profile.strategy, StateMode.FRESH_SOLVE)
        tier = _STRATEGY_TIER.get(profile.strategy, 2)
        requires_whitebox = profile.strategy in _WHITEBOX_STRATEGIES

        self._spec = OperatorSpec(
            operator_id=profile.profile_id,
            operator_version=operator_version,
            provider="thinkbooster",
            strategy=profile.strategy,
            strategy_version="1",
            state_mode=state_mode,
            tier=tier,
            backend_requirements=("openai_compatible",),
            supports_budget=True,
            supports_seed=True,
            requires_logprobs=requires_whitebox,
            requires_whitebox=requires_whitebox,
            max_context_tokens=8192,
            provenance={
                "thinkbooster_strategy": profile.strategy,
                "thinkbooster_scorer": profile.scorer,
                "thinkbooster_params": dict(profile.strategy_params),
                "backend_url": backend.base_url,
            },
        )

    @property
    def spec(self) -> OperatorSpec:
        return self._spec

    def estimate_cost(
        self,
        state: RuntimeState,
        budget: BudgetEnvelope | None = None,
    ) -> CostVector:
        """Rough cost estimate based on strategy parameters."""
        params = self._profile.strategy_params
        n_samples = params.get("tts_n_samples", 1)
        beam_size = params.get("tts_beam_size", 1)
        max_depth = params.get("tts_max_depth", 1)
        effective_calls = max(n_samples, beam_size * max_depth, 1)

        return CostVector(
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            gateway_calls=1,
            underlying_model_calls=effective_calls,
            wall_ms=None,
            gpu_ms=None,
            api_cost_usd=None,
        )

    def execute(
        self,
        state: RuntimeState,
        backend: Any | None = None,
        budget: BudgetEnvelope | None = None,
        replicate_id: int = 42,
    ) -> OperatorResult:
        """Execute the ThinkBooster strategy via the OpenAI-compatible gateway."""
        active_backend = backend if backend is not None else self._backend
        if not isinstance(active_backend, OpenAICompatibleBackend):
            return OperatorResult(
                terminal_answer="",
                status="FAILURE",
                error_code="BACKEND_TYPE_ERROR",
                error_message=f"Expected OpenAICompatibleBackend, got {type(active_backend).__name__}",
            )

        # Construct the ThinkBooster-specific base URL with strategy/scorer path.
        # The ThinkBooster gateway uses the strategy/scorer as part of the
        # OpenAI base_url, so chat completions go to:
        #   {root}/v1/{strategy}/{scorer}/chat/completions
        # We use base_url_override for thread-safe routing without mutating
        # the shared backend instance.
        tb_base_url = active_backend.base_url + self._profile.url_path()

        # Build the chat request from the task prompt
        messages = (
            ChatMessage(role="system", content="You are a helpful reasoning assistant. Provide a clear final answer."),
            ChatMessage(role="user", content=state.task_prompt),
        )

        # Merge strategy params into extra_params
        extra_params: dict[str, Any] = {}
        for k, v in self._profile.strategy_params.items():
            extra_params[k] = v

        # For STATE_CONDITIONED operators, include current answer as context
        if self._spec.state_mode == StateMode.STATE_CONDITIONED and state.current_answer:
            extra_params["current_answer"] = state.current_answer

        # For CANDIDATE_RERANK, include existing candidates
        if self._spec.state_mode == StateMode.CANDIDATE_RERANK and state.candidates:
            extra_params["candidates"] = [c.answer for c in state.candidates]

        max_tokens = 4096
        if budget is not None and budget.max_tokens is not None:
            max_tokens = min(max_tokens, budget.max_tokens)

        request = ExternalGenerationRequest(
            messages=messages,
            model=active_backend.model,
            temperature=self._profile.strategy_params.get("temperature", 0.0),
            max_tokens=max_tokens,
            seed=replicate_id,
            extra_params=extra_params,
        )

        # Use base_url_override for thread-safe routing. Do NOT mutate the
        # shared backend instance.
        result = active_backend.generate(request, base_url_override=tb_base_url)

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
                    "thinkbooster_strategy": self._profile.strategy,
                    "thinkbooster_scorer": self._profile.scorer,
                    "request_hash": result.request_hash,
                    **active_backend.service_identity.to_dict(),
                },
            )

        # Build candidate from the response
        # Use canonical answer extractor for terminal_answer, full text for trace
        terminal = extract_answer(result.text, state.answer_type)
        candidate = Candidate(
            candidate_id=f"tb_{self._profile.profile_id}_{replicate_id}",
            answer=terminal,
            reasoning_trace=result.text,
            temperature=self._profile.strategy_params.get("temperature", 0.0),
            seed=replicate_id,
            generation_index=0,
            metadata={
                "strategy": self._profile.strategy,
                "scorer": self._profile.scorer,
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
                "strategy": self._profile.strategy,
                "scorer": self._profile.scorer,
                "profile_id": self._profile.profile_id,
            },
            status="SUCCESS",
            cost=CostVector(
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                total_tokens=result.total_tokens,
                # gateway_calls: DAPH-X made 1 HTTP request to ThinkBooster
                gateway_calls=1,
                # underlying_model_calls: unknown until ThinkBooster reports it
                underlying_model_calls=None,
                wall_ms=result.latency_ms,
            ),
            provenance={
                "thinkbooster_strategy": self._profile.strategy,
                "thinkbooster_scorer": self._profile.scorer,
                "thinkbooster_params": dict(self._profile.strategy_params),
                "backend_url": active_backend.base_url,
                "endpoint_url": tb_base_url + "/chat/completions",
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
) -> ThinkBoosterOperator:
    """Create a ThinkBooster operator from a frozen profile ID."""
    if profile_id not in PROFILES:
        raise ValueError(f"Unknown ThinkBooster profile: {profile_id}. Available: {list(PROFILES.keys())}")
    return ThinkBoosterOperator(
        profile=PROFILES[profile_id],
        backend=backend,
        operator_version=operator_version,
    )
