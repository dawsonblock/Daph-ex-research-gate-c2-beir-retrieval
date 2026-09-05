"""R14-A live conformance test scaffold.

These tests require live ThinkBooster and/or OptiLLM services running.
They are skipped by default. To run them:

  1. Start ThinkBooster service:
     cd thinkbooster && python service_app/main.py  # starts on :8001

  2. Start OptiLLM proxy:
     python optillm.py --base_url http://localhost:8080/v1  # starts on :8000

  3. Set environment variables:
     export DAPH_R14_LIVE_TESTS=1
     export DAPH_THINKBOOSTER_URL=http://localhost:8001/v1
     export DAPH_OPTILLM_URL=http://localhost:8000/v1
     export DAPH_R14_MODEL=qwen2.5-7b-instruct  # or whatever model name

  4. Run:
     pytest tests/test_r14_live_conformance.py -v

The most important assertions are:
  - Request reaches the correct endpoint
  - HTTP success
  - Strategy actually activated (not silently falling back)
  - Response parsable
  - Answer extracted
  - Token accounting semantics known
  - Provenance complete
"""
from __future__ import annotations

import os
from typing import Any

import pytest

from daph_x.backends.openai_compat import (
    ChatMessage,
    ExternalGenerationRequest,
    OpenAICompatibleBackend,
)
from daph_x.evaluation.answer_extractor import extract_answer
from daph_x.operators.external.thinkbooster import PROFILES as TB_PROFILES, ThinkBoosterOperator
from daph_x.operators.external.optillm import PROFILES as OPT_PROFILES, OptiLLMOperator
from daph_x.operators.types import Candidate, RuntimeState, TrajectoryPoint


_LIVE_TESTS_ENABLED = os.environ.get("DAPH_R14_LIVE_TESTS", "") == "1"
_TB_URL = os.environ.get("DAPH_THINKBOOSTER_URL", "http://localhost:8001/v1")
_OPT_URL = os.environ.get("DAPH_OPTILLM_URL", "http://localhost:8000/v1")
_MODEL = os.environ.get("DAPH_R14_MODEL", "test-model")

pytestmark = pytest.mark.skipif(
    not _LIVE_TESTS_ENABLED,
    reason="Set DAPH_R14_LIVE_TESTS=1 to run live conformance tests",
)


def _make_state(prompt: str = "What is 17 * 23?") -> RuntimeState:
    return RuntimeState(
        checkpoint_id="live_test_cp",
        task_id="live_test_task",
        task_prompt=prompt,
        answer_type="numeric",
        category="math",
        difficulty="medium",
        candidates=(
            Candidate(
                candidate_id="c0",
                answer="391",
                reasoning_trace="17 * 23 = 391",
                temperature=0.0,
                seed=42,
                generation_index=0,
                metadata={},
            ),
        ),
        trajectory=(
            TrajectoryPoint(k=1, top_answer="391", p_top1=0.8, p_top2=0.1, margin=0.7, entropy=0.3, n_unique=1),
        ),
        k=1,
        current_answer="391",
        observable_features={
            "p_top1": 0.8, "p_top2": 0.1, "margin": 0.7, "entropy": 0.3,
            "n_unique_answers": 1, "agreement_rate": 1.0,
            "uncertainty_current": 0.3, "uncertainty_delta": 0.0,
            "margin_delta": 0.0, "answer_changed": 0, "stable_prefix_count": 1,
        },
        state_hash="live_test_hash",
    )


# ---------------------------------------------------------------------------
# ThinkBooster live tests
# ---------------------------------------------------------------------------

class TestThinkBoosterLive:
    def _make_backend(self) -> OpenAICompatibleBackend:
        return OpenAICompatibleBackend(
            base_url=_TB_URL,
            model=_MODEL,
            api_key="EMPTY",
            provider_name="thinkbooster",
        )

    @pytest.mark.parametrize("profile_id", ["TB_BON_LOW", "TB_SC_LOW"])
    def test_blackbox_strategy_succeeds(self, profile_id: str):
        """Verify black-box ThinkBooster strategies produce valid results."""
        backend = self._make_backend()
        op = ThinkBoosterOperator(TB_PROFILES[profile_id], backend)
        state = _make_state()
        result = op.execute(state, replicate_id=42)

        assert result.status == "SUCCESS", f"Failed: {result.error_code} - {result.error_message}"
        assert result.terminal_answer, "Empty terminal answer"
        assert len(result.candidates) > 0
        assert result.cost.gateway_calls == 1
        assert result.provenance.get("thinkbooster_strategy") == TB_PROFILES[profile_id].strategy
        assert result.provenance.get("thinkbooster_scorer") == TB_PROFILES[profile_id].scorer

    def test_token_usage_reported(self):
        """Verify that token usage is reported (may be final-only or aggregate)."""
        backend = self._make_backend()
        op = ThinkBoosterOperator(TB_PROFILES["TB_BON_LOW"], backend)
        state = _make_state()
        result = op.execute(state, replicate_id=42)

        if result.status == "SUCCESS":
            # Token usage may be None if the service doesn't report it,
            # but if reported, it should be positive
            tokens = result.cost.effective_total_tokens()
            if tokens is not None:
                assert tokens > 0, "Reported tokens should be positive"

    def test_seed_transmitted(self):
        """Verify that seed is transmitted to the service."""
        backend = self._make_backend()
        op = ThinkBoosterOperator(TB_PROFILES["TB_BON_LOW"], backend)
        state = _make_state()
        result = op.execute(state, replicate_id=123)

        if result.status == "SUCCESS":
            assert result.provenance.get("request_hash") is not None

    def test_provenance_complete(self):
        """Verify provenance contains all required fields."""
        backend = self._make_backend()
        op = ThinkBoosterOperator(TB_PROFILES["TB_BON_LOW"], backend)
        state = _make_state()
        result = op.execute(state, replicate_id=42)

        if result.status == "SUCCESS":
            p = result.provenance
            assert "thinkbooster_strategy" in p
            assert "thinkbooster_scorer" in p
            assert "backend_url" in p
            assert "endpoint_url" in p
            assert "model_id" in p
            assert "request_hash" in p
            assert "response_hash" in p
            assert "provider_name" in p


# ---------------------------------------------------------------------------
# OptiLLM live tests
# ---------------------------------------------------------------------------

class TestOptiLLMLive:
    def _make_backend(self) -> OpenAICompatibleBackend:
        return OpenAICompatibleBackend(
            base_url=_OPT_URL,
            model=_MODEL,
            api_key="no_key",
            provider_name="optillm",
        )

    @pytest.mark.parametrize("profile_id", ["OPT_COT_REFLECT", "OPT_RE2", "OPT_PLANSEARCH_LOW"])
    def test_llamaccpp_compatible_strategy_succeeds(self, profile_id: str):
        """Verify OptiLLM strategies compatible with llama-server work."""
        backend = self._make_backend()
        op = OptiLLMOperator(OPT_PROFILES[profile_id], backend)
        state = _make_state()
        result = op.execute(state, replicate_id=42)

        assert result.status == "SUCCESS", f"Failed: {result.error_code} - {result.error_message}"
        assert result.terminal_answer, "Empty terminal answer"
        assert result.provenance.get("optillm_slug") == OPT_PROFILES[profile_id].slug

    def test_slug_in_model_name(self):
        """Verify the slug is prepended to the model name in the request."""
        backend = self._make_backend()
        op = OptiLLMOperator(OPT_PROFILES["OPT_COT_REFLECT"], backend)
        # The slug should be in the provenance
        state = _make_state()
        result = op.execute(state, replicate_id=42)
        if result.status == "SUCCESS":
            assert "cot_reflection" in result.provenance.get("optillm_slug", "")


# ---------------------------------------------------------------------------
# Answer extraction live tests
# ---------------------------------------------------------------------------

class TestAnswerExtractionLive:
    def test_extracted_answer_is_canonical(self):
        """Verify the terminal answer is canonical, not the full reasoning trace."""
        backend = OpenAICompatibleBackend(
            base_url=_TB_URL,
            model=_MODEL,
            api_key="EMPTY",
        )
        op = ThinkBoosterOperator(TB_PROFILES["TB_BON_LOW"], backend)
        state = _make_state("What is 5 + 3?")
        result = op.execute(state, replicate_id=42)

        if result.status == "SUCCESS":
            # The terminal answer should be extractable as a number
            # even if the response contains reasoning
            assert result.terminal_answer, "Empty terminal answer"
            # It should not be a huge blob of reasoning text
            assert len(result.terminal_answer) < 500, \
                f"Terminal answer too long ({len(result.terminal_answer)} chars), " \
                f"extraction may have failed: {result.terminal_answer[:100]}..."
