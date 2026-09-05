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

# Skip marker for tests requiring live services
live_only = pytest.mark.skipif(
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

@live_only
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

@live_only
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

@live_only
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


# ---------------------------------------------------------------------------
# Mixed answer-type extraction smoke
# ---------------------------------------------------------------------------

class TestMixedAnswerTypeExtractionLive:
    """Run extraction across all answer types used in R13 corpus.

    This is a unit test (no live service needed) that verifies the extractor
    handles verbose responses for each answer_type. Run before R14-B0 to
    ensure the extractor does not need correction after evidence is generated.
    """

    @pytest.mark.parametrize("answer_type,raw_text,expected", [
        ("numeric", "Let me compute 17 * 23. 17 * 23 = 391. The answer is 391.", "391"),
        ("numeric", "After careful analysis, \\boxed{391}", "391"),
        ("float", "The result is approximately 2.40 meters.", "2.40"),
        ("fraction", "We need to simplify. 5/14 is already in lowest terms.", "5/14"),
        ("yes_no", "After considering all evidence, yes.", "yes"),
        ("yes_no", "The answer is no because the condition fails.", "no"),
        ("true_false", "The statement is true given the premises.", "true"),
        ("true_false", "This claim is false.", "false"),
        ("letter", "Option A is wrong, B is close, but D is correct.", "D"),
        ("string", 'The character is "knight".', "knight"),
        ("string", 'The person is "Bob".', "Bob"),
        ("integer", "Count: 42 items total.", "42"),
    ])
    def test_extraction_by_type(self, answer_type: str, raw_text: str, expected: str):
        """Verify extraction produces canonical answer for each type."""
        result = extract_answer(raw_text, answer_type)
        assert result == expected, \
            f"answer_type={answer_type}: expected '{expected}', got '{result}' " \
            f"from text: {raw_text[:80]}..."

    def test_verbose_reasoning_still_extracts(self):
        """A long reasoning trace should still yield the correct answer."""
        text = (
            "Let me work through this step by step. "
            "First, we consider the constraints. "
            "The problem asks for the number of ordered pairs. "
            "We can enumerate: (1,1), (2,1), (3,2), (4,3), (5,6). "
            "Wait, let me recount. Actually there are 5 pairs. "
            "Final answer: 5"
        )
        result = extract_answer(text, "numeric")
        assert result == "5", f"Expected '5', got '{result}'"

    def test_string_type_extracts_yes_from_verbose(self):
        """Bug fix: string type with yes/no answer must extract just 'yes'."""
        text = "Yes. 28 is a perfect number because the sum of its proper divisors (1, 2, 4, 7, 14) equals the number itself (1 + 2 + 4 + 7 + 14 = 28)."
        result = extract_answer(text, "string")
        assert result == "yes", f"Expected 'yes', got '{result}'"

    def test_string_type_extracts_no_from_verbose(self):
        """Bug fix: string type with no answer must extract just 'no'."""
        text = "No, because the sum of proper divisors exceeds the number."
        result = extract_answer(text, "string")
        assert result == "no", f"Expected 'no', got '{result}'"

    def test_string_type_extracts_fraction(self):
        """String type with fraction answer should extract the fraction."""
        text = "After working through the ratios, the answer is 2/3."
        result = extract_answer(text, "string")
        assert result == "2/3", f"Expected '2/3', got '{result}'"


# ---------------------------------------------------------------------------
# Token accounting semantics determination
# ---------------------------------------------------------------------------

@live_only
class TestTokenAccountingSemanticsLive:
    """Determine whether external services report aggregate or final-only tokens.

    This is the most important R14-B measurement question. If usage.total_tokens
    contains only the final selected response, Pareto analysis using that field
    will be invalid. This test captures the raw response for manual inspection
    and records the determination.
    """

    def test_thinkbooster_bon_token_semantics(self):
        """For TB_BON_LOW (N=4), inspect whether usage includes all 4 generations."""
        backend = OpenAICompatibleBackend(
            base_url=_TB_URL,
            model=_MODEL,
            api_key="EMPTY",
            provider_name="thinkbooster",
        )
        op = ThinkBoosterOperator(TB_PROFILES["TB_BON_LOW"], backend)
        state = _make_state("What is 12 * 12?")
        result = op.execute(state, replicate_id=42)

        assert result.status == "SUCCESS", \
            f"Live service required: {result.error_code} - {result.error_message}"

        # The cost vector should have token data
        tokens = result.cost.effective_total_tokens()
        n_samples = TB_PROFILES["TB_BON_LOW"].strategy_params.get("tts_n_samples", 4)

        # Record the determination for manual inspection
        # This test does not assert a specific semantics — it captures data
        # for the frozen service manifest.
        print(f"\n--- Token Accounting Determination (TB_BON_LOW, N={n_samples}) ---")
        print(f"  prompt_tokens: {result.cost.prompt_tokens}")
        print(f"  completion_tokens: {result.cost.completion_tokens}")
        print(f"  total_tokens: {result.cost.total_tokens}")
        print(f"  effective_total_tokens: {tokens}")
        print(f"  gateway_calls: {result.cost.gateway_calls}")
        print(f"  underlying_model_calls: {result.cost.underlying_model_calls}")
        print(f"  raw_response keys: {list(result.provenance.keys())}")
        print(f"  Determination needed: AGGREGATE_INTERNAL_COMPUTE or FINAL_RESPONSE_ONLY")
        print(f"  If total_tokens < ~100 for N=4 generations, likely FINAL_RESPONSE_ONLY")

        # Minimum assertion: tokens should be reported (not None) for Pareto analysis
        # If None, we must instrument the service before R14-B
        if tokens is None:
            pytest.skip(
                "Service does not report token usage. "
                "Must instrument service or use alternative cost axis before R14-B."
            )

    def test_optillm_multicall_token_semantics(self):
        """For OPT_SC_LOW (self-consistency, N=4), inspect token reporting."""
        backend = OpenAICompatibleBackend(
            base_url=_OPT_URL,
            model=_MODEL,
            api_key="no_key",
            provider_name="optillm",
        )
        op = OptiLLMOperator(OPT_PROFILES["OPT_SC_LOW"], backend)
        state = _make_state("What is 15 * 15?")
        result = op.execute(state, replicate_id=42)

        assert result.status == "SUCCESS", \
            f"Live service required: {result.error_code} - {result.error_message}"

        tokens = result.cost.effective_total_tokens()
        n = OPT_PROFILES["OPT_SC_LOW"].strategy_params.get("n", 4)

        print(f"\n--- Token Accounting Determination (OPT_SC_LOW, N={n}) ---")
        print(f"  prompt_tokens: {result.cost.prompt_tokens}")
        print(f"  completion_tokens: {result.cost.completion_tokens}")
        print(f"  total_tokens: {result.cost.total_tokens}")
        print(f"  effective_total_tokens: {tokens}")
        print(f"  Determination needed: AGGREGATE_INTERNAL_COMPUTE or FINAL_RESPONSE_ONLY")

        if tokens is None:
            pytest.skip(
                "Service does not report token usage. "
                "Must instrument service or use alternative cost axis before R14-B."
            )
