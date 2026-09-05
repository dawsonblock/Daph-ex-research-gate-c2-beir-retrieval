"""Conformance tests for R14 external operator ABI.

Verifies G0 (Adapter Conformance) per R14_PROTOCOL.md §7:
  - OperatorResult schema validity
  - OperatorSpec completeness
  - CostVector None-vs-zero discipline
  - BudgetEnvelope enforcement (with None-safe budget checks)
  - Registry lifecycle
  - StateMode classification
  - White-box capability enforcement
  - OptiLLM llama-server compatibility restrictions
  - Canonical answer extraction
  - Immutable backend (with_base_url, no mutation)
"""
from __future__ import annotations

from daph_x.executive.budget import BudgetEnvelope
from daph_x.executive.registry import OperatorRegistry, OperatorStatus
from daph_x.operators.external.base import (
    CognitiveOperator,
    CostVector,
    OperatorResult,
    OperatorSpec,
    StateMode,
)
from daph_x.operators.external.thinkbooster import PROFILES as TB_PROFILES, ThinkBoosterOperator
from daph_x.operators.external.optillm import (
    PROFILES as OPT_PROFILES,
    OptiLLMOperator,
    is_slug_compatible_with_capabilities,
)
from daph_x.backends.openai_compat import OpenAICompatibleBackend, ServiceIdentity
from daph_x.evaluation.answer_extractor import extract_answer
from daph_x.operators.types import (
    Candidate,
    EvaluationLabels,
    RuntimeState,
    TrajectoryPoint,
)

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_state() -> RuntimeState:
    """Build a minimal RuntimeState for testing."""
    return RuntimeState(
        checkpoint_id="test_cp_001",
        task_id="test_task_001",
        task_prompt="What is 2 + 2?",
        answer_type="numeric",
        category="math",
        difficulty="easy",
        candidates=(
            Candidate(
                candidate_id="c0",
                answer="4",
                reasoning_trace="2 + 2 = 4",
                temperature=0.0,
                seed=42,
                generation_index=0,
                metadata={},
            ),
        ),
        trajectory=(
            TrajectoryPoint(k=1, top_answer="4", p_top1=0.9, p_top2=0.05, margin=0.85, entropy=0.1, n_unique=1),
        ),
        k=1,
        current_answer="4",
        observable_features={
            "p_top1": 0.9,
            "p_top2": 0.05,
            "margin": 0.85,
            "entropy": 0.1,
            "n_unique_answers": 1,
            "agreement_rate": 1.0,
            "uncertainty_current": 0.1,
            "uncertainty_delta": 0.0,
            "margin_delta": 0.0,
            "answer_changed": 0,
            "stable_prefix_count": 1,
        },
        state_hash="abc123",
    )


def _make_backend(capabilities: set[str] | None = None) -> OpenAICompatibleBackend:
    backend = OpenAICompatibleBackend(
        base_url="http://localhost:8001/v1",
        model="test-model",
        api_key="test-key",
        provider_name="test",
    )
    if capabilities:
        for cap in capabilities:
            backend = backend.with_capability(cap)
    return backend


# ---------------------------------------------------------------------------
# CostVector tests
# ---------------------------------------------------------------------------

class TestCostVector:
    def test_unmeasured_is_none_not_zero(self):
        cv = CostVector()
        assert cv.prompt_tokens is None
        assert cv.completion_tokens is None
        assert cv.total_tokens is None
        assert cv.gateway_calls is None
        assert cv.underlying_model_calls is None
        assert cv.wall_ms is None
        assert cv.gpu_ms is None

    def test_effective_total_tokens_with_total(self):
        cv = CostVector(total_tokens=500)
        assert cv.effective_total_tokens() == 500

    def test_effective_total_tokens_with_parts(self):
        cv = CostVector(prompt_tokens=200, completion_tokens=300)
        assert cv.effective_total_tokens() == 500

    def test_effective_total_tokens_empty_returns_none(self):
        """Unknown cost must be None, not 0."""
        cv = CostVector()
        assert cv.effective_total_tokens() is None

    def test_effective_total_tokens_partial_returns_none(self):
        """If only one of prompt/completion is known, still None."""
        cv = CostVector(prompt_tokens=200)
        assert cv.effective_total_tokens() is None

    def test_to_dict_preserves_none(self):
        cv = CostVector(prompt_tokens=100)
        d = cv.to_dict()
        assert d["prompt_tokens"] == 100
        assert d["completion_tokens"] is None
        assert d["effective_tokens"] is None

    def test_from_dict_roundtrip(self):
        cv = CostVector(
            prompt_tokens=100, completion_tokens=200, total_tokens=300,
            gateway_calls=1, underlying_model_calls=8,
        )
        d = cv.to_dict()
        cv2 = CostVector.from_dict(d)
        assert cv2.prompt_tokens == 100
        assert cv2.completion_tokens == 200
        assert cv2.total_tokens == 300
        assert cv2.gateway_calls == 1
        assert cv2.underlying_model_calls == 8

    def test_gateway_vs_underlying_calls_separate(self):
        cv = CostVector(gateway_calls=1, underlying_model_calls=8)
        assert cv.gateway_calls == 1
        assert cv.underlying_model_calls == 8


# ---------------------------------------------------------------------------
# OperatorSpec tests
# ---------------------------------------------------------------------------

class TestOperatorSpec:
    def test_spec_is_frozen(self):
        spec = OperatorSpec(
            operator_id="TEST",
            operator_version="1",
            provider="test",
            strategy="test_strategy",
            strategy_version="1",
            state_mode=StateMode.FRESH_SOLVE,
            tier=1,
        )
        with pytest.raises(Exception):
            spec.operator_id = "OTHER"  # type: ignore

    def test_to_dict_serializes_state_mode(self):
        spec = OperatorSpec(
            operator_id="TEST",
            operator_version="1",
            provider="test",
            strategy="test",
            strategy_version="1",
            state_mode=StateMode.PARALLEL_SEARCH,
            tier=2,
        )
        d = spec.to_dict()
        assert d["state_mode"] == "PARALLEL_SEARCH"
        assert d["tier"] == 2


# ---------------------------------------------------------------------------
# OperatorResult tests
# ---------------------------------------------------------------------------

class TestOperatorResult:
    def test_success_result(self):
        r = OperatorResult(terminal_answer="42")
        assert r.is_success
        assert r.status == "SUCCESS"

    def test_failure_result(self):
        r = OperatorResult(
            terminal_answer="",
            status="FAILURE",
            error_code="TIMEOUT",
            error_message="timed out",
        )
        assert not r.is_success
        assert r.error_code == "TIMEOUT"

    def test_to_dict_preserves_candidates(self):
        c = Candidate(
            candidate_id="c1", answer="42", reasoning_trace="...",
            temperature=0.0, seed=42, generation_index=0, metadata={},
        )
        r = OperatorResult(terminal_answer="42", candidates=(c,))
        d = r.to_dict()
        assert len(d["candidates"]) == 1
        assert d["candidates"][0]["answer"] == "42"


# ---------------------------------------------------------------------------
# BudgetEnvelope tests
# ---------------------------------------------------------------------------

class TestBudgetEnvelope:
    def test_no_budget_allows_everything(self):
        b = BudgetEnvelope()
        assert not b.is_exceeded_by(tokens=999999)

    def test_token_budget_exceeded(self):
        b = BudgetEnvelope(max_tokens=500)
        assert b.is_exceeded_by(tokens=501)
        assert not b.is_exceeded_by(tokens=500)

    def test_multi_dim_budget(self):
        b = BudgetEnvelope(max_tokens=500, max_calls=3, max_wall_ms=10000.0)
        assert b.is_exceeded_by(tokens=600)
        assert b.is_exceeded_by(calls=4)
        assert b.is_exceeded_by(wall_ms=15000.0)
        assert not b.is_exceeded_by(tokens=500, calls=3, wall_ms=9000.0)

    def test_from_dict_roundtrip(self):
        b = BudgetEnvelope(max_tokens=500, max_calls=3, priority=1)
        d = b.to_dict()
        b2 = BudgetEnvelope.from_dict(d)
        assert b2.max_tokens == 500
        assert b2.max_calls == 3
        assert b2.priority == 1


# ---------------------------------------------------------------------------
# Backend immutability tests
# ---------------------------------------------------------------------------

class TestBackendImmutability:
    def test_with_base_url_creates_new_instance(self):
        b1 = _make_backend()
        b2 = b1.with_base_url("http://localhost:8001/v1/beam_search/prm")
        assert b1 is not b2
        assert b1.base_url == "http://localhost:8001/v1"
        assert b2.base_url == "http://localhost:8001/v1/beam_search/prm"

    def test_with_capability_creates_new_instance(self):
        b1 = _make_backend()
        b2 = b1.with_capability("multi_sample")
        assert b1 is not b2
        assert "multi_sample" not in b1.capabilities
        assert "multi_sample" in b2.capabilities

    def test_chat_completions_url(self):
        b = OpenAICompatibleBackend(base_url="http://localhost:8000/v1", model="m")
        assert b.chat_completions_url == "http://localhost:8000/v1/chat/completions"

    def test_thinkbooster_url_construction(self):
        b = OpenAICompatibleBackend(base_url="http://localhost:8001/v1", model="m")
        tb_b = b.with_base_url("http://localhost:8001/v1/beam_search/prm")
        assert tb_b.chat_completions_url == "http://localhost:8001/v1/beam_search/prm/chat/completions"

    def test_service_identity_preserved(self):
        b1 = _make_backend()
        b2 = b1.with_base_url("http://other:8002/v1")
        assert b2.service_identity.provider_name == b1.service_identity.provider_name
        assert b2.service_identity.base_url == "http://other:8002/v1"


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

class _DummyOperator(CognitiveOperator):
    """Minimal operator for registry testing."""

    def __init__(self, op_id: str, tier: int = 1, requires_whitebox: bool = False):
        self._spec = OperatorSpec(
            operator_id=op_id,
            operator_version="1",
            provider="test",
            strategy="dummy",
            strategy_version="1",
            state_mode=StateMode.FRESH_SOLVE,
            tier=tier,
            requires_whitebox=requires_whitebox,
        )

    @property
    def spec(self) -> OperatorSpec:
        return self._spec

    def estimate_cost(self, state, budget=None):
        return CostVector(gateway_calls=1)

    def execute(self, state, backend=None, budget=None, replicate_id=42):
        return OperatorResult(terminal_answer="ok")


class TestOperatorRegistry:
    def test_register_and_get(self):
        reg = OperatorRegistry()
        op = _DummyOperator("DUMMY_1")
        entry = reg.register(op)
        assert entry.operator_id == "DUMMY_1"
        assert entry.status == OperatorStatus.EXPERIMENTAL
        assert reg.get("DUMMY_1") is entry

    def test_duplicate_register_fails(self):
        reg = OperatorRegistry()
        reg.register(_DummyOperator("DUP"))
        with pytest.raises(ValueError):
            reg.register(_DummyOperator("DUP"))

    def test_mark_routable(self):
        reg = OperatorRegistry()
        reg.register(_DummyOperator("OP1"))
        reg.pass_gate("OP1", "G0")
        reg.mark_routable("OP1", "G0")
        assert reg.get("OP1").is_routable
        assert "G0" in reg.get("OP1").admission_gates_passed

    def test_mark_retired(self):
        reg = OperatorRegistry()
        reg.register(_DummyOperator("OP1"))
        reg.mark_retired("OP1", "dominated by OP2")
        assert reg.get("OP1").status == OperatorStatus.RETIRED

    def test_admissible_filtering_by_budget(self):
        reg = OperatorRegistry()
        reg.register(_DummyOperator("CHEAP", tier=1))
        reg.register(_DummyOperator("EXPENSIVE", tier=3))
        reg.mark_routable("CHEAP")
        reg.mark_routable("EXPENSIVE")

        state = _make_state()
        # With unknown cost (None), budget check should pass (can't verify)
        budget = BudgetEnvelope(max_tokens=100)
        admissible = reg.admissible_operators(state, budget=budget)
        ids = [op.spec.operator_id for op in admissible]
        assert "CHEAP" in ids
        assert "EXPENSIVE" in ids

    def test_only_routable_filter(self):
        reg = OperatorRegistry()
        reg.register(_DummyOperator("ROUTED"))
        reg.register(_DummyOperator("UNROUTED"))
        reg.mark_routable("ROUTED")

        state = _make_state()
        admissible = reg.admissible_operators(state, only_routable=True)
        ids = [op.spec.operator_id for op in admissible]
        assert "ROUTED" in ids
        assert "UNROUTED" not in ids

    def test_whitebox_capability_enforcement(self):
        """Operator requiring whitebox should be inadmissible without it."""
        reg = OperatorRegistry()
        reg.register(_DummyOperator("WB_OP", requires_whitebox=True))
        reg.mark_routable("WB_OP")

        state = _make_state()
        # Without whitebox capability
        admissible = reg.admissible_operators(state, capabilities={"openai_compatible"})
        assert "WB_OP" not in [op.spec.operator_id for op in admissible]

        # With whitebox capability
        admissible = reg.admissible_operators(state, capabilities={"openai_compatible", "whitebox"})
        assert "WB_OP" in [op.spec.operator_id for op in admissible]

        # With service-level whitebox (e.g. thinkbooster.whitebox)
        admissible = reg.admissible_operators(state, capabilities={"openai_compatible", "thinkbooster.whitebox"})
        assert "WB_OP" in [op.spec.operator_id for op in admissible]


# ---------------------------------------------------------------------------
# ThinkBooster profile tests
# ---------------------------------------------------------------------------

class TestThinkBoosterProfiles:
    def test_all_profiles_have_valid_strategies(self):
        for pid, profile in TB_PROFILES.items():
            assert profile.strategy in {"baseline", "majority_voting", "best_of_n", "beam_search",
                                        "extended_thinking", "mur", "deepconf_online",
                                        "deepconf_offline", "phi_decoding", "uncertainty_cot"}, \
                f"Unknown strategy in {pid}: {profile.strategy}"

    def test_profile_url_path(self):
        profile = TB_PROFILES["TB_BON_LOW"]
        assert profile.url_path() == "/v1/best_of_n/prm"

    def test_operator_creation(self):
        backend = _make_backend()
        op = ThinkBoosterOperator(TB_PROFILES["TB_BON_LOW"], backend)
        assert op.spec.operator_id == "TB_BON_LOW"
        assert op.spec.provider == "thinkbooster"
        assert op.spec.state_mode == StateMode.FRESH_SOLVE
        assert op.spec.tier == 1

    def test_estimate_cost_returns_costvector(self):
        backend = _make_backend()
        op = ThinkBoosterOperator(TB_PROFILES["TB_BON_MED"], backend)
        state = _make_state()
        cost = op.estimate_cost(state)
        assert isinstance(cost, CostVector)
        assert cost.gateway_calls == 1
        assert cost.underlying_model_calls is not None
        assert cost.underlying_model_calls >= 1

    def test_whitebox_profiles_require_whitebox(self):
        """MUR and DeepConf should require whitebox capability."""
        backend = _make_backend()
        mur_op = ThinkBoosterOperator(TB_PROFILES["TB_MUR_LOW"], backend)
        assert mur_op.spec.requires_whitebox

        dc_op = ThinkBoosterOperator(TB_PROFILES["TB_DEEPCONF_LOW"], backend)
        assert dc_op.spec.requires_whitebox

    def test_blackbox_profiles_dont_require_whitebox(self):
        """Best-of-N and majority voting should not require whitebox."""
        backend = _make_backend()
        bon_op = ThinkBoosterOperator(TB_PROFILES["TB_BON_LOW"], backend)
        assert not bon_op.spec.requires_whitebox

        sc_op = ThinkBoosterOperator(TB_PROFILES["TB_SC_LOW"], backend)
        assert not sc_op.spec.requires_whitebox


# ---------------------------------------------------------------------------
# OptiLLM profile tests
# ---------------------------------------------------------------------------

class TestOptiLLMProfiles:
    def test_all_profiles_have_valid_slugs(self):
        for pid, profile in OPT_PROFILES.items():
            assert profile.slug in {"cot_reflection", "plansearch", "re2", "self_consistency",
                                    "z3", "bon", "moa", "mcts", "rstar", "rto", "leap",
                                    "executive"}, f"Unknown slug in {pid}: {profile.slug}"

    def test_operator_creation(self):
        backend = _make_backend()
        op = OptiLLMOperator(OPT_PROFILES["OPT_PLANSEARCH_LOW"], backend)
        assert op.spec.operator_id == "OPT_PLANSEARCH_LOW"
        assert op.spec.provider == "optillm"
        assert op.spec.state_mode == StateMode.PARALLEL_SEARCH

    def test_estimate_cost_returns_costvector(self):
        backend = _make_backend()
        op = OptiLLMOperator(OPT_PROFILES["OPT_MOA_LOW"], backend)
        state = _make_state()
        cost = op.estimate_cost(state)
        assert isinstance(cost, CostVector)
        assert cost.gateway_calls == 1
        assert cost.underlying_model_calls is not None


class TestOptiLLMCompatibility:
    def test_llamaccpp_compatible_slugs(self):
        caps = {"openai_compatible"}  # No multi_sample
        assert is_slug_compatible_with_capabilities("cot_reflection", caps)
        assert is_slug_compatible_with_capabilities("plansearch", caps)
        assert is_slug_compatible_with_capabilities("re2", caps)
        assert is_slug_compatible_with_capabilities("self_consistency", caps)
        assert is_slug_compatible_with_capabilities("z3", caps)

    def test_llamaccpp_incompatible_slugs(self):
        caps = {"openai_compatible"}  # No multi_sample
        assert not is_slug_compatible_with_capabilities("bon", caps)
        assert not is_slug_compatible_with_capabilities("moa", caps)

    def test_multi_sample_capability_allows_all(self):
        caps = {"openai_compatible", "multi_sample"}
        assert is_slug_compatible_with_capabilities("bon", caps)
        assert is_slug_compatible_with_capabilities("moa", caps)

    def test_bon_inadmissible_with_llamaccpp_caps(self):
        """OPT_BON_LOW should be inadmissible when only openai_compatible is available."""
        backend = _make_backend(capabilities={"openai_compatible"})
        op = OptiLLMOperator(OPT_PROFILES["OPT_BON_LOW"], backend)
        state = _make_state()
        assert not op.is_admissible(state, capabilities={"openai_compatible"})

    def test_bon_admissible_with_multi_sample(self):
        """OPT_BON_LOW should be admissible with multi_sample capability."""
        backend = _make_backend()
        op = OptiLLMOperator(OPT_PROFILES["OPT_BON_LOW"], backend)
        state = _make_state()
        assert op.is_admissible(state, capabilities={"openai_compatible", "multi_sample"})

    def test_plansearch_always_admissible(self):
        """OPT_PLANSEARCH_LOW should be admissible even without multi_sample."""
        backend = _make_backend()
        op = OptiLLMOperator(OPT_PROFILES["OPT_PLANSEARCH_LOW"], backend)
        state = _make_state()
        assert op.is_admissible(state, capabilities={"openai_compatible"})


# ---------------------------------------------------------------------------
# Answer extractor tests
# ---------------------------------------------------------------------------

class TestAnswerExtractor:
    def test_boxed_extraction(self):
        assert extract_answer("The answer is \\boxed{42}", "numeric") == "42"

    def test_final_answer_pattern(self):
        assert extract_answer("After reasoning... Final answer: 420", "numeric") == "420"

    def test_integer_extraction(self):
        assert extract_answer("The result is 17", "integer") == "17"

    def test_fraction_extraction(self):
        assert extract_answer("We get 5/14", "fraction") == "5/14"

    def test_yes_no_extraction(self):
        assert extract_answer("After analysis, yes.", "yes_no") == "yes"
        assert extract_answer("The answer is no.", "yes_no") == "no"

    def test_letter_extraction(self):
        assert extract_answer("The correct option is C", "letter") == "C"

    def test_string_extraction_quoted(self):
        assert extract_answer('The answer is "knight"', "string") == "knight"

    def test_default_fallback_last_number(self):
        assert extract_answer("blah blah 42 blah", "default") == "42"

    def test_empty_input(self):
        assert extract_answer("", "numeric") == ""

    def test_reasoning_trace_not_truncated(self):
        """The extractor should find the answer in a long reasoning trace."""
        text = "Let me think about this step by step. " * 20 + "Final answer: 1337"
        assert extract_answer(text, "numeric") == "1337"
