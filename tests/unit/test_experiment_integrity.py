"""Tests for the shared fail-closed integrity package
(hrm_adaptive_memory/experiment_integrity/). Includes mutation-style tests:
take a known-good value/object and mutate ONE thing at a time, asserting the
expected rejection fires -- stronger evidence of fail-closed behavior than
happy-path tests alone.
"""
from __future__ import annotations

import subprocess

import pytest

from hrm_adaptive_memory.experiment_integrity.certified_memory import (
    MEMORY_V1_CONFIG_HASH, CertifiedMemoryDriftError,
    assert_certified_memory_v1_unchanged, current_certified_memory_v1_identity,
    pin_certified_memory_v1_boundary_policy)
from hrm_adaptive_memory.experiment_integrity.execution_identity import (
    ExecutionIdentity, resume_is_valid)
from hrm_adaptive_memory.experiment_integrity.executive_bootstrap import (
    grouped_lcb_executive_opportunity)
from hrm_adaptive_memory.experiment_integrity.metric_validation import (
    NOT_COMPUTABLE, MetricValidationError, require_finite_number,
    require_hash_format, require_nonempty_rate, require_nonneg_int,
    require_probability)
from hrm_adaptive_memory.experiment_integrity.result_schema import (
    FailureClass, GateResult, MechanismStatus, ScientificVerdict, SplitStatus)
from hrm_adaptive_memory.experiment_integrity.split_lineage import (
    PermittedUse, SplitLineageError, SplitRole, parse_split_manifest,
    require_permitted_use)
from hrm_adaptive_memory.experiment_integrity.subprocess_safe import (
    SubprocessTimeoutPolicyError, check_output_safe, run_safe)


class TestRequireFiniteNumber:
    def test_accepts_a_plain_float(self):
        assert require_finite_number(0.42, field="x") == 0.42

    def test_accepts_an_int(self):
        assert require_finite_number(3, field="x") == 3.0

    @pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), float("-inf"),
                                     True, False, "0.5", [], {}])
    def test_rejects_every_non_finite_or_wrong_type_value(self, bad):
        with pytest.raises(MetricValidationError):
            require_finite_number(bad, field="x")

    def test_error_message_names_the_field(self):
        with pytest.raises(MetricValidationError, match="mean_delta"):
            require_finite_number(float("nan"), field="mean_delta")


class TestRequireProbability:
    def test_accepts_bounds(self):
        assert require_probability(0.0, field="p") == 0.0
        assert require_probability(1.0, field="p") == 1.0

    def test_rejects_out_of_range(self):
        with pytest.raises(MetricValidationError):
            require_probability(1.5, field="p")
        with pytest.raises(MetricValidationError):
            require_probability(-0.5, field="p")

    def test_tolerates_tiny_float_noise(self):
        assert require_probability(1.0000000001, field="p") == 1.0


class TestRequireNonnegInt:
    def test_accepts_zero_and_positive(self):
        assert require_nonneg_int(0, field="n") == 0
        assert require_nonneg_int(7, field="n") == 7

    def test_rejects_negative(self):
        with pytest.raises(MetricValidationError):
            require_nonneg_int(-1, field="n")

    def test_rejects_non_integer_float(self):
        with pytest.raises(MetricValidationError):
            require_nonneg_int(2.5, field="n")

    def test_rejects_bool(self):
        with pytest.raises(MetricValidationError):
            require_nonneg_int(True, field="n")


class TestRequireHashFormat:
    def test_accepts_valid_hex(self):
        assert require_hash_format("a1b2c3", field="h") == "a1b2c3"

    def test_rejects_uppercase(self):
        with pytest.raises(MetricValidationError):
            require_hash_format("A1B2C3", field="h")

    def test_rejects_wrong_length(self):
        with pytest.raises(MetricValidationError):
            require_hash_format("a1b2", field="h", expected_length=16)

    def test_rejects_non_hex(self):
        with pytest.raises(MetricValidationError):
            require_hash_format("not-a-hash!", field="h")


class TestRequireNonemptyRate:
    def test_normal_rate(self):
        assert require_nonempty_rate(3, 10, field="r") == 0.3

    def test_empty_denominator_returns_sentinel_not_zero(self):
        result = require_nonempty_rate(0, 0, field="r")
        assert result == NOT_COMPUTABLE
        assert result != 0.0  # the exact bug this prevents: NOT_COMPUTABLE == 0 would be a lie

    def test_hits_exceeding_total_is_rejected(self):
        with pytest.raises(MetricValidationError):
            require_nonempty_rate(5, 3, field="r")


class TestExecutionIdentity:
    def _identity(self, **overrides):
        base = dict(task_id="t1", arm_id="K1", prompt_hash="a" * 16,
                   retrieval_config_hash="b" * 16, selector_config_hash="c" * 16,
                   graph_compressor_config_hash="d" * 16, model_revision="hrm-v1",
                   pipeline_version="g2v2", source_commit="e" * 40)
        base.update(overrides)
        return ExecutionIdentity(**base)

    def test_identical_identities_hash_identically(self):
        assert self._identity().canonical_sha256() == self._identity().canonical_sha256()

    @pytest.mark.parametrize("field,new_value", [
        ("task_id", "t2"), ("arm_id", "K0"), ("prompt_hash", "f" * 16),
        ("retrieval_config_hash", "f" * 16), ("selector_config_hash", "f" * 16),
        ("graph_compressor_config_hash", "f" * 16), ("model_revision", "hrm-v2"),
        ("pipeline_version", "g2v3"), ("source_commit", "f" * 40),
    ])
    def test_mutating_any_single_field_changes_the_hash(self, field, new_value):
        """The core resume-key guarantee: changing ANY one of these fields
        must produce a different identity. A resume key that only bound
        packet_hash would miss every one of these mutations."""
        original = self._identity()
        mutated = self._identity(**{field: new_value})
        assert original.canonical_sha256() != mutated.canonical_sha256()
        assert not resume_is_valid(original, mutated)

    def test_extra_config_hashes_participate_in_the_hash(self):
        base = self._identity()
        with_extra = self._identity(
            extra_config_hashes={"endpoint_recognizer_config_hash": "1" * 16})
        assert base.canonical_sha256() != with_extra.canonical_sha256()

    def test_extra_config_hash_key_order_does_not_matter(self):
        a = self._identity(extra_config_hashes={"x": "1", "y": "2"})
        b = self._identity(extra_config_hashes={"y": "2", "x": "1"})
        assert a.canonical_sha256() == b.canonical_sha256()

    def test_resume_valid_only_when_every_field_matches(self):
        a = self._identity()
        b = self._identity()
        assert resume_is_valid(a, b)

    def test_receipt_contains_the_hash_and_every_field(self):
        receipt = self._identity().as_receipt()
        assert "execution_identity_sha256" in receipt
        assert receipt["task_id"] == "t1"


class TestSubprocessSafe:
    def test_missing_timeout_and_exemption_raises_before_running(self):
        with pytest.raises(SubprocessTimeoutPolicyError):
            run_safe(["echo", "hi"])

    def test_exempt_reason_allows_no_timeout(self):
        result = run_safe(["echo", "hi"], exempt_reason="trivial local echo",
                          capture_output=True, text=True)
        assert result.stdout.strip() == "hi"

    def test_explicit_timeout_is_honored(self):
        result = run_safe(["echo", "hi"], timeout=5, capture_output=True, text=True)
        assert result.returncode == 0

    def test_timeout_actually_fires_on_a_slow_command(self):
        with pytest.raises(subprocess.TimeoutExpired):
            run_safe(["sleep", "5"], timeout=0.1)

    def test_check_output_safe_requires_timeout_too(self):
        with pytest.raises(SubprocessTimeoutPolicyError):
            check_output_safe(["echo", "hi"])
        assert check_output_safe(["echo", "hi"], timeout=5).strip() == b"hi"


class TestSplitLineage:
    def _good_manifest(self, **overrides):
        raw = {
            "split_id": "cal_700", "split_role": "calibration",
            "consumed_status": False,
            "permitted_uses": ["mechanism_selection", "threshold_selection"],
            "dataset_hash": "a" * 16,
        }
        raw.update(overrides)
        return raw

    def test_valid_manifest_parses(self):
        m = parse_split_manifest(self._good_manifest())
        assert m.split_role == SplitRole.CALIBRATION
        assert not m.consumed_status

    @pytest.mark.parametrize("missing_field", [
        "split_id", "split_role", "consumed_status", "permitted_uses", "dataset_hash"])
    def test_missing_any_required_field_is_rejected(self, missing_field):
        raw = self._good_manifest()
        del raw[missing_field]
        with pytest.raises(SplitLineageError):
            parse_split_manifest(raw)

    def test_unrecognized_split_role_is_rejected(self):
        with pytest.raises(SplitLineageError):
            parse_split_manifest(self._good_manifest(split_role="totally_fresh"))

    def test_consumed_status_mismatched_with_role_is_rejected(self):
        with pytest.raises(SplitLineageError):
            parse_split_manifest(self._good_manifest(
                split_role="qualification_consumed", consumed_status=False))
        with pytest.raises(SplitLineageError):
            parse_split_manifest(self._good_manifest(consumed_status=True))

    def test_consumed_split_forbids_mechanism_selection(self):
        """The exact rule this project has enforced by hand for qualification_1
        and confirmation_1, now mechanical."""
        raw = self._good_manifest(
            split_role="confirmation_consumed", consumed_status=True,
            permitted_uses=["mechanism_selection"])
        with pytest.raises(SplitLineageError):
            parse_split_manifest(raw)

    def test_consumed_split_permits_diagnosis_only(self):
        m = parse_split_manifest(self._good_manifest(
            split_role="confirmation_consumed", consumed_status=True,
            permitted_uses=["diagnosis_only"]))
        assert m.split_role == SplitRole.CONFIRMATION_CONSUMED

    def test_require_permitted_use_blocks_disallowed_use(self):
        m = parse_split_manifest(self._good_manifest(
            split_role="confirmation_consumed", consumed_status=True,
            permitted_uses=["diagnosis_only"]))
        with pytest.raises(SplitLineageError):
            require_permitted_use(m, PermittedUse.MECHANISM_SELECTION)
        require_permitted_use(m, PermittedUse.DIAGNOSIS_ONLY)  # does not raise

    def test_future_confirmation_split_cannot_be_used_for_mechanism_selection(self):
        """The scenario the whole module exists to prevent: a fresh
        confirmation split accidentally used to pick a mechanism/threshold."""
        raw = self._good_manifest(
            split_id="confirmation_2", split_role="future_confirmation",
            consumed_status=False, permitted_uses=["confirmation"])
        m = parse_split_manifest(raw)
        with pytest.raises(SplitLineageError):
            require_permitted_use(m, PermittedUse.MECHANISM_SELECTION)


class TestCertifiedMemoryV1:
    """The hard boundary an executive/controller experiment must assert
    against before invoking the confirmed memory operation as a black box.

    Each test pins boundary_policy explicitly at its own start rather than
    relying on module-load-time state or other tests' side effects -- this
    process-global is exactly the kind of shared mutable state that produces
    order-dependent test failures if left implicit. The autouse fixture below
    snapshots and restores it around every test in this class: an earlier
    version of these tests mutated it without restoring, which silently
    polluted test_g2_v4e_entity_boundary.py's default-policy assumptions
    whenever the two files ran in the same pytest session -- exactly the
    failure mode this fixture exists to prevent.
    """

    @pytest.fixture(autouse=True)
    def _restore_boundary_policy(self):
        from hrm_adaptive_memory.c4.bridge_extraction import (
            get_default_boundary_policy, set_default_boundary_policy)
        original = get_default_boundary_policy()
        yield
        set_default_boundary_policy(original)

    def test_current_code_state_matches_the_frozen_identity(self):
        """The stack as it stands today (post confirmation-2), WITH the
        deliberate pin applied, must match the hash frozen in
        configs/certified_memory_v1.json -- if this fails, the certificate is
        stale relative to the code, not the other way round."""
        pin_certified_memory_v1_boundary_policy()
        identity = assert_certified_memory_v1_unchanged()
        assert identity.canonical_sha256() == MEMORY_V1_CONFIG_HASH

    def test_identity_pins_the_grammar_v4_boundary_policy(self):
        pin_certified_memory_v1_boundary_policy()
        identity = current_certified_memory_v1_identity()
        assert identity.boundary_policy == "grammar_v4"

    def test_identity_pins_packet_budget_six(self):
        identity = current_certified_memory_v1_identity()
        assert identity.packet_budget == 6

    def test_drift_in_any_single_field_is_rejected(self):
        """Mutation-style: flip ONE field of the frozen identity at a time and
        confirm the resulting hash no longer matches -- proves the hash is
        actually sensitive to each component, not silently ignoring one.
        Pins boundary_policy explicitly first so `base` has a known starting
        value -- otherwise the boundary_policy->"legacy" mutation below could
        silently become a no-op if the ambient state already happened to be
        "legacy"."""
        from hrm_adaptive_memory.experiment_integrity.certified_memory import (
            CertifiedMemoryV1Identity)
        pin_certified_memory_v1_boundary_policy()
        base = current_certified_memory_v1_identity()
        base_hash = base.canonical_sha256()
        for field, bad_value in [
            ("retrieval_config_hash", "C3"),
            ("selector_config_hash", "s2_v1"),
            ("graph_compressor_config_hash", "0000000000000000"),
            ("model_revision", "sapientinc/HRM-Text-1B@deadbeef"),
            ("pipeline_version", "hrm_qualification_v2"),
            ("packet_budget", 8),
            ("boundary_policy", "legacy"),
        ]:
            kwargs = {**base.__dict__, field: bad_value}
            mutated = CertifiedMemoryV1Identity(**kwargs)
            assert mutated.canonical_sha256() != base_hash, f"hash insensitive to {field}"

    def test_boundary_policy_left_as_legacy_is_detected_as_drift(self):
        """The exact failure mode this module exists to catch: a caller that
        forgets to pin boundary_policy=grammar_v4 before invoking the memory
        operation must be rejected, not silently scored against the wrong
        (legacy, unconfirmed) entity-boundary treatment. Restoration to the
        true pre-test state is handled by the _restore_boundary_policy fixture,
        not by this test."""
        from hrm_adaptive_memory.c4.bridge_extraction import set_default_boundary_policy
        set_default_boundary_policy("legacy")
        with pytest.raises(CertifiedMemoryDriftError):
            assert_certified_memory_v1_unchanged()

    def test_config_hash_is_stable_across_repeated_calls(self):
        assert (current_certified_memory_v1_identity().canonical_sha256()
               == current_certified_memory_v1_identity().canonical_sha256())


class TestGroupedLcbExecutiveOpportunity:
    """LCB of ExecutiveOpportunity = mean(max(q0,q1)) - max(mean(q0),mean(q1)),
    which grouped_lcb's single-value-per-task interface cannot express because
    of the max() term -- these pin the generalization against hand-computable
    cases."""

    def test_no_heterogeneity_gives_zero_opportunity(self):
        """Q_E0 == Q_E1 for every task -> true opportunity is exactly 0, and
        every bootstrap replicate reproduces the same degenerate case."""
        triples = [("fam_a", 1.0, 1.0)] * 20 + [("fam_b", 0.0, 0.0)] * 20
        assert grouped_lcb_executive_opportunity(triples) == 0.0

    def test_perfect_50_50_heterogeneity_gives_large_positive_lcb(self):
        """Half the tasks strictly favor E0, half strictly favor E1 -- oracle
        wins every task (mean 1.0), each fixed policy wins only half (mean
        0.5) -> true opportunity = 0.5."""
        triples = ([("fam_a", 1.0, 0.0)] * 25 + [("fam_a", 0.0, 1.0)] * 25 +
                  [("fam_b", 1.0, 0.0)] * 25 + [("fam_b", 0.0, 1.0)] * 25)
        lcb = grouped_lcb_executive_opportunity(triples)
        assert lcb is not None and lcb > 0.3

    def test_single_group_lcb_equals_the_point_estimate(self):
        """With only one group, every bootstrap replicate resamples that same
        group every time -- no resampling diversity is possible, so the LCB
        must equal the point estimate exactly."""
        triples = [("only_fam", 1.0, 0.0)] * 10 + [("only_fam", 0.0, 1.0)] * 10
        q0 = [t[1] for t in triples]
        q1 = [t[2] for t in triples]
        point_estimate = (sum(max(a, b) for a, b in zip(q0, q1)) / len(triples)
                          - max(sum(q0) / len(q0), sum(q1) / len(q1)))
        assert grouped_lcb_executive_opportunity(triples) == round(point_estimate, 4)

    def test_empty_input_returns_none(self):
        assert grouped_lcb_executive_opportunity([]) is None

    def test_one_action_strictly_dominates_gives_zero_or_negative_opportunity(self):
        """E1 always at least as good as E0, everywhere -- oracle just mirrors
        E1, so max(mean(E0),mean(E1)) already equals U(E3): true opportunity
        is exactly 0, nothing for an executive to capture."""
        triples = [("fam_a", 0.0, 1.0)] * 30 + [("fam_b", 0.3, 0.8)] * 30
        lcb = grouped_lcb_executive_opportunity(triples)
        assert lcb is not None and lcb <= 0.0001


class TestGateResult:
    def test_valid_negative_result_like_confirmation_1(self):
        r = GateResult(run_valid=True, scientific_verdict=ScientificVerdict.NEGATIVE,
                       mechanism_status=MechanismStatus.NOT_PROMOTED,
                       failure_class=FailureClass.SELECTOR_LIMITED,
                       split_status=SplitStatus.CONSUMED)
        d = r.as_dict()
        assert d["run_valid"] is True
        assert d["scientific_verdict"] == "NEGATIVE"

    def test_valid_negative_result_like_g2_v1_outcome_c(self):
        r = GateResult(run_valid=True, scientific_verdict=ScientificVerdict.NEGATIVE,
                       mechanism_status=MechanismStatus.NOT_PROMOTED,
                       failure_class=FailureClass.CONSTRUCTION_DEFICIENT,
                       split_status=SplitStatus.NOT_YET_RUN)
        assert r.run_valid and r.mechanism_status == MechanismStatus.NOT_PROMOTED

    def test_invalid_run_cannot_be_promoted(self):
        with pytest.raises(ValueError):
            GateResult(run_valid=False, scientific_verdict=ScientificVerdict.POSITIVE,
                      mechanism_status=MechanismStatus.PROMOTED,
                      failure_class=FailureClass.NONE, split_status=SplitStatus.FRESH)

    def test_benchmark_has_no_action_heterogeneity_like_executive_opportunity_v1(self):
        """The Executive Opportunity Study's actual result: a valid run, a
        negative verdict, and a failure class that names the DATASET as the
        limitation rather than the mechanism -- no executive was tested, so
        it cannot be an executive failure."""
        r = GateResult(run_valid=True, scientific_verdict=ScientificVerdict.NEGATIVE,
                       mechanism_status=MechanismStatus.NOT_PROMOTED,
                       failure_class=FailureClass.BENCHMARK_HAS_NO_ACTION_HETEROGENEITY,
                       split_status=SplitStatus.CONSUMED)
        d = r.as_dict()
        assert d["failure_class"] == "BENCHMARK_HAS_NO_ACTION_HETEROGENEITY"
        assert d["scientific_verdict"] == "NEGATIVE"

    def test_positive_not_promoted_requires_a_reason(self):
        with pytest.raises(ValueError):
            GateResult(run_valid=True, scientific_verdict=ScientificVerdict.POSITIVE,
                      mechanism_status=MechanismStatus.NOT_PROMOTED,
                      failure_class=FailureClass.NONE, split_status=SplitStatus.FRESH)
        # pending-further-evidence is the correct way to express this instead
        GateResult(run_valid=True, scientific_verdict=ScientificVerdict.POSITIVE,
                  mechanism_status=MechanismStatus.PENDING_FURTHER_EVIDENCE,
                  failure_class=FailureClass.NONE, split_status=SplitStatus.FRESH)
