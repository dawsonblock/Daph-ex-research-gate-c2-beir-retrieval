"""Tests for I3.5.1 report invariants — impossible count rejection."""
import pytest
from hrm_adaptive_memory.executive.i3_5_1.report import (
    verify_count_invariants, verify_sign_invariants,
    verify_subgroup_totals, verify_receipt_consistency,
    ReportInvariantError,
)


class TestCountInvariants:
    def test_valid_counts_pass(self):
        verify_count_invariants(
            n_tasks=300,
            both_success=100,
            both_fail=100,
            gov_blind_only=50,
            gov_aware_only=50,
        )  # Should not raise

    def test_invalid_counts_abort(self):
        with pytest.raises(ReportInvariantError, match="Count invariant failed"):
            verify_count_invariants(
                n_tasks=300,
                both_success=100,
                both_fail=100,
                gov_blind_only=50,
                gov_aware_only=51,  # 100+100+50+51=301 != 300
            )

    def test_zero_counts_pass(self):
        verify_count_invariants(
            n_tasks=0,
            both_success=0,
            both_fail=0,
            gov_blind_only=0,
            gov_aware_only=0,
        )


class TestSignInvariants:
    def test_valid_signs_pass(self):
        verify_sign_invariants(
            n_tasks=300,
            positive=150,
            negative=100,
            zero=50,
        )

    def test_invalid_signs_abort(self):
        with pytest.raises(ReportInvariantError, match="Sign invariant failed"):
            verify_sign_invariants(
                n_tasks=300,
                positive=150,
                negative=100,
                zero=51,  # 150+100+51=301 != 300
            )


class TestSubgroupTotals:
    def test_valid_subgroups_pass(self):
        verify_subgroup_totals(
            n_tasks=300,
            depth_counts={"DEPTH_1": 100, "DEPTH_4_PLUS": 200},
        )

    def test_invalid_subgroups_abort(self):
        with pytest.raises(ReportInvariantError, match="subgroup invariant failed"):
            verify_subgroup_totals(
                n_tasks=300,
                depth_counts={"DEPTH_1": 100, "DEPTH_4_PLUS": 201},
            )


class TestReceiptConsistency:
    def test_matching_counts_pass(self):
        verify_receipt_consistency(
            reported_model_calls=1000,
            receipt_model_calls=1000,
            reported_backend_errors=0,
            receipt_backend_errors=0,
            reported_decoder_failures=5,
            receipt_decoder_failures=5,
        )

    def test_mismatched_model_calls_abort(self):
        with pytest.raises(ReportInvariantError, match="Model call count mismatch"):
            verify_receipt_consistency(
                reported_model_calls=1000,
                receipt_model_calls=999,
                reported_backend_errors=0,
                receipt_backend_errors=0,
                reported_decoder_failures=0,
                receipt_decoder_failures=0,
            )

    def test_mismatched_backend_errors_abort(self):
        with pytest.raises(ReportInvariantError, match="Backend error count mismatch"):
            verify_receipt_consistency(
                reported_model_calls=100,
                receipt_model_calls=100,
                reported_backend_errors=0,
                receipt_backend_errors=5,
                reported_decoder_failures=0,
                receipt_decoder_failures=0,
            )
