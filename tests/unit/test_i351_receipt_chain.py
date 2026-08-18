"""Tests for I3.5.1 receipt chain integrity."""
import pytest
from hrm_adaptive_memory.executive.i3_5_1.receipts import (
    ReceiptLedger, make_receipt, compute_receipt_sha256, ReceiptEntry,
)


def _make_test_receipt(
    run_id="run_test",
    prev_sha="",
    step_id=0,
    experiment_sha="abc123",
    condition_sha="def456",
):
    """Build a minimal receipt for testing."""
    return make_receipt(
        run_id=run_id,
        experiment_identity_sha256=experiment_sha,
        condition_identity_sha256=condition_sha,
        task_id="test_task",
        pair_or_block_id="block_001",
        trajectory_id="traj_001",
        step_id=step_id,
        attempt_index=0,
        input_packet={"schema": "TEST", "task": "test"},
        system_prompt="test prompt",
        generation_config={"temperature": 0.0},
        provider="test",
        requested_model="test-model",
        reported_model="test-model",
        system_fingerprint="fp_001",
        timestamp_start="2026-01-01T00:00:00Z",
        timestamp_end="2026-01-01T00:00:01Z",
        latency_ms=1000.0,
        http_status=200,
        result_class="OK",
        raw_output='{"action": "ANSWER", "reason_code": "test"}',
        parsed_output={"action": "ANSWER", "reason_code": "test"},
        decoder_status="VALID",
        previous_receipt_sha256=prev_sha,
    )


class TestReceiptChain:
    def test_empty_ledger_verifies(self):
        ledger = ReceiptLedger()
        assert ledger.verify_chain() is True
        assert ledger.receipt_count == 0

    def test_single_receipt_chain(self):
        ledger = ReceiptLedger(run_id="run_001")
        r = _make_test_receipt(run_id="run_001")
        ledger.add(r)
        assert ledger.verify_chain() is True
        assert ledger.receipt_count == 1
        assert ledger.receipt_chain_root == r.receipt_sha256

    def test_multi_receipt_chain(self):
        ledger = ReceiptLedger(run_id="run_002")
        prev = ""
        for i in range(5):
            r = _make_test_receipt(run_id="run_002", prev_sha=prev, step_id=i)
            ledger.add(r)
            prev = r.receipt_sha256
        assert ledger.verify_chain() is True
        assert ledger.receipt_count == 5

    def test_chain_break_detected(self):
        """Tampering with a receipt should break the chain."""
        ledger = ReceiptLedger(run_id="run_003")
        r1 = _make_test_receipt(run_id="run_003", prev_sha="")
        ledger.add(r1)
        # Create a receipt with wrong previous_sha
        r2 = _make_test_receipt(run_id="run_003", prev_sha="wrong_sha")
        with pytest.raises(ValueError, match="chain broken"):
            ledger.add(r2)

    def test_receipt_has_all_required_fields(self):
        r = _make_test_receipt()
        d = r.as_dict()
        required = [
            "schema", "run_id", "experiment_identity_sha256",
            "condition_identity_sha256", "task_id_hash",
            "pair_or_block_id", "trajectory_id", "step_id",
            "attempt_index", "input_packet_sha256",
            "system_prompt_sha256", "generation_config_sha256",
            "request_sha256", "provider", "requested_model",
            "reported_model", "system_fingerprint",
            "timestamp_start", "timestamp_end", "latency_ms",
            "http_status", "result_class",
            "raw_output_sha256", "parsed_output_sha256",
            "decoder_status", "previous_receipt_sha256",
            "receipt_sha256",
        ]
        for field in required:
            assert field in d, f"Missing required field: {field}"

    def test_receipt_sha256_computed(self):
        r = _make_test_receipt()
        assert r.receipt_sha256 != ""
        assert len(r.receipt_sha256) == 64  # SHA-256 hex

    def test_save_and_load(self, tmp_path):
        ledger = ReceiptLedger(run_id="run_004")
        for i in range(3):
            r = _make_test_receipt(
                run_id="run_004",
                prev_sha=ledger.receipt_chain_root,
                step_id=i,
            )
            ledger.add(r)
        path = tmp_path / "receipts.jsonl"
        ledger.save(path)
        loaded = ReceiptLedger.load(path)
        assert loaded.receipt_count == 3
        assert loaded.verify_chain() is True
        assert loaded.receipt_chain_root == ledger.receipt_chain_root

    def test_chain_hash_depends_on_previous(self):
        """Changing the previous hash changes the receipt hash."""
        r1 = _make_test_receipt(prev_sha="aaa")
        r2 = _make_test_receipt(prev_sha="bbb")
        assert r1.receipt_sha256 != r2.receipt_sha256
