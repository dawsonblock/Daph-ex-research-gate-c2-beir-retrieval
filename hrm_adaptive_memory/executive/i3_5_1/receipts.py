"""Append-only event ledger receipts for I3.5.1.

Each receipt is a node in a tamper-evident hash chain:
  R_i = SHA256(R_{i-1} || canonical(R_i))

Receipts are never overwritten. Each run gets its own:
  run_<uuid>/receipts.jsonl

A run is immutable.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RECEIPT_SCHEMA = "DAPH_V2B_I3_5_1_RECEIPT_V1"
RECEIPT_VERSION = 1


@dataclass(frozen=True)
class ReceiptEntry:
    """One API attempt receipt — a node in the hash chain."""
    schema: str
    schema_version: int
    run_id: str
    experiment_identity_sha256: str
    condition_identity_sha256: str
    task_id_hash: str
    pair_or_block_id: str
    trajectory_id: str
    step_id: int
    attempt_index: int
    input_packet_sha256: str
    system_prompt_sha256: str
    generation_config_sha256: str
    request_sha256: str
    provider: str
    requested_model: str
    reported_model: str | None
    system_fingerprint: str | None
    timestamp_start: str
    timestamp_end: str
    latency_ms: float
    http_status: int
    result_class: str  # OK, BACKEND_ERROR, DECODER_FAILURE
    raw_output_sha256: str | None
    parsed_output_sha256: str | None
    decoder_status: str  # VALID, INVALID, NOT_RUN
    previous_receipt_sha256: str  # "" for genesis
    receipt_sha256: str  # computed hash

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "experiment_identity_sha256": self.experiment_identity_sha256,
            "condition_identity_sha256": self.condition_identity_sha256,
            "task_id_hash": self.task_id_hash,
            "pair_or_block_id": self.pair_or_block_id,
            "trajectory_id": self.trajectory_id,
            "step_id": self.step_id,
            "attempt_index": self.attempt_index,
            "input_packet_sha256": self.input_packet_sha256,
            "system_prompt_sha256": self.system_prompt_sha256,
            "generation_config_sha256": self.generation_config_sha256,
            "request_sha256": self.request_sha256,
            "provider": self.provider,
            "requested_model": self.requested_model,
            "reported_model": self.reported_model,
            "system_fingerprint": self.system_fingerprint,
            "timestamp_start": self.timestamp_start,
            "timestamp_end": self.timestamp_end,
            "latency_ms": self.latency_ms,
            "http_status": self.http_status,
            "result_class": self.result_class,
            "raw_output_sha256": self.raw_output_sha256,
            "parsed_output_sha256": self.parsed_output_sha256,
            "decoder_status": self.decoder_status,
            "previous_receipt_sha256": self.previous_receipt_sha256,
            "receipt_sha256": self.receipt_sha256,
        }


def _canonical_json(obj: dict[str, Any]) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def compute_receipt_sha256(
    receipt_fields: dict[str, Any],
    previous_receipt_sha256: str,
) -> str:
    """Compute the tamper-evident chain hash.

    R_i = SHA256(R_{i-1} || canonical(R_i_fields))
    """
    # Remove receipt_sha256 and previous_receipt_sha256 from fields
    # for the canonical computation
    fields = {k: v for k, v in receipt_fields.items()
              if k not in ("receipt_sha256", "previous_receipt_sha256")}
    canonical = _canonical_json(fields)
    chain_input = previous_receipt_sha256 + canonical
    return _sha256_str(chain_input)


def make_receipt(
    *,
    run_id: str,
    experiment_identity_sha256: str,
    condition_identity_sha256: str,
    task_id: str,
    pair_or_block_id: str,
    trajectory_id: str,
    step_id: int,
    attempt_index: int,
    input_packet: dict[str, Any],
    system_prompt: str,
    generation_config: dict[str, Any],
    provider: str,
    requested_model: str,
    reported_model: str | None,
    system_fingerprint: str | None,
    timestamp_start: str,
    timestamp_end: str,
    latency_ms: float,
    http_status: int,
    result_class: str,
    raw_output: str | None,
    parsed_output: dict[str, Any] | None,
    decoder_status: str,
    previous_receipt_sha256: str,
) -> ReceiptEntry:
    """Build a receipt entry with computed chain hash."""
    fields = {
        "schema": RECEIPT_SCHEMA,
        "schema_version": RECEIPT_VERSION,
        "run_id": run_id,
        "experiment_identity_sha256": experiment_identity_sha256,
        "condition_identity_sha256": condition_identity_sha256,
        "task_id_hash": _sha256_str(task_id),
        "pair_or_block_id": pair_or_block_id,
        "trajectory_id": trajectory_id,
        "step_id": step_id,
        "attempt_index": attempt_index,
        "input_packet_sha256": _sha256_str(_canonical_json(input_packet)),
        "system_prompt_sha256": _sha256_str(system_prompt),
        "generation_config_sha256": _sha256_str(_canonical_json(generation_config)),
        "request_sha256": _sha256_str(
            _canonical_json(input_packet) + system_prompt
        ),
        "provider": provider,
        "requested_model": requested_model,
        "reported_model": reported_model,
        "system_fingerprint": system_fingerprint,
        "timestamp_start": timestamp_start,
        "timestamp_end": timestamp_end,
        "latency_ms": latency_ms,
        "http_status": http_status,
        "result_class": result_class,
        "raw_output_sha256": _sha256_str(raw_output) if raw_output else None,
        "parsed_output_sha256": (
            _sha256_str(_canonical_json(parsed_output)) if parsed_output else None
        ),
        "decoder_status": decoder_status,
    }
    receipt_sha = compute_receipt_sha256(fields, previous_receipt_sha256)
    return ReceiptEntry(
        **fields,
        previous_receipt_sha256=previous_receipt_sha256,
        receipt_sha256=receipt_sha,
    )


class ReceiptLedger:
    """Append-only receipt ledger with hash chain verification."""

    def __init__(
        self,
        run_id: str | None = None,
        experiment_identity_sha256: str = "",
    ):
        self.run_id = run_id or f"run_{uuid.uuid4()}"
        self.experiment_identity_sha256 = experiment_identity_sha256
        self._receipts: list[ReceiptEntry] = []
        self._last_sha = ""

    @property
    def receipt_chain_root(self) -> str:
        """SHA-256 of the last receipt in the chain (empty if no receipts)."""
        return self._last_sha

    @property
    def receipt_count(self) -> int:
        return len(self._receipts)

    @property
    def receipts(self) -> tuple[ReceiptEntry, ...]:
        return tuple(self._receipts)

    def add(self, receipt: ReceiptEntry) -> None:
        """Append a receipt to the ledger."""
        if receipt.previous_receipt_sha256 != self._last_sha:
            raise ValueError(
                "Receipt chain broken: previous_receipt_sha256 mismatch")
        self._receipts.append(receipt)
        self._last_sha = receipt.receipt_sha256

    def verify_chain(self) -> bool:
        """Verify the entire hash chain is intact."""
        prev = ""
        for r in self._receipts:
            if r.previous_receipt_sha256 != prev:
                return False
            expected = compute_receipt_sha256(r.as_dict(), prev)
            if r.receipt_sha256 != expected:
                return False
            prev = r.receipt_sha256
        return True

    def save(self, path: str | Path) -> str:
        """Save receipts to an append-only JSONL file. Return file SHA-256."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for r in self._receipts:
                f.write(json.dumps(r.as_dict(), sort_keys=True) + "\n")
        return _sha256_bytes(path.read_bytes())

    @classmethod
    def build_chain_from_receipts(
        cls,
        receipts: list[dict[str, Any]],
        run_id: str,
    ) -> "ReceiptLedger":
        """Rebuild a hash chain from standalone receipts (from parallel runs).

        Receipts are re-chained in the given order. Each receipt's
        previous_receipt_sha256 and receipt_sha256 are recomputed.
        """
        ledger = cls(run_id=run_id)
        for r_dict in receipts:
            # Update run_id to the unified run_id before computing hash
            r_dict = dict(r_dict, run_id=run_id)
            # Recompute the receipt hash with the current chain root
            new_sha = compute_receipt_sha256(r_dict, ledger._last_sha)
            new_receipt = ReceiptEntry(
                schema=r_dict["schema"],
                schema_version=r_dict["schema_version"],
                run_id=run_id,
                experiment_identity_sha256=r_dict["experiment_identity_sha256"],
                condition_identity_sha256=r_dict["condition_identity_sha256"],
                task_id_hash=r_dict["task_id_hash"],
                pair_or_block_id=r_dict["pair_or_block_id"],
                trajectory_id=r_dict["trajectory_id"],
                step_id=r_dict["step_id"],
                attempt_index=r_dict["attempt_index"],
                input_packet_sha256=r_dict["input_packet_sha256"],
                system_prompt_sha256=r_dict["system_prompt_sha256"],
                generation_config_sha256=r_dict["generation_config_sha256"],
                request_sha256=r_dict["request_sha256"],
                provider=r_dict["provider"],
                requested_model=r_dict["requested_model"],
                reported_model=r_dict["reported_model"],
                system_fingerprint=r_dict["system_fingerprint"],
                timestamp_start=r_dict["timestamp_start"],
                timestamp_end=r_dict["timestamp_end"],
                latency_ms=r_dict["latency_ms"],
                http_status=r_dict["http_status"],
                result_class=r_dict["result_class"],
                raw_output_sha256=r_dict["raw_output_sha256"],
                parsed_output_sha256=r_dict["parsed_output_sha256"],
                decoder_status=r_dict["decoder_status"],
                previous_receipt_sha256=ledger._last_sha,
                receipt_sha256=new_sha,
            )
            ledger._receipts.append(new_receipt)
            ledger._last_sha = new_sha
        return ledger

    @classmethod
    def load(cls, path: str | Path) -> "ReceiptLedger":
        """Load receipts from a JSONL file and verify the chain."""
        path = Path(path)
        ledger = cls()
        with open(path) as f:
            for line in f:
                entry = json.loads(line)
                receipt = ReceiptEntry(**entry)
                ledger.add(receipt)
        assert ledger.verify_chain(), "Receipt chain verification failed"
        return ledger
