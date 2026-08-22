#!/usr/bin/env python3
"""R13-RUNTIME-001: Runtime replacement validator (read-only).

Verifies that every quarantined deviant-runtime trajectory key has exactly
one accepted frozen-runtime replacement. Does NOT mutate any files.

Usage:
    python scripts/r13_validate_runtime_replacements.py \
        --accepted /path/to/results.jsonl \
        --quarantine /path/to/quarantine/runtime_deviation/results.jsonl \
        --frozen-runtime-sha c64eb7b828feeac599e4bb001bf14a790efabe0d8e39c4f9cc4486062ad024c3 \
        [--output /path/to/replacement_status.json] \
        [--run-active]

Exit codes:
    0 = all replacements present and verified (QUARANTINED_AND_FULLY_RERUN)
    1 = replacements in progress (REPLACEMENT_IN_PROGRESS) — non-fatal during active run
    2 = validation failure (wrong runtime, duplicates, quarantine corruption)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


FROZEN_RUNTIME_SHA = "c64eb7b828feeac599e4bb001bf14a790efabe0d8e39c4f9cc4486062ad024c3"
FROZEN_EXPERIMENT_COMMIT = "5454246b7e61adfb7a093eb5a1f731347071270d"
FROZEN_PROTOCOL_SHA = "9590440d2744a6409cc19bc7ba8168d22cb7cee80952fb520a54134815c312c5"
FROZEN_GGUF_SHA = "2ad4c9ce431a2d5b80af37983828c2cfb8f4909792ca5075e0370e3a71ca013d"
FROZEN_BACKEND = "2ad4c9ce431a2d5b"
EXPECTED_QUARANTINE_COUNT = 28


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def record_hash(record: dict) -> str:
    """Stable hash of a record for identity comparison."""
    return hashlib.sha256(
        json.dumps(record, sort_keys=True, default=str).encode()
    ).hexdigest()


def validate(
    accepted_path: Path,
    quarantine_path: Path,
    frozen_runtime_sha: str,
    run_active: bool = False,
) -> dict:
    """Run the full validation and return a status report."""

    # --- Step 1: Validate quarantine itself ---
    quarantined = load_jsonl(quarantine_path)
    q_keys = [r.get("trajectory_key", "") for r in quarantined]
    q_key_set = set(q_keys)
    q_duplicates = len(q_keys) - len(q_key_set)

    quarantine_valid = (
        len(quarantined) == EXPECTED_QUARANTINE_COUNT
        and len(q_key_set) == EXPECTED_QUARANTINE_COUNT
        and q_duplicates == 0
    )

    # --- Step 2: Validate accepted results ---
    accepted = load_jsonl(accepted_path)
    a_keys = [r.get("trajectory_key", "") for r in accepted]
    a_key_set = set(a_keys)
    a_duplicates = len(a_keys) - len(a_key_set)
    a_key_counts = Counter(a_keys)

    # --- Step 3: Calculate replacement status ---
    Q = q_key_set
    A = a_key_set
    replaced = Q & A
    missing = Q - A

    # --- Step 4: Verify every replacement uses frozen runtime ---
    # Per-trajectory records carry: backend_identity, protocol_id
    # Run-level identities (runtime_config_sha256, protocol_sha256, gguf_sha256)
    # are in identity_frozen.json, not in individual trajectory records.
    wrong_backend = []
    wrong_protocol_id = []
    duplicate_replacements = []

    # Build accepted lookup: key -> list of records
    accepted_by_key: dict[str, list[dict]] = {}
    for r in accepted:
        k = r.get("trajectory_key", "")
        accepted_by_key.setdefault(k, []).append(r)

    per_key_reports = []
    for k in Q:
        q_record = next(r for r in quarantined if r.get("trajectory_key") == k)
        a_records = accepted_by_key.get(k, [])

        report = {
            "trajectory_key": k,
            "deviant_present": True,
            "accepted_replacement_present": len(a_records) > 0,
            "accepted_replacement_count": len(a_records),
            "deviant_record_sha256": record_hash(q_record),
        }

        if len(a_records) > 1:
            duplicate_replacements.append(k)
            report["status"] = "DUPLICATE_REPLACEMENT"
        elif len(a_records) == 1:
            a_record = a_records[0]
            report["accepted_record_sha256"] = record_hash(a_record)
            report["same_record_identity"] = (
                report["accepted_record_sha256"] == report["deviant_record_sha256"]
            )

            # Check per-record identities that ARE present
            backend_ok = a_record.get("backend_identity", "") == FROZEN_BACKEND
            protocol_ok = a_record.get("protocol_id", "") == "I3_15C_CONFIRMATION_PROTOCOL_V2"

            if not backend_ok:
                wrong_backend.append(k)
            if not protocol_ok:
                wrong_protocol_id.append(k)

            # Runtime SHA is verified at run-level via identity_frozen.json,
            # not per-trajectory. The server config is verified separately.
            report["accepted_backend_identity"] = a_record.get("backend_identity", "")
            report["accepted_protocol_id"] = a_record.get("protocol_id", "")
            report["backend_identity_is_frozen"] = backend_ok
            report["protocol_id_is_frozen"] = protocol_ok

            is_frozen = backend_ok and protocol_ok
            report["status"] = "REPLACED" if is_frozen else "WRONG_IDENTITY"
        else:
            report["status"] = "MISSING_REPLACEMENT"

        per_key_reports.append(report)

    # --- Step 5: Determine overall status ---
    wrong_identity_count = len(wrong_backend) + len(wrong_protocol_id)
    has_errors = (
        not quarantine_valid
        or wrong_identity_count > 0
        or len(duplicate_replacements) > 0
        or a_duplicates > 0
    )

    if has_errors:
        status = "VALIDATION_FAILURE"
        exit_code = 2
    elif len(missing) > 0:
        if run_active:
            status = "REPLACEMENT_IN_PROGRESS"
            exit_code = 1
        else:
            status = "INCOMPLETE_AT_CLOSURE"
            exit_code = 2
    else:
        # All replaced — verify all have frozen identities
        all_frozen = all(
            r.get("backend_identity_is_frozen", False) and r.get("protocol_id_is_frozen", False)
            for r in per_key_reports
            if r["status"] == "REPLACED"
        )
        if all_frozen:
            status = "QUARANTINED_AND_FULLY_RERUN"
            exit_code = 0
        else:
            status = "VALIDATION_FAILURE"
            exit_code = 2

    # --- Step 6: Leakage check (deviant record hashes in accepted?) ---
    q_hashes = {record_hash(r) for r in quarantined}
    a_hashes = {record_hash(r) for r in accepted}
    leaked = q_hashes & a_hashes

    report = {
        "deviation_id": "R13-RUNTIME-001",
        "quarantined_count": len(quarantined),
        "quarantined_unique_keys": len(q_key_set),
        "quarantine_duplicates": q_duplicates,
        "quarantine_valid": quarantine_valid,
        "accepted_count": len(accepted),
        "accepted_unique_keys": len(a_key_set),
        "accepted_duplicates": a_duplicates,
        "replacement_count": len(replaced),
        "missing_replacement_count": len(missing),
        "wrong_identity_count": wrong_identity_count,
        "duplicate_replacement_count": len(duplicate_replacements),
        "wrong_backend_count": len(wrong_backend),
        "wrong_protocol_id_count": len(wrong_protocol_id),
        "record_leakage_count": len(leaked),
        "status": status,
        "missing_keys": sorted(missing),
        "wrong_backend_keys": sorted(wrong_backend),
        "wrong_protocol_keys": sorted(wrong_protocol_id),
        "duplicate_replacement_keys": sorted(duplicate_replacements),
        "leaked_record_hashes": sorted(leaked),
        "per_key_reports": per_key_reports,
        "frozen_runtime_sha256": frozen_runtime_sha,
        "note": "Per-trajectory records carry backend_identity and protocol_id. Runtime config SHA is verified at run-level via identity_frozen.json, not per-trajectory.",
        "run_active": run_active,
    }

    return report, exit_code


def main():
    parser = argparse.ArgumentParser(
        description="R13-RUNTIME-001 runtime replacement validator (read-only)"
    )
    parser.add_argument("--accepted", required=True, type=Path, help="Path to accepted results.jsonl")
    parser.add_argument("--quarantine", required=True, type=Path, help="Path to quarantined results.jsonl")
    parser.add_argument(
        "--frozen-runtime-sha",
        default=FROZEN_RUNTIME_SHA,
        help="Frozen runtime config SHA256 (default: %(default)s)",
    )
    parser.add_argument("--output", type=Path, default=None, help="Write status JSON to this path")
    parser.add_argument("--run-active", action="store_true", help="R13 is still running (missing replacements are non-fatal)")
    args = parser.parse_args()

    if not args.accepted.exists():
        print(f"ERROR: accepted file not found: {args.accepted}", file=sys.stderr)
        sys.exit(2)
    if not args.quarantine.exists():
        print(f"ERROR: quarantine file not found: {args.quarantine}", file=sys.stderr)
        sys.exit(2)

    report, exit_code = validate(
        args.accepted,
        args.quarantine,
        args.frozen_runtime_sha,
        run_active=args.run_active,
    )

    # Print summary
    print(f"R13-RUNTIME-001 Replacement Validation")
    print(f"  quarantined: {report['quarantined_count']} ({report['quarantined_unique_keys']} unique)")
    print(f"  accepted: {report['accepted_count']} ({report['accepted_unique_keys']} unique)")
    print(f"  replaced: {report['replacement_count']}")
    print(f"  missing: {report['missing_replacement_count']}")
    print(f"  wrong identity: {report['wrong_identity_count']}")
    print(f"  duplicate replacements: {report['duplicate_replacement_count']}")
    print(f"  record leakage: {report['record_leakage_count']}")
    print(f"  status: {report['status']}")

    if report["missing_keys"]:
        print(f"  missing keys: {report['missing_keys'][:5]}{'...' if len(report['missing_keys']) > 5 else ''}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"  written to: {args.output}")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
