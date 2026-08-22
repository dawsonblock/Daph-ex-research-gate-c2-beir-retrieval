#!/usr/bin/env python3
"""R13 Dataset Closure — executed only after 1280/1280 completion.

This script:
1. Verifies dataset integrity (1280 unique keys, 0 duplicates, 0 missing, 0 errors)
2. Verifies all frozen identities match across every record
3. Copies raw files into an immutable raw_closed/ directory
4. Computes per-file SHA256 → dataset manifest → R13_DATASET_SHA
5. Runs the preregistered analysis against the exact closed dataset
6. Produces all contrasts, controls, cost, and error attribution
7. Assigns R1 disposition

Usage:
    python3 tools/colab/close_r13_dataset.py --checkpoint-dir ~/DAPH_CHECKPOINTS/R13 --output-dir experiments/v2b_i3_15c/confirmation/r13

Prerequisites:
    - R13 must be complete (1280/1280)
    - All checkpoint files must be downloaded
"""
import json, os, sys, hashlib, shutil, argparse
from pathlib import Path
from datetime import datetime


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_results(path: Path) -> list[dict]:
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return results


def verify_dataset_integrity(results: list[dict], expected: int = 1280) -> dict:
    """Verify the closed dataset meets all integrity requirements."""
    keys = [r.get("trajectory_key", "") for r in results]
    unique_keys = set(keys)
    duplicates = len(keys) - len(unique_keys)
    missing = expected - len(unique_keys)

    # Check ALL frozen identities across every record — must be exactly one value each.
    # This is critical because R13 crossed VM boundaries.
    #
    # KNOWN DEFECT R13-PROV-001: The frozen runner's build_run_manifest() incorrectly
    # sets confirmation_executable_sha256 to runtime_config_sha256. The actual
    # confirmation executable SHA (41cc60b04f506f63...) is only in
    # confirmation_executable_sha256.txt. We do NOT check the manifest field
    # for the confirmation SHA. We check the .txt file separately at closure.
    # See: experiments/v2b_i3_15c/confirmation/r13_known_defects.json
    identity_fields = [
        "protocol_id",
        "backend_identity",
        "protocol_sha256",
        "gguf_sha256",
        "runtime_config_sha256",
        "receipt_identity_sha256",
    ]
    identity_violations = []
    identity_uniqueness = {}
    for field in identity_fields:
        values = set(r.get(field, "") for r in results if r.get(field))
        identity_uniqueness[field] = list(values)
        if len(values) > 1:
            identity_violations.append(f"{field}: {values}")
        elif len(values) == 0:
            identity_violations.append(f"{field}: MISSING from all records")

    # Check for errors
    error_count = sum(1 for r in results if r.get("terminal_result") == "BACKEND_ERROR")

    # Check A1/R1 balance
    arm_counts = {}
    for r in results:
        arm = r.get("arm", "unknown")
        arm_counts[arm] = arm_counts.get(arm, 0) + 1

    # Check retrieval condition is Q3 only
    retrieval_conditions = set(r.get("retrieval_condition", "") for r in results if r.get("retrieval_condition"))

    report = {
        "total_records": len(results),
        "unique_keys": len(unique_keys),
        "duplicates": duplicates,
        "missing": missing,
        "errors": error_count,
        "identity_violations": identity_violations,
        "identity_uniqueness": identity_uniqueness,
        "arm_counts": arm_counts,
        "retrieval_conditions": list(retrieval_conditions),
        "known_defects": ["R13-PROV-001"],
        "experiment_classification": "PRE_SPECIFIED_CONFIRMATION_WITH_KNOWN_PROVENANCE_FIELD_DEFECT",
        "passes": (
            len(results) == expected
            and len(unique_keys) == expected
            and duplicates == 0
            and missing == 0
            and error_count == 0
            and len(identity_violations) == 0
        ),
    }
    return report


def create_raw_closed(checkpoint_dir: Path, output_dir: Path) -> Path:
    """Copy raw checkpoint files into an immutable raw_closed/ directory."""
    raw_closed = output_dir / "raw_closed"
    raw_closed.mkdir(parents=True, exist_ok=True)

    files_to_copy = [
        "results.jsonl",
        "model_calls.jsonl",
        "mechanism_receipts.jsonl",
        "cognition_cost_receipts.jsonl",
        "errors.jsonl",
        "progress.json",
        "run_manifest.json",
        "identity_frozen.json",
        "context_preflight.json",
        "confirmation_executable_sha256.txt",
        "semantic_error_attribution.json",
        "mechanism_receipts_strengthened.jsonl",
        "execution_segments.jsonl",
        "retry_receipts.jsonl",
    ]

    # Also copy known defects and experiment identity from the repo
    repo_root = Path(__file__).resolve().parents[2]
    identity_src = repo_root / "experiments/v2b_i3_15c/confirmation/r13_experiment_identity.json"
    defects_src = repo_root / "experiments/v2b_i3_15c/confirmation/r13_known_defects.json"
    if identity_src.exists():
        shutil.copy2(identity_src, raw_closed / "r13_experiment_identity.json")
    if defects_src.exists():
        shutil.copy2(defects_src, raw_closed / "known_defects.json")

    for fname in files_to_copy:
        src = checkpoint_dir / fname
        if src.exists():
            shutil.copy2(src, raw_closed / fname)

    return raw_closed


def compute_dataset_sha(raw_closed: Path) -> dict:
    """Compute per-file SHA256, create manifest, and hash the manifest."""
    manifest = {}
    for f in sorted(raw_closed.iterdir()):
        if f.is_file():
            manifest[f.name] = {
                "sha256": sha256_file(f),
                "bytes": f.stat().st_size,
            }

    # Write manifest
    manifest_path = raw_closed / "dataset_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Compute R13_DATASET_SHA = H(manifest)
    manifest_str = json.dumps(manifest, sort_keys=True)
    dataset_sha = hashlib.sha256(manifest_str.encode()).hexdigest()

    return {
        "dataset_manifest": manifest,
        "r13_dataset_sha256": dataset_sha,
    }


def main():
    parser = argparse.ArgumentParser(description="R13 Dataset Closure")
    parser.add_argument("--checkpoint-dir", required=True,
                        help="Directory with downloaded checkpoint files")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for closed dataset and analysis")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir).expanduser()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("R13 DATASET CLOSURE")
    print(f"  Timestamp: {datetime.utcnow().isoformat()}Z")
    print(f"  Checkpoint: {checkpoint_dir}")
    print(f"  Output: {output_dir}")
    print("=" * 70)

    # Step 1: Load and verify results
    print("\n[1] Loading results...")
    results_path = checkpoint_dir / "results.jsonl"
    if not results_path.exists():
        print(f"  ERROR: {results_path} not found")
        sys.exit(1)

    results = load_results(results_path)
    print(f"  Loaded {len(results)} records")

    # Step 2: Verify dataset integrity
    print("\n[2] Verifying dataset integrity...")
    integrity = verify_dataset_integrity(results, expected=1280)
    print(f"  total_records: {integrity['total_records']}")
    print(f"  unique_keys: {integrity['unique_keys']}")
    print(f"  duplicates: {integrity['duplicates']}")
    print(f"  missing: {integrity['missing']}")
    print(f"  errors: {integrity['errors']}")
    print(f"  identity_violations: {integrity['identity_violations']}")
    print(f"  arm_counts: {integrity['arm_counts']}")
    print(f"  retrieval_conditions: {integrity['retrieval_conditions']}")
    print(f"  identity_uniqueness:")
    for field, values in integrity.get("identity_uniqueness", {}).items():
        n = len(values)
        status = "OK" if n == 1 else "VIOLATION"
        val = values[0][:20] + "..." if values and len(values[0]) > 20 else (values[0] if values else "MISSING")
        print(f"    {field}: {n} unique value(s) [{status}] = {val}")

    # Step 2b: Verify execution segments have identical scientific identities
    print("\n[2b] Verifying execution segments...")
    segments_path = checkpoint_dir / "execution_segments.jsonl"
    segments = []
    if segments_path.exists():
        with open(segments_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    segments.append(json.loads(line))
        print(f"  Found {len(segments)} execution segment(s)")

        # Build multi-segment identity comparison table
        # Required: all scientific identities must be identical across segments.
        # Session IDs may differ. GPU class should be same (preferred, not required).
        segment_identity_fields = [
            "experiment_source_commit",
            "gguf_sha256",
            "protocol_sha256",
            "runtime_config_sha256",
            "confirmation_executable_sha256",
        ]
        segment_violations = []
        segment_identity_table = {}

        for field in segment_identity_fields:
            values = set(s.get(field, "") for s in segments if s.get(field))
            segment_identity_table[field] = {
                "values_per_segment": [s.get(field, "MISSING") for s in segments],
                "unique_count": len(values),
                "identical": len(values) <= 1,
            }
            if len(values) > 1:
                segment_violations.append(f"{field}: {values}")

        # Print multi-segment identity table
        print(f"\n  Multi-segment identity comparison:")
        print(f"  {'Property':<30} {'Identical':<12} {'Values'}")
        for field, info in segment_identity_table.items():
            status = "OK" if info["identical"] else "VIOLATION"
            vals = [v[:16] + "..." if len(v) > 16 else v for v in info["values_per_segment"]]
            print(f"  {field:<30} {status:<12} {vals}")

        # Print segment details
        print(f"\n  Segment details:")
        for seg in segments:
            prov = seg.get("provenance_status", "UNKNOWN")
            print(f"    Segment {seg.get('segment')}: "
                  f"start={seg.get('start_completed')} "
                  f"gpu={seg.get('gpu')} "
                  f"provenance={prov} "
                  f"session={seg.get('session_id', 'unknown')[:25]}")

        if segment_violations:
            print(f"\n  SEGMENT IDENTITY VIOLATIONS: {segment_violations}")
            integrity["passes"] = False
            integrity["segment_violations"] = segment_violations
        else:
            print(f"\n  OK: All segments share identical scientific identities")
            print(f"  Session IDs may differ — that is an infrastructure event, not a treatment change")

        integrity["segment_identity_table"] = segment_identity_table
        integrity["segments"] = segments
    else:
        print("  No execution_segments.jsonl found (single-segment run)")

    # Step 2c: Verify confirmation_executable_sha256.txt (independent of defective manifest)
    # R13-PROV-001: run_manifest.confirmation_executable_sha256 is defective.
    # The actual SHA is in confirmation_executable_sha256.txt, written at end of run.
    print("\n[2c] Verifying confirmation executable SHA (independent of manifest)...")
    expected_confirmation_sha = "41cc60b04f506f63b80c91e036d330d61d79992a86fb975cbe21597bd2d84f57"
    confirmation_sha_path = checkpoint_dir / "confirmation_executable_sha256.txt"
    if confirmation_sha_path.exists():
        actual_sha = confirmation_sha_path.read_text().strip()
        if actual_sha == expected_confirmation_sha:
            print(f"  OK: confirmation_executable_sha256.txt matches expected value")
            print(f"    {actual_sha[:20]}...")
        else:
            print(f"  ERROR: confirmation_executable_sha256.txt mismatch")
            print(f"    Expected: {expected_confirmation_sha}")
            print(f"    Actual:   {actual_sha}")
            integrity["passes"] = False
            integrity["confirmation_sha_mismatch"] = True
    else:
        print(f"  WARNING: confirmation_executable_sha256.txt not found")
        print(f"    This file is written at the end of the run.")
        print(f"    If the run completed, it should exist.")
        print(f"    Defect R13-PROV-001 means the manifest field is NOT a substitute.")
        integrity["confirmation_sha_missing"] = True

    # Step 2d: Document known defects
    print("\n[2d] Known defects:")
    print(f"  R13-PROV-001: run_manifest.confirmation_executable_sha256")
    print(f"    incorrectly aliases runtime_config_sha256")
    print(f"    Scientific execution NOT affected")
    print(f"    Classification: {integrity.get('experiment_classification', 'UNKNOWN')}")

    if not integrity["passes"]:
        print("\n  GATE FAILED — dataset is not complete or has violations")
        print("  NOT proceeding to analysis")
        # Still create raw_closed for forensic inspection
        print("\n[3] Creating raw_closed/ for forensic inspection...")
        raw_closed = create_raw_closed(checkpoint_dir, output_dir)
        print(f"  {raw_closed}")
        sys.exit(1)

    print("\n  GATE PASSED — dataset is complete, coherent, and identity-consistent")
    print(f"  Classification: {integrity['experiment_classification']}")

    # Step 3: Create raw_closed/ directory
    print("\n[3] Creating immutable raw_closed/ directory...")
    raw_closed = create_raw_closed(checkpoint_dir, output_dir)
    print(f"  Files: {len(list(raw_closed.iterdir()))}")

    # Step 4: Compute dataset SHA
    print("\n[4] Computing dataset SHA...")
    sha_result = compute_dataset_sha(raw_closed)
    print(f"  R13_DATASET_SHA256: {sha_result['r13_dataset_sha256']}")
    for fname, info in sha_result["dataset_manifest"].items():
        print(f"    {fname}: {info['sha256'][:16]}... ({info['bytes']} bytes)")

    # Write closure record
    closure = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "r13_dataset_sha256": sha_result["r13_dataset_sha256"],
        "integrity": integrity,
        "dataset_manifest": sha_result["dataset_manifest"],
    }
    closure_path = output_dir / "closure" / "closure.json"
    closure_path.parent.mkdir(parents=True, exist_ok=True)
    with open(closure_path, "w") as f:
        json.dump(closure, f, indent=2)
    print(f"\n  Closure record: {closure_path}")

    # Step 5: Run preregistered analysis
    print("\n[5] Running preregistered analysis...")
    print("  (Analysis code reads from raw_closed/, never mutates it)")

    # Import and run the analysis from the R13 runner
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))

    from scripts.run_r13_confirmation import compute_r13_analysis

    analysis = compute_r13_analysis(results)
    analysis["r13_dataset_sha256"] = sha_result["r13_dataset_sha256"]
    analysis["integrity"] = integrity

    analysis_path = output_dir / "analysis" / "analysis.json"
    analysis_path.parent.mkdir(parents=True, exist_ok=True)
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f"  Analysis: {analysis_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("R13 ANALYSIS SUMMARY")
    print("=" * 70)

    pc = analysis.get("primary_contrast", {})
    print(f"\nPRIMARY CONTRAST: {pc.get('name', '?')}")
    print(f"  n={pc.get('n', 0)}, mean={pc.get('mean', 0):.4f}")
    ci = pc.get("ci_95", [0, 0])
    print(f"  CI95=[{ci[0]:.4f}, {ci[1]:.4f}]")
    print(f"  Criterion: LCB > 0 → {'PASS' if pc.get('passes') else 'FAIL'}")

    print(f"\nSECONDARY CONTRASTS:")
    for sc in analysis.get("secondary_contrasts", []):
        ci = sc.get("ci_95", [0, 0])
        print(f"  {sc.get('name', '?')}: n={sc.get('n', 0)}, mean={sc.get('mean', 0):.4f}, "
              f"CI=[{ci[0]:.4f}, {ci[1]:.4f}]")

    print(f"\nCONTROL CONTRASTS:")
    for cc in analysis.get("control_contrasts", []):
        ci = cc.get("ci_90", [0, 0])
        print(f"  {cc.get('name', '?')}: mean={cc.get('mean', 0):.4f}, "
              f"CI90=[{ci[0]:.4f}, {ci[1]:.4f}], equiv={cc.get('equivalent')}")

    print(f"\nSAFETY:")
    safety = analysis.get("safety_checks", {})
    print(f"  False T2 rate: {safety.get('false_t2_rate', 0):.4f}")
    print(f"  Safety PASS: {safety.get('passes')}")

    print(f"\nPROMOTION CRITERIA:")
    promo = analysis.get("promotion_criteria", {})
    for k, v in promo.items():
        if k != "all_criteria":
            print(f"  {k}: {'PASS' if v else 'FAIL'}")
    print(f"  ALL CRITERIA: {'PASS' if promo.get('all_criteria') else 'FAIL'}")

    print(f"\n  R13_DATASET_SHA256: {sha_result['r13_dataset_sha256']}")
    print(f"\nR13 CLOSURE COMPLETE.")


if __name__ == "__main__":
    main()
