#!/usr/bin/env python3
"""Qualify research artifacts for I3.30R2 live study.

Verifies:
- Declared files exist
- All SHAs match frozen manifest
- Model identity matches
- Dataset identity matches
- Schema identity matches
- Benchmark identity matches
- Source files match freeze
- No mutable freeze overwrite occurred

Usage:
    python scripts/qualify_research_artifacts.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "experiments" / "i3_30r" / "live_study_frozen_manifest.json"


def sha256_file(path: Path) -> str:
    if not path.exists():
        return "MISSING"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def check(label: str, expected: str, actual: str) -> bool:
    ok = expected == actual
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}")
    if not ok:
        print(f"         expected: {expected}")
        print(f"         actual:   {actual}")
    return ok


def main():
    print("=" * 70)
    print("I3.30R2 Artifact Qualification")
    print("=" * 70)

    if not MANIFEST_PATH.exists():
        print(f"FAIL: Manifest not found at {MANIFEST_PATH}")
        sys.exit(1)

    manifest = json.loads(MANIFEST_PATH.read_text())
    all_pass = True

    # --- Candidate model ---
    print("\n--- Candidate (V3R2-A) ---")
    p = REPO_ROOT / manifest["candidate"]["model_path"]
    all_pass &= check("V3R2-A model exists + SHA", manifest["candidate"]["model_sha256"], sha256_file(p))
    p = REPO_ROOT / manifest["candidate"]["feature_schema_path"]
    all_pass &= check("V3R2-A feature schema SHA", manifest["candidate"]["feature_schema_sha256"], sha256_file(p))

    # --- Baseline model ---
    print("\n--- Baseline (V1) ---")
    p = REPO_ROOT / manifest["baseline"]["model_path"]
    all_pass &= check("V1 model exists + SHA", manifest["baseline"]["model_sha256"], sha256_file(p))
    p = REPO_ROOT / manifest["baseline"]["schema_path"]
    all_pass &= check("V1 schema SHA", manifest["baseline"]["schema_sha256"], sha256_file(p))

    # --- Pinned Qwen ---
    print("\n--- Pinned Model (Qwen) ---")
    p = Path(manifest["pinned_model"]["path"])
    all_pass &= check("Qwen GGUF exists + SHA", manifest["pinned_model"]["sha256"], sha256_file(p))

    # --- Canonical semantics ---
    print("\n--- Canonical Semantics ---")
    cs = manifest["canonical_semantics"]
    all_pass &= check("EPISTEMIC_SEMANTICS_V1.md SHA", cs["spec_sha256"], sha256_file(REPO_ROOT / cs["spec_path"]))
    all_pass &= check("topology.py SHA", cs["topology_sha256"], sha256_file(REPO_ROOT / cs["topology_path"]))
    all_pass &= check("types.py SHA", cs["types_sha256"], sha256_file(REPO_ROOT / cs["types_path"]))
    all_pass &= check("v3_features.py SHA", cs["v3_features_sha256"], sha256_file(REPO_ROOT / cs["v3_features_path"]))

    # --- Authority ---
    print("\n--- Authority ---")
    auth = manifest["authority"]
    all_pass &= check("policy_v3.py SHA", auth["policy_v3_sha256"], sha256_file(REPO_ROOT / auth["policy_v3_path"]))

    # --- Executor ---
    print("\n--- Executor ---")
    ex = manifest["executor"]
    all_pass &= check("executor.py SHA", ex["sha256"], sha256_file(REPO_ROOT / ex["path"]))
    all_pass &= check("schema.py SHA", ex["schema_sha256"], sha256_file(REPO_ROOT / ex["schema_path"]))

    # --- Benchmark ---
    print("\n--- Benchmark ---")
    bm = manifest["benchmark"]
    all_pass &= check("D1-D4 generator SHA", bm["d1_d4_generator_sha256"], sha256_file(REPO_ROOT / bm["d1_d4_generator_path"]))
    all_pass &= check("D5 generator SHA", bm["d5_generator_sha256"], sha256_file(REPO_ROOT / bm["d5_generator_path"]))

    # Verify benchmark hashes by regeneration
    sys.path.insert(0, str(REPO_ROOT))
    from hrm_adaptive_memory.executive.evidence_benchmark.i3_29_safety_generator import generate_i3_29_benchmark, compute_benchmark_hash
    from hrm_adaptive_memory.executive.evidence_benchmark.i3_30_d5_generator import generate_d5_tasks
    tasks = generate_i3_29_benchmark(seed=bm["seed"])
    bench_hash = compute_benchmark_hash(tasks)
    all_pass &= check("D1-D4 benchmark hash (regenerated)", bm["d1_d4_benchmark_sha256"], bench_hash)
    d5 = generate_d5_tasks(seed=bm["seed"], n_tasks=35)
    import hashlib as h2
    d5_json = json.dumps([{
        "task_id": t.task_id, "category": t.category, "n_hyps": len(t.hypotheses),
        "n_ev": len(t.evidence_items), "correct_hyp": t.correct_hypothesis_id,
        "expected_terminal": t.expected_terminal.value if hasattr(t.expected_terminal, "value") else str(t.expected_terminal),
        "oracle": list(t.oracle_resolution_path),
    } for t in d5], sort_keys=True)
    d5_hash = h2.sha256(d5_json.encode()).hexdigest()
    all_pass &= check("D5R benchmark hash (regenerated)", bm["d5_benchmark_sha256"], d5_hash)

    # --- Runner ---
    print("\n--- Runner ---")
    all_pass &= check("runner SHA", manifest["runner"]["sha256"], sha256_file(REPO_ROOT / manifest["runner"]["path"]))

    # --- Tests ---
    print("\n--- Tests ---")
    tests = manifest["tests"]
    test_files = {
        "test_epistemic_topology": "tests/unit/test_epistemic_topology.py",
        "test_semantic_conformance": "tests/unit/test_semantic_conformance.py",
        "test_authority_v3": "tests/unit/test_authority_v3.py",
        "test_authority_v2": "tests/unit/test_authority_v2.py",
        "test_mdsg_topology_invariant": "tests/unit/test_mdsg_topology_invariant.py",
    }
    for key, path in test_files.items():
        all_pass &= check(f"{key} SHA", tests[key], sha256_file(REPO_ROOT / path))

    # --- Structural holdout ---
    print("\n--- Structural Holdout ---")
    sh = manifest["structural_holdout"]
    all_pass &= check("structural_holdout_gates.json SHA", sh["gates_sha256"], sha256_file(REPO_ROOT / sh["gates_path"]))
    all_pass &= check("V3R2-A all_pass", True, sh["v3r2_a_all_pass"])
    all_pass &= check("V3R2-A FAR_ANSWER = 0", 0.0, sh["v3r2_a_far_answer"])
    all_pass &= check("V3R2-A FAR_DEFER = 0", 0.0, sh["v3r2_a_far_defer"])
    all_pass &= check("V3R2-A precision = 1.0", 1.0, sh["v3r2_a_precision"])

    # --- Causal boundary data ---
    print("\n--- Causal Boundary Data ---")
    cb = manifest["causal_boundary_data"]
    all_pass &= check("causal_actions_v3.jsonl SHA", cb["sha256"], sha256_file(REPO_ROOT / cb["path"]))

    # --- Preregistration ---
    print("\n--- Preregistration ---")
    pr = manifest["preregistration"]
    all_pass &= check("I3_30R_PREREGISTRATION.json SHA", pr["sha256"], sha256_file(REPO_ROOT / pr["path"]))

    # --- Manifest immutability ---
    print("\n--- Manifest Immutability ---")
    all_pass &= check("manifest immutable flag", True, manifest.get("immutable", False))

    # --- Summary ---
    print("\n" + "=" * 70)
    if all_pass:
        print("OVERALL: PASS — All artifacts qualified for live study.")
    else:
        print("OVERALL: FAIL — Some artifacts do not match freeze manifest.")
    print("=" * 70)

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
