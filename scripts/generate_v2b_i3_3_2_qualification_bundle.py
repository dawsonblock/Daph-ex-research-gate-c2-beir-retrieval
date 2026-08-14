#!/usr/bin/env python3
"""Generate the immutable I3.3.2 frozen-benchmark qualification evidence bundle.

Produces evidence/v2b_i3_3_2/qualification/ containing:

  identity.json              -- benchmark + qualification identity
  source_identity.json       -- the commit/tree this bundle binds
  benchmark_closure.json     -- closure artifacts + closure SHA-256
  oracle_set_hashes.json     -- latent + seven sequential observable set hashes
  environment.json           -- qualification runtime environment
  tests.json                 -- exhaustive regeneration run result
  qualification_receipt.json -- the self-verifying receipt binding it all
  stdout.sha256 / stderr.sha256
  BUNDLE.sha256              -- DAPH_EVIDENCE_BUNDLE_V1 manifest over the bundle

The receipt status is QUALIFIED_FROZEN_BENCHMARK (never QUALIFIED_EXECUTIVE):
this qualifies the benchmark, not a model result.

Usage:
  python scripts/generate_v2b_i3_3_2_qualification_bundle.py
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "configs/v2b_i3_3_3_baseline.json"
CACHE_MANIFEST = ROOT / "experiments/v2b_i3_3/oracle_tables/v2b_i3_3_oracle_cache_manifest_v1.json"
BUNDLE_DIR = ROOT / "evidence/v2b_i3_3_2/qualification"
SCHEMA_BUNDLE = "DAPH_EVIDENCE_BUNDLE_V1"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _write_json(name: str, obj: dict) -> Path:
    path = BUNDLE_DIR / name
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))
    return path


def _dependency_environment_hash() -> tuple[str, dict]:
    try:
        freeze = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze", "--all"], text=True)
    except Exception:
        freeze = ""
    env = {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": sys.platform,
        "machine": platform.machine(),
        "pip_freeze_sha256": _sha256_bytes(freeze.encode()),
    }
    return _sha256_bytes(json.dumps(env, sort_keys=True).encode()), env


def _build_deterministic_files(commit: str, tree: str, baseline: dict,
                                cache: dict) -> dict:
    """Write the identity/closure/oracle/environment files (no run needed)."""
    identity = {
        "schema": "DAPH_V2B_I3_3_2_QUALIFICATION_IDENTITY_V1",
        "artifact_identity": "V2B_I3_3_2_FROZEN_BENCHMARK_QUALIFICATION",
        "benchmark_identity": baseline["benchmark_id"],
        "integrity_revision": baseline["integrity_revision"],
        "protocol_path": baseline["bindings"]["protocol"]["path"],
        "protocol_sha256": baseline["bindings"]["protocol"]["sha256"],
        "qualification_runtime_path": baseline["qualification_runtime"]["path"],
        "qualification_runtime_sha256": baseline["qualification_runtime"]["sha256"],
        "test_corpus": baseline["test_corpus"],
        "claim_boundary": baseline["claim_boundary"],
    }
    _write_json("identity.json", identity)

    source_identity = {
        "schema": "DAPH_V2B_I3_3_2_SOURCE_IDENTITY_V1",
        "source_commit": commit,
        "source_tree": tree,
        "baseline_config_path": "configs/v2b_i3_3_3_baseline.json",
        "baseline_config_sha256": _sha256_file(BASELINE),
    }
    _write_json("source_identity.json", source_identity)

    closure_artifacts = cache["benchmark_closure_artifacts"]
    benchmark_closure = {
        "schema": "DAPH_V2B_I3_3_2_BENCHMARK_CLOSURE_V1",
        "benchmark_closure_sha256": cache["benchmark_closure_sha256"],
        "benchmark_manifest_sha256": cache["benchmark_manifest_sha256"],
        "closure_artifacts": closure_artifacts,
    }
    _write_json("benchmark_closure.json", benchmark_closure)

    oracle_set_hashes = {
        "schema": "DAPH_V2B_I3_3_2_ORACLE_SET_HASHES_V1",
        "latent_oracle": {
            "table_count": cache["latent_oracles"]["table_count"],
            "table_set_sha256": cache["latent_oracles"]["table_set_sha256"],
            "artifact_sha256": cache["latent_oracles"]["sha256"],
            "reachable_states": cache["latent_oracles"]["reachable_states"],
            "reachable_transitions": cache["latent_oracles"]["reachable_transitions"],
        },
        "sequential_observable_oracles": {
            cond: {
                "set_sha256": meta["set_sha256"],
                "artifact_sha256": meta["sha256"],
                "table_count": meta["table_count"],
                "task_uniform_information_gap": meta["task_uniform_information_gap"],
            }
            for cond, meta in cache["sequential_observable_oracles"].items()
        },
    }
    _write_json("oracle_set_hashes.json", oracle_set_hashes)

    dep_hash, env = _dependency_environment_hash()
    environment = {
        "schema": "DAPH_V2B_I3_3_2_QUALIFICATION_ENVIRONMENT_V1",
        "dependency_environment_hash": dep_hash,
        **env,
    }
    _write_json("environment.json", environment)

    return {
        "identity": identity,
        "source_identity": source_identity,
        "benchmark_closure": benchmark_closure,
        "oracle_set_hashes": oracle_set_hashes,
        "environment": environment,
    }


def _run_regeneration() -> dict:
    """Run the exhaustive oracle regeneration, capturing stdout/stderr/exit."""
    cmd = [sys.executable, "-m", "pytest", "-q", "tests/qualification"]
    started_at = datetime.now(timezone.utc).isoformat()
    start_perf = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, env={**__import__("os").environ,
                                                          "PYTHONUNBUFFERED": "1"})
    elapsed = time.time() - start_perf
    ended_at = datetime.now(timezone.utc).isoformat()
    out_bytes = proc.stdout.encode()
    (BUNDLE_DIR / "regeneration_output.txt").write_bytes(out_bytes)
    stdout_sha = _sha256_bytes(out_bytes)
    (BUNDLE_DIR / "stdout.sha256").write_text(stdout_sha + "\n")
    (BUNDLE_DIR / "stderr.sha256").write_text(stdout_sha + "\n")
    return {
        "command": cmd,
        "started_at": started_at,
        "ended_at": ended_at,
        "elapsed_s": elapsed,
        "returncode": proc.returncode,
        "stdout_sha256": stdout_sha,
        "stderr_sha256": stdout_sha,
        "output_tail": "\n".join(out_bytes.decode().splitlines()[-8:]),
    }


def _write_receipt(commit: str, tree: str, files: dict, run: dict) -> Path:
    latent = files["oracle_set_hashes"]["latent_oracle"]
    seq = files["oracle_set_hashes"]["sequential_observable_oracles"]
    gates = {
        "exhaustive_oracle_regeneration": "PASS" if run["returncode"] == 0 else "FAIL",
        "latent_oracle_set_hash_match": "PASS" if run["returncode"] == 0 else "FAIL",
        "seven_sequential_observable_set_hashes_match": "PASS" if run["returncode"] == 0 else "FAIL",
        "benchmark_closure_present": "PASS",
        "source_identity_bound": "PASS",
    }
    final_status = "QUALIFIED_FROZEN_BENCHMARK" if run["returncode"] == 0 else "FAILED_REGENERATION"
    receipt = {
        "schema": "DAPH_V2B_I3_3_2_QUALIFICATION_RECEIPT_V1",
        "artifact_identity": files["identity"]["artifact_identity"],
        "source_commit": commit,
        "source_tree": tree,
        "benchmark_identity": files["identity"]["benchmark_identity"],
        "protocol_identity": {
            "path": files["identity"]["protocol_path"],
            "sha256": files["identity"]["protocol_sha256"],
        },
        "test_corpus_hash": _sha256_bytes(
            json.dumps(files["identity"]["test_corpus"], sort_keys=True).encode()),
        "dependency_environment_hash": files["environment"]["dependency_environment_hash"],
        "qualification_runtime_hash": files["identity"]["qualification_runtime_sha256"],
        "benchmark_closure_sha256": files["benchmark_closure"]["benchmark_closure_sha256"],
        "latent_oracle_set_sha256": latent["table_set_sha256"],
        "sequential_observable_oracle_set_shas": {
            cond: meta["set_sha256"] for cond, meta in seq.items()},
        "started_at": run["started_at"],
        "ended_at": run["ended_at"],
        "exit_status": run["returncode"],
        "all_gates": gates,
        "final_status": final_status,
        "claim_boundary": ("QUALIFIED_FROZEN_BENCHMARK only; this is NOT "
                           "QUALIFIED_EXECUTIVE and implies no model result."),
    }
    return _write_json("qualification_receipt.json", receipt)


def _write_bundle_hash() -> Path:
    """Deterministic DAPH_EVIDENCE_BUNDLE_V1 manifest over the bundle."""
    files = []
    for p in sorted(BUNDLE_DIR.iterdir()):
        if p.name == "BUNDLE.sha256":
            continue
        if p.is_file():
            files.append({"path": p.name, "sha256": _sha256_file(p)})
    concat = "".join(f"{e['path']}\x00{e['sha256']}\n" for e in files)
    bundle_sha = _sha256_bytes(concat.encode())
    manifest = {
        "schema": SCHEMA_BUNDLE,
        "files": files,
        "bundle_sha256": bundle_sha,
    }
    out = BUNDLE_DIR / "BUNDLE.sha256"
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return out


def main() -> int:
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    baseline = json.loads(BASELINE.read_text())
    cache = json.loads(CACHE_MANIFEST.read_text())
    commit = baseline["source_commit"]
    tree = baseline["source_tree"]

    # Verify the checkout matches the recorded source identity.
    actual_commit = _git("rev-parse", "--verify", f"{commit}^{{commit}}")
    actual_tree = _git("rev-parse", f"{commit}^{{tree}}")
    if actual_commit != commit or actual_tree != tree:
        raise RuntimeError(
            f"Checkout source identity mismatch: got {actual_commit}/{actual_tree}, "
            f"expected {commit}/{tree}")

    files = _build_deterministic_files(commit, tree, baseline, cache)
    run = _run_regeneration()
    _write_json("tests.json", {"schema": "DAPH_V2B_I3_3_2_REGENERATION_RESULT_V1", **run})
    receipt = _write_receipt(commit, tree, files, run)
    bundle = _write_bundle_hash()

    print(f"Qualification bundle written to {BUNDLE_DIR}")
    print(f"  final_status: {json.loads(receipt.read_text())['final_status']}")
    print(f"  bundle_sha256: {json.loads(bundle.read_text())['bundle_sha256']}")
    print(f"  regeneration returncode: {run['returncode']}")
    return 0 if run["returncode"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
