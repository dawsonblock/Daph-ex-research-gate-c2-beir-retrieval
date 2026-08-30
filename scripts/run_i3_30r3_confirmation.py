"""I3.30R3 Confirmation runner: untouched structural confirmation.

Runs V3-SHADOW and V3-HARD on the fresh 400-task confirmation benchmark.
Does NOT include V1 (not needed for primary confirmation hypothesis).

Per the audit's section 21 design:
- Do not alter Q_V3R2-A, epsilon, authority threshold, certificate logic, or prompt
- Use V3-SHADOW and V3-HARD as primary arms
- Generate fresh tasks from structural configurations not in development benchmark
- Primary hypothesis: E[U_HARD - U_SHADOW] > 0
- For promotion: CI_95%,lower(ΔU) > 0

Usage:
    PYTHONPATH=. python3 scripts/run_i3_30r3_confirmation.py \\
        --gguf-path /path/to/Qwen2.5-7B-Instruct-Q4_K_M.gguf \\
        --output-dir experiments/i3_30r3/confirmation
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_confirmation_manifest(gguf_path: str) -> dict:
    """Compute manifest for the confirmation run.

    Freezes ALL executable components, dependencies, runtime parameters,
    and model artifacts. The execution path can never modify this identity.
    """
    from hrm_adaptive_memory.executive.evidence_benchmark.i3_30r3_confirmation_generator import (
        generate_confirmation_benchmark, compute_confirmation_benchmark_hash,
    )
    import numpy, sklearn, joblib, pytest

    tasks = generate_confirmation_benchmark(seed=43291)
    bench_hash = compute_confirmation_benchmark_hash(tasks)

    # Use relative paths from REPO_ROOT (same as development runner)
    import os
    os.chdir(REPO_ROOT)

    manifest = {
        "experiment": "I3.30R3-CONFIRMATION",
        "status": "FROZEN",
        "source_commit": "current",
        "benchmark_seed": 43291,
        "task_count": len(tasks),
        "trajectory_count": len(tasks) * 2,
        "arms": ["v3_shadow", "v3_hard"],
        "arm_count": 2,

        # === Model artifacts (unchanged from development) ===
        "Q_V3R_model_sha256": sha256_file("experiments/i3_30r/Q_V3R2_A.pkl"),
        "Q_V3R_schema_sha256": sha256_file("experiments/i3_30r/v3r2_feature_schema.json"),
        "Q_V1_model_sha256": sha256_file("experiments/i3_5/pinned_policy/frozen_estimators/QCAUSAL_gbt.pkl"),
        "Q_V1_schema_sha256": sha256_file("experiments/i3_5/pinned_policy/frozen_estimators/feature_schema.json"),

        # === Epistemic / authority modules ===
        "topology_sha256": sha256_file("daph/epistemic/topology.py"),
        "v3_features_sha256": sha256_file("daph/epistemic/v3_features.py"),
        "authority_policy_v2_sha256": sha256_file("daph/authority/policy.py"),
        "authority_policy_v3_sha256": sha256_file("daph/authority/policy_v3.py"),
        "authority_isolation_sha256": sha256_file("daph/authority/isolation.py"),
        "utility_config_sha256": sha256_file("configs/v2b_i3_1_utility_v1.json"),

        # === Runner / evaluator / intervention ===
        "runner_sha256": sha256_file("scripts/run_i3_30r3_confirmation.py"),
        "evaluator_sha256": sha256_file("scripts/evaluate_i3_30r3_authority_isolation.py"),
        "checkpoint_sha256": sha256_file("daph/intervention/checkpoint.py"),
        "restore_sha256": sha256_file("daph/intervention/restore.py"),

        # === Schema / grammar / backend / snapshot ===
        "schema_grammar_sha256": sha256_file("scripts/r2_schema.py"),
        "r2_allowed_actions_sha256": sha256_file("scripts/r2_allowed_actions.py"),
        "i3_7e_snapshot_builder_sha256": sha256_file("scripts/run_i3_7e_compact_governor.py"),
        "model_backend_sha256": sha256_file("hrm_adaptive_memory/executive/model_backend.py"),

        # === Benchmark generator ===
        "confirmation_generator_sha256": sha256_file("hrm_adaptive_memory/executive/evidence_benchmark/i3_30r3_confirmation_generator.py"),

        # === Model weights ===
        "qwen_gguf_sha256": sha256_file(gguf_path),
        "qwen_gguf_path": gguf_path,

        # === Dependency versions ===
        "llama_cpp_python_version": "0.3.7",
        "numpy_version": numpy.__version__,
        "scikit_learn_version": sklearn.__version__,
        "joblib_version": joblib.__version__,
        "pytest_version": pytest.__version__,

        # === Runtime parameters ===
        "runtime_n_ctx": 4096,
        "runtime_n_gpu_layers": -1,
        "runtime_temperature": 0.0,
        "runtime_max_tokens": 256,

        # === Benchmark ===
        "benchmark_sha256": bench_hash,
        "benchmark_strata": {
            "D1": 80, "D2": 80, "D3": 80, "D4": 80, "D5": 80,
        },

        # === Frozen constants (unchanged from development) ===
        "authority_threshold": 5.0,
        "near_optimal_epsilon": 3.0,
        "v3_frozen_rule": "A2AD_V3_POSITIVE_CERTIFICATE",

        # === Gates (confirmation-specific — G5 is STRICTER) ===
        "gates": {
            "G1": "treatment_purity",
            "G2": "authority_breaks == 0",
            "G3": "false_answer_authority == 0",
            "G4": "false_defer_authority == 0",
            "G5": "ci_95_lower > 0",
            "G6": "rescues > breaks",
            "G7": "effective_answer_interventions > 0",
            "G8": "defer_coverage (informative, not required)",
            "G9": "semantic_disagreements == 0",
            "G10": "reliability_errors == 0",
            "G11": "manifest_mismatches == 0",
            "G12": "complete_receipt_rate == 1.0",
        },
    }

    return manifest


def main():
    parser = argparse.ArgumentParser(description="I3.30R3 Confirmation runner")
    parser.add_argument("--gguf-path", required=True)
    parser.add_argument("--output-dir", default="experiments/i3_30r3/confirmation")
    parser.add_argument("--freeze-manifest-only", action="store_true",
                        help="Only create the frozen manifest, don't run trajectories")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing trajectory files")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute and freeze manifest (write-once, fail-closed)
    manifest = compute_confirmation_manifest(args.gguf_path)
    manifest_path = output_dir / "frozen_manifest.json"

    if manifest_path.exists():
        with open(manifest_path) as f:
            existing = json.load(f)
        mismatch = False
        for key in sorted(set(list(existing.keys()) + list(manifest.keys()))):
            if key in ("frozen_at", "timestamp", "source_commit"):
                continue
            if existing.get(key) != manifest.get(key):
                print(f"  MISMATCH: {key}")
                print(f"    existing: {str(existing.get(key))[:40]}")
                print(f"    computed: {str(manifest.get(key))[:40]}")
                mismatch = True
        if mismatch:
            print(f"\n*** FROZEN MANIFEST MISMATCH — ABORTING ***")
            print(f"The frozen identity has been violated. The execution cannot proceed.")
            print(f"To run a new experiment, use a fresh output directory.")
            sys.exit(1)
        print(f"Manifest verified (write-once): {manifest_path}")
        manifest = existing
    else:
        if not args.freeze_manifest_only:
            print(f"\n*** NO FROZEN MANIFEST — ABORTING ***")
            print(f"Run with --freeze-manifest-only first.")
            sys.exit(1)
        # Add frozen_at timestamp
        manifest["frozen_at"] = __import__("datetime").datetime.now().isoformat()
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        # Make manifest read-only at filesystem level
        manifest_path.chmod(0o444)
        print(f"Manifest created (freeze-only): {manifest_path}")
        print(f"  File permissions set to read-only (0o444)")

    print(f"  benchmark_seed: {manifest['benchmark_seed']}")
    print(f"  task_count: {manifest['task_count']}")
    print(f"  trajectory_count: {manifest['trajectory_count']}")
    print(f"  arms: {manifest['arms']}")
    print(f"  benchmark_sha256: {manifest['benchmark_sha256'][:16]}...")

    if args.freeze_manifest_only:
        print("\nManifest frozen. Run without --freeze-manifest-only to execute.")
        return

    # Import the development runner's components
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

    from hrm_adaptive_memory.executive.evidence_benchmark.i3_30r3_confirmation_generator import (
        generate_confirmation_benchmark, get_confirmation_budget_for_task,
        _CONFIRMATION_BUDGET_OVERRIDES,
    )
    # Register confirmation budgets in the I3.29 generator's override dict
    from hrm_adaptive_memory.executive.evidence_benchmark.i3_29_safety_generator import (
        _BUDGET_OVERRIDES as DEV_BUDGET_OVERRIDES,
    )
    from run_i3_30r3_authority_isolation import (
        run_trajectory, ArmMode, QModelV3R,
    )
    from run_i3_29_live_safety import QModelV1
    from hrm_adaptive_memory.executive.model_backend import R2DirectLlamaBackend
    from hrm_adaptive_memory.executive.metareasoning_utility import MetareasoningUtility
    import run_i3_7e_compact_governor as i3_7e

    # Generate confirmation tasks
    tasks = generate_confirmation_benchmark(seed=43291)

    # Register confirmation budgets in the dev generator's override dict
    # so get_budget_for_task() finds them
    DEV_BUDGET_OVERRIDES.update(_CONFIRMATION_BUDGET_OVERRIDES)

    strata_counts = {}
    for t in tasks:
        for s in ["d1", "d2", "d3", "d4", "d5"]:
            if f"_{s}_" in t.task_id:
                strata_counts[s.upper()] = strata_counts.get(s.upper(), 0) + 1
    print(f"\nConfirmation tasks: {len(tasks)} ({strata_counts})")
    print(f"Arms: V3_SHADOW, V3_HARD (no V1 — primary confirmation)")
    print(f"Total trajectories: {len(tasks) * 2}")

    # Load models (same as development — unchanged)
    q_v1 = QModelV1.load(
        REPO_ROOT / "experiments/i3_5/pinned_policy/frozen_estimators/QCAUSAL_gbt.pkl",
        REPO_ROOT / "experiments/i3_5/pinned_policy/frozen_estimators/feature_schema.json",
    )
    q_v3r = QModelV3R.load(
        REPO_ROOT / "experiments/i3_30r/Q_V3R2_A.pkl",
        REPO_ROOT / "experiments/i3_30r/v3r2_feature_schema.json",
    )

    # Load backend (same GGUF, same parameters)
    backend = R2DirectLlamaBackend(
        model_path=args.gguf_path,
        n_ctx=4096,
        n_gpu_layers=-1,
    )

    # Load utility
    utility = MetareasoningUtility.from_file(
        REPO_ROOT / "configs" / "v2b_i3_1_utility_v1.json")

    # Trajectory output paths
    arm_files = {}
    for arm in [ArmMode.V3_SHADOW, ArmMode.V3_HARD]:
        arm_files[arm] = output_dir / f"trajectories_{arm.value}.jsonl"

    error_path = output_dir / "errors.jsonl"
    auth_events_path = output_dir / "authority_events.jsonl"

    # Resume logic
    done = {arm: set() for arm in arm_files}
    if args.resume:
        for arm, path in arm_files.items():
            if path.exists():
                with open(path) as f:
                    for line in f:
                        r = json.loads(line)
                        done[arm].add(r["task_id"])
        print(f"  Resuming: " + ", ".join(f"{arm.value}={len(d)}" for arm, d in done.items()))

    traj_files = {arm: open(path, "a") for arm, path in arm_files.items()}
    error_file = open(error_path, "a")
    auth_events_file = open(auth_events_path, "a")

    total = len(tasks) * 2
    completed = 0
    errors = 0

    for task in tasks:
        d2_pre = "_d2_" in task.task_id

        for arm in [ArmMode.V3_SHADOW, ArmMode.V3_HARD]:
            if task.task_id in done[arm]:
                completed += 1
                continue

            try:
                result = run_trajectory(
                    task, backend, i3_7e, utility,
                    q_v1, q_v3r, arm,
                    d2_pre_verify=d2_pre,
                )

                traj_files[arm].write(json.dumps(result, default=str) + "\n")
                traj_files[arm].flush()

                for evt in result.get("authority_events", []):
                    auth_events_file.write(json.dumps(evt, default=str) + "\n")
                auth_events_file.flush()

                done[arm].add(task.task_id)
                completed += 1

                if completed % 20 == 0:
                    traj = result
                    print(f"  [{completed}/{total}] {task.task_id} {arm.value}: "
                          f"success={traj.get('success', '?')} util={traj.get('realized_utility', 0):.1f}")

            except Exception as e:
                error_file.write(json.dumps({
                    "task_id": task.task_id,
                    "arm": arm.value,
                    "error": str(e),
                }) + "\n")
                error_file.flush()
                errors += 1
                completed += 1

    for f in traj_files.values():
        f.close()
    error_file.close()
    auth_events_file.close()

    print(f"\nDone. {completed}/{total} trajectories completed.")
    print(f"  v3_shadow: {output_dir}/trajectories_v3_shadow.jsonl")
    print(f"  v3_hard: {output_dir}/trajectories_v3_hard.jsonl")
    print(f"  Authority events: {output_dir}/authority_events.jsonl")
    print(f"  Errors: {output_dir}/errors.jsonl")
    print(f"  Manifest: {output_dir}/frozen_manifest.json")


if __name__ == "__main__":
    main()
