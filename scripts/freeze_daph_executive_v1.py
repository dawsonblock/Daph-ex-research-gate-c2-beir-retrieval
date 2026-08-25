#!/usr/bin/env python3
"""Freeze DAPH_EXECUTIVE_V1: QCAUSAL_V1 + I2 epsilon interface.

Binds all artifact identities into a single manifest:
  - Q model SHA (QCAUSAL_gbt.pkl)
  - training/intervention dataset SHA (pinned_causal_actions_v1.jsonl)
  - feature-schema SHA (feature_schema.json)
  - epsilon (3.0)
  - I2 packet-builder SHA (run_i3_5_interface_ablation.py)
  - phase classifier SHA (classify_phase_simple in runner)
  - Qwen backend SHA
  - utility SHA
  - model SHA
  - benchmark SHA
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main():
    timestamp = datetime.now(timezone.utc).isoformat()

    est_dir = REPO_ROOT / "experiments/i3_5/pinned_policy/frozen_estimators"
    pinned_dir = REPO_ROOT / "experiments/i3_5/pinned_policy"
    config_dir = REPO_ROOT / "configs"
    scripts_dir = REPO_ROOT / "scripts"

    # ================================================================
    # Compute all SHAs
    # ================================================================

    # Q model
    qcausal_pkl_sha = sha256_file(est_dir / "QCAUSAL_gbt.pkl")

    # Training/intervention dataset
    causal_dataset_sha = sha256_file(pinned_dir / "pinned_causal_actions_v1.jsonl")

    # Feature schema
    feature_schema_sha = sha256_file(est_dir / "feature_schema.json")
    with open(est_dir / "feature_schema.json") as f:
        feature_schema = json.load(f)
    feature_keys = feature_schema["feature_keys"]

    # I2 packet-builder (the interface ablation runner contains the I2 logic)
    interface_runner_sha = sha256_file(scripts_dir / "run_i3_5_interface_ablation.py")

    # Phase classifier (classify_phase_simple is in the runner)
    # We also need the i3_7e module for the base packet builder
    i3_7e_sha = sha256_file(scripts_dir / "run_i3_7e_compact_governor.py")

    # R2 schema and allowed actions
    r2_schema_sha = sha256_file(scripts_dir / "r2_schema.py")
    r2_allowed_sha = sha256_file(scripts_dir / "r2_allowed_actions.py")

    # Utility
    utility_path = config_dir / "v2b_i3_1_utility_v1.json"
    utility_sha = sha256_file(utility_path)

    # Model
    model_path = Path("/Users/dawsonblock/Downloads/qwen_gguf/Qwen2.5-7B-Instruct-Q4_K_M.gguf")
    model_sha = sha256_file(model_path) if model_path.exists() else "UNKNOWN"

    # Backend
    backend_sha = sha256_file(REPO_ROOT / "hrm_adaptive_memory/executive/model_backend.py")
    decoder_sha = sha256_file(REPO_ROOT / "hrm_adaptive_memory/executive/model_decoder.py")

    # Benchmark generator
    benchmark_gen_sha = sha256_file(
        REPO_ROOT / "hrm_adaptive_memory/executive/evidence_benchmark/i3_5_state_discrimination_generator.py")

    # Evidence snapshot/executor
    evidence_bench_sha = sha256_file(
        REPO_ROOT / "hrm_adaptive_memory/executive/evidence_benchmark/__init__.py")
    executor_sha = sha256_file(
        REPO_ROOT / "hrm_adaptive_memory/executive/evidence_benchmark/executor.py")

    # Checkpoint (state feature computation)
    checkpoint_sha = sha256_file(REPO_ROOT / "daph/intervention/checkpoint.py")

    # Resources
    resources_sha = sha256_file(REPO_ROOT / "hrm_adaptive_memory/executive/resources.py")

    # Cognitive control core
    cognitive_core_sha = sha256_file(
        REPO_ROOT / "hrm_adaptive_memory/cognitive_control/core.py")

    # ================================================================
    # Build the frozen executive manifest
    # ================================================================

    manifest = {
        "executive_name": "DAPH_EXECUTIVE_V1",
        "version": "1.0",
        "timestamp": timestamp,
        "description": (
            "DAPH cognitive-control executive V1. "
            "Frozen QCAUSAL_V1 estimator + I2 epsilon near-optimal set interface. "
            "The estimator predicts Q^pi_Qwen(s,a) from 35 state features. "
            "The interface exposes the near-optimal action set A_epsilon(s) = "
            "{a: Q(s,a) >= Q_max(s) - epsilon} without numerical values, "
            "letting the base model decide among near-equivalent actions."
        ),
        "components": {
            "q_estimator": {
                "name": "QCAUSAL_V1",
                "type": "GradientBoostedTreesRegressor",
                "artifact_path": "experiments/i3_5/pinned_policy/frozen_estimators/QCAUSAL_gbt.pkl",
                "sha256": qcausal_pkl_sha,
                "n_features": len(feature_keys),
                "training_data": {
                    "dataset_path": "experiments/i3_5/pinned_policy/pinned_causal_actions_v1.jsonl",
                    "dataset_sha256": causal_dataset_sha,
                    "n_records": 1056,
                    "collection_method": "pinned_policy_forced_action",
                    "policy": "Qwen2.5-7B-Instruct-Q4_K_M at temperature=0",
                },
            },
            "feature_schema": {
                "path": "experiments/i3_5/pinned_policy/frozen_estimators/feature_schema.json",
                "sha256": feature_schema_sha,
                "feature_keys": feature_keys,
            },
            "interface": {
                "name": "I2",
                "type": "epsilon_near_optimal_set",
                "epsilon": 3.0,
                "description": (
                    "Exposes near_optimal_actions (actions with Q >= Q_max - epsilon) "
                    "and lower_value_actions (all others). No numerical values. "
                    "No ranking. The LLM chooses freely from the near-optimal set."
                ),
                "implementation": {
                    "packet_builder_path": "scripts/run_i3_5_interface_ablation.py",
                    "packet_builder_sha256": interface_runner_sha,
                    "class": "InterfaceVariant",
                    "method": "build_packet_fields",
                    "variant": "I2",
                },
            },
            "phase_classifier": {
                "name": "classify_phase_simple",
                "implementation": "scripts/run_i3_5_interface_ablation.py",
                "sha256": interface_runner_sha,
                "phases": ["T2", "READY", "EXPLORE", "DISCRIMINATE"],
            },
            "base_packet_builder": {
                "name": "i3_7e MDSG state with affordances",
                "implementation": "scripts/run_i3_7e_compact_governor.py",
                "sha256": i3_7e_sha,
            },
            "action_schema": {
                "schema_builder": "scripts/r2_schema.py",
                "schema_sha256": r2_schema_sha,
                "allowed_actions": "scripts/r2_allowed_actions.py",
                "allowed_actions_sha256": r2_allowed_sha,
            },
            "utility": {
                "path": "configs/v2b_i3_1_utility_v1.json",
                "sha256": utility_sha,
            },
            "model": {
                "name": "Qwen2.5-7B-Instruct-Q4_K_M",
                "path": str(model_path),
                "sha256": model_sha,
                "quantization": "Q4_K_M",
                "format": "GGUF",
            },
            "backend": {
                "class": "R2DirectLlamaBackend",
                "path": "hrm_adaptive_memory/executive/model_backend.py",
                "sha256": backend_sha,
                "decoder_path": "hrm_adaptive_memory/executive/model_decoder.py",
                "decoder_sha256": decoder_sha,
                "temperature": 0.0,
            },
            "benchmark": {
                "generator": "i3_5_state_discrimination_generator.py",
                "generator_sha256": benchmark_gen_sha,
                "seed": 9137,
                "n_per_subtype": 24,
                "n_per_two_live_subtype": 20,
            },
            "evidence_runtime": {
                "evidence_benchmark_sha256": evidence_bench_sha,
                "executor_sha256": executor_sha,
            },
            "state_features": {
                "checkpoint_path": "daph/intervention/checkpoint.py",
                "checkpoint_sha256": checkpoint_sha,
            },
            "resources": {
                "path": "hrm_adaptive_memory/executive/resources.py",
                "sha256": resources_sha,
                "budget": {
                    "max_executive_steps": 10,
                    "max_retrieval_calls": 3,
                    "max_search_calls": 2,
                    "max_verification_calls": 5,
                },
            },
            "cognitive_control": {
                "path": "hrm_adaptive_memory/cognitive_control/core.py",
                "sha256": cognitive_core_sha,
            },
        },
        "invariants": [
            "QCAUSAL_V1 is frozen — no retraining allowed",
            "I2 interface is frozen — no packet representation changes",
            "epsilon=3.0 is frozen",
            "Feature schema is frozen (35 features)",
            "Model identity is frozen (Qwen2.5-7B-Instruct-Q4_K_M)",
            "Backend is frozen (temperature=0, deterministic)",
            "Utility function is frozen",
            "Action schema is frozen",
            "Phase classifier is frozen",
            "No .fit() calls in the live execution path",
            "Only serialized estimators may be loaded",
        ],
        "control_law": {
            "formula": "A_epsilon(s) = {a : Q(s,a) >= Q_max(s) - epsilon}",
            "epsilon": 3.0,
            "packet_fields": {
                "near_optimal_actions": "list of actions in A_epsilon(s)",
                "lower_value_actions": "list of actions not in A_epsilon(s)",
                "epistemic_phase": "phase classification string",
            },
            "no_numeric_values": True,
            "no_ranking": True,
        },
        "validation": {
            "phase_19_six_arm": {
                "QCAUSAL_vs_B0": "PASS (+7.02, CI=[0.12, 13.96])",
                "QCAUSAL_vs_QOBS": "FAIL (-11.91, CI=[-16.88, -7.47])",
                "promotion_gate": "PASS (4/4 required gates)",
            },
            "phase_21_interface_ablation": {
                "I2_vs_C0": "PASS (+0.23, CI=[0.12, 0.37])",
                "I2_vs_I0": "PASS (+17.86, CI=[12.23, 23.84])",
                "ol_retrieve_E_retrieve": "1.00 (target: ~1, was 3.00 with I0)",
                "ol_retrieve_success": "100% (was 75% with I0)",
                "overall_success": "100%",
            },
        },
    }

    # ================================================================
    # Write the manifest
    # ================================================================

    output_dir = REPO_ROOT / "experiments/i3_5/executive_v1"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "DAPH_EXECUTIVE_V1.json"

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    manifest_sha = sha256_file(manifest_path)

    print(f"DAPH_EXECUTIVE_V1 frozen")
    print(f"  Manifest: {manifest_path}")
    print(f"  Manifest SHA256: {manifest_sha}")
    print()
    print(f"  Q model SHA:      {qcausal_pkl_sha[:16]}...")
    print(f"  Dataset SHA:      {causal_dataset_sha[:16]}...")
    print(f"  Feature schema:   {feature_schema_sha[:16]}...")
    print(f"  Interface (I2):   {interface_runner_sha[:16]}...")
    print(f"  Model SHA:        {model_sha[:16]}...")
    print(f"  Utility SHA:      {utility_sha[:16]}...")
    print(f"  Backend SHA:      {backend_sha[:16]}...")
    print(f"  Benchmark SHA:    {benchmark_gen_sha[:16]}...")
    print()
    print(f"  epsilon: 3.0")
    print(f"  n_features: {len(feature_keys)}")
    print(f"  n_training_records: 1056")
    print(f"  control_law: A_epsilon(s) = {{a : Q(s,a) >= Q_max(s) - epsilon}}")
    print(f"  no_numeric_values: True")
    print(f"  no_ranking: True")

    # Also write a compact binding file
    binding = {
        "executive_name": "DAPH_EXECUTIVE_V1",
        "manifest_sha256": manifest_sha,
        "qcausal_model_sha256": qcausal_pkl_sha,
        "causal_dataset_sha256": causal_dataset_sha,
        "feature_schema_sha256": feature_schema_sha,
        "interface_sha256": interface_runner_sha,
        "model_sha256": model_sha,
        "utility_sha256": utility_sha,
        "backend_sha256": backend_sha,
        "epsilon": 3.0,
        "interface_variant": "I2",
    }
    binding_path = output_dir / "binding.json"
    with open(binding_path, "w") as f:
        json.dump(binding, f, indent=2, sort_keys=True)
    print(f"\n  Binding: {binding_path}")


if __name__ == "__main__":
    main()
