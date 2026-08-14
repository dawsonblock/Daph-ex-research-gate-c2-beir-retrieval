"""Generate and bind the I3.4 controller identity.

This script computes SHA-256 hashes of every component that affects
model-output or evaluation reproducibility and writes a frozen
``controller_identity.json`` to the I3.4 experiment directory.

Usage:
    python scripts/generate_v2b_i3_4_controller_identity.py

The DeepSeek API key is NOT required for identity generation; only the
model name and provider are bound.  The system_fingerprint is captured
per-call during actual model runs and stored in the model output store.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from hrm_adaptive_memory.executive.controller_identity import (
    build_identity, save_identity)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    policy_path = ROOT / "configs/v2b_i3_policy_v1.json"
    utility_path = ROOT / "configs/v2b_i3_1_utility_v1.json"
    masks_path = ROOT / "configs/v2b_i3_observation_masks_v1.json"
    manifest_path = ROOT / "experiments/v2b/benchmark/v2b_i3_benchmark_manifest_v1.json"

    identity = build_identity(
        model_name="deepseek-v4-flash",
        model_provider="deepseek",
        model_revision=None,
        system_fingerprint=None,
        temperature=0.0,
        max_tokens=2048,
        policy_path=str(policy_path.relative_to(ROOT)),
        policy_sha256=_file_sha256(policy_path),
        utility_path=str(utility_path.relative_to(ROOT)),
        utility_sha256=_file_sha256(utility_path),
        observation_masks_path=str(masks_path.relative_to(ROOT)),
        observation_masks_sha256=_file_sha256(masks_path),
        benchmark_manifest_path=str(manifest_path.relative_to(ROOT)),
        benchmark_manifest_sha256=_file_sha256(manifest_path),
        python_version=sys.version.split()[0],
        platform=platform.platform(),
    )

    output_path = ROOT / "experiments/v2b_i3_4/manifests/v2b_i3_4_controller_identity_v1.json"
    identity_hash = save_identity(identity, output_path)
    print(f"Controller identity written to: {output_path}")
    print(f"Identity SHA-256: {identity_hash}")
    print(f"Model: deepseek-v4-flash (deepseek)")
    print(f"Prompt SHA-256: {identity.system_prompt['sha256']}")
    print(f"Serializer SHA-256: {identity.serializer['source_sha256']}")
    print(f"Decoder SHA-256: {identity.decoder['source_sha256']}")
    print(f"Controller code SHA-256: {identity.controller_code['source_sha256']}")


if __name__ == "__main__":
    main()
