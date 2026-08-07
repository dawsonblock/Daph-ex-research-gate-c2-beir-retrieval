#!/usr/bin/env python3
"""Phase 13: Qualify model determinism separately.

Only after packet hashes are stable should you test HRM.

This script:
1. Selects N representative frozen prompts.
2. Runs each several times with deterministic decoding settings.
3. Records prompt hash, generation config, output text, output token IDs, output hash.
4. Quantifies: same prompt → same tokens?

If output varies, that becomes a distinct model-runtime nondeterminism issue
rather than being confused with S2c.

Usage:
    python scripts/c4_model_determinism.py [--n-prompts 20] [--n-repeats 3]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def qualify_model_determinism(n_prompts: int = 20, n_repeats: int = 3):
    """Test HRM determinism on frozen prompts."""
    from scripts.run_gate_c4 import _load_hrm, _run_hrm

    frozen_dir = ROOT / "evidence/gate_c4/frozen_packets/development"
    if not frozen_dir.exists():
        print("ERROR: Frozen packets not found. Run c4_freeze_packets.py first.")
        sys.exit(1)

    # Collect frozen prompts from C4_4
    arm_dir = frozen_dir / "C4_4"
    if not arm_dir.exists():
        print(f"ERROR: {arm_dir} not found")
        sys.exit(1)

    prompt_paths = sorted(arm_dir.glob("*/prompt.txt"))[:n_prompts]
    print(f"=== Phase 13: Model Determinism Qualification ===")
    print(f"Prompts: {len(prompt_paths)}")
    print(f"Repeats: {n_repeats}")
    print()

    # Load HRM with deterministic settings
    print("Loading HRM model...")
    adapter, condition = _load_hrm()
    print(f"  Model: {adapter.spec.model_id}")
    print(f"  Condition: {condition}")
    print()

    # Run each prompt multiple times
    results = []
    all_deterministic = True

    for i, prompt_path in enumerate(prompt_paths):
        tid = prompt_path.parent.name
        prompt_text = prompt_path.read_text()
        prompt_hash = hashlib.sha256(prompt_text.encode()).hexdigest()

        outputs = []
        for repeat in range(n_repeats):
            hrm_result = _run_hrm(adapter, condition, prompt_text)
            output_hash = hashlib.sha256(hrm_result.output.encode()).hexdigest()
            outputs.append({
                "repeat": repeat,
                "output": hrm_result.output,
                "output_hash": output_hash,
            })

        # Check if all outputs are identical
        output_hashes = [o["output_hash"] for o in outputs]
        is_deterministic = len(set(output_hashes)) == 1

        if not is_deterministic:
            all_deterministic = False
            print(f"  [{i+1}/{len(prompt_paths)}] {tid}: NONDETERMINISTIC ({len(set(output_hashes))} unique outputs)")
        else:
            print(f"  [{i+1}/{len(prompt_paths)}] {tid}: deterministic")

        results.append({
            "task_id": tid,
            "prompt_hash": prompt_hash,
            "is_deterministic": is_deterministic,
            "outputs": outputs,
        })

    # Summary
    n_det = sum(1 for r in results if r["is_deterministic"])
    n_non = len(results) - n_det
    print(f"\n=== Summary ===")
    print(f"  Prompts tested: {len(results)}")
    print(f"  Deterministic:  {n_det}/{len(results)}")
    print(f"  Nondeterministic: {n_non}/{len(results)}")
    print(f"  Result: {'PASS' if all_deterministic else 'FAIL — model-runtime nondeterminism detected'}")

    # Write receipt
    receipt = {
        "schema_version": "c4-model-determinism-v1",
        "n_prompts": len(results),
        "n_repeats": n_repeats,
        "n_deterministic": n_det,
        "n_nondeterministic": n_non,
        "result": "PASS" if all_deterministic else "FAIL",
        "model_id": adapter.spec.model_id,
        "results": [
            {
                "task_id": r["task_id"],
                "prompt_hash": r["prompt_hash"],
                "is_deterministic": r["is_deterministic"],
                "output_hashes": [o["output_hash"] for o in r["outputs"]],
            }
            for r in results
        ],
    }

    receipt_path = ROOT / "evidence/gate_c4/model_determinism.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"\n  Receipt: {receipt_path}")


def main():
    parser = argparse.ArgumentParser(description="C4 model determinism qualification")
    parser.add_argument("--n-prompts", type=int, default=20)
    parser.add_argument("--n-repeats", type=int, default=3)
    args = parser.parse_args()

    qualify_model_determinism(
        n_prompts=args.n_prompts,
        n_repeats=args.n_repeats,
    )


if __name__ == "__main__":
    main()
