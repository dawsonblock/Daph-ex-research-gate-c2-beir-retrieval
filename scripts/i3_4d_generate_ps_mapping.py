#!/usr/bin/env python3
"""DAPH I3.4d — Generate frozen PS mapping with SHA-256 receipt.

Pre-computes the PS (shuffled-value) permutation for all phases using
deterministic SHA-256 seeding, serializes it to JSON, and writes a
receipt with the mapping SHA-256.

This replaces the old dynamic shuffle (which used Python's process-randomized
hash()) with a frozen, process-independent mapping that is identical across
all Colab sessions and resume cycles.

Usage:
    PYTHONPATH=scripts:. python3 scripts/i3_4d_generate_ps_mapping.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from daph.executive.packet_builder import (
    generate_frozen_ps_mapping,
    save_frozen_ps_mapping,
    stable_shuffle_seed,
)
from daph.value.empirical import PhaseActionTable
from daph.phase.ontology import ALL_PHASES


def main():
    b1_path = REPO_ROOT / "experiments/i3_4/value/frozen_b1_table.json"
    ps_mapping_path = REPO_ROOT / "experiments/i3_4/value/ps_frozen_mapping.json"
    ps_receipt_path = REPO_ROOT / "experiments/i3_4/value/ps_mapping_receipt.json"

    shuffle_seed = 42  # Must match the experiment's --shuffle-seed

    print("=" * 60)
    print("I3.4d — Generate Frozen PS Mapping")
    print("=" * 60)

    # Load frozen B1
    print(f"\nLoading frozen B1 from: {b1_path}")
    b1 = PhaseActionTable.load(b1_path)
    b1_sha = b1.sha256()
    print(f"  B1 model SHA256: {b1_sha}")
    print(f"  Entries: {len(b1._values)}")

    # Generate frozen PS mapping
    print(f"\nGenerating frozen PS mapping (shuffle_seed={shuffle_seed})...")
    actions = ["ANSWER", "VERIFY", "DEFER", "SEARCH_MORE", "RETRIEVE"]
    mapping = generate_frozen_ps_mapping(b1, shuffle_seed=shuffle_seed, actions=actions)

    print(f"\nFrozen PS mapping:")
    for phase in ALL_PHASES:
        pval = phase.value
        if pval in mapping:
            print(f"  {pval}:")
            for a, v in sorted(mapping[pval].items()):
                print(f"    {a:15s} → {v:.6f}")

    # Verify deterministic seed
    print(f"\nDeterministic seed verification:")
    for phase in ALL_PHASES:
        seed = stable_shuffle_seed(phase.value, shuffle_seed)
        print(f"  {phase.value:30s} seed={seed}")

    # Save mapping
    ps_sha = save_frozen_ps_mapping(mapping, ps_mapping_path)
    print(f"\nFrozen PS mapping written to: {ps_mapping_path}")
    print(f"  PS mapping SHA256: {ps_sha}")

    # Write receipt
    receipt = {
        "ps_mapping_sha256": ps_sha,
        "b1_model_sha256": b1_sha,
        "shuffle_seed": shuffle_seed,
        "actions": actions,
        "phases": [p.value for p in ALL_PHASES],
        "seed_algorithm": "SHA-256(phase|shuffle_seed) → first 8 bytes as uint64",
        "note": "Replaces Python hash() which is process-randomized via PYTHONHASHSEED",
    }
    with open(ps_receipt_path, "w") as f:
        json.dump(receipt, f, indent=2, sort_keys=True)
    print(f"\nPS mapping receipt written to: {ps_receipt_path}")

    print("\n" + "=" * 60)
    print("DONE — Frozen PS mapping generated")
    print("=" * 60)
    print(f"\nNext: run the corrective P2 vs PS_FIXED experiment with:")
    print(f"  --ps-mapping {ps_mapping_path}")


if __name__ == "__main__":
    main()
