#!/usr/bin/env python3
"""DAPH I3.4e — Freeze control decomposition artifacts.

Generates all frozen artifacts needed for the I3.4e control decomposition:

1. B0 — Global action prior E[U | a] from R2 transitions (no phase info)
2. 16 frozen PS mappings (PS01-PS16) using SHA-256 deterministic seeding
   with distinct shuffle seeds. None are chosen by performance.
3. CONST mapping — every action gets the same value (0.5)
4. DEFER-HEURISTIC mapping — DEFER=1.0, all others=0.5
5. A receipt binding all SHAs together.

All mappings preserve the B1 value multiset within each phase (where
applicable) so that the packet structure, numeric distribution, and
field count are held constant across arms. The only thing that changes
is the action→value association.

For B0, CONST, and DEFER-HEURISTIC, the values are not constrained to
the B1 multiset — they use their own value definitions. This is by
design: these arms test different hypotheses about what information
the model uses.

Usage:
    PYTHONPATH=scripts:. python3 scripts/i3_4e_freeze_controls.py
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
from daph.value.empirical import GlobalActionMean, PhaseActionTable
from daph.value.dataset import load_transitions, get_action_value_target
from daph.phase.ontology import ALL_PHASES


ACTIONS = ["ANSWER", "VERIFY", "DEFER", "SEARCH_MORE", "RETRIEVE"]


def freeze_b0() -> tuple[dict, str, str]:
    """Compute B0 global action prior from R2 transitions.

    Returns (mapping, model_sha, file_sha) where mapping is
    {action: value} (same for all phases).
    """
    transitions_path = REPO_ROOT / "experiments/i3_4/datasets/transitions_r2_dev_v2.jsonl"
    with open(transitions_path) as f:
        transitions = [json.loads(line) for line in f]

    b0 = GlobalActionMean()
    b0.fit(transitions, get_action_value_target)

    # B0 is phase-independent — same values for every phase
    mapping: dict[str, dict[str, float]] = {}
    for phase in ALL_PHASES:
        mapping[phase.value] = {
            a: round(b0.predict(phase.value, a, {}), 6) for a in ACTIONS
        }

    # Compute model SHA from the values
    model_sha = hashlib.sha256(
        json.dumps({a: round(v, 6) for a, v in sorted(b0._values.items())},
                   sort_keys=True).encode()
    ).hexdigest()

    # Save
    output_path = REPO_ROOT / "experiments/i3_4/value/b0_global_prior.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "model": "B0_global_action_mean",
            "values": {a: round(v, 6) for a, v in sorted(b0._values.items())},
            "counts": dict(b0._counts),
            "mapping": mapping,
        }, f, indent=2, sort_keys=True)
    file_sha = hashlib.sha256(output_path.read_bytes()).hexdigest()

    print(f"B0 global action prior:")
    for a in ACTIONS:
        print(f"  {a:15s} → {b0._values.get(a, 0.0):.6f}  (n={b0._counts.get(a, 0)})")
    print(f"  Model SHA: {model_sha}")
    print(f"  File SHA:  {file_sha}")
    print(f"  Written to: {output_path}")

    return mapping, model_sha, file_sha


def freeze_const_mapping() -> tuple[dict, str, str]:
    """CONST mapping — every action gets 0.5 in every phase.

    Tests whether merely adding an action_value_estimates object alters
    behavior, independent of any ranking information.
    """
    mapping: dict[str, dict[str, float]] = {}
    for phase in ALL_PHASES:
        mapping[phase.value] = {a: 0.5 for a in ACTIONS}

    model_sha = hashlib.sha256(
        json.dumps(mapping, sort_keys=True).encode()
    ).hexdigest()

    output_path = REPO_ROOT / "experiments/i3_4/value/const_mapping.json"
    with open(output_path, "w") as f:
        json.dump({
            "model": "CONST_uniform",
            "value": 0.5,
            "mapping": mapping,
        }, f, indent=2, sort_keys=True)
    file_sha = hashlib.sha256(output_path.read_bytes()).hexdigest()

    print(f"\nCONST mapping: all actions = 0.5")
    print(f"  Model SHA: {model_sha}")
    print(f"  File SHA:  {file_sha}")
    print(f"  Written to: {output_path}")

    return mapping, model_sha, file_sha


def freeze_defer_heuristic() -> tuple[dict, str, str]:
    """DEFER-HEURISTIC mapping — DEFER=1.0, all others=0.5.

    Tests whether a simple DEFER bias explains the PSF improvement.
    If this beats everything, the benchmark has a strong DEFER prior.
    """
    mapping: dict[str, dict[str, float]] = {}
    for phase in ALL_PHASES:
        mapping[phase.value] = {a: (1.0 if a == "DEFER" else 0.5) for a in ACTIONS}

    model_sha = hashlib.sha256(
        json.dumps(mapping, sort_keys=True).encode()
    ).hexdigest()

    output_path = REPO_ROOT / "experiments/i3_4/value/defer_heuristic_mapping.json"
    with open(output_path, "w") as f:
        json.dump({
            "model": "DEFER_HEURISTIC",
            "defer_value": 1.0,
            "other_value": 0.5,
            "mapping": mapping,
        }, f, indent=2, sort_keys=True)
    file_sha = hashlib.sha256(output_path.read_bytes()).hexdigest()

    print(f"\nDEFER-HEURISTIC mapping: DEFER=1.0, others=0.5")
    print(f"  Model SHA: {model_sha}")
    print(f"  File SHA:  {file_sha}")
    print(f"  Written to: {output_path}")

    return mapping, model_sha, file_sha


def freeze_ps_ensemble(n_mappings: int = 16) -> list[dict]:
    """Generate 16 frozen PS mappings with distinct deterministic seeds.

    Each mapping uses a different shuffle_seed (101, 102, ..., 116).
    None are chosen by performance — they are pre-frozen before any
    evaluation.

    Returns list of {name, mapping, model_sha, file_sha, shuffle_seed}.
    """
    b1_path = REPO_ROOT / "experiments/i3_4/value/frozen_b1_table.json"
    b1 = PhaseActionTable.load(b1_path)

    ensemble_dir = REPO_ROOT / "experiments/i3_4/value/ps_ensemble"
    ensemble_dir.mkdir(parents=True, exist_ok=True)

    mappings = []
    print(f"\nGenerating {n_mappings} frozen PS mappings (PS01-PS{n_mappings:02d})...")

    for i in range(1, n_mappings + 1):
        name = f"PS{i:02d}"
        shuffle_seed = 100 + i  # 101, 102, ..., 116
        mapping = generate_frozen_ps_mapping(
            b1, shuffle_seed=shuffle_seed, actions=ACTIONS,
        )

        output_path = ensemble_dir / f"{name.lower()}_mapping.json"
        with open(output_path, "w") as f:
            json.dump({
                "model": name,
                "shuffle_seed": shuffle_seed,
                "seed_algorithm": "SHA-256(phase|shuffle_seed) → first 8 bytes as uint64",
                "mapping": mapping,
            }, f, indent=2, sort_keys=True)
        file_sha = hashlib.sha256(output_path.read_bytes()).hexdigest()
        model_sha = hashlib.sha256(
            json.dumps(mapping, sort_keys=True).encode()
        ).hexdigest()

        mappings.append({
            "name": name,
            "shuffle_seed": shuffle_seed,
            "model_sha": model_sha,
            "file_sha": file_sha,
            "path": str(output_path),
        })
        print(f"  {name} (seed={shuffle_seed}): model_sha={model_sha[:16]}...")

    return mappings


def main():
    print("=" * 60)
    print("I3.4e — Freeze Control Decomposition Artifacts")
    print("=" * 60)

    # 1. B0
    print("\n--- B0: Global action prior ---")
    b0_mapping, b0_model_sha, b0_file_sha = freeze_b0()

    # 2. CONST
    print("\n--- CONST: Uniform value control ---")
    const_mapping, const_model_sha, const_file_sha = freeze_const_mapping()

    # 3. DEFER-HEURISTIC
    print("\n--- DEFER-HEURISTIC ---")
    defer_mapping, defer_model_sha, defer_file_sha = freeze_defer_heuristic()

    # 4. PS ensemble (16 mappings)
    ps_ensemble = freeze_ps_ensemble(n_mappings=16)

    # 5. Master receipt
    receipt = {
        "experiment": "i3_4e_control_decomposition",
        "b0": {
            "model_sha": b0_model_sha,
            "file_sha": b0_file_sha,
            "path": "experiments/i3_4/value/b0_global_prior.json",
            "description": "E[U | a] — global action prior, no phase info",
        },
        "const": {
            "model_sha": const_model_sha,
            "file_sha": const_file_sha,
            "path": "experiments/i3_4/value/const_mapping.json",
            "description": "All actions = 0.5 — tests packet-structure effect",
        },
        "defer_heuristic": {
            "model_sha": defer_model_sha,
            "file_sha": defer_file_sha,
            "path": "experiments/i3_4/value/defer_heuristic_mapping.json",
            "description": "DEFER=1.0, others=0.5 — tests DEFER-bias hypothesis",
        },
        "ps_ensemble": {
            "n_mappings": len(ps_ensemble),
            "mappings": ps_ensemble,
            "description": "16 frozen shuffled B1 mappings — permutation distribution",
        },
        "b1_reference": {
            "model_sha": "c72b8641c5452d79e92fd9cb14ca289948b54385f2481a893ecb8c6bf22f1f68",
            "path": "experiments/i3_4/value/frozen_b1_table.json",
        },
        "note": (
            "All mappings are frozen BEFORE any evaluation. "
            "No mapping is selected by performance. "
            "The PS ensemble establishes the permutation distribution "
            "against which B1 and any future Q_phi must be compared."
        ),
    }

    receipt_path = REPO_ROOT / "experiments/i3_4/value/i3_4e_control_receipt.json"
    with open(receipt_path, "w") as f:
        json.dump(receipt, f, indent=2, sort_keys=True)
    print(f"\nMaster receipt written to: {receipt_path}")

    print("\n" + "=" * 60)
    print("DONE — All I3.4e control artifacts frozen")
    print("=" * 60)
    print(f"\nArms ready for I3.4e:")
    print(f"  P0     — baseline (no phase, no values)")
    print(f"  P2     — phase + correct B1 values")
    print(f"  B0     — phase + global action prior (no phase conditioning)")
    print(f"  CONST  — phase + uniform values (structure only)")
    print(f"  DEFER  — phase + DEFER heuristic")
    print(f"  PV     — phase + numeric values only (no ranking field)")
    print(f"  PR     — phase + ranking only (no numeric values)")
    print(f"  PS01-PS16 — 16 frozen shuffled mappings")


if __name__ == "__main__":
    main()
