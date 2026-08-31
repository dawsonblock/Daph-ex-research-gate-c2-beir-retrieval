#!/usr/bin/env python3
"""Build the DAPH-X M4 causal corpus using procedural generation + multi-step rollout.

Generates states procedurally, splits at the family/mechanism level,
runs multi-step counterfactual rollouts, and checks balance.

Usage:
    python scripts/build_m4_corpus.py [--n_states 2000] [--seed 42]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from daph_x.benchmark.procedural_generator import (
    generate_state, generate_paired_worlds, GeneratorConfig, HARM_MECHANISMS,
    MECHANISM_FAMILIES,
)
from daph_x.benchmark.novelty_signatures import compute_all_signatures
from daph_x.benchmark.balance_checker import check_balance
from daph_x.receipts.checkpoint import checkpoint_from_task_and_runtime
from daph_x.receipts.rollout_engine import (
    evaluate_all_actions_rollout, DownstreamPolicy, RolloutResult,
)
from daph_x.actions.candidate_generator import generate_and_prune
from daph_x.graph.epistemic_graph import build_graph_from_evidence_task


def split_mechanisms(seed: int) -> dict:
    """Split mechanism families into train/calibration/structural_ood/mechanism_ood.

    TRAIN: 55% — evidence_quality, action_selection, resource, benign
    CALIBRATION: 15% — subset of train mechanisms (for conformal calibration)
    STRUCTURAL_OOD: 15% — same mechanisms, held-out topology families
    MECHANISM_OOD: 15% — model_error, structural (held-out mechanisms)
    """
    import random
    rng = random.Random(seed)

    all_families = list(MECHANISM_FAMILIES.keys())

    # Mechanism OOD: model_error and structural families
    mechanism_ood_families = ["model_error", "structural"]
    mechanism_ood_mechs = []
    for f in mechanism_ood_families:
        mechanism_ood_mechs.extend(MECHANISM_FAMILIES[f])

    # Train + calibration + structural_ood: the rest
    train_families = [f for f in all_families if f not in mechanism_ood_families]
    train_mechs = []
    for f in train_families:
        train_mechs.extend(MECHANISM_FAMILIES[f])

    return {
        "train_mechanisms": train_mechs,
        "mechanism_ood_mechanisms": mechanism_ood_mechs,
        "mechanism_ood_families": mechanism_ood_families,
        "train_families": train_families,
    }


def build_corpus(
    n_states: int = 2000,
    seed: int = 42,
    output_dir: Path | None = None,
) -> dict:
    """Build the M4 causal corpus.

    Args:
        n_states: Target number of states to generate
        seed: Master RNG seed
        output_dir: Where to save the corpus

    Returns:
        Metadata dict with corpus statistics
    """
    if output_dir is None:
        output_dir = REPO_ROOT / "experiments/daph_x/m4"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = GeneratorConfig()
    mechanism_split = split_mechanisms(seed)

    # Split: 55% train, 15% calibration, 15% structural_ood, 15% mechanism_ood
    n_train = int(n_states * 0.55)
    n_cal = int(n_states * 0.15)
    n_struct_ood = int(n_states * 0.15)
    n_mech_ood = int(n_states * 0.15)

    print(f"Generating {n_states} states:")
    print(f"  TRAIN: {n_train}")
    print(f"  CALIBRATION: {n_cal}")
    print(f"  STRUCTURAL_OOD: {n_struct_ood}")
    print(f"  MECHANISM_OOD: {n_mech_ood}")
    print()

    # Generate states for each split
    all_states = {}
    split_states = {
        "train": [],
        "calibration": [],
        "structural_ood": [],
        "mechanism_ood": [],
    }

    # Train + calibration: use train mechanisms
    for i in range(n_train + n_cal):
        s = seed + i
        state = generate_state(
            seed=s,
            config=config,
            allowed_mechanisms=mechanism_split["train_mechanisms"],
        )
        split_name = "train" if i < n_train else "calibration"
        split_states[split_name].append(state)

    # Structural OOD: same mechanisms but we'll filter by family signature
    for i in range(n_struct_ood * 3):  # Generate extra, filter for novelty
        s = seed + n_train + n_cal + i
        state = generate_state(
            seed=s,
            config=config,
            allowed_mechanisms=mechanism_split["train_mechanisms"],
        )
        split_states["structural_ood"].append(state)
        if len(split_states["structural_ood"]) >= n_struct_ood:
            break

    # Mechanism OOD: use held-out mechanisms
    for i in range(n_mech_ood):
        s = seed + n_train + n_cal + n_struct_ood + i
        state = generate_state(
            seed=s,
            config=config,
            allowed_mechanisms=mechanism_split["mechanism_ood_mechanisms"],
        )
        split_states["mechanism_ood"].append(state)

    # Generate paired worlds (10% of train)
    n_paired = n_train // 10
    paired_states = []
    for i in range(n_paired):
        s = seed + 100000 + i
        state_a, state_b = generate_paired_worlds(
            seed=s,
            config=config,
            allowed_mechanisms=mechanism_split["train_mechanisms"],
        )
        paired_states.append((state_a, state_b))
        split_states["train"].append(state_a)
        split_states["train"].append(state_b)

    # Enforce family-level novelty: structural_ood families must not overlap with train
    train_families = set(s.signatures.family for s in split_states["train"])
    struct_ood_filtered = []
    for state in split_states["structural_ood"]:
        if state.signatures.family not in train_families:
            struct_ood_filtered.append(state)
    split_states["structural_ood"] = struct_ood_filtered[:n_struct_ood]

    # If we don't have enough structural_ood after filtering, keep what we have
    print(f"After family-level filtering:")
    for split_name, states in split_states.items():
        print(f"  {split_name}: {len(states)} states")

    # Check family overlap
    train_fams = set(s.signatures.family for s in split_states["train"])
    cal_fams = set(s.signatures.family for s in split_states["calibration"])
    struct_fams = set(s.signatures.family for s in split_states["structural_ood"])
    mech_fams = set(s.signatures.family for s in split_states["mechanism_ood"])

    print(f"\nFamily signature overlap:")
    print(f"  train ∩ calibration: {len(train_fams & cal_fams)} (expected: some overlap, calibration is from train mechanisms)")
    print(f"  train ∩ structural_ood: {len(train_fams & struct_fams)} (should be 0)")
    print(f"  train ∩ mechanism_ood: {len(train_fams & mech_fams)} (may overlap — different mechanism, same structure)")

    # Mechanism overlap
    train_mechs_set = set(s.harm_mechanism for s in split_states["train"])
    mech_ood_mechs_set = set(s.harm_mechanism for s in split_states["mechanism_ood"])
    print(f"\nMechanism overlap:")
    print(f"  train ∩ mechanism_ood: {len(train_mechs_set & mech_ood_mechs_set)} (should be 0)")

    # Run rollouts for each split
    downstream_policy = DownstreamPolicy()

    for split_name, states in split_states.items():
        print(f"\nRunning rollouts for {split_name}...")
        rollout_records = []
        harm_labels = []
        intervention_features = []

        for i, state in enumerate(states):
            if i % 100 == 0:
                print(f"  {split_name}: {i}/{len(states)}")

            # Create checkpoint
            checkpoint = checkpoint_from_task_and_runtime(state.task, None, seed=state.generator_seed)

            # Generate candidates
            candidates = generate_and_prune(state.graph)
            if not candidates:
                continue

            # Run multi-step rollout for each action
            results = evaluate_all_actions_rollout(
                checkpoint=checkpoint,
                actions=candidates,
                downstream_policy=downstream_policy,
                world_model_config=state.world_model_config,
                max_steps=8,
                seed=state.generator_seed,
            )

            # Find best action by utility (oracle)
            best_result = max(results, key=lambda r: r.utility)
            oracle_utility = best_result.utility

            # Simulate base policy: always DEFER
            base_result = None
            for r in results:
                if "DEFER" in r.first_action:
                    base_result = r
                    break
            if base_result is None:
                base_result = results[0]

            # Compute intervention metrics
            delta_u = best_result.utility - base_result.utility
            is_harmful = 1 if delta_u < 0 else 0

            # Build intervention features (pre-decision only)
            features = {
                "n_hyp": len(state.graph.hypothesis_ids()),
                "n_ev": len(state.graph.evidence_ids()),
                "steps_remaining": state.graph.steps_remaining,
                "verify_remaining": state.graph.verify_remaining,
                "search_remaining": state.graph.search_remaining,
                "n_verified": sum(1 for n in state.graph.nodes.values()
                                  if n.node_type == "evidence" and n.verification_state != "UNVERIFIED"),
                "n_unverified": sum(1 for n in state.graph.nodes.values()
                                    if n.node_type == "evidence" and n.verification_state == "UNVERIFIED"),
                "has_competition": 1.0 if any(
                    s.signatures.family == "competing_support" for s in [state]
                ) else 0.0,
                "harm_mechanism": state.harm_mechanism,
                "mechanism_family": state.mechanism_family,
            }

            harm_labels.append(is_harmful)
            intervention_features.append(features)

            # Build rollout records
            for j, result in enumerate(results):
                record = result.to_dict()
                record["record_id"] = f"{state.task.task_id}:{j}"
                record["counterfactual_group_id"] = state.task.task_id
                record["task_id"] = state.task.task_id
                record["correct_hypothesis_id"] = state.correct_hypothesis_id
                record["harm_mechanism"] = state.harm_mechanism
                record["mechanism_family"] = state.mechanism_family
                record["signatures"] = state.signatures.to_dict()
                record["pair_id"] = state.pair_id
                record["pair_polarity"] = state.pair_polarity
                record["oracle_utility"] = oracle_utility
                record["regret"] = oracle_utility - result.utility
                record["is_harmful_intervention"] = is_harmful
                record["delta_u"] = delta_u
                rollout_records.append(record)

        # Save rollout records
        output_file = output_dir / f"m4_{split_name}.jsonl"
        with open(output_file, "w") as f:
            for r in rollout_records:
                f.write(json.dumps(r, default=str) + "\n")
        print(f"  Saved {len(rollout_records)} records to {output_file}")

        # Run balance check for this split
        if harm_labels and len(set(harm_labels)) > 1:
            balance = check_balance(intervention_features, harm_labels)
            balance_file = output_dir / f"m4_{split_name}_balance.json"
            with open(balance_file, "w") as f:
                json.dump(balance, f, indent=2)
            print(f"  Balance check: {'PASS' if balance['passed'] else 'FAIL'}")
            if balance["flagged_features"]:
                print(f"  Flagged features (AUROC > {balance['threshold']}):")
                for ff in balance["flagged_features"]:
                    print(f"    {ff['feature']}: AUROC={ff['auroc']}")

    # Save metadata
    metadata = {
        "n_states_target": n_states,
        "seed": seed,
        "splits": {k: len(v) for k, v in split_states.items()},
        "mechanism_split": mechanism_split,
        "family_overlap": {
            "train_cal": len(train_fams & cal_fams),
            "train_struct_ood": len(train_fams & struct_fams),
            "train_mech_ood": len(train_fams & mech_fams),
        },
        "mechanism_overlap": {
            "train_mech_ood": len(train_mechs_set & mech_ood_mechs_set),
        },
        "n_paired_worlds": len(paired_states),
        "generator_config": {
            "n_hyp_range": config.n_hyp_range,
            "n_ev_range": config.n_ev_range,
            "steps_range": config.steps_range,
            "verify_range": config.verify_range,
        },
        "downstream_policy": DownstreamPolicy.POLICY_VERSION,
    }
    metadata_file = output_dir / "m4_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved to {metadata_file}")

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Build DAPH-X M4 causal corpus")
    parser.add_argument("--n_states", type=int, default=2000,
                        help="Target number of states to generate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Master RNG seed")
    args = parser.parse_args()

    metadata = build_corpus(n_states=args.n_states, seed=args.seed)
    print(f"\nDone. Generated {sum(metadata['splits'].values())} states.")


if __name__ == "__main__":
    main()
