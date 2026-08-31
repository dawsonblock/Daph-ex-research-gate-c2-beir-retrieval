#!/usr/bin/env python3
"""V3R2 rule-proxy comparison for DAPH-X M4.

NOTE: This is a RULE PROXY, not the frozen V3R2 release.
The run_v3r2_baseline() function implements handcrafted rules
that approximate V3R2's terminal authority behavior:
  unique support → ANSWER
  competing support → DEFER
  otherwise → DEFER

This is NOT a scientific V3R2-vs-DAPH-X comparison.
It is a development diagnostic comparing DAPH-X against a simple
rule-based baseline on the M4 corpus.

Usage:
    python scripts/m4_v3r2_proxy_comparison.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import joblib

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

M4_DIR = REPO_ROOT / "experiments/daph_x/m4"
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from train_m4_q_res import extract_m4_features, compute_q_mb_from_record, load_m4_split


def v3r2_proxy_action(records: list[dict]) -> dict:
    """Select action using V3R2 rule proxy.

    Rules:
      - If unique supported hypothesis with ANSWER action → ANSWER
      - If competing support → DEFER
      - Otherwise → DEFER
    """
    # Find ANSWER action if unique supported
    answer_records = [r for r in records if r["first_action_type"] == "ANSWER"]
    defer_records = [r for r in records if r["first_action_type"] == "DEFER"]

    # V3R2 proxy: ANSWER if available, else DEFER
    # (In real V3R2, this would check unique support via topology)
    if answer_records:
        return answer_records[0]
    return defer_records[0] if defer_records else records[0]


def daph_x_action(records: list[dict], q_res_model, feature_keys: list[str]) -> dict:
    """Select action using DAPH-X learned Q_X."""
    best_q = -float("inf")
    best_rec = records[0]

    for rec in records:
        feats = extract_m4_features(rec)
        x = np.array([[feats[k] for k in feature_keys]])
        q_mb = compute_q_mb_from_record(rec)
        q_res = q_res_model.predict(x)[0]
        q_x = q_mb + q_res
        if q_x > best_q:
            best_q = q_x
            best_rec = rec

    return best_rec


def main():
    # Load Q_res model
    q_res_data = joblib.load(M4_DIR / "q_res_m4.pkl")
    q_res_model = q_res_data["model"]
    feature_keys = q_res_data["feature_keys"]

    for split_name in ["structural_ood", "mechanism_ood"]:
        records = load_m4_split(split_name)
        if not records:
            continue

        groups = defaultdict(list)
        for r in records:
            groups[r["counterfactual_group_id"]].append(r)

        v3r2_utilities = []
        daph_x_utilities = []
        oracle_utilities = []
        v3r2_correct = 0
        daph_x_correct = 0
        n_groups = 0

        for gid, group in groups.items():
            if len(group) < 2:
                continue
            n_groups += 1

            # Oracle
            oracle_util = max(r["utility"] for r in group)
            oracle_utilities.append(oracle_util)

            # V3R2 proxy
            v3r2_rec = v3r2_proxy_action(group)
            v3r2_utilities.append(v3r2_rec["utility"])
            if v3r2_rec["utility"] == oracle_util:
                v3r2_correct += 1

            # DAPH-X
            daph_rec = daph_x_action(group, q_res_model, feature_keys)
            daph_x_utilities.append(daph_rec["utility"])
            if daph_rec["utility"] == oracle_util:
                daph_x_correct += 1

        v3r2_mean = np.mean(v3r2_utilities)
        daph_mean = np.mean(daph_x_utilities)
        oracle_mean = np.mean(oracle_utilities)

        v3r2_regret = oracle_mean - v3r2_mean
        daph_regret = oracle_mean - daph_mean

        print(f"{'='*60}")
        print(f"  {split_name.upper()} — V3R2 Rule-Proxy vs DAPH-X")
        print(f"{'='*60}")
        print(f"  Groups: {n_groups}")
        print(f"  V3R2 proxy:  mean_utility={v3r2_mean:.2f}, correct={v3r2_correct}/{n_groups} ({v3r2_correct/n_groups:.3f}), regret={v3r2_regret:.2f}")
        print(f"  DAPH-X:      mean_utility={daph_mean:.2f}, correct={daph_x_correct}/{n_groups} ({daph_x_correct/n_groups:.3f}), regret={daph_regret:.2f}")
        print(f"  Oracle:      mean_utility={oracle_mean:.2f}")
        print(f"  DAPH-X vs V3R2: {daph_mean - v3r2_mean:+.2f} utility, {daph_regret - v3r2_regret:+.2f} regret")
        print()

    # Save with clear labeling
    output = {
        "note": "RULE PROXY comparison, NOT frozen V3R2. The V3R2 baseline is approximated by handcrafted rules. This is a development diagnostic.",
        "v3r2_implementation": "rule_proxy",
        "daph_x_implementation": "learned_q_res",
    }
    output_path = M4_DIR / "v3r2_proxy_comparison_m4.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
