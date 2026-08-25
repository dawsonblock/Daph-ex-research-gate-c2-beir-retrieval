#!/usr/bin/env python3
"""I3.5-PQ Phase 21b Part B: Repeated-action stress test (simulated depths).

The causal data only has RETRIEVE records at depth 0 (initial states).
To test whether QCAUSAL_V1 represents diminishing returns, we simulate
states at retrieval depths 0, 1, 2 by modifying state features:
  - retrieval_count: 0 -> 1 -> 2
  - retrieval_remaining: 3 -> 2 -> 1
  - n_visible_evidence: increment (retrieval adds evidence)
  - n_supporting: increment (retrieval finds supporting evidence)
  - same_action_run_length: 0 -> 1 -> 2
  - last_action: None -> RETRIEVE -> RETRIEVE

Then ask QCAUSAL_V1 to predict Q(s, RETRIEVE) and Q(s, VERIFY) at each
simulated depth. If Q(s, RETRIEVE) remains constant despite diminishing
true returns, history features are necessary.

We also compute the actual realized utility of RETRIEVE at each depth
from the six-arm experiment trajectories (which have states at various
depths in their action sequences).
"""
from __future__ import annotations

import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def extract_features(state_features: dict, action: str) -> dict:
    feats = {
        "n_live": state_features.get("n_live", 0),
        "n_eliminated": state_features.get("n_eliminated", 0),
        "n_untested": state_features.get("n_untested", 0),
        "n_total_hypotheses": state_features.get("n_total_hypotheses", 0),
        "n_visible_evidence": state_features.get("n_visible_evidence", 0),
        "n_verified": state_features.get("n_verified", 0),
        "n_supporting": state_features.get("n_supporting", 0),
        "n_contradicting": state_features.get("n_contradicting", 0),
        "n_stale": state_features.get("n_stale", 0),
        "retrieval_remaining": state_features.get("retrieval_remaining", 0),
        "search_remaining": state_features.get("search_remaining", 0),
        "verify_remaining": state_features.get("verify_remaining", 0),
        "steps_remaining": state_features.get("steps_remaining", 0),
        "can_retrieve": int(state_features.get("can_retrieve", False)),
        "can_search": int(state_features.get("can_search", False)),
        "can_verify": int(state_features.get("can_verify", False)),
        "searched": int(state_features.get("searched", False)),
        "reasoning_complete": int(state_features.get("reasoning_complete", False)),
        "same_action_run_length": state_features.get("same_action_run_length", 0),
        "retrieval_count": state_features.get("retrieval_count", 0),
        "search_count": state_features.get("search_count", 0),
        "verify_count": state_features.get("verify_count", 0),
    }
    for a in ["ANSWER", "DEFER", "RETRIEVE", "VERIFY", "SEARCH_MORE", "REASON_MORE"]:
        feats[f"a_{a}"] = int(action == a)
    feats["n_live_x_retrieve"] = feats["n_live"] * feats["a_RETRIEVE"]
    feats["n_live_x_verify"] = feats["n_live"] * feats["a_VERIFY"]
    feats["n_live_x_search"] = feats["n_live"] * feats["a_SEARCH_MORE"]
    feats["n_untested_x_retrieve"] = feats["n_untested"] * feats["a_RETRIEVE"]
    feats["n_untested_x_verify"] = feats["n_untested"] * feats["a_VERIFY"]
    feats["n_supporting_x_answer"] = feats["n_supporting"] * feats["a_ANSWER"]
    feats["n_eliminated_x_defer"] = feats["n_eliminated"] * feats["a_DEFER"]
    return feats


def simulate_retrieval_depth(base_sf: dict, depth: int) -> dict:
    """Simulate state features after `depth` retrievals.

    Each retrieval:
    - Increments retrieval_count
    - Decrements retrieval_remaining
    - Adds 1 visible evidence (supporting)
    - Increments same_action_run_length (if consecutive)
    - Decrements steps_remaining
    """
    sf = dict(base_sf)
    sf["retrieval_count"] = depth
    sf["retrieval_remaining"] = max(0, sf.get("retrieval_remaining", 3) - depth)
    sf["n_visible_evidence"] = sf.get("n_visible_evidence", 0) + depth
    sf["n_supporting"] = sf.get("n_supporting", 0) + depth
    sf["same_action_run_length"] = depth
    sf["steps_remaining"] = max(0, sf.get("steps_remaining", 10) - depth)
    # After 2+ retrievals, can_retrieve may become False if budget exhausted
    if sf["retrieval_remaining"] <= 0:
        sf["can_retrieve"] = False
    return sf


def main():
    est_dir = REPO_ROOT / "experiments/i3_5/pinned_policy/frozen_estimators"
    pinned_dir = REPO_ROOT / "experiments/i3_5/pinned_policy"
    six_arm_dir = REPO_ROOT / "experiments/i3_5/six_arm"

    # Load frozen QCAUSAL
    print("Loading frozen QCAUSAL_V1...")
    with open(est_dir / "QCAUSAL_gbt.pkl", "rb") as f:
        qcausal_model = pickle.load(f)
    with open(est_dir / "feature_schema.json") as f:
        feature_schema = json.load(f)
    feature_keys = feature_schema["feature_keys"]

    # Load causal data for base states
    print("Loading pinned-policy causal data...")
    causal_records = []
    with open(pinned_dir / "pinned_causal_actions_v1.jsonl") as f:
        for line in f:
            causal_records.append(json.loads(line))

    # Get unique base states (one per checkpoint)
    by_checkpoint = defaultdict(dict)
    checkpoint_sf = {}
    checkpoint_category = {}
    for r in causal_records:
        cp = r["checkpoint_id"]
        by_checkpoint[cp][r["forced_action"]] = r["pinned_policy_utility"]
        if cp not in checkpoint_sf:
            checkpoint_sf[cp] = r["state_features"]
            checkpoint_category[cp] = r["category"]

    # Find states where RETRIEVE is legal and has been tested
    retrieve_states = []
    for cp, action_qs in by_checkpoint.items():
        if "RETRIEVE" in action_qs:
            sf = checkpoint_sf[cp]
            if sf.get("can_retrieve", False) and sf.get("retrieval_remaining", 0) > 0:
                retrieve_states.append({
                    "checkpoint_id": cp,
                    "category": checkpoint_category[cp],
                    "state_features": sf,
                    "q_actual_retrieve": action_qs["RETRIEVE"],
                    "q_actual_verify": action_qs.get("VERIFY"),
                    "all_q": action_qs,
                })

    print(f"  {len(retrieve_states)} states with RETRIEVE tested and legal")

    # ================================================================
    # Simulate depths 0, 1, 2 and predict Q
    # ================================================================
    print("\n" + "=" * 80)
    print("REPEATED-ACTION STRESS TEST: Q(s, RETRIEVE) vs retrieval depth")
    print("=" * 80)
    print()
    print("  Simulating states at retrieval depths 0, 1, 2 by modifying:")
    print("    retrieval_count, retrieval_remaining, n_visible_evidence,")
    print("    n_supporting, same_action_run_length, steps_remaining")
    print()

    epsilon = 3.0
    depths = [0, 1, 2]

    # For each state, predict Q at each depth
    results_by_state = []
    for st in retrieve_states:
        base_sf = st["state_features"]
        depth_preds = {}
        for depth in depths:
            sim_sf = simulate_retrieval_depth(base_sf, depth)
            # Predict Q for RETRIEVE and VERIFY
            legal = []
            if sim_sf.get("can_retrieve", False) or depth == 0:
                legal.append("RETRIEVE")
            if sim_sf.get("can_verify", False):
                legal.append("VERIFY")
            if sim_sf.get("can_search", False):
                legal.append("SEARCH_MORE")
            legal.extend(["ANSWER", "DEFER", "REASON_MORE"])

            preds = {}
            for a in legal:
                feats = extract_features(sim_sf, a)
                X = np.array([[feats[k] for k in feature_keys]])
                preds[a] = float(qcausal_model.predict(X)[0])

            depth_preds[depth] = {
                "q_retrieve": preds.get("RETRIEVE"),
                "q_verify": preds.get("VERIFY"),
                "all_preds": preds,
                "can_retrieve": sim_sf.get("can_retrieve", False),
                "retrieval_remaining": sim_sf.get("retrieval_remaining", 0),
            }

            # Compute near-optimal set
            if preds:
                q_max = max(preds.values())
                optimal_set = {a for a, q in preds.items() if q >= q_max - epsilon}
                depth_preds[depth]["near_optimal_set"] = sorted(optimal_set)
                depth_preds[depth]["retrieve_in_set"] = "RETRIEVE" in optimal_set
                depth_preds[depth]["set_size"] = len(optimal_set)

        results_by_state.append({
            "checkpoint_id": st["checkpoint_id"],
            "category": st["category"],
            "q_actual_retrieve_depth0": st["q_actual_retrieve"],
            "depth_preds": depth_preds,
        })

    # ================================================================
    # Aggregate: Mean Q_pred for RETRIEVE and VERIFY by depth
    # ================================================================
    print(f"  {'Depth':>5s} {'n':>5s} {'Q_pred RETRIEVE':>16s} {'Q_pred VERIFY':>14s} "
          f"{'P(RETR in A_eps)':>18s} {'MeanSetSize':>12s}")

    depth_summary = []
    for depth in depths:
        q_rets = [r["depth_preds"][depth]["q_retrieve"] for r in results_by_state
                  if r["depth_preds"][depth]["q_retrieve"] is not None]
        q_ver = [r["depth_preds"][depth]["q_verify"] for r in results_by_state
                 if r["depth_preds"][depth]["q_verify"] is not None]
        retr_in_set = [r["depth_preds"][depth]["retrieve_in_set"] for r in results_by_state
                       if "retrieve_in_set" in r["depth_preds"][depth]]
        set_sizes = [r["depth_preds"][depth]["set_size"] for r in results_by_state
                     if "set_size" in r["depth_preds"][depth]]

        mean_q_ret = sum(q_rets) / len(q_rets) if q_rets else 0
        mean_q_ver = sum(q_ver) / len(q_ver) if q_ver else 0
        p_retr_in = sum(retr_in_set) / len(retr_in_set) if retr_in_set else 0
        mean_size = sum(set_sizes) / len(set_sizes) if set_sizes else 0

        print(f"  {depth:5d} {len(q_rets):5d} {mean_q_ret:16.2f} {mean_q_ver:14.2f} "
              f"{p_retr_in:18.4f} {mean_size:12.2f}")

        depth_summary.append({
            "depth": depth,
            "n": len(q_rets),
            "mean_q_retrieve": round(mean_q_ret, 4),
            "mean_q_verify": round(mean_q_ver, 4),
            "p_retrieve_in_set": round(p_retr_in, 4),
            "mean_set_size": round(mean_size, 4),
        })

    # ================================================================
    # The critical test
    # ================================================================
    print()
    print("  CRITICAL TEST: Does QCAUSAL_V1 predict decreasing Q for RETRIEVE?")
    print()

    if len(depth_summary) >= 2:
        q0 = depth_summary[0]["mean_q_retrieve"]
        q1 = depth_summary[1]["mean_q_retrieve"]
        q2 = depth_summary[2]["mean_q_retrieve"] if len(depth_summary) > 2 else q1

        decrease_01 = q0 - q1
        decrease_02 = q0 - q2
        decrease_12 = q1 - q2

        print(f"  Q_pred(RETRIEVE | 0 retrieves) = {q0:.2f}")
        print(f"  Q_pred(RETRIEVE | 1 retrieve)  = {q1:.2f}")
        print(f"  Q_pred(RETRIEVE | 2 retrieves) = {q2:.2f}")
        print(f"  Decrease 0->1: {decrease_01:.2f}")
        print(f"  Decrease 1->2: {decrease_12:.2f}")
        print(f"  Decrease 0->2: {decrease_02:.2f}")
        print()

        # Also check VERIFY trend
        v0 = depth_summary[0]["mean_q_verify"]
        v1 = depth_summary[1]["mean_q_verify"]
        v2 = depth_summary[2]["mean_q_verify"] if len(depth_summary) > 2 else v1
        print(f"  Q_pred(VERIFY | 0 retrieves) = {v0:.2f}")
        print(f"  Q_pred(VERIFY | 1 retrieve)  = {v1:.2f}")
        print(f"  Q_pred(VERIFY | 2 retrieves) = {v2:.2f}")
        print()

        # Check if RETRIEVE drops out of the near-optimal set
        p0 = depth_summary[0]["p_retrieve_in_set"]
        p1 = depth_summary[1]["p_retrieve_in_set"]
        p2 = depth_summary[2]["p_retrieve_in_set"]
        print(f"  P(RETRIEVE in A_eps | 0 retrieves) = {p0:.4f}")
        print(f"  P(RETRIEVE in A_eps | 1 retrieve)  = {p1:.4f}")
        print(f"  P(RETRIEVE in A_eps | 2 retrieves) = {p2:.4f}")
        print()

        # Verdict
        if decrease_02 < 1.0:
            print(f"  VERDICT: QCAUSAL_V1 does NOT represent diminishing returns.")
            print(f"  Q_pred(RETRIEVE) decreases by only {decrease_02:.2f} from depth 0 to 2.")
            print(f"  The estimator treats RETRIEVE at step 2 and RETRIEVE after")
            print(f"  already retrieving twice as nearly equivalent.")
            print(f"  History-aware Q_V2 IS necessary.")
            verdict = "history_necessary"
        elif decrease_02 < 3.0:
            print(f"  VERDICT: QCAUSAL_V1 partially represents diminishing returns.")
            print(f"  Q_pred(RETRIEVE) decreases by {decrease_02:.2f} from depth 0 to 2.")
            print(f"  This is small but non-zero. The I2 interface may compensate")
            print(f"  by letting the LLM decide on near-ties.")
            print(f"  History-aware Q_V2 may help but is not critical.")
            verdict = "history_marginal"
        else:
            print(f"  VERDICT: QCAUSAL_V1 DOES represent diminishing returns.")
            print(f"  Q_pred(RETRIEVE) decreases by {decrease_02:.2f} from depth 0 to 2.")
            print(f"  History-aware Q_V2 is NOT necessary — V1 already captures this.")
            verdict = "history_not_needed"

        # Also check: does RETRIEVE drop out of A_epsilon?
        if p2 < p0:
            print(f"  P(RETRIEVE in A_eps) drops from {p0:.2f} to {p2:.2f} at depth 2.")
            print(f"  The I2 interface would stop recommending RETRIEVE after 2 retrieves.")
            print(f"  This is additional protection against over-retrieval.")
        else:
            print(f"  P(RETRIEVE in A_eps) remains at {p2:.2f} even at depth 2.")
            print(f"  The I2 interface would still include RETRIEVE in the near-optimal set.")
            print(f"  The LLM's own judgment is the primary protection against over-retrieval.")
    else:
        verdict = "insufficient_data"
        print("  Insufficient depth variation")

    # ================================================================
    # Also: empirical evidence from six-arm trajectories
    # ================================================================
    print()
    print("=" * 80)
    print("  EMPIRICAL EVIDENCE: Actual RETRIEVE outcomes by depth")
    print("  (from six-arm trajectories where QCAUSAL over-retrieved)")
    print("=" * 80)
    print()

    # Load six-arm trajectories
    six_arm_traj = []
    with open(six_arm_dir / "trajectories_v1.jsonl") as f:
        for line in f:
            six_arm_traj.append(json.loads(line))

    # For QCAUSAL arm, find trajectories with multiple RETRIEVEs
    # and compute the realized utility contribution of each RETRIEVE
    qcausal_traj = [r for r in six_arm_traj if r["arm"] == "QCAUSAL"]
    print(f"  QCAUSAL trajectories: {len(qcausal_traj)}")

    # Count trajectories with 1, 2, 3 RETRIEVEs
    from collections import Counter
    retr_counts = Counter()
    for r in qcausal_traj:
        retr_counts[r["actions_taken"].count("RETRIEVE")] += 1
    print(f"  RETRIEVE count distribution: {dict(retr_counts)}")

    # Compare success/utility by RETRIEVE count
    print()
    print(f"  {'n_RETRIEVE':>10s} {'n_traj':>7s} {'mean_U':>10s} {'success':>8s}")
    for n_retr in sorted(retr_counts.keys()):
        recs = [r for r in qcausal_traj if r["actions_taken"].count("RETRIEVE") == n_retr]
        us = [r["realized_utility"] for r in recs]
        sr = sum(1 for r in recs if r["success"]) / len(recs)
        print(f"  {n_retr:10d} {len(recs):7d} {sum(us)/len(us):10.2f} {sr:8.2f}")

    # ================================================================
    # Compare with I2 from interface ablation
    # ================================================================
    print()
    i2_traj = [r for r in six_arm_traj if r["arm"] == "QCAUSAL"]  # placeholder
    # Actually load from interface ablation
    ablation_traj = []
    with open(REPO_ROOT / "experiments/i3_5/interface_ablation/trajectories_v1.jsonl") as f:
        for line in f:
            ablation_traj.append(json.loads(line))

    i2_traj = [r for r in ablation_traj if r["arm"] == "I2"]
    print(f"  I2 trajectories: {len(i2_traj)}")
    i2_retr_counts = Counter()
    for r in i2_traj:
        i2_retr_counts[r["actions_taken"].count("RETRIEVE")] += 1
    print(f"  I2 RETRIEVE count distribution: {dict(i2_retr_counts)}")

    print()
    print(f"  {'n_RETRIEVE':>10s} {'n_traj':>7s} {'mean_U':>10s} {'success':>8s}")
    for n_retr in sorted(i2_retr_counts.keys()):
        recs = [r for r in i2_traj if r["actions_taken"].count("RETRIEVE") == n_retr]
        us = [r["realized_utility"] for r in recs]
        sr = sum(1 for r in recs if r["success"]) / len(recs)
        print(f"  {n_retr:10d} {len(recs):7d} {sum(us)/len(us):10.2f} {sr:8.2f}")

    # ================================================================
    # Save results
    # ================================================================
    results = {
        "depth_summary": depth_summary,
        "verdict": verdict,
        "q_pred_decrease_0_to_2": decrease_02 if len(depth_summary) >= 2 else None,
        "p_retrieve_in_set_by_depth": [d["p_retrieve_in_set"] for d in depth_summary],
    }
    output_path = REPO_ROOT / "experiments/i3_5/executive_v1/stress_test_v1.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
