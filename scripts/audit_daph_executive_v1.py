#!/usr/bin/env python3
"""I3.5-PQ Phase 21b: Gap/set-size audit + repeated-action stress test.

Two-part audit of DAPH_EXECUTIVE_V1 (frozen QCAUSAL_V1 + I2):

PART A: Gap/set-size audit
  For each trajectory state in the interface-ablation experiment:
    - Compute the near-optimal set A_epsilon(s)
    - Measure MeanSetSize by gap class (near_tie, moderate, clear)
    - Measure NearOptimalSetCoverage (does A_epsilon contain the true optimum?)
    - Measure P(a_LLM in A_epsilon) overall and stratified by gap class
    - Measure P(a_LLM in A_epsilon | gap > 3) — where DAPH has real information

PART B: Repeated-action stress test
  Create states with 0, 1, 2 retrieves performed (holding task fixed).
  Ask the frozen QCAUSAL_V1:
    Q(s_0, RETRIEVE)
    Q(s_1, RETRIEVE)
    Q(s_2, RETRIEVE)
  If Q remains constant despite diminishing true returns, history features
  are necessary. If Q decreases, V1 already represents diminishing returns.

  Also measure the actual realized utility of RETRIEVE at each depth
  from the pinned-policy causal data.
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


def sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


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


def main():
    est_dir = REPO_ROOT / "experiments/i3_5/pinned_policy/frozen_estimators"
    pinned_dir = REPO_ROOT / "experiments/i3_5/pinned_policy"
    ablation_dir = REPO_ROOT / "experiments/i3_5/interface_ablation"

    # Load frozen QCAUSAL
    print("Loading frozen QCAUSAL_V1...")
    with open(est_dir / "QCAUSAL_gbt.pkl", "rb") as f:
        qcausal_model = pickle.load(f)
    with open(est_dir / "feature_schema.json") as f:
        feature_schema = json.load(f)
    feature_keys = feature_schema["feature_keys"]

    # Load causal data for Q values and true returns
    print("Loading pinned-policy causal data...")
    causal_records = []
    with open(pinned_dir / "pinned_causal_actions_v1.jsonl") as f:
        for line in f:
            causal_records.append(json.loads(line))
    print(f"  {len(causal_records)} causal records")

    # Load interface ablation trajectories
    print("Loading interface ablation trajectories...")
    trajectories = []
    with open(ablation_dir / "trajectories_v1.jsonl") as f:
        for line in f:
            trajectories.append(json.loads(line))
    print(f"  {len(trajectories)} trajectories")

    # ================================================================
    # PART A: Gap/set-size audit
    # ================================================================
    print("\n" + "=" * 80)
    print("PART A: GAP/SET-SIZE AUDIT")
    print("=" * 80)

    # For each causal record, compute the near-optimal set and gap
    # Group by checkpoint_id (each checkpoint = one state)
    by_checkpoint = defaultdict(dict)
    checkpoint_meta = {}
    for r in causal_records:
        cp = r["checkpoint_id"]
        by_checkpoint[cp][r["forced_action"]] = r["pinned_policy_utility"]
        if cp not in checkpoint_meta:
            checkpoint_meta[cp] = {
                "category": r["category"],
                "state_features": r["state_features"],
                "legal_actions": r.get("legal_actions", []),
            }

    # Compute gap and near-optimal set for each checkpoint
    epsilon = 3.0
    gap_classes = {"near_tie": [], "moderate": [], "clear": []}
    set_sizes = {"near_tie": [], "moderate": [], "clear": []}
    contains_optimum = {"near_tie": [], "moderate": [], "clear": []}
    all_checkpoints = []

    for cp, action_qs in by_checkpoint.items():
        if len(action_qs) < 2:
            continue
        q_max = max(action_qs.values())
        q_sorted = sorted(action_qs.values(), reverse=True)
        gap = q_sorted[0] - q_sorted[1]
        optimal_set = {a for a, q in action_qs.items() if q >= q_max - epsilon}
        true_best = max(action_qs, key=action_qs.get)

        if gap <= epsilon:
            gap_class = "near_tie"
        elif gap <= 10:
            gap_class = "moderate"
        else:
            gap_class = "clear"

        gap_classes[gap_class].append(gap)
        set_sizes[gap_class].append(len(optimal_set))
        contains_optimum[gap_class].append(1 if true_best in optimal_set else 0)
        all_checkpoints.append({
            "checkpoint_id": cp,
            "category": checkpoint_meta[cp]["category"],
            "gap": gap,
            "gap_class": gap_class,
            "set_size": len(optimal_set),
            "optimal_set": sorted(optimal_set),
            "contains_optimum": true_best in optimal_set,
            "q_values": {a: round(q, 2) for a, q in action_qs.items()},
        })

    # Report
    print(f"\n  Total checkpoints with >= 2 actions: {len(all_checkpoints)}")
    print(f"\n  {'Gap class':15s} {'n':>5s} {'MeanSetSize':>12s} {'ContainsOpt':>12s} {'MeanGap':>10s}")
    for gc in ["clear", "moderate", "near_tie"]:
        n = len(gap_classes[gc])
        if n == 0:
            continue
        mean_size = sum(set_sizes[gc]) / n
        mean_cov = sum(contains_optimum[gc]) / n
        mean_gap = sum(gap_classes[gc]) / n
        print(f"  {gc:15s} {n:5d} {mean_size:12.2f} {mean_cov:12.4f} {mean_gap:10.2f}")

    # Overall
    all_sizes = [s for ss in set_sizes.values() for s in ss]
    all_cov = [c for cc in contains_optimum.values() for c in cc]
    print(f"  {'ALL':15s} {len(all_sizes):5d} {sum(all_sizes)/len(all_sizes):12.2f} "
          f"{sum(all_cov)/len(all_cov):12.4f}")

    # ================================================================
    # P(a_LLM in A_epsilon) from interface ablation trajectories
    # ================================================================
    print("\n" + "=" * 80)
    print("  P(a_LLM in A_epsilon) — does Qwen follow the near-optimal set?")
    print("=" * 80)

    # Build category -> {action: Q_value} for computing A_epsilon
    by_cat_action = defaultdict(list)
    for r in causal_records:
        by_cat_action[(r["category"], r["forced_action"])].append(r["pinned_policy_utility"])
    category_q = {}
    for (cat, action), values in by_cat_action.items():
        if cat not in category_q:
            category_q[cat] = {}
        category_q[cat][action] = sum(values) / len(values)

    # For each trajectory, check if first action is in A_epsilon
    by_task_arm = {}
    for r in trajectories:
        by_task_arm[(r["task_id"], r["arm"])] = r

    task_ids = sorted(set(r["task_id"] for r in trajectories))

    def get_subtype(tid):
        parts = tid.split("_")
        return "_".join(parts[2:4])

    # Compute gap per subtype (using mean Q values)
    subtype_gaps = {}
    for cat, action_qs in category_q.items():
        if len(action_qs) >= 2:
            qs = sorted(action_qs.values(), reverse=True)
            subtype_gaps[cat] = qs[0] - qs[1]
        else:
            subtype_gaps[cat] = 0

    print(f"\n  {'Arm':4s} {'P(in A_eps)':>12s} {'P(in A_eps | gap>3)':>20s} "
          f"{'P(in A_eps | gap<=3)':>20s}")

    for arm in ["C0", "I0", "I2", "I3", "I4"]:
        in_set_all = 0
        in_set_gap_gt3 = 0
        n_gap_gt3 = 0
        in_set_gap_le3 = 0
        n_gap_le3 = 0
        for tid in task_ids:
            r = by_task_arm.get((tid, arm))
            if not r or not r["actions_taken"]:
                continue
            subtype = get_subtype(tid)
            q_vals = category_q.get(subtype, {})
            if not q_vals:
                continue
            q_max = max(q_vals.values())
            optimal_set = {a for a, q in q_vals.items() if q >= q_max - epsilon}
            gap = subtype_gaps.get(subtype, 0)
            first_action = r["actions_taken"][0]
            in_set = first_action in optimal_set
            if in_set:
                in_set_all += 1
            if gap > 3:
                n_gap_gt3 += 1
                if in_set:
                    in_set_gap_gt3 += 1
            else:
                n_gap_le3 += 1
                if in_set:
                    in_set_gap_le3 += 1
        total = n_gap_gt3 + n_gap_le3
        p_all = in_set_all / total if total else 0
        p_gt3 = in_set_gap_gt3 / n_gap_gt3 if n_gap_gt3 else 0
        p_le3 = in_set_gap_le3 / n_gap_le3 if n_gap_le3 else 0
        print(f"  {arm:4s} {p_all:12.4f} {p_gt3:20.4f} {p_le3:20.4f}")

    # ================================================================
    # Set size distribution
    # ================================================================
    print("\n" + "=" * 80)
    print("  SET SIZE DISTRIBUTION")
    print("=" * 80)
    from collections import Counter
    for gc in ["clear", "moderate", "near_tie"]:
        sizes = set_sizes[gc]
        if not sizes:
            continue
        dist = Counter(sizes)
        print(f"\n  {gc}:")
        for size in sorted(dist.keys()):
            print(f"    size={size}: {dist[size]} checkpoints ({dist[size]/len(sizes):.2%})")

    # ================================================================
    # PART B: Repeated-action stress test
    # ================================================================
    print("\n" + "=" * 80)
    print("PART B: REPEATED-ACTION STRESS TEST")
    print("=" * 80)
    print()
    print("  Question: Does QCAUSAL_V1 represent diminishing returns for RETRIEVE?")
    print("  Method: Compare Q(s, RETRIEVE) at retrieval depths 0, 1, 2")
    print("          against actual realized utility from the causal data.")
    print()

    # Group causal records by (checkpoint_id, forced_action=RETRIEVE)
    # and by retrieval depth
    # The causal data has state_features which include retrieval_count

    # First, find all RETRIEVE records and group by retrieval_count
    retrieve_records = [r for r in causal_records if r["forced_action"] == "RETRIEVE"]
    print(f"  Total RETRIEVE records: {len(retrieve_records)}")

    by_depth = defaultdict(list)
    for r in retrieve_records:
        sf = r["state_features"]
        depth = sf.get("retrieval_count", 0)
        by_depth[depth].append(r)

    print(f"  By retrieval depth:")
    for depth in sorted(by_depth.keys()):
        recs = by_depth[depth]
        utils = [r["pinned_policy_utility"] for r in recs]
        print(f"    depth={depth}: n={len(recs)} mean_realized_U={sum(utils)/len(utils):.2f}")

    # Now ask QCAUSAL_V1 to predict Q(s, RETRIEVE) for each depth
    print()
    print("  QCAUSAL_V1 predictions for Q(s, RETRIEVE) by retrieval depth:")
    print(f"  {'Depth':>5s} {'n':>5s} {'Mean Q_pred':>12s} {'Mean Q_actual':>14s} {'Diff':>8s}")

    depth_results = []
    for depth in sorted(by_depth.keys()):
        recs = by_depth[depth]
        if not recs:
            continue
        # Get Q predictions for each state
        q_preds = []
        q_actuals = []
        for r in recs:
            sf = r["state_features"]
            feats = extract_features(sf, "RETRIEVE")
            X = np.array([[feats[k] for k in feature_keys]])
            q_pred = float(qcausal_model.predict(X)[0])
            q_actual = r["pinned_policy_utility"]
            q_preds.append(q_pred)
            q_actuals.append(q_actual)
        mean_pred = sum(q_preds) / len(q_preds)
        mean_actual = sum(q_actuals) / len(q_actuals)
        diff = mean_pred - mean_actual
        print(f"  {depth:5d} {len(recs):5d} {mean_pred:12.2f} {mean_actual:14.2f} {diff:8.2f}")
        depth_results.append({
            "depth": depth,
            "n": len(recs),
            "mean_q_pred": round(mean_pred, 4),
            "mean_q_actual": round(mean_actual, 4),
            "diff": round(diff, 4),
        })

    # ================================================================
    # The critical test: Does Q_pred decrease with depth?
    # ================================================================
    print()
    print("  CRITICAL TEST: Does QCAUSAL_V1 predict decreasing Q for RETRIEVE?")
    print()

    if len(depth_results) >= 2:
        q_pred_0 = depth_results[0]["mean_q_pred"]
        q_pred_1 = depth_results[1]["mean_q_pred"] if len(depth_results) > 1 else q_pred_0
        q_pred_2 = depth_results[2]["mean_q_pred"] if len(depth_results) > 2 else q_pred_1

        q_actual_0 = depth_results[0]["mean_q_actual"]
        q_actual_1 = depth_results[1]["mean_q_actual"] if len(depth_results) > 1 else q_actual_0
        q_actual_2 = depth_results[2]["mean_q_actual"] if len(depth_results) > 2 else q_actual_1

        pred_decrease = q_pred_0 - q_pred_2 if len(depth_results) > 2 else q_pred_0 - q_pred_1
        actual_decrease = q_actual_0 - q_actual_2 if len(depth_results) > 2 else q_actual_0 - q_actual_1

        print(f"  Q_pred(0 retrieves) = {q_pred_0:.2f}")
        print(f"  Q_pred(1 retrieve)  = {q_pred_1:.2f}")
        print(f"  Q_pred(2 retrieves) = {q_pred_2:.2f}")
        print(f"  Q_pred decrease (0->2) = {pred_decrease:.2f}")
        print()
        print(f"  Q_actual(0 retrieves) = {q_actual_0:.2f}")
        print(f"  Q_actual(1 retrieve)  = {q_actual_1:.2f}")
        print(f"  Q_actual(2 retrieves) = {q_actual_2:.2f}")
        print(f"  Q_actual decrease (0->2) = {actual_decrease:.2f}")
        print()

        if pred_decrease < 1.0:
            print(f"  VERDICT: QCAUSAL_V1 does NOT represent diminishing returns.")
            print(f"  Q_pred decreases by only {pred_decrease:.2f} from depth 0 to 2.")
            print(f"  Q_actual decreases by {actual_decrease:.2f}.")
            print(f"  History-aware Q_V2 IS necessary.")
        else:
            print(f"  VERDICT: QCAUSAL_V1 DOES represent diminishing returns.")
            print(f"  Q_pred decreases by {pred_decrease:.2f} from depth 0 to 2.")
            print(f"  Q_actual decreases by {actual_decrease:.2f}.")
            print(f"  History-aware Q_V2 is NOT necessary — V1 already captures this.")
    else:
        print(f"  Insufficient depth variation in causal data (only {len(depth_results)} depths)")

    # ================================================================
    # Also check: does the I2 interface produce different sets at different depths?
    # ================================================================
    print()
    print("  Does the I2 near-optimal set change with retrieval depth?")
    print()

    for depth in sorted(by_depth.keys()):
        recs = by_depth[depth]
        if not recs:
            continue
        # For each state at this depth, compute A_epsilon
        set_sizes_at_depth = []
        retrieve_in_set_count = 0
        for r in recs:
            sf = r["state_features"]
            # Get all legal actions at this state
            # We need to predict Q for all actions
            legal = ["ANSWER", "DEFER", "RETRIEVE", "VERIFY", "SEARCH_MORE", "REASON_MORE"]
            # Filter by affordances
            if not sf.get("can_retrieve", False):
                legal = [a for a in legal if a != "RETRIEVE"]
            if not sf.get("can_search", False):
                legal = [a for a in legal if a != "SEARCH_MORE"]
            if not sf.get("can_verify", False):
                legal = [a for a in legal if a != "VERIFY"]

            X = np.array([[extract_features(sf, a)[k] for k in feature_keys] for a in legal])
            preds = qcausal_model.predict(X)
            q_vals = dict(zip(legal, [float(p) for p in preds]))
            q_max = max(q_vals.values())
            optimal_set = {a for a, q in q_vals.items() if q >= q_max - epsilon}
            set_sizes_at_depth.append(len(optimal_set))
            if "RETRIEVE" in optimal_set:
                retrieve_in_set_count += 1

        mean_size = sum(set_sizes_at_depth) / len(set_sizes_at_depth)
        p_retrieve_in_set = retrieve_in_set_count / len(recs)
        print(f"    depth={depth}: n={len(recs)} mean_set_size={mean_size:.2f} "
              f"P(RETRIEVE in A_eps)={p_retrieve_in_set:.4f}")

    # ================================================================
    # Save results
    # ================================================================
    results = {
        "part_a": {
            "gap_class_summary": {
                gc: {
                    "n": len(gap_classes[gc]),
                    "mean_set_size": sum(set_sizes[gc]) / len(set_sizes[gc]) if set_sizes[gc] else 0,
                    "contains_optimum_rate": sum(contains_optimum[gc]) / len(contains_optimum[gc]) if contains_optimum[gc] else 0,
                    "mean_gap": sum(gap_classes[gc]) / len(gap_classes[gc]) if gap_classes[gc] else 0,
                }
                for gc in ["clear", "moderate", "near_tie"]
            },
        },
        "part_b": {
            "depth_results": depth_results,
            "q_pred_decrease_0_to_2": pred_decrease if len(depth_results) >= 2 else None,
            "q_actual_decrease_0_to_2": actual_decrease if len(depth_results) >= 2 else None,
        },
    }
    output_path = REPO_ROOT / "experiments/i3_5/executive_v1/audit_v1.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
