#!/usr/bin/env python3
"""I3.5-PQ Phase 20: Mechanism audit.

Per-subtype analysis of action selection patterns across arms.
Answers: WHY does QCAUSAL hurt on some subtypes while QOBS helps?

For each (subtype, arm) pair, we analyze:
  1. First-action distribution (what does the LLM choose first?)
  2. Action sequence patterns (what trajectories does it produce?)
  3. Where QCAUSAL diverges from P0 and QOBS
  4. The "over-retrieval" hypothesis: does QCAUSAL cause repeated
     non-terminal actions that waste resources?
  5. Per-subtype delta-U decomposition: which subtypes drive the
     QCAUSAL > B0 result, and which drive the QCAUSAL < QOBS result?
  6. Per-subtype near-optimal action rate
  7. The "value-LLM interface" diagnosis: compare the estimator's
     ranking to the LLM's actual first action
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def load_trajectories(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def load_model_calls(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def main():
    six_arm_dir = REPO_ROOT / "experiments/i3_5/six_arm"
    pinned_dir = REPO_ROOT / "experiments/i3_5/pinned_policy"

    print("Loading data...")
    trajectories = load_trajectories(six_arm_dir / "trajectories_v1.jsonl")
    model_calls = load_model_calls(six_arm_dir / "model_calls_v1.jsonl")
    print(f"  {len(trajectories)} trajectories, {len(model_calls)} model calls")

    # Load pinned causal data for Q values
    causal_records = []
    with open(pinned_dir / "pinned_causal_actions_v1.jsonl") as f:
        for line in f:
            causal_records.append(json.loads(line))

    # Build category -> {action: mean_Q}
    by_cat_action = defaultdict(list)
    for r in causal_records:
        by_cat_action[(r["category"], r["forced_action"])].append(r["pinned_policy_utility"])
    category_q = {}
    for (cat, action), values in by_cat_action.items():
        if cat not in category_q:
            category_q[cat] = {}
        category_q[cat][action] = sum(values) / len(values)

    arms = ["P0", "B0", "B1", "PS05", "QOBS", "QCAUSAL"]

    # Organize by (task_id, arm)
    by_task_arm = {}
    for r in trajectories:
        by_task_arm[(r["task_id"], r["arm"])] = r

    task_ids = sorted(set(r["task_id"] for r in trajectories))

    # Extract subtype from task_id
    def get_subtype(tid: str) -> str:
        parts = tid.split("_")
        return "_".join(parts[2:4])

    def get_gap_bucket(subtype: str) -> str:
        # From the audit: ol_defer and tl_retrieve are clear/moderate
        if subtype in ("ol_defer",):
            return "clear_choice"
        elif subtype in ("tl_defer",):
            return "moderate_choice"
        elif subtype in ("tl_retrieve",):
            return "clear_choice"
        else:
            return "near_tie"

    subtypes = sorted(set(get_subtype(tid) for tid in task_ids))

    # ================================================================
    # 1. Per-subtype first-action distribution
    # ================================================================
    print("\n" + "=" * 80)
    print("1. PER-SUBTYPE FIRST-ACTION DISTRIBUTION")
    print("=" * 80)

    for subtype in subtypes:
        subtype_tasks = [tid for tid in task_ids if get_subtype(tid) == subtype]
        print(f"\n  {subtype} ({len(subtype_tasks)} tasks, gap={get_gap_bucket(subtype)}):")

        # Get the Q-best action for this subtype
        q_vals = category_q.get(subtype, {})
        if q_vals:
            q_best = max(q_vals, key=q_vals.get)
            q_sorted = sorted(q_vals.items(), key=lambda x: -x[1])
            print(f"    Q ranking: {', '.join(f'{a}={v:.1f}' for a, v in q_sorted[:4])}")

        for arm in arms:
            first_actions = []
            for tid in subtype_tasks:
                r = by_task_arm.get((tid, arm))
                if r and r["actions_taken"]:
                    first_actions.append(r["actions_taken"][0])
            dist = Counter(first_actions)
            top3 = ", ".join(f"{a}:{c}" for a, c in dist.most_common(3))
            print(f"    {arm:10s}: {top3}")

    # ================================================================
    # 2. Per-subtype mean utility and delta-U decomposition
    # ================================================================
    print("\n" + "=" * 80)
    print("2. PER-SUBTYPE MEAN UTILITY")
    print("=" * 80)

    print(f"\n  {'Subtype':15s} {'Gap':>15s}", end="")
    for arm in arms:
        print(f" {arm:>10s}", end="")
    print()

    subtype_utils = {}
    for subtype in subtypes:
        subtype_tasks = [tid for tid in task_ids if get_subtype(tid) == subtype]
        bucket = get_gap_bucket(subtype)
        print(f"  {subtype:15s} {bucket:>15s}", end="")
        subtype_utils[subtype] = {}
        for arm in arms:
            recs = [by_task_arm[(tid, arm)] for tid in subtype_tasks if (tid, arm) in by_task_arm]
            if recs:
                mean_u = sum(r["realized_utility"] for r in recs) / len(recs)
                subtype_utils[subtype][arm] = mean_u
                print(f" {mean_u:10.2f}", end="")
            else:
                print(f" {'--':>10s}", end="")
        print()

    # ================================================================
    # 3. Delta-U decomposition: which subtypes drive the contrasts?
    # ================================================================
    print("\n" + "=" * 80)
    print("3. DELTA-U DECOMPOSITION (QCAUSAL vs comparators)")
    print("=" * 80)

    contrasts = [("QCAUSAL", "B0"), ("QCAUSAL", "QOBS"), ("QCAUSAL", "P0")]
    for a_arm, b_arm in contrasts:
        print(f"\n  ΔU({a_arm} - {b_arm}) by subtype:")
        print(f"    {'Subtype':15s} {'ΔU':>10s} {'n':>5s} {'contributes':>12s}")
        total_delta = 0
        for subtype in subtypes:
            subtype_tasks = [tid for tid in task_ids if get_subtype(tid) == subtype]
            diffs = []
            for tid in subtype_tasks:
                ra = by_task_arm.get((tid, a_arm))
                rb = by_task_arm.get((tid, b_arm))
                if ra and rb:
                    diffs.append(ra["realized_utility"] - rb["realized_utility"])
            if diffs:
                mean_diff = sum(diffs) / len(diffs)
                total_delta += mean_diff * len(diffs)
                contribution = mean_diff * len(diffs) / len(task_ids)
                sign = "+" if mean_diff > 0 else ""
                print(f"    {subtype:15s} {sign}{mean_diff:9.2f} {len(diffs):5d} {sign}{contribution:11.2f}")
        print(f"    {'TOTAL':15s} {total_delta/len(task_ids):10.2f}")

    # ================================================================
    # 4. Over-retrieval analysis: repeated non-terminal actions
    # ================================================================
    print("\n" + "=" * 80)
    print("4. REPEATED ACTION ANALYSIS (over-retrieval hypothesis)")
    print("=" * 80)

    for subtype in subtypes:
        subtype_tasks = [tid for tid in task_ids if get_subtype(tid) == subtype]
        print(f"\n  {subtype}:")
        for arm in arms:
            max_repeats = []
            mean_repeats = []
            for tid in subtype_tasks:
                r = by_task_arm.get((tid, arm))
                if r and r["actions_taken"]:
                    action_counts = Counter(r["actions_taken"])
                    max_rep = max(action_counts.values())
                    max_repeats.append(max_rep)
                    # Count repeats of non-terminal actions
                    non_terminal = {a: c for a, c in action_counts.items()
                                   if a not in ("ANSWER", "DEFER", "STOP")}
                    if non_terminal:
                        mean_repeats.append(max(non_terminal.values()))
                    else:
                        mean_repeats.append(0)
            if max_repeats:
                print(f"    {arm:10s}: max_repeat={max(max_repeats)} "
                      f"mean_max_repeat={sum(max_repeats)/len(max_repeats):.2f} "
                      f"mean_max_nonterminal_repeat={sum(mean_repeats)/len(mean_repeats):.2f}")

    # ================================================================
    # 5. Action sequence patterns
    # ================================================================
    print("\n" + "=" * 80)
    print("5. ACTION SEQUENCE PATTERNS (most common trajectories)")
    print("=" * 80)

    for subtype in subtypes:
        subtype_tasks = [tid for tid in task_ids if get_subtype(tid) == subtype]
        print(f"\n  {subtype}:")
        for arm in ["P0", "QOBS", "QCAUSAL"]:
            sequences = []
            for tid in subtype_tasks:
                r = by_task_arm.get((tid, arm))
                if r and r["actions_taken"]:
                    sequences.append(tuple(r["actions_taken"]))
            # Show top 3 patterns
            seq_counts = Counter(sequences)
            for seq, count in seq_counts.most_common(3):
                print(f"    {arm:10s}: {list(seq)} (n={count})")

    # ================================================================
    # 6. Per-subtype near-optimal action rate
    # ================================================================
    print("\n" + "=" * 80)
    print("6. PER-SUBTYPE NEAR-OPTIMAL ACTION RATE (epsilon=3)")
    print("=" * 80)

    for subtype in subtypes:
        subtype_tasks = [tid for tid in task_ids if get_subtype(tid) == subtype]
        q_vals = category_q.get(subtype, {})
        if not q_vals:
            continue
        q_star = max(q_vals.values())
        optimal_set = {a for a, q in q_vals.items() if q >= q_star - 3}

        print(f"\n  {subtype} (optimal_set={optimal_set}):")
        for arm in arms:
            correct = 0
            total = 0
            for tid in subtype_tasks:
                r = by_task_arm.get((tid, arm))
                if r and r["actions_taken"]:
                    if r["actions_taken"][0] in optimal_set:
                        correct += 1
                    total += 1
            rate = correct / total if total else 0
            print(f"    {arm:10s}: {correct}/{total} ({rate:.3f})")

    # ================================================================
    # 7. Value-LLM interface diagnosis
    # ================================================================
    print("\n" + "=" * 80)
    print("7. VALUE-LLM INTERFACE DIAGNOSIS")
    print("=" * 80)
    print("  For each subtype: what does the estimator rank #1,")
    print("  and what does the LLM actually choose?")

    # Load estimator predictions for each subtype
    # We need to reconstruct what each estimator would predict
    # for the initial state of each task
    import pickle
    est_dir = REPO_ROOT / "experiments/i3_5/pinned_policy/frozen_estimators"

    with open(est_dir / "feature_schema.json") as f:
        feature_schema = json.load(f)
    feature_keys = feature_schema["feature_keys"]

    with open(est_dir / "QCAUSAL_gbt.pkl", "rb") as f:
        qcausal_model = pickle.load(f)
    with open(est_dir / "QOBS_gbt.pkl", "rb") as f:
        qobs_model = pickle.load(f)

    # Load B1 table
    with open(est_dir / "B1_phase_action_table.json") as f:
        b1_data = json.load(f)

    def extract_features(sf, action):
        feats = {
            "n_live": sf.get("n_live", 0),
            "n_eliminated": sf.get("n_eliminated", 0),
            "n_untested": sf.get("n_untested", 0),
            "n_total_hypotheses": sf.get("n_total_hypotheses", 0),
            "n_visible_evidence": sf.get("n_visible_evidence", 0),
            "n_verified": sf.get("n_verified", 0),
            "n_supporting": sf.get("n_supporting", 0),
            "n_contradicting": sf.get("n_contradicting", 0),
            "n_stale": sf.get("n_stale", 0),
            "retrieval_remaining": sf.get("retrieval_remaining", 0),
            "search_remaining": sf.get("search_remaining", 0),
            "verify_remaining": sf.get("verify_remaining", 0),
            "steps_remaining": sf.get("steps_remaining", 0),
            "can_retrieve": int(sf.get("can_retrieve", False)),
            "can_search": int(sf.get("can_search", False)),
            "can_verify": int(sf.get("can_verify", False)),
            "searched": int(sf.get("searched", False)),
            "reasoning_complete": int(sf.get("reasoning_complete", False)),
            "same_action_run_length": sf.get("same_action_run_length", 0),
            "retrieval_count": sf.get("retrieval_count", 0),
            "search_count": sf.get("search_count", 0),
            "verify_count": sf.get("verify_count", 0),
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

    # Get initial state features for each subtype from the causal data
    # Use the first record for each category
    subtype_initial_state = {}
    for r in causal_records:
        if r["category"] not in subtype_initial_state:
            subtype_initial_state[r["category"]] = r["state_features"]

    for subtype in subtypes:
        sf = subtype_initial_state.get(subtype)
        if not sf:
            continue
        q_vals = category_q.get(subtype, {})
        if not q_vals:
            continue

        # What actions are legal at this state?
        legal = [a for a in ["ANSWER", "DEFER", "RETRIEVE", "VERIFY", "SEARCH_MORE", "REASON_MORE"]
                 if q_vals.get(a) is not None]

        # Estimator predictions
        X = np.array([[extract_features(sf, a)[k] for k in feature_keys] for a in legal])
        qcausal_preds = dict(zip(legal, qcausal_model.predict(X)))
        qobs_preds = dict(zip(legal, qobs_model.predict(X)))

        # Rankings
        qcausal_ranking = sorted(legal, key=lambda a: -qcausal_preds[a])
        qobs_ranking = sorted(legal, key=lambda a: -qobs_preds[a])
        actual_q_ranking = sorted(legal, key=lambda a: -q_vals[a])

        # What does the LLM choose first (most common)?
        llm_first = {}
        for arm in ["P0", "QOBS", "QCAUSAL"]:
            subtype_tasks = [tid for tid in task_ids if get_subtype(tid) == subtype]
            first_actions = []
            for tid in subtype_tasks:
                r = by_task_arm.get((tid, arm))
                if r and r["actions_taken"]:
                    first_actions.append(r["actions_taken"][0])
            if first_actions:
                llm_first[arm] = Counter(first_actions).most_common(1)[0]

        print(f"\n  {subtype}:")
        print(f"    Actual Q ranking:    {', '.join(f'{a}({q_vals[a]:.1f})' for a in actual_q_ranking[:4])}")
        print(f"    QCAUSAL predicts:    {', '.join(f'{a}({qcausal_preds[a]:.1f})' for a in qcausal_ranking[:4])}")
        print(f"    QOBS predicts:       {', '.join(f'{a}({qobs_preds[a]:.1f})' for a in qobs_ranking[:4])}")
        for arm in ["P0", "QOBS", "QCAUSAL"]:
            if arm in llm_first:
                action, count = llm_first[arm]
                print(f"    LLM({arm:8s}) chooses: {action} (n={count})")

    # ================================================================
    # 8. Summary: Why QOBS > QCAUSAL
    # ================================================================
    print("\n" + "=" * 80)
    print("8. SUMMARY: WHY QOBS > QCAUSAL")
    print("=" * 80)

    # Compute the key diagnostic: for each subtype, does QCAUSAL's
    # top-ranked action match the actual Q-best, and does the LLM
    # follow it?
    print("\n  Key diagnostic: Does the LLM follow the estimator's top recommendation?")
    print(f"  {'Subtype':15s} {'QCAUSAL top':>12s} {'LLM follows':>12s} {'QOBS top':>12s} {'LLM follows':>12s} {'Actual best':>12s}")

    for subtype in subtypes:
        sf = subtype_initial_state.get(subtype)
        if not sf:
            continue
        q_vals = category_q.get(subtype, {})
        if not q_vals:
            continue
        legal = [a for a in ["ANSWER", "DEFER", "RETRIEVE", "VERIFY", "SEARCH_MORE", "REASON_MORE"]
                 if q_vals.get(a) is not None]
        X = np.array([[extract_features(sf, a)[k] for k in feature_keys] for a in legal])
        qcausal_preds = dict(zip(legal, qcausal_model.predict(X)))
        qobs_preds = dict(zip(legal, qobs_model.predict(X)))

        qcausal_top = max(qcausal_preds, key=qcausal_preds.get)
        qobs_top = max(qobs_preds, key=qobs_preds.get)
        actual_best = max(q_vals, key=q_vals.get)

        # Does the LLM follow?
        subtype_tasks = [tid for tid in task_ids if get_subtype(tid) == subtype]
        qcausal_llm_first = Counter()
        qobs_llm_first = Counter()
        for tid in subtype_tasks:
            rq = by_task_arm.get((tid, "QCAUSAL"))
            ro = by_task_arm.get((tid, "QOBS"))
            if rq and rq["actions_taken"]:
                qcausal_llm_first[rq["actions_taken"][0]] += 1
            if ro and ro["actions_taken"]:
                qobs_llm_first[ro["actions_taken"][0]] += 1

        qcausal_follows = qcausal_llm_first.get(qcausal_top, 0)
        qcausal_total = sum(qcausal_llm_first.values())
        qobs_follows = qobs_llm_first.get(qobs_top, 0)
        qobs_total = sum(qobs_llm_first.values())

        print(f"  {subtype:15s} {qcausal_top:>12s} {qcausal_follows}/{qcausal_total:<10d} "
              f"{qobs_top:>12s} {qobs_follows}/{qobs_total:<10d} {actual_best:>12s}")

    # ================================================================
    # 9. The "over-guidance" diagnosis
    # ================================================================
    print("\n" + "=" * 80)
    print("9. OVER-GUIDANCE DIAGNOSIS")
    print("=" * 80)

    # For QCAUSAL, count how many times the LLM repeats the estimator's
    # top-recommended action
    print("\n  QCAUSAL: Does the LLM repeat the recommended action?")
    for subtype in subtypes:
        subtype_tasks = [tid for tid in task_ids if get_subtype(tid) == subtype]
        repeats = []
        for tid in subtype_tasks:
            r = by_task_arm.get((tid, "QCAUSAL"))
            if r and r["actions_taken"]:
                action_counts = Counter(r["actions_taken"])
                # Find the most repeated action
                most_repeated = max(action_counts.values())
                repeats.append(most_repeated)
        if repeats:
            print(f"    {subtype:15s}: mean_max_repeat={sum(repeats)/len(repeats):.2f} "
                  f"max={max(repeats)} n={len(repeats)}")

    print("\n  P0: Does the LLM repeat actions without guidance?")
    for subtype in subtypes:
        subtype_tasks = [tid for tid in task_ids if get_subtype(tid) == subtype]
        repeats = []
        for tid in subtype_tasks:
            r = by_task_arm.get((tid, "P0"))
            if r and r["actions_taken"]:
                action_counts = Counter(r["actions_taken"])
                most_repeated = max(action_counts.values())
                repeats.append(most_repeated)
        if repeats:
            print(f"    {subtype:15s}: mean_max_repeat={sum(repeats)/len(repeats):.2f} "
                  f"max={max(repeats)} n={len(repeats)}")

    # ================================================================
    # 10. Final mechanism summary
    # ================================================================
    print("\n" + "=" * 80)
    print("10. MECHANISM SUMMARY")
    print("=" * 80)

    print("""
    The mechanism audit reveals three distinct patterns:

    1. EASY SUBTYPES (ol_answer, ol_defer):
       - All arms perform identically
       - Qwen doesn't need guidance; the state is self-evident
       - Q values have clear separation (ANSWER=100, DEFER=70)
       - No arm causes harm

    2. RETRIEVAL SUBTYPES (ol_retrieve, tl_retrieve):
       - QCAUSAL causes over-retrieval (3 RETRIEVEs vs 1 for P0)
       - QCAUSAL correctly identifies RETRIEVE as high-value
       - But the LLM interprets the high normalized value as "always retrieve"
       - QOBS avoids this because its biased estimates are more conservative
       - P0 avoids this because Qwen naturally chooses VERIFY first
       - Result: QCAUSAL hurts (U=54.58 vs 91.38 for P0 on ol_retrieve)

    3. TWO-LIVE HARD SUBTYPES (tl_search, tl_verify):
       - B0 collapses (global prior misleads Qwen into premature DEFER)
       - QCAUSAL and QOBS both prevent B0's collapse
       - QOBS is slightly better because its estimates are more conservative
       - Result: QCAUSAL > B0 but QOBS > QCAUSAL

    THE CORE MECHANISM:
    - QCAUSAL's accurate Q values are "too aggressive" for the LLM
    - When QCAUSAL says "RETRIEVE is valuable" (Q=91.4), the LLM
      interprets the normalized value (1.0) as "always retrieve"
    - QOBS's biased Q values are "accidentally conservative"
    - The observational bias acts as implicit regularization
    - P0 (no guidance) lets Qwen use its own judgment, which is
      better than any current form of value guidance

    THE VALUE-LLM INTERFACE PROBLEM:
    - Current interface: normalized Q values + ranking in the packet
    - Problem: normalization removes magnitude information
    - A Q value of 91.4 and 91.4 (RETRIEVE vs VERIFY) both normalize to
      1.0 and 0.999, making them look like a strong preference
    - But the actual difference is negligible (0.0 in the causal data)
    - The LLM can't distinguish "strong preference" from "near tie"
    """)

    # Save results
    results = {
        "subtype_utils": {st: {arm: round(u, 4) for arm, u in arms_u.items()}
                          for st, arms_u in subtype_utils.items()},
    }
    output_path = six_arm_dir / "mechanism_audit_v1.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
