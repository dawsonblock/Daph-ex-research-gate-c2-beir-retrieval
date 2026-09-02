#!/usr/bin/env python3
"""DAPH-X R6: Alternative gate formulations.

Tests several ways to convert pairwise/correctness predictions into
intervention decisions, to find what actually beats simple baselines.

Gate variants:
  A. MaxCal + margin: pick max P(correct), intervene only if margin > threshold
  B. Bradley-Terry: pairwise win-rate ranking, pick top if gap > threshold
  C. Direct Z model: train P(Z=+1) and P(Z=-1) directly
  D. Conditional gate: only intervene when P(base correct) < threshold
  E. Ensemble: combine all signals with a meta-classifier
  F. MaxCal + pairwise confirmation: pick max P(correct) only if pairwise agrees

Usage:
    python scripts/run_r6_evaluation.py \\
        --corpus experiments/daph_x/reasoning/reasoning_corpus_v2.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from daph_x.coding.reasoning_tasks import check_answer

R6_DIR = REPO_ROOT / "experiments/daph_x/r6"


def load_corpus(path):
    tasks = []
    with open(path) as f:
        for line in f:
            tasks.append(json.loads(line))
    return tasks


def split_tasks(tasks, train_frac=0.6, cal_frac=0.15, seed=42):
    rng = np.random.RandomState(seed)
    n = len(tasks)
    indices = rng.permutation(n)
    n_train = int(n * train_frac)
    n_cal = int(n * cal_frac)
    return (
        [tasks[i] for i in indices[:n_train]],
        [tasks[i] for i in indices[n_train:n_train + n_cal]],
        [tasks[i] for i in indices[n_train + n_cal:]],
    )


def get_feature_keys(records):
    all_keys = set()
    for r in records:
        all_keys.update(r["features"].keys())
    return sorted(all_keys)


def build_feature_vector(features, feature_keys):
    return np.array([float(features.get(k, 0.0)) for k in feature_keys])


def build_pair_features(a, b, feature_keys):
    fa = build_feature_vector(a["features"], feature_keys)
    fb = build_feature_vector(b["features"], feature_keys)
    return np.concatenate([fa - fb, fa, fb, np.abs(fa - fb)])


def flatten_candidates(tasks):
    records = []
    for task in tasks:
        for cand in task["candidates"]:
            records.append(cand)
    return records


# ─── Train all models ───
def train_all_models(train_tasks, cal_tasks, feature_keys):
    train_records = flatten_candidates(train_tasks)
    cal_records = flatten_candidates(cal_tasks)

    # 1. Correctness model
    X_c, y_c = [], []
    for r in train_records:
        X_c.append(build_feature_vector(r["features"], feature_keys))
        y_c.append(1 if r["is_correct"] else 0)
    corr_model = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.1,
        subsample=0.8, random_state=42)
    corr_model.fit(np.array(X_c), np.array(y_c))

    # Calibrate
    raw_p, true_l = [], []
    for r in cal_records:
        f = build_feature_vector(r["features"], feature_keys)
        raw_p.append(corr_model.predict_proba([f])[0, 1])
        true_l.append(1 if r["is_correct"] else 0)
    corr_cal = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
    corr_cal.fit(np.array(raw_p), np.array(true_l))

    # 2. Pairwise preference model
    X_p, y_p = [], []
    for task in train_tasks:
        cands = task["candidates"]
        for i in range(len(cands)):
            for j in range(len(cands)):
                if i == j: continue
                if cands[i]["is_correct"] == cands[j]["is_correct"]: continue
                X_p.append(build_pair_features(cands[i], cands[j], feature_keys))
                y_p.append(1 if cands[i]["is_correct"] else 0)
    pref_model = None
    if len(set(y_p)) >= 2 and len(X_p) >= 10:
        pref_model = GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1,
            subsample=0.8, random_state=42)
        pref_model.fit(np.array(X_p), np.array(y_p))

    # Calibrate pairwise
    raw_pp, true_pp = [], []
    for task in cal_tasks:
        cands = task["candidates"]
        for i in range(len(cands)):
            for j in range(len(cands)):
                if i == j: continue
                if cands[i]["is_correct"] == cands[j]["is_correct"]: continue
                feats = build_pair_features(cands[i], cands[j], feature_keys)
                raw_pp.append(pref_model.predict_proba([feats])[0, 1])
                true_pp.append(1 if cands[i]["is_correct"] else 0)
    pref_cal = None
    if len(set(true_pp)) >= 2:
        pref_cal = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
        pref_cal.fit(np.array(raw_pp), np.array(true_pp))

    # 3. Direct Z model: P(Z=+1) and P(Z=-1)
    # Z=+1: alt correct, base wrong. Z=-1: alt wrong, base correct.
    X_z, z_rescue, z_break = [], [], []
    for task in train_tasks:
        cands = task["candidates"]
        base = cands[0]
        for c in cands[1:]:
            feats = build_pair_features(c, base, feature_keys)
            X_z.append(feats)
            z_rescue.append(1 if (c["is_correct"] and not base["is_correct"]) else 0)
            z_break.append(1 if (not c["is_correct"] and base["is_correct"]) else 0)

    rescue_model = None
    break_model = None
    if len(set(z_rescue)) >= 2:
        rescue_model = GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1,
            subsample=0.8, random_state=42)
        rescue_model.fit(np.array(X_z), np.array(z_rescue))
    if len(set(z_break)) >= 2:
        break_model = GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1,
            subsample=0.8, random_state=42)
        break_model.fit(np.array(X_z), np.array(z_break))

    # 4. Ensemble meta-classifier: combine all signals
    # Features: [P(base correct), P(alt correct), P(alt>base), P(base>alt), P(rescue), P(break),
    #            confidence_base, confidence_alt, agreement_base, agreement_alt]
    X_meta, y_meta_rescue, y_meta_break = [], [], []
    for task in train_tasks:
        cands = task["candidates"]
        base = cands[0]
        for c in cands[1:]:
            p_base = _predict_correctness(corr_model, corr_cal, base["features"], feature_keys)
            p_alt = _predict_correctness(corr_model, corr_cal, c["features"], feature_keys)
            p_aw = _predict_pref(pref_model, pref_cal, c, base, feature_keys)
            p_bw = _predict_pref(pref_model, pref_cal, base, c, feature_keys)
            p_rescue = 0.5
            p_break = 0.5
            if rescue_model is not None:
                pf = build_pair_features(c, base, feature_keys)
                p_rescue = rescue_model.predict_proba([pf])[0, 1]
            if break_model is not None:
                pf = build_pair_features(c, base, feature_keys)
                p_break = break_model.predict_proba([pf])[0, 1]

            meta_feats = np.array([
                p_base, p_alt, p_aw, p_bw, p_rescue, p_break,
                base["self_confidence"], c["self_confidence"],
                base["features"].get("agreement_rate", 0),
                c["features"].get("agreement_rate", 0),
                base["features"].get("n_agreeing", 0),
                c["features"].get("n_agreeing", 0),
            ])
            X_meta.append(meta_feats)
            y_meta_rescue.append(1 if (c["is_correct"] and not base["is_correct"]) else 0)
            y_meta_break.append(1 if (not c["is_correct"] and base["is_correct"]) else 0)

    meta_rescue = None
    meta_break = None
    if len(set(y_meta_rescue)) >= 2:
        meta_rescue = GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1,
            subsample=0.8, random_state=42)
        meta_rescue.fit(np.array(X_meta), np.array(y_meta_rescue))
    if len(set(y_meta_break)) >= 2:
        meta_break = GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1,
            subsample=0.8, random_state=42)
        meta_break.fit(np.array(X_meta), np.array(y_meta_break))

    return {
        "corr_model": corr_model, "corr_cal": corr_cal,
        "pref_model": pref_model, "pref_cal": pref_cal,
        "rescue_model": rescue_model, "break_model": break_model,
        "meta_rescue": meta_rescue, "meta_break": meta_break,
        "feature_keys": feature_keys,
    }


def _predict_correctness(model, cal, features, fk):
    f = build_feature_vector(features, fk)
    raw = model.predict_proba([f])[0, 1]
    return float(cal.predict([raw])[0]) if cal else float(raw)


def _predict_pref(model, cal, a, b, fk):
    if model is None: return 0.5
    f = build_pair_features(a, b, fk)
    raw = model.predict_proba([f])[0, 1]
    return float(cal.predict([raw])[0]) if cal else float(raw)


# ─── Baselines ───
def sys_base(task):
    c = task["candidates"][0]
    return {"correct": c["is_correct"], "utility": 100.0 if c["is_correct"] else 0.0}

def sys_majority_vote(task):
    answers = [c["answer"] for c in task["candidates"]]
    mv = Counter(answers).most_common(1)[0][0]
    pick = next(c for c in task["candidates"] if c["answer"] == mv)
    return {"correct": pick["is_correct"], "utility": 100.0 if pick["is_correct"] else 0.0}

def sys_max_confidence(task):
    pick = max(task["candidates"], key=lambda c: c["self_confidence"])
    return {"correct": pick["is_correct"], "utility": 100.0 if pick["is_correct"] else 0.0}

def sys_max_calibrated(task, models):
    fk = models["feature_keys"]
    best, best_p = None, -1
    for c in task["candidates"]:
        p = _predict_correctness(models["corr_model"], models["corr_cal"], c["features"], fk)
        if p > best_p:
            best_p = p
            best = c
    return {"correct": best["is_correct"], "utility": 100.0 if best["is_correct"] else 0.0}


# ─── Gate variants ───

def gate_A_maxcal_margin(task, models, margin=0.2):
    """Pick max P(correct), intervene only if margin over base > threshold."""
    fk = models["feature_keys"]
    base = task["candidates"][0]
    p_base = _predict_correctness(models["corr_model"], models["corr_cal"], base["features"], fk)

    best, best_p = base, p_base
    for c in task["candidates"][1:]:
        p = _predict_correctness(models["corr_model"], models["corr_cal"], c["features"], fk)
        if p > best_p:
            best_p = p
            best = c

    would_force = (best["candidate_id"] != base["candidate_id"]) and (best_p - p_base > margin)
    pick = best if would_force else base
    return {"correct": pick["is_correct"], "utility": 100.0 if pick["is_correct"] else 0.0,
            "would_force": would_force}


def gate_B_bradley_terry(task, models, margin=0.3):
    """Bradley-Terry: compute win-rate for each candidate, pick top if gap > margin."""
    fk = models["feature_keys"]
    cands = task["candidates"]
    base = cands[0]

    # Compute win rate for each candidate
    win_rates = []
    for c in cands:
        scores = []
        for other in cands:
            if c["candidate_id"] == other["candidate_id"]: continue
            p = _predict_pref(models["pref_model"], models["pref_cal"], c, other, fk)
            scores.append(p)
        win_rates.append(np.mean(scores) if scores else 0.5)

    best_idx = np.argmax(win_rates)
    best = cands[best_idx]

    would_force = (best["candidate_id"] != base["candidate_id"]) and (win_rates[best_idx] - win_rates[0] > margin)
    pick = best if would_force else base
    return {"correct": pick["is_correct"], "utility": 100.0 if pick["is_correct"] else 0.0,
            "would_force": would_force}


def gate_C_direct_z(task, models, tau_r=0.5, tau_b=0.2):
    """Direct Z model: FORCE when P(rescue) > tau_r AND P(break) < tau_b."""
    fk = models["feature_keys"]
    cands = task["candidates"]
    base = cands[0]

    best_alt = None
    best_p_rescue = -1
    best_p_break = 1.0

    for c in cands[1:]:
        pf = build_pair_features(c, base, fk)
        p_rescue = 0.0
        p_break = 0.0
        if models["rescue_model"] is not None:
            p_rescue = models["rescue_model"].predict_proba([pf])[0, 1]
        if models["break_model"] is not None:
            p_break = models["break_model"].predict_proba([pf])[0, 1]

        if p_rescue > best_p_rescue:
            best_p_rescue = p_rescue
            best_p_break = p_break
            best_alt = c

    if best_alt is None:
        return {"correct": base["is_correct"], "utility": 100.0 if base["is_correct"] else 0.0,
                "would_force": False}

    would_force = best_p_rescue > tau_r and best_p_break < tau_b
    pick = best_alt if would_force else base
    return {"correct": pick["is_correct"], "utility": 100.0 if pick["is_correct"] else 0.0,
            "would_force": would_force}


def gate_D_conditional(task, models, p_base_thresh=0.4, tau_r=0.5, tau_b=0.3):
    """Conditional: only consider intervening when P(base correct) < threshold."""
    fk = models["feature_keys"]
    cands = task["candidates"]
    base = cands[0]
    p_base = _predict_correctness(models["corr_model"], models["corr_cal"], base["features"], fk)

    if p_base > p_base_thresh:
        return {"correct": base["is_correct"], "utility": 100.0 if base["is_correct"] else 0.0,
                "would_force": False}

    # Base likely wrong — find best alternative
    best_alt = None
    best_p = -1
    for c in cands[1:]:
        p = _predict_correctness(models["corr_model"], models["corr_cal"], c["features"], fk)
        if p > best_p:
            best_p = p
            best_alt = c

    would_force = best_p > 0.5
    pick = best_alt if would_force else base
    return {"correct": pick["is_correct"], "utility": 100.0 if pick["is_correct"] else 0.0,
            "would_force": would_force}


def gate_E_ensemble(task, models, tau_r=0.5, tau_b=0.2):
    """Ensemble: use meta-classifier combining all signals."""
    fk = models["feature_keys"]
    cands = task["candidates"]
    base = cands[0]

    if models["meta_rescue"] is None or models["meta_break"] is None:
        return {"correct": base["is_correct"], "utility": 100.0 if base["is_correct"] else 0.0,
                "would_force": False}

    best_alt = None
    best_p_rescue = -1
    best_p_break = 1.0

    for c in cands[1:]:
        p_base = _predict_correctness(models["corr_model"], models["corr_cal"], base["features"], fk)
        p_alt = _predict_correctness(models["corr_model"], models["corr_cal"], c["features"], fk)
        p_aw = _predict_pref(models["pref_model"], models["pref_cal"], c, base, fk)
        p_bw = _predict_pref(models["pref_model"], models["pref_cal"], base, c, fk)
        p_rescue = 0.5
        p_break = 0.5
        if models["rescue_model"] is not None:
            pf = build_pair_features(c, base, fk)
            p_rescue = models["rescue_model"].predict_proba([pf])[0, 1]
        if models["break_model"] is not None:
            pf = build_pair_features(c, base, fk)
            p_break = models["break_model"].predict_proba([pf])[0, 1]

        meta_feats = np.array([[
            p_base, p_alt, p_aw, p_bw, p_rescue, p_break,
            base["self_confidence"], c["self_confidence"],
            base["features"].get("agreement_rate", 0),
            c["features"].get("agreement_rate", 0),
            base["features"].get("n_agreeing", 0),
            c["features"].get("n_agreeing", 0),
        ]])

        p_meta_rescue = models["meta_rescue"].predict_proba(meta_feats)[0, 1]
        p_meta_break = models["meta_break"].predict_proba(meta_feats)[0, 1]

        if p_meta_rescue > best_p_rescue:
            best_p_rescue = p_meta_rescue
            best_p_break = p_meta_break
            best_alt = c

    would_force = best_p_rescue > tau_r and best_p_break < tau_b
    pick = best_alt if would_force else base
    return {"correct": pick["is_correct"], "utility": 100.0 if pick["is_correct"] else 0.0,
            "would_force": would_force}


def gate_F_maxcal_pairwise_confirm(task, models, margin=0.15, pairwise_thresh=0.6):
    """MaxCal selection, but only intervene if pairwise model confirms."""
    fk = models["feature_keys"]
    cands = task["candidates"]
    base = cands[0]
    p_base = _predict_correctness(models["corr_model"], models["corr_cal"], base["features"], fk)

    best, best_p = base, p_base
    for c in cands[1:]:
        p = _predict_correctness(models["corr_model"], models["corr_cal"], c["features"], fk)
        if p > best_p:
            best_p = p
            best = c

    if best["candidate_id"] == base["candidate_id"]:
        return {"correct": base["is_correct"], "utility": 100.0 if base["is_correct"] else 0.0,
                "would_force": False}

    # Check margin and pairwise confirmation
    p_pairwise = _predict_pref(models["pref_model"], models["pref_cal"], best, base, fk)
    would_force = (best_p - p_base > margin) and (p_pairwise > pairwise_thresh)
    pick = best if would_force else base
    return {"correct": pick["is_correct"], "utility": 100.0 if pick["is_correct"] else 0.0,
            "would_force": would_force}


def evaluate_all(eval_tasks, models):
    results = {}
    for name in ["base", "majority_vote", "max_confidence", "max_calibrated",
                 "gate_A", "gate_B", "gate_C", "gate_D", "gate_E", "gate_F"]:
        results[name] = []

    for task in eval_tasks:
        results["base"].append(sys_base(task))
        results["majority_vote"].append(sys_majority_vote(task))
        results["max_confidence"].append(sys_max_confidence(task))
        results["max_calibrated"].append(sys_max_calibrated(task, models))

        # Try different thresholds for each gate
        results["gate_A"].append(gate_A_maxcal_margin(task, models, margin=0.15))
        results["gate_B"].append(gate_B_bradley_terry(task, models, margin=0.2))
        results["gate_C"].append(gate_C_direct_z(task, models, tau_r=0.4, tau_b=0.2))
        results["gate_D"].append(gate_D_conditional(task, models, p_base_thresh=0.4))
        results["gate_E"].append(gate_E_ensemble(task, models, tau_r=0.4, tau_b=0.2))
        results["gate_F"].append(gate_F_maxcal_pairwise_confirm(task, models, margin=0.1, pairwise_thresh=0.55))

    return results


def summarize(results, eval_tasks):
    summary = {}
    base_utils = [100.0 if eval_tasks[i]["candidates"][0]["is_correct"] else 0.0
                  for i in range(len(eval_tasks))]
    rescue_opp = sum(1 for t in eval_tasks if t["rescue_available"])

    for name, res_list in results.items():
        utils = [r["utility"] for r in res_list]
        successes = sum(1 for r in res_list if r["correct"])
        improvements = sum(1 for u, bu in zip(utils, base_utils) if u > bu + 0.5)
        regressions = sum(1 for u, bu in zip(utils, base_utils) if u < bu - 0.5)

        n_force = sum(1 for r in res_list if r.get("would_force", False))
        rescues = sum(1 for r, bu in zip(res_list, base_utils)
                      if r.get("would_force", False) and r["utility"] > bu + 0.5)
        breaks = sum(1 for r, bu in zip(res_list, base_utils)
                     if r.get("would_force", False) and r["utility"] < bu - 0.5)

        summary[name] = {
            "mean_utility": np.mean(utils),
            "success_rate": successes / len(res_list),
            "improvements": improvements,
            "regressions": regressions,
            "n_force": n_force,
            "rescues": rescues,
            "breaks": breaks,
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default=str(R6_DIR / "r6_corpus.jsonl"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    R6_DIR.mkdir(parents=True, exist_ok=True)

    tasks = load_corpus(args.corpus)
    print(f"Loaded {len(tasks)} tasks")

    all_results = {}

    for seed in [42, 123, 7, 99, 2024]:
        train_tasks, cal_tasks, eval_tasks = split_tasks(tasks, seed=seed)
        train_records = flatten_candidates(train_tasks)
        feature_keys = get_feature_keys(train_records)

        print(f"\n=== seed={seed} ({len(train_tasks)} dev, {len(eval_tasks)} eval) ===")
        models = train_all_models(train_tasks, cal_tasks, feature_keys)
        results = evaluate_all(eval_tasks, models)
        summary = summarize(results, eval_tasks)

        # Print table
        print(f"{'System':<20} {'Util':>7} {'Acc':>7} {'Imp':>5} {'Reg':>5} {'Force':>6} {'Resc':>5} {'Brk':>5}")
        print("-" * 70)
        for name in ["base", "majority_vote", "max_confidence", "max_calibrated",
                     "gate_A", "gate_B", "gate_C", "gate_D", "gate_E", "gate_F"]:
            s = summary[name]
            print(f"{name:<20} {s['mean_utility']:>7.1f} {s['success_rate']:>6.1%} "
                  f"{s['improvements']:>5} {s['regressions']:>5} "
                  f"{s['n_force']:>6} {s['rescues']:>5} {s['breaks']:>5}")

        all_results[seed] = summary

    # Aggregate
    print(f"\n{'='*90}")
    print(f"  AGGREGATE ACROSS 5 SEEDS")
    print(f"{'='*90}")
    print(f"{'System':<20} {'Mean':>7} {'Std':>7} {'Min':>7} {'Max':>7} {'TotF':>6} {'TotR':>6} {'TotB':>6}")
    print("-" * 90)

    names = ["base", "majority_vote", "max_confidence", "max_calibrated",
             "gate_A", "gate_B", "gate_C", "gate_D", "gate_E", "gate_F"]

    agg = {}
    for name in names:
        utils = [all_results[s][name]["mean_utility"] for s in [42, 123, 7, 99, 2024]]
        tot_f = sum(all_results[s][name]["n_force"] for s in [42, 123, 7, 99, 2024])
        tot_r = sum(all_results[s][name]["rescues"] for s in [42, 123, 7, 99, 2024])
        tot_b = sum(all_results[s][name]["breaks"] for s in [42, 123, 7, 99, 2024])
        agg[name] = {"mean": np.mean(utils), "std": np.std(utils),
                     "min": min(utils), "max": max(utils),
                     "tot_f": tot_f, "tot_r": tot_r, "tot_b": tot_b}
        print(f"{name:<20} {np.mean(utils):>7.1f} {np.std(utils):>7.1f} "
              f"{min(utils):>7.1f} {max(utils):>7.1f} "
              f"{tot_f:>6} {tot_r:>6} {tot_b:>6}")

    # Find best gate
    best_simple = max(agg["majority_vote"]["mean"], agg["max_confidence"]["mean"],
                      agg["max_calibrated"]["mean"])
    print(f"\n  Best simple baseline: {best_simple:.1f}")
    for name in ["gate_A", "gate_B", "gate_C", "gate_D", "gate_E", "gate_F"]:
        diff = agg[name]["mean"] - best_simple
        verdict = "BEATS" if diff > 0.5 else ("MATCHES" if abs(diff) < 0.5 else "WORSE")
        print(f"  {name}: {agg[name]['mean']:.1f} ({diff:+.1f}) {verdict} "
              f"[F={agg[name]['tot_f']}, R={agg[name]['tot_r']}, B={agg[name]['tot_b']}]")

    # Save
    output = {"aggregate": agg, "per_seed": all_results}
    output_path = R6_DIR / "r6_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    main()
