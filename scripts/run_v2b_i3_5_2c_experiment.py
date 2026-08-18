#!/usr/bin/env python3
"""Run the I3.5.2c End-to-End Selective Governor Trajectory Experiment.

Compares three arms on 300 development tasks:
  - OFF: No governor (clean base packet)
  - ALWAYS_ON: Full governor always injected
  - SELECTIVE_FRAME: Gate decides → governor advisory packet → model chooses

Uses deterministic counterbalancing: HMAC(seed, task_id) % 6 selects arm
ordering from the 6 permutations to eliminate temporal confounds.

Primary hypothesis:
  ΔDG_S = DG_OFF - DG_SELECTIVE > 0  (LCB_95 > 0)

Primary utility hypothesis:
  ΔU_S = U_SELECTIVE - U_OFF > 0  (LCB_95 > 0)

Ideal ordering:
  DG_SELECTIVE < DG_OFF < DG_ALWAYS
  U_SELECTIVE > U_OFF > U_ALWAYS

Analysis includes:
  - Task-paired bootstrap CIs for DG and utility
  - McNemar's test on discordant success pairs
  - Per-rule firing distribution
  - Cascade (consecutive intervention) diagnostics
  - Token/latency cost accounting
  - Intervention instrumentation

Usage:
    DEEPSEEK_API_KEY=... python scripts/run_v2b_i3_5_2c_experiment.py \\
        --split structure_dev_v2 --workers 8
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hrm_adaptive_memory.executive.metareasoning_benchmark import (
    load_metareasoning_benchmark,
)
from hrm_adaptive_memory.executive.model_backend import DeepSeekBackend
from hrm_adaptive_memory.executive.metareasoning_utility import MetareasoningUtility
from hrm_adaptive_memory.executive.i3_5_1.receipts import ReceiptLedger
from hrm_adaptive_memory.executive.i3_5_2.modes import GovernorMode
from hrm_adaptive_memory.executive.i3_5_2.trajectory_runner import (
    I352FactorialRunner,
)
from hrm_adaptive_memory.executive.selective_governor import (
    SelectiveGovernorGate,
    RuleBasedInterventionPredictor,
    compute_experiment_identity,
    save_gate_identity,
)


def bootstrap_ci_paired(
    diffs: list[float],
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Task-paired bootstrap CI for the mean of paired differences.

    Returns (mean, lcb, ucb).
    """
    if not diffs:
        return 0.0, 0.0, 0.0
    n = len(diffs)
    mean = sum(diffs) / n

    import random
    rng = random.Random(seed)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = [diffs[rng.randint(0, n - 1)] for _ in range(n)]
        boot_means.append(sum(sample) / n)

    boot_means.sort()
    alpha = (1 - confidence) / 2
    lcb_idx = int(alpha * n_bootstrap)
    ucb_idx = int((1 - alpha) * n_bootstrap)
    lcb = boot_means[lcb_idx]
    ucb = boot_means[min(ucb_idx, n_bootstrap - 1)]
    return mean, lcb, ucb


def mcnemar_test(b_success: int, c_success: int) -> dict[str, Any]:
    """McNemar's test on discordant pairs.

    b = off_only_success count
    c = selective_only_success count
    """
    n = b_success + c_success
    if n == 0:
        return {"statistic": 0.0, "p_value": 1.0, "b": 0, "c": 0, "n_discordant": 0}
    # Exact binomial test (McNemar exact)
    # Under null, c ~ Binomial(n, 0.5)
    # Two-sided p-value
    from math import comb
    k = min(b_success, c_success)
    p_one_sided = sum(comb(n, j) for j in range(k + 1)) / (2 ** n)
    p_two = min(1.0, 2 * p_one_sided)

    # Continuity-corrected chi-square for comparison
    if n > 0:
        chi2 = (abs(b_success - c_success) - 1) ** 2 / n if n > 0 else 0.0
    else:
        chi2 = 0.0

    return {
        "statistic": chi2,
        "p_value_exact": round(p_two, 6),
        "b_off_only_success": b_success,
        "c_selective_only_success": c_success,
        "n_discordant": n,
    }


def compute_dg(success: bool, optimal_success: bool) -> int:
    """Binary terminal-success degradation: 1 if optimal succeeds but agent fails.

    This is the TERMINAL degradation indicator, NOT the full DG.
    The full DG = V_O - V_π uses realized utility, not binary success.
    Since both arms use the same AWARE condition, V_O cancels in the contrast:
        ΔDG = DG_OFF - DG_SEL = V_{π,SEL} - V_{π,OFF} = ΔU
    """
    return 1 if (optimal_success and not success) else 0


def main():
    parser = argparse.ArgumentParser(description="Run I3.5.2c End-to-End Selective Governor Experiment")
    parser.add_argument("--split", default="structure_dev_v2")
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument(
        "--benchmark-manifest",
        default="experiments/v2b_i3_5/manifests/v2b_i3_5_benchmark_manifest_v2.json",
    )
    parser.add_argument("--utility", default="configs/v2b_i3_1_utility_v1.json")
    parser.add_argument("--policy", default="configs/v2b_i3_policy_v1.json")
    parser.add_argument("--output-dir", default="experiments/v2b_i3_5_2/development")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--counterbalance-seed", default="daph_v2b_i3_5_2c_v1")
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    print(f"Loading benchmark from {args.benchmark_manifest}...")
    benchmark = load_metareasoning_benchmark(args.benchmark_manifest, verify_oracle_cache=False)
    split_bm = benchmark.for_split(args.split)
    tasks = split_bm.tasks
    if args.max_tasks is not None:
        tasks = tasks[:args.max_tasks]
    print(f"Loaded {len(tasks)} tasks for split '{args.split}'")

    utility = MetareasoningUtility.from_file(ROOT / args.utility)

    # Compute and save full experiment identity
    print("\nComputing experiment identity...")
    identity = compute_experiment_identity(
        repo_root=ROOT,
        benchmark_manifest_path=args.benchmark_manifest,
        utility_path=args.utility,
        policy_path=args.policy,
    )
    exp_sha = identity["experiment_identity_sha256"]
    gate_sha = identity["gate_identity_sha256"]
    print(f"  Gate identity:        {gate_sha}")
    print(f"  Experiment identity:  {exp_sha}")
    print(f"  Source commit:        {identity.get('source_commit', 'N/A')[:12]}")

    output_dir = Path(args.output_dir) / f"i352c_{exp_sha[:12]}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save identity file
    identity_path = output_dir / "experiment_identity.json"
    identity_path.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")
    print(f"  Identity saved:       {identity_path}")

    modes = (
        GovernorMode.OFF,
        GovernorMode.ALWAYS_ON,
        GovernorMode.SELECTIVE_FRAME,
    )

    print(f"\nRunning I3.5.2c ({len(tasks)} tasks, modes={[m.value for m in modes]}, {args.workers} workers)")
    print(f"Counterbalance seed: {args.counterbalance_seed}")
    print(f"Arm ordering: HMAC('{args.counterbalance_seed}', task_id) % 6")

    def run_one_task(item):
        idx, task = item
        budget = split_bm.budget_for(task)
        worker_backend = DeepSeekBackend()
        worker_gate = SelectiveGovernorGate(predictor=RuleBasedInterventionPredictor())
        worker_runner = I352FactorialRunner(
            backend=worker_backend,
            gate=worker_gate,
            utility=utility,
            experiment_id="v2b_i3_5_2c_experiment_v1",
            experiment_identity_sha256=exp_sha,
            max_steps=24,
            strict_json=True,
            temperature=0.0,
            max_tokens=2048,
        )
        block_result, block_receipts = worker_runner.run_comparison_block_standalone(
            task, budget, modes=modes,
            counterbalance_seed=args.counterbalance_seed,
        )
        return idx, block_result, block_receipts

    results = []
    all_receipts = []
    completed = 0
    t_start = time.monotonic()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_one_task, (i, t)): i
            for i, t in enumerate(tasks)
        }
        for future in as_completed(futures):
            idx, block_result, block_receipts = future.result()
            results.append((idx, block_result, block_receipts))
            completed += 1
            if completed % args.progress_every == 0:
                elapsed = time.monotonic() - t_start
                rate = completed / elapsed
                eta = (len(tasks) - completed) / rate if rate > 0 else 0
                trajs = block_result["trajectories"]
                print(f"  [{completed}/{len(tasks)}] {block_result['task_id']} "
                      f"order={block_result['execution_order']}: "
                      f"OFF={trajs['OFF']['task_success']} "
                      f"ALWAYS={trajs['ALWAYS_ON']['task_success']} "
                      f"SEL={trajs['SELECTIVE_FRAME']['task_success']} "
                      f"(interv={trajs['SELECTIVE_FRAME']['interventions_approved']}) "
                      f"eta={eta:.0f}s")

    results.sort(key=lambda x: x[0])
    block_results = [r[1] for r in results]
    for r in results:
        all_receipts.extend(r[2])

    elapsed = time.monotonic() - t_start
    print(f"\nCompleted {completed} tasks in {elapsed:.0f}s")

    # Build receipt chain
    run_id = f"run_i352c_{exp_sha[:12]}"
    ledger = ReceiptLedger.build_chain_from_receipts(all_receipts, run_id=run_id)
    print(f"Built receipt chain: {ledger.receipt_count} receipts")
    assert ledger.verify_chain(), "Receipt chain verification failed!"

    receipts_path = output_dir / "receipts.jsonl"
    receipts_sha = ledger.save(receipts_path)
    print(f"Receipts saved: {receipts_path} (SHA-256: {receipts_sha[:16]}...)")

    # Save results
    results_payload = {
        "schema": "DAPH_V2B_I3_5_2C_RESULTS_V1",
        "schema_version": 1,
        "experiment_identity_sha256": exp_sha,
        "gate_identity_sha256": gate_sha,
        "counterbalance_seed": args.counterbalance_seed,
        "receipt_chain_root": ledger.receipt_chain_root,
        "source_receipts_sha256": receipts_sha,
        "results": block_results,
    }
    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(results_payload, indent=2, sort_keys=True) + "\n")
    print(f"Results saved: {results_path}")

    # =========================================================================
    # ANALYSIS
    # =========================================================================
    print("\n" + "=" * 78)
    print("V2B-I3.5.2c END-TO-END SELECTIVE GOVERNOR ANALYSIS")
    print("=" * 78)

    # Extract per-task results
    task_data = []
    for block in block_results:
        tid = block["task_id"]
        trajs = block["trajectories"]
        task_data.append({
            "task_id": tid,
            "off_success": trajs["OFF"]["task_success"],
            "off_utility": trajs["OFF"]["realized_utility"],
            "off_calls": trajs["OFF"]["model_calls"],
            "off_tokens": trajs["OFF"]["total_tokens"],
            "off_latency": trajs["OFF"]["total_latency_ms"],
            "off_steps": trajs["OFF"]["total_decisions"],
            "always_success": trajs["ALWAYS_ON"]["task_success"],
            "always_utility": trajs["ALWAYS_ON"]["realized_utility"],
            "always_calls": trajs["ALWAYS_ON"]["model_calls"],
            "always_tokens": trajs["ALWAYS_ON"]["total_tokens"],
            "always_latency": trajs["ALWAYS_ON"]["total_latency_ms"],
            "always_steps": trajs["ALWAYS_ON"]["total_decisions"],
            "sel_success": trajs["SELECTIVE_FRAME"]["task_success"],
            "sel_utility": trajs["SELECTIVE_FRAME"]["realized_utility"],
            "sel_calls": trajs["SELECTIVE_FRAME"]["model_calls"],
            "sel_tokens": trajs["SELECTIVE_FRAME"]["total_tokens"],
            "sel_latency": trajs["SELECTIVE_FRAME"]["total_latency_ms"],
            "sel_steps": trajs["SELECTIVE_FRAME"]["total_decisions"],
            "sel_interventions": trajs["SELECTIVE_FRAME"]["interventions_approved"],
            "sel_max_cascade": trajs["SELECTIVE_FRAME"]["max_consecutive_interventions"],
            "sel_chain_lengths": trajs["SELECTIVE_FRAME"]["intervention_chain_lengths"],
            "sel_intervention_details": trajs["SELECTIVE_FRAME"]["interventions"],
        })

    n_tasks = len(task_data)

    # Success rates
    off_succ = sum(1 for t in task_data if t["off_success"])
    always_succ = sum(1 for t in task_data if t["always_success"])
    sel_succ = sum(1 for t in task_data if t["sel_success"])

    print(f"\n--- Success Rates (N={n_tasks}) ---")
    print(f"  OFF:              {off_succ}/{n_tasks} ({off_succ/n_tasks:.1%})")
    print(f"  ALWAYS_ON:        {always_succ}/{n_tasks} ({always_succ/n_tasks:.1%})")
    print(f"  SELECTIVE_FRAME:  {sel_succ}/{n_tasks} ({sel_succ/n_tasks:.1%})")

    # Decision Degradation: DG = V_O - V_π (frozen I3.5.1 definition)
    # V_π = realized utility (controller value under policy π)
    # Since both arms use the same AWARE condition, V_O cancels in the contrast:
    #   ΔDG = DG_OFF - DG_SEL = (V_O - V_{π,OFF}) - (V_O - V_{π,SEL})
    #       = V_{π,SEL} - V_{π,OFF} = ΔU
    # So the continuous DG contrast is identical to the utility contrast.
    # We also report terminal-success degradation separately.

    # Get optimal success from benchmark for terminal degradation
    task_map = {t.task_id: t for t in split_bm.tasks}
    term_dg_off = []
    term_dg_always = []
    term_dg_sel = []
    for t in task_data:
        task = task_map.get(t["task_id"])
        if task and task.latent.expected_terminal:
            optimal = True
        else:
            optimal = False
        term_dg_off.append(compute_dg(t["off_success"], optimal))
        term_dg_always.append(compute_dg(t["always_success"], optimal))
        term_dg_sel.append(compute_dg(t["sel_success"], optimal))

    mean_term_dg_off = sum(term_dg_off) / n_tasks
    mean_term_dg_always = sum(term_dg_always) / n_tasks
    mean_term_dg_sel = sum(term_dg_sel) / n_tasks

    # Continuous DG = V_O - V_π, using realized utility as V_π
    # V_O is the same for both arms (same AWARE condition), so it cancels
    dg_off = [t["off_utility"] for t in task_data]  # V_{π,OFF} (negated DG)
    dg_always = [t["always_utility"] for t in task_data]
    dg_sel = [t["sel_utility"] for t in task_data]

    mean_dg_off = -sum(dg_off) / n_tasks  # DG = V_O - V_π ≈ -V_π (V_O constant)
    mean_dg_always = -sum(dg_always) / n_tasks
    mean_dg_sel = -sum(dg_sel) / n_tasks

    print(f"\n--- Terminal Success Degradation ---")
    print(f"  TermDG_OFF:        {mean_term_dg_off:.4f} ({sum(term_dg_off)}/{n_tasks})")
    print(f"  TermDG_ALWAYS_ON:  {mean_term_dg_always:.4f} ({sum(term_dg_always)}/{n_tasks})")
    print(f"  TermDG_SELECTIVE:  {mean_term_dg_sel:.4f} ({sum(term_dg_sel)}/{n_tasks})")

    print(f"\n--- Continuous Decision Degradation (DG = V_O - V_π) ---")
    print(f"  DG_OFF:            {mean_dg_off:.4f}  (V_π = {sum(dg_off)/n_tasks:.2f})")
    print(f"  DG_ALWAYS_ON:      {mean_dg_always:.4f}  (V_π = {sum(dg_always)/n_tasks:.2f})")
    print(f"  DG_SELECTIVE:      {mean_dg_sel:.4f}  (V_π = {sum(dg_sel)/n_tasks:.2f})")

    # Primary hypothesis: ΔDG_S = DG_OFF - DG_SEL > 0
    # = (V_O - V_{π,OFF}) - (V_O - V_{π,SEL}) = V_{π,SEL} - V_{π,OFF} = ΔU
    dg_diffs_sel = [dg_sel[i] - dg_off[i] for i in range(n_tasks)]  # V_{π,SEL} - V_{π,OFF}
    dg_diffs_always = [dg_always[i] - dg_off[i] for i in range(n_tasks)]
    mean_dg_sel_vs_off, lcb_dg_sel, ucb_dg_sel = bootstrap_ci_paired(
        dg_diffs_sel, n_bootstrap=args.n_bootstrap)
    mean_dg_always_vs_off, lcb_dg_always, ucb_dg_always = bootstrap_ci_paired(
        dg_diffs_always, n_bootstrap=args.n_bootstrap)

    print(f"\n  ΔDG_S = V_{{π,SEL}} - V_{{π,OFF}}:   mean={mean_dg_sel_vs_off:+.4f} "
          f"LCB_95={lcb_dg_sel:+.4f} UCB_95={ucb_dg_sel:+.4f}")
    print(f"  (equivalently ΔU_S = U_SEL - U_OFF)")
    print(f"  ΔDG_A = V_{{π,ALW}} - V_{{π,OFF}}:   mean={mean_dg_always_vs_off:+.4f} "
          f"LCB_95={lcb_dg_always:+.4f} UCB_95={ucb_dg_always:+.4f}")

    # Terminal success difference
    term_diffs_sel = [term_dg_off[i] - term_dg_sel[i] for i in range(n_tasks)]
    mean_term_diff_sel, lcb_term_sel, ucb_term_sel = bootstrap_ci_paired(
        term_diffs_sel, n_bootstrap=args.n_bootstrap)
    print(f"\n  ΔTermDG_S = TermDG_OFF - TermDG_SEL: mean={mean_term_diff_sel:+.4f} "
          f"LCB_95={lcb_term_sel:+.4f} UCB_95={ucb_term_sel:+.4f}")

    dg_sel_hypothesis = "SUPPORTED" if lcb_dg_sel > 0 else "NOT_SUPPORTED"

    # Utility
    u_off = [t["off_utility"] for t in task_data]
    u_always = [t["always_utility"] for t in task_data]
    u_sel = [t["sel_utility"] for t in task_data]

    mean_u_off = sum(u_off) / n_tasks
    mean_u_always = sum(u_always) / n_tasks
    mean_u_sel = sum(u_sel) / n_tasks

    print(f"\n--- Realized Utility ---")
    print(f"  U_OFF:             {mean_u_off:.2f}")
    print(f"  U_ALWAYS_ON:       {mean_u_always:.2f}")
    print(f"  U_SELECTIVE:       {mean_u_sel:.2f}")

    u_diffs_sel = [u_sel[i] - u_off[i] for i in range(n_tasks)]
    u_diffs_always = [u_always[i] - u_off[i] for i in range(n_tasks)]
    mean_u_sel_vs_off, lcb_u_sel, ucb_u_sel = bootstrap_ci_paired(
        u_diffs_sel, n_bootstrap=args.n_bootstrap)
    mean_u_always_vs_off, lcb_u_always, ucb_u_always = bootstrap_ci_paired(
        u_diffs_always, n_bootstrap=args.n_bootstrap)

    print(f"\n  ΔU_S = U_SEL - U_OFF:       mean={mean_u_sel_vs_off:+.2f} "
          f"LCB_95={lcb_u_sel:+.2f} UCB_95={ucb_u_sel:+.2f}")
    print(f"  ΔU_A = U_ALWAYS - U_OFF:    mean={mean_u_always_vs_off:+.2f} "
          f"LCB_95={lcb_u_always:+.2f} UCB_95={ucb_u_always:+.2f}")

    u_sel_hypothesis = "SUPPORTED" if lcb_u_sel > 0 else "NOT_SUPPORTED"

    # Ideal ordering check
    print(f"\n--- Ideal Ordering Check ---")
    print(f"  DG: DG_SEL < DG_OFF < DG_ALWAYS?")
    print(f"       {mean_dg_sel:.4f} < {mean_dg_off:.4f} < {mean_dg_always:.4f} "
          f"→ {'YES' if mean_dg_sel < mean_dg_off < mean_dg_always else 'NO'}")
    print(f"  U:  U_SEL > U_OFF > U_ALWAYS?")
    print(f"       {mean_u_sel:.2f} > {mean_u_off:.2f} > {mean_u_always:.2f} "
          f"→ {'YES' if mean_u_sel > mean_u_off > mean_u_always else 'NO'}")

    # McNemar's test (OFF vs SELECTIVE)
    both_succ = sum(1 for t in task_data if t["off_success"] and t["sel_success"])
    both_fail = sum(1 for t in task_data if not t["off_success"] and not t["sel_success"])
    off_only = sum(1 for t in task_data if t["off_success"] and not t["sel_success"])
    sel_only = sum(1 for t in task_data if not t["off_success"] and t["sel_success"])

    mcnemar_sel = mcnemar_test(off_only, sel_only)

    print(f"\n--- McNemar's Test (OFF vs SELECTIVE) ---")
    print(f"  both_success:            {both_succ}")
    print(f"  both_fail:               {both_fail}")
    print(f"  off_only_success:        {off_only}")
    print(f"  selective_only_success:  {sel_only}")
    print(f"  p-value (exact):         {mcnemar_sel.get('p_value_exact', mcnemar_sel.get('p_value', 'N/A'))}")

    # McNemar's test (OFF vs ALWAYS)
    off_only_a = sum(1 for t in task_data if t["off_success"] and not t["always_success"])
    always_only = sum(1 for t in task_data if not t["off_success"] and t["always_success"])
    mcnemar_always = mcnemar_test(off_only_a, always_only)

    print(f"\n--- McNemar's Test (OFF vs ALWAYS_ON) ---")
    print(f"  off_only_success:        {off_only_a}")
    print(f"  always_only_success:     {always_only}")
    print(f"  p-value (exact):         {mcnemar_always.get('p_value_exact', mcnemar_always.get('p_value', 'N/A'))}")

    # Intervention statistics
    total_interventions = sum(t["sel_interventions"] for t in task_data)
    tasks_with_intervention = sum(1 for t in task_data if t["sel_interventions"] > 0)
    int_rate = total_interventions / sum(t["sel_steps"] for t in task_data) if task_data else 0

    print(f"\n--- Intervention Statistics ---")
    print(f"  Total interventions:       {total_interventions}")
    print(f"  Tasks with intervention:   {tasks_with_intervention}/{n_tasks}")
    print(f"  Intervention rate (per step): {int_rate:.1%}")

    # Rule firing distribution
    rule_counter = Counter()
    rule_outcomes = defaultdict(list)
    rule_actions = defaultdict(list)
    for t in task_data:
        for iv in t["sel_intervention_details"]:
            # Extract rule name from gate_reason
            reason = iv["gate_reason"]
            if "POST_VERIFY" in reason:
                rule_name = "POST_VERIFY"
            elif "POST_SEARCH" in reason:
                rule_name = "POST_SEARCH"
            elif "STEP0" in reason:
                rule_name = "STEP0_HAZARD"
            elif "ALWAYS_ON" in reason:
                rule_name = "ALWAYS_ON"
            else:
                rule_name = reason.split(":")[-1] if ":" in reason else reason
            rule_counter[rule_name] += 1
            rule_outcomes[rule_name].append(iv["outcome"])
            rule_actions[rule_name].append(iv["model_action"])

    print(f"\n--- Rule Firing Distribution ---")
    for rule, cnt in rule_counter.most_common():
        actions = Counter(rule_actions[rule])
        outcomes = Counter(rule_outcomes[rule])
        print(f"  {rule:<40}: {cnt:>4} interventions")
        print(f"    actions: {dict(actions.most_common(3))}")
        print(f"    outcomes: {dict(outcomes.most_common(3))}")

    # Cascade diagnostics
    all_chains = []
    for t in task_data:
        all_chains.extend(t["sel_chain_lengths"])
    chain_counter = Counter(all_chains)

    print(f"\n--- Cascade Diagnostics ---")
    print(f"  Total intervention chains: {len(all_chains)}")
    for length in sorted(chain_counter.keys()):
        cnt = chain_counter[length]
        print(f"    Chain length {length}: {cnt} chains")
    max_cascade = max(t["sel_max_cascade"] for t in task_data) if task_data else 0
    print(f"  Max consecutive interventions: {max_cascade}")

    # Cost accounting
    mean_calls_off = sum(t["off_calls"] for t in task_data) / n_tasks
    mean_calls_always = sum(t["always_calls"] for t in task_data) / n_tasks
    mean_calls_sel = sum(t["sel_calls"] for t in task_data) / n_tasks
    mean_tokens_off = sum(t["off_tokens"] for t in task_data) / n_tasks
    mean_tokens_always = sum(t["always_tokens"] for t in task_data) / n_tasks
    mean_tokens_sel = sum(t["sel_tokens"] for t in task_data) / n_tasks
    mean_latency_off = sum(t["off_latency"] for t in task_data) / n_tasks
    mean_latency_always = sum(t["always_latency"] for t in task_data) / n_tasks
    mean_latency_sel = sum(t["sel_latency"] for t in task_data) / n_tasks

    print(f"\n--- Cost Accounting ---")
    print(f"  NOTE: Utility is charged by the executor (simulated resource consumption).")
    print(f"  Model tokens are telemetry, NOT directly charged by MetareasoningUtility.")
    print(f"  {'Arm':<20} {'Calls':>6} {'Tokens':>8} {'Latency(ms)':>12} {'Steps':>6} {'Utility':>10}")
    print(f"  {'OFF':<20} {mean_calls_off:>6.1f} {mean_tokens_off:>8.0f} {mean_latency_off:>12.0f} {sum(t['off_steps'] for t in task_data)/n_tasks:>6.1f} {mean_u_off:>10.2f}")
    print(f"  {'ALWAYS_ON':<20} {mean_calls_always:>6.1f} {mean_tokens_always:>8.0f} {mean_latency_always:>12.0f} {sum(t['always_steps'] for t in task_data)/n_tasks:>6.1f} {mean_u_always:>10.2f}")
    print(f"  {'SELECTIVE_FRAME':<20} {mean_calls_sel:>6.1f} {mean_tokens_sel:>8.0f} {mean_latency_sel:>12.0f} {sum(t['sel_steps'] for t in task_data)/n_tasks:>6.1f} {mean_u_sel:>10.2f}")

    delta_tokens = mean_tokens_sel - mean_tokens_off
    delta_u = mean_u_sel - mean_u_off
    delta_steps = sum(t['sel_steps'] for t in task_data)/n_tasks - sum(t['off_steps'] for t in task_data)/n_tasks

    print(f"\n  ΔU (SEL - OFF):         {delta_u:+.2f}  (from longer/costlier executor trajectories)")
    print(f"  Δsteps (SEL - OFF):     {delta_steps:+.1f}  (executor action steps)")
    print(f"  Δtokens (SEL - OFF):    {delta_tokens:+.0f}  (model token consumption, telemetry only)")
    print(f"  ΔU is caused by executor resource costs, NOT by model token consumption.")

    # Development acceptance gates
    print(f"\n{'='*78}")
    print("DEVELOPMENT ACCEPTANCE GATES")
    print(f"{'='*78}")

    gates = {
        "G1_validity": {
            "description": "Receipt chain valid, all 3 arms complete, no errors",
            "passed": ledger.verify_chain() and all(
                t["off_calls"] > 0 and t["always_calls"] > 0 and t["sel_calls"] > 0
                for t in task_data
            ),
        },
        "G2_nontrivial_intervention": {
            "description": "intervention_rate > 0",
            "passed": total_interventions > 0,
            "value": total_interventions,
        },
        "G3_primary_dg": {
            "description": "mean(ΔDG_S) > 0",
            "passed": mean_dg_sel_vs_off > 0,
            "value": round(mean_dg_sel_vs_off, 4),
        },
        "G4_primary_utility": {
            "description": "mean(ΔU_S) > 0",
            "passed": mean_u_sel_vs_off > 0,
            "value": round(mean_u_sel_vs_off, 4),
        },
        "G5_always_on_dominance": {
            "description": "U_SEL > U_ALWAYS",
            "passed": mean_u_sel > mean_u_always,
            "value": f"U_SEL={mean_u_sel:.2f} > U_ALWAYS={mean_u_always:.2f}",
        },
        "G6_no_catastrophic_harm": {
            "description": "off_only_success <= selective_only_success",
            "passed": off_only <= sel_only,
            "value": f"off_only={off_only} <= sel_only={sel_only}",
        },
        "G7_sequential_stability": {
            "description": "No runaway cascades (max_consecutive <= 5)",
            "passed": max_cascade <= 5,
            "value": max_cascade,
        },
    }

    all_passed = True
    for gate_name, gate_info in gates.items():
        status = "PASS" if gate_info["passed"] else "FAIL"
        if not gate_info["passed"]:
            all_passed = False
        print(f"  {gate_name}: {status} — {gate_info['description']}")
        if "value" in gate_info:
            print(f"    value: {gate_info['value']}")

    print(f"\n  OVERALL: {'ALL GATES PASSED' if all_passed else 'SOME GATES FAILED'}")
    print(f"\n  NOTE: Development CIs are NOT confirmatory because rules were")
    print(f"  derived from the same development corpus. Freeze before validation.")

    # Save analysis report
    analysis = {
        "schema": "DAPH_V2B_I3_5_2C_ANALYSIS_V1",
        "schema_version": 1,
        "experiment_identity_sha256": exp_sha,
        "n_tasks": n_tasks,
        "success_rates": {
            "OFF": {"success": off_succ, "total": n_tasks, "rate": round(off_succ/n_tasks, 4)},
            "ALWAYS_ON": {"success": always_succ, "total": n_tasks, "rate": round(always_succ/n_tasks, 4)},
            "SELECTIVE_FRAME": {"success": sel_succ, "total": n_tasks, "rate": round(sel_succ/n_tasks, 4)},
        },
        "decision_degradation": {
            "DG_OFF": round(mean_dg_off, 4),
            "DG_ALWAYS_ON": round(mean_dg_always, 4),
            "DG_SELECTIVE": round(mean_dg_sel, 4),
            "delta_DG_sel_vs_off": {
                "mean": round(mean_dg_sel_vs_off, 4),
                "lcb_95": round(lcb_dg_sel, 4),
                "ucb_95": round(ucb_dg_sel, 4),
            },
            "delta_DG_always_vs_off": {
                "mean": round(mean_dg_always_vs_off, 4),
                "lcb_95": round(lcb_dg_always, 4),
                "ucb_95": round(ucb_dg_always, 4),
            },
        },
        "utility": {
            "U_OFF": round(mean_u_off, 4),
            "U_ALWAYS_ON": round(mean_u_always, 4),
            "U_SELECTIVE": round(mean_u_sel, 4),
            "delta_U_sel_vs_off": {
                "mean": round(mean_u_sel_vs_off, 4),
                "lcb_95": round(lcb_u_sel, 4),
                "ucb_95": round(ucb_u_sel, 4),
            },
            "delta_U_always_vs_off": {
                "mean": round(mean_u_always_vs_off, 4),
                "lcb_95": round(lcb_u_always, 4),
                "ucb_95": round(ucb_u_always, 4),
            },
        },
        "mcnemar_sel_vs_off": mcnemar_sel,
        "mcnemar_always_vs_off": mcnemar_always,
        "intervention_stats": {
            "total_interventions": total_interventions,
            "tasks_with_intervention": tasks_with_intervention,
            "intervention_rate": round(int_rate, 4),
        },
        "rule_firing_distribution": dict(rule_counter.most_common()),
        "cascade": {
            "total_chains": len(all_chains),
            "chain_length_distribution": dict(chain_counter),
            "max_consecutive": max_cascade,
        },
        "cost": {
            "OFF": {"calls": round(mean_calls_off, 2), "tokens": round(mean_tokens_off, 0), "latency_ms": round(mean_latency_off, 0)},
            "ALWAYS_ON": {"calls": round(mean_calls_always, 2), "tokens": round(mean_tokens_always, 0), "latency_ms": round(mean_latency_always, 0)},
            "SELECTIVE": {"calls": round(mean_calls_sel, 2), "tokens": round(mean_tokens_sel, 0), "latency_ms": round(mean_latency_sel, 0)},
            "delta_tokens_sel_vs_off": round(delta_tokens, 0),
            "delta_u_per_delta_token": round(cost_adjusted, 6) if cost_adjusted != float('inf') else None,
        },
        "acceptance_gates": {k: v["passed"] for k, v in gates.items()},
        "all_gates_passed": all_passed,
        "hypothesis_results": {
            "primary_dg_hypothesis": dg_sel_hypothesis,
            "primary_utility_hypothesis": u_sel_hypothesis,
        },
        "scientific_caveats": [
            "Development CIs are NOT confirmatory — rules were derived from this corpus.",
            "Must freeze all components before validation.",
            "Do not tune against validation and then proceed to held-out.",
        ],
    }

    analysis_path = output_dir / "analysis.json"
    analysis_path.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    print(f"\nAnalysis saved: {analysis_path}")


if __name__ == "__main__":
    main()
