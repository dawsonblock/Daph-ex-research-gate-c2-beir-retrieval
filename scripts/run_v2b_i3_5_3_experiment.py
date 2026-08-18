#!/usr/bin/env python3
"""Run the I3.5.3 End-to-End Q^{π_B}-Based Selective Governor Experiment.

Same structure as I3.5.2c but uses the Q^{π_B} regression gate instead of
the Q*-based rule gate.

Three arms on 300 development tasks:
  - OFF: No governor (clean base packet)
  - ALWAYS_ON: Full governor always injected
  - SELECTIVE_QPIB: Q^{π_B} gate decides → governor advisory packet → model

The Q^{π_B} gate was trained from:
  - OFF trajectory data (Q^{π_B}(s, a_taken) = realized utility)
  - I3.5.2d fork data (Q^{π_B}(s, a_G) = fork B realized utility)

Primary hypothesis:
  ΔDG_S = V_{π,SEL} - V_{π,OFF} > 0  (LCB_95 > 0)

Usage:
    DEEPSEEK_API_KEY=... python scripts/run_v2b_i3_5_3_experiment.py \\
        --split structure_dev_v2 --workers 8 \\
        --gate-model experiments/v2b_i3_5_2/development/i352d/q_pib_gate_v1.pkl
"""
from __future__ import annotations

import argparse
import json
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
from hrm_adaptive_memory.executive.selective_governor.q_pib_predictor import (
    QPiBInterventionPredictor,
)


def bootstrap_ci_paired(
    diffs: list[float],
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
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
    return mean, boot_means[lcb_idx], boot_means[min(ucb_idx, n_bootstrap - 1)]


def mcnemar_test(b: int, c: int) -> dict[str, Any]:
    n = b + c
    if n == 0:
        return {"statistic": 0.0, "p_value_exact": 1.0, "b": 0, "c": 0, "n_discordant": 0}
    from math import comb
    k = min(b, c)
    p_one = sum(comb(n, j) for j in range(k + 1)) / (2 ** n)
    p_two = min(1.0, 2 * p_one)
    chi2 = (abs(b - c) - 1) ** 2 / n if n > 0 else 0.0
    return {
        "statistic": chi2,
        "p_value_exact": round(p_two, 6),
        "b_off_only": b,
        "c_sel_only": c,
        "n_discordant": n,
    }


def main():
    parser = argparse.ArgumentParser(description="Run I3.5.3 Q^{π_B} Selective Governor Experiment")
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
    parser.add_argument("--counterbalance-seed", default="daph_v2b_i3_5_3_v1")
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument(
        "--gate-model",
        default="experiments/v2b_i3_5_2/development/i352d/q_pib_gate_v1.pkl",
    )
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    # Load the Q^{π_B} gate model
    print(f"Loading Q^{{π_B}} gate model from {args.gate_model}...")
    q_pib_predictor = QPiBInterventionPredictor.load(args.gate_model)
    print(f"  Model loaded: {type(q_pib_predictor.model).__name__}")
    print(f"  Threshold: delta_q > {q_pib_predictor.delta_q_threshold}")
    print(f"  Max harm prob: {q_pib_predictor.max_harm_probability}")
    print(f"  Min confidence: {q_pib_predictor.min_confidence}")

    print(f"\nLoading benchmark from {args.benchmark_manifest}...")
    benchmark = load_metareasoning_benchmark(args.benchmark_manifest, verify_oracle_cache=False)
    split_bm = benchmark.for_split(args.split)
    tasks = split_bm.tasks
    if args.max_tasks is not None:
        tasks = tasks[:args.max_tasks]
    print(f"Loaded {len(tasks)} tasks for split '{args.split}'")

    utility = MetareasoningUtility.from_file(ROOT / args.utility)

    # Compute experiment identity
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

    output_dir = Path(args.output_dir) / f"i353_{exp_sha[:12]}"
    output_dir.mkdir(parents=True, exist_ok=True)

    identity_path = output_dir / "experiment_identity.json"
    identity_path.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")
    print(f"  Identity saved:       {identity_path}")

    # Save gate model info
    gate_info = {
        "gate_type": "QPiBInterventionPredictor",
        "gate_model_path": args.gate_model,
        "delta_q_threshold": q_pib_predictor.delta_q_threshold,
        "max_harm_probability": q_pib_predictor.max_harm_probability,
        "min_confidence": q_pib_predictor.min_confidence,
        "model_class": type(q_pib_predictor.model).__name__,
    }
    (output_dir / "gate_info.json").write_text(
        json.dumps(gate_info, indent=2, sort_keys=True) + "\n")

    modes = (
        GovernorMode.OFF,
        GovernorMode.ALWAYS_ON,
        GovernorMode.SELECTIVE_FRAME,  # SELECTIVE_FRAME with Q^{π_B} gate
    )

    print(f"\nRunning I3.5.3 ({len(tasks)} tasks, modes={[m.value for m in modes]}, {args.workers} workers)")
    print(f"Gate: Q^{{π_B}} regression (GradientBoostingRegressor)")
    print(f"Counterbalance seed: {args.counterbalance_seed}")

    def run_one_task(item):
        idx, task = item
        budget = split_bm.budget_for(task)
        worker_backend = DeepSeekBackend()
        # Use Q^{π_B} predictor instead of rule-based
        worker_gate = SelectiveGovernorGate(predictor=q_pib_predictor)
        worker_runner = I352FactorialRunner(
            backend=worker_backend,
            gate=worker_gate,
            utility=utility,
            experiment_id="v2b_i3_5_3_experiment_v1",
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
    run_id = f"run_i353_{exp_sha[:12]}"
    ledger = ReceiptLedger.build_chain_from_receipts(all_receipts, run_id=run_id)
    print(f"Built receipt chain: {ledger.receipt_count} receipts")
    assert ledger.verify_chain(), "Receipt chain verification failed!"

    receipts_path = output_dir / "receipts.jsonl"
    receipts_sha = ledger.save(receipts_path)
    print(f"Receipts saved: {receipts_path} (SHA-256: {receipts_sha[:16]}...)")

    # Save results
    results_payload = {
        "schema": "DAPH_V2B_I3_5_3_RESULTS_V1",
        "schema_version": 1,
        "experiment_identity_sha256": exp_sha,
        "gate_identity_sha256": gate_sha,
        "gate_type": "QPiBInterventionPredictor",
        "gate_model_path": args.gate_model,
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
    print("V2B-I3.5.3 Q^{π_B} SELECTIVE GOVERNOR ANALYSIS")
    print("=" * 78)

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
            "off_steps": trajs["OFF"]["total_decisions"],
            "always_success": trajs["ALWAYS_ON"]["task_success"],
            "always_utility": trajs["ALWAYS_ON"]["realized_utility"],
            "always_calls": trajs["ALWAYS_ON"]["model_calls"],
            "always_tokens": trajs["ALWAYS_ON"]["total_tokens"],
            "always_steps": trajs["ALWAYS_ON"]["total_decisions"],
            "sel_success": trajs["SELECTIVE_FRAME"]["task_success"],
            "sel_utility": trajs["SELECTIVE_FRAME"]["realized_utility"],
            "sel_calls": trajs["SELECTIVE_FRAME"]["model_calls"],
            "sel_tokens": trajs["SELECTIVE_FRAME"]["total_tokens"],
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
    print(f"  SELECTIVE_QPIB:   {sel_succ}/{n_tasks} ({sel_succ/n_tasks:.1%})")

    # Continuous DG = V_O - V_π, contrast = V_{π,SEL} - V_{π,OFF} = ΔU
    u_off = [t["off_utility"] for t in task_data]
    u_always = [t["always_utility"] for t in task_data]
    u_sel = [t["sel_utility"] for t in task_data]

    mean_u_off = sum(u_off) / n_tasks
    mean_u_always = sum(u_always) / n_tasks
    mean_u_sel = sum(u_sel) / n_tasks

    print(f"\n--- Realized Utility ---")
    print(f"  U_OFF:             {mean_u_off:.4f}")
    print(f"  U_ALWAYS_ON:       {mean_u_always:.4f}")
    print(f"  U_SELECTIVE_QPIB:  {mean_u_sel:.4f}")

    u_diffs_sel = [u_sel[i] - u_off[i] for i in range(n_tasks)]
    u_diffs_always = [u_always[i] - u_off[i] for i in range(n_tasks)]
    mean_u_sel_vs_off, lcb_u_sel, ucb_u_sel = bootstrap_ci_paired(
        u_diffs_sel, n_bootstrap=args.n_bootstrap)
    mean_u_always_vs_off, lcb_u_always, ucb_u_always = bootstrap_ci_paired(
        u_diffs_always, n_bootstrap=args.n_bootstrap)

    print(f"\n  ΔU_S = U_SEL - U_OFF:       mean={mean_u_sel_vs_off:+.4f} "
          f"LCB_95={lcb_u_sel:+.4f} UCB_95={ucb_u_sel:+.4f}")
    print(f"  ΔU_A = U_ALWAYS - U_OFF:    mean={mean_u_always_vs_off:+.4f} "
          f"LCB_95={lcb_u_always:+.4f} UCB_95={ucb_u_always:+.4f}")

    print(f"\n  ΔDG_S = V_{{π,SEL}} - V_{{π,OFF}} = ΔU_S = {mean_u_sel_vs_off:+.4f}")
    print(f"  95% CI: [{lcb_u_sel:+.4f}, {ucb_u_sel:+.4f}]")

    dg_hypothesis = "SUPPORTED" if lcb_u_sel > 0 else "NOT_SUPPORTED"
    u_hypothesis = "SUPPORTED" if lcb_u_sel > 0 else "NOT_SUPPORTED"

    # McNemar
    both_succ = sum(1 for t in task_data if t["off_success"] and t["sel_success"])
    both_fail = sum(1 for t in task_data if not t["off_success"] and not t["sel_success"])
    off_only = sum(1 for t in task_data if t["off_success"] and not t["sel_success"])
    sel_only = sum(1 for t in task_data if not t["off_success"] and t["sel_success"])

    mcnemar_sel = mcnemar_test(off_only, sel_only)

    print(f"\n--- McNemar's Test (OFF vs SELECTIVE_QPIB) ---")
    print(f"  both_success:            {both_succ}")
    print(f"  both_fail:               {both_fail}")
    print(f"  off_only_success:        {off_only}")
    print(f"  selective_only_success:  {sel_only}")
    print(f"  p-value (exact):         {mcnemar_sel['p_value_exact']}")

    off_only_a = sum(1 for t in task_data if t["off_success"] and not t["always_success"])
    always_only = sum(1 for t in task_data if not t["off_success"] and t["always_success"])
    mcnemar_always = mcnemar_test(off_only_a, always_only)

    # Intervention statistics
    total_interventions = sum(t["sel_interventions"] for t in task_data)
    tasks_with_intervention = sum(1 for t in task_data if t["sel_interventions"] > 0)
    total_steps = sum(t["sel_steps"] for t in task_data)
    int_rate = total_interventions / total_steps if total_steps else 0

    print(f"\n--- Intervention Statistics ---")
    print(f"  Total interventions:       {total_interventions}")
    print(f"  Tasks with intervention:   {tasks_with_intervention}/{n_tasks}")
    print(f"  Intervention rate (per step): {int_rate:.1%}")

    # Rule/gate reason firing
    reason_counter = Counter()
    for t in task_data:
        for iv in t["sel_intervention_details"]:
            reason = iv["gate_reason"]
            # Extract QPIB reason
            if "QPI" in reason:
                reason_counter["QPIB_GATE"] += 1
            else:
                reason_counter[reason.split(":")[-1] if ":" in reason else reason] += 1

    print(f"\n--- Gate Reason Distribution ---")
    for reason, cnt in reason_counter.most_common():
        print(f"  {reason:<40}: {cnt}")

    # Cascade diagnostics
    all_chains = []
    for t in task_data:
        all_chains.extend(t["sel_chain_lengths"])
    chain_counter = Counter(all_chains)

    print(f"\n--- Cascade Diagnostics ---")
    print(f"  Total intervention chains: {len(all_chains)}")
    for length in sorted(chain_counter.keys()):
        print(f"    Chain length {length}: {chain_counter[length]} chains")
    max_cascade = max((t["sel_max_cascade"] for t in task_data), default=0)
    print(f"  Max consecutive interventions: {max_cascade}")

    # Cost accounting
    mean_calls_off = sum(t["off_calls"] for t in task_data) / n_tasks
    mean_calls_always = sum(t["always_calls"] for t in task_data) / n_tasks
    mean_calls_sel = sum(t["sel_calls"] for t in task_data) / n_tasks
    mean_tokens_off = sum(t["off_tokens"] for t in task_data) / n_tasks
    mean_tokens_always = sum(t["always_tokens"] for t in task_data) / n_tasks
    mean_tokens_sel = sum(t["sel_tokens"] for t in task_data) / n_tasks
    mean_steps_off = sum(t["off_steps"] for t in task_data) / n_tasks
    mean_steps_always = sum(t["always_steps"] for t in task_data) / n_tasks
    mean_steps_sel = sum(t["sel_steps"] for t in task_data) / n_tasks

    print(f"\n--- Cost Accounting ---")
    print(f"  NOTE: Utility loss comes from executor trajectory costs, NOT model tokens.")
    print(f"  {'Arm':<20} {'Steps':>6} {'Calls':>6} {'Tokens':>8} {'Utility':>10}")
    print(f"  {'OFF':<20} {mean_steps_off:>6.1f} {mean_calls_off:>6.1f} {mean_tokens_off:>8.0f} {mean_u_off:>10.2f}")
    print(f"  {'ALWAYS_ON':<20} {mean_steps_always:>6.1f} {mean_calls_always:>6.1f} {mean_tokens_always:>8.0f} {mean_u_always:>10.2f}")
    print(f"  {'SELECTIVE_QPIB':<20} {mean_steps_sel:>6.1f} {mean_calls_sel:>6.1f} {mean_tokens_sel:>8.0f} {mean_u_sel:>10.2f}")

    delta_u = mean_u_sel - mean_u_off
    delta_steps = mean_steps_sel - mean_steps_off
    delta_tokens = mean_tokens_sel - mean_tokens_off
    print(f"\n  ΔU (SEL - OFF):     {delta_u:+.2f}")
    print(f"  Δsteps (SEL - OFF): {delta_steps:+.1f}")
    print(f"  Δtokens (SEL - OFF): {delta_tokens:+.0f}  (telemetry)")

    # Acceptance gates
    print(f"\n{'='*78}")
    print("DEVELOPMENT ACCEPTANCE GATES")
    print(f"{'='*78}")

    gates = {
        "G1_validity": {
            "description": "Receipt chain valid, all 3 arms complete",
            "passed": ledger.verify_chain() and all(
                t["off_calls"] > 0 and t["always_calls"] > 0 and t["sel_calls"] > 0
                for t in task_data),
        },
        "G2_nontrivial_intervention": {
            "description": "intervention_rate > 0",
            "passed": total_interventions > 0,
            "value": total_interventions,
        },
        "G3_primary_dg": {
            "description": "ΔDG_S > 0 (LCB > 0)",
            "passed": lcb_u_sel > 0,
            "value": f"ΔDG={mean_u_sel_vs_off:+.4f}, LCB={lcb_u_sel:+.4f}",
        },
        "G4_primary_utility": {
            "description": "ΔU_S > 0 (LCB > 0)",
            "passed": lcb_u_sel > 0,
            "value": f"ΔU={mean_u_sel_vs_off:+.4f}, LCB={lcb_u_sel:+.4f}",
        },
        "G5_always_on_dominance": {
            "description": "U_SEL > U_ALWAYS",
            "passed": mean_u_sel > mean_u_always,
            "value": f"U_SEL={mean_u_sel:.2f} > U_ALWAYS={mean_u_always:.2f}",
        },
        "G6_no_catastrophic_harm": {
            "description": "off_only <= sel_only",
            "passed": off_only <= sel_only,
            "value": f"off_only={off_only} <= sel_only={sel_only}",
        },
        "G7_sequential_stability": {
            "description": "max_consecutive <= 5",
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

    n_passed = sum(1 for g in gates.values() if g["passed"])
    print(f"\n  OVERALL: {n_passed}/{len(gates)} gates passed")

    # Comparison with I3.5.2c
    print(f"\n{'='*78}")
    print("COMPARISON WITH I3.5.2c (Q*-based gate)")
    print(f"{'='*78}")
    print(f"  I3.5.2c (Q* gate):     ΔU = -3.28, interventions = 536, success = 83/300")
    print(f"  I3.5.3 (Q^{{π_B}} gate): ΔU = {mean_u_sel_vs_off:+.2f}, interventions = {total_interventions}, success = {sel_succ}/{n_tasks}")

    if total_interventions < 536:
        print(f"  → Q^{{π_B}} gate reduced interventions by {536 - total_interventions} ({(536-total_interventions)/536:.0%})")
    if mean_u_sel_vs_off > -3.28:
        print(f"  → Q^{{π_B}} gate improved utility by {mean_u_sel_vs_off - (-3.28):+.2f} vs Q* gate")
    if sel_succ > 83:
        print(f"  → Q^{{π_B}} gate improved success by {sel_succ - 83} tasks")
    elif sel_succ == 83:
        print(f"  → Q^{{π_B}} gate preserved success (same as Q* gate)")
    else:
        print(f"  → Q^{{π_B}} gate changed success by {sel_succ - 83:+d} tasks")

    # Save analysis
    analysis = {
        "schema": "DAPH_V2B_I3_5_3_ANALYSIS_V1",
        "experiment_identity_sha256": exp_sha,
        "gate_type": "QPiBInterventionPredictor",
        "gate_model_path": args.gate_model,
        "n_tasks": n_tasks,
        "success_rates": {
            "OFF": {"success": off_succ, "total": n_tasks, "rate": round(off_succ/n_tasks, 4)},
            "ALWAYS_ON": {"success": always_succ, "total": n_tasks, "rate": round(always_succ/n_tasks, 4)},
            "SELECTIVE_QPIB": {"success": sel_succ, "total": n_tasks, "rate": round(sel_succ/n_tasks, 4)},
        },
        "utility": {
            "U_OFF": round(mean_u_off, 4),
            "U_ALWAYS_ON": round(mean_u_always, 4),
            "U_SELECTIVE_QPIB": round(mean_u_sel, 4),
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
        "cascade": {
            "total_chains": len(all_chains),
            "chain_length_distribution": dict(chain_counter),
            "max_consecutive": max_cascade,
        },
        "cost": {
            "OFF": {"steps": round(mean_steps_off, 2), "calls": round(mean_calls_off, 2),
                    "tokens": round(mean_tokens_off, 0), "utility": round(mean_u_off, 4)},
            "ALWAYS_ON": {"steps": round(mean_steps_always, 2), "calls": round(mean_calls_always, 2),
                          "tokens": round(mean_tokens_always, 0), "utility": round(mean_u_always, 4)},
            "SELECTIVE_QPIB": {"steps": round(mean_steps_sel, 2), "calls": round(mean_calls_sel, 2),
                               "tokens": round(mean_tokens_sel, 0), "utility": round(mean_u_sel, 4)},
            "delta_steps": round(delta_steps, 2),
            "delta_tokens": round(delta_tokens, 0),
            "delta_u": round(delta_u, 4),
        },
        "acceptance_gates": {k: v["passed"] for k, v in gates.items()},
        "n_gates_passed": n_passed,
        "hypothesis_results": {
            "primary_dg_hypothesis": dg_hypothesis,
            "primary_utility_hypothesis": u_hypothesis,
        },
        "comparison_with_i352c": {
            "i352c_delta_u": -3.2814,
            "i352c_interventions": 536,
            "i352c_success": 83,
            "i353_delta_u": round(mean_u_sel_vs_off, 4),
            "i353_interventions": total_interventions,
            "i353_success": sel_succ,
        },
    }

    analysis_path = output_dir / "analysis.json"
    analysis_path.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    print(f"\nAnalysis saved: {analysis_path}")


if __name__ == "__main__":
    main()
