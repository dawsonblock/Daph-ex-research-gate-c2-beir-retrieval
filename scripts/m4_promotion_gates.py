#!/usr/bin/env python3
"""Check M4R2 promotion gates against user-specified criteria.

Evaluates whether the M4R2 results meet the promotion gates for
eventual hard authority qualification. ALL gates must pass.

Gates:
  1. Regret_hybrid^structOOD < Regret_MB^structOOD
  2. Regret_hybrid^mechOOD < Regret_MB^mechOOD
  3. Coverage_90^structOOD >= 0.88 (stratified)
  4. Mechanism OOD harm FNR < 10%
  5. N_effective_intervention >= 300 (total across OOD splits)
  6. Breaks == 0 on structural OOD
  7. UCB_95(break_rate) < 5% on structural OOD
  8. Rescue recall > 0 (authority provides value)

Usage:
    python scripts/m4_promotion_gates.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
M4_DIR = REPO_ROOT / "experiments/daph_x/m4"


def main():
    # Load all results
    q_res_results = json.loads(open(M4_DIR / "q_res_m4_results.json").read())
    conformal = json.loads(open(M4_DIR / "conformal_calibration_m4.json").read())
    risk = json.loads(open(M4_DIR / "intervention_risk_m4.json").read())
    shadow = json.loads(open(M4_DIR / "shadow_authority_m4.json").read())

    gates = []

    # Gate 1: Regret_hybrid^structOOD < Regret_MB^structOOD
    struct = q_res_results.get("structural_ood", {})
    r_mb = struct.get("regret_mb", 999)
    r_hyb = struct.get("regret_hybrid", 999)
    pass1 = r_hyb < r_mb
    gates.append({
        "name": "regret_structOOD_hybrid_lt_MB",
        "description": "Regret_hybrid^structOOD < Regret_MB^structOOD",
        "value_left": r_hyb,
        "value_right": r_mb,
        "pass": pass1,
    })

    # Gate 2: Regret_hybrid^mechOOD < Regret_MB^mechOOD
    mech = q_res_results.get("mechanism_ood", {})
    r_mb_m = mech.get("regret_mb", 999)
    r_hyb_m = mech.get("regret_hybrid", 999)
    pass2 = r_hyb_m < r_mb_m
    gates.append({
        "name": "regret_mechOOD_hybrid_lt_MB",
        "description": "Regret_hybrid^mechOOD < Regret_MB^mechOOD",
        "value_left": r_hyb_m,
        "value_right": r_mb_m,
        "pass": pass2,
    })

    # Gate 3: Coverage_90^structOOD >= 0.88 (stratified)
    strat_results = conformal.get("results", {}).get("structural_ood_stratified", {})
    cov_90 = strat_results.get("coverage_0.90", {}).get("empirical_coverage", 0)
    pass3 = cov_90 >= 0.88
    gates.append({
        "name": "coverage_90_structOOD_stratified",
        "description": "Stratified conformal coverage_90^structOOD >= 0.88",
        "value_left": cov_90,
        "value_right": 0.88,
        "pass": pass3,
    })

    # Gate 4: Mechanism OOD harm FNR < 10%
    mech_risk = risk.get("results", {}).get("mechanism_ood", {})
    mech_fnr = mech_risk.get("harm_fnr", 999)
    pass4 = mech_fnr < 0.10
    gates.append({
        "name": "mechOOD_harm_fnr_lt_10pct",
        "description": "Mechanism OOD harm FNR < 10%",
        "value_left": mech_fnr,
        "value_right": 0.10,
        "pass": pass4,
    })

    # Gate 5: N_effective_intervention >= 300
    shadow_results = shadow.get("results", {})
    n_struct = shadow_results.get("structural_ood", {}).get("n_force", 0)
    n_mech = shadow_results.get("mechanism_ood", {}).get("n_force", 0)
    n_total = n_struct + n_mech
    pass5 = n_total >= 300
    gates.append({
        "name": "n_effective_intervention_gte_300",
        "description": "N_effective_intervention >= 300 (total OOD)",
        "value_left": n_total,
        "value_right": 300,
        "pass": pass5,
    })

    # Gate 6: Breaks == 0 on structural OOD
    breaks_struct = shadow_results.get("structural_ood", {}).get("breaks", 999)
    pass6 = breaks_struct == 0
    gates.append({
        "name": "breaks_structOOD_zero",
        "description": "Breaks == 0 on structural OOD",
        "value_left": breaks_struct,
        "value_right": 0,
        "pass": pass6,
    })

    # Gate 7: UCB_95(break_rate) < 5% on structural OOD
    bu95 = shadow_results.get("structural_ood", {}).get("break_rate_upper_95", 999)
    pass7 = bu95 is not None and bu95 < 0.05
    gates.append({
        "name": "ucb95_breakrate_structOOD_lt_5pct",
        "description": "UCB_95(break_rate) < 5% on structural OOD",
        "value_left": bu95 if bu95 is not None else "N/A",
        "value_right": 0.05,
        "pass": pass7,
    })

    # Gate 8: Rescue recall > 0 on both splits
    recall_struct = shadow_results.get("structural_ood", {}).get("rescue_recall", 0)
    recall_mech = shadow_results.get("mechanism_ood", {}).get("rescue_recall", 0)
    pass8 = recall_struct > 0 and recall_mech > 0
    gates.append({
        "name": "rescue_recall_positive",
        "description": "Rescue recall > 0 on both OOD splits",
        "value_left": f"struct={recall_struct:.4f}, mech={recall_mech:.4f}",
        "value_right": "both > 0",
        "pass": pass8,
    })

    # Print results
    print(f"{'='*70}")
    print(f"  M4R2 PROMOTION GATE CHECK")
    print(f"{'='*70}")
    print()
    all_pass = True
    for g in gates:
        status = "✓ PASS" if g["pass"] else "✗ FAIL"
        print(f"  {status}  {g['description']}")
        print(f"         {g['value_left']} vs {g['value_right']}")
        if not g["pass"]:
            all_pass = False
        print()

    print(f"{'='*70}")
    n_pass = sum(1 for g in gates if g["pass"])
    print(f"  Gates passed: {n_pass}/{len(gates)}")
    print(f"  Overall: {'ALL PASS' if all_pass else 'NOT QUALIFIED'}")
    print(f"{'='*70}")

    # Save
    output = {
        "gates": gates,
        "all_pass": all_pass,
        "n_pass": n_pass,
        "n_total": len(gates),
    }
    output_path = M4_DIR / "promotion_gates_m4.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")

    return all_pass


if __name__ == "__main__":
    main()
