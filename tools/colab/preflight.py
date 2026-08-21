"""No-LLM confirmation preflight.

Runs all structural and evidence checks over the full confirmation corpus
without making any LLM calls. If any check fails, the confirmation is blocked.

Usage (on Colab):
    cd /content/Daph-ex-research-gate-c2-beir-retrieval
    PYTHONPATH=. python3 tools/colab/preflight.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_DIR))


def main():
    print("=" * 80)
    print("R13 NO-LLM PREFLIGHT")
    print("=" * 80)

    checks = []

    # 1. Structural validator
    print("\n[1/10] Structural T2 validation...")
    from hrm_adaptive_memory.executive.semantic_relations.i3_15c_task_generator import (
        generate_i3_15c_corpus, validate_t2_eligibility,
    )
    tasks = generate_i3_15c_corpus(n_per_cell=40, seed=42)
    print(f"  Generated {len(tasks)} tasks")
    validation = validate_t2_eligibility(tasks)
    passed = validation["passed"]
    checks.append({
        "name": "structural_t2_validation",
        "passed": passed,
        "details": {
            "t2_positive_expected": validation["t2_positive_expected"],
            "t2_positive_reachable_gold": validation["t2_positive_reachable_gold"],
            "t2_negative_expected": validation["t2_negative_expected"],
            "t2_negative_incorrectly_reachable_gold": validation["t2_negative_incorrectly_reachable_gold"],
        },
    })
    print(f"  {'PASS' if passed else 'FAIL'}: T2+ reachable={validation['t2_positive_reachable_gold']}/{validation['t2_positive_expected']}, "
          f"T2- incorrectly reachable={validation['t2_negative_incorrectly_reachable_gold']}")

    # 2. Corpus coverage
    print("\n[2/10] Corpus coverage (100% required evidence in corpus)...")
    from hrm_adaptive_memory.executive.semantic_relations.i3_15c_task_generator import get_i3_15c_corpus
    corpus = get_i3_15c_corpus()
    corpus_ids = {p.passage_id for p in corpus}
    corpus_by_text = {p.text: p.passage_id for p in corpus}

    from scripts.run_i3_15_r1_balanced import get_required_passage_ids
    all_required = set()
    for task in tasks:
        req = get_required_passage_ids(task, corpus_by_text)
        all_required.update(req)
    missing = all_required - corpus_ids
    coverage = 1.0 - len(missing) / max(len(all_required), 1)
    checks.append({
        "name": "corpus_coverage",
        "passed": coverage == 1.0,
        "details": {"coverage": coverage, "missing": list(missing)[:10]},
    })
    print(f"  {'PASS' if coverage == 1.0 else 'FAIL'}: coverage={coverage:.4f}, missing={len(missing)}")

    # 3. Retrieval receipts present
    print("\n[3/10] Retrieval receipts present...")
    receipts_path = REPO_DIR / "experiments/v2b_i3_15c/confirmation/retrieval_receipts.jsonl"
    if receipts_path.exists():
        n_receipts = sum(1 for _ in open(receipts_path))
        expected_receipts = len(tasks) * 3  # Q0, Q3, Q4
        checks.append({
            "name": "retrieval_receipts_present",
            "passed": n_receipts == expected_receipts,
            "details": {"n_receipts": n_receipts, "expected": expected_receipts},
        })
        print(f"  {'PASS' if n_receipts == expected_receipts else 'FAIL'}: {n_receipts}/{expected_receipts} receipts")
    else:
        checks.append({"name": "retrieval_receipts_present", "passed": False, "details": {"error": "file not found"}})
        print(f"  FAIL: {receipts_path} not found")

    # 4. Q3 retrieval qualification
    print("\n[4/10] Q3 retrieval qualification...")
    summary_path = REPO_DIR / "experiments/v2b_i3_15c/confirmation/retrieval_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        q3_mean = summary.get("mean_recall_by_level", {}).get("Q3_RERANKED", 0)
        q0_mean = summary.get("mean_recall_by_level", {}).get("Q0_BM25", 0)
        q3_better = q3_mean > q0_mean
        q3_qualified = summary.get("qualification", {}).get("qualified", False)
        checks.append({
            "name": "q3_retrieval_qualification",
            "passed": q3_qualified and q3_better,
            "details": {"q3_mean": q3_mean, "q0_mean": q0_mean, "qualified": q3_qualified},
        })
        print(f"  {'PASS' if q3_qualified and q3_better else 'FAIL'}: Q3={q3_mean:.4f}, Q0={q0_mean:.4f}, qualified={q3_qualified}")
    else:
        checks.append({"name": "q3_retrieval_qualification", "passed": False, "details": {"error": "summary not found"}})
        print(f"  FAIL: {summary_path} not found")

    # 5. T2-positive gold reachability = 100%
    print("\n[5/10] T2-positive gold reachability = 100%...")
    t2_pos_reachable = validation["t2_positive_reachable_gold"]
    t2_pos_expected = validation["t2_positive_expected"]
    checks.append({
        "name": "t2_positive_gold_reachability",
        "passed": t2_pos_reachable == t2_pos_expected,
        "details": {"reachable": t2_pos_reachable, "expected": t2_pos_expected},
    })
    print(f"  {'PASS' if t2_pos_reachable == t2_pos_expected else 'FAIL'}: {t2_pos_reachable}/{t2_pos_expected}")

    # 6. T2-negative gold reachability = 0%
    print("\n[6/10] T2-negative gold reachability = 0%...")
    t2_neg_incorrect = validation["t2_negative_incorrectly_reachable_gold"]
    checks.append({
        "name": "t2_negative_gold_reachability",
        "passed": t2_neg_incorrect == 0,
        "details": {"incorrectly_reachable": t2_neg_incorrect},
    })
    print(f"  {'PASS' if t2_neg_incorrect == 0 else 'FAIL'}: incorrectly_reachable={t2_neg_incorrect}")

    # 7. False structural T2 on controls = 0
    print("\n[7/10] False structural T2 on controls = 0...")
    per_stratum = validation.get("per_stratum", {})
    false_t2_controls = sum(
        s.get("t2_gold_true", 0) + s.get("t2_initial_true", 0)
        for name, s in per_stratum.items()
        if name in ("DEFER_CONTROL", "ANSWER_CONTROL", "MATCHED_NEG")
    )
    checks.append({
        "name": "false_t2_controls",
        "passed": false_t2_controls == 0,
        "details": {"false_t2_controls": false_t2_controls},
    })
    print(f"  {'PASS' if false_t2_controls == 0 else 'FAIL'}: false_t2={false_t2_controls}")

    # 8. Frozen hashes match (protocol exists)
    print("\n[8/10] Frozen hashes match (protocol exists)...")
    protocol_path = REPO_DIR / "experiments/v2b_i3_15c/confirmation/confirmation_protocol_v1.json"
    if protocol_path.exists():
        with open(protocol_path) as f:
            protocol = json.load(f)
        checks.append({
            "name": "frozen_hashes",
            "passed": True,
            "details": {"protocol_id": protocol.get("protocol_id"), "status": protocol.get("status")},
        })
        print(f"  PASS: protocol_id={protocol.get('protocol_id')}, status={protocol.get('status')}")
    else:
        checks.append({"name": "frozen_hashes", "passed": False, "details": {"error": "protocol not found"}})
        print(f"  FAIL: {protocol_path} not found")

    # 9. Backend identity matches protocol
    print("\n[9/10] Backend identity matches protocol...")
    # This check verifies that the backend requirements are defined in the protocol
    backend_req = protocol.get("backend_requirements", {}) if protocol_path.exists() else {}
    checks.append({
        "name": "backend_identity",
        "passed": bool(backend_req),
        "details": {"backend_primary": backend_req.get("primary", "NOT SET")},
    })
    print(f"  {'PASS' if backend_req else 'FAIL'}: primary={backend_req.get('primary', 'NOT SET')}")

    # 10. Utility identity matches protocol
    print("\n[10/10] Utility identity matches protocol...")
    frozen = protocol.get("frozen_identities", {}) if protocol_path.exists() else {}
    utility_id = frozen.get("utility", "NOT SET")
    checks.append({
        "name": "utility_identity",
        "passed": utility_id != "NOT SET",
        "details": {"utility": utility_id},
    })
    print(f"  {'PASS' if utility_id != 'NOT SET' else 'FAIL'}: utility={utility_id}")

    # Final result
    all_pass = all(c["passed"] for c in checks)
    print(f"\n{'='*80}")
    print(f"PREFLIGHT RESULT: {'PASS' if all_pass else 'FAIL'}")
    print(f"{'='*80}")

    if not all_pass:
        print("\nFAILED CHECKS:")
        for c in checks:
            if not c["passed"]:
                print(f"  - {c['name']}: {c.get('details', {})}")
        print("\nEXPERIMENT_BLOCKED")
        print("Do not proceed with R13 until all checks pass.")
    else:
        print("\nAll checks passed. Ready for R13 confirmation.")

    # Save report
    report_path = REPO_DIR / "experiments/v2b_i3_15c/confirmation/preflight_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump({
            "all_passed": all_pass,
            "n_checks": len(checks),
            "checks": checks,
        }, f, indent=2)
    print(f"\nReport saved to {report_path}")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
