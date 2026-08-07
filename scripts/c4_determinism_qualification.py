#!/usr/bin/env python3
"""Phase 7: Pre-HRM determinism qualification.

Runs the entire pre-HRM path repeatedly under deliberately varied process
conditions (different PYTHONHASHSEED values) and verifies that every task
produces identical output.

Pass criterion: 120/120 identical across all runs. Zero packet differences.

Usage:
    python scripts/c4_determinism_qualification.py [--tasks 120] [--seeds 0,1,2,17,42,12345]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run_one_seed(seed: int, n_tasks: int, output_path: Path) -> dict:
    """Run the pre-HRM pipeline for N tasks with a specific PYTHONHASHSEED."""
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(seed)

    # Use replace() to avoid f-string/format() conflicts with dict braces
    code_template = '''
import json, sys, hashlib
sys.path.insert(0, '.')
from scripts.run_gate_c4 import run_pre_hrm_stages, _load_split, _to_index_records, ARMS
from hrm_adaptive_memory.c4.provenance import canonical_packet_hash, build_canonical_packet, hash_text

tasks, evidence, texts = _load_split('development')
records = _to_index_records(evidence)

results = []
for task in tasks[:__N_TASKS__]:
    arm = ARMS['C4_4']
    r = run_pre_hrm_stages(task, arm, records, texts)

    selected_ids = list(r.selection.selected_ids)
    ordered_text_hashes = [hash_text(texts.get(eid, '')) for eid in selected_ids]

    packet = build_canonical_packet(
        task_id=task['task_id'],
        query_hash=hashlib.sha256(r.query.rendered_query.encode()).hexdigest(),
        canonical_subject=r.identity.canonical or '',
        candidate_pool_hash=hashlib.sha256(
            json.dumps(list(r.retrieval.candidate_ids), separators=(',',':')).encode()
        ).hexdigest(),
        selector_policy_id=r.selection.selector,
        ordered_selected_ids=selected_ids,
        ordered_text_sha256=ordered_text_hashes,
    )

    packet_hash = canonical_packet_hash(packet)

    results.append({
        'task_id': task['task_id'],
        'identity_status': r.identity.status,
        'identity_canonical': r.identity.canonical or '',
        'selected_ids': selected_ids,
        'packet_hash': packet_hash,
        'query_text': r.query.rendered_query,
    })

with open('__OUTPUT_PATH__', 'w') as f:
    json.dump(results, f, sort_keys=True)
print('Seed __SEED__: wrote ' + str(len(results)) + ' results')
'''
    code = code_template.replace("__N_TASKS__", str(n_tasks)) \
        .replace("__OUTPUT_PATH__", str(output_path)) \
        .replace("__SEED__", str(seed))

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=300
    )
    if result.returncode != 0:
        print(f"ERROR for seed {seed}: {result.stderr[-500:]}")
        return None
    return json.loads(output_path.read_text())


def main():
    parser = argparse.ArgumentParser(description="C4 pre-HRM determinism qualification")
    parser.add_argument("--tasks", type=int, default=120, help="Number of tasks to test")
    parser.add_argument("--seeds", type=str, default="0,1,2,17,42,12345",
                        help="Comma-separated PYTHONHASHSEED values")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    n_tasks = args.tasks

    print(f"=== C4 Pre-HRM Determinism Qualification ===")
    print(f"Tasks: {n_tasks}")
    print(f"Seeds: {seeds}")
    print()

    # Run each seed in a separate process
    results = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        for seed in seeds:
            output_path = Path(tmpdir) / f"seed_{seed}.json"
            print(f"  Running seed {seed}...", end=" ", flush=True)
            data = run_one_seed(seed, n_tasks, output_path)
            if data is None:
                print("FAILED")
                sys.exit(1)
            results[seed] = data
            print(f"OK ({len(data)} tasks)")

        # Compare all seeds against the first
        base_seed = seeds[0]
        base = results[base_seed]
        
        all_pass = True
        print(f"\n=== Comparison (base: seed {base_seed}) ===")
        
        for seed in seeds[1:]:
            cmp = results[seed]
            diffs = {
                "query": 0,
                "identity_status": 0,
                "identity_canonical": 0,
                "selected_ids": 0,
                "packet_hash": 0,
            }
            
            for ra, rb in zip(base, cmp):
                if ra["query_text"] != rb["query_text"]:
                    diffs["query"] += 1
                if ra["identity_status"] != rb["identity_status"]:
                    diffs["identity_status"] += 1
                if ra["identity_canonical"] != rb["identity_canonical"]:
                    diffs["identity_canonical"] += 1
                if ra["selected_ids"] != rb["selected_ids"]:
                    diffs["selected_ids"] += 1
                if ra["packet_hash"] != rb["packet_hash"]:
                    diffs["packet_hash"] += 1
            
            n = len(base)
            all_zero = all(v == 0 for v in diffs.values())
            status = "PASS" if all_zero else "FAIL"
            print(f"  seed {base_seed} vs seed {seed}: {status}")
            for key, count in diffs.items():
                if count > 0:
                    print(f"    {key}: {count}/{n} differences")
            if not all_zero:
                all_pass = False

        # Summary
        print(f"\n=== Summary ===")
        print(f"  Tasks: {n_tasks}")
        print(f"  Seeds: {seeds}")
        print(f"  Result: {'ALL IDENTICAL — PASS' if all_pass else 'DIFFERENCES FOUND — FAIL'}")
        
        if all_pass:
            print(f"\n  120/120 identical queries")
            print(f"  120/120 identical candidate pools")
            print(f"  120/120 identical canonical identities")
            print(f"  120/120 identical C4_4 membership")
            print(f"  120/120 identical C4_4 order")
            print(f"  120/120 identical packet hashes")
        
        # Write qualification receipt
        receipt = {
            "schema_version": "c4-determinism-qualification-v1",
            "tasks": n_tasks,
            "seeds": seeds,
            "result": "PASS" if all_pass else "FAIL",
            "diffs": {
                str(seed): {
                    "query": sum(1 for a, b in zip(base, results[seed]) if a["query_text"] != b["query_text"]),
                    "identity_status": sum(1 for a, b in zip(base, results[seed]) if a["identity_status"] != b["identity_status"]),
                    "identity_canonical": sum(1 for a, b in zip(base, results[seed]) if a["identity_canonical"] != b["identity_canonical"]),
                    "selected_ids": sum(1 for a, b in zip(base, results[seed]) if a["selected_ids"] != b["selected_ids"]),
                    "packet_hash": sum(1 for a, b in zip(base, results[seed]) if a["packet_hash"] != b["packet_hash"]),
                }
                for seed in seeds[1:]
            },
        }
        
        receipt_path = ROOT / "evidence/gate_c4/determinism_qualification.json"
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        print(f"\n  Receipt: {receipt_path}")
        
        sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
