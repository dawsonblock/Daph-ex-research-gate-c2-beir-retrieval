#!/usr/bin/env python3
"""Phase 7: Pre-HRM determinism qualification.

Runs the entire pre-HRM path repeatedly under deliberately varied process
conditions (different PYTHONHASHSEED values) and verifies that every task
produces identical output.

Pass criterion: N/N identical on EVERY compared field. Zero differences.

Every field named in the summary is a field this script actually compared. The
previous version printed "120/120 identical candidate pools" and "identical
C4_4 order" while its diff object compared neither, so the summary asserted
more than the code measured. The comparison set is now derived from one list,
:data:`COMPARED_FIELDS`, which drives the replay payload, the diff, and the
printed summary alike.

Usage:
    python scripts/c4_determinism_qualification.py [--tasks 120] \
        [--seeds 0,1,2,17,42,12345] [--arms C4_4]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Fields compared across seeds. Order matters only for display.
# Each entry: (receipt key, human label used in the summary)
COMPARED_FIELDS: list[tuple[str, str]] = [
    ("query_text", "identical queries"),
    ("candidate_ids", "identical candidate pool IDs"),
    ("candidate_pool_hash", "identical candidate pool hashes"),
    ("identity_status", "identical identity statuses"),
    ("identity_canonical", "identical canonical identities"),
    ("selected_ids", "identical selected IDs"),
    ("membership_hash", "identical membership hashes"),
    ("order_hash", "identical packet order hashes"),
    ("ordered_selected_ids", "identical packet orders"),
    ("packet_hash", "identical packet hashes"),
    ("prompt_hash", "identical prompt hashes"),
]

# The replay body. Every value it records is one of COMPARED_FIELDS, so the
# payload and the comparison cannot drift apart.
_REPLAY_TEMPLATE = '''
import json, sys
sys.path.insert(0, '.')
from scripts.run_gate_c4 import run_pre_hrm_stages, _load_split, _to_index_records, ARMS

tasks, evidence, texts = _load_split('__SPLIT__')
records = _to_index_records(evidence)

arm = ARMS['__ARM__']
results = []
for task in tasks[:__N_TASKS__]:
    r = run_pre_hrm_stages(task, arm, records, texts)
    p = r.packet
    results.append({
        'task_id': task['task_id'],
        'query_text': r.query.rendered_query,
        'candidate_ids': list(r.retrieval.candidate_ids),
        'candidate_pool_hash': p.candidate_pool_hash,
        'identity_status': r.identity.status,
        'identity_canonical': r.identity.canonical or '',
        'selected_ids': list(r.selection.selected_ids),
        'membership_hash': p.membership_hash,
        'order_hash': p.order_hash,
        'ordered_selected_ids': list(p.packet_ids),
        'packet_hash': p.packet_hash,
        'prompt_hash': p.prompt_hash,
    })

with open('__OUTPUT_PATH__', 'w') as f:
    json.dump(results, f, sort_keys=True)
print('Seed __SEED__: wrote ' + str(len(results)) + ' results')
'''


def run_one_seed(seed: int, n_tasks: int, arm: str, split: str,
                 output_path: Path) -> list[dict] | None:
    """Run the pre-HRM pipeline for N tasks with a specific PYTHONHASHSEED."""
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(seed)

    code = (_REPLAY_TEMPLATE
            .replace("__N_TASKS__", str(n_tasks))
            .replace("__OUTPUT_PATH__", str(output_path))
            .replace("__SEED__", str(seed))
            .replace("__SPLIT__", split)
            .replace("__ARM__", arm))

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=900
    )
    if result.returncode != 0:
        print(f"ERROR for seed {seed}: {result.stderr[-800:]}")
        return None
    return json.loads(output_path.read_text())


def _assert_payload_complete(records: list[dict]) -> None:
    """Fail closed if the replay did not record a field we claim to compare.

    An empty-string hash counts as missing: it would compare equal across
    seeds and manufacture a pass.
    """
    if not records:
        raise AssertionError("Replay produced no records")
    missing = [key for key, _ in COMPARED_FIELDS if key not in records[0]]
    if missing:
        raise AssertionError(
            f"Replay payload is missing compared fields {missing}. The summary "
            f"would claim invariance that was never measured.")
    for key, _ in COMPARED_FIELDS:
        blank = [r["task_id"] for r in records if r[key] in ("", [], None)]
        if blank:
            raise AssertionError(
                f"Field {key!r} is empty for {len(blank)} task(s) "
                f"(e.g. {blank[:3]}); it cannot be used as a determinism "
                f"witness. Fix the pipeline before qualifying.")


def _diff(base: list[dict], other: list[dict]) -> dict[str, int]:
    """Per-field difference counts between two seed replays."""
    diffs = {key: 0 for key, _ in COMPARED_FIELDS}
    for ra, rb in zip(base, other):
        for key, _ in COMPARED_FIELDS:
            if ra[key] != rb[key]:
                diffs[key] += 1
    return diffs


def main() -> int:
    parser = argparse.ArgumentParser(description="C4 pre-HRM determinism qualification")
    parser.add_argument("--tasks", type=int, default=120, help="Number of tasks to test")
    parser.add_argument("--seeds", type=str, default="0,1,2,17,42,12345",
                        help="Comma-separated PYTHONHASHSEED values")
    parser.add_argument("--arms", type=str, default="C4_4",
                        help="Comma-separated arms to qualify")
    parser.add_argument("--split", default="development")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    arm_ids = [a.strip() for a in args.arms.split(",") if a.strip()]
    n_tasks = args.tasks

    if len(seeds) < 2:
        print("ERROR: at least two seeds are required to compare anything.")
        return 1

    print("=== C4 Pre-HRM Determinism Qualification ===")
    print(f"Tasks:  {n_tasks}")
    print(f"Seeds:  {seeds}")
    print(f"Arms:   {arm_ids}")
    print(f"Fields: {[k for k, _ in COMPARED_FIELDS]}")
    print()

    overall_pass = True
    receipt_arms: dict[str, dict] = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        for arm in arm_ids:
            print(f"--- Arm {arm} ---")
            results: dict[int, list[dict]] = {}
            for seed in seeds:
                output_path = Path(tmpdir) / f"{arm}_seed_{seed}.json"
                print(f"  Running seed {seed}...", end=" ", flush=True)
                data = run_one_seed(seed, n_tasks, arm, args.split, output_path)
                if data is None:
                    print("FAILED")
                    return 1
                try:
                    _assert_payload_complete(data)
                except AssertionError as exc:
                    print(f"FAILED\n  {exc}")
                    return 1
                results[seed] = data
                print(f"OK ({len(data)} tasks)")

            base_seed = seeds[0]
            base = results[base_seed]
            n = len(base)
            arm_pass = True
            arm_diffs: dict[str, dict[str, int]] = {}

            print(f"\n  Comparison (base: seed {base_seed})")
            for seed in seeds[1:]:
                if len(results[seed]) != n:
                    print(f"    seed {seed}: FAIL — record count "
                          f"{len(results[seed])} != {n}")
                    arm_pass = False
                    arm_diffs[str(seed)] = {"record_count_mismatch": 1}
                    continue
                diffs = _diff(base, results[seed])
                arm_diffs[str(seed)] = diffs
                all_zero = all(v == 0 for v in diffs.values())
                print(f"    seed {base_seed} vs seed {seed}: "
                      f"{'PASS' if all_zero else 'FAIL'}")
                for key, count in diffs.items():
                    if count > 0:
                        print(f"      {key}: {count}/{n} differences")
                if not all_zero:
                    arm_pass = False

            print(f"\n  Arm {arm}: {'PASS' if arm_pass else 'FAIL'}")
            if arm_pass:
                for key, label in COMPARED_FIELDS:
                    print(f"    {n}/{n} {label}")
            receipt_arms[arm] = {
                "result": "PASS" if arm_pass else "FAIL",
                "tasks": n,
                "diffs": arm_diffs,
            }
            overall_pass = overall_pass and arm_pass
            print()

    print("=== Summary ===")
    print(f"  Tasks:  {n_tasks}")
    print(f"  Seeds:  {seeds}")
    print(f"  Arms:   {arm_ids}")
    print(f"  Result: {'ALL IDENTICAL — PASS' if overall_pass else 'DIFFERENCES FOUND — FAIL'}")

    receipt = {
        "schema_version": "c4-determinism-qualification-v2",
        "tasks": n_tasks,
        "seeds": seeds,
        "arms": arm_ids,
        "split": args.split,
        "compared_fields": [key for key, _ in COMPARED_FIELDS],
        "result": "PASS" if overall_pass else "FAIL",
        "per_arm": receipt_arms,
    }

    receipt_path = ROOT / "evidence/gate_c4/determinism_qualification.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"  Receipt: {receipt_path}")

    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
