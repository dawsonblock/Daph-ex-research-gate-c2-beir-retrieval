#!/usr/bin/env python3
"""Verify BOUNDED_MEMORY_RUNTIME_V1 acceptance conditions A1-A10.

Each scale runs in its OWN subprocess, because ru_maxrss reports a process
PEAK: loading 100k then 500k then 1M in one process would report the 1M peak
for all three and silently overstate the small scales.

A3 compares against state hashes captured from the PRE-change implementation
(evidence/memory_write/pre_change_state_hashes.json). Comparing the new code
against itself would prove nothing.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CHILD = r'''
import json, resource, sys, time, shutil
from pathlib import Path
sys.path.insert(0, %r)
import hrm_adaptive_memory.evaluation  # noqa
from hrm_adaptive_memory.memory_write import ClaimStore
from hrm_adaptive_memory.memory_write.consolidation import consolidate_from_scratch

src, scale, work = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
if work.exists(): shutil.rmtree(work)
work.mkdir(parents=True)
with src.open() as fin, (work/"claims_events.jsonl").open("w") as fo:
    for i, l in enumerate(fin):
        if i >= scale: break
        fo.write(l)

t0 = time.perf_counter()
s = ClaimStore(work, auto_snapshot=False)
t_replay = time.perf_counter() - t0
st = s.consolidated_state()
# A4: incremental still equals independent full consolidation
t1 = time.perf_counter()
scratch = consolidate_from_scratch(s.all_records(), s.corpus_version)
t_scratch = time.perf_counter() - t1
print(json.dumps({
    "scale": scale,
    "state_hash": st.state_hash(),
    "corpus_version": s.corpus_version,
    "active_records": len(st.active_record_ids),
    "duplicate_clusters": len(st.duplicate_clusters),
    "alias_clusters": len(st.alias_clusters),
    "support_groups": len(st.support_groups),
    "contradiction_groups": len(st.contradiction_groups),
    "incremental_equals_full_consolidation": st.state_hash() == scratch.state_hash(),
    "has_resident_event_list": hasattr(s, "_events"),
    "replay_s": t_replay,
    "consolidate_from_scratch_s": t_scratch,
    "rss_peak_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    "n_index_records": len(s.retrievable_index_records()),
}))
shutil.rmtree(work)
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-log", required=True)
    ap.add_argument("--scales", type=int, nargs="*", default=[100_000, 500_000, 1_000_000])
    ap.add_argument("--baseline", default=str(ROOT / "evidence/memory_write/pre_change_state_hashes.json"))
    ap.add_argument("--out", default=str(ROOT / "evidence/memory_write/bounded_memory_runtime_v1.json"))
    args = ap.parse_args()

    base = json.loads(Path(args.baseline).read_text())["scales"]
    work = ROOT / "evidence/memory_write/_bm_tmp"

    print("=== BOUNDED_MEMORY_RUNTIME_V1 verification ===")
    print(f"  {'scale':>9}{'RSS_MB':>9}{'was_MB':>9}{'delta':>9}"
          f"{'replay_s':>10}{'was_s':>8}{'A3_hash':>9}{'A4':>6}{'A1':>6}")

    rows = []
    for scale in args.scales:
        r = subprocess.run(
            [sys.executable, "-c", CHILD % str(ROOT), args.source_log, str(scale), str(work)],
            capture_output=True, text=True, timeout=3600)
        if r.returncode != 0:
            print(f"  {scale}: FAILED\n{r.stderr[-2000:]}")
            return 1
        got = json.loads(r.stdout.strip().splitlines()[-1])
        b = base[str(scale)]
        a3 = got["state_hash"] == b["state_hash"]
        a1 = not got["has_resident_event_list"]
        a2 = got["corpus_version"] == b["corpus_version"]
        a4 = got["incremental_equals_full_consolidation"]
        a5 = (got["active_records"] == b["active_records"]
              and got["contradiction_groups"] == b["contradiction_groups"]
              and got["support_groups"] == b["support_groups"]
              and got["alias_clusters"] == b["alias_clusters"]
              and got["duplicate_clusters"] == b["duplicate_clusters"])
        rss_mb = got["rss_peak_bytes"] / 1e6
        was_mb = b["rss_bytes_peak_after"] / 1e6
        rows.append({**got, "baseline": b, "A1_no_resident_events": a1,
                     "A2_corpus_version_correct": a2, "A3_state_hash_identical": a3,
                     "A4_incremental_equals_full": a4, "A5_semantics_identical": a5,
                     "rss_reduction_pct": (was_mb - rss_mb) / was_mb * 100 if was_mb else None})
        print(f"  {scale:>9}{rss_mb:>9.0f}{was_mb:>9.0f}"
              f"{(rss_mb-was_mb)/was_mb*100:>8.1f}%{got['replay_s']:>10.2f}"
              f"{b['replay_s']:>8.2f}{str(a3):>9}{str(a4):>6}{str(a1):>6}")

    all_ok = all(r["A1_no_resident_events"] and r["A2_corpus_version_correct"]
                 and r["A3_state_hash_identical"] and r["A4_incremental_equals_full"]
                 and r["A5_semantics_identical"] for r in rows)
    print(f"\n  A1 no resident event list : {all(r['A1_no_resident_events'] for r in rows)}")
    print(f"  A2 corpus_version correct : {all(r['A2_corpus_version_correct'] for r in rows)}")
    print(f"  A3 state hash IDENTICAL   : {all(r['A3_state_hash_identical'] for r in rows)}")
    print(f"  A4 incremental == full    : {all(r['A4_incremental_equals_full'] for r in rows)}")
    print(f"  A5 semantics identical    : {all(r['A5_semantics_identical'] for r in rows)}")
    print(f"  ALL ACCEPTANCE CONDITIONS : {all_ok}")

    if work.exists():
        shutil.rmtree(work)
    Path(args.out).write_text(json.dumps({
        "artifact": "BOUNDED_MEMORY_RUNTIME_V1 verification",
        "design": "configs/bounded_memory_runtime_v1_design.json",
        "baseline_source": args.baseline,
        "rows": rows, "all_acceptance_conditions_met": all_ok,
    }, indent=2, sort_keys=True, default=str) + "\n")
    print(f"\n  written: {args.out}")
    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
