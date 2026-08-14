#!/usr/bin/env python3
"""Corrected restart/recovery measurement for VERIFIED_MEMORY_CONSOLIDATION_V1.

The recovery sweep inside scripts/pressure_test_verified_memory.py was WRONG:
it reopened the SAME store after the ladder had already grown to 1M events,
so all three rows measured a 1M cold replay while being labelled 100k / 500k
/ 1M. Its numbers are not a scale sweep and must not be read as one.

This script measures it properly by truncating a copy of the canonical log to
each scale -- replaying a log prefix is exactly what a restart at that scale
would have done, since the log is append-only and prefix-stable (C12).

Three startup paths are timed separately, which is what the original request
actually asked for:

    cold replay              no snapshot present
    snapshot-validated       a snapshot that matches this log
    snapshot-invalidated     a snapshot describing a DIFFERENT log

If all three are equal, that is itself the finding: the snapshot is being
published and validated but never used to accelerate startup, so there is no
fast-start path and restart cost is unavoidably O(n).
"""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import hrm_adaptive_memory.evaluation  # noqa: E402,F401  (cycle-breaker)

from hrm_adaptive_memory.memory_write import ClaimStore  # noqa: E402


def timed_open(root: Path) -> tuple[float, ClaimStore]:
    gc.collect()
    t0 = time.perf_counter()
    s = ClaimStore(root, auto_snapshot=False)
    return time.perf_counter() - t0, s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-log", required=True, help="an existing claims_events.jsonl")
    ap.add_argument("--scales", type=int, nargs="*", default=[100_000, 500_000, 1_000_000])
    ap.add_argument("--out", default=str(ROOT / "evidence/memory_write/restart_recovery_v1.json"))
    args = ap.parse_args()

    src = Path(args.source_log)
    workdir = Path(args.out).parent / "_recovery_tmp"
    workdir.mkdir(parents=True, exist_ok=True)

    print("=== restart / recovery (corrected) ===")
    print(f"  source log: {src}  ({src.stat().st_size/1e6:.0f} MB)\n")
    print(f"  {'scale':>9}{'cold_s':>9}{'snap_valid_s':>14}{'snap_invalid_s':>16}"
          f"{'active':>10}{'hash_match':>12}")

    rows = []
    for scale in args.scales:
        d = workdir / f"s{scale}"
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        # truncate the canonical log to the first `scale` events
        with src.open() as fin, (d / "claims_events.jsonl").open("w") as fout:
            for i, line in enumerate(fin):
                if i >= scale:
                    break
                fout.write(line)

        # 1) cold: no snapshot on disk
        t_cold, s = timed_open(d)
        h_cold = s.consolidated_state().state_hash()
        n_active = len(s.consolidated_state().active_record_ids)
        s.publish_snapshot()
        del s; gc.collect()

        # 2) snapshot present AND valid for this log
        t_valid, s = timed_open(d)
        assert s.snapshot_is_valid(), "snapshot should validate against its own log"
        h_valid = s.consolidated_state().state_hash()
        del s; gc.collect()

        # 3) snapshot present but describing a DIFFERENT log -> must be discarded
        snap = json.loads((d / "consolidated_snapshot.json").read_text())
        snap["derived_from_log_sha256"] = "0" * 64
        (d / "consolidated_snapshot.json").write_text(json.dumps(snap))
        t_invalid, s = timed_open(d)
        assert not s.snapshot_is_valid(), "mismatched snapshot must be rejected"
        h_invalid = s.consolidated_state().state_hash()
        del s; gc.collect()

        match = h_cold == h_valid == h_invalid
        rows.append({
            "scale": scale, "t_cold_replay_s": t_cold,
            "t_snapshot_validated_startup_s": t_valid,
            "t_snapshot_invalidated_rebuild_s": t_invalid,
            "active_records": n_active,
            "all_three_paths_agree_on_state_hash": match,
        })
        print(f"  {scale:>9}{t_cold:>9.2f}{t_valid:>14.2f}{t_invalid:>16.2f}"
              f"{n_active:>10}{str(match):>12}")
        shutil.rmtree(d)

    spread = max(abs(r["t_snapshot_validated_startup_s"] - r["t_cold_replay_s"])
                 / max(r["t_cold_replay_s"], 1e-9) for r in rows)
    no_fast_path = spread < 0.25
    print(f"\n  max relative difference between cold and snapshot-validated startup: {spread:.1%}")
    if no_fast_path:
        print("  FINDING: the snapshot does NOT accelerate startup. It is published and")
        print("  validated, but ClaimStore always replays the full log, so restart cost")
        print("  is O(n) regardless. This is the motivation for checkpointed replay.")

    out = {
        "measurement": "restart/recovery, corrected",
        "supersedes": ("the 'recovery' block in evidence/memory_write/pressure_test_v1.json, "
                       "which reopened the same 1M store three times and mislabelled the rows "
                       "by scale"),
        "method": "truncate a copy of the canonical log to each scale; replaying a log prefix "
                  "is what a restart at that scale would have done (C12 prefix stability)",
        "rows": rows,
        "snapshot_does_not_accelerate_startup": bool(no_fast_path),
        "max_relative_gap_cold_vs_snapshot_validated": spread,
    }
    p = Path(args.out)
    p.write_text(json.dumps(out, indent=2, sort_keys=True, default=str) + "\n")
    print(f"\n  written: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
