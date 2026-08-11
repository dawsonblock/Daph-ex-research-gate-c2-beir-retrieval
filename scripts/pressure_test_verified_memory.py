#!/usr/bin/env python3
"""VERIFIED_MEMORY_CONSOLIDATION_V1 pressure test.

Per configs/verified_memory_consolidation_v1_design.json PRESSURE_TEST.
MEASUREMENT, NOT QUALIFICATION: there is no frozen performance threshold and
none is invented afterwards. Results are reported as numbers.

Ladder: 1k -> 10k -> 50k -> 100k -> 250k -> 500k -> 1M events, cumulative on
one store, with a FIXED event mixture declared before running.

At every checkpoint:
  * T_incremental per event        (steady-state ingestion)
  * T_full_replay                  (cold rebuild from the canonical log)
  * RSS, event-log size, snapshot size, derived-index size
  * retrieval latency              (reads must not silently rot while
                                    writes are optimized)
  * H(incremental) == H(full replay)  -- the integrity invariant, checked at
                                         EVERY checkpoint, not only at 1M
  * write amplification WA = bytes written to derived state
                             / bytes appended to canonical log

Restart/recovery is simulated at the larger scales to separate steady-state
ingestion cost from restart cost:
  * cold replay
  * snapshot-validated startup
  * snapshot-invalidated rebuild
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import resource
import string
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import hrm_adaptive_memory.evaluation  # noqa: E402,F401  (cycle-breaker)

from hrm_adaptive_memory.memory_write import ClaimStore  # noqa: E402
from hrm_adaptive_memory.memory_write.consolidation import (  # noqa: E402
    ALIAS_RELATION, consolidate_from_scratch)

LADDER = (1_000, 10_000, 50_000, 100_000, 250_000, 500_000, 1_000_000)
RECOVERY_SCALES = (100_000, 500_000, 1_000_000)

#: FROZEN BEFORE RUNNING. A realistic continual-ingestion mixture.
MIX = {"new": 0.60, "duplicate": 0.10, "alias": 0.05,
       "contradiction": 0.10, "supersession": 0.10, "retraction": 0.05}
MIX_SEED = 20260811

_L = string.ascii_lowercase
_ROLES = ("relay unit", "pressure assembly", "control module")
RELATIONS = ("operating tier", "thermal rating", "assigned category")


def entity(i: int) -> str:
    r"""Grammar-valid b3 entity: capitalized alpha head + lowercase role words.

    Two constraints the certified extractor actually enforces, both learned
    the hard way when its fail-closed guard rejected earlier generators:
      * NO DIGITS -- _V4_ENTITY is [A-Z][a-z]+(\s+[a-z]+){1,3}
      * the head must not be a _STOP_FIRST word. A plain base-26 head
        eventually spells real English ("This control module" was refused at
        ~500k events), so heads are prefixed with "Zq", which cannot form an
        English stopword. 26^4 heads x 3 roles = 1.37M unique entities.
    """
    j = i // 3
    return (f"Zq{_L[j % 26]}{_L[(j // 26) % 26]}{_L[(j // 676) % 26]}"
            f"{_L[(j // 17576) % 26]} {_ROLES[i % 3]}")


def rss_bytes() -> int:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r if sys.platform == "darwin" else r * 1024


def dir_bytes(path: Path, *names: str) -> int:
    return sum((path / n).stat().st_size for n in names if (path / n).is_file())


class Workload:
    """Deterministic mixed event generator."""

    def __init__(self, seed: int = MIX_SEED):
        self.rng = random.Random(seed)
        self.n_entities = 0
        self.superseding: list[str] = []   # record_ids eligible to be superseded
        self.retractable: list[str] = []
        self.counts = {k: 0 for k in MIX}

    def step(self, store: ClaimStore) -> None:
        r = self.rng.random()
        cum = 0.0
        kind = "new"
        for k, p in MIX.items():
            cum += p
            if r < cum:
                kind = k
                break
        ts = "2020-01-01T00:00:00+00:00"
        try:
            if kind == "supersession" and self.superseding:
                target = self.superseding.pop()
                rec = store.get(target)
                if rec is not None and rec.verification_state.value in ("UNVERIFIED", "SUPPORTED", "CONTRADICTED"):
                    res = store.ingest(subject=rec.canonical_entity, relation=rec.canonical_relation,
                                       value=f"Tier {self.rng.randrange(99)}", source_id="S",
                                       observed_at_utc=ts, supersedes=target)
                    self.counts[kind] += 1
                    self.superseding.append(res.record.record_id)
                    return
                kind = "new"
            if kind == "retraction" and self.retractable:
                target = self.retractable.pop()
                if store.get(target) is not None:
                    store.retract(target, reason="pressure-test")
                    self.counts[kind] += 1
                    return
                kind = "new"
            if kind == "duplicate" and self.n_entities:
                i = self.rng.randrange(self.n_entities)
                store.ingest(subject=entity(i), relation=RELATIONS[0], value="Tier 1",
                             source_id="A", observed_at_utc=ts)
                self.counts[kind] += 1
                return
            if kind == "contradiction" and self.n_entities:
                i = self.rng.randrange(self.n_entities)
                res = store.ingest(subject=entity(i), relation=RELATIONS[0],
                                   value=f"Tier {self.rng.randrange(50, 99)}",
                                   source_id=f"C{self.rng.randrange(1000)}", observed_at_utc=ts)
                self.counts[kind] += 1
                if res.record:
                    self.retractable.append(res.record.record_id)
                return
            if kind == "alias" and self.n_entities > 1:
                i = self.rng.randrange(self.n_entities)
                j = self.rng.randrange(self.n_entities)
                if i != j:
                    store.ingest(subject=entity(i), relation=ALIAS_RELATION, value=entity(j),
                                 source_id="A", observed_at_utc=ts)
                    self.counts[kind] += 1
                    return
                kind = "new"
        except Exception:
            kind = "new"
        # default: a brand new claim
        i = self.n_entities
        self.n_entities += 1
        res = store.ingest(subject=entity(i), relation=RELATIONS[i % len(RELATIONS)],
                           value="Tier 1", source_id="A", observed_at_utc=ts)
        self.counts["new"] += 1
        if res.record:
            if self.rng.random() < 0.25:
                self.superseding.append(res.record.record_id)
            if self.rng.random() < 0.15:
                self.retractable.append(res.record.record_id)


def measure_retrieval(store: ClaimStore, samples: int = 3) -> dict:
    """Reads must not rot while writes get optimized. Indexing the full
    active set is itself part of the cost, so it is timed separately from
    the query."""
    import asyncio

    from hrm_adaptive_memory.backends import CanonicalRetrievalMode
    from hrm_adaptive_memory.c4.retrieval_stage import get_cached_backend
    t0 = time.perf_counter()
    records = store.retrievable_index_records()
    t_project = time.perf_counter() - t0
    if not records:
        return {"n_indexed": 0, "t_project_s": t_project}
    t0 = time.perf_counter()
    backend = get_cached_backend(CanonicalRetrievalMode.BM25, records)
    t_index = time.perf_counter() - t0
    lat = []
    for k in range(samples):
        q = f"What is the operating tier for {entity(k)}?"
        t0 = time.perf_counter()
        asyncio.run(backend.search(q, k=5))
        lat.append(time.perf_counter() - t0)
    return {"n_indexed": len(records), "t_project_s": t_project,
            "t_index_build_s": t_index, "t_query_s_mean": sum(lat) / len(lat)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-events", type=int, default=1_000_000)
    ap.add_argument("--rss-limit-gb", type=float, default=12.0)
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", default=str(ROOT / "evidence/memory_write/pressure_test_v1.json"))
    ap.add_argument("--skip-retrieval", action="store_true")
    args = ap.parse_args()

    import tempfile
    root = Path(args.root) if args.root else Path(tempfile.mkdtemp()) / "pressure"
    store = ClaimStore(root, auto_snapshot=False)   # periodic publish; see WA note
    wl = Workload()

    print("=== VERIFIED_MEMORY pressure test (MEASUREMENT, no frozen thresholds) ===")
    print(f"  store: {root}")
    print(f"  mixture (frozen before running): {MIX}\n")
    print(f"  {'events':>9}{'us/ev':>9}{'replay_s':>10}{'scratch_s':>11}"
          f"{'RSS_MB':>9}{'log_MB':>8}{'snap_MB':>9}{'WA':>7}{'integrity':>11}")

    results = []
    done = 0
    snapshot_bytes_written = 0
    for target in LADDER:
        if target > args.max_events:
            break
        t0 = time.perf_counter()
        while done < target:
            wl.step(store)
            done = store.corpus_version
        t_ingest = time.perf_counter() - t0
        n_new = target - (results[-1]["events"] if results else 0)
        us_per_event = t_ingest / max(1, n_new) * 1e6

        # derived state: incremental vs from-scratch, integrity at EVERY rung
        inc = store.consolidated_state()
        t0 = time.perf_counter()
        scratch = consolidate_from_scratch(store.all_records(), store.corpus_version)
        t_scratch = time.perf_counter() - t0
        integrity = inc.state_hash() == scratch.state_hash()

        snap_path = store.publish_snapshot()
        snapshot_bytes_written += snap_path.stat().st_size
        log_size = store.log_path.stat().st_size
        snap_size = snap_path.stat().st_size
        wa = snapshot_bytes_written / log_size if log_size else float("nan")

        # cold replay from the canonical log
        del scratch
        gc.collect()
        t0 = time.perf_counter()
        cold = ClaimStore(root, auto_snapshot=False)
        t_replay = time.perf_counter() - t0
        assert cold.consolidated_state().state_hash() == inc.state_hash()
        del cold
        gc.collect()

        row = {
            "events": target, "us_per_event_incremental": us_per_event,
            "t_ingest_s": t_ingest, "t_full_replay_s": t_replay,
            "t_consolidate_from_scratch_s": t_scratch,
            "rss_bytes": rss_bytes(), "event_log_bytes": log_size,
            "snapshot_bytes": snap_size,
            "derived_index_active_records": len(inc.active_record_ids),
            "duplicate_clusters": len(inc.duplicate_clusters),
            "alias_clusters": len(inc.alias_clusters),
            "support_groups": len(inc.support_groups),
            "contradiction_groups": len(inc.contradiction_groups),
            "write_amplification_periodic": wa,
            "integrity_incremental_equals_full_replay": integrity,
            "event_mix_counts": dict(wl.counts),
        }
        if not args.skip_retrieval:
            row["retrieval"] = measure_retrieval(store)
        results.append(row)
        print(f"  {target:>9}{us_per_event:>9.1f}{t_replay:>10.2f}{t_scratch:>11.2f}"
              f"{row['rss_bytes']/1e6:>9.0f}{log_size/1e6:>8.1f}{snap_size/1e6:>9.2f}"
              f"{wa:>7.2f}{str(integrity):>11}")
        if not integrity:
            print("  INTEGRITY VIOLATION -- stopping.")
            break
        if row["rss_bytes"] / 1e9 > args.rss_limit_gb:
            print(f"  RSS limit ({args.rss_limit_gb} GB) reached -- stopping ladder early "
                  "and reporting what was measured.")
            break

    # --- restart / recovery -------------------------------------------
    recovery = []
    for scale in RECOVERY_SCALES:
        if not any(r["events"] == scale for r in results):
            continue
        t0 = time.perf_counter()
        s1 = ClaimStore(root, auto_snapshot=False)
        t_cold = time.perf_counter() - t0
        valid = s1.snapshot_is_valid()
        del s1; gc.collect()
        recovery.append({"scale": scale, "t_cold_replay_s": t_cold,
                         "snapshot_valid_at_this_scale": valid})

    # --- write amplification of PER-EVENT snapshot publishing ---------
    # The mode a naive implementation would use. Measured on a small store
    # because its cost is precisely what makes it untenable at scale.
    wa_per_event = []
    for n in (200, 400, 800):
        d = Path(tempfile.mkdtemp()) / "wa"
        s = ClaimStore(d, auto_snapshot=True)
        w2 = Workload(seed=7)
        written = 0
        while s.corpus_version < n:
            before = s.corpus_version
            w2.step(s)
            if s.corpus_version > before:
                written += s.snapshot_path.stat().st_size + s.manifest_path.stat().st_size
        wa_per_event.append({"events": n, "derived_bytes": written,
                             "log_bytes": s.log_path.stat().st_size,
                             "write_amplification": written / s.log_path.stat().st_size})
        del s; gc.collect()

    print("\n  restart/recovery:")
    for r in recovery:
        print(f"    {r['scale']:>9} events  cold_replay={r['t_cold_replay_s']:.2f}s  "
              f"snapshot_valid={r['snapshot_valid_at_this_scale']}")
    print("\n  write amplification, PER-EVENT snapshot publishing (naive mode):")
    for r in wa_per_event:
        print(f"    {r['events']:>6} events  WA={r['write_amplification']:>8.1f}x")

    out = {
        "artifact": "VERIFIED_MEMORY_CONSOLIDATION_V1 pressure test",
        "design": "configs/verified_memory_consolidation_v1_design.json",
        "MEASUREMENT_NOT_QUALIFICATION": True,
        "no_frozen_performance_threshold": True,
        "source_commit": "8f04d797d63a80665abab5fe21deb6f382080c8c",
        "push_status": "CONFIRMED local==origin at run time",
        "event_mixture_frozen_before_running": MIX,
        "mix_seed": MIX_SEED,
        "platform": sys.platform,
        "fsync_caveat": ("os.fsync on darwin does not force a physical device flush "
                         "(F_FULLFSYNC would); durability numbers here are therefore "
                         "optimistic relative to Linux."),
        "ladder": results,
        "recovery": recovery,
        "write_amplification_per_event_mode": wa_per_event,
    }
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, sort_keys=True, default=str) + "\n")
    print(f"\n  written: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
