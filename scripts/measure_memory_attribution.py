#!/usr/bin/env python3
"""Attribute ClaimStore RSS to individual structures.

Motivated by an explicit instruction not to ASSUME that removing the
in-memory event list halves RSS at 1M. _records, the indexes, string data and
dict overhead may well account for more than half, and the fix should be
aimed by measurement rather than intuition.

Method: a recursive sizeof walker. Two numbers are reported per structure
because neither alone is honest:

  independent  each structure walked with a FRESH memo, so shared objects
               (notably interned/shared strings and ClaimRecord instances
               referenced from several indexes) are counted in every
               structure that reaches them. This is an UPPER bound on a
               structure's cost and the two columns will sum to more than
               the union.
  marginal     the increase in the union walk when this structure is added
               last, i.e. what would actually be RECLAIMED by deleting it.
               This is the number that predicts a fix's benefit.

The gap between the two is precisely the shared data, so reporting both makes
the sharing visible instead of hiding it in one arbitrary attribution order.
"""
from __future__ import annotations

import argparse
import gc
import json
import resource
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import hrm_adaptive_memory.evaluation  # noqa: E402,F401  (cycle-breaker)

from hrm_adaptive_memory.memory_write import ClaimStore  # noqa: E402


def deep_size(obj, seen: set[int]) -> int:
    oid = id(obj)
    if oid in seen:
        return 0
    seen.add(oid)
    try:
        size = sys.getsizeof(obj)
    except TypeError:
        return 0
    if isinstance(obj, dict):
        for k, v in obj.items():
            size += deep_size(k, seen) + deep_size(v, seen)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for i in obj:
            size += deep_size(i, seen)
    elif hasattr(obj, "__dict__"):
        size += deep_size(vars(obj), seen)
    return size


def rss_bytes() -> int:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r if sys.platform == "darwin" else r * 1024


def structures(store: ClaimStore) -> dict:
    idx = store._index
    out = {
        "_records": store._records,
        "_active_by_claim_key": store._active_by_claim_key,
        "consolidation._active": idx._active,
        "consolidation._by_content": idx._by_content,
        "consolidation._alias_edges": idx._alias_edges,
        "consolidation._entities": idx._entities,
    }
    if hasattr(store, "_events"):
        out["_events"] = store._events
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-log", required=True)
    ap.add_argument("--scale", type=int, default=100_000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    src = Path(args.source_log)
    work = ROOT / "evidence/memory_write/_attr_tmp"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    with src.open() as fin, (work / "claims_events.jsonl").open("w") as fout:
        for i, line in enumerate(fin):
            if i >= args.scale:
                break
            fout.write(line)

    rss0 = rss_bytes()
    store = ClaimStore(work, auto_snapshot=False)
    gc.collect()
    rss1 = rss_bytes()

    st = structures(store)
    independent = {name: deep_size(obj, set()) for name, obj in st.items()}

    # marginal: what deleting this structure would actually reclaim
    marginal = {}
    for name in st:
        seen: set[int] = set()
        for other, obj in st.items():
            if other != name:
                deep_size(obj, seen)
        before = len(seen)
        total_without = sum(sys.getsizeof(o) for o in ())  # placeholder, use walk below
        union_without = 0
        seen2: set[int] = set()
        for other, obj in st.items():
            if other != name:
                union_without += deep_size(obj, seen2)
        seen3: set[int] = set()
        union_with = 0
        for other, obj in st.items():
            union_with += deep_size(obj, seen3)
        marginal[name] = union_with - union_without
        del seen, seen2, seen3, before, total_without

    seen_all: set[int] = set()
    union_total = sum(deep_size(obj, seen_all) for obj in st.values())

    print(f"=== ClaimStore memory attribution @ {args.scale:,} events ===")
    print(f"  process RSS after load : {(rss1)/1e6:>10.0f} MB")
    print(f"  RSS delta from load    : {(rss1-rss0)/1e6:>10.0f} MB")
    print(f"  union of all structures: {union_total/1e6:>10.0f} MB "
          f"(python-object accounting, excludes allocator overhead)\n")
    print(f"  {'structure':<30}{'independent_MB':>16}{'marginal_MB':>14}{'marg_share':>12}")
    for name in sorted(st, key=lambda n: -marginal[n]):
        share = marginal[name] / union_total if union_total else 0.0
        print(f"  {name:<30}{independent[name]/1e6:>16.1f}{marginal[name]/1e6:>14.1f}{share:>11.1%}")

    ev = marginal.get("_events", 0)
    print(f"\n  removing _events would reclaim ~{ev/1e6:.0f} MB of "
          f"{union_total/1e6:.0f} MB accounted = {ev/union_total if union_total else 0:.1%}")
    print("  (measured, NOT assumed -- the remainder stays resident regardless)")

    out = {
        "scale": args.scale,
        "rss_after_load_bytes": rss1,
        "union_of_structures_bytes": union_total,
        "independent_bytes": independent,
        "marginal_bytes": marginal,
        "events_marginal_share_of_accounted": (ev / union_total) if union_total else None,
        "caveat": ("independent counts shared objects once per structure and is an upper "
                   "bound; marginal is what deleting the structure would reclaim. Neither "
                   "includes CPython allocator overhead, so union < RSS is expected."),
    }
    p = Path(args.out) if args.out else (
        ROOT / f"evidence/memory_write/memory_attribution_{args.scale}.json")
    p.write_text(json.dumps(out, indent=2, sort_keys=True, default=str) + "\n")
    print(f"\n  written: {p}")
    del store
    shutil.rmtree(work)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
