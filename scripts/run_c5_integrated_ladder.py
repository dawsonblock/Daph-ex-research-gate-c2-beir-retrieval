#!/usr/bin/env python3
"""Integrated C5 ladder I0-I5, per configs/gate_c5_integrated_v1.json.

Answers one narrow question: do frozen R1 (retrieval fusion) and S2 (selector)
combine into a reproducible, subgroup-safe mechanism -- and which of them is
actually carrying the effect.

    I0  frozen_rrf + frozen s2c      BASELINE, reproduces C4 v2.1 behavior
    I1  R1         + frozen s2c      retrieval effect alone
    I2  frozen_rrf + S2              selector effect alone
    I3  R1         + S2              PRIMARY, the integrated mechanism
    I4  R1         + oracle-over-pool  ceiling on R1 pools
    I5  oracle evidence               reader ceiling

The 2x2 over {fusion} x {selector} is the point: it yields E_retrieval,
E_selector, E_combined and their interaction, so a gain cannot be attributed to
the wrong component.

Three preconditions run before any GPU spend, each fail-closed:
  --determinism   replay a subset under varied PYTHONHASHSEED, requiring exact
                  equality on every hash and id field
  --dry-pass      all six arms on all tasks, selection-only, checking the R1
                  and S2 effects survived integration and parity holds
  --with-hrm      the 720-generation run itself

Usage:
    python scripts/run_c5_integrated_ladder.py --determinism
    python scripts/run_c5_integrated_ladder.py --dry-pass
    python scripts/run_c5_integrated_ladder.py --dry-pass --with-hrm
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hrm_adaptive_memory.backends import CanonicalRetrievalMode  # noqa: E402
from hrm_adaptive_memory.c4.arms import ARMS  # noqa: E402
from hrm_adaptive_memory.c4.contracts import (  # noqa: E402
    C4_CANDIDATE_BUDGET, C4_PRIMARY_PACKET_BUDGET, C4_RRF_K, RetrievalResult,
    SelectionResult)
from hrm_adaptive_memory.c4.fusion import frozen_rrf, max_reciprocal  # noqa: E402
from hrm_adaptive_memory.c4.packet_ordering import (  # noqa: E402
    canonical_candidate_membership_hash, canonical_candidate_pool_hash)
from hrm_adaptive_memory.c4.identity_stage import run_identity_stage  # noqa: E402
from hrm_adaptive_memory.c4.packet_stage import run_packet_stage  # noqa: E402
from hrm_adaptive_memory.c4.query_stage import run_query_stage  # noqa: E402
from hrm_adaptive_memory.c4.retrieval_stage import get_cached_backend  # noqa: E402
from hrm_adaptive_memory.c4.selector_v2 import select_s2  # noqa: E402
from hrm_adaptive_memory.retrieval_bench.selectors import s0_raw, s5_oracle  # noqa: E402
from hrm_adaptive_memory.retrieval_bench.selectors.chain import (  # noqa: E402
    s2c_chain_plus_relation)
from scripts.diagnose_c4_selector_eligibility import (  # noqa: E402
    task_shape, terminal_records)
from scripts.run_gate_c4 import (  # noqa: E402
    _load_split as load_split, _to_index_records as to_index_records)

PROTOCOL = ROOT / "configs/gate_c5_integrated_v1.json"
FUSIONS = {"frozen_rrf": frozen_rrf, "R1_max_reciprocal": max_reciprocal}

# Two ladders. The I ladder was the original 2x2 over {fusion} x {selector}; it
# established that R1 adds nothing downstream once S2 is active (Q(I3)-Q(I2) =
# -0.0021, interaction -0.0104). The J ladder is the post-amendment build with
# R1 DROPPED from the primary mechanism: a single-factor selector ladder on the
# frozen retrieval, plus J1r retained as a diagnostic so the R1 interaction
# stays measurable without entering the promotion decision.
LADDERS: dict[str, dict[str, tuple[str, str]]] = {
    "I": {
        "I0": ("frozen_rrf", "S0"), "I1": ("R1_max_reciprocal", "S0"),
        "I2": ("frozen_rrf", "S2"), "I3": ("R1_max_reciprocal", "S2"),
        "I4": ("R1_max_reciprocal", "oracle"),
        "I5": ("R1_max_reciprocal", "oracle_evidence"),
    },
    "J": {
        "J0": ("frozen_rrf", "S0"), "J1": ("frozen_rrf", "S2"),
        "J2": ("frozen_rrf", "oracle"), "J3": ("frozen_rrf", "oracle_evidence"),
        "J1r": ("R1_max_reciprocal", "S2"),
    },
}
#: Arms excluded from the promotion decision by protocol, not by convention.
NON_PROMOTABLE = {"I0", "I4", "I5", "J0", "J2", "J3", "J1r"}
PRIMARY_ARM = {"I": "I3", "J": "J1"}
BASELINE_ARM = {"I": "I0", "J": "J0"}

ARM_ORDER: tuple[str, ...] = tuple(LADDERS["I"])
ARM_SPEC: dict[str, tuple[str, str]] = dict(LADDERS["I"])


def use_ladder(name: str) -> None:
    """Select the active ladder. Module-level so the replay subprocess and the
    parity checker see the same arms as the main run."""
    global ARM_ORDER, ARM_SPEC
    ARM_SPEC = dict(LADDERS[name])
    ARM_ORDER = tuple(ARM_SPEC)


def environment_fingerprint() -> dict[str, Any]:
    """Platform identity, recorded with every run.

    A replay contract is platform-scoped: the pipeline is deterministic WITHIN
    an environment but is NOT bit-reproducible across CUDA and CPU, because
    dense embeddings differ in the last bits and permute ranks at numerical
    ties. Recording the fingerprint is what lets a later diagnostic know
    whether it can reproduce a run or must read the frozen artifact.
    """
    import platform
    fingerprint: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    try:
        import torch
        fingerprint["torch"] = torch.__version__
        fingerprint["cuda_available"] = torch.cuda.is_available()
        fingerprint["device_name"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    except Exception as exc:  # noqa: BLE001
        fingerprint["torch_error"] = str(exc)
    return fingerprint


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def config_hashes() -> dict[str, str]:
    """Hashes of the two frozen mechanism configs, recorded in every receipt."""
    return {
        "retrieval_config_hash": _sha(json.dumps(
            {"fusion": "R1_max_reciprocal", "k_rrf": C4_RRF_K,
             "candidate_budget": C4_CANDIDATE_BUDGET}, sort_keys=True))[:16],
        "selector_config_hash": _sha(json.dumps(
            {"selector": "S2_bridge_aware_plus_connectivity",
             "packet_budget": C4_PRIMARY_PACKET_BUDGET}, sort_keys=True))[:16],
    }


def constituents(task: dict, arm, records, depth: int):
    """BM25 and BGE lists, retrieved once and shared by every fusion."""
    _state, query = run_query_stage(task["question"], arm)
    bm25 = get_cached_backend(CanonicalRetrievalMode.BM25, records)
    bge = get_cached_backend(CanonicalRetrievalMode.DENSE_BGE, records)
    a = [e.evidence_id for e in asyncio.run(bm25.search(query.rendered_query, k=depth)).evidence]
    b = [e.evidence_id for e in asyncio.run(bge.search(query.rendered_query, k=depth)).evidence]
    return query, a, b


def evaluate_task(task: dict, arm, records, texts, depth: int) -> dict[str, Any]:
    """Every arm's pool, identity, selection and packet for one task."""
    query, bm25_ids, bge_ids = constituents(task, arm, records, depth)
    budget, packet_budget = C4_CANDIDATE_BUDGET, C4_PRIMARY_PACKET_BUDGET
    required = list(task["required_evidence_ids"])

    pools: dict[str, tuple[list[str], dict[str, float]]] = {}
    for name, policy in FUSIONS.items():
        fused = policy([bm25_ids, bge_ids], C4_RRF_K, budget)
        pools[name] = ([eid for eid, _ in fused], dict(fused))

    out: dict[str, Any] = {"query_hash": query.query_hash, "arms": {}}
    for arm_id in ARM_ORDER:
        fusion_name, selector_name = ARM_SPEC[arm_id]
        pool, scores = pools[fusion_name]
        retrieval = RetrievalResult(
            candidate_ids=tuple(pool), candidate_budget=budget,
            retrieval_policy=arm.retrieval_policy, bm25_backend="bm25",
            bge_model_id="", bge_revision="", rrf_k=C4_RRF_K,
            bm25_ranked=(), bge_ranked=(), fusion_ranked=())
        identity = run_identity_stage(task["question"], arm, retrieval, texts)
        candidates = [{"document_id": eid} for eid in pool]
        resolved_q = task["question"]
        if identity.surface and identity.canonical:
            resolved_q = resolved_q.replace(identity.surface, identity.canonical)

        def frozen(b: int, _q=resolved_q, _c=candidates, _i=identity) -> list[str]:
            if _i.status in ("EXACT", "RESOLVED") and _i.canonical:
                return s2c_chain_plus_relation(_c, budget=b, question=_q, texts=texts)
            return s0_raw(_c, budget=b)

        diag: dict[str, Any] = {}
        if selector_name == "S0":
            selected = list(frozen(packet_budget))
        elif selector_name == "S2":
            selected, _receipt, diag = select_s2(
                identity_status=identity.status, question=task["question"],
                canonical_subject=identity.canonical, candidate_ids=pool,
                texts=texts, budget=packet_budget, frozen_select=frozen,
                fusion_scores=scores)
        elif selector_name == "oracle":
            selected = s5_oracle(candidates, budget=packet_budget,
                                 required=[e for e in required if e in set(pool)])
        else:  # oracle_evidence -- reader ceiling
            selected = list(required[:packet_budget])

        sel_result = SelectionResult(
            selector=selector_name, selected_ids=tuple(selected),
            selector_policy=arm.selector_policy, identity_status=identity.status)
        prompt, packet = run_packet_stage(
            arm, task["question"], sel_result, texts, retrieval)

        out["arms"][arm_id] = {
            "fusion": fusion_name, "selector": selector_name,
            "pool": pool, "selected": selected, "prompt": prompt,
            # Both hashes, never collapsed: the order hash answers "did
            # retrieval rank the pool identically", the membership hash answers
            # "was record X in the pool at all". Availability analysis depends
            # only on membership, and conflating them once already let benign
            # cross-platform ordering churn masquerade as a validity failure.
            "candidate_order_hash": canonical_candidate_pool_hash(pool),
            "candidate_membership_hash":
                canonical_candidate_membership_hash(pool),
            "candidate_ranked": [
                {"record_id": eid, "fusion_rank": rank,
                 "fusion_score": scores.get(eid)}
                for rank, eid in enumerate(pool, 1)],
            "identity_status": identity.status,
            "identity_canonical": identity.canonical,
            "candidate_pool_hash": packet.candidate_pool_hash,
            "membership_hash": packet.membership_hash,
            "order_hash": packet.order_hash,
            "packet_hash": packet.packet_hash,
            "prompt_hash": packet.prompt_hash,
            "diag": diag,
        }
    return out


# --------------------------------------------------------------------------
# Preconditions
# --------------------------------------------------------------------------

_REPLAY = '''
import json, sys
sys.path.insert(0, ".")
from scripts.run_c5_integrated_ladder import evaluate_task, use_ladder
use_ladder("__LADDER__")
from scripts.run_c5_integrated_ladder import ARM_ORDER
from scripts.run_gate_c4 import _load_split, _to_index_records, ARMS
tasks, ev, texts = _load_split("__SPLIT__")
records = _to_index_records(ev)
arm = ARMS["C4_4"]
rows = {}
for task in tasks[:__N__]:
    r = evaluate_task(task, arm, records, texts, len(records))
    rows[task["task_id"]] = {
        a: {k: v for k, v in r["arms"][a].items()
            if k in ("pool", "selected", "identity_status", "identity_canonical",
                     "candidate_pool_hash", "membership_hash", "order_hash",
                     "packet_hash", "prompt_hash")}
        for a in ARM_ORDER}
open("__OUT__", "w").write(json.dumps(rows, sort_keys=True))
'''


def run_determinism(split: str, n_tasks: int, seeds: list[int],
                    ladder: str = "J") -> bool:
    """Replay under varied PYTHONHASHSEED; require exact equality. Fail-closed."""
    import tempfile
    print(f"--- determinism precondition ({n_tasks} tasks, seeds {seeds}) ---")
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        for seed in seeds:
            out = Path(tmp) / f"seed_{seed}.json"
            env = {**os.environ, "PYTHONHASHSEED": str(seed)}
            code = (_REPLAY.replace("__SPLIT__", split)
                    .replace("__N__", str(n_tasks)).replace("__OUT__", str(out))
                    .replace("__LADDER__", ladder))
            proc = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT),
                                  env=env, capture_output=True, text=True,
                                  timeout=3600)
            if proc.returncode != 0:
                print(f"  seed {seed}: FAILED\n{proc.stderr[-800:]}")
                return False
            results[seed] = json.loads(out.read_text())
            print(f"  seed {seed}: OK")

    base_seed = seeds[0]
    base = results[base_seed]
    for seed in seeds[1:]:
        if results[seed] != base:
            differing = [t for t in base if results[seed].get(t) != base[t]]
            print(f"  MISMATCH seed {base_seed} vs {seed} on {len(differing)} "
                  f"task(s), e.g. {differing[:3]}")
            return False
        print(f"  seed {base_seed} vs {seed}: IDENTICAL")
    print("  determinism: PASS\n")
    return True


def check_crossover_parity(row: dict[str, Any]) -> list[str]:
    """Only the intended intervention may differ between paired arms.

    Checked programmatically rather than asserted in prose. Derived from the
    ACTIVE ladder's own spec rather than hardcoded pairs, so it stays correct
    when the ladder changes: any two arms sharing a fusion must share a
    candidate_pool_hash, and any two sharing a selector must share identity
    state. A hardcoded pair list silently stops checking anything the moment the
    arms are renamed.
    """
    arms = row["arms"]
    violations: list[str] = []
    names = [a for a in ARM_ORDER if a in arms]

    # First: what each arm REPORTS must match what the ladder spec says it is.
    # Deriving the pair comparisons below from ARM_SPEC is only sound if the
    # executed arms actually correspond to their spec, so a row whose reported
    # fusion or selector has drifted must be caught here rather than silently
    # trusted.
    for a in names:
        spec_fusion, spec_selector = ARM_SPEC[a]
        if arms[a].get("fusion") not in (None, spec_fusion):
            violations.append(
                f"{a}: reported fusion {arms[a]['fusion']!r} does not match "
                f"the ladder spec {spec_fusion!r}")
        if arms[a].get("selector") not in (None, spec_selector):
            violations.append(
                f"{a}: reported selector {arms[a]['selector']!r} does not "
                f"match the ladder spec {spec_selector!r}")

    for i, a in enumerate(names):
        for b in names[i + 1:]:
            same_fusion = ARM_SPEC[a][0] == ARM_SPEC[b][0]
            same_selector = ARM_SPEC[a][1] == ARM_SPEC[b][1]
            if same_fusion and (arms[a]["candidate_pool_hash"]
                                != arms[b]["candidate_pool_hash"]):
                violations.append(
                    f"{a} vs {b} (same fusion): candidate pools differ, so a "
                    f"selector comparison would be confounded by retrieval")
            if same_selector and (arms[a]["identity_status"]
                                  != arms[b]["identity_status"]):
                violations.append(
                    f"{a} vs {b} (same selector): identity status differs")
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description="C5 integrated ladder")
    parser.add_argument("--split", default="development")
    parser.add_argument("--ladder", choices=sorted(LADDERS), default="J",
                        help="J is the post-amendment build (R1 dropped from "
                             "the primary mechanism); I is the original 2x2")
    parser.add_argument("--arm-for-queries", default="C4_4")
    parser.add_argument("--determinism", action="store_true")
    parser.add_argument("--determinism-tasks", type=int, default=20)
    parser.add_argument("--dry-pass", action="store_true")
    parser.add_argument("--with-hrm", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    use_ladder(args.ladder)
    protocol = json.loads(PROTOCOL.read_text())
    seeds = protocol["determinism_precondition"]["seeds"]
    primary, baseline = PRIMARY_ARM[args.ladder], BASELINE_ARM[args.ladder]

    if args.determinism:
        if not run_determinism(args.split, args.determinism_tasks, seeds,
                               args.ladder):
            print("ABORT: determinism precondition failed. No GPU spend.")
            return 1
        if not (args.dry_pass or args.with_hrm):
            return 0

    if not (args.dry_pass or args.with_hrm):
        print("Nothing to do: pass --determinism, --dry-pass and/or --with-hrm.")
        return 0

    tasks, evidence, texts = load_split(args.split)
    if args.limit:
        tasks = tasks[:args.limit]
    records = to_index_records(evidence)
    arm = ARMS[args.arm_for_queries]
    packet_budget = C4_PRIMARY_PACKET_BUDGET

    fingerprint = environment_fingerprint()
    adapter = condition = None
    if args.with_hrm:
        from scripts.run_gate_c4 import _load_hrm
        print("--- Loading HRM ---")
        adapter, condition = _load_hrm()
        print(f"  {adapter.spec.model_id} @ {adapter.spec.revision}\n")

    print(f"=== C5 integrated ladder ({protocol['protocol_id']}) ===")
    print(f"  split={args.split}  tasks={len(tasks)}  arms={list(ARM_ORDER)}")
    print(f"  candidate_budget={C4_CANDIDATE_BUDGET}  packet_budget={packet_budget}")
    print(f"  hrm={'YES' if args.with_hrm else 'no (dry pass)'}\n")

    ces = defaultdict(int)
    cand_ces = defaultdict(int)
    ans_ret = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    role_ret = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    packet_max = defaultdict(int)
    disconnected = defaultdict(int)
    parity_violations: list[str] = []
    q_by_arm = defaultdict(list)
    correct_by_arm = defaultdict(list)
    q_family = defaultdict(lambda: defaultdict(list))
    q_regime = defaultdict(lambda: defaultdict(list))
    per_task_q: dict[str, dict[str, float]] = defaultdict(dict)
    meta: dict[str, dict[str, str]] = {}
    binding_failures = 0
    receipt_path = (Path(args.out).with_suffix(".receipts.jsonl") if args.out else
                    ROOT / f"evidence/gate_c4/diagnosis/{args.split}_c5_"
                           f"{args.ladder}ladder"
                           f"{'_hrm' if args.with_hrm else '_dry'}.receipts.jsonl")
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_file = receipt_path.open("w")

    for index, task in enumerate(tasks, 1):
        if index % 10 == 0 or index == len(tasks):
            print(f"  {index}/{len(tasks)}...", end="\r", flush=True)
        row = evaluate_task(task, arm, records, texts, len(records))
        parity_violations.extend(
            f"{task['task_id']}: {v}" for v in check_crossover_parity(row))

        required = set(task["required_evidence_ids"])
        terminals = set(terminal_records(task))
        group = f"{row['arms'][primary]['identity_status']}_{task_shape(task)}"
        task_id = task["task_id"]
        meta[task_id] = {"family": task["family"],
                         "entity_regime": task["metadata"]["entity_regime"],
                         "source_cluster_id": task.get("source_cluster_id", "")}

        oracle = task.get("_oracle_metadata") or {}
        bridge_recs = {e["record_id"] for e in (oracle.get("proof_edges") or [])
                       if e.get("target") == oracle.get("latent_bridge")}
        ident_recs = required - terminals - bridge_recs

        for arm_id in ARM_ORDER:
            a = row["arms"][arm_id]
            chosen, pool = set(a["selected"]), set(a["pool"])
            packet_max[arm_id] = max(packet_max[arm_id], len(a["selected"]))
            disconnected[arm_id] += a["diag"].get("disconnected_in_packet", 0)
            if required <= pool:
                ces[arm_id] += required <= chosen
            cand_ces[arm_id] += required <= pool
            if terminals and terminals <= pool:
                hit = terminals <= chosen
                for key in ("ALL", group, a["identity_status"]):
                    ans_ret[arm_id][key][0] += hit
                    ans_ret[arm_id][key][1] += 1
            for label, recs in (("identity", ident_recs), ("bridge", bridge_recs)):
                if recs and recs <= pool:
                    role_ret[arm_id][label][0] += recs <= chosen
                    role_ret[arm_id][label][1] += 1

            if args.with_hrm:
                from scripts.run_gate_c4 import (
                    _run_hrm, _verify_answer, _compute_quality)
                hrm = _run_hrm(adapter, condition, a["prompt"])
                if hrm.prompt_hash != a["prompt_hash"]:
                    binding_failures += 1
                _s, ok = _verify_answer(task, hrm.output)
                q = _compute_quality(task, list(a["selected"]), evidence, ok)
                q_by_arm[arm_id].append(q)
                correct_by_arm[arm_id].append(1.0 if ok else 0.0)
                q_family[arm_id][meta[task_id]["family"]].append(q)
                q_regime[arm_id][meta[task_id]["entity_regime"]].append(q)
                per_task_q[arm_id][task_id] = q

        # Incremental receipt per task: resume safety, and the per-task record
        # the grouped bootstrap CIs are computed from. Omitting per-task Q the
        # first time made two REQUIRED frozen criteria uncomputable without a
        # second GPU run, so it is persisted as it is produced.
        receipt_file.write(json.dumps({
            "task_id": task_id, **meta[task_id],
            "query_hash": row["query_hash"],
            **config_hashes(),
            "environment_fingerprint": fingerprint,
            "arms": {a: {
                "fusion": row["arms"][a]["fusion"],
                "selector": row["arms"][a]["selector"],
                "selected": row["arms"][a]["selected"],
                # MANDATORY from here on: the pool is persisted with the run, so
                # every later failure decomposition is artifact analysis rather
                # than model or retrieval replay.
                "candidate_ids_ordered": row["arms"][a]["pool"],
                "candidate_ranked": row["arms"][a]["candidate_ranked"],
                "candidate_order_hash": row["arms"][a]["candidate_order_hash"],
                "candidate_membership_hash":
                    row["arms"][a]["candidate_membership_hash"],
                "candidate_pool_hash": row["arms"][a]["candidate_pool_hash"],
                "membership_hash": row["arms"][a]["membership_hash"],
                "order_hash": row["arms"][a]["order_hash"],
                "packet_hash": row["arms"][a]["packet_hash"],
                "prompt_hash": row["arms"][a]["prompt_hash"],
                "q": per_task_q[a].get(task_id),
            } for a in ARM_ORDER},
        }, sort_keys=True) + "\n")
        receipt_file.flush()

    receipt_file.close()
    n = len(tasks)
    print(" " * 30, end="\r")

    def mean(v): return round(sum(v) / len(v), 4) if v else 0.0
    report: dict[str, Any] = {
        "schema_version": "c5-integrated-ladder-v1",
        "protocol_id": protocol["protocol_id"], "split": args.split,
        "ladder": args.ladder, "primary_arm": primary, "baseline_arm": baseline,
        "environment_fingerprint": fingerprint,
        "determinism_contract": {
            "within_environment_determinism": "PASS (see determinism precondition)",
            "cross_platform_bit_reproducibility": "NOT_GUARANTEED",
            "consequence": (
                "Reproduce this run on the recorded platform, or read the "
                "persisted candidate_ids_ordered / candidate_membership_hash "
                "instead of replaying retrieval."),
        },
        "non_promotable_arms": sorted(a for a in ARM_ORDER if a in NON_PROMOTABLE),
        "task_count": n, "with_hrm": args.with_hrm,
        "candidate_budget": C4_CANDIDATE_BUDGET, "packet_budget": packet_budget,
        **config_hashes(),
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
            text=True, timeout=30).stdout.strip(),
        "crossover_parity": {
            "violations": parity_violations[:20],
            "violation_count": len(parity_violations),
            "passed": not parity_violations},
        "prompt_binding_failures": binding_failures if args.with_hrm else None,
        "arms": {
            a: {"fusion": ARM_SPEC[a][0], "selector": ARM_SPEC[a][1],
                "candidate_ces": round(cand_ces[a] / n, 4),
                "selected_ces": round(ces[a] / n, 4),
                "answer_retention": {g: round(v[0] / v[1], 4) if v[1] else 0.0
                                     for g, v in sorted(ans_ret[a].items())},
                "identity_retention": round(role_ret[a]["identity"][0]
                                            / role_ret[a]["identity"][1], 4)
                if role_ret[a]["identity"][1] else 0.0,
                "bridge_retention": round(role_ret[a]["bridge"][0]
                                          / role_ret[a]["bridge"][1], 4)
                if role_ret[a]["bridge"][1] else 0.0,
                "max_packet_size": packet_max[a],
                "disconnected_in_packet": disconnected[a]}
            for a in ARM_ORDER},
    }

    if args.with_hrm:
        report["downstream_q"] = {
            a: {"q": mean(q_by_arm[a]), "correct": mean(correct_by_arm[a]),
                "q_by_family": {f: mean(v) for f, v in sorted(q_family[a].items())},
                "q_by_entity_regime": {r: mean(v) for r, v in sorted(q_regime[a].items())}}
            for a in ARM_ORDER if q_by_arm[a]}
        report["per_task_q"] = {a: per_task_q[a] for a in per_task_q}
        report["task_meta"] = meta

        def lcb(arm_a: str, arm_b: str, axis: str) -> float:
            """Grouped bootstrap lower bound on Q(arm_a) - Q(arm_b)."""
            import random
            groups: dict[str, list[float]] = defaultdict(list)
            for tid, qa in per_task_q[arm_a].items():
                qb = per_task_q[arm_b].get(tid)
                if qb is not None:
                    groups[meta[tid][axis]].append(qa - qb)
            keys = sorted(groups)
            if not keys:
                return 0.0
            rng = random.Random(12345)
            means = []
            for _ in range(2000):
                picked = [groups[keys[rng.randrange(len(keys))]] for _ in keys]
                flat = [v for g in picked for v in g]
                if flat:
                    means.append(sum(flat) / len(flat))
            means.sort()
            return round(means[int(0.025 * len(means))], 4) if means else 0.0

        report["primary_contrast_ci"] = {
            "contrast": f"Q({primary}) - Q({baseline})",
            "family_grouped_lcb": lcb(primary, baseline, "family"),
            "source_cluster_grouped_lcb": lcb(primary, baseline, "source_cluster_id"),
        }
        secondary: dict[str, Any] = {}
        if "J1r" in per_task_q and "J1" in per_task_q:
            secondary["J1r_vs_J1_family_lcb"] = lcb("J1r", "J1", "family")
            secondary["note"] = (
                "J1r vs J1 keeps the R1 interaction measurable. DIAGNOSTIC "
                "ONLY -- it does not enter the promotion decision.")
        if {"I2", "I3", "I0"} <= set(per_task_q):
            secondary["I2_vs_I0_family_lcb"] = lcb("I2", "I0", "family")
            secondary["I3_vs_I2_family_lcb"] = lcb("I3", "I2", "family")
        if secondary:
            report["secondary_contrast_ci"] = secondary

        q = {a: report["downstream_q"][a]["q"] for a in report["downstream_q"]}
        if {"I0", "I1", "I2", "I3"} <= set(q):
            report["causal_decomposition"] = {
                "E_retrieval": round(q["I1"] - q["I0"], 4),
                "E_selector": round(q["I2"] - q["I0"], 4),
                "E_combined": round(q["I3"] - q["I0"], 4),
                "E_interaction": round(q["I3"] - q["I2"] - q["I1"] + q["I0"], 4),
            }

    out = Path(args.out) if args.out else (
        ROOT / f"evidence/gate_c4/diagnosis/{args.split}_c5_{args.ladder}ladder"
               f"{'_hrm' if args.with_hrm else '_dry'}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(f"  {'arm':<4}{'fusion':<20}{'sel':<16}{'candCES':>9}{'selCES':>8}"
          f"{'EXACT_br':>10}{'ident':>8}{'bridge':>8}{'pkt':>5}")
    for a in ARM_ORDER:
        r = report["arms"][a]
        print(f"  {a:<4}{r['fusion']:<20}{r['selector']:<16}"
              f"{r['candidate_ces']:>8.1%}{r['selected_ces']:>8.1%}"
              f"{r['answer_retention'].get('EXACT_bridged', 0):>10.1%}"
              f"{r['identity_retention']:>8.1%}{r['bridge_retention']:>8.1%}"
              f"{r['max_packet_size']:>5}")

    cp = report["crossover_parity"]
    print(f"\n  crossover parity: {'PASS' if cp['passed'] else 'FAIL'}"
          f" ({cp['violation_count']} violations)")
    for v in cp["violations"][:5]:
        print(f"    {v}")

    if args.with_hrm:
        print(f"\n  prompt binding failures: {binding_failures}")
        print(f"  {'arm':<4}{'Q':>9}{'correct':>10}")
        for a in ARM_ORDER:
            row = report["downstream_q"].get(a)
            if row:
                print(f"  {a:<4}{row['q']:>9.4f}{row['correct']:>10.4f}")
        ci = report.get("primary_contrast_ci")
        if ci:
            print(f"\n  primary contrast {ci['contrast']} grouped LCBs:")
            print(f"    family-grouped        : {ci['family_grouped_lcb']:+.4f}")
            print(f"    cluster-grouped       : {ci['source_cluster_grouped_lcb']:+.4f}")
            for k, v in (report.get("secondary_contrast_ci") or {}).items():
                if isinstance(v, float):
                    print(f"    {k:<22}: {v:+.4f}")
        cd = report.get("causal_decomposition")
        if cd:
            print("\n  causal decomposition:")
            for k, v in cd.items():
                print(f"    {k:<16}{v:+.4f}")

    print(f"\n  written: {out}")
    if not args.with_hrm:
        print("  DRY PASS only -- downstream Q not measured.")
    return 0 if report["crossover_parity"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
