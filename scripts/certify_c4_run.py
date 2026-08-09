#!/usr/bin/env python3
"""Authoritative certification of a Gate C4 v2_1 result bundle.

    VALID_RUN = all(gates)

and every gate is *derived from the artifacts*, never asserted by the author of
the notebook. The failure mode this replaces:

    valid_run_status = (protocol_sha256 == expected_protocol_sha)
    ...
    "determinism_gate": "PASSED (100% Parity)",   # hardcoded string
    "all_arms_complete": True,                    # hardcoded bool
    "result_hashes_verified": True,               # hardcoded bool

A certificate whose prerequisites are string literals certifies nothing. Here,
each gate is a function over the bundle; a gate that cannot be evaluated is
FAILED, never skipped, and any failed gate makes VALID_RUN false.

Usage:
    python scripts/certify_c4_run.py \
        --bundle evidence/gate_c4/full/development \
        --protocol configs/gate_c4_protocol_v2_1.json \
        --lock configs/c4_requirements.lock

    # inspect gates without a full test run (VALID_RUN will be false)
    python scripts/certify_c4_run.py --bundle ... --no-tests

Outputs <bundle>/certification/CERTIFICATION.json. Lineage artifacts are
written under <bundle>/certification/ so the bundle's own RESULTS.sha256
(computed over files directly in <bundle>) stays valid and verifiable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hrm_adaptive_memory.c4 import git_state  # noqa: E402
from hrm_adaptive_memory.c4.arms import PRIMARY_ORDER  # noqa: E402
from hrm_adaptive_memory.c4.environment_lock import (  # noqa: E402
    load_lock, verify_environment)
from hrm_adaptive_memory.experiment_integrity.metric_validation import (  # noqa: E402
    MetricValidationError, require_finite_number)
from hrm_adaptive_memory.c4.metrics import (  # noqa: E402
    compute_quality, evidence_complete, oracle_gap_capture, selector_gap_capture)
from hrm_adaptive_memory.c4.packet_ordering import ORDERING_POLICY_ID  # noqa: E402
from hrm_adaptive_memory.c4.protocol_validation import (  # noqa: E402
    ProtocolViolation, load_and_validate_protocol)
from hrm_adaptive_memory.c4.development_lineage import (  # noqa: E402
    check_development_lineage)
from hrm_adaptive_memory.c4.provenance import verify_results_hash  # noqa: E402
from hrm_adaptive_memory.c4.receipts import assert_runtime_clean  # noqa: E402

# Source tree that must be snapshotted and hashed for lineage. Defined once in
# git_state so the snapshot and the cleanliness check cannot disagree.
SOURCE_PATHS = git_state.SOURCE_PATHS

# Determinism fields the qualification receipt must have compared.
REQUIRED_DETERMINISM_FIELDS = frozenset({
    "candidate_pool_hash", "membership_hash", "order_hash", "packet_hash",
    "prompt_hash",
})

# Protocol D4 threshold for the primary mechanism delta.
PRIMARY_DELTA_THRESHOLD = 0.15

# Arms expected to apply the deterministic ordering policy.
DETERMINISTIC_ORDER_ARMS = frozenset({"C4_4", "C4_3o"})

# Agreement tolerance between analysis.json and recomputation from receipts.
# The analyzer's bootstrap is seeded, so CIs are reproducible exactly.
METRIC_TOLERANCE = 1e-9


def _load_analyzer():
    """Import scripts/analyze_gate_c4.py as a module.

    Certification recomputes metrics with the same authoritative functions the
    analyzer uses, so agreement means "the analyzer ran on these receipts",
    not "two independent implementations happen to agree".
    """
    import importlib.util

    path = ROOT / "scripts/analyze_gate_c4.py"
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("_c4_analyzer", path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:  # noqa: BLE001
        return None
    return module


# --- Gate plumbing ----------------------------------------------------------

@dataclass
class Gate:
    name: str
    passed: bool = False
    detail: dict[str, Any] = field(default_factory=dict)
    violations: list[str] = field(default_factory=list)

    def fail(self, message: str) -> "Gate":
        self.violations.append(message)
        self.passed = False
        return self

    def finalize(self) -> "Gate":
        self.passed = not self.violations
        return self

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "violations": self.violations,
            "detail": self.detail,
        }


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _packet(receipt: dict) -> dict:
    return receipt.get("runtime_payload", {}).get("packet", {})


# --- Gates ------------------------------------------------------------------

def gate_source_lineage(bundle: Path, manifest: dict,
                        expect_source_sha: str | None) -> tuple[Gate, dict]:
    """Hash the source tree and bind it to the bundle's git commit."""
    g = Gate("source_lineage")

    entries: dict[str, str] = {}
    for rel in SOURCE_PATHS:
        p = ROOT / rel
        if not p.exists():
            g.fail(f"source path missing: {rel}")
            continue
        if p.is_file():
            entries[rel] = _sha256_file(p)
            continue
        for f in sorted(p.rglob("*")):
            if not f.is_file() or "__pycache__" in f.parts or f.suffix == ".pyc":
                continue
            entries[str(f.relative_to(ROOT))] = _sha256_file(f)

    tree_lines = "".join(f"{h}  {name}\n" for name, h in sorted(entries.items()))
    source_sha = hashlib.sha256(tree_lines.encode()).hexdigest()
    snapshot = {
        "schema_version": "c4-source-snapshot-v1",
        "source_paths": list(SOURCE_PATHS),
        "file_count": len(entries),
        "source_tree_sha256": source_sha,
        "files": dict(sorted(entries.items())),
    }
    g.detail["source_tree_sha256"] = source_sha
    g.detail["file_count"] = len(entries)

    if expect_source_sha and expect_source_sha != source_sha:
        g.fail(f"source tree sha mismatch: expected {expect_source_sha}, "
               f"computed {source_sha}")

    commit = manifest.get("git_commit")
    g.detail["manifest_git_commit"] = commit
    if not commit or commit == "unknown":
        g.fail("manifest.git_commit is missing or 'unknown': the bundle cannot "
               "be tied to a source revision")

    state = git_state.inspect(ROOT)
    g.detail["git"] = state.summary()

    if not state.is_repo:
        g.fail(f"{state.error}: cannot verify the source revision")
        return g.finalize(), snapshot

    if commit and commit != "unknown" and not git_state.revision_matches(
            state.head, commit):
        g.fail(f"bundle was produced at {commit[:12]} but the working tree is "
               f"at {(state.head or '?')[:12]}")

    # Cleanliness is scoped to the paths that DEFINE the revision. A certifying
    # run necessarily rewrites tracked files under evidence/ (frozen packets,
    # determinism receipt, dry-run and smoke receipts), so an unscoped dirty
    # check would make VALID_RUN unreachable by construction.
    if state.source_changes:
        g.fail(f"source tree is dirty in {len(state.source_changes)} place(s) "
               f"(protocol abort condition D8_clean_release): "
               f"{[c[3:] for c in state.source_changes[:5]]}")
    if state.other_changes:
        g.fail(f"unclassified changes outside source and evidence paths: "
               f"{[c[3:] for c in state.other_changes[:5]]}")
    # Recorded, never fatal: this is the run's own output.
    g.detail["evidence_change_count"] = len(state.output_changes)

    return g.finalize(), snapshot


def gate_protocol(protocol_path: Path, manifest: dict,
                  sha_sidecar: Path | None) -> tuple[Gate, dict | None, str | None]:
    """Validate protocol semantics and bind its hash to the bundle."""
    g = Gate("protocol")
    try:
        protocol, sha, checks = load_and_validate_protocol(protocol_path)
    except ProtocolViolation as exc:
        g.fail(str(exc))
        return g.finalize(), None, None

    g.detail["protocol_sha256"] = sha
    g.detail["protocol_id"] = protocol.get("protocol_id")
    g.detail["semantic_checks"] = checks

    if sha_sidecar and sha_sidecar.is_file():
        declared = sha_sidecar.read_text().strip().split()[0]
        g.detail["declared_sha256"] = declared
        if declared != sha:
            g.fail(f"protocol sha mismatch: sidecar {sha_sidecar.name} declares "
                   f"{declared}, file hashes to {sha}")
    else:
        g.fail(f"protocol sha sidecar not found: {sha_sidecar}")

    bundle_sha = manifest.get("protocol_sha256")
    g.detail["manifest_protocol_sha256"] = bundle_sha
    if not bundle_sha:
        g.fail("manifest.protocol_sha256 is missing")
    elif bundle_sha != sha:
        g.fail(f"bundle was produced under protocol {bundle_sha[:16]} but "
               f"certification is running against {sha[:16]}")

    return g.finalize(), protocol, sha


def gate_environment(lock_path: Path, manifest: dict) -> tuple[Gate, dict | None]:
    """Verify the live environment against the lock and the bundle manifest."""
    g = Gate("environment_lock")
    try:
        lock = load_lock(lock_path)
    except AssertionError as exc:
        g.fail(str(exc))
        return g.finalize(), None

    ok, violations, observed = verify_environment(lock)
    g.detail["lock"] = lock
    g.detail["observed"] = observed
    for v in violations:
        g.fail(v)

    # The bundle records its own versions; they must agree with the lock too,
    # or the lock describes a different environment than the one that ran.
    for manifest_key, lock_key in (("torch_version", "torch"),
                                   ("transformers_version", "transformers")):
        recorded = manifest.get(manifest_key)
        locked = (lock.get("packages") or {}).get(lock_key)
        if recorded and locked and recorded != locked:
            g.fail(f"manifest.{manifest_key}={recorded} but lock pins "
                   f"{lock_key}=={locked}")
    if manifest.get("python_version") and lock.get("python") and \
            manifest["python_version"] != lock["python"]:
        g.fail(f"manifest.python_version={manifest['python_version']} but lock "
               f"pins python=={lock['python']}")

    return g.finalize(), lock


def gate_tests(run_tests: bool, tests_path: str) -> Gate:
    """Run the test suite. A skipped suite is a FAILED gate, not a pass."""
    g = Gate("test_suite")
    if not run_tests:
        g.detail["skipped"] = True
        g.fail("test suite was not run (--no-tests); protocol abort condition "
               "'test suite fails' cannot be cleared without running it")
        return g.finalize()

    proc = subprocess.run(
        [sys.executable, "-m", "pytest", tests_path, "-q", "--tb=line"],
        cwd=ROOT, capture_output=True, text=True, timeout=1800)
    tail = proc.stdout.strip().splitlines()[-8:]
    g.detail["exit_code"] = proc.returncode
    g.detail["summary"] = tail
    if proc.returncode != 0:
        g.fail(f"pytest exited {proc.returncode}: {tail[-1] if tail else 'no output'}")
    return g.finalize()


def gate_determinism(receipt_path: Path, task_count: int) -> Gate:
    """Require a determinism receipt that measured every claimed field."""
    g = Gate("determinism")
    if not receipt_path.is_file():
        g.fail(f"determinism receipt not found: {receipt_path}")
        return g.finalize()

    receipt = json.loads(receipt_path.read_text())
    g.detail["schema_version"] = receipt.get("schema_version")
    g.detail["result"] = receipt.get("result")
    g.detail["seeds"] = receipt.get("seeds")
    g.detail["tasks"] = receipt.get("tasks")
    g.detail["arms"] = receipt.get("arms")

    if receipt.get("schema_version") != "c4-determinism-qualification-v2":
        g.fail(f"determinism receipt schema is "
               f"{receipt.get('schema_version')!r}; v2 is required because "
               f"earlier versions claimed pool/order invariance they never "
               f"measured")
        return g.finalize()

    compared = set(receipt.get("compared_fields") or [])
    g.detail["compared_fields"] = sorted(compared)
    missing = REQUIRED_DETERMINISM_FIELDS - compared
    if missing:
        g.fail(f"determinism run did not compare {sorted(missing)}")
    if receipt.get("result") != "PASS":
        g.fail(f"determinism result is {receipt.get('result')!r}")
    if len(receipt.get("seeds") or []) < 2:
        g.fail("determinism run used fewer than two hash seeds")
    if (receipt.get("tasks") or 0) < task_count:
        g.fail(f"determinism covered {receipt.get('tasks')} tasks but the run "
               f"has {task_count}")
    if "C4_4" not in (receipt.get("arms") or []):
        g.fail("determinism run did not include C4_4, the primary arm")
    for arm, arm_result in (receipt.get("per_arm") or {}).items():
        for seed, diffs in (arm_result.get("diffs") or {}).items():
            nonzero = {k: v for k, v in diffs.items() if v}
            if nonzero:
                g.fail(f"{arm} seed {seed} differences: {nonzero}")
    return g.finalize()


def gate_arms_complete(arms: dict[str, list[dict]], task_count: int) -> Gate:
    """Every primary arm must have exactly task_count unique receipts."""
    g = Gate("arms_complete")
    g.detail["task_count"] = task_count
    counts: dict[str, int] = {}
    for arm_id in PRIMARY_ORDER:
        recs = arms.get(arm_id)
        if recs is None:
            g.fail(f"{arm_id}: receipts missing")
            continue
        counts[arm_id] = len(recs)
        ids = [r.get("task_id") for r in recs]
        if len(recs) != task_count:
            g.fail(f"{arm_id}: {len(recs)}/{task_count} receipts")
        if len(set(ids)) != len(ids):
            g.fail(f"{arm_id}: duplicate task_ids")
        if any(not t for t in ids):
            g.fail(f"{arm_id}: receipts with empty task_id")
    g.detail["per_arm_counts"] = counts
    return g.finalize()


def gate_no_oracle_leakage(arms: dict[str, list[dict]]) -> Gate:
    """No oracle key may appear anywhere in a runtime payload."""
    g = Gate("no_oracle_leakage")
    checked = 0
    for arm_id, recs in sorted(arms.items()):
        for r in recs:
            payload = r.get("runtime_payload")
            if payload is None:
                g.fail(f"{arm_id}/{r.get('task_id')}: runtime_payload missing")
                continue
            try:
                assert_runtime_clean(payload)
                checked += 1
            except AssertionError as exc:
                g.fail(f"{arm_id}/{r.get('task_id')}: {exc}")
    g.detail["payloads_checked"] = checked
    return g.finalize()


def gate_receipt_hash_fields(arms: dict[str, list[dict]]) -> Gate:
    """Receipts must carry the protocol's packet boundary hashes."""
    g = Gate("receipt_hash_fields")
    required = ("candidate_pool_hash", "membership_hash", "order_hash",
                "packet_hash", "prompt_hash")
    for arm_id, recs in sorted(arms.items()):
        for r in recs:
            p = _packet(r)
            missing = [k for k in required if not p.get(k)]
            if missing:
                g.fail(f"{arm_id}/{r.get('task_id')}: missing packet hashes "
                       f"{missing} — receipts predate protocol v2 hash fields")
                break  # one report per arm is enough
    g.detail["required_fields"] = list(required)
    return g.finalize()


def gate_prompt_binding(arms: dict[str, list[dict]]) -> Gate:
    """The model must have generated from the frozen packet prompt."""
    g = Gate("prompt_binding")
    mismatches = 0
    for arm_id, recs in sorted(arms.items()):
        for r in recs:
            packet_hash = _packet(r).get("prompt_hash")
            hrm_hash = r.get("runtime_payload", {}).get("hrm", {}).get("prompt_hash")
            if not packet_hash or not hrm_hash:
                g.fail(f"{arm_id}/{r.get('task_id')}: prompt hash missing "
                       f"(packet={bool(packet_hash)} hrm={bool(hrm_hash)})")
                break
            if packet_hash != hrm_hash:
                mismatches += 1
                if mismatches <= 3:
                    g.fail(f"{arm_id}/{r.get('task_id')}: HRM consumed a prompt "
                           f"that is not the frozen packet "
                           f"({hrm_hash[:12]} != {packet_hash[:12]})")
    if mismatches > 3:
        g.fail(f"...and {mismatches - 3} further prompt-binding mismatches")
    g.detail["mismatches"] = mismatches
    return g.finalize()


def gate_ordering_conformance(arms: dict[str, list[dict]]) -> Gate:
    """Ordering policy must be applied exactly where the protocol says."""
    g = Gate("ordering_conformance")
    observed: dict[str, dict[str, Any]] = {}
    for arm_id, recs in sorted(arms.items()):
        applied = {bool(_packet(r).get("ordering_applied")) for r in recs}
        policies = {_packet(r).get("ordering_policy_id") for r in recs}
        observed[arm_id] = {"ordering_applied": sorted(applied),
                            "ordering_policy_ids": sorted(p or "" for p in policies)}
        if len(applied) != 1:
            g.fail(f"{arm_id}: ordering_applied is inconsistent across tasks")
            continue
        should_apply = arm_id in DETERMINISTIC_ORDER_ARMS
        if applied.pop() != should_apply:
            g.fail(f"{arm_id}: ordering_applied should be {should_apply}")
        expected_policy = ORDERING_POLICY_ID if should_apply else "pool_order"
        if policies != {expected_policy}:
            g.fail(f"{arm_id}: ordering_policy_id should be "
                   f"{expected_policy!r}, got {sorted(policies)}")
    g.detail["per_arm"] = observed
    return g.finalize()


def gate_iterative_excluded(arms: dict[str, list[dict]]) -> Gate:
    """Iterative retrieval is OUTSIDE primary C4: no second pass may have run."""
    g = Gate("iterative_retrieval_excluded")
    offenders: dict[str, int] = {}
    for arm_id, recs in sorted(arms.items()):
        n = sum(1 for r in recs
                if r.get("runtime_payload", {}).get("query", {})
                .get("second_pass_performed"))
        if n:
            offenders[arm_id] = n
            g.fail(f"{arm_id}: {n} receipts performed a second retrieval pass")
    g.detail["second_pass_counts"] = offenders
    return g.finalize()


def gate_parity(arms: dict[str, list[dict]], analysis: dict) -> Gate:
    """Causal parity: arms downstream of retrieval must share the pool."""
    g = Gate("causal_parity")
    parity = analysis.get("parity") or {}
    g.detail["analysis_parity"] = parity
    if not parity.get("all_arms_same_tasks"):
        g.fail("analysis reports arms do not share a task set")

    # C4_3 -> C4_4 -> C4_5 differ only in the selector, so the candidate pool
    # must be byte-identical across them for the contrast to be causal.
    for a, b in (("C4_3", "C4_4"), ("C4_4", "C4_5")):
        if a not in arms or b not in arms:
            g.fail(f"cannot check pool parity {a} vs {b}: arm missing")
            continue
        by_a = {r["task_id"]: _packet(r).get("candidate_pool_hash") for r in arms[a]}
        by_b = {r["task_id"]: _packet(r).get("candidate_pool_hash") for r in arms[b]}
        common = sorted(set(by_a) & set(by_b))
        diff = [t for t in common if by_a[t] != by_b[t]]
        g.detail[f"pool_parity_{a}_{b}"] = {
            "compared": len(common), "differing": len(diff)}
        if diff:
            g.fail(f"{a} vs {b}: candidate pool differs on {len(diff)} tasks "
                   f"(e.g. {diff[:3]}) — the selector contrast is confounded")

    # C4_4 vs C4_4m must share membership and differ in order; that is the
    # entire point of the diagnostic pair.
    if "C4_4" in arms and "C4_4m" in arms:
        m4 = {r["task_id"]: _packet(r).get("membership_hash") for r in arms["C4_4"]}
        m4m = {r["task_id"]: _packet(r).get("membership_hash") for r in arms["C4_4m"]}
        o4 = {r["task_id"]: _packet(r).get("order_hash") for r in arms["C4_4"]}
        o4m = {r["task_id"]: _packet(r).get("order_hash") for r in arms["C4_4m"]}
        common = sorted(set(m4) & set(m4m))
        bad_member = [t for t in common if m4[t] != m4m[t]]
        same_order = [t for t in common if o4[t] == o4m[t]]
        g.detail["c4_4_vs_c4_4m"] = {
            "compared": len(common),
            "membership_differs": len(bad_member),
            "order_identical": len(same_order),
        }
        if bad_member:
            g.fail(f"C4_4 vs C4_4m membership differs on {len(bad_member)} tasks; "
                   f"the pair must isolate ordering only")
    return g.finalize()


def gate_metric_correctness(arms: dict[str, list[dict]], analysis: dict) -> Gate:
    """Recompute every quality score and both gap-capture metrics."""
    g = Gate("metric_correctness")
    bad = 0
    for arm_id, recs in sorted(arms.items()):
        for r in recs:
            ann = r.get("evaluator_annotation") or {}
            required = ann.get("required_evidence_ids")
            if required is None:
                g.fail(f"{arm_id}/{r.get('task_id')}: required_evidence_ids missing")
                break
            selected = r.get("runtime_payload", {}).get("selection", {}).get(
                "selected_ids", [])
            expected = compute_quality(
                correct=bool(ann.get("correct")),
                evidence_complete=evidence_complete(required, selected))
            stored = ann.get("quality")
            if stored is None or abs(float(stored) - expected) > 1e-12:
                bad += 1
                if bad <= 3:
                    g.fail(f"{arm_id}/{r.get('task_id')}: stored quality "
                           f"{stored} != recomputed {expected}")
            csr = ann.get("csr")
            expected_csr = 1.0 if set(required) <= set(selected) else 0.0
            if csr is not None and abs(float(csr) - expected_csr) > 1e-12:
                bad += 1
                if bad <= 3:
                    g.fail(f"{arm_id}/{r.get('task_id')}: stored csr {csr} != "
                           f"recomputed {expected_csr}")
    if bad > 3:
        g.fail(f"...and {bad - 3} further metric mismatches")
    g.detail["receipt_mismatches"] = bad

    qualities = {
        arm_id: sum(r["evaluator_annotation"]["quality"] for r in recs) / len(recs)
        for arm_id, recs in arms.items() if recs
    }
    g.detail["recomputed_arm_quality"] = qualities

    for name, fn in (("selector_gap_capture", selector_gap_capture),
                     ("oracle_gap_capture", oracle_gap_capture)):
        recomputed = fn(qualities)
        reported = analysis.get(name)
        g.detail[f"recomputed_{name}"] = recomputed
        g.detail[f"reported_{name}"] = reported
        if recomputed is None:
            g.fail(f"{name} could not be recomputed (missing arms)")
        elif reported is None:
            g.fail(f"analysis.json does not report {name}")
        elif abs(float(reported) - recomputed) > 1e-9:
            g.fail(f"{name}: analysis reports {reported}, recomputed {recomputed}")

    # Arm quality in analysis.json must match the receipts it was built from.
    for arm_id, summary in (analysis.get("arm_summary") or {}).items():
        if arm_id in qualities and \
                abs(float(summary.get("quality", -1)) - qualities[arm_id]) > 1e-9:
            g.fail(f"analysis arm_summary.{arm_id}.quality="
                   f"{summary.get('quality')} != recomputed {qualities[arm_id]}")
    return g.finalize()


def gate_derived_metric_agreement(arms: dict[str, list[dict]],
                                  analysis: dict) -> Gate:
    """Recompute every derived metric from raw receipts and compare.

    Protocol v2_1 ``certification.recompute_from_raw_receipts``. The rule is:
    never trust analysis.json merely because it is present. This gate caught a
    historical bundle reporting OGC=0.7947 where the receipts give 0.2263,
    because that analysis was written when the numerator used C4_5.

    Grouped CIs are recomputed with the analyzer's own bootstrap, which is
    seeded, so the comparison is exact rather than approximate.
    """
    g = Gate("derived_metric_agreement")
    g.detail["tolerance"] = METRIC_TOLERANCE

    analyzer = _load_analyzer()
    if analyzer is None:
        g.fail("could not import scripts/analyze_gate_c4.py to recompute metrics")
        return g.finalize()

    def compare(name: str, recomputed: Any, reported: Any) -> None:
        g.detail[name] = {"recomputed": recomputed, "reported": reported}
        if recomputed is None:
            g.fail(f"{name}: could not be recomputed from receipts")
            return
        if reported is None:
            g.fail(f"{name}: absent from analysis.json")
            return
        if abs(float(reported) - float(recomputed)) > METRIC_TOLERANCE:
            g.fail(f"{name}: analysis reports {reported}, receipts give "
                   f"{recomputed}")

    # 1. Arm counts and 2. task-set equality, recomputed over primary arms.
    primary = {a: recs for a, recs in arms.items() if a in PRIMARY_ORDER}
    counts = {a: len(recs) for a, recs in sorted(primary.items())}
    g.detail["recomputed_arm_counts"] = counts
    reported_counts = {a: s.get("n") for a, s in
                       (analysis.get("arm_summary") or {}).items()
                       if a in PRIMARY_ORDER}
    for arm_id, n in counts.items():
        if reported_counts.get(arm_id) != n:
            g.fail(f"arm_summary.{arm_id}.n={reported_counts.get(arm_id)} but "
                   f"{n} receipts are present")

    task_sets = {a: {r.get("task_id") for r in recs}
                 for a, recs in primary.items()}
    equal = len({frozenset(s) for s in task_sets.values()}) <= 1
    g.detail["recomputed_task_set_equality"] = equal
    reported_equal = (analysis.get("parity") or {}).get("all_arms_same_tasks")
    if equal is not bool(reported_equal):
        g.fail(f"parity.all_arms_same_tasks={reported_equal} but recomputation "
               f"over receipts gives {equal}")
    if not equal:
        g.fail("primary arms do not share a task set")

    # 3. Arm quality and 4. binary accuracy and 5. CSR, per arm.
    for arm_id, recs in sorted(primary.items()):
        if not recs:
            continue
        summary = (analysis.get("arm_summary") or {}).get(arm_id) or {}
        compare(f"arm_quality.{arm_id}",
                analyzer.arm_quality(recs), summary.get("quality"))
        compare(f"correct_rate.{arm_id}",
                analyzer.arm_correct_rate(recs), summary.get("correct_rate"))
        recomputed_csr = sum(
            r["evaluator_annotation"].get("csr", 0.0) for r in recs) / len(recs)
        reported_csr = (analysis.get("csr_stats") or {}).get(arm_id, {}).get("mean_csr")
        compare(f"mean_csr.{arm_id}", recomputed_csr, reported_csr)

    # 6. Primary delta with 7. family and 8. cluster CIs, all recomputed.
    if "C4_0" in primary and "C4_4" in primary:
        recomputed_delta = analyzer.paired_deltas(primary["C4_0"], primary["C4_4"])
        reported_delta = analysis.get("primary_delta") or {}
        for key in ("mean_delta", "ci_lower", "ci_upper", "ci_lower_cluster",
                    "ci_upper_cluster", "ci_lower_template", "ci_upper_template"):
            compare(f"primary_delta.{key}", recomputed_delta.get(key),
                    reported_delta.get(key))
        for flip, count in (recomputed_delta.get("flips") or {}).items():
            reported_flip = (reported_delta.get("flips") or {}).get(flip)
            if reported_flip != count:
                g.fail(f"primary_delta.flips.{flip}: analysis reports "
                       f"{reported_flip}, receipts give {count}")
    else:
        g.fail("cannot recompute primary delta: C4_0 or C4_4 missing")

    # 9. SGC and 10. OGC, from recomputed arm quality only.
    qualities = {a: analyzer.arm_quality(recs) for a, recs in primary.items() if recs}
    compare("selector_gap_capture", selector_gap_capture(qualities),
            analysis.get("selector_gap_capture"))
    compare("oracle_gap_capture", oracle_gap_capture(qualities),
            analysis.get("oracle_gap_capture"))

    # Diagnostic decomposition, when the arms are present. Reported for
    # interpretation; it is not part of the primary promotion decision.
    decomposition = analysis.get("ordering_membership_decomposition") or {}
    g.detail["decomposition_available"] = bool(decomposition.get("available"))
    if decomposition.get("available"):
        recomputed_dec = analyzer.ordering_membership_decomposition(arms)
        for key in ("ordering_effect", "membership_effect", "combined_effect",
                    "interaction_effect"):
            compare(f"decomposition.{key}", recomputed_dec.get(key),
                    decomposition.get(key))

    return g.finalize()


def gate_statistical(analysis: dict) -> Gate:
    """Protocol D4: predeclared threshold and grouped CI lower bounds.

    D5 (family regression) is gate_family_regression, below -- kept separate
    so a failed certificate distinguishes "the aggregate gain is too small or
    unstable" from "one family collapsed while the aggregate looked fine".
    This docstring previously claimed D5 too; the function never actually
    checked it.
    """
    g = Gate("statistical_gate")
    delta = analysis.get("primary_delta") or {}
    if not delta:
        g.fail("analysis.json has no primary_delta (C4_4 vs C4_0)")
        return g.finalize()

    raw_mean = delta.get("mean_delta")
    g.detail["mean_delta"] = raw_mean
    g.detail["threshold"] = PRIMARY_DELTA_THRESHOLD
    try:
        mean = require_finite_number(raw_mean, field="primary_delta.mean_delta")
    except MetricValidationError as exc:
        # Catches None, NaN, +/-Inf, bool, and non-numeric strings alike --
        # a NaN or Inf mean_delta must never silently satisfy or fail the
        # threshold comparison below by falling through Python's normal
        # (and here, misleading) NaN comparison semantics.
        g.fail(f"primary_delta.mean_delta: {exc}")
    else:
        if mean < PRIMARY_DELTA_THRESHOLD:
            g.fail(f"primary delta {mean:+.4f} is below the predeclared threshold "
                   f"{PRIMARY_DELTA_THRESHOLD:+.2f}")

    for label, key in (("family", "ci_lower"), ("cluster", "ci_lower_cluster"),
                       ("template", "ci_lower_template")):
        raw_lo = delta.get(key)
        g.detail[f"ci_lower_{label}"] = raw_lo
        try:
            lo = require_finite_number(raw_lo, field=f"primary_delta.{key}")
        except MetricValidationError as exc:
            g.fail(f"{label}-grouped CI lower bound: {exc}")
        else:
            if lo <= 0:
                g.fail(f"{label}-grouped CI lower bound {lo:+.4f} does not exclude 0")

    flips = delta.get("flips") or {}
    g.detail["flips"] = flips
    return g.finalize()


def gate_family_regression(arms: dict[str, list[dict]], analysis: dict) -> Gate:
    """D5: no task-structure family may regress worse than the frozen
    threshold. Frozen BEFORE any qualification run, per RESEARCH_STATUS.json.

    Recomputed independently from raw receipts using the same
    family_regression() function analyze_gate_c4.py uses to write
    analysis.json, then cross-checked against what analysis.json reports --
    the same "never trust analysis.json because it is present" discipline
    every other derived metric in this certifier already follows.

    A point-estimate check, not CI-based: development has ~12 tasks per
    family, too few for a per-family CI to mean much; qualification's larger
    per-family N is what makes the point estimate itself trustworthy there.
    """
    g = Gate("family_regression")

    analyzer = _load_analyzer()
    if analyzer is None:
        g.fail("could not import scripts/analyze_gate_c4.py to recompute "
               "per-family deltas")
        return g.finalize()

    threshold = analyzer.D5_FAMILY_REGRESSION_THRESHOLD
    g.detail["threshold"] = threshold

    if "C4_0" not in arms or "C4_4" not in arms:
        g.fail("C4_0 or C4_4 missing: cannot compute the primary delta this "
               "gate is scoped to")
        return g.finalize()

    recomputed = analyzer.family_regression(arms, threshold=threshold)
    g.detail["recomputed"] = recomputed

    reported = analysis.get("family_regression")
    g.detail["reported"] = reported
    if reported is None:
        g.fail("analysis.json does not report family_regression")
    else:
        for fam, r_detail in (recomputed or {}).get("per_family", {}).items():
            rep_detail = (reported.get("per_family") or {}).get(fam)
            if rep_detail is None:
                g.fail(f"analysis.json is missing family {fam!r}")
                continue
            try:
                rep_mean = require_finite_number(
                    rep_detail.get("mean_delta"), field=f"family_regression.{fam}.mean_delta")
                recomputed_mean = require_finite_number(
                    r_detail["mean_delta"], field=f"recomputed.{fam}.mean_delta")
            except MetricValidationError as exc:
                # A NaN/missing mean_delta must hard-fail this cross-check, not
                # silently pass it: abs(NaN - x) > 1e-9 is ALWAYS False in
                # Python (NaN comparisons never return True), so the original
                # `> 1e-9` guard let a NaN reported delta slip through as if it
                # matched the recomputed value.
                g.fail(f"family {fam}: {exc}")
                continue
            if abs(rep_mean - recomputed_mean) > 1e-9:
                g.fail(f"family {fam}: analysis reports delta "
                       f"{rep_detail.get('mean_delta')}, receipts give "
                       f"{r_detail['mean_delta']}")

    if recomputed and not recomputed["safe"]:
        for fam in recomputed["regressed_families"]:
            d = recomputed["per_family"][fam]
            g.fail(f"family {fam!r} regressed {d['mean_delta']:+.4f}, "
                   f"below the D5 threshold {threshold:+.2f} "
                   f"(n={d['n']} tasks)")

    return g.finalize()


def gate_results_hash(bundle: Path) -> Gate:
    """Every file in the bundle must match RESULTS.sha256."""
    g = Gate("results_hash")
    hash_file = bundle / "RESULTS.sha256"
    if not hash_file.is_file():
        g.fail(f"RESULTS.sha256 not found in {bundle}")
        return g.finalize()

    if not verify_results_hash(bundle):
        g.fail("RESULTS.sha256 does not match the bundle's current contents")

    per_file: dict[str, str] = {}
    for line in hash_file.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 2:
            g.fail(f"malformed RESULTS.sha256 line: {line!r}")
            continue
        digest, name = parts[0], parts[-1]
        target = bundle / name
        if not target.is_file():
            g.fail(f"{name}: listed in RESULTS.sha256 but missing")
            continue
        actual = _sha256_file(target)
        per_file[name] = "OK" if actual == digest else "MISMATCH"
        if actual != digest:
            g.fail(f"{name}: sha256 mismatch")
    g.detail["files"] = per_file
    return g.finalize()


def gate_development_lineage(manifest: dict, dev_certification_path: Path,
                             *, repo: Path) -> Gate:
    """For non-development splits: prove this run did not silently diverge
    from the development configuration that earned VALID_RUN.

    Thin wrapper around hrm_adaptive_memory.c4.development_lineage's
    check_development_lineage -- the same function
    scripts/colab_c4_requalify.py calls EARLY (before any GPU work) so a
    mismatch aborts before the run's cost is spent, not after it. This gate
    is the authoritative, non-bypassable version of that same check, run
    again at certification time regardless of what the launcher already did.

    'No development-side changes' is enforced at the granularity this project
    already uses to define a frozen scientific configuration: the protocol.
    Any change to the actual mechanism -- selection, ordering, retrieval,
    metrics -- requires a new protocol version by this project's own
    convention (v2 -> v2_1 is the precedent), and protocol_validation.py
    cross-checks the live code against the declared protocol on every run. So
    protocol_sha256 equality is both necessary and, together with that
    cross-check, a strong proxy for 'the mechanism has not changed'. The
    second hard requirement is that development's certified commit must be an
    ancestor of this run's commit (history was not rewritten or reset
    backward). A full file-level diff against development's
    SOURCE_SNAPSHOT.json is always attached as detail -- visible to any
    reviewer -- but does not itself fail the gate: tooling and
    certification-infrastructure files are expected to keep evolving between
    splits, and hiding that diff would be worse than reporting it while
    trusting the two hard checks above to catch what actually matters.
    """
    g = Gate("development_lineage")
    g.detail["dev_certification_path"] = str(dev_certification_path)

    result = check_development_lineage(
        dev_certification_path=dev_certification_path,
        this_protocol_sha256=manifest.get("protocol_sha256"),
        repo=repo,
    )
    g.detail.update(result.summary())
    for v in result.violations:
        g.fail(v)
    return g.finalize()


# --- Orchestration ----------------------------------------------------------

def certify(bundle: Path, protocol_path: Path, lock_path: Path,
            determinism_receipt: Path, run_tests: bool, tests_path: str,
            expect_source_sha: str | None,
            dev_certification_path: Path | None = None) -> dict:
    """Evaluate every gate and assemble the certificate."""
    gates: list[Gate] = []
    prelude: list[str] = []

    manifest_path = bundle / "manifest.json"
    analysis_path = bundle / "analysis.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    analysis = json.loads(analysis_path.read_text()) if analysis_path.is_file() else {}
    if not manifest:
        prelude.append(f"manifest.json missing or empty in {bundle}")
    if not analysis:
        prelude.append(f"analysis.json missing or empty in {bundle}")

    arms: dict[str, list[dict]] = {}
    for p in sorted(bundle.glob("C4_*.jsonl")):
        arms[p.stem] = _load_jsonl(p)

    task_count = manifest.get("task_count") or (
        max((len(v) for v in arms.values()), default=0))

    src_gate, snapshot = gate_source_lineage(bundle, manifest, expect_source_sha)
    gates.append(src_gate)
    proto_gate, _protocol, protocol_sha = gate_protocol(
        protocol_path, manifest, protocol_path.with_suffix(".sha256"))
    gates.append(proto_gate)
    env_gate, lock = gate_environment(lock_path, manifest)
    gates.append(env_gate)
    gates.append(gate_tests(run_tests, tests_path))
    gates.append(gate_determinism(determinism_receipt, task_count))
    gates.append(gate_arms_complete(arms, task_count))
    gates.append(gate_no_oracle_leakage(arms))
    gates.append(gate_receipt_hash_fields(arms))
    gates.append(gate_prompt_binding(arms))
    gates.append(gate_ordering_conformance(arms))
    gates.append(gate_iterative_excluded(arms))
    gates.append(gate_parity(arms, analysis))
    gates.append(gate_metric_correctness(arms, analysis))
    gates.append(gate_derived_metric_agreement(arms, analysis))
    gates.append(gate_statistical(analysis))
    gates.append(gate_family_regression(arms, analysis))
    gates.append(gate_results_hash(bundle))

    # Only non-development splits are required to prove lineage back to a
    # certified development configuration. Development has nothing prior to
    # compare itself against, and this must never add a 17th gate to the
    # already-certified development bundle's own history.
    if manifest.get("split") and manifest.get("split") != "development":
        # bundle is e.g. evidence/gate_c4/full/qualification; the development
        # bundle is the SIBLING evidence/gate_c4/full/development, so only
        # ONE .parent step up from bundle, not two.
        default_dev_cert = (bundle.parent / "development" /
                            "certification" / "CERTIFICATION.json")
        gates.append(gate_development_lineage(
            manifest, dev_certification_path or default_dev_cert, repo=ROOT))

    gate_map = {g.name: g.to_dict() for g in gates}
    valid_run = bool(gates) and not prelude and all(g.passed for g in gates)

    certificate = {
        "schema_version": "c4-certification-v1",
        "VALID_RUN": valid_run,
        "verdict": ("VALID_CONFORMANT_C4_V2_RESULT" if valid_run
                    else "NOT_CERTIFIED"),
        "bundle": str(bundle.relative_to(ROOT) if bundle.is_relative_to(ROOT) else bundle),
        "split": manifest.get("split"),
        "task_count": task_count,
        "arms_present": sorted(arms),
        "protocol_sha256": protocol_sha,
        "source_tree_sha256": snapshot.get("source_tree_sha256"),
        "prelude_errors": prelude,
        "gates_total": len(gates),
        "gates_passed": sum(1 for g in gates if g.passed),
        "gates_failed": sorted(g.name for g in gates if not g.passed),
        "gates": gate_map,
        # Reported for context only. These never influence VALID_RUN: a good
        # number is not evidence that the run was reproducible.
        "performance_summary": {
            "arm_quality": {
                arm: sum(r["evaluator_annotation"]["quality"] for r in recs) / len(recs)
                for arm, recs in sorted(arms.items()) if recs
            },
            "selector_gap_capture": analysis.get("selector_gap_capture"),
            "oracle_gap_capture": analysis.get("oracle_gap_capture"),
            "ordering_membership_decomposition":
                analysis.get("ordering_membership_decomposition"),
        },
    }
    return certificate, snapshot, lock


def write_certification_artifacts(bundle: Path, certificate: dict,
                                  snapshot: dict, lock: dict | None,
                                  protocol_path: Path) -> Path:
    """Write lineage artifacts into <bundle>/certification/.

    A subdirectory keeps the bundle's RESULTS.sha256 (computed over files
    directly in the bundle) valid, so certification never has to rewrite the
    hash file it just verified.
    """
    out = bundle / "certification"
    out.mkdir(parents=True, exist_ok=True)

    (out / "SOURCE_SNAPSHOT.json").write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
    if protocol_path.is_file():
        (out / "PROTOCOL.json").write_bytes(protocol_path.read_bytes())
    if lock is not None:
        (out / "ENVIRONMENT.lock").write_text(
            json.dumps(lock, indent=2, sort_keys=True) + "\n")

    # Zip the source tree named in the snapshot so the bundle is self-contained.
    snapshot_zip = out / "source_snapshot.zip"
    with zipfile.ZipFile(snapshot_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in sorted(snapshot.get("files", {})):
            src = ROOT / rel
            if src.is_file():
                zf.write(src, rel)

    certificate["certification_artifacts"] = {
        name: _sha256_file(out / name)
        for name in sorted(p.name for p in out.iterdir()
                           if p.is_file() and p.name != "CERTIFICATION.json")
    }
    cert_path = out / "CERTIFICATION.json"
    cert_path.write_text(json.dumps(certificate, indent=2, sort_keys=True) + "\n")
    return cert_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Certify a Gate C4 v2 result bundle (fail-closed)")
    parser.add_argument("--bundle", type=Path,
                        default=ROOT / "evidence/gate_c4/full/development")
    parser.add_argument("--protocol", type=Path,
                        default=ROOT / "configs/gate_c4_protocol_v2_1.json")
    parser.add_argument("--lock", type=Path,
                        default=ROOT / "configs/c4_requirements.lock")
    parser.add_argument("--determinism-receipt", type=Path,
                        default=ROOT / "evidence/gate_c4/determinism_qualification.json")
    parser.add_argument("--tests-path", default="tests")
    parser.add_argument("--no-tests", action="store_true",
                        help="Do not run the suite. The test gate then FAILS.")
    parser.add_argument("--expect-source-sha", default=None)
    parser.add_argument("--dev-certification", type=Path, default=None,
                        help="Development CERTIFICATION.json to prove lineage "
                             "against. Only used when this bundle's split is "
                             "not 'development'. Defaults to "
                             "<bundle>/../../development/certification/"
                             "CERTIFICATION.json.")
    args = parser.parse_args()

    if not args.bundle.is_dir():
        print(f"FATAL: bundle directory not found: {args.bundle}")
        return 2

    print("=" * 70)
    print("  GATE C4 v2 CERTIFICATION")
    print("=" * 70)
    print(f"  bundle:   {args.bundle}")
    print(f"  protocol: {args.protocol}")
    print(f"  lock:     {args.lock}")
    print()

    certificate, snapshot, lock = certify(
        bundle=args.bundle,
        protocol_path=args.protocol,
        lock_path=args.lock,
        determinism_receipt=args.determinism_receipt,
        run_tests=not args.no_tests,
        tests_path=args.tests_path,
        expect_source_sha=args.expect_source_sha,
        dev_certification_path=args.dev_certification,
    )

    for name, result in certificate["gates"].items():
        status = "PASS" if result["passed"] else "FAIL"
        print(f"  [{status}] {name}")
        for v in result["violations"][:6]:
            print(f"         - {v}")
        if len(result["violations"]) > 6:
            print(f"         ... {len(result['violations']) - 6} more")

    for err in certificate["prelude_errors"]:
        print(f"  [FAIL] bundle: {err}")

    cert_path = write_certification_artifacts(
        args.bundle, certificate, snapshot, lock, args.protocol)

    # Root hash written LAST, after CERTIFICATION.json and every other
    # artifact exist, so it is the one file that covers the whole bundle:
    # raw receipts, analysis, manifest, and everything under certification/
    # (including the certificate itself). RESULTS.sha256 stays narrow and
    # keeps verifying just the raw results, exactly as before; this is the
    # broader root of trust for the bundle as a whole.
    from hrm_adaptive_memory.c4.provenance import write_bundle_hash
    write_bundle_hash(args.bundle)

    print()
    print(f"  gates passed: {certificate['gates_passed']}/{certificate['gates_total']}")
    print(f"  VALID_RUN:    {certificate['VALID_RUN']}")
    print(f"  verdict:      {certificate['verdict']}")
    print(f"  certificate:  {cert_path}")
    print(f"  bundle hash:  {args.bundle / 'BUNDLE.sha256'} "
          f"(covers everything, including certification/)")
    if not certificate["VALID_RUN"]:
        print()
        print("  NOT CERTIFIED. Do not name the bundle "
              "VALID_CONFORMANT_C4_V2_*.zip.")
    return 0 if certificate["VALID_RUN"] else 1


if __name__ == "__main__":
    sys.exit(main())
