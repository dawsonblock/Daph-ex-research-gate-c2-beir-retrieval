#!/usr/bin/env python3
"""Gate C4 runner — integrated non-oracle memory pipeline.

Orchestrates the stage-composition harness. Each arm is a configuration over
shared stages, not a separate code path.

Modes:
    dry-run: CPU-only pre-HRM stages for all tasks, validate parity + leakage.
    smoke:   5 tasks × 7 arms with HRM (model load check).
    full:    All tasks × all arms with HRM, resumable.

Protocol hash is embedded in every manifest to prevent config drift.
"""
from __future__ import annotations

import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path
from dataclasses import asdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hrm_adaptive_memory.contracts import IndexRecord
from hrm_adaptive_memory.c4.contracts import *
from hrm_adaptive_memory.c4.arms import ARMS, DIAGNOSTIC_ORDER, PRIMARY_ORDER
from hrm_adaptive_memory.c4.query_stage import run_query_stage, extract_subject
from hrm_adaptive_memory.retrieval.information_state import (
    InformationState, formulate_followup, FOLLOWUP_FORMULATION)
from hrm_adaptive_memory.c4.retrieval_stage import run_retrieval_stage, clear_backend_cache
from hrm_adaptive_memory.c4.identity_stage import run_identity_stage
from hrm_adaptive_memory.c4.selection_stage import run_selection_stage
from hrm_adaptive_memory.c4.packet_stage import run_packet_stage
from hrm_adaptive_memory.c4.bridge_extraction import extract_bridge
from hrm_adaptive_memory.c4.receipts import build_pre_hrm_receipt, assert_runtime_clean, build_full_receipt
from hrm_adaptive_memory.c4.parity import (
    validate_all_parity, validate_no_leakage,
    validate_selected_in_pool, validate_packet_budgets,
    validate_q3_query_formulation, validate_merge_provenance,
    validate_causal_parity, validate_all_conformance)
from hrm_adaptive_memory.evaluation.verifiers import verify_answer as _verify_answer_shared
from hrm_adaptive_memory.c4.provenance import (
    build_manifest, write_results_hash, sha256_corpus)

CORPUS = ROOT / "data/hrm/controlled_gate_a_v4"
OUT = ROOT / "evidence/gate_c4"
HRM_MAX_NEW_TOKENS = 64
# Prompts sharing one HRM forward pass in run_full(). Must stay 1 by default:
# tests/unit/test_batched_generation.py's real-checkpoint equivalence check
# (HRM_EQUIVALENCE_TEST=1) currently FAILS for sapientinc/HRM-Text-1B at both
# float16 and bfloat16 -- on a Lightning A100, batching produced a genuinely
# different completion (not just a trailing-token artifact) for the shortest
# prompt in a mixed-length batch. Root cause not yet found; suspected
# interaction between left-padding and this model's recurrent H/L-cycle
# architecture, which is not a standard decoder-only transformer. Do not
# raise this default until that equivalence test passes.
HRM_BATCH_SIZE = int(os.environ.get("HRM_BATCH_SIZE", "1"))


ACTIVE_PROTOCOL = "v2_1"

# Only v2_1 is certifiable. v2 specified packet ordering twice, inconsistently,
# and v1 predates the determinism repair. Both remain readable for lineage.
PROTOCOL_FILES = {
    "v2_1": "configs/gate_c4_protocol_v2_1.json",
    "v2": "configs/gate_c4_protocol_v2.json",
    "v1": "configs/gate_c4_protocol.json",
}


def _protocol_path() -> Path:
    """Path to the active C4 protocol file.

    Selected by the C4_PROTOCOL environment variable; defaults to the active
    protocol rather than to v1, so a bundle produced without setting the
    variable is not silently stamped with a superseded protocol hash.
    """
    version = os.environ.get("C4_PROTOCOL", ACTIVE_PROTOCOL)
    if version not in PROTOCOL_FILES:
        raise ValueError(
            f"C4_PROTOCOL={version!r} is not a known protocol. "
            f"Known: {sorted(PROTOCOL_FILES)}")
    return ROOT / PROTOCOL_FILES[version]


def _protocol_hash() -> str:
    """Hash of the active C4 protocol file."""
    return hashlib.sha256(_protocol_path().read_bytes()).hexdigest()


def _detect_device() -> str:
    """Detect torch device without importing at module level."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _load_split(split: str) -> tuple[list[dict], list[dict], dict[str, str]]:
    tasks = [json.loads(l) for l in (CORPUS / split / "oracle_tasks.jsonl").read_text().splitlines() if l.strip()]
    evidence = [json.loads(l) for l in (CORPUS / split / "evidence.jsonl").read_text().splitlines() if l.strip()]
    texts = {r["evidence_id"]: r["content"] for r in evidence}
    return tasks, evidence, texts


def _to_index_records(evidence: list[dict]) -> list[IndexRecord]:
    return [IndexRecord(
        evidence_id=r["evidence_id"], source_id=r["source_id"],
        content=r["content"], token_count=max(1, len(r["content"].split())),
        source_type=r["source_type"], metadata=r["metadata"],
    ) for r in evidence]


def _state_to_dict(state) -> dict:
    return {
        "subject": state.subject,
        "target_relation": state.target_relation,
        "bridge": state.bridge,
        "canonical_subject": state.canonical_subject,
        "resolved_identities": list(state.resolved_identities),
        "hop": state.hop,
    }


def _verify_answer(task: dict, output: str) -> tuple[float, bool]:
    """Verify HRM output against the task answer (evaluator-side only).

    Delegates to the shared verifier in ``hrm_adaptive_memory.evaluation.verifiers``
    so that C4 semantics are identical to the historical V4 verifier, including
    stripping of HRM control tokens (e.g. ``<|box_end|>``).
    """
    verifier = task.get("verifier", "exact")
    answer = task["answer"]
    return _verify_answer_shared(verifier, answer, output)


def _compute_quality(task: dict, selected_ids: list[str],
                     evidence: list[dict], correct: bool) -> float:
    """Compute quality score via the shared metrics module."""
    from hrm_adaptive_memory.c4.metrics import compute_quality, evidence_complete
    complete = evidence_complete(task["required_evidence_ids"], selected_ids)
    return compute_quality(correct=correct, evidence_complete=complete)


def _compute_csr(task: dict, selected_ids: list[str]) -> float:
    """Compute Complete Set Retention (evaluator-side)."""
    required = set(task["required_evidence_ids"])
    selected = set(selected_ids)
    return 1.0 if required <= selected else 0.0


def _compute_role_retention(task: dict, selected_ids: list[str],
                             evidence: list[dict]) -> dict:
    """Compute role retention metrics (evaluator-side)."""
    ev_by_id = {r["evidence_id"]: r for r in evidence}
    selected = set(selected_ids)
    meta = task["_oracle_metadata"]
    edges = meta["proof_edges"]
    roles = {"answer": 0, "bridge": 0, "identity": 0}
    role_eligible = {"answer": 0, "bridge": 0, "identity": 0}

    for edge in edges:
        rid = edge["record_id"]
        if rid not in task["required_evidence_ids"]:
            continue
        kind = ev_by_id.get(rid, {}).get("metadata", {}).get("record_kind", "")
        if "answer" in kind or kind in ("required", "required_current", "direct_answer"):
            role_eligible["answer"] += 1
            if rid in selected:
                roles["answer"] += 1
        elif "identity" in kind:
            role_eligible["identity"] += 1
            if rid in selected:
                roles["identity"] += 1
        else:
            role_eligible["bridge"] += 1
            if rid in selected:
                roles["bridge"] += 1

    return {
        "answer_retention": roles["answer"] / role_eligible["answer"] if role_eligible["answer"] else 1.0,
        "bridge_retention": roles["bridge"] / role_eligible["bridge"] if role_eligible["bridge"] else 1.0,
        "identity_retention": roles["identity"] / role_eligible["identity"] if role_eligible["identity"] else 1.0,
    }


def _merge_retrieval(first: RetrievalResult, second: RetrievalResult,
                     arm: C4Arm) -> RetrievalResult:
    """Merge two retrieval results by interleaving candidates (RRF-style)."""
    # Combine rankings with RRF
    all_bm25 = list(first.bm25_ranked) + list(second.bm25_ranked)
    all_bge = list(first.bge_ranked) + list(second.bge_ranked)

    # RRF merge of candidate IDs
    k = first.rrf_k
    scores: dict[str, float] = {}
    for ranking in [list(first.bm25_ranked), list(second.bm25_ranked)]:
        for i, (eid, _) in enumerate(ranking, 1):
            scores[eid] = scores.get(eid, 0.0) + 1.0 / (k + i)
    if first.bge_ranked or second.bge_ranked:
        for ranking in [list(first.bge_ranked), list(second.bge_ranked)]:
            for i, (eid, _) in enumerate(ranking, 1):
                scores[eid] = scores.get(eid, 0.0) + 1.0 / (k + i)

    merged_ranked = tuple(sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:first.candidate_budget])
    candidate_ids = tuple(eid for eid, _ in merged_ranked)

    return RetrievalResult(
        bm25_ranked=tuple(all_bm25),
        bge_ranked=tuple(all_bge),
        fusion_ranked=merged_ranked,
        candidate_ids=candidate_ids,
        candidate_budget=first.candidate_budget,
        retrieval_policy=first.retrieval_policy,
        bm25_backend=first.bm25_backend,
        bge_model_id=first.bge_model_id,
        bge_revision=first.bge_revision,
        rrf_k=first.rrf_k,
    )


def run_pre_hrm_stages(task: dict, arm: C4Arm, records: list[IndexRecord],
                       texts: dict[str, str]) -> PreHRMResult:
    """Run all pre-HRM stages for one task under one arm.

    For subject_preserving query policy (C4-1+), this implements the qualified
    two-pass iterative retrieval:

        question → first retrieval → discover bridge → subject + bridge + relation
        → second retrieval → merge candidates

    For original query policy (C4-0), this is a single-pass retrieval.

    NOTE: Iterative retrieval is DISABLED based on the C4-BRIDGE gate negative
    result (no runtime bridge mechanism beats the one-pass baseline). The
    second pass introduces distractors that displace required evidence in the
    bounded candidate pool. See evidence/gate_c4/bridge/metrics.json and
    RESEARCH_STATUS.json gate_c4_bridge_runtime_acquisition.
    The bridge extraction code is retained for provenance and future
    re-evaluation but the second pass is not performed.
    """
    question = task["question"]
    split = task.get("split", "development")

    # Query stage (first pass)
    state_before, query_result = run_query_stage(question, arm)

    # Retrieval stage (first pass)
    retrieval_result = run_retrieval_stage(query_result.rendered_query, arm, records)

    # Iterative retrieval DISABLED (C4-BRIDGE negative result).
    # Bridge is still extracted for provenance/diagnostics but no second
    # retrieval pass is performed.
    bridge = None
    second_query = None
    second_pass_performed = False
    state_after_query = state_before

    if arm.query_policy == "subject_preserving":
        subject = extract_subject(question)
        if subject:
            bridge = extract_bridge(
                subject, question, retrieval_result.candidate_ids, texts)
            # Second pass disabled — see C4-BRIDGE gate result
            # if bridge:
            #     state_after_query = state_before.with_bridge(bridge)
            #     second_query = formulate_followup(
            #         state_after_query, formulation=FOLLOWUP_FORMULATION)
            #     second_pass_performed = True
            #     second_retrieval = run_retrieval_stage(second_query, arm, records)
            #     retrieval_result = _merge_retrieval(retrieval_result, second_retrieval, arm)

    # Update query result with iterative info
    query_result = QueryResult(
        original_question=query_result.original_question,
        rendered_query=query_result.rendered_query,
        query_hash=query_result.query_hash,
        query_policy=query_result.query_policy,
        query_policy_version=query_result.query_policy_version,
        bridge=bridge,
        second_query=second_query,
        second_pass_performed=second_pass_performed,
    )

    # Identity stage
    identity_result = run_identity_stage(question, arm, retrieval_result, texts)

    # If identity resolved, update state
    state_after = state_after_query
    canonical_question = None
    if identity_result.status == "RESOLVED" and identity_result.canonical:
        state_after = state_after_query.with_identity(
            identity_result.surface or state_after_query.subject,
            identity_result.canonical)
        canonical_question = question.replace(
            identity_result.surface or "",
            identity_result.canonical)

    # Selection stage
    selection_result = run_selection_stage(
        arm, question, retrieval_result, identity_result, texts,
        task.get("required_evidence_ids", []),
        canonical_question=canonical_question)

    # Packet stage. `retrieval_result` supplies the candidate pool hash and the
    # fusion scores used by the ordering policy's tie-break rules.
    prompt, packet_result = run_packet_stage(
        arm, question, selection_result, texts, retrieval=retrieval_result)

    return PreHRMResult(
        task_id=task["task_id"],
        arm_id=arm.arm_id,
        split=split,
        query=query_result,
        retrieval=retrieval_result,
        identity=identity_result,
        selection=selection_result,
        packet=packet_result,
        information_state_before=_state_to_dict(state_before),
        information_state_after=_state_to_dict(state_after),
        prompt=prompt,
    )


def _load_hrm():
    """Load the HRM model (lazy, only when needed).

    Automatically uses GPU (CUDA) when available with fp16 for speed.
    Set HRM_DEVICE=cuda and HRM_DTYPE=float16 via environment to override.
    """
    from hrm_adaptive_memory.hrm.model import HRMAdapter, HRMModelSpec, PromptCondition
    import torch

    device_map = None
    dtype = None

    if torch.cuda.is_available():
        device_map = "auto"
        # Use fp16 on GPU for ~2x speedup
        dtype = torch.float16
        print(f"  [HRM] Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"  [HRM] dtype: {dtype}")
    else:
        print("  [HRM] Using CPU (no CUDA detected)")

    adapter = HRMAdapter.from_pretrained(
        spec=HRMModelSpec(), dtype=dtype, device_map=device_map)
    return adapter, PromptCondition.DIRECT


def _merged_arm_manifest_fields(out_dir: Path, arm_ids: list[str]) -> dict:
    """Merge this invocation's arm_ids into any manifest.json already present.

    run_full() is called twice per requalify run -- once with PRIMARY_ORDER,
    once with the diagnostic arms -- and both write manifest.json to the same
    out_dir. Without merging, the second call's manifest.json overwrote the
    first entirely, leaving arm_ids=["C4_3o","C4_4m"] as the on-disk record
    even though the bundle directory holds receipts for all nine arms.
    certify_c4_run.py never depended on this field (it reconstructs arm
    completeness from the actual C4_*.jsonl files present), so the bug never
    affected a certification verdict -- but the manifest is supposed to be a
    readable record of what's in the bundle, and it wasn't one.

    Returns the union of arm_ids (this call's plus any prior), split into
    primary_arm_ids and diagnostic_arm_ids by membership in the canonical
    orderings, plus the union itself under the pre-existing "arm_ids" key for
    backward compatibility with anything that already reads it.
    """
    existing_path = out_dir / "manifest.json"
    prior_arm_ids: list[str] = []
    if existing_path.is_file():
        try:
            prior = json.loads(existing_path.read_text())
            prior_arm_ids = list(prior.get("arm_ids") or [])
            # Absorb old-schema split fields too, in case of a future format
            # change; union is idempotent either way.
            prior_arm_ids += list(prior.get("primary_arm_ids") or [])
            prior_arm_ids += list(prior.get("diagnostic_arm_ids") or [])
        except (json.JSONDecodeError, OSError):
            pass

    union = set(prior_arm_ids) | set(arm_ids)
    primary = [a for a in PRIMARY_ORDER if a in union]
    diagnostic = [a for a in DIAGNOSTIC_ORDER if a in union]
    other = sorted(union - set(primary) - set(diagnostic))
    return {
        "arm_ids": primary + diagnostic + other,
        "primary_arm_ids": primary,
        "diagnostic_arm_ids": diagnostic,
    }


def _run_hrm(adapter, condition, prompt: str) -> HRMResult:
    """Run HRM generation for one prompt."""
    import time as _time
    t0 = _time.perf_counter()
    result = adapter.generate(
        prompt, condition=condition, max_new_tokens=HRM_MAX_NEW_TOKENS)
    latency = _time.perf_counter() - t0
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
    return HRMResult(
        output=result["text"],
        prompt_hash=prompt_hash,
        prompt_tokens=result["prompt_tokens"],
        completion_tokens=result["completion_tokens"],
        model_id=adapter.spec.model_id,
        model_revision=adapter.spec.revision,
        latency_seconds=latency,
    )


def _run_hrm_batch(adapter, condition, prompts: list[str]) -> list[HRMResult]:
    """Run HRM generation for several prompts in one forward pass.

    Only usable where tests/unit/test_batched_generation.py's
    test_batched_matches_sequential_on_the_real_checkpoint passes: batched and
    sequential generation are not assumed equivalent, they are verified
    equivalent (greedy, byte-identical) before this is wired into a real run.
    latency_seconds is the shared batch latency divided evenly across its
    members -- it is receipt metadata only, never read by any gate.
    """
    import time as _time
    if not prompts:
        return []
    t0 = _time.perf_counter()
    rows = adapter.generate_batch(
        prompts, condition=condition, max_new_tokens=HRM_MAX_NEW_TOKENS)
    per_task_latency = (_time.perf_counter() - t0) / len(prompts)
    results = []
    for prompt, row in zip(prompts, rows):
        results.append(HRMResult(
            output=row["text"],
            prompt_hash=hashlib.sha256(prompt.encode()).hexdigest(),
            prompt_tokens=row["prompt_tokens"],
            completion_tokens=row["completion_tokens"],
            model_id=adapter.spec.model_id,
            model_revision=adapter.spec.revision,
            latency_seconds=per_task_latency,
        ))
    return results


def _assert_prompt_binding(pre_hrm: PreHRMResult, hrm: HRMResult) -> None:
    """Fail closed if HRM did not consume the frozen packet prompt.

    ``PacketResult.prompt_hash`` is computed pre-HRM from the ordered packet.
    ``HRMResult.prompt_hash`` is computed from whatever text was actually
    generated on. If they diverge, the ordering policy was bypassed and the
    receipt would misdescribe the run.
    """
    if pre_hrm.packet.prompt_hash != hrm.prompt_hash:
        raise AssertionError(
            f"Prompt binding violated for {pre_hrm.task_id}/{pre_hrm.arm_id}: "
            f"packet prompt_hash={pre_hrm.packet.prompt_hash[:16]} but HRM "
            f"consumed prompt_hash={hrm.prompt_hash[:16]}. HRM must generate "
            f"from PreHRMResult.prompt.")


# Bumped whenever a change alters the prompt HRM sees for an unchanged packet
# hash. Without this, resumption silently reuses receipts generated under the
# previous prompt-construction path.
#   v2: packet ordering + fusion tie-breaks actually applied to the prompt
PIPELINE_VERSION = "c4_pipeline_v2_ordered_packet"


def _resume_key(task: dict, arm: C4Arm, packet_hash: str) -> str:
    """Build a resume key that invalidates on config/code changes."""
    return hashlib.sha256(
        f"{task['task_id']}|{arm.arm_id}|{packet_hash}|{arm.query_policy}|"
        f"{arm.retrieval_policy}|{arm.identity_policy}|{arm.selector_policy}|"
        f"{PIPELINE_VERSION}".encode()
    ).hexdigest()[:16]


def _select_smoke_tasks(tasks: list[dict]) -> list[dict]:
    """Select 5 tasks covering canonical, abbreviation, alias, description, multi-hop."""
    by_regime: dict[str, list[dict]] = {}
    for t in tasks:
        regime = t["metadata"]["entity_regime"]
        by_regime.setdefault(regime, []).append(t)

    selected = []
    for regime in ("canonical", "abbreviation", "alias", "description"):
        if regime in by_regime and by_regime[regime]:
            selected.append(by_regime[regime][0])
    # Add a multi-hop task if available
    multi_hop = [t for t in tasks if t["metadata"].get("opportunity_group") == "B_SECOND_PASS_REQUIRED"]
    if multi_hop:
        selected.append(multi_hop[0])
    elif len(selected) < 5 and tasks:
        selected.append(tasks[-1])
    return selected[:5]


def run_smoke(split: str = "development"):
    """5-task × 7-arm smoke test with HRM."""
    tasks_all, evidence, texts = _load_split(split)
    tasks = _select_smoke_tasks(tasks_all)
    records = _to_index_records(evidence)
    arm_ids = PRIMARY_ORDER  # all 7 arms
    print(f"Smoke test: {len(tasks)} tasks × {len(arm_ids)} arms on {split}")
    print(f"Task regimes: {[t['metadata']['entity_regime'] for t in tasks]}")

    # Run pre-HRM stages first (with cached backends)
    print("\n--- Pre-HRM stages ---")
    all_pre_hrm: dict[str, list[PreHRMResult]] = {}
    for arm_id in arm_ids:
        arm = ARMS[arm_id]
        print(f"  {arm_id}...", end=" ", flush=True)
        t0 = time.time()
        results = []
        for task in tasks:
            r = run_pre_hrm_stages(task, arm, records, texts)
            results.append(r)
        all_pre_hrm[arm_id] = results
        print(f"{time.time()-t0:.1f}s")

    # Quick parity check
    ok, violations = validate_all_parity(all_pre_hrm)
    print(f"\n  Parity: {'PASS' if ok else 'FAIL'} ({len(violations)} violations)")
    for v in violations[:5]: print(f"    {v}")

    # Load HRM model
    print("\n--- Loading HRM model ---")
    t0 = time.time()
    adapter, condition = _load_hrm()
    print(f"  HRM loaded in {time.time()-t0:.1f}s ({adapter.spec.model_id})")

    # Run HRM generation
    print("\n--- HRM generation ---")
    out_dir = OUT / "smoke" / split
    out_dir.mkdir(parents=True, exist_ok=True)

    all_receipts: dict[str, list] = {}
    for arm_id in arm_ids:
        arm = ARMS[arm_id]
        print(f"  {arm_id}...", end=" ", flush=True)
        t0 = time.time()
        receipts = []
        for i, task in enumerate(tasks):
            pre_hrm = all_pre_hrm[arm_id][i]
            pre_receipt = build_pre_hrm_receipt(pre_hrm)
            # HRM consumes the frozen packet prompt — never a recomposition
            # from selected_ids, which would bypass the ordering policy.
            hrm_result = _run_hrm(adapter, condition, pre_hrm.prompt)
            _assert_prompt_binding(pre_hrm, hrm_result)
            _binary_score, correct = _verify_answer(task, hrm_result.output)
            quality_score = _compute_quality(
                task, list(pre_hrm.selection.selected_ids), evidence, correct)
            csr = _compute_csr(task, list(pre_hrm.selection.selected_ids))
            roles = _compute_role_retention(task, list(pre_hrm.selection.selected_ids), evidence)

            evaluator = {
                "answer": task["answer"],
                "verifier": task.get("verifier", "exact"),
                "correct": correct,
                "quality": quality_score,
                "csr": csr,
                "required_evidence_ids": task["required_evidence_ids"],
                "oracle_evidence_ids": task.get("oracle_evidence_ids", task["required_evidence_ids"]),
                "_oracle_metadata": task["_oracle_metadata"],
                "metadata": task["metadata"],
                "family": task["family"],
                "source_cluster_id": task["source_cluster_id"],
                "role_retention": roles,
            }

            full_receipt = build_full_receipt(pre_receipt, hrm_result, evaluator)
            receipts.append({
                "task_id": full_receipt.task_id,
                "arm_id": full_receipt.arm_id,
                "split": full_receipt.split,
                "runtime_payload": full_receipt.runtime_payload,
                "evaluator_annotation": full_receipt.evaluator_annotation,
            })
        all_receipts[arm_id] = receipts
        elapsed = time.time() - t0
        q_scores = [r["evaluator_annotation"]["quality"] for r in receipts]
        print(f"{elapsed:.1f}s  Q={sum(q_scores)/len(q_scores):.2f}")

    # Write smoke receipts
    for arm_id, receipts in all_receipts.items():
        (out_dir / f"{arm_id}_smoke.jsonl").write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in receipts))

    # Summary table
    print("\n=== Smoke Summary ===")
    print(f"{'Arm':<8} {'Q':>6} {'Correct':>8} {'CSR':>6}")
    for arm_id in arm_ids:
        receipts = all_receipts[arm_id]
        n = len(receipts)
        q = sum(r["evaluator_annotation"]["quality"] for r in receipts) / n
        correct = sum(1 for r in receipts if r["evaluator_annotation"]["correct"]) / n
        csr = sum(r["evaluator_annotation"]["csr"] for r in receipts) / n
        print(f"{arm_id:<8} {q:6.2f} {correct:8.2f} {csr:6.2f}")

    manifest = {
        "mode": "smoke",
        "split": split,
        "protocol_sha256": _protocol_hash(),
        "arm_ids": arm_ids,
        "task_count": len(tasks),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
        "hrm_model_id": adapter.spec.model_id,
        "hrm_model_revision": adapter.spec.revision,
        "hrm_max_new_tokens": HRM_MAX_NEW_TOKENS,
        "device": _detect_device(),
        "dtype": os.environ.get("HRM_DTYPE", "float32"),
        "generation_condition": "DIRECT",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"\n  receipts: {out_dir}")
    print("  smoke test complete.")


def run_full(split: str = "development", arm_ids: list[str] | None = None):
    """Full development run with resumability."""
    if arm_ids is None:
        arm_ids = PRIMARY_ORDER
    tasks, evidence, texts = _load_split(split)
    records = _to_index_records(evidence)
    print(f"Full run: {len(tasks)} tasks × {len(arm_ids)} arms on {split}")

    out_dir = OUT / "full" / split
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load HRM
    print("\n--- Loading HRM model ---")
    t0 = time.time()
    adapter, condition = _load_hrm()
    print(f"  HRM loaded in {time.time()-t0:.1f}s")

    all_receipts: dict[str, list] = {}
    for arm_id in arm_ids:
        arm = ARMS[arm_id]
        arm_file = out_dir / f"{arm_id}.jsonl"
        # Check for existing results (resumability)
        existing: dict[str, dict] = {}
        if arm_file.exists():
            for line in arm_file.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    existing[r["task_id"]] = r
            print(f"  {arm_id}: {len(existing)}/{len(tasks)} existing results")

        print(f"  {arm_id}...", end=" ", flush=True)
        t0 = time.time()
        receipts: list[dict | None] = [None] * len(tasks)
        completed = 0

        # Pre-HRM stages (retrieval/identity/selection/ordering) are CPU-only
        # and unaffected by batching; run them for every task up front so the
        # GPU-bound HRM calls below can be grouped. Anything already resumed
        # from a prior run is filled in immediately and never re-sent to HRM.
        pending: list[tuple[int, dict, str]] = []  # (task index, pre_hrm, resume_key)
        pre_hrm_by_index: dict[int, object] = {}
        for i, task in enumerate(tasks):
            tid = task["task_id"]
            pre_hrm = run_pre_hrm_stages(task, arm, records, texts)
            rkey = _resume_key(task, arm, pre_hrm.packet.packet_hash)
            if tid in existing and existing[tid].get("resume_key") == rkey:
                receipts[i] = existing[tid]
                completed += 1
                continue
            pre_hrm_by_index[i] = pre_hrm
            pending.append((i, pre_hrm, rkey))

        def _build_receipt(task: dict, pre_hrm, rkey: str, hrm_result: HRMResult) -> dict:
            _assert_prompt_binding(pre_hrm, hrm_result)
            pre_receipt = build_pre_hrm_receipt(pre_hrm)
            _binary_score, correct = _verify_answer(task, hrm_result.output)
            quality_score = _compute_quality(
                task, list(pre_hrm.selection.selected_ids), evidence, correct)
            csr = _compute_csr(task, list(pre_hrm.selection.selected_ids))
            roles = _compute_role_retention(task, list(pre_hrm.selection.selected_ids), evidence)
            evaluator = {
                "answer": task["answer"],
                "verifier": task.get("verifier", "exact"),
                "correct": correct,
                "quality": quality_score,
                "csr": csr,
                "required_evidence_ids": task["required_evidence_ids"],
                "oracle_evidence_ids": task.get("oracle_evidence_ids", task["required_evidence_ids"]),
                "_oracle_metadata": task["_oracle_metadata"],
                "metadata": task["metadata"],
                "family": task["family"],
                "source_cluster_id": task["source_cluster_id"],
                "role_retention": roles,
            }
            full_receipt = build_full_receipt(pre_receipt, hrm_result, evaluator)
            return {
                "task_id": full_receipt.task_id,
                "arm_id": full_receipt.arm_id,
                "split": full_receipt.split,
                "resume_key": rkey,
                "runtime_payload": full_receipt.runtime_payload,
                "evaluator_annotation": full_receipt.evaluator_annotation,
            }

        # HRM generation, grouped into batches. Only usable because
        # tests/unit/test_batched_generation.py verifies batched output is
        # byte-identical to the sequential path -- this is not a mechanism
        # change, only how many prompts share one forward pass.
        for batch_start in range(0, len(pending), HRM_BATCH_SIZE):
            batch = pending[batch_start:batch_start + HRM_BATCH_SIZE]
            prompts = [pre_hrm_by_index[i].prompt for i, _, _ in batch]
            hrm_results = _run_hrm_batch(adapter, condition, prompts)
            for (i, pre_hrm, rkey), hrm_result in zip(batch, hrm_results):
                receipts[i] = _build_receipt(tasks[i], pre_hrm, rkey, hrm_result)
                completed += 1

            # Incremental write (append to file)
            arm_file.write_text(
                "".join(json.dumps(r, sort_keys=True) + "\n"
                        for r in receipts if r is not None))
            print(f"\r  {arm_id}: {completed}/{len(tasks)}", end="", flush=True)

        # Final write
        arm_file.write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in receipts if r is not None))
        all_receipts[arm_id] = receipts
        elapsed = time.time() - t0
        q_scores = [r["evaluator_annotation"]["quality"] for r in receipts]
        print(f"\r  {arm_id}: {completed}/{len(tasks)} in {elapsed:.1f}s  Q={sum(q_scores)/len(q_scores):.3f}")

    # Summary table
    print(f"\n=== Development Results ({split}) ===")
    print(f"{'Arm':<8} {'Q':>8} {'dQ':>8} {'Correct':>8} {'CSR':>8} {'AnsRet':>8} {'BrgRet':>8} {'IdRet':>8} {'Tokens':>8}")
    base_q = None
    for arm_id in arm_ids:
        receipts = all_receipts[arm_id]
        n = len(receipts)
        q = sum(r["evaluator_annotation"]["quality"] for r in receipts) / n
        if base_q is None:
            base_q = q
        dq = q - base_q
        correct = sum(1 for r in receipts if r["evaluator_annotation"]["correct"]) / n
        csr = sum(r["evaluator_annotation"]["csr"] for r in receipts) / n
        ans = sum(r["evaluator_annotation"]["role_retention"]["answer_retention"] for r in receipts) / n
        brg = sum(r["evaluator_annotation"]["role_retention"]["bridge_retention"] for r in receipts) / n
        idr = sum(r["evaluator_annotation"]["role_retention"]["identity_retention"] for r in receipts) / n
        tokens = sum(r["runtime_payload"]["packet"]["packet_token_count"] for r in receipts) / n
        print(f"{arm_id:<8} {q:8.3f} {dq:8.3f} {correct:8.3f} {csr:8.3f} {ans:8.3f} {brg:8.3f} {idr:8.3f} {tokens:8.1f}")

    # Manifest (self-describing, fully populated)
    from hrm_adaptive_memory.retrieval.embedding import BGE_SMALL
    from hrm_adaptive_memory.c4.contracts import (
        C4_CANDIDATE_BUDGET, C4_RRF_K, C4_PRIMARY_PACKET_BUDGET)
    import torch as _torch
    _device = _detect_device()
    _dtype = os.environ.get("HRM_DTYPE", "float32")
    manifest = build_manifest(
        repo=ROOT, mode="full", split=split, arm_ids=arm_ids,
        task_count=len(tasks),
        protocol_sha256=_protocol_hash(),
        task_corpus_sha256=sha256_corpus(CORPUS / split / "oracle_tasks.jsonl"),
        evidence_corpus_sha256=sha256_corpus(CORPUS / split / "evidence.jsonl"),
        hrm_model_id=adapter.spec.model_id,
        hrm_model_revision=adapter.spec.revision,
        hrm_max_new_tokens=HRM_MAX_NEW_TOKENS,
        bge_model_id=BGE_SMALL.get("model_id", ""),
        bge_revision=BGE_SMALL.get("revision", ""),
        pooling=BGE_SMALL.get("pooling", ""),
        normalization=str(BGE_SMALL.get("normalize", False)),
        query_prefix=BGE_SMALL.get("query_prefix", ""),
        candidate_budget=C4_CANDIDATE_BUDGET,
        packet_budget=C4_PRIMARY_PACKET_BUDGET,
        rrf_k=C4_RRF_K,
        query_policy_version="v1_frozen",
        bridge_policy_version="excluded_from_primary_c4",
        identity_policy_version="i3_v1",
        selector_version="s2c_deterministic_v1",
        selector_config_hash=hashlib.sha256(
            json.dumps({
                "selector": "s2c_with_s0_fallback",
                "packet_budget": C4_PRIMARY_PACKET_BUDGET,
                "candidate_budget": C4_CANDIDATE_BUDGET,
            }, sort_keys=True).encode()).hexdigest()[:16],
        device=_device,
        dtype=_dtype,
        batch_size=HRM_BATCH_SIZE,
        generation_condition="DIRECT",
    )
    # Merge arm_ids with any manifest.json already at this path: run_full()
    # is called once for the primary ladder and again for the diagnostic
    # arms, both targeting the same out_dir, and a plain overwrite would lose
    # the first call's arm list.
    manifest.update(_merged_arm_manifest_fields(out_dir, arm_ids))
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    # Run analyzer to produce analysis.json BEFORE hashing
    print("\n--- Running analyzer ---")
    analyzer_cmd = [sys.executable, str(ROOT / "scripts/analyze_gate_c4.py"),
                    "--dir", str(out_dir),
                    "--output", str(out_dir / "analysis.json")]
    analyzer_ret = subprocess.run(analyzer_cmd, capture_output=True, text=True)
    if analyzer_ret.stdout:
        print(analyzer_ret.stdout)
    if analyzer_ret.returncode != 0:
        print(f"  WARNING: Analyzer failed: {analyzer_ret.stderr[-300:]}")
    else:
        print("  Analysis complete.")

    # RESULTS.sha256 written LAST, after all artifacts exist
    write_results_hash(out_dir)
    print(f"\n  receipts: {out_dir}")
    print(f"  RESULTS.sha256 written last (after analysis)")


def run_dry_run(split: str = "development", arm_ids: list[str] | None = None):
    """CPU-only dry run: all pre-HRM stages, validate parity + leakage."""
    if arm_ids is None:
        arm_ids = [a for a in PRIMARY_ORDER if a != "C4_6"]
    tasks, evidence, texts = _load_split(split)
    records = _to_index_records(evidence)
    print(f"Dry run: {len(tasks)} tasks × {len(arm_ids)} arms on {split}")

    all_results: dict[str, list[PreHRMResult]] = {}
    for arm_id in arm_ids:
        arm = ARMS[arm_id]
        print(f"  {arm_id}...", end=" ", flush=True)
        t0 = time.time()
        results = []
        for task in tasks:
            r = run_pre_hrm_stages(task, arm, records, texts)
            results.append(r)
        all_results[arm_id] = results
        print(f"{time.time()-t0:.1f}s")

    # Validation gates
    print("\n=== Validation ===")
    ok, violations = validate_no_leakage(all_results)
    print(f"  {'PASS' if ok else 'FAIL'}: no oracle leakage ({len(violations)} violations)")
    for v in violations[:5]: print(f"    {v}")
    ok, violations = validate_all_parity(all_results)
    print(f"  {'PASS' if ok else 'FAIL'}: arm parity ({len(violations)} violations)")
    for v in violations[:10]: print(f"    {v}")
    ok, violations = validate_selected_in_pool(all_results)
    print(f"  {'PASS' if ok else 'FAIL'}: selected IDs in pool ({len(violations)} violations)")
    for v in violations[:5]: print(f"    {v}")
    ok, violations = validate_packet_budgets(all_results)
    print(f"  {'PASS' if ok else 'FAIL'}: packet budgets ({len(violations)} violations)")
    for v in violations[:5]: print(f"    {v}")
    ok, violations = validate_q3_query_formulation(all_results)
    print(f"  {'PASS' if ok else 'FAIL'}: Q3 query formulation ({len(violations)} violations)")
    for v in violations[:5]: print(f"    {v}")
    ok, violations = validate_merge_provenance(all_results)
    print(f"  {'PASS' if ok else 'FAIL'}: merge provenance ({len(violations)} violations)")
    for v in violations[:5]: print(f"    {v}")
    ok, violations = validate_causal_parity(all_results)
    print(f"  {'PASS' if ok else 'FAIL'}: causal parity ({len(violations)} violations)")
    for v in violations[:5]: print(f"    {v}")

    # Write dry-run receipts
    out_dir = OUT / "dry_run" / split
    out_dir.mkdir(parents=True, exist_ok=True)
    for arm_id, results in all_results.items():
        receipts = [build_pre_hrm_receipt(r) for r in results]
        (out_dir / f"{arm_id}_dry.jsonl").write_text(
            "".join(json.dumps({
                "task_id": r.task_id, "arm_id": r.arm_id, "split": r.split,
                "runtime_payload": r.runtime_payload,
                "evaluator_annotation": r.evaluator_annotation,
            }, sort_keys=True) + "\n" for r in receipts))

    manifest = build_manifest(
        repo=ROOT, mode="dry_run", split=split, arm_ids=arm_ids,
        task_count=len(tasks),
        protocol_sha256=_protocol_hash(),
        task_corpus_sha256=sha256_corpus(CORPUS / split / "oracle_tasks.jsonl"),
        evidence_corpus_sha256=sha256_corpus(CORPUS / split / "evidence.jsonl"),
    )
    manifest["validation"] = {
        "no_leakage": validate_no_leakage(all_results)[0],
        "parity": validate_all_parity(all_results)[0],
        "selected_in_pool": validate_selected_in_pool(all_results)[0],
        "packet_budgets": validate_packet_budgets(all_results)[0],
        "q3_query_formulation": validate_q3_query_formulation(all_results)[0],
        "merge_provenance": validate_merge_provenance(all_results)[0],
        "causal_parity": validate_causal_parity(all_results)[0],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    # RESULTS.sha256
    write_results_hash(out_dir)
    print(f"\n  receipts: {out_dir}")
    all_pass = all(manifest["validation"].values())
    if all_pass:
        print("  ALL VALIDATION GATES PASSED — safe to proceed to HRM")
    else:
        print("  VALIDATION FAILED — do NOT start HRM")
    return all_pass


def main():
    parser = argparse.ArgumentParser(description="Gate C4 runner")
    parser.add_argument("mode", choices=["dry-run", "smoke", "full"])
    parser.add_argument("--split", default="development")
    parser.add_argument("--arms", nargs="*", default=None)
    args = parser.parse_args()

    if args.mode == "dry-run":
        run_dry_run(args.split, args.arms)
    elif args.mode == "smoke":
        run_smoke(args.split)
    elif args.mode == "full":
        run_full(args.split, args.arms)


if __name__ == "__main__":
    main()
