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

import argparse, hashlib, json, sys, time, re, math
from pathlib import Path
from dataclasses import asdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hrm_adaptive_memory.contracts import IndexRecord
from hrm_adaptive_memory.c4.contracts import *
from hrm_adaptive_memory.c4.arms import ARMS, PRIMARY_ORDER
from hrm_adaptive_memory.c4.query_stage import run_query_stage
from hrm_adaptive_memory.c4.retrieval_stage import run_retrieval_stage, clear_backend_cache
from hrm_adaptive_memory.c4.identity_stage import run_identity_stage
from hrm_adaptive_memory.c4.selection_stage import run_selection_stage
from hrm_adaptive_memory.c4.packet_stage import run_packet_stage
from hrm_adaptive_memory.c4.receipts import build_pre_hrm_receipt, assert_runtime_clean, build_full_receipt
from hrm_adaptive_memory.c4.parity import (
    validate_all_parity, validate_no_leakage,
    validate_selected_in_pool, validate_packet_budgets)

CORPUS = ROOT / "data/hrm/controlled_gate_a_v4"
OUT = ROOT / "evidence/gate_c4"
HRM_MAX_NEW_TOKENS = 64


def _protocol_hash() -> str:
    return hashlib.sha256(
        (ROOT / "configs/gate_c4_protocol.json").read_bytes()).hexdigest()


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
    """Verify HRM output against the task answer (evaluator-side only)."""
    verifier = task.get("verifier", "exact")
    answer = task["answer"]

    def _norm(s):
        return " ".join(re.findall(r"\w+", s.lower()))

    if verifier == "exact":
        passed = _norm(output) == _norm(answer)
    elif verifier == "numeric":
        numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", output)
        passed = bool(numbers) and math.isclose(
            float(numbers[-1]), float(answer), rel_tol=1e-9, abs_tol=1e-9)
    elif verifier == "canonical":
        answer_terms = tuple(re.findall(r"\w+", _norm(answer)))
        output_terms = tuple(re.findall(r"\w+", _norm(output)))
        width = len(answer_terms)
        starts = [i for i in range(len(output_terms) - width + 1)
                  if output_terms[i:i + width] == answer_terms] if width else []
        passed = bool(starts) and starts[-1] + width == len(output_terms)
    else:
        passed = _norm(output) == _norm(answer)
    return float(passed), passed


def _compute_quality(task: dict, selected_ids: list[str],
                     evidence: list[dict], correct: bool) -> float:
    """Compute quality score (matches selector ladder definition)."""
    required = set(task["required_evidence_ids"])
    selected = set(selected_ids)
    complete = required <= selected
    # Quality = 1.0 if complete and HRM correct, 0.5 if complete but HRM wrong,
    # 0.25 if partial but HRM correct, 0.0 otherwise.
    if complete and correct:
        return 1.0
    elif complete and not correct:
        return 0.5
    elif not complete and correct:
        return 0.25
    return 0.0


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


def run_pre_hrm_stages(task: dict, arm: C4Arm, records: list[IndexRecord],
                       texts: dict[str, str]) -> PreHRMResult:
    """Run all pre-HRM stages for one task under one arm."""
    question = task["question"]
    split = task.get("split", "development")

    # Query stage
    state_before, query_result = run_query_stage(question, arm)

    # Retrieval stage
    retrieval_result = run_retrieval_stage(query_result.rendered_query, arm, records)

    # Identity stage
    identity_result = run_identity_stage(question, arm, retrieval_result, texts)

    # If identity resolved, update state
    state_after = state_before
    canonical_question = None
    if identity_result.status == "RESOLVED" and identity_result.canonical:
        state_after = state_before.with_identity(
            identity_result.surface or state_before.subject,
            identity_result.canonical)
        canonical_question = question.replace(
            identity_result.surface or "",
            identity_result.canonical)

    # Selection stage
    selection_result = run_selection_stage(
        arm, question, retrieval_result, identity_result, texts,
        task.get("required_evidence_ids", []),
        canonical_question=canonical_question)

    # Packet stage
    prompt, packet_result = run_packet_stage(arm, question, selection_result, texts)

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
    )


def _load_hrm():
    """Load the HRM model (lazy, only when needed)."""
    from hrm_adaptive_memory.hrm.model import HRMAdapter, HRMModelSpec, PromptCondition
    adapter = HRMAdapter.from_pretrained(spec=HRMModelSpec())
    return adapter, PromptCondition.DIRECT


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


def _resume_key(task: dict, arm: C4Arm, packet_hash: str) -> str:
    """Build a resume key that invalidates on config/code changes."""
    return hashlib.sha256(
        f"{task['task_id']}|{arm.arm_id}|{packet_hash}|{arm.query_policy}|"
        f"{arm.retrieval_policy}|{arm.identity_policy}|{arm.selector_policy}".encode()
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
            prompt = pre_hrm.packet.packet_contents
            from hrm_adaptive_memory.evidence.packing import compose_evidence_prompt
            full_prompt = compose_evidence_prompt(
                task["question"],
                [texts.get(eid, "") for eid in pre_hrm.selection.selected_ids if eid in texts])

            hrm_result = _run_hrm(adapter, condition, full_prompt)
            quality_score, correct = _verify_answer(task, hrm_result.output)
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
        "hrm_model_id": adapter.spec.model_id,
        "hrm_model_revision": adapter.spec.revision,
        "hrm_max_new_tokens": HRM_MAX_NEW_TOKENS,
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
        receipts = []
        completed = 0
        for i, task in enumerate(tasks):
            tid = task["task_id"]
            # Check resume cache
            pre_hrm = run_pre_hrm_stages(task, arm, records, texts)
            rkey = _resume_key(task, arm, pre_hrm.packet.packet_hash)
            if tid in existing and existing[tid].get("resume_key") == rkey:
                receipts.append(existing[tid])
                completed += 1
                continue

            # Run HRM
            pre_receipt = build_pre_hrm_receipt(pre_hrm)
            from hrm_adaptive_memory.evidence.packing import compose_evidence_prompt
            full_prompt = compose_evidence_prompt(
                task["question"],
                [texts.get(eid, "") for eid in pre_hrm.selection.selected_ids if eid in texts])
            hrm_result = _run_hrm(adapter, condition, full_prompt)
            quality_score, correct = _verify_answer(task, hrm_result.output)
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
            receipt_dict = {
                "task_id": full_receipt.task_id,
                "arm_id": full_receipt.arm_id,
                "split": full_receipt.split,
                "resume_key": rkey,
                "runtime_payload": full_receipt.runtime_payload,
                "evaluator_annotation": full_receipt.evaluator_annotation,
            }
            receipts.append(receipt_dict)
            completed += 1

            # Incremental write (append to file)
            if completed % 10 == 0 or i == len(tasks) - 1:
                arm_file.write_text(
                    "".join(json.dumps(r, sort_keys=True) + "\n" for r in receipts))
                print(f"\r  {arm_id}: {completed}/{len(tasks)}", end="", flush=True)

        # Final write
        arm_file.write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in receipts))
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

    # Manifest
    manifest = {
        "mode": "full",
        "split": split,
        "protocol_sha256": _protocol_hash(),
        "arm_ids": arm_ids,
        "task_count": len(tasks),
        "hrm_model_id": adapter.spec.model_id,
        "hrm_model_revision": adapter.spec.revision,
        "hrm_max_new_tokens": HRM_MAX_NEW_TOKENS,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"\n  receipts: {out_dir}")


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

    manifest = {
        "mode": "dry_run",
        "split": split,
        "protocol_sha256": _protocol_hash(),
        "arm_ids": arm_ids,
        "task_count": len(tasks),
        "validation": {
            "no_leakage": validate_no_leakage(all_results)[0],
            "parity": validate_all_parity(all_results)[0],
            "selected_in_pool": validate_selected_in_pool(all_results)[0],
            "packet_budgets": validate_packet_budgets(all_results)[0],
        },
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
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
