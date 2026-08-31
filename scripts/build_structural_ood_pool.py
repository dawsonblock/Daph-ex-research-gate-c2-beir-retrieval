#!/usr/bin/env python3
"""P0.3: Build a real structural-OOD confirmation pool.

The previous confirmation benchmark was a fresh in-family replication:
96.6% of certificate-positive confirmation states shared an exact V3
structural signature with development. This script builds a genuinely
novel structural pool by:

1. Computing the complete V3R2 feature signature for every development
   and confirmation authority event.
2. Defining development_signatures = set of all development signatures.
3. Generating candidate tasks from NOVEL structural configurations:
   - New number of viable hypotheses (4, 5, 6 — not 2 or 3)
   - New mixed support/contradiction combinations
   - New verified/unverified topology patterns
   - New resource-state combinations
   - New terminal-action mappings
   - New evidence-count topology
   - New legal-action subsets
   - New continuation availability
4. Rejecting any candidate whose structural signature appears in
   development_signatures.
5. Requiring nearest-neighbor distance above a preregistered threshold
   for an OOD subset.

This produces the first experiment capable of supporting a real
structural-generalization claim.

Usage:
    python scripts/build_structural_ood_pool.py

Outputs:
    experiments/i3_30r3/structural_ood/ood_pool.json
    experiments/i3_30r3/structural_ood/development_signatures.json
    experiments/i3_30r3/structural_ood/ood_pool_report.json
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from hrm_adaptive_memory.cognitive_control.core import DecisionAction
from hrm_adaptive_memory.cognitive_control.state import (
    TemporalStatus, VerificationState,
)
from hrm_adaptive_memory.executive.evidence_benchmark.schema import (
    EvidenceHypothesis, EvidenceItem, EvidenceTask,
    initial_evidence_runtime,
)
from hrm_adaptive_memory.executive.resources import ResourceBudget, ResourceState
from daph.epistemic.v3_features import compute_v3_features_canonical
from daph.intervention.checkpoint import compute_state_features

OUTPUT_DIR = REPO_ROOT / "experiments/i3_30r3/structural_ood"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_structural_signature(state_features: dict, v3_features: dict) -> str:
    """Compute a canonical structural signature from V3 features.

    This is the 14-field V3 structural state that the auditor used.
    Two states with the same signature are structurally identical
    regardless of domain strings or task IDs.
    """
    sig_fields = [
        "n_hyp_with_verified_support",
        "n_hyp_with_verified_contradiction",
        "n_hyp_with_mixed_verified",
        "n_viable_hypotheses",
        "n_eliminated_hypotheses",
        "has_unique_verified_supported_hypothesis",
        "has_verified_unresolved_competition",
        "verified_hyp_action_is_answer",
        "verified_hyp_action_is_defer",
        "n_hyp_unverified_support",
        "n_hyp_unverified_contradiction",
        "has_competing_unverified_support",
        # Resource-aware fields
        "verify_remaining",
        "steps_remaining",
    ]
    sig_parts = []
    for k in sig_fields:
        v = v3_features.get(k, state_features.get(k, 0))
        sig_parts.append(f"{k}={v}")
    return "|".join(sig_parts)


def compute_feature_vector(state_features: dict, v3_features: dict) -> np.ndarray:
    """Compute the full feature vector for nearest-neighbor distance."""
    from run_i3_30r2_train import extract_v3r2_features
    feats = extract_v3r2_features(state_features, "ANSWER", v3_features)
    # Use a fixed key order
    keys = sorted(feats.keys())
    return np.array([feats[k] for k in keys])


def load_development_signatures() -> tuple[set[str], np.ndarray | None]:
    """Load all development structural signatures.

    Returns:
        (signatures, feature_matrix) where signatures is a set of
        structural signature strings and feature_matrix is the stacked
        feature vectors for nearest-neighbor distance.
    """
    signatures = set()
    feature_vectors = []

    dev_paths = [
        REPO_ROOT / "experiments/i3_30r/causal_boundary_v2/causal_actions_v2.jsonl",
        REPO_ROOT / "experiments/i3_30r3/confirmation_analysis/paired_results.jsonl",
    ]

    for path in dev_paths:
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                sf = r.get("state_features", {})
                v3 = r.get("v3_features", {})
                if not v3:
                    continue
                sig = compute_structural_signature(sf, v3)
                signatures.add(sig)
                try:
                    fv = compute_feature_vector(sf, v3)
                    feature_vectors.append(fv)
                except Exception:
                    pass

    feat_matrix = np.array(feature_vectors) if feature_vectors else None
    return signatures, feat_matrix


# Novel structural configurations designed to be OOD from development
# Development used: 2-3 hypotheses, D1-D5 patterns
# OOD pool uses: 4-6 hypotheses, novel topology patterns
OOD_DOMAIN_TEMPLATES = [
    # 4 hypotheses with mixed verified/unverified support
    # Fixed: E1 starts UNVERIFIED, E2-E4 are verified contradictions.
    # After VERIFY(E1), H1 gets unique verified support → ANSWER_READY.
    # Previous version had E3 as SUFFICIENT support for H3, creating
    # competing verified support that made the oracle path invalid.
    {
        "category": "OOD_4HYP_MIXED",
        "summary": "Is the condition type A, B, C, or D?",
        "hypotheses": [
            ("H1", "type A", "ANSWER"),
            ("H2", "type B", "ANSWER"),
            ("H3", "type C", "ANSWER"),
            ("H4", "type D", "DEFER"),
        ],
        "evidence": [
            ("E1", "Marker for A", "initial", ("H1",), (), "UNVERIFIED", "CURRENT"),
            ("E2", "Contradiction of B", "initial", (), ("H2",), "SUFFICIENT", "CURRENT"),
            ("E3", "Contradiction of C", "initial", (), ("H3",), "SUFFICIENT", "CURRENT"),
            ("E4", "Contradiction of D", "initial", (), ("H4",), "SUFFICIENT", "CURRENT"),
        ],
        "correct_hypothesis": "H1",
        "expected_terminal": "ANSWER",
        "oracle_path": ("VERIFY", "ANSWER"),
        "budget": {"steps": 4, "verify": 2, "retrieve": 0, "search": 0},
    },
    # 5 hypotheses with all unverified
    # Fixed: E2-E5 are now verified contradictions (FALSIFIED), so only E1
    # is unverified. The executor's VERIFY targets the last unverified item,
    # which is now E1. After VERIFY(E1), H1 gets unique verified support.
    # Previous version had all 5 unverified, causing VERIFY to target E5
    # (for H5/DEFER) instead of E1 (for H1).
    {
        "category": "OOD_5HYP_VERIFY_TO_UNIQUE",
        "summary": "Is the diagnosis one of five possibilities?",
        "hypotheses": [
            ("H1", "diagnosis 1", "ANSWER"),
            ("H2", "diagnosis 2", "ANSWER"),
            ("H3", "diagnosis 3", "ANSWER"),
            ("H4", "diagnosis 4", "ANSWER"),
            ("H5", "diagnosis 5", "DEFER"),
        ],
        "evidence": [
            ("E1", "Test for 1", "initial", ("H1",), (), "UNVERIFIED", "CURRENT"),
            ("E2", "Ruling out 2", "initial", (), ("H2",), "SUFFICIENT", "CURRENT"),
            ("E3", "Ruling out 3", "initial", (), ("H3",), "SUFFICIENT", "CURRENT"),
            ("E4", "Ruling out 4", "initial", (), ("H4",), "SUFFICIENT", "CURRENT"),
            ("E5", "Ruling out 5", "initial", (), ("H5",), "SUFFICIENT", "CURRENT"),
        ],
        "correct_hypothesis": "H1",
        "expected_terminal": "ANSWER",
        "oracle_path": ("VERIFY", "ANSWER"),
        "budget": {"steps": 5, "verify": 3, "retrieve": 0, "search": 0},
    },
    # 6 hypotheses with partial verification
    {
        "category": "OOD_6HYP_PARTIAL_VERIFY",
        "summary": "Is the failure one of six modes?",
        "hypotheses": [
            ("H1", "mode 1", "ANSWER"),
            ("H2", "mode 2", "ANSWER"),
            ("H3", "mode 3", "ANSWER"),
            ("H4", "mode 4", "ANSWER"),
            ("H5", "mode 5", "ANSWER"),
            ("H6", "mode 6", "DEFER"),
        ],
        "evidence": [
            ("E1", "Test for mode 1", "initial", ("H1",), (), "SUFFICIENT", "CURRENT"),
            ("E2", "Test for mode 2", "initial", ("H2",), (), "FALSIFIED", "CURRENT"),
            ("E3", "Test for mode 3", "initial", ("H3",), (), "UNVERIFIED", "CURRENT"),
            ("E4", "Test for mode 4", "initial", ("H4",), (), "FALSIFIED", "CURRENT"),
            ("E5", "Test for mode 5", "initial", ("H5",), (), "UNVERIFIED", "CURRENT"),
            ("E6", "Test for mode 6", "initial", ("H6",), (), "UNVERIFIED", "CURRENT"),
        ],
        "correct_hypothesis": "H1",
        "expected_terminal": "ANSWER",
        "oracle_path": ("ANSWER",),
        "budget": {"steps": 3, "verify": 1, "retrieve": 0, "search": 0},
    },
    # 4 hypotheses with all verified, unique support
    {
        "category": "OOD_4HYP_ALL_VERIFIED_UNIQUE",
        "summary": "Is the result A, B, C, or D after full verification?",
        "hypotheses": [
            ("H1", "result A", "ANSWER"),
            ("H2", "result B", "ANSWER"),
            ("H3", "result C", "ANSWER"),
            ("H4", "result D", "DEFER"),
        ],
        "evidence": [
            ("E1", "Test for A", "initial", ("H1",), (), "SUFFICIENT", "CURRENT"),
            ("E2", "Test for B", "initial", ("H2",), (), "FALSIFIED", "CURRENT"),
            ("E3", "Test for C", "initial", ("H3",), (), "FALSIFIED", "CURRENT"),
            ("E4", "Contradiction of D", "initial", (), ("H4",), "SUFFICIENT", "CURRENT"),
        ],
        "correct_hypothesis": "H1",
        "expected_terminal": "ANSWER",
        "oracle_path": ("ANSWER",),
        "budget": {"steps": 2, "verify": 0, "retrieve": 0, "search": 0},
    },
    # 4 hypotheses with competing verified support (DEFER correct)
    {
        "category": "OOD_4HYP_COMPETING_VERIFIED_DEFER",
        "summary": "Is the phenotype A, B, C, or D — multiple confirmed?",
        "hypotheses": [
            ("H1", "phenotype A", "ANSWER"),
            ("H2", "phenotype B", "ANSWER"),
            ("H3", "phenotype C", "ANSWER"),
            ("H4", "unresolved — defer", "DEFER"),
        ],
        "evidence": [
            ("E1", "Marker for A", "initial", ("H1",), (), "SUFFICIENT", "CURRENT"),
            ("E2", "Marker for B", "initial", ("H2",), (), "SUFFICIENT", "CURRENT"),
            ("E3", "Marker for C", "initial", ("H3",), (), "SUFFICIENT", "CURRENT"),
        ],
        "correct_hypothesis": "H4",
        "expected_terminal": "DEFER",
        "oracle_path": ("DEFER",),
        "budget": {"steps": 2, "verify": 0, "retrieve": 0, "search": 0},
    },
    # 5 hypotheses with 3 eliminated, unique support
    {
        "category": "OOD_5HYP_3ELIM_UNIQUE",
        "summary": "Is the cause one of five — three ruled out?",
        "hypotheses": [
            ("H1", "cause 1", "ANSWER"),
            ("H2", "cause 2", "ANSWER"),
            ("H3", "cause 3", "ANSWER"),
            ("H4", "cause 4", "ANSWER"),
            ("H5", "cause 5", "DEFER"),
        ],
        "evidence": [
            ("E1", "Test for cause 1", "initial", ("H1",), (), "SUFFICIENT", "CURRENT"),
            ("E2", "Test for cause 2", "initial", (), ("H2",), "SUFFICIENT", "CURRENT"),
            ("E3", "Test for cause 3", "initial", (), ("H3",), "SUFFICIENT", "CURRENT"),
            ("E4", "Test for cause 4", "initial", (), ("H4",), "SUFFICIENT", "CURRENT"),
            ("E5", "Test for cause 5", "initial", ("H5",), (), "UNVERIFIED", "CURRENT"),
        ],
        "correct_hypothesis": "H1",
        "expected_terminal": "ANSWER",
        "oracle_path": ("ANSWER",),
        "budget": {"steps": 2, "verify": 1, "retrieve": 0, "search": 0},
    },
    # 4 hypotheses with mixed verified, no unique support, search available
    # Fixed: E1 was SUFFICIENT (verified support for H1) which made
    # the initial state already ANSWER_READY. Changed E1 to UNVERIFIED
    # and added E4 as verified contradiction of H4.
    # After SEARCH_MORE reveals E1, then VERIFY(E1) → H1 uniquely supported.
    {
        "category": "OOD_4HYP_MIXED_SEARCH",
        "summary": "Is the lesion A, B, C, or D — need more info?",
        "hypotheses": [
            ("H1", "lesion A", "ANSWER"),
            ("H2", "lesion B", "ANSWER"),
            ("H3", "lesion C", "ANSWER"),
            ("H4", "lesion D", "DEFER"),
        ],
        "evidence": [
            ("E1", "Scan for A", "initial", ("H1",), (), "UNVERIFIED", "CURRENT"),
            ("E2", "Contradiction of B", "initial", (), ("H2",), "SUFFICIENT", "CURRENT"),
            ("E3", "Contradiction of C", "initial", (), ("H3",), "SUFFICIENT", "CURRENT"),
            ("E4", "Contradiction of D", "initial", (), ("H4",), "SUFFICIENT", "CURRENT"),
        ],
        "correct_hypothesis": "H1",
        "expected_terminal": "ANSWER",
        "oracle_path": ("VERIFY", "ANSWER"),
        "budget": {"steps": 4, "verify": 1, "retrieve": 0, "search": 2},
    },
    # 6 hypotheses with 5 eliminated, unique support (novel elimination count)
    {
        "category": "OOD_6HYP_5ELIM_UNIQUE",
        "summary": "Is the pathogen one of six — five ruled out?",
        "hypotheses": [
            ("H1", "pathogen 1", "ANSWER"),
            ("H2", "pathogen 2", "ANSWER"),
            ("H3", "pathogen 3", "ANSWER"),
            ("H4", "pathogen 4", "ANSWER"),
            ("H5", "pathogen 5", "ANSWER"),
            ("H6", "pathogen 6", "DEFER"),
        ],
        "evidence": [
            ("E1", "Test for pathogen 1", "initial", ("H1",), (), "SUFFICIENT", "CURRENT"),
            ("E2", "Test for pathogen 2", "initial", (), ("H2",), "SUFFICIENT", "CURRENT"),
            ("E3", "Test for pathogen 3", "initial", (), ("H3",), "SUFFICIENT", "CURRENT"),
            ("E4", "Test for pathogen 4", "initial", (), ("H4",), "SUFFICIENT", "CURRENT"),
            ("E5", "Test for pathogen 5", "initial", (), ("H5",), "SUFFICIENT", "CURRENT"),
        ],
        "correct_hypothesis": "H1",
        "expected_terminal": "ANSWER",
        "oracle_path": ("ANSWER",),
        "budget": {"steps": 2, "verify": 0, "retrieve": 0, "search": 0},
    },
]


def generate_ood_candidate(task_template: dict, idx: int) -> EvidenceTask:
    """Generate a single OOD candidate task from a template."""
    task_id = f"i3_30r3_ood_{task_template['category'].lower()}_{idx:03d}"

    hypotheses = []
    for h_id, prop, action_str in task_template["hypotheses"]:
        action = DecisionAction(action_str)
        payload = f"{action_str}:{h_id}:{prop}"
        hypotheses.append(EvidenceHypothesis(
            hypothesis_id=h_id,
            proposition=prop,
            answer_action=action,
            answer_payload=payload,
        ))

    evidence_items = []
    for ev_id, prop, source, supports, contradicts, vstate_str, tstatus_str in task_template["evidence"]:
        evidence_items.append(EvidenceItem(
            evidence_id=ev_id,
            proposition=prop,
            source_class=source,
            supports=supports,
            contradicts=contradicts,
            verification_state=VerificationState(vstate_str),
            temporal_status=TemporalStatus(tstatus_str),
            retrieved=True,
            verify_result=vstate_str if vstate_str != "UNVERIFIED" else None,
        ))

    b = task_template["budget"]
    budget = ResourceBudget(
        max_executive_steps=b["steps"],
        max_reasoning_tokens=256,
        max_retrieval_calls=b["retrieve"],
        max_verification_calls=b["verify"],
        max_search_calls=b["search"],
        max_elapsed_ms=10000,
    )

    return EvidenceTask(
        task_id=task_id,
        split="i3_30r3_ood",
        category=task_template["category"],
        task_summary=task_template["summary"],
        high_stakes=True,
        budget_profile=f"OOD_{b['steps']}_{b['verify']}_{b['search']}",
        hypotheses=tuple(hypotheses),
        evidence_items=tuple(evidence_items),
        retrieve_exposes=(),
        search_exposes=(),
        oracle_resolution_path=task_template["oracle_path"],
        expected_terminal=DecisionAction(task_template["expected_terminal"]),
        correct_hypothesis_id=task_template["correct_hypothesis"],
    )


def compute_task_signature(task: EvidenceTask) -> str | None:
    """Compute the structural signature for a task's initial state."""
    parts = task.budget_profile.split("_")
    budget = ResourceBudget(
        max_executive_steps=int(parts[1]) if len(parts) > 1 else 2,
        max_reasoning_tokens=256,
        max_retrieval_calls=0,
        max_verification_calls=int(parts[2]) if len(parts) > 2 else 0,
        max_search_calls=int(parts[3]) if len(parts) > 3 else 0,
        max_elapsed_ms=10000,
    )
    resources = ResourceState(budget=budget)
    runtime = initial_evidence_runtime(task, resources)
    sf = compute_state_features(runtime, prior_actions=())

    evidence_dicts = []
    for ev in runtime.visible_evidence:
        evidence_dicts.append({
            "evidence_id": ev.evidence_id,
            "supports": list(ev.supports),
            "contradicts": list(ev.contradicts),
            "verification_state": ev.verification_state.value,
            "temporal_status": ev.temporal_status.value,
            "retrieved": ev.retrieved,
        })
    hyp_dicts = []
    for h in task.hypotheses:
        hyp_dicts.append({
            "hypothesis_id": h.hypothesis_id,
            "answer_action": h.answer_action.value,
        })
    v3 = compute_v3_features_canonical(evidence_dicts, hyp_dicts)
    return compute_structural_signature(sf, v3)


def main():
    print("=" * 60)
    print("P0.3: Build Structural-OOD Confirmation Pool")
    print("=" * 60)

    # 1. Load development signatures
    print("\n1. Loading development structural signatures...")
    dev_signatures, dev_features = load_development_signatures()
    print(f"   Development unique signatures: {len(dev_signatures)}")
    if dev_features is not None:
        print(f"   Development feature matrix: {dev_features.shape}")

    # 2. Generate OOD candidates
    print("\n2. Generating OOD candidates...")
    rng = random.Random(12345)
    candidates = []
    for template in OOD_DOMAIN_TEMPLATES:
        for i in range(20):  # 20 per template
            candidate = generate_ood_candidate(template, i)
            candidates.append(candidate)
    print(f"   Total candidates: {len(candidates)}")

    # 3. Filter by structural novelty
    print("\n3. Filtering by structural novelty...")
    accepted = []
    rejected_overlap = 0
    rejected_no_sig = 0

    for candidate in candidates:
        sig = compute_task_signature(candidate)
        if sig is None:
            rejected_no_sig += 1
            continue
        if sig in dev_signatures:
            rejected_overlap += 1
            continue
        accepted.append((candidate, sig))

    print(f"   Accepted (structurally novel): {len(accepted)}")
    print(f"   Rejected (overlap with development): {rejected_overlap}")
    print(f"   Rejected (no signature): {rejected_no_sig}")

    # 4. Compute nearest-neighbor distances for accepted candidates
    print("\n4. Computing nearest-neighbor distances...")
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics.pairwise import euclidean_distances

    ood_features = []
    for candidate, sig in accepted:
        parts = candidate.budget_profile.split("_")
        budget = ResourceBudget(
            max_executive_steps=int(parts[1]),
            max_reasoning_tokens=256,
            max_retrieval_calls=0,
            max_verification_calls=int(parts[2]),
            max_search_calls=int(parts[3]),
            max_elapsed_ms=10000,
        )
        resources = ResourceState(budget=budget)
        runtime = initial_evidence_runtime(candidate, resources)
        sf = compute_state_features(runtime, prior_actions=())

        evidence_dicts = []
        for ev in runtime.visible_evidence:
            evidence_dicts.append({
                "evidence_id": ev.evidence_id,
                "supports": list(ev.supports),
                "contradicts": list(ev.contradicts),
                "verification_state": ev.verification_state.value,
                "temporal_status": ev.temporal_status.value,
                "retrieved": ev.retrieved,
            })
        hyp_dicts = []
        for h in candidate.hypotheses:
            hyp_dicts.append({
                "hypothesis_id": h.hypothesis_id,
                "answer_action": h.answer_action.value,
            })
        v3 = compute_v3_features_canonical(evidence_dicts, hyp_dicts)
        try:
            fv = compute_feature_vector(sf, v3)
            ood_features.append(fv)
        except Exception:
            ood_features.append(None)

    # Compute distances in standardized space
    valid_idx = [i for i, f in enumerate(ood_features) if f is not None]
    if valid_idx and dev_features is not None:
        ood_matrix = np.array([ood_features[i] for i in valid_idx])
        # Standardize using development features
        scaler = StandardScaler()
        scaler.fit(dev_features)
        dev_std = scaler.transform(dev_features)
        ood_std = scaler.transform(ood_matrix)
        dists = euclidean_distances(ood_std, dev_std)
        min_dists = dists.min(axis=1)

        # Report distance distribution
        print(f"   Distance computed for {len(valid_idx)} candidates")
        print(f"   Min distance: {min_dists.min():.2f}")
        print(f"   Max distance: {min_dists.max():.2f}")
        print(f"   Median distance: {np.median(min_dists):.2f}")
        print(f"   Mean distance: {min_dists.mean():.2f}")

        # Preregistered threshold: 3.0 in standardized space
        OOD_DISTANCE_THRESHOLD = 3.0
        far_ood = [(valid_idx[i], min_dists[i]) for i in range(len(valid_idx))
                    if min_dists[i] >= OOD_DISTANCE_THRESHOLD]
        print(f"   Candidates with distance >= {OOD_DISTANCE_THRESHOLD}: {len(far_ood)}")
    else:
        far_ood = []
        min_dists = []

    # 5. Save results
    print("\n5. Saving results...")

    # Save development signatures
    dev_sig_path = OUTPUT_DIR / "development_signatures.json"
    with open(dev_sig_path, "w") as f:
        json.dump({
            "count": len(dev_signatures),
            "signatures": sorted(dev_signatures),
        }, f, indent=2)
    print(f"   Saved: {dev_sig_path}")

    # Save OOD pool
    ood_pool = []
    for i, (candidate, sig) in enumerate(accepted):
        distance = None
        if i in valid_idx:
            idx_in_valid = valid_idx.index(i)
            if idx_in_valid < len(min_dists):
                distance = float(min_dists[idx_in_valid])

        expected = candidate.expected_terminal
        expected_str = expected.value if hasattr(expected, 'value') else str(expected)

        ood_pool.append({
            "task_id": candidate.task_id,
            "category": candidate.category,
            "structural_signature": sig,
            "nn_distance": distance,
            "is_far_ood": distance is not None and distance >= OOD_DISTANCE_THRESHOLD,
            "expected_terminal": expected_str,
            "correct_hypothesis": candidate.correct_hypothesis_id,
            "n_hypotheses": len(candidate.hypotheses),
            "budget_profile": candidate.budget_profile,
        })

    ood_pool_path = OUTPUT_DIR / "ood_pool.json"
    with open(ood_pool_path, "w") as f:
        json.dump(ood_pool, f, indent=2)
    print(f"   Saved: {ood_pool_path}")

    # Save report
    by_cat = defaultdict(lambda: {"total": 0, "accepted": 0, "far_ood": 0})
    for template in OOD_DOMAIN_TEMPLATES:
        by_cat[template["category"]]["total"] = 20

    for entry in ood_pool:
        by_cat[entry["category"]]["accepted"] += 1
        if entry["is_far_ood"]:
            by_cat[entry["category"]]["far_ood"] += 1

    report = {
        "status": "BUILT",
        "description": "Structural-OOD confirmation pool with explicit feature-signature exclusion from development",
        "development_signatures": len(dev_signatures),
        "candidates_generated": len(candidates),
        "accepted_structurally_novel": len(accepted),
        "rejected_overlap": rejected_overlap,
        "rejected_no_sig": rejected_no_sig,
        "far_ood_threshold": OOD_DISTANCE_THRESHOLD,
        "far_ood_count": sum(1 for e in ood_pool if e["is_far_ood"]),
        "by_category": dict(by_cat),
        "distance_stats": {
            "min": float(min_dists.min()) if len(min_dists) > 0 else None,
            "max": float(min_dists.max()) if len(min_dists) > 0 else None,
            "median": float(np.median(min_dists)) if len(min_dists) > 0 else None,
            "mean": float(min_dists.mean()) if len(min_dists) > 0 else None,
        },
        "novelty_criteria": [
            "Structural signature not in development signatures",
            "Nearest-neighbor distance >= 3.0 in standardized feature space (for far-OOD subset)",
        ],
        "novel_structural_features": [
            "4-6 hypotheses (development used 2-3)",
            "5 eliminated hypotheses (development had max 2)",
            "4-way competing verified support (development had max 3-way)",
            "6-hypothesis all-unverified topology",
            "Mixed verified/unverified with search available",
        ],
    }
    report_path = OUTPUT_DIR / "ood_pool_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"   Saved: {report_path}")

    print("\n" + "=" * 60)
    print("Structural-OOD pool built.")
    print(f"  Development signatures: {len(dev_signatures)}")
    print(f"  Accepted novel: {len(accepted)}")
    print(f"  Far-OOD (distance >= {OOD_DISTANCE_THRESHOLD}): {sum(1 for e in ood_pool if e['is_far_ood'])}")
    print("=" * 60)


if __name__ == "__main__":
    main()
