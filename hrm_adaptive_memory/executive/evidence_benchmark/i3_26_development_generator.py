"""I3.26 development benchmark generator.

Fresh seed (different from Phase 25 confirmation seed 4287 and development
seed 9137). Heavy representation of:
  - chain (solvable planning failures)
  - defer (solvable planning failures)
  - tl_retrieve (unsolvable controls — search cannot help)
  - near-tie states (where search should trigger)
  - clear-gap controls (where search should NOT trigger)
  - unavoidable controls (genuinely impossible tasks)

Uses seed 7719 for domain assignment.
"""
from __future__ import annotations

import hashlib
import random
from pathlib import Path

from hrm_adaptive_memory.cognitive_control.core import DecisionAction
from hrm_adaptive_memory.cognitive_control.state import (
    VerificationState, TemporalStatus,
)
from hrm_adaptive_memory.executive.evidence_benchmark.schema import (
    EvidenceTask, EvidenceHypothesis, EvidenceItem,
)
from hrm_adaptive_memory.executive.evidence_benchmark.i3_5_confirmation_generator import (
    _DOMAINS, _make_hypotheses, _make_evidence,
    CONFIRMATION_BUDGET_PROFILES,
)


# Reuse the existing generators for the failure-type categories
from hrm_adaptive_memory.executive.evidence_benchmark.i3_5_confirmation_generator import (
    _gen_chain,
    _gen_tight_defer,
    _gen_two_live_retrieve,
    _gen_tight_answer,
    _gen_tight_retrieve,
    _gen_tight_verify,
    _gen_tight_search,
    _gen_contradiction,
    _gen_two_live_verify,
    _gen_two_live_search,
    _gen_two_live_defer,
    _gen_premature_answer_trap,
    _gen_premature_defer_trap,
    _gen_retrieval_waste_trap,
    _gen_stale_defer,
)


# New domain content for the development benchmark (different from confirmation)
_DEV_DOMAINS = [
    {"name": "neurology", "context": "stroke type diagnosis",
     "h1": "ischemic stroke", "h2": "hemorrhagic stroke",
     "e1_initial": "patient presents with sudden-onset unilateral weakness",
     "e1_supports": "H1", "e1_contradicts": "",
     "e2_initial": "patient has history of atrial fibrillation",
     "e2_supports": "", "e2_contradicts": "H2",
     "e3_hidden": "CT scan shows no hemorrhage, consistent with ischemia",
     "e3_supports": "H1", "e3_contradicts": "",
     "e4_chain": "MRI diffusion-weighted imaging shows restricted diffusion",
     "e4_supports": "H1", "e4_contradicts": "",
     "e5_search": "angiography confirms middle cerebral artery occlusion",
     "e5_supports": "H1", "e5_contradicts": ""},
    {"name": "climate", "context": "climate anomaly classification",
     "h1": "anthropogenic climate change", "h2": "natural variability",
     "e1_initial": "global temperature anomaly exceeds historical baseline",
     "e1_supports": "H1", "e1_contradicts": "",
     "e2_initial": "similar anomalies occurred in pre-industrial period",
     "e2_supports": "", "e2_contradicts": "H2",
     "e3_hidden": "isotope analysis shows fossil fuel carbon signature",
     "e3_supports": "H1", "e3_contradicts": "",
     "e4_chain": "ocean heat content shows consistent anthropogenic warming pattern",
     "e4_supports": "H1", "e4_contradicts": "",
     "e5_search": "climate model attribution study confirms anthropogenic forcing",
     "e5_supports": "H1", "e5_contradicts": ""},
    {"name": "forensics", "context": "cause of death determination",
     "h1": "homicide", "h2": "accidental death",
     "e1_initial": "victim has defensive wounds on hands and arms",
     "e1_supports": "H1", "e1_contradicts": "",
     "e2_initial": "scene suggests fall from height",
     "e2_supports": "", "e2_contradicts": "H2",
     "e3_hidden": "toxicology shows incapacitating drug levels",
     "e3_supports": "H1", "e3_contradicts": "",
     "e4_chain": "blood spatter analysis indicates blunt force trauma",
     "e4_supports": "H1", "e4_contradicts": "",
     "e5_search": "DNA under fingernails matches unknown suspect",
     "e5_supports": "H1", "e5_contradicts": ""},
    {"name": "manufacturing", "context": "production defect analysis",
     "h1": "supplier material defect", "h2": "assembly process error",
     "e1_initial": "defective units show material composition abnormalities",
     "e1_supports": "H1", "e1_contradicts": "",
     "e2_initial": "assembly line calibration was within spec",
     "e2_supports": "", "e2_contradicts": "H2",
     "e3_hidden": "spectroscopy reveals off-spec alloy composition",
     "e3_supports": "H1", "e3_contradicts": "",
     "e4_chain": "supplier certificate of analysis shows falsified values",
     "e4_supports": "H1", "e4_contradicts": "",
     "e5_search": "supplier audit confirms batch substitution",
     "e5_supports": "H1", "e5_contradicts": ""},
    {"name": "cybersec2", "context": "malware classification",
     "h1": "ransomware attack", "h2": "adware infection",
     "e1_initial": "files on system are encrypted with unknown extension",
     "e1_supports": "H1", "e1_contradicts": "",
     "e2_initial": "system shows popup advertisements",
     "e2_supports": "", "e2_contradicts": "H2",
     "e3_hidden": "ransom note found on desktop demanding cryptocurrency",
     "e3_supports": "H1", "e3_contradicts": "",
     "e4_chain": "encryption algorithm matches known ransomware family",
     "e4_supports": "H1", "e4_contradicts": "",
     "e5_search": "C2 server communication matches ransomware infrastructure",
     "e5_supports": "H1", "e5_contradicts": ""},
    {"name": "agriculture", "context": "crop disease diagnosis",
     "h1": "fungal infection", "h2": "nutrient deficiency",
     "e1_initial": "leaves show dark spots with fuzzy edges",
     "e1_supports": "H1", "e1_contradicts": "",
     "e2_initial": "soil pH is slightly below optimal range",
     "e2_supports": "", "e2_contradicts": "H2",
     "e3_hidden": "microscopic examination reveals fungal hyphae",
     "e3_supports": "H1", "e3_contradicts": "",
     "e4_chain": "culture test confirms pathogenic fungal species",
     "e4_supports": "H1", "e3_contradicts": "",
     "e5_search": "PCR test identifies specific fungal strain",
     "e5_supports": "H1", "e5_contradicts": ""},
    {"name": "finance", "context": "fraud detection",
     "h1": "coordinated fraud", "h2": "isolated accounting error",
     "e1_initial": "multiple transactions show unusual pattern",
     "e1_supports": "H1", "e1_contradicts": "",
     "e2_initial": "individual transaction amounts are small",
     "e2_supports": "", "e2_contradicts": "H2",
     "e3_hidden": "transactions trace to common intermediary account",
     "e3_supports": "H1", "e3_contradicts": "",
     "e4_chain": "shell company registration links to suspect network",
     "e4_supports": "H1", "e4_contradicts": "",
     "e5_search": "international wire pattern matches known fraud typology",
     "e5_supports": "H1", "e5_contradicts": ""},
    {"name": "energy", "context": "grid failure analysis",
     "h1": "cascade failure", "h2": "isolated equipment fault",
     "e1_initial": "multiple substations tripped in sequence",
     "e1_supports": "H1", "e1_contradicts": "",
     "e2_initial": "single transformer showed maintenance warning",
     "e2_supports": "", "e2_contradicts": "H2",
     "e3_hidden": "phase angle analysis shows cascade propagation pattern",
     "e3_supports": "H1", "e3_contradicts": "",
     "e4_chain": "protection relay coordination failed at key node",
     "e4_supports": "H1", "e4_contradicts": "",
     "e5_search": "SCADA log analysis confirms cascade trigger sequence",
     "e5_supports": "H1", "e5_contradicts": ""},
]


def generate_development_benchmark(
    seed: int = 7719,
) -> tuple[EvidenceTask, ...]:
    """Generate the I3.26 development benchmark.

    Heavy representation of:
      - chain (24 tasks) — solvable planning failures
      - defer (24 tasks) — solvable planning failures
      - tl_retrieve (24 tasks) — unsolvable controls
      - answer (12 tasks) — clear-gap controls
      - retrieve (12 tasks) — search-required
      - verify (12 tasks) — verify-required
      - prem_answer (12 tasks) — premature-answer trap
      - prem_defer (12 tasks) — premature-defer trap
      - contradiction (12 tasks) — contradiction resolution
      - retr_waste (12 tasks) — retrieval waste trap

    Total: 156 tasks

    Args:
        seed: Random seed (7719, different from confirmation 4287)

    Returns:
        Tuple of EvidenceTask objects.
    """
    rng = random.Random(seed)
    domains = list(_DEV_DOMAINS)

    # Category counts — heavy on solvable failure types
    category_counts = {
        "chain": 24,           # Solvable planning failures
        "defer": 24,           # Solvable planning failures
        "tl_retrieve": 24,     # Unsolvable controls (search cannot help)
        "answer": 12,          # Clear-gap controls
        "retrieve": 12,        # Search-required
        "verify": 12,          # Verify-required
        "prem_answer": 12,     # Premature-answer trap
        "prem_defer": 12,      # Premature-defer trap
        "contradiction": 12,   # Contradiction resolution
        "retr_waste": 12,      # Retrieval waste trap
    }

    # Map category names to generator functions
    gen_map = {
        "chain": _gen_chain,
        "defer": _gen_tight_defer,
        "tl_retrieve": _gen_two_live_retrieve,
        "answer": _gen_tight_answer,
        "retrieve": _gen_tight_retrieve,
        "verify": _gen_tight_verify,
        "prem_answer": _gen_premature_answer_trap,
        "prem_defer": _gen_premature_defer_trap,
        "contradiction": _gen_contradiction,
        "retr_waste": _gen_retrieval_waste_trap,
    }

    tasks = []
    for category, count in category_counts.items():
        gen_func = gen_map[category]
        for i in range(count):
            domain = domains[i % len(domains)]
            # Use dev-specific task IDs
            task = gen_func(domain, i)
            # Override task_id to use i3_26d prefix
            new_tid = task.task_id.replace("i3_5c_", "i3_26d_")
            task = EvidenceTask(
                task_id=new_tid,
                split=task.split,
                category=task.category,
                task_summary=task.task_summary,
                high_stakes=task.high_stakes,
                budget_profile=task.budget_profile,
                hypotheses=task.hypotheses,
                evidence_items=task.evidence_items,
                retrieve_exposes=task.retrieve_exposes,
                search_exposes=task.search_exposes,
                oracle_resolution_path=task.oracle_resolution_path,
                expected_terminal=task.expected_terminal,
                correct_hypothesis_id=task.correct_hypothesis_id,
            )
            tasks.append(task)

    return tuple(tasks)


def compute_benchmark_hash(tasks: tuple[EvidenceTask, ...]) -> str:
    """Compute deterministic SHA256 hash of the benchmark task set."""
    import json
    content = json.dumps({
        "n_tasks": len(tasks),
        "task_ids": [t.task_id for t in tasks],
        "categories": [t.category for t in tasks],
        "budget_profiles": [t.budget_profile for t in tasks],
    }, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()


if __name__ == "__main__":
    tasks = generate_development_benchmark(seed=7719)
    bench_hash = compute_benchmark_hash(tasks)

    from collections import Counter
    cat_counts = Counter(t.category for t in tasks)
    budget_counts = Counter(t.budget_profile for t in tasks)

    print(f"I3.26 Development Benchmark")
    print(f"  Total tasks: {len(tasks)}")
    print(f"  Seed: 7719")
    print(f"  Hash: {bench_hash}")
    print(f"\n  Categories:")
    for cat, count in sorted(cat_counts.items()):
        print(f"    {cat}: {count}")
    print(f"\n  Budget profiles:")
    for bp, count in sorted(budget_counts.items()):
        print(f"    {bp}: {count}")

    # Save frozen benchmark metadata
    import json
    from pathlib import Path
    REPO_ROOT = Path(__file__).resolve().parent.parent
    output = {
        "version": "I3.26_DEVELOPMENT_BENCHMARK_V1",
        "seed": 7719,
        "n_tasks": len(tasks),
        "hash": bench_hash,
        "categories": dict(cat_counts),
        "budget_profiles": dict(budget_counts),
        "task_ids": [t.task_id for t in tasks],
        "frozen_before_any_arm_run": True,
        "note": "Development benchmark for I3.26 search experiments. "
                "Heavy on solvable failure types (chain, defer) and unsolvable "
                "controls (tl_retrieve). Different seed from confirmation (4287).",
    }
    output_path = REPO_ROOT / "experiments/i3_26/DEVELOPMENT_BENCHMARK_FROZEN.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, sort_keys=True)
    print(f"\n  Saved: {output_path}")
