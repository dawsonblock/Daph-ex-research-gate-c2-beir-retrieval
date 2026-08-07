"""Dedicated nondeterminism tests for the C4 pre-HRM pipeline.

Phase 8 of the C4 determinism repair. These tests convert the qualification
principles into permanent CI tests.

Test A — Hash-seed invariance
Test B — Candidate permutation invariance
Test C — Equal-score tie test
Test D — Repeated-process replay (subprocess)
Test E — Serialization stability
Test F — Cross-run full pre-HRM replay
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from hrm_adaptive_memory.c4.provenance import (
    canonical_packet_hash, build_canonical_packet, hash_text)
from hrm_adaptive_memory.c4.packet_ordering import (
    order_packet, canonical_membership_hash, canonical_order_hash)
from hrm_adaptive_memory.retrieval_bench.selectors.chain import (
    s2c_chain_plus_relation, candidate_sort_key, _quantize, ROLE_PRIORITY)


# --- Test A: Hash-seed invariance ---

class TestHashSeedInvariance:
    """Spawn subprocesses with different PYTHONHASHSEED and require identical output."""

    @pytest.mark.parametrize("seed_a,seed_b", [(0, 42), (17, 12345), (1, 2)])
    def test_selector_output_identical_across_seeds(self, seed_a, seed_b):
        """S2c selector must produce identical output regardless of hash seed."""
        candidates = [
            {"document_id": "task-1/identity"},
            {"document_id": "task-1/value"},
            {"document_id": "task-1/link"},
            {"document_id": "task-2/identity"},
            {"document_id": "task-2/value"},
            {"document_id": "distractor/dead-end-0"},
        ]
        texts = {
            "task-1/identity": "Alpha entity refers to Alpha Prime.",
            "task-1/value": "Alpha Prime ownership tier is Tier 3.",
            "task-1/link": "Alpha Prime is assigned to Beta unit.",
            "task-2/identity": "Beta entity refers to Beta Prime.",
            "task-2/value": "Beta Prime status is active.",
            "distractor/dead-end-0": "Unrelated content about Gamma.",
        }
        question = "Which ownership tier is held by Alpha entity?"

        # Run in subprocess with different seeds
        code = f"""
import sys
sys.path.insert(0, '.')
from hrm_adaptive_memory.retrieval_bench.selectors.chain import s2c_chain_plus_relation
candidates = {candidates!r}
texts = {texts!r}
question = {question!r}
result = s2c_chain_plus_relation(candidates, budget=6, question=question, texts=texts)
import json
print(json.dumps(result))
"""
        env_a = {**os.environ, "PYTHONHASHSEED": str(seed_a)}
        env_b = {**os.environ, "PYTHONHASHSEED": str(seed_b)}

        result_a = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, env=env_a, timeout=30)
        result_b = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, env=env_b, timeout=30)

        assert result_a.returncode == 0, f"seed {seed_a} failed: {result_a.stderr}"
        assert result_b.returncode == 0, f"seed {seed_b} failed: {result_b.stderr}"
        assert result_a.stdout == result_b.stdout, \
            f"Selector output differs between seeds {seed_a} and {seed_b}"


# --- Test B: Candidate permutation invariance ---

class TestCandidatePermutationInvariance:
    """Shuffling the input candidate order must not change S2c output."""

    def test_permutation_invariance(self):
        import random
        candidates = [
            {"document_id": f"task-{i}/value"} for i in range(10)
        ]
        candidates.extend([
            {"document_id": "task-0/identity"},
            {"document_id": "task-0/link"},
        ])
        texts = {c["document_id"]: f"Content for {c['document_id']}" for c in candidates}
        texts["task-0/identity"] = "Alpha refers to Alpha Prime."
        texts["task-0/link"] = "Alpha Prime is linked to Beta."
        texts["task-0/value"] = "Alpha Prime value is 42."
        question = "What value is held by Alpha?"

        base_result = s2c_chain_plus_relation(
            candidates, budget=6, question=question, texts=texts)

        for seed in range(50):
            shuffled = candidates[:]
            random.Random(seed).shuffle(shuffled)
            result = s2c_chain_plus_relation(
                shuffled, budget=6, question=question, texts=texts)
            assert result == base_result, \
                f"Permutation seed {seed} produced different output"


# --- Test C: Equal-score tie test ---

class TestEqualScoreTie:
    """Candidates with identical scores must be ordered by the tie-break chain."""

    def test_tie_break_by_record_id(self):
        """When all scores are equal, record_id breaks ties."""
        ids = ["task-3/value", "task-1/value", "task-2/value"]
        ordered = order_packet(ids)
        assert ordered == ["task-1/value", "task-2/value", "task-3/value"]

    def test_tie_break_by_role_priority(self):
        """When scores are equal, role priority breaks ties."""
        ids = ["task-1/value", "task-1/identity", "task-1/link"]
        ordered = order_packet(ids)
        # identity (10) < link (20) < value (50)
        assert ordered == ["task-1/identity", "task-1/link", "task-1/value"]

    def test_candidate_sort_key_total_order(self):
        """No two different records compare equal under candidate_sort_key."""
        key1 = candidate_sort_key(1.0, 0.5, "task-1/value")
        key2 = candidate_sort_key(1.0, 0.5, "task-2/value")
        assert key1 != key2

    def test_quantization_eliminates_fp_noise(self):
        """Scores that differ only at machine precision are equal."""
        assert _quantize(1.0 + 1e-15) == _quantize(1.0)
        assert _quantize(0.1 + 0.2) == _quantize(0.3)


# --- Test D: Repeated-process replay ---

class TestRepeatedProcessReplay:
    """Run the selector in 5 fresh Python processes, require same SHA-256."""

    def test_repeated_process_identical(self):
        candidates = [
            {"document_id": "task/identity"},
            {"document_id": "task/value"},
            {"document_id": "task/link"},
            {"document_id": "distractor/dead-end-0"},
        ]
        texts = {
            "task/identity": "Alpha refers to Alpha Prime.",
            "task/value": "Alpha Prime tier is 3.",
            "task/link": "Alpha Prime assigned to Beta.",
            "distractor/dead-end-0": "Unrelated gamma content.",
        }
        question = "Which tier is held by Alpha?"

        code = f"""
import sys, json, hashlib
sys.path.insert(0, '.')
from hrm_adaptive_memory.retrieval_bench.selectors.chain import s2c_chain_plus_relation
candidates = {candidates!r}
texts = {texts!r}
question = {question!r}
result = s2c_chain_plus_relation(candidates, budget=4, question=question, texts=texts)
h = hashlib.sha256(json.dumps(result).encode()).hexdigest()
print(h)
"""

        hashes = []
        for i in range(5):
            env = {**os.environ, "PYTHONHASHSEED": str(i)}
            result = subprocess.run(
                [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=30)
            assert result.returncode == 0, f"Process {i} failed: {result.stderr}"
            hashes.append(result.stdout.strip())

        assert len(set(hashes)) == 1, f"Different hashes across processes: {hashes}"


# --- Test E: Serialization stability ---

class TestSerializationStability:
    """Serialize → parse → serialize must produce the same bytes."""

    def test_packet_serialization_stable(self):
        packet = build_canonical_packet(
            task_id="task-001",
            query_hash="abc123",
            canonical_subject="Alpha Prime",
            candidate_pool_hash="def456",
            selector_policy_id="s2c",
            ordered_selected_ids=["task/identity", "task/value"],
            ordered_text_sha256=["hash1", "hash2"],
        )
        h1 = canonical_packet_hash(packet)
        # Parse and re-serialize
        parsed = json.loads(json.dumps(packet, sort_keys=True))
        h2 = canonical_packet_hash(parsed)
        assert h1 == h2

    def test_membership_hash_stable(self):
        h1 = canonical_membership_hash(["a", "b", "c"])
        h2 = canonical_membership_hash(["a", "b", "c"])
        assert h1 == h2

    def test_order_hash_stable(self):
        h1 = canonical_order_hash(["a", "b", "c"])
        h2 = canonical_order_hash(["a", "b", "c"])
        assert h1 == h2


# --- Test F: Cross-run full pre-HRM replay ---

class TestCrossRunReplay:
    """Run the pre-HRM pipeline twice in separate processes, require identical hashes."""

    @pytest.mark.skipif(
        not Path("data/hrm/controlled_gate_a_v4/development/oracle_tasks.jsonl").exists(),
        reason="Corpus not available"
    )
    def test_full_pre_hrm_replay_5_tasks(self):
        """Run 5 tasks through the full pre-HRM pipeline in 2 processes."""
        code = """
import sys, json, hashlib
sys.path.insert(0, '.')
from scripts.run_gate_c4 import run_pre_hrm_stages, _load_split, _to_index_records, ARMS
from hrm_adaptive_memory.c4.provenance import canonical_packet_hash, build_canonical_packet, hash_text

tasks, evidence, texts = _load_split('development')
records = _to_index_records(evidence)
arm = ARMS['C4_4']

hashes = []
for task in tasks[:5]:
    r = run_pre_hrm_stages(task, arm, records, texts)
    selected = list(r.selection.selected_ids)
    packet = build_canonical_packet(
        task_id=task['task_id'],
        query_hash=hashlib.sha256(r.query.rendered_query.encode()).hexdigest(),
        canonical_subject=r.identity.canonical or '',
        candidate_pool_hash=hashlib.sha256(
            json.dumps(list(r.retrieval.candidate_ids), separators=(',',':')).encode()
        ).hexdigest(),
        selector_policy_id=r.selection.selector,
        ordered_selected_ids=selected,
        ordered_text_sha256=[hash_text(texts.get(eid, '')) for eid in selected],
    )
    hashes.append(canonical_packet_hash(packet))

print(json.dumps(hashes))
"""

        env_a = {**os.environ, "PYTHONHASHSEED": "0"}
        env_b = {**os.environ, "PYTHONHASHSEED": "42"}

        result_a = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True,
            env=env_a, timeout=120, cwd=str(Path(__file__).resolve().parents[2]))
        result_b = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True,
            env=env_b, timeout=120, cwd=str(Path(__file__).resolve().parents[2]))

        assert result_a.returncode == 0, f"Process A failed: {result_a.stderr[-300:]}"
        assert result_b.returncode == 0, f"Process B failed: {result_b.stderr[-300:]}"

        hashes_a = json.loads(result_a.stdout)
        hashes_b = json.loads(result_b.stdout)
        assert hashes_a == hashes_b, "Packet hashes differ across hash seeds"
