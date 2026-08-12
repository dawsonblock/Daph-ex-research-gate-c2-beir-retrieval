# DAPH-HRM Adaptive Memory Control Plane v3.7.1

> Pretrained-compatible adaptive computation with a physically ordered four-level effort hierarchy, plus a staged retrieval-and-memory research pipeline built on **HRM-Text-1B**.

[![Tests](https://img.shields.io/badge/tests-607%20passed-brightgreen)](#tests)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](#install)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](#license)
[![HRM--Text--1B](https://img.shields.io/badge/HRM--Text--1B-sapientinc-orange)](https://huggingface.co/sapientinc/HRM-Text-1B)

---

## What this repository does

This is a **gate-structured research pipeline** that measures whether each
mechanism in an adaptive memory stack earns its place before it is allowed
into the canonical build. Every gate has a frozen protocol, pre-declared
promotion thresholds, and immutable evidence bundles with SHA-256 hashing.

The pipeline is built on **HRM-Text-1B** (sapientinc), a 1B-parameter
PrefixLM reasoning model, and tests whether retrieval, identity resolution,
evidence selection, and bridge acquisition each add measurable value
before being composed into the integrated pipeline.

```
question → subject-preserving query → BM25+BGE retrieval
         → I3 identity resolution → S2c structural selection
         → bounded evidence packet → HRM-Text-1B → verified answer
```

---

## Gate status dashboard

| Gate | Status | Key result |
|------|--------|------------|
| Gate A0 — PASSED | ✅ PASS | HRM-Text-1B uses correctly supplied external evidence |
| **B** Single-pass retrieval | ✅ PASS | BM25 dominates tested dense representations on identifiable corpus |
| **C0** Controlled iterative retrieval | ⚠️ MECHANISM_SUCCESS, PROMOTION_BLOCKED | Reaches oracle ceiling on v2 but cannot promote under its own statistical rule |
| **C1** Structural generalization | ❌ FAIL | Inert out of distribution (0.080 vs 0.764) — entity chaining is lexical, not inferential |
| **C2-R** Retrieval coverage | ✅ PASS | Candidate-generation policy frozen (P2 RRF bm25+bge k=10) |
| **C2-S** Evidence selection | ✅ PASS | Structural selection captures 30% of oracle gap on ID partition |
| **C3** Surface identity resolution | ✅ MECHANISM_SUCCESS | I3 identity-record retrieval resolves surface defect with 92% accuracy, 0% false resolution |
| **C4** Integrated memory pipeline | ✅ CERTIFIED (development) | H100 certification: +0.2000 quality delta (family CI [+0.1354, +0.2604], threshold +0.15); 17/17 gates passed |
| **C4-BRIDGE** Runtime bridge acquisition | ❌ NEGATIVE RESULT | No runtime bridge mechanism beats one-pass baseline (B0 CES=0.783 vs B2 CES=0.775) |
| **V2A** External background verification | ✅ QUALIFIED | Immutable evidence + lineage + deterministic verification; 1M-event pressure/replay PASS, 10/10 adversarial PASS, recorded live/offline smoke PASS |
| **D–N** Downstream gates | 🔒 BLOCKED | Pending untouched qualification/OOD splits and Gate D decision |

Full machine-readable state: [`RESEARCH_STATUS.json`](RESEARCH_STATUS.json)

V2A's qualification is deliberately narrow: DAPH can capture immutable
external evidence, preserve declared lineage, deterministically verify the
supported exact-field source classes, survive retries/replay/tampering, and
maintain an auditable current verification state. It is not a claim of general
truth determination, arbitrary literature understanding, verification-aware
retrieval, or improved answer accuracy.

---

## C4 integrated pipeline — current focus

The C4 gate composes all qualified mechanisms into a single pipeline and
measures whether the composition adds value beyond any single component.

### Seven-arm ablation

| Arm | Query | Retrieval | Identity | Selector | What it tests |
|-----|-------|-----------|----------|----------|---------------|
| C4_0 | original | BM25 | off | S0 | Baseline (no memory stack) |
| C4_1 | subject-preserving | BM25 | off | S0 | Query formulation only |
| C4_2 | subject-preserving | BM25+BGE | off | S0 | + Dense retrieval |
| C4_3 | subject-preserving | BM25+BGE | I3 | S0 | + Identity resolution |
| C4_4 | subject-preserving | BM25+BGE | I3 | S2c | + Structural selection |
| C4_5 | subject-preserving | BM25+BGE | I3 | oracle | Oracle selector ceiling |
| C4_6 | subject-preserving | BM25+BGE | oracle | oracle_evidence | Full oracle ceiling |

### H100-certified development scores

The fail-closed C4 development certification passed on `NVIDIA H100 80GB HBM3`
with CUDA 12.8, `float16`, and `HRM_BATCH_SIZE=1`. The certified bundle is
[`evidence/gate_c4/full/development/certification/CERTIFICATION.json`](evidence/gate_c4/full/development/certification/CERTIFICATION.json);
its `BUNDLE.sha256` covers the certificate, environment lock, source snapshot,
receipts, and analysis.

| Arm | Quality | Accuracy |
|-----|---------|----------|
| C4_0 | 0.1625 | 0.1667 |
| C4_1 | 0.1708 | 0.1583 |
| C4_2 | 0.2125 | 0.1667 |
| C4_3 | 0.2125 | 0.1667 |
| C4_4 | 0.3625 | 0.2917 |
| C4_5 | 0.7917 | 0.8833 |
| C4_6 | 0.9542 | 0.9083 |

**Primary delta (C4_4 − C4_0): +0.2000**, family bootstrap CI [+0.1354, +0.2604], threshold +0.15 — **PASS**. All 17 certification gates passed; this certifies the development split only, not the later qualification or OOD splits.

### C4-BRIDGE negative result

Iterative retrieval was implemented and tested. No runtime bridge mechanism
beats the one-pass baseline:

| Mechanism | CES | Recall | Second-pass rate |
|-----------|-----|--------|------------------|
| B0 (one-pass baseline) | 0.783 | 0.925 | 0% |
| B1 (heuristic) | 0.767 | 0.917 | 34.2% |
| B2 (relation parser) | 0.775 | 0.921 | 35.0% |
| B3 (connectivity) | 0.775 | 0.921 | 35.0% |
| B4 (oracle bridge) | 0.933 | 0.975 | 60.0% |

**Decision: iterative retrieval is disabled.** C4 uses the one-pass pipeline.
Oracle bridge headroom exists (B4 CES=0.933) but no runtime mechanism captures it.

### Conformance validation (7 gates)

Every run passes all 7 conformance gates before HRM execution:

1. ✅ No oracle leakage
2. ✅ Arm parity (arms differ only where expected)
3. ✅ Selected IDs in pool
4. ✅ Packet budgets
5. ✅ Q3 query formulation
6. ✅ Merge provenance
7. ✅ Causal parity (each mechanism change causes only expected downstream effects)

---

## Quick start

### Install

```bash
pip install -e .
# For HRM execution (requires transformers >= 5.9):
pip install -e ".[hrm]"
```

### Run tests

```bash
python -m pytest -q
# 607 passed, 2 skipped
```

### Run the C4 pipeline

```bash
# CPU-only dry run (validates all 7 conformance gates, ~15 seconds)
python scripts/run_gate_c4.py dry-run

# C4-BRIDGE gate (no HRM, ~2 seconds)
python scripts/run_gate_c4_bridge.py

# HRM smoke test (3 tasks × 7 arms, ~2-3 minutes on GPU)
python scripts/run_gate_c4.py smoke

# Full conformant development run (120 tasks × 7 arms, ~15-25 min on T4 GPU)
python scripts/run_gate_c4.py full --split development

# Analyze results (family/cluster/template CIs, task flips, gap capture)
python scripts/analyze_gate_c4.py --dir evidence/gate_c4/full/development

# Diagnose S2c selection behavior
python scripts/diagnose_c4_composition.py
```

### Run certification on a locked GPU

The full C4 v2_1 fail-closed requalification:

1. Go to [Google Colab](https://colab.research.google.com/)
2. **Runtime → Change runtime type → GPU + High-RAM**
3. **File → Upload notebook** → upload [`notebooks/colab_c4_requalify.ipynb`](notebooks/colab_c4_requalify.ipynb)
4. **Runtime → Run all**

Or run the scripts directly:

```python
!git clone https://github.com/dawsonblock/Daph-ex-research-gate-c2-beir-retrieval.git
%cd Daph-ex-research-gate-c2-beir-retrieval
!pip install -e ".[hrm]"
!python scripts/c4_freeze_environment.py --note 'captured certifying runtime'
!python scripts/colab_c4_requalify.py
```

**Expected time depends on the locked GPU** (the H100 development certification completed in roughly 10 minutes; CPU takes hours).

The run is resumable, and fail-closed: it aborts on any protocol abort
condition rather than producing a result that cannot be certified. It ends by
writing `CERTIFICATION.json`, where `VALID_RUN` is the conjunction of every
derived gate. The archive is named `UNCERTIFIED_*` unless `VALID_RUN` is true.

The notebook contains **no scientific logic** — it only invokes the tested
scripts. See [`notebooks/README.md`](notebooks/README.md) for why, and for the
retired execution paths under `notebooks/superseded/`.

---

## Architecture

### C4 pipeline stages

```
┌─────────────────────────────────────────────────────────────────┐
│  C4 Integrated Memory Pipeline                                   │
│                                                                  │
│  Question                                                        │
│    │                                                             │
│    ▼                                                             │
│  Query Stage (Q3: subject-preserving)                            │
│    │  Keeps subject entity, includes target relation             │
│    ▼                                                             │
│  Retrieval Stage (BM25 + BGE RRF fusion, k=10)                  │
│    │  Pinned: BAAI/bge-small-en-v1.5, CLS pooling               │
│    ▼                                                             │
│  Identity Stage (I3: identity-record resolution)                 │
│    │  Reads surface→canonical mappings from evidence             │
│    │  EXACT / RESOLVED / AMBIGUOUS / UNRESOLVED                  │
│    ▼                                                             │
│  Selection Stage (S2c: structural selection)                     │
│    │  Prefers identity/link/value records over dead-ends         │
│    │  Falls back to S0 (BM25 score) when no structure            │
│    ▼                                                             │
│  Packet Stage (bounded evidence packet)                          │
│    │  Precision packing with hash for provenance                 │
│    ▼                                                             │
│  HRM-Text-1B (PrefixLM reasoning, max 64 new tokens)            │
│    │                                                             │
│    ▼                                                             │
│  Verified Answer (shared verifier)                               │
└─────────────────────────────────────────────────────────────────┘
```

### Key modules

| Module | Purpose |
|--------|---------|
| `hrm_adaptive_memory/c4/arms.py` | 7-arm ablation definitions |
| `hrm_adaptive_memory/c4/query_stage.py` | Q3 subject-preserving query formulation |
| `hrm_adaptive_memory/c4/retrieval_stage.py` | BM25+BGE RRF fusion |
| `hrm_adaptive_memory/c4/identity_stage.py` | I3 identity-record resolution |
| `hrm_adaptive_memory/c4/selection_stage.py` | S2c structural selection |
| `hrm_adaptive_memory/c4/packet_stage.py` | Bounded evidence packet with precision packing |
| `hrm_adaptive_memory/c4/parity.py` | 7 conformance validation gates |
| `hrm_adaptive_memory/c4/provenance.py` | Immutable result hashing (manifest + RESULTS.sha256) |
| `hrm_adaptive_memory/c4/relational_state.py` | V4 link-record parser for relational graph |
| `hrm_adaptive_memory/c4/bridge_extraction.py` | Bridge extraction (retained for provenance, disabled in pipeline) |
| `hrm_adaptive_memory/hrm/model.py` | HRM-Text-1B adapter (PrefixLM, SDPA attention) |
| `hrm_adaptive_memory/evaluation/verifiers.py` | Shared verifier (exact, symbolic, enum, boolean, JSON) |

### Effort hierarchy (legacy architecture)

`QwenExFusionModel` executes distinct compute graphs:

| Mode | Execution | Intent |
|------|-----------|--------|
| E0 | first 50% of layers | Cheapest approximation |
| E1 | first 75% of layers | Intermediate approximation |
| E2 | all layers | Full pretrained anchor |
| E3 | full backbone + bounded recurrent refinement | Additional compute for difficult inputs |

Deterministic `EffortComputeReceipt` accounting proves:
`C(E0) < C(E1) < C(E2) < C(E3)` and `C_norm(E2) = 1.0`.

---

## Repository structure

```
hrm_adaptive_memory/          # Active research implementation
  c4/                         # C4 integrated pipeline (current focus)
    arms.py                   #   7-arm ablation
    query_stage.py            #   Q3 subject-preserving query
    retrieval_stage.py        #   BM25+BGE RRF fusion
    identity_stage.py         #   I3 identity resolution
    selection_stage.py        #   S2c structural selection
    packet_stage.py           #   Bounded evidence packet
    parity.py                 #   7 conformance gates
    provenance.py             #   Immutable result hashing
    relational_state.py       #   V4 link-record parser
    bridge_extraction.py      #   Bridge extraction (disabled)
  hrm/                        # HRM-Text-1B adapter
  retrieval/                  # BM25 + BGE embedder
  evaluation/                 # Shared verifiers

scripts/                      # 67 executable scripts
  run_gate_c4.py              #   C4 harness (dry-run, smoke, full)
  run_gate_c4_bridge.py       #   C4-BRIDGE qualification gate
  analyze_gate_c4.py          #   C4 analyzer (CIs, flips, gap capture)
  diagnose_c4_composition.py  #   S2c selection diagnostic
  colab_c4_requalify.py       #   Authoritative C4 run (Colab T4, fail-closed)
  certify_c4_run.py           #   VALID_RUN = conjunction of derived gates
  c4_freeze_environment.py    #   Environment lock freeze/verify
  c4_void_packets.py          #   Void with proof, preserve the data

notebooks/                    # Launchers only — no scientific logic
  colab_c4_requalify.ipynb    #   Invokes scripts/colab_c4_requalify.py
  superseded/                 #   Retired execution paths (provenance only)

configs/                      # Frozen protocol configurations
  gate_c4_protocol_v2_1.json  #   C4 ACTIVE protocol (single ordering spec)
  gate_c4_protocol_v2.json    #   C4 v2 (superseded: contradictory ordering)
  gate_c4_protocol.json       #   C4 v1 (superseded)
  c4_requirements.lock        #   C4 environment lock (null pins fail closed)
  gate_c3_protocol.json       #   C3 protocol
  gate_c2_protocol.json       #   C2 protocol

evidence/                     # Immutable evidence bundles
  gate_c4/                    #   C4 results (dry_run, smoke, full, bridge)
  gate_c3/                    #   C3 results
  gate_c2/                    #   C2 results

tests/                        # 607 tests (unit + integration)
data/hrm/controlled_gate_a_v4/# Frozen task corpus (120 dev + 120 qual + 120 OOD)
```

---

## Evidence integrity

Every result bundle is cryptographically hashed for reproducibility:

- **Protocol hash**: SHA-256 of the frozen protocol JSON, recorded in manifest
- **Task corpus hash**: SHA-256 of the task corpus, recorded in manifest
- **Evidence corpus hash**: SHA-256 of the evidence corpus, recorded in manifest
- **RESULTS.sha256**: Per-file hashes of all result JSONL files
- **Git commit**: Recorded in manifest for full traceability

```json
{
  "protocol_sha256": "e4e803a2...",
  "task_corpus_sha256": "71d13609...",
  "evidence_corpus_sha256": "de6d710f...",
  "git_commit": "9960806...",
  "validation": {
    "no_leakage": true,
    "parity": true,
    "selected_in_pool": true,
    "packet_budgets": true,
    "q3_query_formulation": true,
    "merge_provenance": true,
    "causal_parity": true
  }
}
```

---

## Metrics

All metrics are defined in the [protocol](configs/gate_c4_protocol.json) under
`metric_definitions`:

| Metric | Definition |
|--------|------------|
| **Quality** | Partial-credit: 1.0 for correct, 0.0 for incorrect (shared verifier) |
| **CES** | Complete Evidence Set: 1.0 if all required evidence is selected, else 0.0 |
| **CSR** | Complete Set Retention: fraction of required evidence selected |
| **S2c live rate** | Fraction of tasks where S2c selector activated |
| **SGC** | Selector Gap Capture: (C4_4 − C4_3) / (C4_5 − C4_3) |
| **OGC** | Oracle Gap Capture: (C4_5 − C4_0) / (C4_6 − C4_0) |
| **Primary delta** | C4_4 quality − C4_0 quality, with grouped bootstrap CI |
| **Family CI** | Bootstrap CI resampling whole families |
| **Cluster CI** | Bootstrap CI resampling whole source clusters |
| **Template CI** | Bootstrap CI resampling whole templates |
| **Task flip** | Per-task quality change: improve / regress / unchanged |

---

## Promotion criteria (frozen before qualification)

1. C4_4 quality > C4_0 by at least +0.15 absolute on development
2. No material canonical/abbreviation regression > 0.05
3. Alias and description both improve over C4_0
4. FalseResolutionRate ≤ 0.02
5. C4_4 materially reduces the C4_5 oracle-selector gap
6. All runtime payloads pass oracle-leak validation
7. Candidate/evidence budgets remain fixed
8. Grouped bootstrap lower bound for primary quality delta > 0
9. No post-hoc mechanism/config changes after freeze

---

## Key reports

- [Gate C4 Protocol](GATE_C4_PROTOCOL.md) — frozen protocol with metric definitions
- [Gate C3 Report](GATE_C3_REPORT.md) — I3 identity-record resolution (MECHANISM_SUCCESS)
- [Gate C1 Report](GATE_C1_REPORT.md) — structural generalization failure
- [Gate B Report](GATE_B_REPORT.md) — single-pass retrieval qualification
- [Research Status](RESEARCH_STATUS.json) — full machine-readable state
- [Changelog](CHANGELOG.md) — version history

---

## Repository ownership

| Package | Status |
|---------|--------|
| `hrm_adaptive_memory/` | **ACTIVE_HRM_RESEARCH** — canonical research implementation |
| `daph/` | LEGACY_QWEN_EXFUSION — frozen, tests maintained |
| `daph_metareasoner/` | LEGACY_METAREASONING — frozen, tests maintained |

---

## Testing policy

Tests assert structured state, never exact strings against narrative prose.
The shared verifier (`hrm_adaptive_memory.evaluation.verifiers`) is the single
source of truth for answer correctness across all gates.

```bash
python -m pytest -q
# 607 passed, 2 skipped
```

---

## License

MIT. See [LICENSE](LICENSE) for details.

---

## Citation

If you use this work, cite the repository and the HRM-Text-1B model:

```bibtex
@misc{daph-hrm-2024,
  title  = {DAPH-HRM Adaptive Memory Control Plane},
  author = {Dawson Block},
  year   = {2024},
  url    = {https://github.com/dawsonblock/Daph-ex-research-gate-c2-beir-retrieval}
}
```

---

## Acknowledgements

Inspired by architectural principles from Kimi K3, adapted for smaller
experimental systems. HRM-Text-1B by sapientinc. BGE embeddings by BAAI.
