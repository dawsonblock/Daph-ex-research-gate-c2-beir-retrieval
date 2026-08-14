# Changelog

All notable changes to the DAPH research repository are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [V2B-I3.4.1] — Scientific Protocol Repair (In Progress)

### Added
- **Pre-repair baseline** (`v2b-i3.4-pre-scientific-repair` tag): Permanent
  baseline at commit `8386fcd` with 12 known audit findings.
- **IG/DG/TR decomposition** (`i3_4_scientific_scoring.py`): Restored the
  frozen I3.2.2 definitions. TR = IG + DG algebraically. Individual task
  contributions are NOT clamped. Non-negativity holds at the aggregate level.
- **Scientific Criteria V2** (`v2b_i3_4_scientific_criteria_v2.json`):
  Primary hypothesis uses decision_gap (DG), not trajectory_regret (TR).
  Primary success criterion: LCB_95(ΔDG) > 0. Four distinct claims.
  Prohibition on substituting TR for DG. 23 validity gates (G01-G23).
- **Paired bootstrap** (`i3_4_statistical_analysis.py`): Task-level
  resampling, 10000 iterations, 95% CI, frozen seed.
- **Topology-cluster bootstrap**: Cluster-level resampling for structural
  held-out inference (51 clusters, not 150 tasks).
- **FrozenGenerationConfig**: Explicit thinking_mode='disabled',
  response_format='json_object', all parameters explicitly bound.
- **RetryPolicy**: Explicit retry matrix (only transport-like failures
  retried). CallReceipt with append-only audit trail per attempt.
- **Counterbalanced paired scheduler**: Deterministic order from
  SHA256(experiment_id || task_id). Adjacent pair calls. Fingerprint
  invalidation rule.
- **Supersession notice** for Scientific Criteria V1 (preserved unmodified).

### Changed
- README updated to reflect I3.3.3, I3.4, and I3.4.1 status.
- Architecture diagram updated to include I3.3.3, I3.4, and I3.4.1.

### Notes
- Scientific Criteria V1 is SUPERSEDED but preserved as historical evidence.
- The I3.3.2 benchmark remains immutable (750 tasks, splits, oracles, etc.).
- No held-out model evaluation has been run.
- Branch: `i3.4.1-scientific-protocol-repair`

## [V2B-I3.4] — Pinned Model Executive (Engineering Implemented)

### Added
- **PinnedModelController**: Condition-agnostic executive controller backed
  by a pinned language model. No if/aware/else branching.
- **DeepSeek backend**: OpenAI-compatible API with retry logic.
- **Strict JSON decoder**: Fail-closed output validation with 7-action
  vocabulary. Rejects malformed JSON, unknown actions, missing fields.
- **Frozen system prompt**: No benchmark-specific heuristics or condition
  identity leakage.
- **Controller identity**: Binds model, prompt, serializer, decoder,
  controller code, backend code, and generation settings.
- **Development metrics**: Model valid-action rate, malformed-output rate,
  latency, token usage, action distribution, backend error tracking.
- **Scientific Criteria V1** (later superseded by V2).

### Fixed
- Exception handling in `PinnedModelController.choose()`: API errors
  fail-closed with `BACKEND_ERROR_PROPOSAL` instead of crashing.
- Broken condition-branching test: regex patterns now use `re.search()`.
- `_extract_json` replaced with `_extract_json_candidates` for reasoning
  text with braces.
- `assert_no_condition_leakage` now checks string values, not just keys.
- Mutable shared list aliases in NULL_* constants fixed.
- Config consistency: added `relevant_memories` to canonical_nulls,
  removed unused generation settings.

## [V2B-I3.3.3] — Release/Qualification Hardening

### Added
- Receipt-based V2A provenance boundary.
- Fail-closed V2A boundary tests.
- Clean-checkout CI on Python 3.10, 3.11, and 3.12.
- Frozen I3.3.2 baseline manifest.
- Immutable qualification bundle.
- Exhaustive latent and sequential oracle regeneration.
- Benchmark-closure reproduction.
- Structural-depth/difficulty preregistration.
- Branch protection on `main` and `v2b-infrastructure`.

### Status
- `QUALIFIED_FROZEN_BENCHMARK` — benchmark qualification only, not
  executive qualification.

## [V2B-I3.3.2] — Scientific Split (Frozen Benchmark)

### Added
- 750 frozen tasks across development, validation, and held-out splits.
- Behavior-derived topology isolation.
- Four Q-margin difficulty bands.
- Latent oracle tables (750 tables, 227248 reachable states).
- Seven sequential observable oracle sets (one per condition/mask).
- Task-uniform information gap values per condition.
- Structural held-out composition: 150 tasks, 51 topologies.
