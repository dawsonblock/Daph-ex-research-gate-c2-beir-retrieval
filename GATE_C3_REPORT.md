# Gate C3 — Surface Identity Resolution

## Final Report

**Gate:** C3_SURFACE_IDENTITY_RESOLUTION  
**Status:** MECHANISM_SUCCESS_I3_IDENTITY_RECORD_RESOLUTION  
**Protocol version:** v2_pre_measurement_amended  
**Corpus:** controlled_gate_c3_v1 (lighthouses, sixth vocabulary)  

## 1. Protocol and frozen configuration

Gate C3 tests whether truncated, alias, and description mentions can be resolved into runtime canonical entities well enough that the qualified structural selector (S2c) can operate on surface tasks.

The protocol was frozen at `configs/gate_c3_protocol.json` before any measurement. Three amendments were applied pre-measurement:

1. **False-resolution metrics** (amendment #7): Added CorrectAnchorRate, WrongAnchorRate, FalseResolutionRate, AmbiguousResolutionRate, UnresolvedRate, plus a resolution contract (EXACT/RESOLVED/AMBIGUOUS/UNRESOLVED) and ResolutionUtility composite.
2. **Sixth-corpus disjointness** (amendment #8): Required zero overlap on entity surfaces, aliases, descriptions, vocabulary, and source clusters — not just noun-family names.
3. **I1 negative-control classification** (amendment #9): Explicitly classified I1 as a NEGATIVE_CONTROL rung, with the prediction that I1 ≈ I0 on truncation failures.

## 2. Data lineage

| Artifact | Path | Status |
|---|---|---|
| C3 protocol | `configs/gate_c3_protocol.json` | Frozen v2_pre_measurement_amended |
| Sixth-vocabulary corpus | `data/hrm/controlled_gate_c3_v1/` | Frozen, all disjointness overlaps = 0 |
| I0/I1/I2/I3 receipts | `evidence/gate_c3/i0_i1_i2_i3_receipt.json` | Frozen, SHA verified |
| I4 opportunity audit | `evidence/gate_c3/i4_opportunity/` | Frozen, SHA verified |

Vocabulary domain: lighthouses (22 heads, 6 roles, 4 descriptors, 8 symbolic codes, 5 enum values, 2 boolean values, 4 JSON keys). All surface vocabulary tokens are disjoint from the five prior corpora (birds, minerals, constellations, rivers, summits).

## 3. Ladder results

| Arm | QAnchorRate | CorrectAnchor | WrongAnchor | FalseRes | Ambiguous | Unresolved | S2cLive | MappingExtract |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| I0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | N/A |
| I1 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | N/A |
| I2 | 0.46 | 0.00 | 0.12 | 0.12 | 0.34 | 0.54 | 0.00 | N/A |
| **I3** | **1.00** | **0.92** | **0.00** | **0.00** | **0.08** | **0.00** | **0.92** | **1.00** |

## 4. Per-rung findings

### I0 — current surface anchor extraction
- **MEASURED**: 0/100 tasks resolved. The mechanism is completely inert.
- The entity type derivation requires CapitalizedHead + qualified trailing noun phrase. The alias surface (e.g., "Beachy prism") has a different head from the canonical and a truncated role, so no entity type is recognized.

### I1 — normalization-only negative control
- **MEASURED**: 0/100 tasks resolved. I1 ≈ I0 exactly as predicted.
- Normalization cannot recover a missing token, and the alias surface is a different name from the canonical. The negative control outcome is confirmed.

### I2 — prefix/suffix tolerant resolution
- **MEASURED**: 0/100 correct, 12% false resolution, 34% ambiguous, 54% unresolved.
- **NEGATIVE_RESULT**: Prefix/suffix matching found candidates on 46% of tasks but never the correct canonical. The alias surface has a different head from the canonical, so head-token preservation prevents correct matching. Suffix extension finds wrong entities (e.g., "Beachy prism" → "Beachy prism bracket" instead of "Fastnet fog siren"). TruncationRecoveryRate = 0.0 on 50 eligible tasks.
- **NOT PROMOTED**: All three promotion criteria failed.

### I3 — identity-record retrieval
- **MEASURED**: 92/100 correct, 0% wrong, 0% false resolution, 8% ambiguous, 0% unresolved.
- **SUPPORTED**: Reading the explicit surface→canonical mapping from runtime-visible identity records resolves the surface identity defect. IdentityMappingExtractionRate = 1.0 (parser extracts correct mapping on all 100 eligible tasks). S2cLiveRate = 0.92 (chains become active when the anchor is resolved).
- Gains across all 10 task families. Per-regime: alias 84% correct / 16% ambiguous, description 100% correct.
- The 8 ambiguous tasks have surfaces appearing in multiple conflicting identity records from different tasks. I3 correctly abstains.
- **PROMOTED**: All three promotion criteria passed.

### I4 — bounded canonicalization through retrieved identity edges
- **NOT RUN**: I4 opportunity audit measured I4OpportunityRate = 0.125 < 0.25 threshold.
- Of 8 ambiguous tasks: 0 resolvable by second identity edge (A), 1 resolvable by non-identity context (B), 7 genuinely conflicting (C), 0 missing evidence (D).
- The 7 genuine conflicts are surface collisions where two different tasks generate the same alias surface but map to different canonicals. In the controlled C3 setup (full evidence pool), both canonicals appear in the pool, so context-based disambiguation cannot distinguish them. In a real retrieval pipeline with a smaller, task-specific candidate pool, context disambiguation might work better — but that is a C4 question, not a C3 question.
- **DECISION**: I4 skipped. Insufficient measured opportunity for deeper identity traversal.

## 5. Claim boundaries

### Supported

- I0 current anchor extraction is inert on the fresh C3 surface corpus.
- I1 normalization does not resolve arbitrary aliases.
- I2 prefix/suffix matching is unsafe (12% false resolution) and fails promotion.
- I3 explicit identity-record resolution recovers 92% of canonical anchors with 0% false resolution.
- I3 activates the structural selector on 92% of tasks.
- Remaining ambiguity (8%) has insufficient measured opportunity for deeper identity traversal (I4OpportunityRate = 0.125 < 0.25).

### Not supported

- Identity resolution is solved generally (8% ambiguity remains).
- Alias ambiguity is solved (genuine conflicts exist).
- Graph traversal is required (I4 opportunity is too low).
- Learned entity resolution is required (not tested).
- The integrated end-to-end pipeline works (C4 has not started).

## 6. Next-gate decision

Gate C3 is closed with **MECHANISM_SUCCESS_I3_IDENTITY_RECORD_RESOLUTION**. The qualified identity-resolution mechanism (I3) should be integrated into the C4 pipeline along with the previously qualified S2c structural selector.

Gate C4 (Integrated Non-Oracle Memory Pipeline) is the correct next gate.
