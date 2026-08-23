# R13 Confirmation Results — I3.15c

> **Documentation revision (R13-F):** The original closing statement "the governor identifies a useful direction, but the model cannot execute it" has been corrected to "T2 reliably identifies the intended epistemic phase, but the tested R1 representation switch is counterproductive under the qualified Gemma policy backend." R13 does not establish that the detected direction is intrinsically useful — only that T2 detection is accurate and the R1 intervention is harmful. Raw R13 artifacts are unchanged.

## Stage A: Dataset Closure

### Closure Gate

| Gate | Required | Actual | Status |
|------|----------|--------|--------|
| Accepted records | 1280 | 1280 | PASS |
| Unique accepted keys | 1280 | 1280 | PASS |
| Missing | 0 | 0 | PASS |
| Duplicates | 0 | 0 | PASS |
| Errors | 0 | 0 | PASS |
| Arm balance (A1/R1) | 640/640 | 640/640 | PASS |
| Retrieval condition | Q3_RERANKED only | Q3_RERANKED | PASS |
| Protocol identity | 1 unique | 1 (I3_15C_CONFIRMATION_PROTOCOL_V2) | PASS |
| Backend identity | 1 unique | 1 (2ad4c9ce431a2d5b) | PASS |
| Protocol SHA | 1 unique | 1 (9590440d...) | PASS |
| GGUF SHA | 1 unique | 1 (2ad4c9ce...) | PASS |
| Runtime config SHA | 1 unique | 1 (c64eb7b8...) | PASS |
| Receipt identity SHA | 1 unique | 1 (bb612a2c...) | PASS |
| Confirmation executable SHA | matches expected | 41cc60b0... | PASS |
| Segment identity constancy | all segments identical | all identical | PASS |
| R13-RUNTIME-001 quarantined | 28 | 28 | PASS |
| R13-RUNTIME-001 frozen replacements | 28 | 28 | PASS |
| R13-RUNTIME-001 missing replacements | 0 | 0 | PASS |
| R13-RUNTIME-001 duplicate replacements | 0 | 0 | PASS |
| R13-RUNTIME-001 deviant record leakage | 0 | 0 | PASS |

**R13_DATASET_VALID = TRUE**

**R13_DATASET_SHA256:** `56cff26a4f13d519810a77f61f7a8280cb6d665e729270ff421966cdeccb62db`

**Classification:** `PRE_SPECIFIED_CONFIRMATION_WITH_KNOWN_PROVENANCE_FIELD_DEFECT`

### Known Defects

**R13-PROV-001:** `run_manifest.confirmation_executable_sha256` incorrectly aliases `runtime_config_sha256`. The actual executable SHA (`41cc60b04f506f63...`) is verified independently via `confirmation_executable_sha256.txt`. Scientific execution unaffected.

**R13-RUNTIME-001:** 28 trajectories (ordinals 616-643) were executed under deviant server parameters (`-t 8`, `--batch-size 4096`, `--ubatch-size 1024`) instead of the frozen config (`-t 4`, `--batch-size 2048`, `--ubatch-size 512`). All 28 were quarantined with their associated receipts, the server was restored to frozen parameters, and all 28 keys were rerun under the frozen runtime. Status: `QUARANTINED_AND_FULLY_RERUN`.

### Execution Segment Provenance

| Segment | Start | End | Provenance | Runtime |
|---------|-------|-----|------------|--------|
| 1 | 0 | 357 | RECONSTRUCTED_FROM_CHECKPOINT_AND_LOG | frozen |
| 2 | 357 | 517 | RECONSTRUCTED_FROM_CHECKPOINT_AND_LOG | frozen |
| 3a | 517 | 615 | RECORDED_AT_RECOVERY | frozen |
| 3b (deviant) | 615 | 643 | historical | deviant (excluded) |
| 3b (replacement) | 615 | 651 | RECORDED_AT_RECOVERY | frozen |
| 4 | 651 | 818 | RECORDED_AT_RECOVERY | frozen |
| 5 | 818 | 991 | RECORDED_AT_RECOVERY | frozen |
| 6 | 991 | 1163 | RECORDED_AT_RECOVERY | frozen |
| 7 | 1163 | 1280 | RECORDED_AT_RECOVERY | frozen |

All accepted segments share identical scientific identities (source commit, GGUF SHA, runtime config SHA, confirmation executable SHA). 7 VM deaths survived with zero data loss.

---

## Stage B: Efficacy Analysis

### Primary Endpoint

**Δ_late = E[U_R1 − U_A1 | T2-late]**

| Metric | Value |
|--------|-------|
| n | 240 |
| Mean Δ | −0.0806 |
| 95% CI | [−0.1344, −0.0358] |
| Criterion | LCB > 0 |
| Result | **FAIL** |

The primary endpoint fails. The lower confidence bound is negative, meaning R1 does not demonstrate a positive utility improvement over A1 on late-T2 trajectories. The effect is significantly negative.

### Secondary Contrasts

| Contrast | n | Mean Δ | 95% CI | Passes? |
|----------|---|--------|--------|---------|
| I_phase | 400 | −0.1277 | [−0.1881, −0.0739] | FAIL |
| Δ_T2_late_1 | 80 | −0.0269 | [−0.0806, 0.0000] | FAIL |
| Δ_T2_late_2 | 80 | −0.2150 | [−0.3763, −0.0806] | FAIL |
| Δ_T2_late_3 | 80 | 0.0000 | [0.0000, 0.0000] | FAIL |
| Δ_T2_immediate | 80 | −0.2688 | [−0.4300, −0.1344] | FAIL |

All secondary contrasts fail. The phase interaction I_phase is significantly negative. The immediate-T2 stratum shows the largest harm (−0.2688), and late_2 shows substantial harm (−0.2150). Late_3 and all control strata show zero variance (all deltas = 0).

### Per-Stratum Detail

| Stratum | n | Mean Δ Utility | Mean Δ Steps | T2 Triggered | Step Limit |
|---------|---|----------------|--------------|--------------|------------|
| T2_CONFLICT_IMMEDIATE | 80 | −0.2688 | +0.125 | 76/80 | 0 |
| T2_CONFLICT_LATE_1 | 80 | −0.0269 | +0.0125 | 76/80 | 0 |
| T2_CONFLICT_LATE_2 | 80 | −0.2150 | +0.100 | 76/80 | 0 |
| T2_CONFLICT_LATE_3 | 80 | 0.0000 | 0.000 | 0/80 | 0 |
| MATCHED_NEG_IMMEDIATE | 80 | 0.0000 | 0.000 | 0/80 | 0 |
| MATCHED_NEG_LATE | 80 | 0.0000 | 0.000 | 0/80 | 0 |
| DEFER_CONTROL | 80 | 0.0000 | 0.000 | 0/80 | 0 |
| ANSWER_CONTROL | 80 | 0.0000 | 0.000 | 0/80 | 0 |

Key observations:
- T2 triggered in 76/80 trajectories in IMMEDIATE, LATE_1, and LATE_2 strata (95% trigger rate)
- T2 never triggered in LATE_3 (0/80) — this stratum has no T2-eligible conflicts
- T2 never triggered in any control stratum (0/240) — safety verified
- R1 adds steps (+0.059 mean across T2+ strata) without adding utility
- All control strata show exactly zero delta — R1 does nothing when T2 is absent

### Control Contrasts

| Control | Mean Δ | 90% CI | Equivalent? | Margin |
|---------|--------|--------|-------------|--------|
| DEFER_CONTROL | 0.0000 | [0.0000, 0.0000] | YES | m=5 |
| ANSWER_CONTROL | 0.0000 | [0.0000, 0.0000] | YES | m=5 |
| MATCHED_NEG | 0.0000 | [0.0000, 0.0000] | YES | m=5 |

All controls are perfectly equivalent. R1 does not perturb behavior when T2 is absent.

### Safety

| Check | Result |
|-------|--------|
| False T2 rate on DEFER_CONTROL | 0/80 = 0% |
| False T2 rate on ANSWER_CONTROL | 0/80 = 0% |
| False T2 rate on MATCHED_NEG | 0/80 = 0% |
| Overall false T2 rate | 0.0000 |
| Safety PASS | **TRUE** |

R1 never falsely triggers T2 on any control trajectory. Safety is perfect.

### Cost Metrics

| Metric | Value | 95% CI |
|--------|-------|--------|
| Δ_Steps (T2+) | +0.0594 | [+0.0344, +0.0875] |
| P(step_limit R1, T2+) | 0.0% | n=320 |
| P(step_limit A1, T2+) | 0.0% | n=320 |

R1 adds a small but statistically significant number of steps (+0.06 mean) on T2+ trajectories. No trajectories hit the step limit in either arm. Cost is nonzero but small.

### Promotion Criteria

| Criterion | Result |
|-----------|--------|
| Effectiveness (primary late-T2 gate) | **FAIL** |
| Phase interaction gate | **FAIL** |
| Safety | PASS |
| Controls equivalent | PASS |
| Cost OK | PASS |
| ALL CRITERIA | **FAIL** |

---

## R1 Disposition

### Mechanical Derivation

| Label | Condition |
|-------|-----------|
| PROMOTED | primary late-T2 gate passes AND phase interaction passes AND no critical safety/control failure |
| CONDITIONALLY PROMOTED | primary benefit positive/convincing BUT cost or interaction criteria weaker |
| NOT PROMOTED | no demonstrated incremental value |
| REJECTED | convincing harm |

Applied:
- Primary late-T2 gate: **FAIL** (LCB = −0.1344 < 0)
- Phase interaction gate: **FAIL** (LCB = −0.1881 < 0)
- Safety: PASS
- Controls: PASS
- Cost: PASS
- Primary effect is significantly **negative** (not merely absent)

**Disposition: REJECTED**

R1 demonstrates convincing harm on this frozen benchmark. The primary endpoint Δ_late is significantly negative (−0.0806, 95% CI [−0.1344, −0.0358]). The phase interaction is also significantly negative. The immediate-T2 stratum shows the worst harm (−0.2688). R1 adds steps without adding utility, and the harm is concentrated in strata where T2 actually triggers.

---

## What This Proves / Does Not Prove

### Proves

1. **R1 does not improve Gemma-3-12B decisions on this frozen benchmark.** The primary endpoint is significantly negative, not merely null. This is a decisive negative result, not an inconclusive one.

2. **The harm is mechanism-specific, not noise.** The negative effect is concentrated in T2-triggering strata (IMMEDIATE: −0.27, LATE_2: −0.22) and absent in controls (all 0.0). T2 triggers correctly (76/80 in eligible strata, 0/240 in controls), but the model cannot execute the suggested direction productively.

3. **R1 is safe but ineffective.** Safety is perfect (0% false T2 rate). Controls are perfectly equivalent. R1 does nothing when T2 is absent. But when T2 is present, R1 makes things worse.

4. **T2 reliably identifies the intended epistemic phase, but the tested R1 representation switch is counterproductive under the qualified Gemma policy backend.** T2 correctly detects the conflict phase (76/80 trigger rate in eligible strata, 0% false activation on controls). However, routing the model into persistent M3 representation at that phase does not improve outcomes — it adds steps and reduces utility. R13 does not establish that the detected direction itself is intrinsically useful; it establishes that the specific R1 intervention at that phase is harmful.

5. **The infrastructure works.** 7 VM deaths, 1 runtime deviation, and 1 provenance defect were all contained without data loss or scientific contamination. The recovery harness is operationally qualified.

### Does Not Prove

1. **Does not prove that phase-conditioned representation routing is universally harmful.** This is one model (Gemma-3-12B Q4), one benchmark (I3.15c), one retrieval condition (Q3_RERANKED), and one governor configuration. The negative result is specific to this frozen configuration.

2. **Does not prove that MDSG/T2 is useless.** T2 triggers correctly and safely. The failure is in the R1 representation switch, not in T2 detection. A different model, different prompt, or different routing target might benefit from the same detected conflicts.

3. **Does not prove that the DAPH V2B executive is worthless.** It proves that R1 (re-verification routing) does not help this model on this benchmark. Other mechanisms (different routing targets, different decision-state compressions) remain untested.

4. **Does not prove broad real-world superiority or inferiority.** The benchmark is a controlled synthetic environment with frozen retrieved evidence. Real-world performance may differ.

5. **Does not prove that the semantic extraction errors are the cause.** The benchmark uses inferred (not oracle) semantic relations. Retrieval incompleteness and extraction error are present but controlled. The negative result holds despite these imperfections, not because of them.

---

## Technical Summary

| Property | Value |
|----------|-------|
| Protocol | I3_15C_CONFIRMATION_PROTOCOL_V2 |
| Model | google/gemma-3-12b-it-qat-q4_0-gguf |
| Backend | llama.cpp (2ad4c9ce431a2d5b) |
| Source commit | 5454246b7e61adfb7a093eb5a1f731347071270d |
| Trajectories | 1280 (640 A1 / 640 R1) |
| Retrieval | Q3_RERANKED |
| Runtime | frozen (-t 4, batch 2048, ubatch 512, ctx 32768, temp 0.0, seed 42) |
| Segments | 7 (5 VM recoveries) |
| VM deaths | 7 |
| Runtime deviations | 1 (R13-RUNTIME-001, quarantined and rerun) |
| Provenance defects | 1 (R13-PROV-001, documented) |
| Dataset SHA | 56cff26a4f13d519810a77f61f7a8280cb6d665e729270ff421966cdeccb62db |
| Primary endpoint | Δ_late = −0.0806, 95% CI [−0.1344, −0.0358] |
| Disposition | **REJECTED** (convincing harm) |
