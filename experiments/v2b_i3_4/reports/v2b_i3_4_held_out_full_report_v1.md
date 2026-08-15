# DAPH V2B-I3.4.2 Held-Out Evaluation Report

**Schema:** `DAPH_V2B_I3_4_HELD_OUT_REPORT_V1`
**Generated:** 2026-08-15T05:53:27.812573+00:00
**Frozen at:** `7af42ed` (FINAL_PRE_HELD_OUT_FREEZE)
**Configuration:** Unchanged across all three held-out phases

## 1. Primary Results Summary

| Split | N | ΔDG | CI lower | CI upper | Excludes 0 | Status |
|-------|--:|----:|---------:|---------:|:----------:|--------|
| held_out_instance | 100 | +9.3458 | -1.5087 | +20.5250 | False | INCONCLUSIVE |
| held_out_surface | 50 | +6.8078 | -7.6208 | +22.0284 | False | INCONCLUSIVE |
| **held_out_structure** | **150** | **-13.4285** | **-17.2140** | **-9.7907** | **True** | **REJECTED** |

## 2. The Decisive Test: HELD_OUT_STRUCTURE

The held_out_structure split tests generalization to **unseen topological structures** — cognitive control topologies that do not appear in development or validation.

### Result

| Metric | Blind | Aware |
|--------|------:|------:|
| Success rate | 32.7% | 32.7% |
| Mean utility | -39.2906 | -38.8628 |
| IG | 31.7543 | 17.8980 |
| DG | 83.8147 | 97.2433 |
| TR | 115.5691 | 115.1413 |

**ΔDG = -13.4285**, 95% CI = [-17.2140, -9.7907]

**PRIMARY CRITERION: LCB_95 > 0 → FAIL**

**HYPOTHESIS: REJECTED_ON_HELD_OUT_STRUCTURE**

The CI is entirely negative. On unseen topological structures, the state-aware controller has a **larger** decision gap than the state-blind controller. The aware condition's DG (97.24) exceeds the blind condition's DG (83.81).

### Interpretation

The aware condition still has lower IG (17.90 vs 31.75) — it has more information. But on unseen structures, that additional information does **not** translate into better executive decisions. In fact, it appears to lead to worse decisions relative to the aware observable oracle.

This is a valid negative result. The development effect (+9.19) does not generalize to unseen topological structures.

## 3. Full Trajectory Across All Splits

| Split | N | ΔDG | CI | Status | Blind succ | Aware succ |
|-------|--:|----:|---:|--------|---------:|---------:|
| development | 300 | +9.188 | [+2.52, +16.28] | SUPPORTED | 42.0% | 48.0% |
| validation | 150 | +5.165 | [-1.12, +12.51] | INCONCLUSIVE | 36.0% | 40.7% |
| held_out_instance | 100 | +9.346 | [-1.51, +20.53] | INCONCLUSIVE | 44.0% | 50.0% |
| held_out_surface | 50 | +6.808 | [-7.62, +22.03] | INCONCLUSIVE | 48.0% | 52.0% |
| **held_out_structure** | **150** | **-13.429** | **[-17.21, -9.79]** | **REJECTED** | **32.7%** | **32.7%** |

## 4. IG/DG/TR Decomposition Across Splits

| Split | Blind IG | Aware IG | Blind DG | Aware DG | Blind TR | Aware TR |
|-------|---------:|---------:|---------:|---------:|---------:|---------:|
| development | 4.63 | 0.47 | 81.62 | 72.43 | 86.25 | 72.90 |
| validation | 5.47 | 2.70 | 109.33 | 104.16 | 114.80 | 106.86 |
| held_out_instance | 4.23 | 0.02 | 77.92 | 68.57 | 82.15 | 68.59 |
| held_out_surface | 2.94 | 0.04 | 71.61 | 64.81 | 74.55 | 64.85 |
| **held_out_structure** | **31.75** | **17.90** | **83.81** | **97.24** | **115.57** | **115.14** |

Key observation: On held_out_structure, IG is much larger for both conditions (31.75 vs ~3-5 on other splits), indicating these tasks have substantially more unobservable structure. The aware condition reduces IG (17.90 vs 31.75) but **increases** DG (97.24 vs 83.81). The additional information does not help the controller make better decisions on these unseen structures.

## 5. Structural Checks

| Check | hoi | hos | host |
|-------|-----|-----|------|
| TR = IG + DG | PASS | PASS | PASS |
| IG >= 0 | PASS | PASS | PASS |
| DG >= 0 | PASS | PASS | PASS |
| Fingerprint-valid | 100/100 | 50/50 | 150/150 |
| Decoder failures | 0 | 0 | 0 |
| Backend errors | 0 | 0 | 0 |

## 6. API Call Summary

| Split | Total calls | Blind | Aware |
|-------|------------:|------:|------:|
| held_out_instance | 431 | 244 | 187 |
| held_out_surface | 217 | 124 | 93 |
| held_out_structure | 788 | 406 | 382 |
| **Total held-out** | **1436** | | |

## 7. Scientific Conclusion

### What the data says

1. **Development**: ΔDG > 0 supported (CI [+2.52, +16.28])
2. **Validation**: Directionally consistent but inconclusive (CI [-1.12, +12.51])
3. **Held-out instance**: Directionally consistent but inconclusive (CI [-1.51, +20.53])
4. **Held-out surface**: Directionally consistent but inconclusive (CI [-7.62, +22.03])
5. **Held-out structure**: **REJECTED** (CI [-17.21, -9.79], entirely negative)

### What this means

The executive-metareasoning hypothesis — that structured cognitive state helps DeepSeek make better executive decisions — is **supported on development** but **rejected on held-out topological structures**.

The pattern across splits suggests:
- The aware condition reliably reduces **information gap** (IG) across all splits
- The aware condition reduces **decision gap** (DG) on development, validation, instance, and surface splits (directionally, though not always significantly)
- On **unseen topological structures**, the aware condition **increases** decision gap — the additional information leads to worse executive decisions relative to the available observable oracle

This is a valid negative result for the generalization claim. The development effect does not survive the decisive held-out structure test.

### Status

- All held-out data was accessed with the same frozen configuration
- No configuration changes were made between phases
- The result is a valid scientific failure of the generalization hypothesis
- The development evidence remains valid as development evidence, not as a general claim

---

*This report covers all three held-out evaluation phases. The configuration was frozen at commit 7af42ed and unchanged throughout. Total held-out API calls: 1436. All structural checks pass.*
