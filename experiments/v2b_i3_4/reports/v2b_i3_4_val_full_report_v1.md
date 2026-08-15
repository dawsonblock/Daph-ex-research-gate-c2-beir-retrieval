# DAPH V2B-I3.4.2 Validation Run Report

**Schema:** `DAPH_V2B_I3_4_VAL_REPORT_V1`
**Generated:** 2026-08-15T04:54:48.715515+00:00
**Experiment ID:** `v2b_i3_4_experiment_v1`
**Frozen at:** `18b81739dc4612f97255705b00328b28a267a957`

## 1. Experiment Configuration

Same frozen configuration as development. No changes to controller, prompt, policy, generation config, retry policy, scoring, or observable oracle views.

| Parameter | Value |
|-----------|-------|
| Model | `deepseek-v4-flash` |
| Thinking mode | disabled |
| Response format | `json_object` |
| Temperature | 0.0 |
| Max tokens | 2048 |
| Split | validation |
| Task count | 150 |
| Max steps per task | 24 |

## 2. Headline Results

| Metric | Blind | Aware | Difference |
|--------|------:|------:|-----------:|
| Success rate | 28.0% | 32.0% | +4.0pp |
| Mean utility | -48.0193 | -40.0811 | +7.9382 |
| Total API calls | 371 | 383 | +12 |
| Fingerprint-valid pairs | 150/150 | | |
| Decoder failures | 0 | | |
| Backend errors | 0 | | |

## 3. IG/DG/TR Decomposition

| Aggregate | Blind | Aware |
|-----------|------:|------:|
| E[V_L^*] | 66.7823 | 66.7823 |
| E[V_O^M] | 61.3097 | 64.0829 |
| E[V_pi^M] | -48.0193 | -40.0811 |
| **IG** | 5.4726 | 2.6993 |
| **DG** | 109.3289 | 104.1640 |
| **TR** | 114.8015 | 106.8633 |

### Identity and Non-negativity Checks

| Check | Result |
|-------|--------|
| TR = IG + DG (all 300 contributions) | PASS |
| Blind IG >= 0 | PASS |
| Blind DG >= 0 | PASS |
| Aware IG >= 0 | PASS |
| Aware DG >= 0 | PASS |

## 4. Statistical Analysis (DG = DG_blind - DG_aware)

| Statistic | Value |
|-----------|-------|
| N paired tasks | 150 |
| Mean DG | 5.164933 |
| Bootstrap CI lower | -1.115733 |
| Bootstrap CI upper | 12.505300 |
| Bootstrap iterations | 10,000 |
| CI excludes zero | False |
| **Hypothesis status** | **INCONCLUSIVE_ON_VALIDATION** |

## 5. DG Distribution

| Category | Count |
|----------|------:|
| Positive (blind DG > aware DG) | 33 |
| Negative (blind DG < aware DG) | 46 |
| Approximately zero | 71 |

## 6. Disagreement Analysis

| Category | Count |
|----------|------:|
| Both succeed | 138 |
| Both fail | 138 |
| Blind succeeds, aware fails | 0 |
| Aware succeeds, blind fails | 6 |

## 7. Action Distribution

| Action | Blind | Aware |
|--------|------:|------:|
| ANSWER | 133 | 136 |
| DEFER | 4 | 1 |
| REASON_MORE | 0 | 27 |
| RETRIEVE | 113 | 92 |
| SEARCH_MORE | 0 | 22 |
| STOP | 13 | 13 |
| VERIFY | 108 | 92 |

## 8. Comparison: Development vs Validation

| Metric | Development | Validation |
|--------|------------:|-----------:|
| N tasks | 300 | 150 |
| Mean DG | +9.188 | +5.165 |
| CI lower | +2.517 | -1.116 |
| CI upper | +16.278 | 12.505 |
| CI excludes zero | Yes | False |
| Blind success | 42.0% | 28.0% |
| Aware success | 48.0% | 32.0% |
| Blind DG | 81.62 | 109.33 |
| Aware DG | 72.43 | 104.16 |
| Blind IG | 4.63 | 5.47 |
| Aware IG | 0.47 | 2.70 |

## 9. Scientific Interpretation

### What the validation data shows

1. **Mean DG = +5.1649** (CI: [-1.1157, 12.5053])
   - The point estimate is positive, consistent with development
   - The CI **includes zero**, so the hypothesis is **inconclusive** at 95% on validation
   - The point estimate (+5.16) is smaller than development (+9.19) but in the same direction

2. **All structural checks pass**: TR = IG + DG, IG >= 0, DG >= 0

3. **Direction is consistent**: The aware condition still shows lower DG than blind, but the smaller sample (150 vs 300) and smaller effect size produce a wider CI

4. **33 of 150 tasks show positive DG** (blind worse), 46 show negative

### Status

- **Validation split only** — no held-out data was accessed
- **Same frozen configuration** as development — no tuning
- **Hypothesis DG > 0: INCONCLUSIVE_ON_VALIDATION**
- The positive direction is preserved but not confirmed at 95% on validation alone

### Assessment

The validation result is **consistent with development but not independently conclusive**. The point estimate remains positive (+5.16), the direction is preserved, and all structural checks pass. However, the smaller sample and effect size produce a CI that includes zero.

This is not a failure — it is a weaker confirmation. The combined development + validation evidence (450 tasks) would be the next thing to examine, but the preregistered protocol calls for sequential evaluation of held-out splits.

### Next steps

The held-out splits (instance, surface, structure) remain untouched. The decisive generalization test is held_out_structure.

---

*This report was generated with the exact same frozen configuration as the development run. No controller, prompt, policy, scoring, or oracle view changes were made.*
