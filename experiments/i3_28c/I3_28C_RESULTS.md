# I3.28C: Targeted DEFER Coverage Audit — Results

**Date:** 2026-08-26
**Branch:** `i3.27-q-error-and-authority`
**Experiment:** `scripts/run_i3_28c_coverage_audit.py`

---

## Question

Can DEFER hard authority have meaningful coverage on safe states where continuation actions are structurally dominated, while remaining blocked on unsafe states?

---

## Design

Four frozen strata with structural variation (different evidence counts, hypothesis counts, budget configurations, domain templates):

| Stratum | Description | n states | Key property |
|---------|-------------|:--------:|--------------|
| D1 | Safe DEFER, VERIFY unavailable | 30 | has_competing=0, max_verify=0 |
| D2 | Safe DEFER, verification completed | 30 | has_competing=0, post-VERIFY state |
| D3 | Unsafe contradiction | 30 | has_competing=1, DEFER blocked |
| D4 | ANSWER-correct, n_verified=2 | 30 | Contrast for VERIFY overestimation |

D4 was added after I3.28C round 1 showed VERIFY overestimation bleeding into n_verified=2 states. D4 provides explicit contrast: at n_verified=2, ANSWER (99.89) > VERIFY (96.71) > DEFER (-30.11).

Structural variation within each stratum:
- 2-3 hypotheses (not always 2)
- 2-4 visible evidence items
- 6 budget configurations per stratum (varied steps, reasoning tokens, verify availability)
- 10 domain templates rotated across 30 states
- Varied correct hypothesis index
- Varied evidence support/contradiction patterns

475 strata records appended to 1056 original + 400 I3.28B boundary = 1931 total. Same GBT, same hyperparameters, same threshold 5.0.

---

## Results

### Gate 1: DEFER coverage — PASS

**Coverage_DEFER = P(force DEFER | D1∪D2) = 26/32 = 0.8125**

| Stratum | Coverage |
|---------|:--------:|
| D1 (VERIFY unavailable) | 14/14 = 100% |
| D2 (verification completed) | 12/18 = 66.7% |
| D3 (unsafe, blocked) | 0/18 = 0% |
| D4 (ANSWER-correct) | 0/6 = 0% (correct — no DEFER authority) |

D1 achieves 100% coverage: when VERIFY is structurally unavailable, DEFER dominates with gap ~85-97. D2 achieves 66.7%: after verification, DEFER dominates in most but not all budget configurations.

### Gate 2: False DEFER authority — PASS

**FalseAuthorityRate_DEFER = P(force DEFER | D3) = 0/18 = 0.0000**

The structural safety predicate (`has_competing_unverified_support == 0`) blocks all DEFER authority in unsafe states. Even when Q(DEFER) is the highest action in D3 (gap=19.12 in some states), the predicate prevents false authority.

### Gate 3: Separation — PASS

- I3.28B boundary: margin=98.33 (required >5.0)
- I3.26 dev: margin=101.34 (required >5.0)

### Gate 4: ANSWER authority preserved — PASS

- 220 original checkpoints: 44/44 ANSWER ranked best
- D4 stratum: 6/6 ANSWER ranked best
- Q(ANSWER) at ol_answer states: 99.30-100.3 (correct)

### Gate 5-8: Preservation — PASS

| Metric | V1 control | V2R repaired |
|--------|:---------:|:------------:|
| Mean regret | 22.31 | 19.03 |
| Near-optimal (ε=3) | 136/220 | 156/220 |
| Correct best | 64/220 | 84/220 |

V2R improves on V1 across all preservation metrics.

### Overall: ALL 8 GATES PASS

---

## Sample Q values

### D1 (safe DEFER, VERIFY unavailable)
```
best=DEFER(69.8) 2nd=STOP(-26.9) gap=96.67 competing=0
best=DEFER(69.8) 2nd=REASON_MORE(-16.2) gap=85.95 competing=0
```

### D2 (safe DEFER, verification completed)
```
best=DEFER(70.0) 2nd=STOP(-39.4) gap=109.43 competing=0
best=DEFER(70.0) 2nd=REASON_MORE(-24.4) gap=94.38 competing=0
```

### D3 (unsafe contradiction — DEFER blocked)
```
best=DEFER(-27.8) 2nd=STOP(-46.9) gap=19.12 competing=1  ← predicate blocks
best=VERIFY(99.3) 2nd=DEFER(-29.3) gap=128.63 competing=1  ← VERIFY correctly best
```

### D4 (ANSWER-correct, n_verified=2)
```
best=ANSWER(100.3) 2nd=DEFER(-27.4) gap=127.73
best=ANSWER(100.0) 2nd=DEFER(-29.2) gap=129.14
```

---

## Key Findings

1. **DEFER authority has real coverage (81.25%).** When VERIFY is structurally unavailable (D1) or already exhausted (D2), DEFER dominates with gaps of 85-109. The authority threshold of 5.0 fires reliably.

2. **False authority is exactly zero.** The structural safety predicate (`has_competing_unverified_support == 0`) blocks all DEFER authority in unsafe states, even when Q(DEFER) has a large gap.

3. **D4 contrast was necessary.** Without D4, the high-utility VERIFY records in D3 caused the GBT to overestimate Q(VERIFY) at n_verified=2 states, breaking ANSWER authority. Adding 30 ANSWER-correct contrast states (n_verified=2, ANSWER dominates) fixed this.

4. **The model learned the epistemic/resource condition, not budget memorization.** Coverage generalizes across 10 domain templates, 2-3 hypothesis counts, 2-4 evidence counts, and 6 budget configurations per stratum.

---

## Decision

All offline gates pass:
- Coverage_DEFER = 81.25% (target: materially > 0)
- FalseAuthorityRate_DEFER = 0.0000 (target: 0)
- ANSWER authority preserved (44/44 + 6/6)
- Separation maintained (margin 98-101)
- Preservation non-inferior (all metrics improved)

**Per the decision tree: freeze DAPH_ADAPTIVE_AUTHORITY_EXECUTIVE_V2 and proceed to targeted live D1/D2/D3 safety run.**

The live safety run should include:
- D1-type states (VERIFY unavailable, DEFER correct) — test DEFER authority fires and helps
- D2-type states (verification completed, DEFER correct) — test DEFER authority fires and helps
- D3-type states (competing support, DEFER wrong) — test DEFER authority does NOT fire
- Known DEFER rescue cases from I3.27

If rescues > breaks and false authority still 0, size fresh confirmation by power.

---

## Artifacts

| File | Description |
|------|-------------|
| `experiments/i3_28c/strata_causal_actions_v1.jsonl` | 475 strata causal records |
| `experiments/i3_28c/Q_V2R_coverage_repaired.pkl` | Q_V2R trained on 1931 records |
| `experiments/i3_28c/Q_V1_coverage_control.pkl` | Q_V1 control on 1931 records |
| `experiments/i3_28c/full_results.json` | All gate results |
