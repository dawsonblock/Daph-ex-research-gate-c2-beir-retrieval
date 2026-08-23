# R13-F: Post-Hoc Exploratory Forensic Analysis

> **Label: POST_HOC_EXPLORATORY**
>
> R13-F can identify likely failure mechanisms; it cannot confirm them causally.
> Any hypothesis produced by R13-F must be tested in new held-out development data.

## Source

| Property | Value |
|----------|-------|
| R13_DATASET_SHA256 | 56cff26a4f13d519810a77f61f7a8280cb6d665e729270ff421966cdeccb62db |
| Source | raw_closed/ (immutable) |
| R13_F_ANALYSIS_SHA256 | a2c8d8bc0a379b7dc75348418c6cced03cf93183530d274f0a9c26cedc3ebcd2 |
| Pairs | 640 (A1/R1 matched by task_id + retrieval_level + backend_identity) |

---

## 1. Pre-T2 Variance Classification

| Class | Count | Description |
|-------|-------|-------------|
| IMMEDIATE_T2 | 76 | T2 triggered at step 0 (no pre-T2 prefix) |
| PREFIX_IDENTICAL | 152 | A1/R1 actions identical through trigger_step−1 |
| PRE_T2_DIVERGED | 0 | Trajectories diverged before T2 |
| NO_TRIGGER | 412 | R1 did not trigger T2 |

**Key finding:** Zero pre-T2 divergences. Every late-trigger pair (152/152) has identical A1/R1 action sequences before T2 fires. This means all observed harm is attributable to the post-T2 intervention, not to independent model call variance.

---

## 2. First Post-T2 Divergence Matrix

| Transition | Count | Mean ΔU |
|------------|-------|---------|
| VERIFY→VERIFY | 185 | −0.1743 |
| VERIFY→RETRIEVE | 4 | −2.1500 |
| No divergence | 451 | — |

**Key finding:** In 185 of 189 divergent pairs, the first divergence is **VERIFY→VERIFY** — both arms execute VERIFY, but they diverge on **target_id** (different evidence items selected for verification). The action label is identical; the target differs.

Only 4 pairs show an action-type divergence (VERIFY→RETRIEVE). The harm in those is much larger (−2.15) but the count is tiny.

This means the primary failure mechanism is **not action displacement** but **target selection within VERIFY**. R1's M3 representation causes the model to select different verification targets than A1, and those targets are systematically worse.

---

## 3. Action Distribution Displacement ΔP(a|T2)

| Stratum | Action | P_R1 | P_A1 | ΔP |
|---------|--------|------|------|-----|
| ALL_T2 | VERIFY | 0.9836 | 1.0000 | −0.0164 |
| ALL_T2 | RETRIEVE | 0.0164 | 0.0000 | +0.0164 |
| IMMEDIATE | VERIFY | 0.9785 | 1.0000 | −0.0215 |
| IMMEDIATE | RETRIEVE | 0.0215 | 0.0000 | +0.0215 |
| LATE_1 | VERIFY | 0.9974 | 1.0000 | −0.0026 |
| LATE_1 | RETRIEVE | 0.0026 | 0.0000 | +0.0026 |
| LATE_2 | VERIFY | 0.9744 | 1.0000 | −0.0256 |
| LATE_2 | RETRIEVE | 0.0256 | 0.0000 | +0.0256 |
| LATE_3 | (no T2 triggers) | — | — | — |

**Key finding:** The action distribution is almost unchanged. Both A1 and R1 overwhelmingly choose VERIFY after T2. R1 introduces a small ~2% shift toward RETRIEVE, but the dominant action in both arms is VERIFY. The harm is not from choosing a different action type — it's from choosing different VERIFY targets.

---

## 4. VERIFY Forensic Audit

| Metric | R1 | A1 |
|--------|----|----|
| Total VERIFY actions | 1140 | 1140 |
| VERIFY_COMPLETED | 912 (80.0%) | 639 (56.1%) |
| INVALID_VERIFY_TARGET | 0 (0.0%) | 273 (24.0%) |
| RESOURCE_EXHAUSTED | 228 (20.0%) | 228 (20.0%) |
| Repeated target rate | 0.0% | 0.0% |
| Useful verify rate | 0.0% | — |
| Epistemic usefulness | OBSERVABLE (no state change detected) | — |

**Key finding:** This is the most diagnostic result. R1's M3 representation eliminates INVALID_VERIFY_TARGET entirely (0% vs 24% for A1). R1 always selects valid verification targets. But this does not help — R1's VERIFY_COMPLETED rate is higher (80% vs 56%), yet utility is worse.

The reason: A1's invalid verifies are "wasted but cheap" — they fail fast. R1's valid verifies are "successful but useless" — they complete without changing the epistemic state. R1 verifies the right targets in a technical sense, but those verifications do not change any hypothesis status.

**Useful verify rate = 0.0%**: Across all 1140 R1 VERIFY actions post-T2, not a single one changed the MDSG state (eliminated_hypotheses or live_hypotheses). Every R1 VERIFY is state-neutral. The model enters NEEDS_DISCRIMINATION, verifies valid targets, and remains in NEEDS_DISCRIMINATION until resources are exhausted.

---

## 5. Harm Conditioned on First Divergence

| Divergence class | n | Mean ΔU | R1 breaks | R1 rescues |
|------------------|---|---------|-----------|------------|
| VERIFY→VERIFY | 185 | −0.1743 | 0 | 0 |
| VERIFY→RETRIEVE | 4 | −2.1500 | 0 | 0 |

**Key finding:** Zero breaks and zero rescues across all 640 pairs. No trajectory pair has A1 success with R1 failure or vice versa. All trajectories fail in both arms. The harm is purely utility degradation within failed trajectories, not success/failure displacement.

The VERIFY→VERIFY class (target selection difference) accounts for the majority of harm at −0.1743 mean ΔU. The rare VERIFY→RETRIEVE cases are much worse per-instance (−2.15) but too few to drive the aggregate.

---

## 6. Persistent-M3 vs First-M3-Action Harm

| Metric | Value |
|--------|-------|
| n (T2-triggered pairs) | 228 |
| Mean consecutive VERIFY | 4.92 |
| First post-T2 action | VERIFY (228/228 = 100%) |

Harm by consecutive VERIFY bucket:

| Bucket | n | Mean ΔU | Mean repeated targets | Mean invalid |
|--------|---|---------|----------------------|--------------|
| 3-4 | 77 | −0.2513 | 0.0 | 0.0 |
| 5+ | 151 | −0.1424 | 0.0 | 0.0 |

**Key finding:** The first post-T2 action is always VERIFY (100%). There is no action-selection failure on the first M3 decision. The harm accumulates over persistent M3 steps.

Trajectories with 3-4 consecutive VERIFYs show worse harm (−0.25) than those with 5+ (−0.14). This is counterintuitive but explained by the step budget: shorter VERIFY runs that still fail represent more "wasted" verification, while longer runs at least exhaust the budget doing something consistent. Both buckets show zero repeated targets and zero invalid targets.

The critical pattern: R1 enters M3, selects valid VERIFY targets, completes those verifications, but **none change the epistemic state**. The model remains stuck in NEEDS_DISCRIMINATION with no live hypotheses, verifying evidence that doesn't move the decision forward. This is a representation-induced loop: the M3 packet makes the model "feel" it should keep verifying, even when no hypothesis can be confirmed or eliminated.

---

## 7. Rescue/Break Analysis

| Category | Count |
|----------|-------|
| Breaks (A1 success, R1 fail) | 0 |
| Rescues (A1 fail, R1 success) | 0 |

**Key finding:** There are no success/failure displacements at all. Every task fails in both arms. R1 does not break any A1 success, and R1 does not rescue any A1 failure. The entire harm is within-trajectory utility cost, not binary outcome changes.

---

## 8. Failure Mechanism Summary

The forensic evidence points to a specific failure chain:

```
T2 fires correctly (76/80 eligible, 0% false positive)
  → R1 routes into M3 representation
    → Model selects valid VERIFY targets (0% invalid vs 24% A1)
      → VERIFY completes successfully (80% vs 56%)
        → But no MDSG state changes (0% useful verify rate)
          → Model remains in NEEDS_DISCRIMINATION
            → Persistent VERIFY loop until RESOURCE_EXHAUSTED
              → Utility degraded by extra verification cost
```

The harm is **not** from:
- Wrong action type (both arms choose VERIFY ~98% of the time)
- Invalid targets (R1 has 0% invalid vs A1's 24%)
- Pre-T2 divergence (zero cases)
- Safety failures (zero false T2)
- Success/failure displacement (zero breaks/rescues)

The harm **is** from:
- Target selection that produces state-neutral verifications
- Persistent M3 representation that traps the model in a VERIFY loop
- The M3 packet making the model "over-deliberate" — verifying valid but irrelevant targets without changing the decision state

---

## 9. Implications for R2 Design

The forensic evidence suggests three testable hypotheses:

### H1: The full M3 packet is the problem
R1's M3 packet includes the complete MDSG state with affordances. This may overwhelm the model with state information, causing it to verify everything rather than make a decision.
**Test:** R2a — T2 flag only, no M3 packet. Tell the model "conflict detected" but let it choose actions with A1's representation.

### H2: Permanent latching is the problem
R1 latches into M3 permanently after T2. The model never returns to A1's simpler representation. This may trap it in an over-deliberative mode.
**Test:** R2c — Transient M3 for one decision, then return to A1. This preserves the "attention shift" without the persistent loop.

### H3: The intervention direction is wrong
T2 correctly identifies NEEDS_DISCRIMINATION, but routing the model to VERIFY is the wrong response. When all hypotheses are eliminated, verification cannot help — the model needs to retrieve new evidence or reason about the conflict, not verify existing evidence.
**Test:** R2b — Compact hypothesis summary that tells the model what to do next, not just what state it's in.

### Recommended priority

Based on the forensic evidence:

1. **R2c (transient M3)** — highest priority. The data shows the first M3 action is always VERIFY (same as A1), but persistent M3 creates the loop. Transient routing may preserve the attention shift without the trap.

2. **R2a (T2 flag only)** — second priority. If the harm comes from the M3 packet itself rather than the persistence, this will show it.

3. **R2b (compact summary)** — third priority. If the problem is that the model doesn't know what to do with the state information, a directive summary may help.

---

## 10. What Cannot Be Inferred

R13-F is observational, not interventional. It cannot confirm:

1. **That removing M3 will help.** The forensic data shows M3 is associated with harm, but removing it might not improve outcomes — the model might still fail in the same way with A1's representation.

2. **That transient M3 will avoid the loop.** The data shows persistent M3 creates loops, but a single M3 decision might still trigger the same target-selection failure.

3. **That the target selection is the causal mechanism.** R1 selects different targets than A1, and those targets are worse, but we cannot prove the M3 representation caused the worse selection — it could be a side effect of the model's response to the packet format.

4. **That the 0% useful verify rate is caused by M3.** A1 also shows state-neutral verification behavior — it just happens to select some invalid targets too. The fundamental problem (verifying without changing state) may be a model limitation, not a representation issue.

All hypotheses must be tested in new held-out development data with the proposed R2 variants.

---

## Technical Summary

| Property | Value |
|----------|-------|
| Label | POST_HOC_EXPLORATORY |
| Pairs | 640 |
| Pre-T2 divergences | 0 |
| First divergence: VERIFY→VERIFY (target) | 185 |
| First divergence: VERIFY→RETRIEVE | 4 |
| R1 invalid verify rate | 0.0% |
| A1 invalid verify rate | 24.0% |
| R1 useful verify rate | 0.0% |
| Breaks | 0 |
| Rescues | 0 |
| First post-T2 action | VERIFY (100%) |
| Mean consecutive VERIFY | 4.92 |
| Primary failure mode | State-neutral VERIFY loop in persistent M3 |
