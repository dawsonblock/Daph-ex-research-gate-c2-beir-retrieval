# R13-F1.1: Corrected Post-Hoc Exploratory Forensic Analysis

> **Label: POST_HOC_EXPLORATORY**
> **Version: R13-F1.1** (corrected from R13-F1)
>
> R13-F can identify likely failure mechanisms; it cannot confirm them causally.
> Any hypothesis produced by R13-F must be tested in new held-out development data.

## Source

| Property | Value |
|----------|-------|
| R13_DATASET_SHA256 | 56cff26a4f13d519810a77f61f7a8280cb6d665e729270ff421966cdeccb62db |
| Source | raw_closed/ (immutable) |
| R13_F_ANALYSIS_SHA256 | fe1f18258a40eb769d72821744f10da387cde62e3f198600dabe0f07022aaab8 |
| Pairs | 640 (A1/R1 matched by task_id + retrieval_level + backend_identity) |

## R13-F1.1 Corrections from R13-F1

1. **Prefix comparison** now uses full step signatures `(action, target_id, execution_outcome)`, not just action labels. This ensures VERIFY(E1) vs VERIFY(E7) is correctly detected as a divergence.
2. **RepeatedTargetRate** is now computed trajectory-locally. Numerator and denominator are aggregated across trajectories, never comparing targets across trajectory boundaries.
3. **UsefulVerify** now reports two variants (V1: hypothesis sets, V2: expanded with decision_state and representation) with explicit denominators. Terminal VERIFY actions without adjacent state snapshots are classified as NOT_OBSERVABLE, not as useless.
4. **New: VERIFY target-value analysis** — classifies whether any VERIFY target could theoretically change the epistemic state at T2 time.
5. **New: A1 vs R1 target quality comparison** — measures target overlap between arms.
6. **Corrected causal language** throughout — "associated with" not "creates" or "causes."

---

## 1. Pre-T2 Variance Classification (Corrected)

| Class | Count | Description |
|-------|-------|-------------|
| IMMEDIATE_T2 | 76 | T2 triggered at step 0 (no pre-T2 prefix) |
| PREFIX_IDENTICAL | 152 | A1/R1 step signatures (action, target, outcome) identical through trigger_step−1 |
| PRE_T2_DIVERGED | 0 | Trajectories diverged before T2 |
| NO_TRIGGER | 412 | R1 did not trigger T2 |

**Key finding:** Zero pre-T2 divergences even with full (action, target, outcome) signatures. All 152 late-trigger pairs have identical A1/R1 action sequences AND identical target selections AND identical outcomes before T2 fires. All observed harm is attributable to the post-T2 intervention.

---

## 2. First Post-T2 Divergence Matrix

| Transition | Count | Mean ΔU |
|------------|-------|---------|
| VERIFY→VERIFY (different target) | 185 | −0.1743 |
| VERIFY→RETRIEVE | 4 | −2.1500 |
| No divergence | 451 | — |

**Key finding:** In 185 of 189 divergent pairs, both arms execute VERIFY but select different targets. The action label is identical; the target differs. Only 4 pairs show an action-type divergence.

---

## 3. Action Distribution Displacement ΔP(a|T2)

| Stratum | Action | P_R1 | P_A1 | ΔP |
|---------|--------|------|------|-----|
| ALL_T2 | VERIFY | 0.9836 | 1.0000 | −0.0164 |
| ALL_T2 | RETRIEVE | 0.0164 | 0.0000 | +0.0164 |

**A1 baseline observation:** P(VERIFY|A1,T2) = 1.0. A1 is also stuck in VERIFY ~100% of the time post-T2. R1 does not introduce VERIFY behavior — both policies are trapped in the same broad action mode. R1 introduces a small ~2% shift toward RETRIEVE.

---

## 4. VERIFY Forensic Audit (Corrected)

| Metric | R1 | A1 |
|--------|----|----|
| Total VERIFY actions | 1140 | 1140 |
| VERIFY_COMPLETED | 912 (80.0%) | 639 (56.1%) |
| INVALID_VERIFY_TARGET | 0 (0.0%) | 273 (24.0%) |
| RESOURCE_EXHAUSTED | 228 (20.0%) | 228 (20.0%) |
| Repeated target rate (trajectory-local) | **0.0%** (0/912) | **31.1%** (284/912) |
| UsefulVerifyV1 (hypothesis sets) | **0/912** | — |
| UsefulVerifyV2 (expanded) | **0/912** | — |
| Not observable (terminal) | 228 | — |

**Corrected claim:** Among 912 R1 VERIFY transitions for which adjacent state snapshots were available, none changed the live- or eliminated-hypothesis sets (V1), nor the decision_state or representation (V2). The remaining 228 VERIFY actions were terminal RESOURCE_EXHAUSTED calls without a following state snapshot and cannot be classified by this method.

**New finding — repeated targets:** A1 repeats verification targets 31.1% of the time (284/912 adjacent pairs). R1 never repeats (0/912). R1's M3 representation causes the model to select a different valid target each step, while A1 sometimes re-verifies the same target. R1 explores more targets, but none are useful.

---

## 5. VERIFY Target-Value Analysis (New)

| State at T2 trigger | Count | Pct |
|---------------------|-------|-----|
| HAS_LIVE_HYPOTHESES | 0 | 0% |
| ALL_ELIMINATED | 228 | 100% |
| EMPTY_STATE | 0 | 0% |
| Any post-T2 state change | 0 | 0% |

**Critical finding:** In 228/228 (100%) of T2-triggered trajectories, all hypotheses are already eliminated at the moment T2 fires. There are zero live hypotheses that verification could potentially confirm or eliminate. VERIFY is **structurally useless** regardless of which target the model selects.

This is the most important forensic finding. The failure is not:
- Wrong action selection (both arms choose VERIFY)
- Wrong target selection (R1 selects valid targets, 0% invalid)
- Persistent M3 trapping the model (though it is associated with the loop)

The failure is:
- **T2 fires when all hypotheses are already eliminated**
- **The only available action affordance is VERIFY**
- **VERIFY cannot change the state when no hypotheses are live**
- **The model is forced to verify evidence that cannot move the decision forward**

The problem may be bigger than R1: the action affordance itself is wrong after T2 when no live hypotheses remain.

---

## 6. A1 vs R1 Target Quality Comparison (New)

| Metric | Value |
|--------|-------|
| n (T2-triggered pairs) | 228 |
| Identical target sets | 48/228 (21.1%) |
| Mean shared targets | 3.09 |
| Mean R1-only targets | 1.91 |
| Mean A1-only targets | 0.41 |

**Finding:** R1 and A1 select different verification targets in 79% of pairs. R1 explores more unique targets (1.91 R1-only vs 0.41 A1-only). But since all targets are structurally useless (no live hypotheses), the target selection difference only affects cost, not outcome.

---

## 7. Harm Conditioned on First Divergence

| Divergence class | n | Mean ΔU | R1 breaks | R1 rescues |
|------------------|---|---------|-----------|------------|
| VERIFY→VERIFY (target diff) | 185 | −0.1743 | 0 | 0 |
| VERIFY→RETRIEVE | 4 | −2.1500 | 0 | 0 |

Zero breaks and zero rescues. All tasks fail in both arms. Harm is purely utility cost within failed trajectories.

---

## 8. Persistent-M3 Association

| Metric | Value |
|--------|-------|
| n (T2-triggered) | 228 |
| Mean consecutive VERIFY | 4.92 |
| First post-T2 action | VERIFY (228/228 = 100%) |

**Corrected language:** Persistent M3 is **associated with** sustained VERIFY behavior and negative incremental utility. Whether persistence itself causes the loop is exactly what R2c must test. R13-F is observational and cannot confirm this causal claim.

---

## 9. Failure Mechanism (Updated)

The corrected forensic evidence points to a deeper failure than representation persistence:

```
T2 fires when ALL hypotheses already eliminated (228/228 = 100%)
  → Only affordance: VERIFY
    → VERIFY cannot change state (0 live hypotheses)
      → Both A1 and R1 stuck in VERIFY loop
        → R1's M3 selects different valid targets (0% invalid, 0% repeated)
          → But those targets are all structurally useless
            → RESOURCE_EXHAUSTED → utility harm
```

The deepest failure is **not** the M3 representation or persistent latching. It is:

**The action affordance is wrong after T2 when no live hypotheses remain.**

T2 correctly identifies NEEDS_DISCRIMINATION, but the only available action is VERIFY, which cannot help when there is nothing left to verify. The model needs to retrieve new evidence, reason about the conflict, or make a decision — not verify already-eliminated hypotheses.

---

## 10. Updated R2 Priority

Based on the corrected forensics:

### R2d — Decision-relevant affordance gating (HIGHEST PRIORITY)

After T2, expose `can_verify=false` if no visible verification target can change the epistemic state. When all hypotheses are eliminated, the model must choose among:
- RETRIEVE (get new evidence)
- SEARCH_MORE (find additional sources)
- DEFER (acknowledge inability to resolve)
- ANSWER (make best guess)
- REASON_MORE (deliberate without new evidence)

This addresses the deepest failure: the affordance itself is wrong.

### R2c — Transient M3 (SECOND PRIORITY)

Still worth testing. Persistent M3 is associated with the VERIFY loop. But R2d is more fundamental — even transient M3 with VERIFY-only affordance would fail.

### R2a — T2 flag only (THIRD PRIORITY)

If the M3 packet is removed but VERIFY is still the only affordance, the model may still loop. R2d is more fundamental.

### R2b — Compact hypothesis summary (FOURTH PRIORITY)

A directive summary might help, but only if it changes the action affordance. If the summary says "all hypotheses eliminated, choose RETRIEVE or DEFER," that effectively becomes R2d.

---

## 11. What Cannot Be Inferred

R13-F1.1 is observational, not interventional. It cannot confirm:

1. **That affordance gating will help.** The data shows VERIFY is structurally useless when all hypotheses are eliminated, but removing VERIFY from the affordance set might not improve outcomes — the model might still fail with RETRIEVE or DEFER.

2. **That the 100% ALL_ELIMINATED rate is caused by T2 timing.** T2 might fire at the right time, but the benchmark may be structured so that all hypotheses are always eliminated before a decision can be made.

3. **That A1's 31% repeated target rate is worse than R1's 0%.** A1's repeated targets might be "checking again" which is cheaper than R1's "checking new things that are also useless."

4. **That the 0% useful verify rate is caused by the affordance rather than the benchmark.** The benchmark may not have decision-relevant evidence available at T2 time, making any VERIFY useless regardless of affordance design.

All hypotheses must be tested in new held-out development data.

---

## Technical Summary

| Property | Value |
|----------|-------|
| Label | POST_HOC_EXPLORATORY |
| Version | R13-F1.1 |
| Pairs | 640 |
| Pre-T2 divergences (full signatures) | 0 |
| First divergence: VERIFY→VERIFY (target) | 185 |
| R1 invalid verify rate | 0.0% |
| A1 invalid verify rate | 24.0% |
| R1 repeated target rate (trajectory-local) | 0.0% |
| A1 repeated target rate (trajectory-local) | 31.1% |
| UsefulVerifyV1 (R1) | 0/912 |
| UsefulVerifyV2 (R1) | 0/912 |
| Not observable (terminal) | 228 |
| T2 trajectories with ALL_ELIMINATED | 228/228 (100%) |
| T2 trajectories with HAS_LIVE_HYPOTHESES | 0/228 (0%) |
| Post-T2 state changes | 0/228 (0%) |
| Breaks | 0 |
| Rescues | 0 |
| A1 also stuck in VERIFY | P(VERIFY\|A1,T2) = 1.0 |
| Primary failure mode | Structurally useless VERIFY when all hypotheses eliminated |
| Deepest identified failure | Wrong action affordance after T2, not merely wrong representation |
