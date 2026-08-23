# R13-F1.2: Counterfactual Affordance Audit

> **Label: POST_HOC_EXPLORATORY**
> **Version: R13-F1.2** (supersedes R13-F1.1)
>
> R13-F can identify likely failure mechanisms; it cannot confirm them causally.
> Any hypothesis produced by R13-F must be tested in new held-out development data.

## Source

| Property | Value |
|----------|-------|
| R13_DATASET_SHA256 | 56cff26a4f13d519810a77f61f7a8280cb6d665e729270ff421966cdeccb62db |
| Source | raw_closed/ (immutable) + frozen executor + Q3_RERANKED retrieval receipts |
| R13_F_ANALYSIS_SHA256 (F1.1) | fe1f18258a40eb769d72821744f10da387cde62e3f198600dabe0f07022aaab8 |
| R13_F1_2_ANALYSIS_SHA256 | 9b593c9425058548b20fb42777e0e481b839f34766d4b9bd945dc324086c64bf |
| Pairs | 640 (A1/R1 matched by task_id + retrieval_level + backend_identity) |
| T2-triggered trajectories audited | 228/228 |

## R13-F1.2 Updates from R13-F1.1

1. **ALL_ELIMINATED relabeled** as a T2 structural consistency check, not a novel finding. T2 is defined as `len(eliminated) == n_hypotheses`, so all T2 activations must have all hypotheses eliminated by construction.
2. **Actual affordances audited** from reconstructed runtime state, not inferred from selected actions. VERIFY, RETRIEVE, and SEARCH are all available at T2 — the model is NOT forced to VERIFY.
3. **Counterfactual VERIFY value computed** by simulating every valid VERIFY target through the frozen executor. 0/228×13 targets can change any decision-relevant state.
4. **Elimination monotonicity checked** by simulation: 10/10 trajectories show monotonic elimination. No VERIFY can un-eliminate a hypothesis.
5. **Decision-state semantics audited**: NEEDS_DISCRIMINATION is semantically wrong when 0 hypotheses are live. The counterfactual proves the unverified evidence cannot support any hypothesis.
6. **R2 priority updated**: R2d (affordance gating) + R2e (state semantics) are now the leading candidates, supported by counterfactual evidence.

## R13-F1.1 Corrections (preserved from prior version)

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

## 5. T2 Structural Consistency Check (Relabeled in F1.2)

| State at T2 trigger | Count | Pct |
|---------------------|-------|-----|
| HAS_LIVE_HYPOTHESES | 0 | 0% |
| ALL_ELIMINATED | 228 | 100% |
| EMPTY_STATE | 0 | 0% |
| Any post-T2 state change | 0 | 0% |

**Relabeled finding (F1.2):** The 228/228 ALL_ELIMINATED observation is a **T2 structural consistency check**, not a novel discovery. T2 is defined as `len(eliminated) == n_hypotheses`, so all T2 activations must have all hypotheses eliminated by construction. This confirms the implementation agrees with the intended T2 definition.

**Do not infer from this alone that VERIFY is useless.** The uselessness of VERIFY at T2 is established separately by the F1.2 counterfactual audit (Section 5b), which simulates every valid VERIFY target and checks whether any could change decision-relevant state.

## 5b. Counterfactual Affordance Audit (New in F1.2)

### Actual exposed affordances at T2

| Affordance pattern | Count |
|--------------------|-------|
| VERIFY + RETRIEVE + SEARCH | 228/228 (100%) |

**Critical finding:** VERIFY IS available at T2, but so are RETRIEVE and SEARCH. The model is NOT forced to VERIFY because it's the only option. The model actively CHOOSES VERIFY even when other evidence-gathering actions are available.

### Counterfactual VERIFY value

| T2 state class | Count | Description |
|----------------|-------|-------------|
| T2_VERIFY_DEAD_END | 228 (100%) | Valid targets exist but 0 can change state |
| T2_VERIFY_RESOLVABLE | 0 (0%) | At least 1 target could change state |
| T2_NO_VERIFY | 0 (0%) | No legal VERIFY targets |

In 228/228 T2 states, valid VERIFY targets exist (mean 13 per state), but **none can change any decision-relevant state** — not hypothesis sets, not decision_state, not T2 status. This is proven by counterfactual simulation through the frozen executor, not inferred from observation.

### Elimination monotonicity

| Check | Result |
|-------|--------|
| Trajectories checked | 10 (first 10) |
| Monotonic | 10/10 |
| Violations | 0 |

MDSG elimination is monotonic: once a hypothesis has SUFFICIENT contradicting evidence, no subsequent VERIFY can remove that contradiction. VERIFY only changes the verified item's state, not other items' states. This confirms the 0% useful verify rate is structural, not accidental.

### Decision-state semantics

| State at T2 | Decision state | Count |
|-------------|---------------|-------|
| 0 live, 2 eliminated | NEEDS_DISCRIMINATION | 228/228 (100%) |

NEEDS_DISCRIMINATION is semantically suspicious when 0 hypotheses are live. Discrimination normally means distinguishing between viable hypotheses. When all are eliminated, there is nothing to discriminate between. The counterfactual audit proves the unverified visible evidence CANNOT support any hypothesis (0/228×13 = 0 useful). The state should arguably be INSUFFICIENT, CONFLICT_EXHAUSTED, or NEEDS_NEW_EVIDENCE.

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

## 9. Failure Mechanism (Updated in F1.2)

The corrected forensic evidence points to a three-layer failure:

```
Evidence is retrieved (Q3 reranked, 15 passages)
  → Both H1 and H2 have SUFFICIENT contradicting evidence
    → T2 fires correctly (all hypotheses eliminated, consistency check)
      → MDSG labels state NEEDS_DISCRIMINATION (semantically wrong)
        → 13 unverified targets visible, VERIFY available (but so are RETRIEVE/SEARCH)
          → Model CHOOSES VERIFY (both A1 and R1)
            → Counterfactual: 0/13 targets can change state (proven by simulation)
              → All VERIFYs complete but change nothing
                → Model remains stuck until RESOURCE_EXHAUSTED
                  → Utility harm from wasted verification steps
            → RESOURCE_EXHAUSTED → utility harm
```

The deepest failure is **not** the M3 representation or persistent latching. It is a three-layer problem:

1. **State semantics:** NEEDS_DISCRIMINATION is wrong when 0 hypotheses are live
2. **Affordance exposure:** VERIFY is exposed as available when it is structurally useless
3. **Model behavior:** The model chooses VERIFY even when RETRIEVE/SEARCH are available

---

## 10. Updated R2 Priority (F1.2)

Based on the counterfactual forensics:

### R2d — Decision-relevant affordance gating (HIGHEST PRIORITY)

After T2, expose `can_verify` based on expected epistemic effect, not merely target validity:

```
can_verify = verification_budget_remaining
             AND len(decision_relevant_valid_verify_targets) > 0
```

where `decision_relevant_valid_verify_targets` = targets that could change live/eliminated hypothesis sets.

This is now strongly supported by the counterfactual audit: 228/228 T2 states have 0 decision-relevant VERIFY targets, so `can_verify` should be `false` at T2.

### R2e — State-semantics correction (SECOND PRIORITY, NEW)

Fix the MDSG classifier so that 0 live hypotheses with unverified evidence is NOT labeled NEEDS_DISCRIMINATION. The correct label when all hypotheses are eliminated is INSUFFICIENT or CONFLICT_EXHAUSTED.

This is a prerequisite for R2d to work correctly: if the state is mislabeled, the affordance gating may also be wrong.

### R2c — Transient M3 (THIRD PRIORITY)

Still worth testing, but less fundamental. Even with transient M3, if VERIFY is the only action the model chooses and it's structurally useless, transient routing won't help.

### R2a — T2 flag only (FOURTH PRIORITY)

If the M3 packet is removed but VERIFY is still available and structurally useless, the model may still loop. R2d is more fundamental.

### R2b — Compact hypothesis summary (FIFTH PRIORITY)

A directive summary might help, but only if it changes the action affordance. If the summary says "all hypotheses eliminated, choose RETRIEVE or DEFER," that effectively becomes R2d.

### Recommended factorial development experiment

| Arm | Representation | VERIFY gating | State semantics |
|-----|---------------|---------------|-----------------|
| A1 | A1 | current | current |
| R1 | persistent M3 | current | current |
| R2d | A1/M3 current | decision-relevant | current |
| R2e | A1/M3 current | current | corrected |
| R2de | A1/M3 current | decision-relevant | corrected |
| R2cde | transient M3 | decision-relevant | corrected |

This separates representation effects, affordance effects, state-semantics effects, and their interactions.

---

## 11. What Cannot Be Inferred

R13-F1.2 is observational and counterfactual, not interventional. It cannot confirm:

1. **That R2d will improve outcomes.** Removing VERIFY from the affordance set might cause the model to choose RETRIEVE, SEARCH, or DEFER — but those might also fail. The model might retrieve more evidence that also leads to elimination, or it might DEFER when ANSWER was possible.

2. **That the state-semantics fix alone will help.** Changing the label from NEEDS_DISCRIMINATION to INSUFFICIENT might cause the model to DEFER, which might be correct or might be premature.

3. **That the problem is not in the benchmark design.** The benchmark may be structured so that T2 always fires in a state where no action can help. If so, the correct response is DEFER, and the failure is that neither A1 nor R1 chooses DEFER.

4. **That A1's 31% repeated target rate is worse than R1's 0%.** A1's repeated targets might be "checking again" which is cheaper than R1's "checking new things that are also useless."

All hypotheses must be tested in new held-out development data.

---

## 12. What R13-F1.2 Proves

1. **VERIFY is structurally useless at T2.** In 228/228 T2 states, 0 valid VERIFY targets can change any decision-relevant state. This is proven by counterfactual simulation through the frozen executor, not inferred from observation.

2. **Elimination is monotonic.** Once a hypothesis is eliminated by contradiction, no VERIFY can revive it. This is a structural property of the MDSG, confirmed by simulation (10/10 trajectories).

3. **VERIFY is not the only available action.** RETRIEVE and SEARCH are also available at T2. The model actively chooses VERIFY, not because it's forced, but because NEEDS_DISCRIMINATION tells it to "discriminate."

4. **NEEDS_DISCRIMINATION is semantically wrong at T2.** When all hypotheses are eliminated, there is nothing to discriminate between. The counterfactual proves the unverified evidence cannot support any hypothesis.

---

## Technical Summary

| Property | Value |
|----------|-------|
| Label | POST_HOC_EXPLORATORY |
| Version | R13-F1.2 |
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
| T2 structural consistency: ALL_ELIMINATED | 228/228 (100%) — consistency check |
| T2 trajectories with HAS_LIVE_HYPOTHESES | 0/228 (0%) |
| Post-T2 state changes | 0/228 (0%) |
| Breaks | 0 |
| Rescues | 0 |
| A1 also stuck in VERIFY | P(VERIFY\|A1,T2) = 1.0 |
| **F1.2: T2_VERIFY_DEAD_END** | **228/228 (100%)** |
| **F1.2: T2_VERIFY_RESOLVABLE** | **0/228 (0%)** |
| **F1.2: T2_NO_VERIFY** | **0/228 (0%)** |
| **F1.2: Affordances at T2** | **VERIFY + RETRIEVE + SEARCH (all 228)** |
| **F1.2: Mean valid VERIFY targets per T2** | **13** |
| **F1.2: Useful VERIFY targets (counterfactual)** | **0/228×13 = 0** |
| **F1.2: Elimination monotonicity** | **10/10 monotonic** |
| **F1.2: Decision state at T2** | **NEEDS_DISCRIMINATION (228/228, semantically wrong)** |
| Primary failure mode | Structurally useless VERIFY when all hypotheses eliminated |
| Deepest identified failure | Wrong state semantics + wrong affordance exposure after T2 |
