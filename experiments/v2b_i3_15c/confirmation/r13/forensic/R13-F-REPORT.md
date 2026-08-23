# R13-F1.2a: Counterfactual Affordance Audit (Hardened)

> **Label: POST_HOC_EXPLORATORY**
> **Version: R13-F1.2a** (supersedes R13-F1.1)
>
> R13-F can identify likely failure mechanisms; it cannot confirm them causally.
> Any hypothesis produced by R13-F must be tested in new held-out development data.

## Source

| Property | Value |
|----------|-------|
| R13_RESULTS_SHA256 | ad600240bf97cbbdb09126f542d1dc56605b11089b387684883be5784bb8a463 |
| R13_EXPERIMENT_SOURCE_COMMIT | 5454246b7e61adfb7a093eb5a1f731347071270d |
| R13_CONFIRMATION_EXECUTABLE_SHA256 | 41cc60b04f506f63b80c91e036d330d61d79992a86fb975cbe21597bd2d84f57 |
| R13_RETRIEVAL_RECEIPTS_SHA256 | 2329bfe2cf7f5c002ec019b0b9554a2727ee810957b9f74a364a203b99db8e1e |
| Source | raw_closed/ (immutable) + frozen executor + Q3_RERANKED retrieval receipts |
| R13_F_ANALYSIS_SHA256 (F1.1) | fe1f18258a40eb769d72821744f10da387cde62e3f198600dabe0f07022aaab8 |
| R13_F1_2_ANALYSIS_SHA256 | a1b7ec550b28c94ec998b6b396754fbe57084cfa0a35618f2fba352bcae77cc8 |
| Pairs | 640 (A1/R1 matched by task_id + retrieval_level + backend_identity) |
| T2-triggered trajectories audited | 228/228 |
| Identity preflight | 19/19 checks passed |

## R13-F1.2a Updates from R13-F1.1

1. **ALL_ELIMINATED relabeled** as a T2 structural consistency check, not a novel finding. T2 is defined as `len(eliminated) == n_hypotheses`, so all T2 activations must have all hypotheses eliminated by construction.
2. **Actual affordances audited** from reconstructed runtime state, not inferred from selected actions. VERIFY, RETRIEVE, and SEARCH are all available at T2 — the model is NOT forced to VERIFY.
3. **Counterfactual VERIFY value computed** by simulating every valid VERIFY target through the frozen executor. 0/228×13 targets can change any epistemically decision-relevant state.
4. **Elimination monotonicity checked exhaustively**: 228/228 T2 states and all valid targets show monotonic elimination. No VERIFY can un-eliminate a hypothesis.
5. **Decision-state semantics audited**: NEEDS_DISCRIMINATION is semantically wrong when 0 hypotheses are live. The counterfactual proves the unverified evidence cannot support any hypothesis.
6. **R2d renamed** to "Structural Dead-End Affordance Gating" — the rule is structural, not predictive. `T2=true ⟹ can_verify=false`. No counterfactual simulation at runtime.
7. **R2e decoupled** from R2d — not a prerequisite. The two are orthogonal causal questions.
8. **Hard preflight identity verification**: 19 identity properties verified against frozen experiment source commit.
9. **Post-hoc oracle vs runtime-visible distinction**: counterfactual simulation uses frozen hidden task effects (legitimate for diagnosis). R2d must derive from visible structural state only.
10. **Precise useful-target definition**: epistemically useful = changes to decision_state, live/eliminated hypothesis sets, or T2 status ONLY. Changes to verification_state, resource_state, or evidence metadata are NOT counted.

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
| T2_VERIFY_DEAD_END | 228 (100%) | Valid targets exist but 0 can change epistemic state |
| T2_VERIFY_RESOLVABLE | 0 (0%) | At least 1 target could change epistemic state |
| T2_NO_VERIFY | 0 (0%) | No legal VERIFY targets |

**Precise definition**: A VERIFY target is "epistemically useful" if and only if simulating it changes any of: decision_state, live_hypotheses, eliminated_hypotheses, or T2 status. Changes to verification_state, resource_state, evidence metadata, or prior_actions are NOT counted.

**Finding**: Across all 2964 counterfactually simulated valid VERIFY targets (228 states × 13 targets each), none changed the MDSG decision state, live/eliminated hypothesis sets, or T2 status.

### Elimination monotonicity (exhaustive)

| Check | Result |
|-------|--------|
| T2 states checked | 228/228 (exhaustive) |
| Total valid VERIFY targets tested | 2964 |
| Monotonic | 228/228 |
| Violations | 0 |

**Empirically exhaustive invariant**: For all s ∈ S_T2, for all v ∈ ValidVerify(s): Eliminated(s) ⊆ Eliminated(T(s,v)).

This is an empirically exhaustive invariant over the 228 audited R13 T2 states, not a mathematical theorem over all possible DAPH states. The property follows from the executor's transition semantics (formal lemma in R13-F1-2-REPORT.md): VERIFY only changes the target item's state, not other items' states, so the contradiction responsible for elimination persists.

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
            → Counterfactual: 0/13 targets can change epistemic state (proven by simulation)
              → All VERIFYs complete but change nothing epistemically
                → Model remains stuck until RESOURCE_EXHAUSTED
                  → Utility harm from wasted verification steps
```

The deepest failure is **not** the M3 representation or persistent latching. It is a three-layer problem:

1. **State semantics:** NEEDS_DISCRIMINATION is wrong when 0 hypotheses are live (R2e)
2. **Affordance exposure:** VERIFY is exposed as available when it is structurally useless (R2d)
3. **Model behavior:** The model chooses VERIFY even when RETRIEVE/SEARCH are available

R2d and R2e address layers 1 and 2 independently. The 2×2 factorial separates their effects.

---

## 10. Updated R2 Priority (F1.2a)

### R2d — Structural Dead-End Affordance Gating (HIGHEST PRIORITY)

**Renamed** from "decision-relevant affordance gating" — the rule is structural, not predictive.

**Hard epistemic admissibility gate** (not soft affordance hint):

```
Legal(a, s)              = budget_remaining(a) AND valid_target_exists(a, s)
EpistemicallyAdmissible(a, s) = Legal(a, s) AND NOT EpistemicDeadEnd(s)
Allowed(a, s)            = Legal(a, s) AND EpistemicallyAdmissible(a, s)
```

For VERIFY: `EpistemicallyAdmissible(VERIFY, s) = NOT T2(s)`

Equivalently: `T2 = true ⟹ VERIFY ∉ Allowed(s)`

VERIFY is removed from the admissible action set and cannot be executed. The LLM still owns policy selection — the controller restricts the action space, not the policy. Prefer enforcing before generation via allowed-action schema/enum to avoid: model chooses VERIFY → executor rejects → model chooses VERIFY again → new loop.

**Rule** (runtime-visible, non-leaky):

```python
can_verify = (
    verification_calls_remaining > 0
    and bool(valid_verify_targets)
    and not all_hypotheses_eliminated
)
```

This does NOT require counterfactual simulation, hidden verification outcomes, or oracle access. The F1.2a exhaustive counterfactual audit (228/228, 2964 targets) validates this rule post-hoc.

**Critical**: R2d must NOT be implemented as `can_verify = any(simulate_verify(t).changes_state for t in targets)`. That would recreate the future-information contamination that invalidated M2.

### R2e — State-Semantics Correction (ORTHOGONAL, NOT A PREREQUISITE)

R2e is **NOT a prerequisite** for R2d. The two are orthogonal causal questions:
- R2d: Does changing the affordance help? (gate vs no-gate)
- R2e: Does changing the semantic label help? (NEEDS_DISCRIMINATION vs corrected)

R2d can key directly from T2/all-eliminated state regardless of what label is displayed.

**Corrected label**: `NO_VIABLE_HYPOTHESIS`

```
NO_VIABLE_HYPOTHESIS ⟺ |H| > 0 AND ∀h ∈ H, status(h) = ELIMINATED
```

Descriptive, non-prescriptive, narrow (all-eliminated case only), maps directly onto T2. Preserves NEEDS_DISCRIMINATION for its proper use: |H_viable| ≥ 2.

### R2c — Transient M3 (SECOND STAGE)

Tested conditional on the best gate+semantics combination. Less fundamental — even with transient M3, if VERIFY is structurally useless, transient routing won't help.

### R2a/R2b — Lower priority

If the M3 packet is removed but VERIFY is still available and structurally useless, the model may still loop. R2d is more fundamental.

### Recommended development experiment

**Core 2×2 factorial** (gate × semantics):

| Arm | State label | VERIFY admissibility |
|-----|-------------|----------------------|
| C0 | NEEDS_DISCRIMINATION | current |
| D | NEEDS_DISCRIMINATION | R2d structural hard gate |
| E | NO_VIABLE_HYPOTHESIS | current |
| DE | NO_VIABLE_HYPOTHESIS | R2d structural hard gate |

Everything else identical: same backend, prompt, retrieval, extractor, MDSG, T2, budgets, utility, model parameters, representation routing. No transient M3 yet.

Contrasts:

```
Δ_D       = U(D)  - U(C0)
Δ_E       = U(E)  - U(C0)
I_{D×E}   = [U(DE) - U(E)] - [U(D) - U(C0)]
```

**Second stage** (representation factor, only after choosing best):

| Arm | Representation |
|-----|---------------|
| best(C0,D,E,DE) | persistent M3 / current routing |
| R2cde | transient M3 |

**Development dataset**: new held-out seed (do NOT reuse R13 efficacy trajectories). Include semantic-error cases where false contradictions produce all-eliminated. Add FalseGateRate = P(gate false | gold says VERIFY relevant) as first-class safety gate.

**Instrumentation**: log legal_actions, epistemically_admissible_actions, allowed_actions, can_verify_legal, can_verify_epistemic, t2, decision_state, n_live, n_eliminated, valid_verify_target_count, selected_action, verify_gate_reason.

**Qualification before efficacy**: requalify Gemma with hard action mask (decoder valid=100%, no gated action emitted, no schema failure).

**Primary endpoints**: Δ_D = E[U_D - U_C0]. Mechanistic: P(VERIFY|T2,R2d)=0. Track P(RETRIEVE), P(SEARCH), P(DEFER), P(ANSWER), P(REASON_MORE).

**Failure modes**: VERIFY loop → RETRIEVE/SEARCH loop, premature DEFER, unsupported ANSWER. Success requires ΔU > 0 without control harm.

---

## 11. What Cannot Be Inferred

R13-F1.2a is observational and counterfactual, not interventional. It cannot confirm:

1. **That R2d will improve outcomes.** Removing VERIFY from the admissible action set might cause the model to choose RETRIEVE, SEARCH, or DEFER — but those might also fail. Possible failure modes: VERIFY loop → RETRIEVE loop, VERIFY loop → SEARCH loop, VERIFY loop → premature DEFER, VERIFY loop → unsupported ANSWER. Success requires ΔU > 0 without control harm, not just "VERIFY disappears."

2. **That the state-semantics fix alone will help.** Changing the label from NEEDS_DISCRIMINATION to NO_VIABLE_HYPOTHESIS might cause the model to DEFER, which might be correct or might be premature.

3. **That the problem is not in the benchmark design.** The benchmark may be structured so that T2 always fires in a state where no action can help. If so, the correct response is DEFER, and the failure is that neither A1 nor R1 chooses DEFER.

4. **That A1's 31% repeated target rate is worse than R1's 0%.** A1's repeated targets might be "checking again" which is cheaper than R1's "checking new things that are also useless."

All hypotheses must be tested in new held-out development data.

---

## 12. Post-Hoc Oracle vs Runtime-Visible Logic

The F1.2a audit uses frozen hidden task effects (the executor's `verify_result` field) to simulate counterfactual outcomes. This is legitimate for post-hoc diagnosis.

R2d must NOT replicate this at runtime. A production controller cannot simulate future verification outcomes. If R2d were implemented as `can_verify = any(simulate_verify(t).changes_state for t in targets)`, it would recreate the future-information contamination that invalidated M2.

The F1.2a audit proves that no such simulation is needed. The structural rule `T2=true ⟹ can_verify=false` is sufficient, deterministic, and public.

---

## 13. Architectural Lesson

> **A valid action is not necessarily an epistemically admissible action.**

DAPH currently has:

```
ExecutableAction(s)
```

The evidence supports adding:

```
EpistemicallyAdmissibleAction(s)
```

So the action pipeline becomes:

```
ActionVocabulary → Legal → EpistemicallyAdmissible → LLM_Policy → Executor
```

Where:

```
Legal(a, s)              = budget_remaining(a) AND valid_target_exists(a, s)
EpistemicallyAdmissible(a, s) = Legal(a, s) AND NOT EpistemicDeadEnd(s)
Allowed(a, s)            = Legal(a, s) AND EpistemicallyAdmissible(a, s)
```

The controller should expose an action only when it is both executable AND capable, under the public structural semantics, of advancing the epistemic state. The LLM still owns policy selection — the controller restricts the action space, not the policy.

This distinction could become more important than R1 itself. R1 changed the representation but not the affordance logic. R2d changes the affordance logic itself — the set of actions the model is permitted to consider.

---

## 14. What R13-F1.2a Proves

1. **VERIFY is structurally useless at T2.** In 228/228 T2 states, 0/2964 valid VERIFY targets can change any epistemically decision-relevant state. Proven by counterfactual simulation.

2. **Elimination is monotonic (empirically exhaustive invariant + formal lemma).** For all 228 T2 states and all 2964 valid targets: Eliminated(s) ⊆ Eliminated(T(s,v)). The property follows from the executor's transition semantics.

3. **VERIFY is not the only available action.** RETRIEVE and SEARCH are also available at T2. The model actively chooses VERIFY.

4. **NEEDS_DISCRIMINATION is semantically wrong at T2.** Corrected label: NO_VIABLE_HYPOTHESIS.

5. **The structural gating rule is sufficient.** `T2=true ⟹ VERIFY ∉ Allowed(s)` is validated by exhaustive counterfactual audit. No runtime simulation is needed.

---

## Technical Summary

| Property | Value |
|----------|-------|
| Label | POST_HOC_EXPLORATORY |
| Version | R13-F1.2a |
| Identity preflight | 19/19 passed |
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
| **F1.2a: T2_VERIFY_DEAD_END** | **228/228 (100%)** |
| **F1.2a: T2_VERIFY_RESOLVABLE** | **0/228 (0%)** |
| **F1.2a: T2_NO_VERIFY** | **0/228 (0%)** |
| **F1.2a: Affordances at T2** | **VERIFY + RETRIEVE + SEARCH (all 228)** |
| **F1.2a: Total valid VERIFY targets tested** | **2964** |
| **F1.2a: Epistemically useful targets** | **0/2964 = 0** |
| **F1.2a: Elimination monotonicity** | **228/228 (empirically exhaustive invariant + formal lemma)** |
| **F1.2a: Decision state at T2** | **NEEDS_DISCRIMINATION (228/228, semantically wrong)** |
| **F1.2a: R2d** | **Hard epistemic admissibility gate: T2=true ⟹ VERIFY ∉ Allowed(s)** |
| **F1.2a: R2e** | **NO_VIABLE_HYPOTHESIS label (orthogonal to R2d)** |
| **F1.2a: Core experiment** | **2×2 factorial: C0, D, E, DE (gate × semantics)** |
| **F1.2a: Safety metric** | **FalseGateRate** |
| Primary failure mode | Structurally useless VERIFY when all hypotheses eliminated |
| Architectural lesson | Legal(a,s) ≠ EpistemicallyAdmissible(a,s) |
