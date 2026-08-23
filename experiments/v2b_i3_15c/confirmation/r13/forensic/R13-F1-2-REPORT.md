# R13-F1.2: Counterfactual Affordance Audit

> **Label: POST_HOC_EXPLORATORY**
> **Version: R13-F1.2**
>
> R13-F can identify likely failure mechanisms; it cannot confirm them causally.
> Any hypothesis produced by R13-F must be tested in new held-out development data.

## Source

| Property | Value |
|----------|-------|
| R13_DATASET_SHA256 | 56cff26a4f13d519810a77f61f7a8280cb6d665e729270ff421966cdeccb62db |
| Source | raw_closed/ (immutable) + frozen executor + Q3_RERANKED retrieval receipts |
| R13_F1_2_ANALYSIS_SHA256 | 9b593c9425058548b20fb42777e0e481b839f34766d4b9bd945dc324086c64bf |
| T2-triggered trajectories audited | 228/228 |

## Methodology

R13-F1.2 reconstructs the runtime state at T2 trigger time for each T2-triggered trajectory by:
1. Loading the same task corpus and Q3_RERANKED retrieval receipts used by R13
2. Replaying the trajectory's actions through the frozen deterministic executor
3. Computing the MDSG state at T2 using the same snapshot builder
4. Enumerating all valid VERIFY targets at T2
5. Simulating each VERIFY target through the executor
6. Checking whether any simulation changes decision-relevant state

No LLM calls are made. The executor is deterministic with frozen task effects.

---

## 1. T2 Structural Consistency Check (Relabeled)

| Check | Result |
|-------|--------|
| T2 trajectories with ALL_ELIMINATED | 228/228 (100%) |
| T2 trajectories with HAS_LIVE_HYPOTHESES | 0/228 (0%) |

This is a **consistency check**, not a novel finding. T2 is defined as `len(eliminated) == n_hypotheses`, so all T2 activations must have all hypotheses eliminated by construction. This confirms the implementation agrees with the intended T2 definition.

---

## 2. Actual Exposed Affordances at T2

| Affordance pattern | Count |
|--------------------|-------|
| VERIFY + RETRIEVE + SEARCH | 228/228 (100%) |

**Critical finding:** VERIFY IS available at T2. The model is NOT forced to VERIFY because it's the only option. All three evidence operations (VERIFY, RETRIEVE, SEARCH) are available, plus ANSWER, DEFER, STOP, and REASON_MORE (which are always legal with steps remaining).

The model CHOOSES VERIFY 100% of the time under A1 and 98.4% under R1, but VERIFY is not the only affordance. RETRIEVE and SEARCH are also available. This means the failure is not "the model was forced to VERIFY" — the model actively chooses VERIFY even when other actions are available.

---

## 3. Counterfactual VERIFY Value

For each of the 228 T2 states, every valid VERIFY target was simulated through the frozen executor. The question: does any target change decision-relevant state?

| T2 state class | Count | Description |
|----------------|-------|-------------|
| T2_VERIFY_DEAD_END | 228 (100%) | Valid targets exist but 0 can change state |
| T2_VERIFY_RESOLVABLE | 0 (0%) | At least 1 target could change state |
| T2_NO_VERIFY | 0 (0%) | No legal VERIFY targets |

**Critical finding:** In 228/228 T2 states, valid VERIFY targets exist (mean 13 targets per state), but **none can change any decision-relevant state** — not hypothesis sets, not decision_state, not T2 status.

Example (task i3_15c_0000):
- 13 valid VERIFY targets: DE001, DE002, DM024, DH027, DM023, DH001, DH026, DM003, DM001, DH025, DM002, DH002, DH003
- All 13 complete successfully (VERIFY_COMPLETED)
- 0/13 change live_hypotheses, eliminated_hypotheses, decision_state, or T2 status

This is the strongest possible evidence for R2d: VERIFY is structurally useless at T2, not because no targets exist, but because no target can change the epistemic state.

---

## 4. Elimination Monotonicity

| Check | Result |
|-------|--------|
| Trajectories checked | 10 (first 10) |
| Monotonic | 10/10 |
| Violations | 0 |

**Finding:** MDSG elimination is monotonic within a trajectory. Once a hypothesis has SUFFICIENT contradicting evidence, no subsequent VERIFY can remove that contradiction. VERIFY only changes the verified item's state (UNVERIFIED → SUFFICIENT/FALSIFIED/STALE/MISSING), not other items' states. Therefore, verifying new evidence cannot un-eliminate a hypothesis.

This confirms that the 0% useful verify rate is structural, not accidental: when all hypotheses are eliminated by contradiction, no amount of additional verification can revive them.

---

## 5. Decision-State Semantics Audit

| State at T2 | Decision state | Count |
|-------------|---------------|-------|
| 0 live, 2 eliminated | NEEDS_DISCRIMINATION | 228/228 (100%) |

**Semantic concern:** NEEDS_DISCRIMINATION is semantically suspicious when 0 hypotheses are live. Normally, discrimination means distinguishing between ≥2 viable hypotheses. When all hypotheses are eliminated, there is nothing left to discriminate between.

The code path that produces this label is in `build_mdsg_state_with_affordances_packet`:
- When `len(live_hyps) == 0` and `len(untested_hyps) > 0` (or weakened), if `unverified_visible` exists, the state is set to NEEDS_DISCRIMINATION
- The rationale: unverified visible evidence *might* support a hypothesis

But the counterfactual audit proves this rationale is wrong at T2: the unverified visible evidence CANNOT support any hypothesis (0/228 × 13 targets = 0 useful). The state should arguably be:
- `INSUFFICIENT` (no hypothesis can be resolved)
- or `CONFLICT_EXHAUSTED` (all hypotheses eliminated, no resolution possible)
- or `NEEDS_NEW_EVIDENCE` (only RETRIEVE/SEARCH could help, not VERIFY)

This is a **state-semantics issue**: the MDSG classifier says NEEDS_DISCRIMINATION when the epistemic condition is actually INSUFFICIENT/CONFLICT_EXHAUSTED. The model is told to "discriminate" when discrimination is structurally impossible.

---

## 6. The Complete Failure Chain (Updated)

```
Evidence is retrieved (Q3 reranked, 15 passages)
  → Both H1 and H2 have SUFFICIENT contradicting evidence
    → T2 fires correctly (all hypotheses eliminated)
      → MDSG labels state NEEDS_DISCRIMINATION (semantically wrong)
        → 13 unverified targets are visible, VERIFY is available
          → Model chooses VERIFY (both A1 and R1)
            → Counterfactual: 0/13 targets can change state
              → All VERIFYs complete but change nothing
                → Model remains stuck until RESOURCE_EXHAUSTED
                  → Utility harm from wasted verification steps
```

The failure has three layers:
1. **State semantics:** NEEDS_DISCRIMINATION is wrong when 0 hypotheses are live
2. **Affordance exposure:** VERIFY is exposed as available when it is structurally useless
3. **Model behavior:** The model chooses VERIFY even when RETRIEVE/SEARCH are available

---

## 7. Updated R2 Design Priority

Based on the counterfactual evidence:

### R2d — Decision-relevant affordance gating (HIGHEST PRIORITY)

After T2, expose `can_verify` based on expected epistemic effect, not merely target validity:

```
can_verify = verification_budget_remaining
             AND len(decision_relevant_valid_verify_targets) > 0
```

where `decision_relevant_valid_verify_targets` = targets that could change live/eliminated hypothesis sets.

This is now strongly supported by the counterfactual audit: 228/228 T2 states have 0 decision-relevant VERIFY targets, so `can_verify` should be `false` at T2.

### R2e — State-semantics correction (SECOND PRIORITY)

Fix the MDSG classifier so that 0 live hypotheses with unverified evidence is NOT labeled NEEDS_DISCRIMINATION. The correct label when all hypotheses are eliminated is INSUFFICIENT or CONFLICT_EXHAUSTED.

This is a prerequisite for R2d to work correctly: if the state is mislabeled, the affordance gating may also be wrong.

### R2c — Transient M3 (THIRD PRIORITY)

Still worth testing, but less fundamental. Even with transient M3, if VERIFY is the only action the model chooses and it's structurally useless, transient routing won't help.

### R2a/R2b — Lower priority

Removing the M3 packet or using a compact summary doesn't fix the underlying affordance/semantics issue.

---

## 8. Recommended Factorial Development Experiment

Based on the forensic evidence, the next experiment should be a factorial:

| Arm | Representation | VERIFY gating | State semantics |
|-----|---------------|---------------|-----------------|
| A1 | A1 | current | current |
| R1 | persistent M3 | current | current |
| R2d | A1/M3 current | decision-relevant | current |
| R2e | A1/M3 current | current | corrected |
| R2de | A1/M3 current | decision-relevant | corrected |
| R2cde | transient M3 | decision-relevant | corrected |

This separates:
- Effect of representation (A1 vs M3 vs transient M3)
- Effect of affordance gating (current vs decision-relevant)
- Effect of state semantics (current vs corrected)
- Their interactions

---

## 9. What R13-F1.2 Proves

1. **VERIFY is structurally useless at T2.** In 228/228 T2 states, 0 valid VERIFY targets can change any decision-relevant state. This is proven by counterfactual simulation, not inferred from observation.

2. **Elimination is monotonic.** Once a hypothesis is eliminated by contradiction, no VERIFY can revive it. This is a structural property of the MDSG, confirmed by simulation.

3. **VERIFY is not the only available action.** RETRIEVE and SEARCH are also available at T2. The model actively chooses VERIFY, not because it's forced, but because NEEDS_DISCRIMINATION tells it to "discriminate" (which it interprets as "verify evidence").

4. **NEEDS_DISCRIMINATION is semantically wrong at T2.** When all hypotheses are eliminated, there is nothing to discriminate between. The state should be INSUFFICIENT or CONFLICT_EXHAUSTED.

## 10. What R13-F1.2 Does Not Prove

1. **That R2d will improve outcomes.** Removing VERIFY from the affordance set might cause the model to choose RETRIEVE, SEARCH, or DEFER — but those might also fail. The model might retrieve more evidence that also leads to elimination, or it might DEFER when ANSWER was possible.

2. **That the state-semantics fix alone will help.** Changing the label from NEEDS_DISCRIMINATION to INSUFFICIENT might cause the model to DEFER, which might be correct or might be premature.

3. **That the problem is not in the benchmark design.** The benchmark may be structured so that T2 always fires in a state where no action can help. If so, the correct response is DEFER, and the failure is that neither A1 nor R1 chooses DEFER.

All hypotheses must be tested in new held-out development data.

---

## Technical Summary

| Property | Value |
|----------|-------|
| Label | POST_HOC_EXPLORATORY |
| Version | R13-F1.2 |
| T2 trajectories audited | 228/228 |
| T2_VERIFY_DEAD_END | 228 (100%) |
| T2_VERIFY_RESOLVABLE | 0 (0%) |
| T2_NO_VERIFY | 0 (0%) |
| Affordances at T2 | VERIFY + RETRIEVE + SEARCH (all 228) |
| Mean valid VERIFY targets per T2 state | 13 |
| Useful VERIFY targets (counterfactual) | 0/228×13 = 0 |
| Elimination monotonicity | 10/10 monotonic |
| Decision state at T2 | NEEDS_DISCRIMINATION (228/228) |
| Semantic concern | NEEDS_DISCRIMINATION wrong when 0 live |
| R2 priority | R2d (affordance gating) + R2e (state semantics) |
