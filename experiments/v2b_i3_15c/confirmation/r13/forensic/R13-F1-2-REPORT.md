# R13-F1.2a: Counterfactual Affordance Audit (Hardened)

> **Label: POST_HOC_EXPLORATORY**
> **Version: R13-F1.2a** (hardened from F1.2)
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
| R13_F1_2_ANALYSIS_SHA256 | a1b7ec550b28c94ec998b6b396754fbe57084cfa0a35618f2fba352bcae77cc8 |
| T2-triggered trajectories audited | 228/228 |
| Identity preflight | 19/19 checks passed |

## F1.2a Hardening from F1.2

1. **Hard preflight identity verification**: verifies results.jsonl SHA, dataset manifest, experiment source commit, confirmation executable SHA, protocol SHA, GGUF SHA, retrieval receipts SHA, and 6 source file SHAs (executor, schema, task generator, i3_7e, i3_12j, i3_15_r1) against the frozen experiment source commit. Aborts on any mismatch.

2. **Exhaustive monotonicity**: runs across all 228/228 T2 states and every valid VERIFY target in each (not just first 10). The structural gating theorem is now empirically exhaustive for the frozen benchmark.

3. **Precise useful-target definition**: "epistemically useful" = changes to decision_state, live_hypotheses, eliminated_hypotheses, or T2 status ONLY. Changes to verification_state, resource_state, evidence metadata (verified_count, supporting_count, etc.), or prior_actions/outcomes are NOT counted.

4. **Post-hoc oracle vs runtime-visible distinction**: the counterfactual simulation uses frozen hidden task effects (legitimate for diagnosis). R2d must NOT use counterfactual simulation at runtime — it must derive from visible structural state only.

---

## 1. T2 Structural Consistency Check

| Check | Result |
|-------|--------|
| T2 trajectories with ALL_ELIMINATED | 228/228 (100%) |
| T2 trajectories with HAS_LIVE_HYPOTHESES | 0/228 (0%) |

This is a **T2 structural consistency check**, not a novel finding. T2 is defined as `len(eliminated) == n_hypotheses`, so all T2 activations must have all hypotheses eliminated by construction.

---

## 2. Actual Exposed Affordances at T2

| Affordance pattern | Count |
|--------------------|-------|
| VERIFY + RETRIEVE + SEARCH | 228/228 (100%) |

VERIFY IS available at T2, but so are RETRIEVE and SEARCH. The model is NOT forced to VERIFY. The model actively CHOOSES VERIFY even when other evidence-gathering actions are available.

---

## 3. Counterfactual VERIFY Value

For each of the 228 T2 states, every valid VERIFY target was simulated through the frozen executor (post-hoc oracle).

**Definition**: A VERIFY target is "epistemically useful" if and only if simulating it changes any of:
- `decision_state` (MDSG label)
- `live_hypotheses` set
- `eliminated_hypotheses` set
- T2 status (all-eliminated flag)

Changes to the following are NOT counted as epistemically useful:
- `verification_state` of individual evidence items
- `resource_state` (budgets remaining)
- evidence metadata (`verified_count`, `supporting_count`, etc.)
- `prior_actions` / `prior_outcomes` logs

| T2 state class | Count | Description |
|----------------|-------|-------------|
| T2_VERIFY_DEAD_END | 228 (100%) | Valid targets exist but 0 can change epistemic state |
| T2_VERIFY_RESOLVABLE | 0 (0%) | At least 1 target could change epistemic state |
| T2_NO_VERIFY | 0 (0%) | No legal VERIFY targets |

**Finding**: Across all counterfactually simulated valid VERIFY targets, none changed the MDSG decision state, live/eliminated hypothesis sets, or T2 status.

---

## 4. Elimination Monotonicity (Exhaustive)

| Check | Result |
|-------|--------|
| T2 states checked | 228/228 (exhaustive) |
| Valid targets tested per state | All (mean 13) |
| Monotonic | 228/228 |
| Violations | 0 |

**Empirically exhaustive theorem**:

> For all s ∈ S_T2, for all v ∈ ValidVerify(s): Eliminated(s) ⊆ Eliminated(T(s,v)).

MDSG elimination is monotonic: once a hypothesis has SUFFICIENT contradicting evidence, no subsequent VERIFY can remove that contradiction. VERIFY only changes the verified item's state, not other items' states. This is a structural property of the MDSG, confirmed exhaustively across all 228 T2 states and all valid targets.

---

## 5. Decision-State Semantics Audit

| State at T2 | Decision state | Count |
|-------------|---------------|-------|
| 0 live, 2 eliminated | NEEDS_DISCRIMINATION | 228/228 (100%) |

NEEDS_DISCRIMINATION is semantically suspicious when 0 hypotheses are live. Discrimination normally means distinguishing between viable hypotheses. When all are eliminated, there is nothing to discriminate between. The counterfactual audit proves the unverified visible evidence CANNOT support any hypothesis (0/228×13 = 0 epistemically useful).

The state should arguably be:
- `INSUFFICIENT` (no hypothesis can be resolved)
- `CONFLICT_EXHAUSTED` (all hypotheses eliminated, no resolution possible)
- `NEEDS_NEW_EVIDENCE` (only RETRIEVE/SEARCH could help, not VERIFY)

However, the semantics correction (R2e) is **orthogonal** to the affordance gating (R2d). R2d can key directly from the T2/all-eliminated structural state regardless of what label is displayed.

---

## 6. Post-Hoc Oracle vs Runtime-Visible Logic

**Critical design constraint**:

The F1.2a audit uses frozen hidden task effects (the executor's `verify_result` field) to simulate counterfactual outcomes. This is legitimate for post-hoc diagnosis.

R2d must NOT replicate this at runtime. A production controller cannot simulate future verification outcomes. If R2d were implemented as:

```python
can_verify = any(simulate_verify(t).changes_epistemic_state for t in valid_targets)
```

it would recreate the future-information contamination that invalidated M2.

The good news: F1.2a proves that no such simulation is needed. The structural rule is sufficient:

```python
can_verify = (
    verification_calls_remaining > 0
    and bool(valid_verify_targets)
    and not all_hypotheses_eliminated
)
```

where `all_hypotheses_eliminated = (n_hypotheses > 0 and len(eliminated_hypotheses) == n_hypotheses)`.

This is deterministic, public, and requires no hidden information. The exhaustive monotonicity result (228/228) validates that this rule is correct for the frozen benchmark: whenever all hypotheses are eliminated, no VERIFY can change the epistemic state.

---

## 7. R2d — Structural Dead-End Affordance Gating

**Renamed** from "decision-relevant affordance gating" to emphasize that the rule is structural, not predictive.

**Rule** (runtime-visible, non-leaky):

```
can_verify = budget_verify > 0
             AND |ValidVerifyTargets| > 0
             AND NOT EpistemicDeadEnd(s)

where EpistemicDeadEnd(s) = (
    n_hypotheses > 0
    AND len(eliminated_hypotheses) == n_hypotheses
)
```

Equivalently: `T2 = true ⟹ can_verify = false`

This does NOT require:
- Counterfactual simulation
- Hidden verification outcomes
- Oracle access to task effects
- Prediction of individual target outcomes

It requires ONLY:
- The current visible hypothesis sets (already computed for the MDSG packet)
- The current verification budget (already tracked)
- The current valid verify target set (already computed)

---

## 8. R2e — State-Semantics Correction (Orthogonal to R2d)

R2e is **NOT a prerequisite** for R2d. The two are orthogonal causal questions:

- **R2d**: Does changing the affordance help? (gate vs no-gate)
- **R2e**: Does changing the semantic label help? (NEEDS_DISCRIMINATION vs corrected)

R2d can key directly from T2/all-eliminated state even if the displayed label remains NEEDS_DISCRIMINATION. This separation is scientifically useful because it allows independent measurement of each effect.

Candidate corrected labels (semantics to be defined before naming):
- `CONFLICT_EXHAUSTED` — all hypotheses eliminated by contradiction
- `INSUFFICIENT` — no hypothesis can be resolved with available evidence
- `NEEDS_NEW_EVIDENCE` — only RETRIEVE/SEARCH could help

---

## 9. Recommended Development Experiment

### Core 2×2 factorial (gate × semantics)

| Arm | State label | VERIFY gate |
|-----|-------------|-------------|
| R1/current | NEEDS_DISCRIMINATION | current |
| R2d | NEEDS_DISCRIMINATION | structural dead-end gate |
| R2e | corrected label | current |
| R2de | corrected label | structural dead-end gate |

Interpretable contrasts:

```
Effect_gate       = U(R2d)  - U(R1)
Effect_semantics  = U(R2e)  - U(R1)
Interaction_{gate,semantics} = [U(R2de) - U(R2e)] - [U(R2d) - U(R1)]
```

### Second stage: representation factor

| Arm | Representation |
|-----|---------------|
| R2de | persistent M3 / current routing |
| R2cde | transient M3 |
| optionally R2ade | A1 + T2 flag |

```
Effect_transient = U(R2cde) - U(R2de)
```

This is cleaner than running six loosely related variants. The core 2×2 isolates gate and semantics effects. The second stage isolates representation effects conditional on the best gate+semantics combination.

---

## 10. Architectural Lesson

> **A valid action is not necessarily an epistemically admissible action.**

DAPH currently computes legal VERIFY from budget + target validity:

```
Legal(a, s) = budget_remaining(a) AND valid_target_exists(a, s)
```

F1.2a suggests it needs a second layer:

```
EpistemicallyAdmissible(a, s) = Legal(a, s) AND NOT EpistemicDeadEnd(s)
```

The controller should expose an action only when it is both executable AND capable, under the public structural semantics, of advancing the epistemic state.

This distinction could become more important than R1 itself. The R1 intervention changed the representation but not the affordance logic. R2d changes the affordance logic itself — the set of actions the model is permitted to consider.

---

## 11. What R13-F1.2a Proves

1. **VERIFY is structurally useless at T2.** In 228/228 T2 states, 0 valid VERIFY targets can change any epistemically decision-relevant state (decision_state, live/eliminated hypothesis sets, T2 status). Proven by counterfactual simulation.

2. **Elimination is monotonic (exhaustive).** For all 228 T2 states and all valid targets: Eliminated(s) ⊆ Eliminated(T(s,v)). No VERIFY can un-eliminate a hypothesis.

3. **VERIFY is not the only available action.** RETRIEVE and SEARCH are also available at T2. The model actively chooses VERIFY.

4. **NEEDS_DISCRIMINATION is semantically wrong at T2.** When all hypotheses are eliminated, there is nothing to discriminate between.

5. **The structural gating rule is sufficient.** `T2 = true ⟹ can_verify = false` is validated by exhaustive counterfactual audit. No runtime simulation is needed.

## 12. What R13-F1.2a Does Not Prove

1. **That R2d will improve outcomes.** Removing VERIFY might cause the model to choose RETRIEVE, SEARCH, or DEFER — but those might also fail.

2. **That the state-semantics fix alone will help.** Changing the label might cause DEFER, which might be correct or premature.

3. **That the problem is not in the benchmark design.** The benchmark may be structured so that T2 always fires in an unsolvable state.

All hypotheses must be tested in new held-out development data.

---

## Technical Summary

| Property | Value |
|----------|-------|
| Label | POST_HOC_EXPLORATORY |
| Version | R13-F1.2a |
| Identity preflight | 19/19 passed |
| T2 trajectories audited | 228/228 |
| T2_VERIFY_DEAD_END | 228 (100%) |
| T2_VERIFY_RESOLVABLE | 0 (0%) |
| T2_NO_VERIFY | 0 (0%) |
| Affordances at T2 | VERIFY + RETRIEVE + SEARCH (all 228) |
| Mean valid VERIFY targets per T2 | 13 |
| Epistemically useful VERIFY targets | 0/228×13 = 0 |
| Elimination monotonicity | 228/228 (exhaustive) |
| Decision state at T2 | NEEDS_DISCRIMINATION (228/228, semantically wrong) |
| R2d rule | `T2=true ⟹ can_verify=false` (structural, non-leaky) |
| R2e | Orthogonal to R2d (not a prerequisite) |
| Core experiment | 2×2 factorial: gate × semantics |
| Second stage | Representation factor (transient M3) |
