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

**Finding**: Across all 2964 counterfactually simulated valid VERIFY targets (228 states × 13 targets each), none changed the MDSG decision state, live/eliminated hypothesis sets, or T2 status.

---

## 4. Elimination Monotonicity (Exhaustive)

| Check | Result |
|-------|--------|
| T2 states checked | 228/228 (exhaustive) |
| Total valid VERIFY targets tested | 2964 |
| Targets per state | 13 (uniform: all 228 states have exactly 13) |
| Monotonic | 228/228 |
| Violations | 0 |

**Empirically exhaustive invariant**:

> For all s ∈ S_T2, for all v ∈ ValidVerify(s): Eliminated(s) ⊆ Eliminated(T(s,v)).

This is an empirically exhaustive invariant over the 228 audited R13 T2 states. It is not a mathematical theorem over all possible DAPH states. However, the property follows from the transition semantics of the frozen executor:

### Formal Lemma (from transition semantics)

**Premises** (guaranteed by `EvidenceExecutor`):

1. A hypothesis h is ELIMINATED in state s because some evidence item e_c has `verification_state = SUFFICIENT` and `h ∈ e_c.contradicts`.
2. VERIFY acts only on an UNVERIFIED target e_v (where `e_v.verification_state = UNVERIFIED`).
3. VERIFY changes only e_v's state: `UNVERIFIED → SUFFICIENT | FALSIFIED | STALE | MISSING`.
4. VERIFY does not change any other evidence item's `verification_state`.
5. Therefore e_c's `SUFFICIENT` status persists after `VERIFY(e_v)`.
6. Therefore the contradiction responsible for h's elimination persists.

**Conclusion**:

```
ELIMINATED(h, s) ⟹ ELIMINATED(h, T(s, VERIFY(e_v)))
```

for any h and any valid VERIFY target e_v.

This lemma is confirmed by the exhaustive empirical invariant (228/228 states, 2964/2964 targets). The premises are guaranteed by the executor code at `hrm_adaptive_memory/executive/evidence_benchmark/executor.py` (SHA verified in preflight).

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

## 7. R2d — Structural Dead-End Affordance Gating (Hard Epistemic Admissibility)

**Renamed** from "decision-relevant affordance gating" to emphasize that the rule is structural, not predictive.

### Hard vs soft gate

R2d is a **hard epistemic admissibility gate**, not a soft affordance hint.

- **Soft gate**: packet says `can_verify=false` but VERIFY remains executable. Risk: model outputs VERIFY → executor accepts → model loops on VERIFY.
- **Hard gate**: VERIFY is removed from the admissible action set and cannot be executed. The model's action vocabulary is restricted before generation.

R2d uses hard gating. The action pipeline becomes:

```
ActionVocabulary → Legal → EpistemicallyAdmissible → LLM_Policy → Executor
```

Where:

```
Legal(a, s)              = budget_remaining(a) AND valid_target_exists(a, s)
EpistemicallyAdmissible(a, s) = Legal(a, s) AND NOT EpistemicDeadEnd(s)
Allowed(a, s)            = Legal(a, s) AND EpistemicallyAdmissible(a, s)
```

For VERIFY specifically:

```
EpistemicallyAdmissible(VERIFY, s) = NOT T2(s)
```

Equivalently: `T2 = true ⟹ VERIFY ∉ Allowed(s)`

The LLM still owns policy selection. The controller is not telling it what to do. It is restricting the action space to actions admissible under current public state.

### Enforcement

Prefer enforcing the hard gate **before generation** via the allowed-action schema/enum if the llama.cpp JSON-schema path supports it. This is cleaner than allowing VERIFY and rejecting it afterward, which risks a new loop:

```
model chooses VERIFY → executor rejects VERIFY → model chooses VERIFY again → new loop
```

### Rule (runtime-visible, non-leaky)

```python
can_verify = (
    verification_calls_remaining > 0
    and bool(valid_verify_targets)
    and not all_hypotheses_eliminated
)
```

where `all_hypotheses_eliminated = (n_hypotheses > 0 and len(eliminated_hypotheses) == n_hypotheses)`.

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

### Corrected label: NO_VIABLE_HYPOTHESIS

**Semantics**:

```
NO_VIABLE_HYPOTHESIS ⟺ |H| > 0 AND ∀h ∈ H, status(h) = ELIMINATED
```

This label is:
- **Descriptive**: it describes the observable state (all hypotheses eliminated)
- **Non-prescriptive**: it does not recommend an action (unlike NEEDS_NEW_EVIDENCE)
- **Narrow**: it applies only to the all-eliminated case (unlike INSUFFICIENT, which is broader)
- **Maps directly onto T2**: `NO_VIABLE_HYPOTHESIS ⟺ T2`

R2e changes the label ONLY for the all-eliminated case:

```
NEEDS_DISCRIMINATION → NO_VIABLE_HYPOTHESIS  (when all hypotheses eliminated)
NEEDS_DISCRIMINATION → NEEDS_DISCRIMINATION  (when ≥2 viable hypotheses remain)
```

This preserves NEEDS_DISCRIMINATION for its proper use: `|H_viable| ≥ 2` with evidence required to distinguish candidates.

---

## 9. Recommended Development Experiment

### Core 2×2 factorial (gate × semantics)

| Arm | State label | VERIFY admissibility |
|-----|-------------|----------------------|
| C0 | NEEDS_DISCRIMINATION | current |
| D | NEEDS_DISCRIMINATION | R2d structural hard gate |
| E | NO_VIABLE_HYPOTHESIS | current |
| DE | NO_VIABLE_HYPOTHESIS | R2d structural hard gate |

Everything else identical:
- same Gemma backend, same prompt (except experimentally required packet field)
- same retrieval (Q3_RERANKED), same semantic extractor, same MDSG computation
- same T2, same budgets, same utility, same model parameters
- same representation routing (persistent M3 / current routing)
- **no transient M3 yet**

Interpretable contrasts:

```
Δ_D       = U(D)  - U(C0)
Δ_E       = U(E)  - U(C0)
I_{D×E}   = [U(DE) - U(E)] - [U(D) - U(C0)]
```

### Second stage: representation factor (only after choosing best combination)

| Arm | Representation |
|-----|---------------|
| best(C0,D,E,DE) | persistent M3 / current routing |
| R2cde | transient M3 |

```
Effect_transient = U(R2cde) - U(best)
```

### Development dataset

Do NOT reuse R13 efficacy trajectories. Generate a new held-out seed/set.

Required strata:
- T2_IMMEDIATE, T2_LATE_1, T2_LATE_2, T2_LATE_3 / nontrigger
- MATCHED_NEG_IMMEDIATE, MATCHED_NEG_LATE
- DEFER_CONTROL, ANSWER_CONTROL

**Deliberately add semantic-error cases** where a false contradiction can incorrectly produce all-eliminated. This is critical because R2d makes T2 operationally stronger: a false-positive T2 now removes VERIFY, not just changes representation.

### Safety metric: FalseGateRate

```
FalseGateRate = P(can_verify switched false | gold says VERIFY remains epistemically relevant)
```

This must be a first-class gate. If R2d incorrectly gates VERIFY when VERIFY could still help, the intervention is harmful.

### Instrumentation

For every decision step, log:

```
legal_actions
epistemically_admissible_actions
allowed_actions
can_verify_legal
can_verify_epistemic
t2
decision_state
n_live
n_eliminated
valid_verify_target_count
selected_action
verify_gate_reason  (e.g. "ALL_HYPOTHESES_ELIMINATED" or null)
```

This makes the next forensic pass trivial.

### Qualification before efficacy

Because a hard action mask changes the policy's choice environment, requalify Gemma.

Test states where:
- VERIFY allowed
- VERIFY structurally gated
- RETRIEVE+SEARCH remain
- only DEFER/ANSWER remain

Require:
- decoder valid = 100%
- length failure = 0
- no gated action emitted/executed
- no action-schema failure

Then the usual action competence gates.

### Primary endpoints

For T2-triggered tasks:

```
Δ_D = E[U_D - U_C0]
```

Mechanistic endpoint:

```
P(VERIFY | T2, R2d) = 0  (if hard gate works)
```

Then track what replaces VERIFY:

```
P(RETRIEVE), P(SEARCH), P(DEFER), P(ANSWER), P(REASON_MORE)
```

### Important failure modes

R2d can still fail in several informative ways:
- VERIFY loop → RETRIEVE loop
- VERIFY loop → SEARCH loop
- VERIFY loop → premature DEFER
- VERIFY loop → unsupported ANSWER

Success is not just "VERIFY disappears." The goal is:

```
ΔU > 0  without control harm and without simply moving resource exhaustion to another action
```

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

1. **VERIFY is structurally useless at T2.** In 228/228 T2 states, 0/2964 valid VERIFY targets can change any epistemically decision-relevant state (decision_state, live/eliminated hypothesis sets, T2 status). Proven by counterfactual simulation.

2. **Elimination is monotonic (empirically exhaustive invariant + formal lemma).** For all 228 T2 states and all 2964 valid targets: Eliminated(s) ⊆ Eliminated(T(s,v)). The property follows from the executor's transition semantics (VERIFY only changes the target item's state, not other items' states). No VERIFY can un-eliminate a hypothesis.

3. **VERIFY is not the only available action.** RETRIEVE and SEARCH are also available at T2. The model actively chooses VERIFY, not because it's forced, but because NEEDS_DISCRIMINATION tells it to "discriminate."

4. **NEEDS_DISCRIMINATION is semantically wrong at T2.** When all hypotheses are eliminated, there is nothing to discriminate between. The counterfactual proves the unverified evidence cannot support any hypothesis. The corrected label is NO_VIABLE_HYPOTHESIS.

5. **The structural gating rule is sufficient.** `T2=true ⟹ VERIFY ∉ Allowed(s)` is validated by exhaustive counterfactual audit. No runtime simulation is needed. The rule is deterministic, public, and non-leaky.

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
| Total valid VERIFY targets tested | 2964 |
| T2_VERIFY_DEAD_END | 228 (100%) |
| T2_VERIFY_RESOLVABLE | 0 (0%) |
| T2_NO_VERIFY | 0 (0%) |
| Affordances at T2 | VERIFY + RETRIEVE + SEARCH (all 228) |
| Valid VERIFY targets per T2 state | 13 (uniform) |
| Epistemically useful VERIFY targets | 0/2964 = 0 |
| Elimination monotonicity | 228/228 (empirically exhaustive invariant) |
| Monotonicity lemma | From transition semantics (premises in executor code) |
| Decision state at T2 | NEEDS_DISCRIMINATION (228/228, semantically wrong) |
| R2d | Hard epistemic admissibility gate: `T2=true ⟹ VERIFY ∉ Allowed(s)` |
| R2e | NO_VIABLE_HYPOTHESIS label (orthogonal to R2d) |
| Core experiment | 2×2 factorial: C0, D, E, DE (gate × semantics) |
| Second stage | Representation factor (transient M3) |
| Safety metric | FalseGateRate = P(gate false \| gold says VERIFY relevant) |
| Architectural lesson | Legal(a,s) ≠ EpistemicallyAdmissible(a,s) |
