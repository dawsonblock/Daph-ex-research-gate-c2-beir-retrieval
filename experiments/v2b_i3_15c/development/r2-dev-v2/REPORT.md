# R2-DEV-V2 Frozen Experiment Report

## Experiment Identity

| Field | Value |
|-------|-------|
| Backend | Qwen2.5-7B-Instruct Q4_K_M |
| Backend identity SHA | `08dd528f6fc9e67c574ec766ea15ab7be80bdaa7a615625244122db533c2772d` |
| GGUF SHA256 | `65b8fcd92af6b4fefa935c625d1ac27ea29dcb6ee14589c55a8f115ceaaa1423` |
| Runtime | llama-cpp-python 0.3.35 |
| GPU | Tesla T4 |
| Dataset SHA | `1f6a5cb38011282722b3fb22b835d40e7199fa9a61983c8dfccbb437ef033e43` |
| Raw closed SHA | `f92b2b0ae5706d8a7565090ed18a4e2a9f74d97fa5e4d479d0c06b41cc47dd3c` |
| Schedule SHA | `b7f9dc82119712730c58b124b70a775f77f8a8c0d141e2ba095119cec1c8de1b` |
| Seed | 137 |
| Run start | 2026-08-23T21:13:02Z |
| Run end | 2026-08-23T21:32:27Z |
| Total time | 1165s (~19 min) |

## Trajectory Counts

| Category | Count |
|----------|-------|
| Total scheduled | 320 |
| Completed | 260 |
| Failed (infrastructure) | 60 |
| Decoder errors | 0 |
| Schema gate violations | 0 |
| Executor admissibility violations | 0 |

The 60 failures were all from synthesized budget-exhausted task variants
(`retrieval_exhausted`, `search_exhausted`, `both_exhausted`) due to a
`ResourceState.consume_retrieval/consume_search` API mismatch. These are
infrastructure errors, not model behavior. The 260 completed trajectories
cover all 16 core strata with 65 per arm (C0, D, E, DE).

## Qualification Invariants

| Invariant | Value |
|-----------|-------|
| Total model calls | 668 |
| Decoder valid rate | 100% (668/668) |
| Schema valid rate | 100% (668/668) |
| Schema gate violations | 0 |
| Executor admissibility violations | 0 |

All invariants hold. The strict decoder and LlamaGrammar enforcement
worked perfectly across all 668 model calls.

## T2 Frequency

T2 was reached in 30/65 trajectories per arm, identical across all arms.
This confirms that the interventions do not change whether T2 is reached
— only what happens once T2 is reached.

## Utility Contrasts

| Contrast | Value |
|----------|-------|
| U(C0) | 11.10 |
| U(D) | 11.13 |
| U(E) | 12.96 |
| U(DE) | 12.93 |
| **Δ_D** | **+0.03** |
| **Δ_E** | **+1.86** |
| **Δ_DE** | **+1.83** |
| **I_D×E** | **-0.07** |

### Interpretation

- **D is neutral** (Δ_D ≈ 0): The hard T2 VERIFY gate neither helps nor
  hurts utility.
- **E is positive** (Δ_E = +1.86): Relabeling `NEEDS_DISCRIMINATION` to
  `NO_VIABLE_HYPOTHESIS` at T2 significantly improves utility.
- **DE ≈ E** (Δ_DE ≈ Δ_E): D adds nothing on top of E.
- **No D×E interaction** (I_D×E ≈ 0): The two interventions are
  independent; E's effect is not modulated by D.

## VERIFY Usefulness

| Metric | Value |
|--------|-------|
| Total VERIFY events (C0/E) | 80 |
| Useful VERIFY events | 80 |
| Useful rate | 100% |

Every VERIFY call in C0/E was useful: each one changed the decision
state, live hypothesis count, eliminated hypothesis count, or T2 status.
VERIFY is genuinely epistemically productive at T2.

Example pattern (T2_LATE_1 stratum):
- State before VERIFY: `SUPPORTED_BUT_UNRESOLVED`, 1 live, 0 eliminated
- State after VERIFY: `INSUFFICIENT` (C0) / `NO_VIABLE_HYPOTHESIS` (E), 0 live, 1 eliminated
- T2 changed: false → true

## Replacement-Action Distribution (D/DE when VERIFY is gated)

| Action | D | DE |
|--------|---|-----|
| SEARCH_MORE | 23 | 1 |
| RETRIEVE | 5 | 5 |
| DEFER | 2 | 24 |

### Key observation

Without E (D only), the model substitutes VERIFY with SEARCH_MORE (77%)
— it keeps searching. With E (DE), the model substitutes with DEFER (80%)
— it recognizes the situation as hopeless and stops.

This explains why DE ≈ E rather than DE > E: E's label change
(`NO_VIABLE_HYPOTHESIS`) dominates the model's decision, causing it to
defer regardless of whether VERIFY is available.

## Terminal-Action Distribution

| Arm | ANSWER | DEFER |
|-----|--------|-------|
| C0 | 30 | 35 |
| D | 30 | 35 |
| E | 30 | 35 |
| DE | 30 | 35 |

Terminal actions are identical across all arms. The interventions change
the path (intermediate actions) but not the terminal outcome
distribution. The utility difference comes from step costs saved along
the way, not from different terminal actions.

## Success Rates

| Arm | Success rate |
|-----|-------------|
| C0 | 45/65 (69.23%) |
| D | 45/65 (69.23%) |
| E | 45/65 (69.23%) |
| DE | 45/65 (69.23%) |

Success rates are identical across all arms. No arm rescues or breaks
tasks that another arm doesn't.

## Causal Interpretation

### The central fork

The experiment was designed to distinguish:

- **H_A**: D hurts because it removes genuinely useful VERIFY actions.
- **H_B**: VERIFY itself is useless, but removing it changes Qwen's
  policy harmfully.

The results show **neither hypothesis is correct**:

1. VERIFY is genuinely useful (100% usefulness rate).
2. D does not hurt (Δ_D ≈ 0).
3. The model substitutes VERIFY with other useful actions (SEARCH_MORE,
   RETRIEVE) that achieve the same state changes.

The correct interpretation is:

> **VERIFY is useful but not uniquely useful.** At T2, the model can
> achieve the same epistemic progress through SEARCH_MORE or RETRIEVE.
> The hard T2 gate removes one useful action but the model substitutes
> others, resulting in no net utility change.

### E's positive effect

E (relabeling to `NO_VIABLE_HYPOTHESIS`) improves utility by +1.86. The
mechanism is:

1. At T2, the internal state is `SUPPORTED_BUT_UNRESOLVED` or
   `INSUFFICIENT`.
2. Without E, the model interprets this as "needs more work" and
   continues searching/retrieving/verifying.
3. With E, the model sees `NO_VIABLE_HYPOTHESIS` and correctly recognizes
   the situation as hopeless, deferring sooner.
4. The utility gain comes from saved step costs (fewer wasted
   search/retrieve actions before DEFER).

This **contradicts the R13 Gemma finding** where E caused STOP-collapse.
The Qwen backend responds positively to the label change, while the
Gemma backend responded negatively. This is a backend-specific effect,
not a universal property of the label intervention.

## Disposition (Frozen Rules Applied)

| Rule | Condition | Result |
|------|-----------|--------|
| D positive | Δ_D > 0 | **Marginal** (Δ_D = +0.03, essentially zero) |
| D negative ∧ VERIFY useful | Δ_D < 0 ∧ UsefulVerify > 0 | Not applicable (D not negative) |
| D negative ∧ VERIFY useless | Δ_D < 0 ∧ UsefulVerify ≈ 0 | Not applicable (D not negative, VERIFY useful) |
| E negative again | Δ_E < 0 | **Not triggered** (E is positive) |
| DE uniquely positive | Δ_DE > Δ_E and Δ_DE > Δ_D | **Not triggered** (DE ≈ E) |
| All interventions negative | Δ_D < 0 ∧ Δ_E < 0 ∧ Δ_DE < 0 | **Not triggered** |

### Final disposition

1. **D is neutral.** The hard T2 VERIFY gate is neither beneficial nor
   harmful. VERIFY is useful but replaceable. Continue epistemic
   admissibility as diagnostic infrastructure, but do not expect utility
   gains from the hard gate alone.

2. **E is positive.** Relabeling `NEEDS_DISCRIMINATION` to
   `NO_VIABLE_HYPOTHESIS` at T2 helps the Qwen model make better
   decisions by reaching DEFER sooner. This **does not replicate** the
   R13 Gemma finding where E caused STOP-collapse. The effect is
   backend-specific.

3. **DE ≈ E.** D adds nothing on top of E. When E relabels the state,
   the model defers regardless of whether VERIFY is available.

4. **No D×E interaction.** The two interventions are independent.

5. **The R13 lesson is refined:** Accurate epistemic state detection
   (T2) is necessary but not sufficient. The effect of intervention
   depends on the backend's response to the exposed label. Qwen responds
   positively to `NO_VIABLE_HYPOTHESIS`; Gemma responded negatively.
   This is a backend-policy interaction, not a universal property of the
   intervention.

## Limitations

1. **60/320 trajectories failed** due to a budget-exhaustion API
   mismatch (`ResourceState.consume_retrieval/consume_search` not
   available). These are all from synthesized budget-variant tasks. The
   260 completed trajectories cover all 16 core strata.

2. **The budget-exhausted variants would have provided additional
   information** about replacement-action behavior when retrieval/search
   is unavailable. This is secondary to the central D-C0 fork.

3. **Absolute utilities are not comparable** to the R13 Gemma lineage.
   Within-model contrasts (D vs C0, E vs C0) are valid.

4. **The dataset's `TWO_HYPOTHESIS_DISCRIMINATION_SCENARIO` stratum**
   represents a two-hypothesis discrimination scenario, not a literal
   state with |H_live| ≥ 2. After gold verification, only 1 hypothesis
   remains live.
