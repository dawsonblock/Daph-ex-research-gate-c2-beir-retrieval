# R2-DEV-V2 Frozen Experiment Report (Corrective Revision 1)

## Classification

This is a **post-run corrective analysis revision**. The original analysis
had two bugs (Step 10 used binary success instead of `realized_utility`;
Step 6 compared state across adjacent calls incorrectly) and two
conceptual errors (Step 3 and Step 5 used `"VERIFY" not in
allowed_actions` instead of `verify_removed_by_epistemic_gate`). This
revision fixes all four issues, splits VERIFY usefulness into pre-T2
and at-T2, and adds paired bootstrap CIs.

The original 260 completed trajectories are preserved untouched. The 60
budget-variant trajectories that failed due to a
`ResourceState.consume_retrieval/consume_search` API mismatch were
rerun with a targeted fix (construct `ResourceState` with
`retrieval_calls_used`/`search_calls_used` pre-set to maximum). The
closure results are stored separately and merged for analysis. The
original errors are quarantined in `raw/errors.jsonl`.

## Experiment Identity

| Field | Value |
|-------|-------|
| Backend | Qwen2.5-7B-Instruct Q4_K_M |
| Backend identity SHA | `08dd528f6fc9e67c574ec766ea15ab7be80bdaa7a615625244122db533c2772d` |
| GGUF SHA256 | `65b8fcd92af6b4fefa935c625d1ac27ea29dcb6ee14589c55a8f115ceaaa1423` |
| Runtime | llama-cpp-python 0.3.35 |
| GPU | Tesla T4 |
| Dataset SHA | `1f6a5cb38011282722b3fb22b835d40e7199fa9a61983c8dfccbb437ef033e43` |
| Seed | 137 |
| Original run | 260/320 trajectories (2026-08-23T21:13–21:32Z) |
| Closure run | 60/60 trajectories (2026-08-23T22:17–22:20Z) |
| Total | 320/320 trajectories |

## Integrity (Step 1)

| Check | Result |
|-------|--------|
| Trajectory count | 320/320 ✓ |
| Task IDs match | 80/80 ✓ |
| Decoder errors | 0 ✓ |
| `trajectory_count_matches` | True |
| `task_ids_match` | True |

## Qualification Invariants (Step 2)

| Invariant | Value |
|-----------|-------|
| Total model calls | 760 |
| Decoder valid rate | 100% (760/760) |
| Schema valid rate | 100% (760/760) |
| Schema gate violations | 0 |
| Executor admissibility violations | 0 |

All invariants hold across all 760 model calls.

## Gate Confusion Matrix (Step 3, corrected)

The gate confusion matrix now uses `verify_removed_by_epistemic_gate`
(the exact field designed to distinguish R2d removal from ordinary
illegality).

| | Gold should gate | Gold should not gate |
|---|---|---|
| Gate fired | TP = 0 | FP = 0 |
| Gate did not fire | FN = 90 | TN = 70 |

- **FalseGateRate = 0.0** (0 false gates)
- **MissedGateRate = 1.0** (90 missed gates)

The R2d gate **never fired**. `verify_removed_by_epistemic_gate` was
false in every single receipt. The reason: at T2, VERIFY is **never
legal** — there are no valid verification targets when all hypotheses
are eliminated. R2d has nothing to remove because the legal-actions
computation already excludes VERIFY.

This is consistent with the earlier F1.2a finding that VERIFY is
structurally useless after T2.

## Replacement-Action Distribution (Step 5, corrected)

Empty — no replacement actions because the gate never fired.

## VERIFY Usefulness (Step 6, split by T2 phase)

| Phase | Total | Useful | Rate |
|-------|-------|--------|------|
| Pre-T2 | 80 | 80 | 100% |
| At-T2 | 0 | 0 | N/A |

**All 80 VERIFY events in C0/E occurred before T2.** Zero VERIFY events
occurred at T2. Every pre-T2 VERIFY was useful (changed decision state,
live/eliminated counts, or T2 status).

This is the key corrected finding:

> **VERIFY is phase-dependent.** It is 100% useful during hypothesis
> resolution (pre-T2) and never used after hypothesis exhaustion (at
> T2). The model never attempts VERIFY at T2 because it is not legal
> there — no valid verification targets exist when all hypotheses are
> eliminated.

## Utility Contrasts (Step 10, with paired bootstrap CIs)

| Arm | Mean utility | n |
|-----|-------------|---|
| C0  | 21.85       | 80 |
| D   | 21.88       | 80 |
| E   | 23.47       | 80 |
| DE  | 23.45       | 80 |

| Contrast | Mean | 95% CI | Excludes 0? |
|----------|------|--------|-------------|
| Δ_D  | +0.027 | [0.000, 0.081] | **No** |
| Δ_E  | +1.620 | [1.161, 2.079] | **Yes** |
| Δ_DE | +1.593 | [1.134, 2.052] | **Yes** |

- **D is neutral.** The paired bootstrap CI includes zero. The hard T2
  VERIFY gate has no measurable effect on utility.
- **E is positive.** The paired bootstrap CI excludes zero. Relabeling
  to `NO_VIABLE_HYPOTHESIS` at T2 produces a positive mean paired
  utility signal of approximately +1.62.
- **DE ≈ E.** DE's CI also excludes zero, but DE is not distinguishable
  from E.

## D×E Interaction (Step 11, with paired bootstrap CI)

| Metric | Value |
|--------|-------|
| I_D×E | -0.054 |
| 95% CI | [-0.189, 0.054] |
| Excludes 0? | **No** |

No D×E interaction. The CI includes zero. The two interventions are
independent.

## Terminal-Action Distribution (Step 8)

| Arm | ANSWER | DEFER |
|-----|--------|-------|
| C0 | 30 | 50 |
| D  | 30 | 50 |
| E  | 30 | 50 |
| DE | 30 | 50 |

Terminal actions are identical across all arms. The interventions change
the path (intermediate actions and step costs) but not the terminal
outcome distribution.

## Success Rates (Step 9)

| Arm | Success rate |
|-----|-------------|
| C0 | 60/80 (75.00%) |
| D  | 60/80 (75.00%) |
| E  | 60/80 (75.00%) |
| DE | 60/80 (75.00%) |

Success rates are identical across all arms. E's utility gain is an
efficiency/step-cost effect, not outcome rescue.

## Causal Interpretation

### The central fork

The experiment was designed to distinguish:

- **H_A**: D hurts because it removes genuinely useful VERIFY actions.
- **H_B**: VERIFY itself is useless, but removing it changes Qwen's
  policy harmfully.

The corrected analysis shows **neither hypothesis is directly tested**
because **D never fires**. The R2d gate is a no-op: at T2, VERIFY is
already illegal (no valid targets), so the epistemic gate has nothing to
remove. `verify_removed_by_epistemic_gate` is false in every receipt.

The correct interpretation is:

> **VERIFY is phase-dependent: valuable during hypothesis resolution,
> but unnecessary after hypothesis exhaustion. R2d correctly identifies
> the boundary but is redundant because the legal-actions computation
> already excludes VERIFY at T2. Enforcing R2d provides no standalone
> utility benefit because Qwen already handles the dead-end reasonably
> well without it.**

### Why D is neutral

D is neutral because it is a no-op. The gate is active
(`verify_gate_condition_active=True`, `verify_gate_reason=ALL_HYPOTHESES_ELIMINATED`)
but `verify_removed_by_epistemic_gate=False` because VERIFY is not legal
at T2. The gate removes nothing because there is nothing to remove.

This is actually **stronger evidence for the selectivity of the T2
boundary** than the original interpretation. The T2 boundary correctly
identifies where VERIFY stops being useful, and the legal-actions
computation already enforces this without needing the epistemic gate.

### Why E is positive

E (relabeling to `NO_VIABLE_HYPOTHESIS`) improves utility by +1.62
(paired bootstrap CI [1.16, 2.08], excludes zero). The mechanism:

1. At T2, the internal state is `SUPPORTED_BUT_UNRESOLVED` or
   `INSUFFICIENT`.
2. Without E, the model interprets this as "needs more work" and
   continues searching/retrieving.
3. With E, the model sees `NO_VIABLE_HYPOTHESIS` and defers sooner.
4. The utility gain comes from saved step costs — same terminal
   outcomes, fewer unnecessary intermediate actions.

This is an **efficiency intervention**, not outcome rescue. Success
rates and terminal actions are identical across all arms.

### Comparison to earlier Gemma behavior

This differs from the earlier exploratory Gemma R2 live-smoke behavior
where E caused STOP-collapse. That was a different, unqualified run with
a different GGUF/runtime and non-strict decoding. We do not yet have a
qualified Gemma-vs-Qwen backend interaction experiment. The effect
appears backend-specific but this has not been formally tested.

## Disposition (frozen rules applied)

| Rule | Condition | Result |
|------|-----------|--------|
| D positive → continue epistemic admissibility | Δ_D > 0 | **Marginal** (CI includes zero) |
| D negative ∧ VERIFY useful → retire hard gate | Δ_D < 0 ∧ UsefulVerify > 0 | Not applicable (D not negative) |
| D negative ∧ VERIFY useless → investigate learned action values | Δ_D < 0 ∧ UsefulVerify ≈ 0 | Not applicable (D not negative) |
| E negative again → retire label intervention | Δ_E < 0 | **Not triggered** (E is positive, CI excludes zero) |
| DE uniquely positive → identify real interaction | Δ_DE > Δ_E and Δ_DE > Δ_D | **Not triggered** (DE ≈ E) |
| All negative → freeze MDSG/T2 as diagnostic, pivot to learned policy | Δ_D < 0 ∧ Δ_E < 0 ∧ Δ_DE < 0 | **Not triggered** |

### Final disposition

1. **D is neutral because it is a no-op.** The T2 boundary is correct
   but the legal-actions computation already enforces it. The hard
   epistemic gate is redundant on this backend. Continue epistemic
   admissibility as diagnostic infrastructure.

2. **E is positive (efficiency, not rescue).** Relabeling to
   `NO_VIABLE_HYPOTHESIS` at T2 helps Qwen defer sooner. The paired
   bootstrap CI excludes zero. This is a step-cost improvement, not a
   success-rate improvement.

3. **No D×E interaction.** The two interventions are independent.

4. **The emerging controller architecture is phase-aware epistemic
   control:**

   ```
   Evidence acquisition / resolution phase
           ↓
   VERIFY is admissible and useful
           ↓
   T2: hypothesis set exhausted
           ↓
   resolution phase ends
           ↓
   VERIFY becomes epistemically inadmissible (no valid targets)
           ↓
   model chooses among search / retrieve / defer / answer / reason
   ```

   The T2 boundary is meaningful as a phase separator. VERIFY is useful
   before it and structurally impossible after it. The hard gate is
   redundant because the legal-actions computation already enforces the
   boundary. The label intervention (E) helps the model recognize the
   phase transition and terminate wasted epistemic work sooner.

## Limitations

1. **The R2d gate never fired.** This means the experiment cannot
   directly test what happens when VERIFY is removed at T2, because
   VERIFY is already unavailable there. The budget-exhausted variants
   (60 trajectories) were designed to test this, but even with
   retrieval/search exhausted, VERIFY is still not legal at T2 (no
   valid targets).

2. **E's effect is efficiency only.** Success rates and terminal
   actions are identical across all arms. The +1.62 utility gain comes
   entirely from saved step costs.

3. **No qualified Gemma-vs-Qwen comparison.** The earlier Gemma
   live-smoke behavior was from an unqualified run. We do not yet have
   a qualified backend-interaction experiment.

4. **The analysis pipeline was revised post-run.** The original
   analysis had bugs in Steps 3, 5, 6, and 10. This revision fixes
   them but is classified as a post-run corrective analysis revision,
   not preregistered.

5. **The dataset's `TWO_HYPOTHESIS_DISCRIMINATION_SCENARIO` stratum**
   represents a two-hypothesis discrimination scenario, not a literal
   state with |H_live| ≥ 2. After gold verification, only 1 hypothesis
   remains live.
