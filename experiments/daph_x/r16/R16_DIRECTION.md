# R16 Direction: Cheap Verification Features for Executive Information

**Status:** DRAFT — not yet preregistered. For discussion before freezing a protocol.

## Motivation

R15-A established that a linear router on candidate-vote statistics (agreement, entropy, margin, trajectory) cannot recover enough of the STOP→COT routing oracle to justify deployment. The router achieved 91.4% selective STOP precision but only 16.7% coverage — it could not identify enough safe STOP states.

The key diagnostic: vote entropy cannot distinguish

```
"five samples agree because the answer is obvious and correct"
```

from

```
"five samples agree because the same systematic misconception
was repeated five times"
```

That is why 3/46 STOP-wrong states on the dev set had p_top1=1.0 (unanimous but wrong). Vote statistics capture agreement, not logical soundness.

R16 changes the question from "try a fancier classifier" to:

> **What information is missing from RuntimeState that would let us identify safe STOP states?**

## Core hypothesis

Cheap metacognitive probes can provide signal about reasoning soundness that vote statistics cannot, at a cost low enough that the net economic tradeoff is favorable.

## Candidate new signals

### Tier 1: Cheap base-model probes (target < 1s each)

1. **Answer confidence probe**: Ask the model to rate its confidence in its answer. Cheap (one short generation) but may be poorly calibrated.

2. **Self-check probe**: "Is your answer correct? Explain briefly." One short generation. May catch obvious errors.

3. **Contradiction probe**: Present the answer and ask "Could this be wrong? Give one reason." If the model produces a substantive contradiction, that's a signal of uncertainty.

4. **Alternative-answer probe**: "What other answer might be correct?" If the model produces a different answer, that's a strong signal that STOP is unsafe.

5. **Problem difficulty estimator**: Ask "How difficult is this problem?" before solving. May correlate with STOP sufficiency.

### Tier 2: Lightweight verifier (target < 2s)

6. **Solution-consistency score**: Given the answer and a brief justification, ask the model to verify step-by-step. Cheaper than full COT_REFLECT but more informative than vote statistics.

7. **Token/logprob statistics**: If the base model exposes logprobs on the answer tokens, use those directly. This is the strongest signal in principle but requires API/gateway support.

### Tier 3: Reasoning product features

8. **Reasoning trace features**: Use the actual content of the candidate reasoning traces, not just vote counts. E.g., does the trace contain hedging language, self-correction, or contradiction?

9. **Cross-trace consistency**: Do the reasoning traces cite the same facts/formulas? Or do they reach the same answer via different routes (stronger signal) vs. the same route (weaker, could be shared misconception)?

## Economic framework

The economic test for R16 is:

```
cheap probe cost
      +
router (with probe features)
      ↓
can we safely avoid enough 8.35s COT calls
to pay for the probe?
```

For example, a 300ms verifier would be worthwhile if it lets the router confidently avoid another 15-20% of COT calls. The break-even point depends on:

- Probe cost C_probe (seconds)
- Additional COT calls avoided due to probe signal: Δn
- COT cost per call: L_COT ≈ 8.35s

Net saving = Δn × L_COT - N × C_probe

If C_probe = 0.3s and N = 419:
- Probe total cost: 125.7s
- Need to avoid: 125.7 / 8.35 ≈ 15 additional COT calls
- That's 15/419 ≈ 3.6% more STOP-kept cases

That is a very achievable bar.

## Proposed R16 structure

### R16-A: Probe qualification

1. Define a small set of cheap probes (start with 2-3 from Tier 1)
2. Run probes on the R13 development checkpoints (90 cp, 81 tasks)
3. Measure: probe cost, probe signal (correlation with STOP correctness)
4. Do NOT yet train a router — just qualify the probes

### R16-B: Router with probe features

1. If R16-A shows useful signal, add probe features to the frozen feature set
2. Train a new router on development with the expanded feature set
3. Freeze one champion via task-grouped CV
4. Evaluate on a NEW confirmation set (not the R15-A 419 tasks — those are now tainted by the R15-A result)

### R16-C: Economic analysis

1. Compute net latency saving including probe cost
2. Compare against R15-A router (no probes) and always-COT
3. Apply the same non-inferiority gate and oracle-headroom recovery tiers

## What R16 does NOT change

- The DAPH-X architecture: epistemic state → action value → authority
- The operator set: STOP, RE2, COT_REFLECT
- The evaluation framework: confirmation-relative non-inferiority + oracle-headroom recovery
- The commitment to frozen protocols and evidence boundaries

## What R16 DOES change

- The information available to the executive
- The cost model (probes have nonzero cost)
- The feature set (probe outputs added to candidate-vote statistics)

## Open questions for discussion before preregistration

1. Which probes to qualify first? (Recommend: self-check + alternative-answer, cheapest and most likely to catch unanimous-wrong)
2. Should probes run on all candidates or just the selected answer?
3. How to handle probes that themselves produce wrong answers?
4. Should the R16-B confirmation set be a fresh sample from R12, or a genuinely external benchmark?
5. Should we also try logprob features if the gateway supports them?

## Relationship to DAPH-X thesis

R14 showed: strong external reasoning operator exists, and selective routing could save ~40% latency.

R15-A showed: existing state + linear router cannot capture that saving.

R16 asks: can cheap verification information close the gap?

If R16 succeeds, DAPH-X has demonstrated real engineering leverage: an executive that uses cheap probes to decide when expensive reasoning is unnecessary.

If R16 fails, the negative result is still valuable: it would suggest that distinguishing sound from unsound reasoning requires reasoning-level compute, not just cheap probes. In that case, the executive's value lies in selecting among reasoning strategies (R14's Pareto frontier) rather than avoiding them.
