# R13 Protocol — Cognitive Action Value

## Central Research Question

Can DAPH-X learn the state-dependent causal value of heterogeneous cognitive actions and use them more efficiently than fixed strategies or simple uncertainty routing?

## Background

R12 established that vanilla resampling (GENERATE+2) is a low-value action:

- P(Rescue | GENERATE) ≈ 2.9%
- P(Break | GENERATE) ≈ 0.1%
- P(Waste | GENERATE) ≈ 97.0%

The R12 ΔQ policy matched MaxCal@8 accuracy (68.4% vs 67.6%) with 33% less compute, but did not beat simple uncertainty-based stopping. The dominant limitation is signal sparsity: only ~3% of checkpoint states benefit from additional vanilla sampling.

R13 attacks this limitation by changing the question from "should I spend more compute?" to "what kind of compute has the highest expected value?"

## Hypotheses

### H1 — Operator heterogeneity

There exist cognitive operators a_i, a_j such that Q(s, a_i) - Q(s, a_j) varies meaningfully across epistemic states s.

This is the necessary condition for DAPH-X routing to be worthwhile. If all operators have the same state-dependent value, routing is unnecessary.

### H2 — Action-value predictability

Q(s, a) can be predicted from information available before executing a.

If the value of each action cannot be predicted from the observable state, no learned router can improve on fixed strategies.

### H3 — Routing policy value

π_DAPH Pareto-dominates the best fixed operator and simple adaptive routers under matched compute.

This is the final policy hypothesis. Do not assume it. Test H1 and H2 first.

## Operator Set (R13-A)

Start with the smallest set that can test H1:

```
A_0 = { STOP, SAMPLE_STANDARD, SAMPLE_DIVERSE, CRITIQUE_RETRY, VERIFY_TARGETED }
```

### STOP

Return the current MaxCal answer without additional reasoning. Zero cost.

### SAMPLE_STANDARD

Generate additional candidates using the frozen R12 sampling procedure (same temperature schedule, same prompt). This is the R12 GENERATE(+2) action.

### SAMPLE_DIVERSE

Generate additional candidates using deliberately different reasoning conditions. Diversity comes from controlled strategy prompts (e.g., "work backwards", "use a formal derivation", "try a different approach"), not random prompt mutations.

### CRITIQUE_RETRY

Inspect the current leading answer and reasoning trace, identify a suspected error, then generate a correction attempt. The critique must produce structured diagnostic information (suspected error type, location, correction recommendation).

### VERIFY_TARGETED

Construct targeted checks against the current leading hypothesis and competing answers. Produce structured evidence (check passed/failed, evidence for/against). Update the state based on verification results. Not just another confidence score.

## Compute Accounting

Candidate count K is no longer sufficient. Record for every operator execution:

- T = generated tokens
- L = latency (milliseconds)
- M = model calls
- G = GPU seconds (where measurable)

Define normalized compute score:

```
C(a) = w_T * T_normalized + w_L * L_normalized + w_M * M_normalized + w_G * G_normalized
```

Also preserve raw metrics separately. Report tokens, calls, latency, and normalized cost independently.

## Experimental Design

### R13-A: Operator Tournament (Stages 5-8)

1. Sample 200-300 frozen checkpoints from R12 (stratified by K, correctness, confidence, stability).
2. For every checkpoint s_i, execute every admissible operator a ∈ A_0.
3. Record U_terminal(s_i, a) and C(s_i, a).
4. Compute P(Rescue|a), P(Break|a), P(Waste|a), E[ΔU|a], E[C|a] for each operator.
5. Compute oracle action distribution P(a* = a) and oracle action entropy H(A*).
6. Compute pairwise win matrices P(Y_a > Y_b).

**Early exit gate (R13-Q4)**: If all operators have P(Rescue|a) ≈ 2-3%, stop. Better routing cannot fix fundamentally weak actions.

### R13-B: Oracle Routing Headroom (Stage 9)

Compare:
- Best fixed operator (always use the same operator)
- Per-state oracle operator (select the best operator for each checkpoint)

```
Δ_routing_headroom = J_oracle - J_best_fixed
```

**Early exit gate (R13-Q5)**: If oracle routing gains < 0.5% accuracy or < 5% compute savings, do not build a learned router.

### R13-C: Learned Router (Stages 11-16, only if R13-A/B pass)

1. Build compact R13 state features (confidence, temporal convergence, epistemic evidence, context).
2. Train three routing formulations:
   - Model A: Direct scalar Q(s, a)
   - Model B: Advantage ΔQ(s, a) = Q(s, a) - Q(s, STOP)
   - Model C: Pairwise preference P(a_i ≻ a_j | s)
3. Compare hierarchical (STOP/CONTINUE → operator) vs monolithic routing.
4. Add constrained-budget optimization: max E[U] s.t. E[C] ≤ B.
5. Add uncertainty estimation: LCB(s, a) = μ_Q - z * σ_Q.
6. Preserve authority as a distinct layer.

### R13-D: Confirmation (Stages 21-22)

1. Freeze all operator definitions, prompts, hyperparameters, thresholds, λ, features, model architecture.
2. Run untouched confirmation on held-out task groups.
3. Run mechanism-OOD evaluation (train and test mechanisms are disjoint).

## Qualification Gates

| Gate | Requirement |
|------|-------------|
| R13-Q1 | R12 release fully reproducible (manifest verification passes) |
| R13-Q2 | All operators start from identical frozen states |
| R13-Q3 | Compute accounting valid (tokens, calls, latency recorded) |
| R13-Q4 | At least one nontrivial operator materially beats vanilla resampling on P(Rescue|a) |
| R13-Q5 | Oracle routing materially beats best fixed operator |
| R13-Q6 | Oracle optimal-action distribution is meaningfully heterogeneous (H(A*) > 0.5) |
| R13-Q7 | Learned router beats random matched-cost routing |
| R13-Q8 | Learned router beats simple uncertainty routing |
| R13-Q9 | Learned router beats or Pareto-dominates best fixed operator |
| R13-Q10 | Calibration/risk gates pass (coverage tests) |
| R13-Q11 | Paired confidence intervals support the claimed improvement |
| R13-Q12 | Mechanism-OOD does not collapse |
| R13-Q13 | Provenance/manifest verification passes |
| R13-Q14 | No post-confirmation tuning |

**Critical early exits**: If Q4 or Q5 fails, stop before training a complicated executive.

## Key Principle

> Do not build the sophisticated DAPH-X router until you prove there is something worth routing.

R13-A and R13-B come first. The decisive intermediate result is:

```
J_oracle_heterogeneous > J_best_fixed_operator
```

by enough margin to matter. If true, DAPH-X has a real meta-control problem to solve. If false, the correct answer is to use the best fixed cognitive operator and avoid unnecessary executive complexity.

## R12 Frozen Results (Reference)

| System | Accuracy | Avg K | J(0.1) |
|--------|---------|-------|--------|
| R12 ΔQ (t=0.01) | 68.4% ± 4.1% | 5.4 | 0.650 |
| MaxCal@8 | 67.6% ± 4.5% | 8.0 | 0.616 |
| uncertainty_p50 | 68.2% ± 3.5% | 5.0 | 0.652 |
| Oracle lookahead6 | 68.4% ± 4.1% | 2.3 | 0.681 |

R12 release: `releases/daph_x_r12/manifest.json`
Git commit: `d67869b`
