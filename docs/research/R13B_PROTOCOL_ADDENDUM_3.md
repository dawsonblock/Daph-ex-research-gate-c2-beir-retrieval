# R13-B Protocol Addendum 3

## Blind correction to the eligibility λ rule

This addendum is committed **after** finalizing the R13-A v2.3 analyzer and **before** inspecting the completed seeds 123 and 2024. It removes the only remaining ambiguity in Addendum 2 regarding which λ controls the R13-A architectural eligibility gate.

## 1. Frozen λ grid

The complete frozen λ grid for R13-A architecture and R13-B cost-aware utility is:

```
{0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5}
```

## 2. R13-A architectural eligibility gate

R13-B proceeds **only if** both conditions hold:

1. The replicated three-seed data contain all required `(checkpoint, operator, seed)` receipts as defined by `scripts/analyze_r13a_v2_3.py`:
   - `REQUIRED_REPLICATES = {42, 123, 2024}`
   - `REQUIRED_OPERATORS = {STOP, SAMPLE_STANDARD, SAMPLE_DIVERSE, CRITIQUE_RETRY, VERIFY_TARGETED}`
   - Only checkpoints with exactly `5 operators × 3 seeds = 15` successful receipts are used.

2. For **every** λ in the frozen λ grid:

```
UCB_95(J_het(λ) - J_bin_best(λ)) < 0.005
```

where:

- `J_het(λ)` is the heterogeneous oracle using the replicated state-action averages `Q̄(s,a)` and `c̄(s,a)`.
- `J_bin_best(λ)` is the best fixed binary STOP/CONTINUE policy using the single globally best continuation `v` at that λ.
- `UCB_95` is the one-sided 95% upper confidence bound from a task-clustered bootstrap with explicit multiplicity correction (10,000 replicates, seed `99`).

If the UCB at **any** λ in the frozen grid is ≥ 0.005, the data do not rule out meaningful heterogeneous routing value at that cost regime, and R13-B is **not executed** under its binary-only architecture.

## 3. Why every λ matters

DAPH-X is a cost-aware controller. The relevant architectural question is not whether action heterogeneity exists when compute is free (λ = 0), but whether it exists across the cost regimes the controller may actually use. If heterogeneity reappears at λ = 0.1, it matters for deployment. Therefore the eligibility gate must hold for the entire frozen λ grid, not just at λ = 0.

## 4. Continuation selection remains at λ = 0

The fixed continuation `v_0` for R13-B is still selected using λ = 0:

```
v_0 = argmax_{v ≠ STOP} mean_s Q̄(s, v)
```

tie-broken by lower `c̄(s, v)`, then lower mean model calls, then lexicographically smallest operator id.

The operating `λ` for R13-B is tuned only after `v_0` is frozen.

## 5. Bootstrap and analyzer standard

The only valid analyzer for the R13-A v2.3 architectural gate is `scripts/analyze_r13a_v2_3.py`.

It implements:

- Replicate averaging: `Q̄(s,a)` and `c̄(s,a)` from seeds 42, 123, and 2024.
- Strict completeness: 15 receipts per checkpoint.
- Fail-closed cost decoding by operator version.
- Multiplicity-correct task-clustered bootstrap.
- Full frozen λ grid.

## 6. Audit trail

- `R13B_PROTOCOL.md`: original preregistration, commit `ed2c42a`, unchanged.
- `R13B_PROTOCOL_ADDENDUM_1.md`: first addendum, commit `c0220c6`, unchanged.
- `R13B_PROTOCOL_ADDENDUM_2.md`: cost/ceiling/split correction, preserved; superseded on the eligibility λ rule by this document.
- `R13B_PROTOCOL_ADDENDUM_3.md`: this document, exact λ-grid eligibility gate.
- `scripts/analyze_r13a_v2_3.py`: frozen analyzer standard.

No operator, threshold, feature, or analysis rule is modified while seeds 123 and 2024 complete.
