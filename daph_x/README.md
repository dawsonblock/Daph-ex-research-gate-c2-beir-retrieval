# DAPH-X: Parameterized Model-Based Executive

A clean-break redesign of DAPH. The LLM is no longer the primary policy.

## Architecture

```
Observations / retrieved evidence / tool results
                    ↓
          Typed epistemic world model
                    ↓
        Canonical belief / task state
                    ↓
      Candidate action generation
                    ↓
 ┌─────────────────────────────────────┐
 │  Model-based consequence evaluator  │
 │  Learned action-value estimator     │
 │  Information-value estimator        │
 │  Resource/cost model                │
 │  Uncertainty + support estimator    │
 │  Structural invariants/certificates │
 └─────────────────────────────────────┘
                    ↓
          Executive action selector
                    ↓
        LLM proposal as one signal
                    ↓
          Action / tool execution
                    ↓
             State transition
                    ↓
       Counterfactual + outcome ledger
```

## Key Differences from V3R2

| V3R2 | DAPH-X |
|------|--------|
| LLM → DAPH occasionally overrides | Executive selects action, LLM is one signal |
| Generic VERIFY | VERIFY(e_i) — parameterized by evidence target |
| Generic REASON_MORE | COMPARE, DECOMPOSE, GENERATE_ALTERNATIVE, etc. |
| Flat feature vector | Epistemic graph (primary) + symbolic topology (derived) |
| Heuristic IG | Target-specific expected information value |
| Binary HARD/SHADOW | OBSERVE, ADVISE, CONSTRAIN, FORCE, ABSTAIN |
| Certificates = policy | Certificates = hard invariants around general planner |
| Single Q estimator | Q_MB + Q_residual (hybrid) |

## V3R2 Baseline

V3R2 is frozen in `releases/daph_v3r2_terminal_authority/`.
Every DAPH-X experiment must answer: "Is this better than the confirmed baseline?"

## Directory Structure

```
daph_x/
├── graph/          # Epistemic graph (primary state representation)
├── topology/       # Canonical symbolic topology (inherited from V3R2)
├── belief/         # Belief engine (calibrated P(H|E))
├── actions/        # Typed parameterized actions
├── world_model/    # P(o|s,a) transition layer
├── value/          # Q_MB + Q_residual, information value, cost
├── authority/      # OBSERVE/ADVISE/CONSTRAIN/FORCE/ABSTAIN
├── receipts/       # Decision receipts (hash-chained)
└── evaluation/     # Benchmarks, qualification, metrics
```
