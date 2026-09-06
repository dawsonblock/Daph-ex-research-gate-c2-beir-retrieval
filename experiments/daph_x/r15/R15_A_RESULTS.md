# R15-A Final Results: Predictive Escalation Qualification

**Commit:** 4fa5085 (frozen evidence boundary)
**Date:** 2026-09-05
**Protocol:** R15_A_PROTOCOL.md + R15_A_ADDENDUM_1.md (both preregistered before confirmation)
**Champion hash:** 20c1a06e0c62b67c752b1ef53d4d0567edf5d85dfe18e4cb8dfa9c8cc9b3fe72
**Verdict:** FAIL — failed to demonstrate non-inferiority; failed Bronze

## Confirmation set

- 419 held-out R12 tasks (not in R13 checkpoints)
- 419 unique task IDs (one checkpoint per task)
- K assigned deterministically: hash(task_id) mod 3 → {2, 4, 6}
- Distribution: k=2: 138, k=4: 143, k=6: 138
- Manifest SHA-256: 63420f48e2012b157f3e3d3bee98410804119a8f8dba26876eb3edbc9c580e6f

This is a within-corpus confirmation test, not an external-distribution confirmation.

## Confirmation frontier

| Policy | Accuracy | Mean lat (s) | N correct |
|---|---:|---:|---:|
| Always STOP | 0.5585 | 0.000 | 234/419 |
| Always COT | 0.8663 | 8.348 | 363/419 |
| STOP→COT oracle | 0.8950 | 4.877 | 375/419 |
| Frozen champion router | 0.8568 | 7.448 | 359/419 |

## Oracle opportunity generalized

The STOP→COT oracle on the confirmation set saves:

    1 - 4.877 / 8.348 = 41.6%

while improving accuracy by +2.87 pp (89.50% - 86.63%).

The routing opportunity is real on held-out tasks. What failed to generalize was the learned decision function, not the existence of routing headroom.

## Non-inferiority gate: FAIL

    A_router = 0.8568
    A_COT    = 0.8663
    Gate:    A_router >= A_COT - 0.005 = 0.8613
    Result:  0.8568 < 0.8613 → FAIL

Task-clustered bootstrap 95% CI for A_router - A_COT: [-0.0239, +0.0024].

**Important distinction:** R15-A failed to *demonstrate* non-inferiority. It did *not* prove the router is inferior to COT at conventional significance levels. The CI includes positive values. But the preregistered gate required establishing acceptable performance, and the router did not clear that bar.

## Latency recovery: FAIL

    Router saving:         10.8%
    Oracle saving:         41.6%
    Oracle-headroom recovery R_L: 25.9%

Bronze requires R_L >= 50%. The router recovered only 25.9% of the available latency saving.

## Confusion matrix

|  | COT correct | COT wrong |
|---|---:|---:|
| STOP-kept (70) | 64 correct | 6 wrong |
| COT-escalated (349) | 295 correct | 54 wrong |

Total escalated: 349/419 = 83.3%.

## Selective-prediction analysis

Among the 70 cases where the router chose STOP:

    64/70 = 91.4% were correct.

That is not zero signal. The router can identify *some* safe STOP states at moderate precision.

The problem is **coverage**, not complete inability to recognize safe STOP:

- Router chose STOP on: 70/419 = 16.7%
- STOP was actually correct on: 234/419 = 55.9%
- There are ~170 additional STOP-correct states the router failed to identify

The router learned an excessively conservative policy: escalate almost everything, stop only on the most obvious cases. This was visible on development (77/90 = 85.6% escalated) and generalized closely to confirmation (349/419 = 83.3% escalated).

## Distribution shift

|  | Dev | Confirmation |
|---|---:|---:|
| STOP accuracy | 48.9% | 55.9% |
| COT accuracy | 90.0% | 86.6% |
| Oracle accuracy | 91.1% | 89.5% |
| Router escalation rate | 85.6% | 83.3% |

COT accuracy dropped 3.4pp on confirmation. An absolute 89.5% accuracy gate would have been impossible — no policy except the oracle achieves it. The confirmation-relative gate (A_COT - 0.005) was the correct preregistered choice.

The router's escalation rate generalized closely (85.6% → 83.3%). The policy did not collapse under distribution shift. It simply learned an overly conservative policy on development that was already visible before confirmation.

## Correct conclusion

R15-A established the narrower claim:

> **The frozen linear action-value router trained from the existing observable RuntimeState features did not exploit those features well enough to justify deployment over always-COT.**

R15-A did **not** establish:

> "The observable RuntimeState features are not predictive enough."

The distinction matters because:
1. Only one frozen Ridge champion was tested, not an information-theoretic bound.
2. The router achieved 91.4% selective STOP precision on its 70 STOP decisions — there is signal.
3. The oracle opportunity generalized (41.6% saving on confirmation), so routing value exists.
4. The problem is coverage (identifying more safe STOP states) not absence of signal.

## What R15-A ruled out

The cheap solution: a linear router on candidate-vote statistics (agreement, entropy, margin, trajectory) cannot recover enough of the routing oracle to justify deployment.

## What R15-A did not rule out

- Nonlinear routers on the same features (requires a separate untouched confirmation set per protocol).
- Additional information sources beyond candidate-vote statistics.
- Cheap metacognitive probes that distinguish "agree because correct" from "agree because same misconception."
- Using the base model's actual reasoning product as state, not merely vote summaries.

## Cost limitation

Wall_ms only. True token/GPU compute is not instrumented. Same limitation as R14-C.

## Evidence boundary

Commit 4fa5085 is the frozen R15-A evidence boundary. No further tuning against these 419 confirmation labels. No nonlinear retry on the same confirmation set.
