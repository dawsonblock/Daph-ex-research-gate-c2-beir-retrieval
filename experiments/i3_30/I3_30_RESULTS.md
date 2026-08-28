# I3.30 Results: Post-Verification Representation Repair

## Status: Offline gates PASS. Live study pending.

## Hypothesis

> `H_{I3.30}: Visible verified-evidence topology is sufficient to distinguish safe ANSWER from safe DEFER after verification.`

## Summary

I3.30 repaired the post-verification epistemic representation that caused 11 false-authority events in I3.29. The repair followed the requested sequence: aliasing audit first, then hypothesis-level topology, then new schema, then separability audit, then training support inspection, then boundary collection, then retraining, then offline gates, then positive structural certificates.

## Phase 1: Aliasing Audit

Examined all 11 I3.29 false-authority states against correctly handled D2/D3/D4 post-verification controls.

**Root cause identified:** V2R structural features (`n_hyp_unverified_support`, `n_hyp_unverified_contradiction`, `has_competing_unverified_support`) collapse to trivial values after verification, providing no signal to distinguish post-verification ANSWER-correct from DEFER-correct states.

**Minimal separating feature:** `n_hyp_with_verified_contradiction` separates the 11 false-authority states:
- DEFER-correct (D2): `n_hyp_with_verified_contradiction = 1` (verification eliminated a hypothesis)
- ANSWER-correct (D3): `n_hyp_with_verified_contradiction = 0` (no elimination)

## Phase 2: Separability Audit

Full audit across all 751 reconstructed I3.29 trajectory states (209 terminal decision points).

| Representation | Collision groups |
|---|---|
| V2R (3 features) | 2 |
| V3 structural (7 features) | 4 |
| V3 + `verified_hyp_action` (9 features) | 1 |

The remaining collision is pre-verification (23 D1 DEFER-correct vs 4 D3 ANSWER-correct, both with `verified_hyp_action=None`). This is a premature-terminal issue, not a Q calibration issue.

**Key insight:** `verified_hyp_action` (the `answer_action` of the uniquely verified-supported hypothesis) is observable and perfectly separates post-verification ANSWER-correct from DEFER-correct states.

## Phase 3: Training Support Coverage

**Third occurrence of the training-support lesson:**

| Source | Records | Post-verify | V3 features |
|---|---|---|---|
| i3_5 | 1056 | 906 | Reconstructed from checkpoints |
| i3_28b | 400 | 0 | Not available |
| i3_28c | 475 | 0 | Not available |
| **Total** | **1931** | **906** | — |

6 safety-critical cells had zero support for V3 features, including all 4 `complete_supported, unique_supported` cells.

## Phase 3b: I3.30B Boundary Collection

Collected 984 post-verification causal records across 4 regimes:

| Regime | Records | Topology | VHA | Expected |
|---|---|---|---|---|
| P1a | 72 | unique_supported | ANSWER | ANSWER |
| P1b | 140 | unique_supported_with_elim | ANSWER | ANSWER |
| P2a | 72 | unique_supported | DEFER | DEFER |
| P2b | 140 | unique_supported_with_elim | DEFER | DEFER |
| P3 | 280 | competing_support | None | ANSWER |
| P2_elim | 280 | only_eliminated | None | DEFER |

All legal actions forced at each state. Resource-exhausted budget variants included for P1/P2 to create Q gaps sufficient for authority activation.

Data SHA-256: `a4e04079c536dcf08630ef4692350d410acff60d3b86f18da13b9e9050122ba2`

## Phase 4: Q_V3R Training

- **Training data:** 2915 records (1931 original + 984 I3.30B)
- **Features:** 56 (V1 + V2R + V3 + interactions)
- **Learner:** `GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42)` — frozen, unchanged
- **Train R²:** 0.9850
- **Model SHA-256:** `64044c3c0f32d46a39dff024f3601668d9f8bc0f08c973998c32ffee3fadd9f1`

## Phase 5: Offline Gates

All 6 gates PASS:

| Gate | Result | Detail |
|---|---|---|
| 1. FAR_ANSWER = 0 | PASS | 0 false / 113 events |
| 2. FAR_DEFER = 0 | PASS | 0 false / 194 events |
| 3. TerminalAuthorityPrecision = 1.0 | PASS | 307/307 correct |
| 4. ANSWER coverage > 0 | PASS | 19/20 P1a states |
| 5. DEFER coverage > 0 | PASS | 10/20 P2a states |
| 6. ANSWER preservation | PASS | 24/24 i3_5 states |

## Phase 6: Positive Structural Certificates

Replaced the V2 absence-of-danger pattern:
```
HighQConfidence(a) AND NOT KnownUnsafe(s)
```
with:
```
HighQConfidence(a) AND PositiveStructuralCertificate(a)
```

**ANSWER certificate:**
- `has_unique_verified_supported_hypothesis AND verified_hyp_action_is_answer`, OR
- `all_evidence_verified AND n_hyp_with_verified_contradiction == 0` (legacy D4)

**DEFER certificate:**
- `has_unique_verified_supported_hypothesis AND verified_hyp_action_is_defer`, OR
- `n_eliminated_hypotheses > 0 AND n_viable_hypotheses <= 1` (elimination), OR
- `verify_budget_exhausted AND n_hyp_with_verified_support == 0 AND n_hyp_with_verified_contradiction == 0` (legacy D1)

22 unit tests pass, including regression tests for both I3.29 false-authority patterns:
- D3 false DEFER: blocked (verified hypothesis says ANSWER, not DEFER)
- D2 false ANSWER: blocked (verified hypothesis says DEFER, not ANSWER)

## Artifacts

| Artifact | Path |
|---|---|
| V3 schema | `experiments/i3_30/Q_STATE_SCHEMA_V3_POSTVERIFY.json` |
| Aliasing audit | `experiments/i3_30/aliasing_audit.json` |
| Separability audit | `experiments/i3_30/separability_audit.json` |
| Coverage matrix | `experiments/i3_30/v3_coverage_matrix.json` |
| Q_V3R model | `experiments/i3_30/Q_V3R_postverify.pkl` |
| Feature schema | `experiments/i3_30/v3_feature_schema.json` |
| Offline gates | `experiments/i3_30/offline_gates.json` |
| I3.30B boundary data | `experiments/i3_30b/post_verify_causal_actions_v1.jsonl` |
| V3 authority policy | `daph/authority/policy_v3.py` |
| V3 authority tests | `tests/unit/test_authority_v3.py` |

## What was NOT changed

- V2 policy (`daph/authority/policy.py`) — preserved
- V2R model — preserved
- Frozen authority threshold (5.0) — unchanged
- GBT hyperparameters — unchanged
- Pinned Qwen model — unchanged
- Search, PAV, runtime — unchanged

## Interpretation

I3.30 provides strong development evidence that the V3 representation with positive structural certificates can distinguish safe ANSWER from safe DEFER after verification:

1. The aliasing audit identified the root cause (post-verification feature collapse).
2. The separability audit confirmed that `verified_hyp_action` resolves the collision.
3. The coverage matrix identified the training-support gap (third occurrence of this lesson).
4. I3.30B boundary data filled the gap with matched causal states.
5. Q_V3R achieves zero false authority and perfect terminal authority precision on offline gates.
6. The positive structural certificate architecture blocks both I3.29 false-authority patterns.

However, V3 remains an experimental candidate. V1 remains the confirmed champion. The offline gates are necessary but not sufficient for promotion. A targeted live study with fresh instances is required before V3 can be considered for confirmation.

## Next step

Targeted live study with 5 strata (D1-D5), comparing frozen V1 against candidate V3, with the live promotion rule:
```
FalseAuthorityRate_ANSWER = 0
AND FalseAuthorityRate_DEFER = 0
AND rescues > breaks
AND breaks = 0 for hard-authority interventions
AND success_V3 >= success_V1
AND ΔU > 0
```
