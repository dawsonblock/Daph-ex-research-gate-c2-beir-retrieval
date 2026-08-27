# I3.28B: DEFER Boundary Causal Repair — Results

**Date:** 2026-08-26
**Branch:** `i3.27-q-error-and-authority`
**Experiment:** `scripts/run_i3_28b_boundary_repair.py`

---

## Design

Generate 40 matched state pairs at the exact aliasing boundary (`n_verified=0`):

| State | `has_competing_unverified_support` | Correct action | DEFER utility |
|-------|:---:|:---:|:---:|
| Safe defer (A) | 0 | DEFER | +69.89 |
| Unsafe defer (B) | 1 | VERIFY→ANSWER | -30.11 |

Everything else matched: same n_hypotheses (2), same n_visible_evidence (2), same budget profile (`TIGHT_NO_RETRIEVE_NO_SEARCH`), same action vocabulary. 10 domain templates rotated across 40 pairs for structural variation.

Forced all 5 legal actions (DEFER, VERIFY, ANSWER, REASON_MORE, STOP) at each state. Computed realized utility using the same `MetareasoningUtility` as the original causal data collection.

Appended 400 boundary records to the original 1056 causal records → 1456 total. Retrained with exactly the same GBT (`n_estimators=200, max_depth=4, random_state=42`).

---

## Boundary Data

| Action | Safe (has_competing=0) | Unsafe (has_competing=1) |
|--------|:---:|:---:|
| DEFER | +69.89 | -30.11 |
| VERIFY | +66.71 | +96.71 |
| ANSWER | -120.11 | -120.11 |
| REASON_MORE | +67.75 | -125.32 |
| STOP | -30.11 | -30.11 |

The contrast is exactly what the GBT needs:
- DEFER is good when has_competing=0, bad when has_competing=1
- VERIFY is good when has_competing=1, moderate when has_competing=0
- The relative action value Q(DEFER) - Q(VERIFY) flips sign across the boundary

---

## Results

### Gate 1: Separation audit — PASS

**Boundary (in-sample):**
- Safe Q(DEFER): 68.37
- Unsafe Q(DEFER): -28.44
- **Margin: 96.82** (required >5.0)

**I3.26 dev (out-of-sample):**
- Safe Q(DEFER): 68.37
- Unsafe Q(DEFER): -35.21 (range [-41.97, -28.44])
- **Margin: 96.82** (required >5.0)

The separation is not just above threshold — it is massive. The GBT learned the boundary perfectly and generalizes to the I3.26 dev benchmark states it never saw in training.

### Gate 2: False DEFER authority rate — PASS

- Safe states: 0/200 trigger DEFER authority
- Unsafe states: 0/200 trigger DEFER authority
- **False DEFER authority rate: 0.0000**

### Gate 3: ANSWER authority preservation — PASS

- V2R ranks ANSWER as best in 44/44 ANSWER-correct checkpoints (same as V1)
- Q(ANSWER) = 99.77 at ol_answer states (V1: 99.92 — negligible change)

### Gate 4: Preservation on 220 checkpoints — PASS

| Metric | V1 (boundary control) | V2R (boundary repaired) |
|--------|---:|---:|
| Mean regret | 19.79 | 18.43 |
| Near-optimal (ε=3) | 112/220 | 180/220 |
| Correct best action | 88/220 | 112/220 |

V2R improves on V1 across all preservation metrics.

### Overall: ALL GATES PASS

---

## Important Finding: DEFER Authority Coverage

DEFER authority coverage on safe defer states is **0/200**. This is not a gate failure — it is a structural property of the task.

At safe defer states, the Q values are:

| Action | Q value |
|--------|--------:|
| DEFER | 68.37 |
| REASON_MORE | 65.28 |
| VERIFY | 65.07 |
| STOP | -17.82 |
| ANSWER | -125.14 |

**Q-gap = 68.37 - 65.28 = 3.09** (below authority threshold of 5.0).

The gap is small because VERIFY and REASON_MORE are also reasonable actions in defer-correct states. You can verify first (cost ~3 utility) then defer, or reason more then defer. The model correctly ranks DEFER as best, but the margin to the next-best action is only ~3 points.

This means DEFER authority with threshold 5.0 will not fire on these states. The model is conservative: it will not force DEFER when VERIFY is nearly as good. This is safe but limits the practical coverage of DEFER hard authority.

**This is not a representation failure or a training failure.** The model has learned the correct action ranking. The issue is that in defer-correct states, multiple actions are reasonable, and the Q-gap doesn't reach the authority threshold.

### Options (not yet decided)

1. **Accept conservative behavior:** DEFER authority doesn't fire, but the model correctly advises DEFER. Advisory guidance may be sufficient for Qwen to choose DEFER.
2. **Design defer-correct states where VERIFY is not available:** Use a budget with `max_verification_calls=0`. Then VERIFY is illegal, and the gap would be to REASON_MORE or STOP.
3. **Lower the DEFER threshold:** The user explicitly said not to tune the threshold after viewing results.

---

## Artifacts

| File | Description |
|------|-------------|
| `experiments/i3_28b/boundary_causal_actions_v1.jsonl` | 400 boundary causal records |
| `experiments/i3_28b/Q_V2R_boundary_repaired.pkl` | Q_V2R trained on 1456 records |
| `experiments/i3_28b/Q_V1_boundary_control.pkl` | Q_V1 control on 1456 records |
| `experiments/i3_28b/full_results.json` | All gate results |

---

## Decision

All offline gates pass. The separation is massive (margin 96.82). The false authority rate is zero. ANSWER authority is preserved. V2R improves on V1 across all preservation metrics.

The DEFER authority coverage finding (0/200 on safe states due to Q-gap = 3.09 < 5.0) is a structural property of the task, not a model failure. The model correctly ranks DEFER as best but doesn't reach the hard authority threshold because VERIFY is also reasonable in defer-correct states.

**Per the preregistered protocol: all offline gates pass. The model is eligible for live validation.**

The live validation sequence should proceed as specified:
1. Known DEFER rescue cases
2. Fresh contradiction-heavy negative controls
3. Targeted live safety run
4. Fresh untouched confirmation (effect-driven sample size)

But the DEFER authority coverage finding should be considered when designing the live validation: in states where VERIFY is also reasonable, DEFER authority won't fire, and the model will remain advisory. The live validation should include states where DEFER is clearly better than VERIFY to test the authority path, as well as states where the model is advisory to test whether Qwen follows advisory DEFER guidance.
