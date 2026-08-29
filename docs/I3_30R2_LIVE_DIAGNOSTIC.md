# I3.30R2 LIVE DEVELOPMENT — DIAGNOSTIC REPORT

**Status: REJECTED / DIAGNOSTIC**
**Date: 2026-08-29**
**Run: Colab T4, 370 trajectories, 24.2 min**
**Commit: 5d60bde**

## Provenance

This is a **development live run**, not a preregistered confirmation run.

- **Runner**: Patched on Colab to use V3R2-A artifacts (fixes committed to git as `6ae399e`, `44528fc`, `5d60bde`)
- **Model**: Qwen2.5-7B-Instruct Q4_K_M from DhruvalLabs mirror
  - Colab SHA: `35f9f55b0c7cdd52...`
  - Local SHA: `65b8fcd92af6b4fe...`
  - **NOT byte-identical**. 5-packet behavioral smoke test passed. This is insufficient for model equivalence.
- **Three bugs were fixed during this run** (not part of the original freeze):
  1. Runner used buggy `compute_v3_features` instead of `compute_v3_features_canonical`
  2. Executor's `_check_defer_success` didn't check `answer_action` of supported hypothesis
  3. DEFER certificate didn't check continuation admissibility

## Gate Results: 12/12 PASS

All 12 pre-registered gates pass numerically. **This does not validate the intended mechanism.**

## Authority Event Accounting

### V1 authority (confirmed champion)
- Mode: `A2A_hard_select` (84 events)
- **All 84 force ANSWER** — V1 is ANSWER-only hard authority
- V1 forces ANSWER on:
  - D4: 33 (correct — D4 expected terminal is ANSWER)
  - D5: 35 (correct — D5 expected terminal is ANSWER, not CONTINUE)
  - D2: 8 (**WRONG** — D2 expected terminal is DEFER, V1 forces ANSWER → all 8 fail)
  - D3: 8 (mixed — some correct, some wrong)

### V3 authority (V3R2-A candidate)
- Modes: `A2AD_hard_ANSWER` (78), `A2AD_hard_DEFER` (3)
- V3 forces ANSWER on:
  - D4: 31 (correct)
  - D5: 31 (correct)
  - D3: 16 (rescues — V3 forces ANSWER where V1 didn't)
- V3 forces DEFER on:
  - D2: 3 (rescues — V3 forces DEFER where V1 forced ANSWER)

### Critical accounting finding

**D5's expected_terminal is ANSWER, not CONTINUE.** The study documentation says "D5: CONTINUE — ambiguous post-verification state" but the actual tasks have `expected_terminal=ANSWER`. This means:
- D5 is NOT a "CONTINUE-correct" stratum
- V1 forcing ANSWER on D5 is correct behavior (35/35 success)
- V3 not forcing ANSWER on 4 D5 tasks is a regression

## First-Divergence Analysis: 6 Breaks

### Break classification

| Task | Stratum | Bucket | V1 | V3 |
|------|---------|--------|-----|-----|
| d1_0004 | D1 | BOTH_ADVISORY | DEFER→SUCCESS | REASON_MORE→FAIL |
| d1_0010 | D1 | BOTH_ADVISORY | DEFER→SUCCESS | REASON_MORE→FAIL |
| d2_0003 | D2 | BOTH_ADVISORY | DEFER→SUCCESS | REASON_MORE→FAIL |
| d3_0022 | D3 | BOTH_ADVISORY | VERIFY,VERIFY,ANSWER→SUCCESS | VERIFY,RETRIEVE,VERIFY→STEP_LIMIT |
| d3_0038 | D3 | BOTH_ADVISORY | VERIFY,VERIFY,ANSWER→SUCCESS | VERIFY,RETRIEVE,VERIFY→STEP_LIMIT |
| d5_0026 | D5 | V1_HARD_V3_ADVISORY | VERIFY,ANSWER(forced)→SUCCESS | VERIFY,REASON_MORE→FAIL |

### Break mechanism summary

- **5 of 6 breaks are BOTH_ADVISORY**: V3R2-A's Q model produces different Q values → different epsilon/refined set → LLM makes different (worse) choice. No hard authority involved.
- **1 of 6 breaks is V1_HARD_V3_ADVISORY**: V1 forces ANSWER (correct), V3 doesn't force (Q gap too small or certificate fails), LLM chooses REASON_MORE → fails.

**The certificate authority mechanism is NOT causing breaks.** The breaks are caused by Q-value differences altering advisory guidance.

## Rescue Mechanism: 15 Rescues

### Rescue classification

| Bucket | Count | Mechanism |
|--------|-------|-----------|
| V3_HARD_RESCUE | 10 | V3 certificate forces ANSWER, V1 didn't force (or model chose wrong) |
| BOTH_HARD_DIFFERENT_ACTION | 3 | V1 forced ANSWER (wrong), V3 forced DEFER (correct) — D2 |
| V1_HARD_V3_ADVISORY | 2 | V1 forced ANSWER (wrong), V3 advisory → model chose DEFER (correct) — D2 |

### Rescue mechanism summary

- **10 of 15 rescues are V3_HARD_RESCUE**: V3's ANSWER certificate fires on D3 tasks where V1's ANSWER-only authority didn't fire. This IS the intended mechanism working.
- **5 of 15 rescues are D2 DEFER rescues**: V3 either forces DEFER (3) or advisory guidance leads model to DEFER (2), where V1 forced ANSWER (wrong).

**The certificate authority mechanism IS responsible for 10 of 15 rescues.** This is genuine evidence that the ANSWER certificate works on D3.

## D1 Regression: V1 8/35 → V3 6/35

D1 has no verify budget → no verified evidence → no certificates can fire. D1 is purely advisory.

- V1 refined set at step 0: DEFER (31/35 tasks)
- V3 refined set at step 0: DEFER (14/35), REASON_MORE (21/35)

V3R2-A's Q model ranks REASON_MORE higher than DEFER on many D1 tasks. This changes the epsilon set, and the LLM chooses REASON_MORE instead of DEFER. Since D1 has no useful continuations, REASON_MORE wastes steps and fails.

**Root cause**: V3R2-A was trained on data where DEFER was correct only after verification. On D1 (no verification possible), the model doesn't confidently recommend DEFER.

## Causal Picture

The intended mechanism:
```
canonical topology → positive terminal certificate → correct hard ANSWER/DEFER → rescues
```

The actual picture:
```
V3R2-A Q model
      ↓
  different Q values
      ↓
  ┌──────────────────────────────────┐
  │                                  │
  ▼                                  ▼
different epsilon/refined set    certificate fires
      ↓                                  ↓
LLM makes different choices      hard ANSWER/DEFER force
      ↓                                  ↓
  ┌─────┐                         ┌─────┐
  │     │                         │     │
  ▼     ▼                         ▼     ▼
D1:    D3/D5:                  D3: 10 rescues    D2: 3 rescues
worse  mixed                   (ANSWER cert)     (DEFER cert)
       results
```

## Three Questions Answered

### Q1: Why did D1 improve (first run) / regress (final run) if DEFER authority never fired?

D1 has no verify budget → no certificates fire. D1 changes are entirely from Q-value differences altering advisory guidance. In the first run (buggy features), V3's Q values happened to guide the LLM toward better D1 choices. In the final run (canonical features), V3's Q values guide the LLM toward REASON_MORE on D1, which is worse. **The D1 effect is purely a Q-model side effect, not authority mechanism.**

### Q2: Why did D5 degrade if false terminal authority was zero?

D5's expected terminal is ANSWER (not CONTINUE as documented). V1 forces ANSWER on all 35 D5 tasks (hard authority) → 35/35 success. V3 only forces ANSWER on 31 D5 tasks. On the 1 failing task (d5_0026), V3's Q gap was too small for hard authority, so V3 stayed advisory, and the LLM chose REASON_MORE → failed. **The D5 regression is caused by V3's certificate being more restrictive than V1's ANSWER-only authority.**

### Q3: What exactly are the 81 hard-authority events and 0 hard-authority breaks?

- 78 ANSWER forces: 31 on D4 (correct), 31 on D5 (correct), 16 on D3 (10 rescues, 6 neutral)
- 3 DEFER forces: all on D2 (all rescues)
- 0 breaks: no hard authority event caused a TERMINAL_WRONG outcome

**The 81 events are genuine hard authority interventions.** The certificate mechanism works when it fires. The issue is that it fires less often than V1's simpler ANSWER-only authority, causing breaks where V1 would have forced correctly.

## Scientific Conclusion

1. **The ANSWER certificate mechanism works** — 10 D3 rescues are directly caused by V3's ANSWER certificate firing where V1's didn't.
2. **The DEFER certificate mechanism works** — 3 D2 rescues are directly caused by V3's DEFER certificate firing where V1 forced wrong ANSWER.
3. **The mechanism is too restrictive** — V3's certificate requires more epistemic evidence than V1's simple Q-gap threshold, causing it to miss 1 D5 case and 2 D1 cases where V1's blunt force was correct.
4. **The Q model has a D1 weakness** — V3R2-A doesn't confidently recommend DEFER on resource-exhausted states, causing advisory regressions.
5. **D5 is mislabeled** — The documentation says CONTINUE but the tasks expect ANSWER. This is a benchmark design issue.

## Recommendations

1. **Do NOT retrain yet.** The current results provide diagnostic value that retraining would destroy.
2. **Fix D5 benchmark labeling** — Either change D5 tasks to have expected_terminal=CONTINUE, or update documentation to reflect ANSWER.
3. **Investigate V1's D2 false ANSWER force** — V1 forces ANSWER on 8 D2 tasks and fails all 8. This is a V1 weakness that V3's DEFER certificate correctly fixes.
4. **Consider relaxing V3's certificate threshold** — The 1 D5 break and 2 D1 breaks suggest V3's certificate is too conservative compared to V1's blunt Q-gap authority.
5. **Run a stronger model equivalence test** — 5 packets is insufficient. Use hundreds of packets near action boundaries.
6. **Preserve these 370 paired trajectories as the diagnostic dataset for I3.30R3.**

## V1 Remains the Confirmed Champion

V3R2-A shows genuine mechanism improvements (10 ANSWER-certificate rescues, 3 DEFER-certificate rescues) but also introduces regressions from Q-model differences and overly restrictive certificates. The net effect is positive (+9 success, +6.86 ΔU) but the mechanism is not yet reliable enough for promotion.
