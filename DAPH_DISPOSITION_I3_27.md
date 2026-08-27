# DAPH Research Disposition — I3.27

**Date:** 2026-08-26
**Branch:** `i3.27-q-error-and-authority`
**Commit:** `90dd77d`
**Frozen champion:** `DAPH_ADAPTIVE_AUTHORITY_EXECUTIVE_V1`

---

## Executive Summary

The central causal question was:

> "same model + same prompt + same controller + same policy/runtime + different observable cognitive state"

The answer is now: **yes, when the executive has demonstrated reliable causal discrimination, adaptive authority improves a pinned model's decisions without breaking safety.**

The data does not support "hard authority everywhere." It supports calibrated authority where Q has proven it can distinguish correct from incorrect terminal actions.

---

## Control Regimes

```
ambiguous state       → advisory authority
clear high-gap ANSWER  → hard authority (threshold = 5.0)
DEFER state            → remain advisory until representation improves
```

---

## Dispositions

### SEARCH: REJECTED

- **Branch:** I3.26 selective search / PAV
- **Result:** `QCAUSAL_V2` rejected because retrieval values remained approximately constant across retrieval depths (`Q(s, RETRIEVE) ≈ 93.53`).
- **Stage B:** VP vs VS — selective search rejected. VP matched or beat VS on all categories.
- **Archive:** PAV/search code preserved in `daph/pav/`, `daph/search/`, `daph/executive/pav_search_controller.py`. Not deleted.
- **Rejection record:** Frozen at commit `bc4edc5`.
- **Verdict:** Do not revisit unless authority validation fails or a later experiment specifically requires deeper planning/search.

### Q_V2 / history-augmented Q: NOT JUSTIFIED

- **Rationale:** Q_CAUSAL_V1 is approximately correct on chain tasks. The bottleneck was policy compliance, not Q quality. Q identified the correct action (ANSWER) with strong gaps (12.20–13.54), but Qwen ignored advisory guidance.
- **OOD analysis:** Chain states are OOD (0/1056 match to training distribution), but Q extrapolates correctly to them.
- **Verdict:** Q retraining is not the bottleneck. Do not invest in Q_V2 unless a specific Q error is identified that authority calibration cannot address.

### PAV learned critic: NOT JUSTIFIED YET

- **Rationale:** The PAV/search branch was tested as a challenger and rejected. The learned critic did not improve over Q_CAUSAL_V1 + progress tie-break.
- **Verdict:** Archived. Do not revisit unless the current champion fails on a new benchmark category that PAV could address.

### ANSWER adaptive authority: CONFIRMED

- **Rule:** `A2A_ANSWER_ONLY_HARD_SELECT` — when `confidence == "clear"`, `refined_set == {ANSWER}`, and `q_gap > 5.0`, restrict schema to `{ANSWER}`. Otherwise advisory.
- **Development benchmark:** 15 rescues, 0 breaks, Delta U = +10.77 (CI [+5.78, +15.76]).
- **Confirmation benchmark:** 5 rescues, 0 breaks, Delta U = +3.17 (CI [+0.55, +5.79], p=0.019).
- **FalseAuthorityRate:** 0.0 on both benchmarks (0/34 dev, 0/44 confirmation).
- **Rescue audit:** All 15 development rescues verified as legitimate. Q had genuine ANSWER advantage, Qwen ignored it, A2A forced the causally-best action (forced-action return 99.89 vs 97.74).
- **Safety gates:** All 14 gates pass on the fresh confirmation benchmark.
- **B0 vs C0:** B0 (global prior) is NOT significantly better than C0 (no guidance), p=0.079. The value comes from Q+I2+Progress, not packet format.
- **Verdict:** Frozen as `DAPH_ADAPTIVE_AUTHORITY_EXECUTIVE_V1`. This is the current champion.

### DEFER hard authority: REJECTED with current representation

- **Blocker:** Q gives identical or insufficiently differentiated values for defer-correct and contradiction-incorrect states because `verify_result` is not in the feature representation.
- **Observed:** `Q(DEFER) = 69.34`, `Q(REASON_MORE) = 62.18`, gap = 7.16.
- **A2R with threshold 5.0:** Rescued defer but broke 10 contradiction tasks (122/156 vs VP 108/156, 24 rescues, 10 breaks, net 14).
- **Verdict:** Do NOT force DEFER with the current Q representation. The blocker is representation, not threshold tuning.

---

## Frozen Champion

```
DAPH_ADAPTIVE_AUTHORITY_EXECUTIVE_V1

base:
  Q_CAUSAL_V1          (sha256: d90d72da...)
  I2 epsilon_q = 3.0
  PROGRESS_RULE_V1      (sha256: 9f0bfc5e...)
  epsilon_p = 0.05

authority:
  ANSWER-only hard selection
  threshold = 5.0
  FalseAuthorityRate requirement = 0
  otherwise advisory

model:
  Qwen2.5-7B-Instruct-Q4_K_M.gguf  (sha256: 65b8fcd9...)

action vocabulary:
  R13 canonical: ANSWER, RETRIEVE, VERIFY, SEARCH_MORE, REASON_MORE, DEFER, STOP
```

Full frozen record: `experiments/i3_27/DAPH_ADAPTIVE_AUTHORITY_EXECUTIVE_V1.json`

---

## Next Research Target

### Representation repair for DEFER

**Hypothesis:** Adding verification-resolution state enables safe DEFER authority.

**Approach:**
1. Add the smallest missing causal feature set to Q state representation — starting with `verify_result` and whatever directly represents unresolved contradiction state.
2. Test whether DEFER-correct and contradiction-correct states become separable **offline** before any live hard-selection experiment.
3. If separable: run the same validation sequence used for ANSWER (freeze → rescue audit → negative controls → safety audit → fresh confirmation).
4. If not separable: stop. Do not force DEFER.

**Not approach:** Do NOT tune DEFER threshold. The blocker is representation, not threshold.

---

## Provenance

| Component | SHA256 (prefix) | Path |
|-----------|-----------------|------|
| Qwen GGUF | `65b8fcd9` | `/Users/dawsonblock/Downloads/qwen_gguf/Qwen2.5-7B-Instruct-Q4_K_M.gguf` |
| QCAUSAL_V1 | `d90d72da` | `experiments/i3_5/pinned_policy/frozen_estimators/QCAUSAL_gbt.pkl` |
| PROGRESS_RULE_V1 | `9f0bfc5e` | `daph/progress/progress_rule_v1.py` |
| Packet builder | `93e1b576` | `daph/executive/packet_builder.py` |
| A2A rule | `570c81e4` | `experiments/i3_27/authority_a2a/A2A_RULE_V1.json` |
| Utility config | `e5c6d34a` | `configs/v2b_i3_1_utility_v1.json` |
| Source commit | `90dd77d` | branch `i3.27-q-error-and-authority` |

| Benchmark | Seed | N | Hash (prefix) |
|-----------|------|---|---------------|
| Development | 7719 | 156 | `8d82d0a8` |
| Confirmation | 97861090 | 180 | `e54a3fec` |
