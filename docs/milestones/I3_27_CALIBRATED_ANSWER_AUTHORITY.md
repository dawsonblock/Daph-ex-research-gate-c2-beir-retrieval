# I3.27: Calibrated ANSWER Authority — validated

**Date:** 2026-08-26
**Branch:** `i3.27-q-error-and-authority`
**Commit:** `90dd77d` (validation state), `5bb7aa6` (corrected claims + protections)
**Frozen champion:** `DAPH_ADAPTIVE_AUTHORITY_EXECUTIVE_V1`

---

## Milestone

I3.27 establishes that ANSWER-only adaptive authority beats advisory VP on a fresh replication benchmark, with disciplined scope and five engineering protections. The milestone is **Calibrated ANSWER Authority — validated**, not "general adaptive authority."

## Strongest defensible conclusion

ANSWER-only adaptive authority beat advisory VP on the 180-task replication benchmark, with +3.17 mean utility, a 95% CI of [+0.55, +5.79], 5 paired rescues, and 0 observed paired breaks.

## Release assessment

| Finding | Status |
|---------|--------|
| VP materially beats C0/B0 | Strong evidence |
| ANSWER A2A improves over VP | Replicated positive evidence |
| A2A caused observed breaks | None in this benchmark |
| True false-authority rate = 0 | Not established |
| B0 and C0 equivalent | Not established |
| ANSWER hard authority | Ready for narrow experimental deployment |
| General adaptive authority | Not established |
| DEFER authority | Unsafe with current representation |
| Threshold 5.0 | Keep frozen |
| More threshold tuning | Do not do it |

## What was corrected before freezing

1. **Benchmark label:** "fresh confirmation" → "fresh replication benchmark." The A2A-vs-VP result was partially inspected during the safety audit before the final replication run. Two gate interpretations were revised after inspection. An untouched seed is needed for publication-grade confirmation.

2. **Gate revision provenance:** Both original and revised gate definitions are preserved in `DAPH_ADAPTIVE_AUTHORITY_EXECUTIVE_V1.json` under `gate_revision`, with `madeAfterInspection: true`. The revisions are based on decision-state semantics, not observed outcomes.

3. **FalseAuthorityRate claim:** Softened from "0, the rule is safe" to "0/44 observed; true rate not established as zero; 44 is a small sample."

4. **B0≈C0 claim:** Softened from "equivalent" to "no statistically detectable improvement." A non-significant difference is not evidence of equivalence.

5. **Effect shrinkage explanation:** Softened from "due to fewer chain tasks" to "consistent with fewer chain opportunities." The category-level decomposition supports this but does not formally prove it.

## Category-level treatment effect

All 5 rescues and the entire +47.56 per-task delta-U on chain tasks come from the chain category only. A2A triggers on other categories (contradiction 9, search 12, tl_verify 10) but changes no outcomes there. A2A is essentially a chain-state repair rule, not a broadly useful authority mechanism.

Full decomposition: `experiments/i3_27/safety_audit/CATEGORY_TREATMENT_EFFECT.json`

## Architecture

```
State
  ↓
Q / causal evaluator
  ↓
action ranking + confidence gap
  ↓
Authority Gate
  ├─ weak/ambiguous → advisory
  └─ high-confidence eligible state → authoritative
                                      ↓
                                 selected action
```

Q does not directly execute an action. Q feeds an AuthorityPolicy that returns ALLOW_OVERRIDE or ADVISORY_ONLY. This leaves room to incorporate action allowlist, confidence, state class, verification state, budget, risk, and history without contaminating Q.

## Engineering protections

1. **Action allowlist:** `authoritativeActions = ["ANSWER"]`
2. **Frozen threshold:** `answerAuthorityGap = 5.0`, tuning prohibited
3. **Provenance event:** `authority/override` with `recommendedAction`, `modelAction`, `forcedAction`, `qForced`, `qRunnerUp`, `gap`, `ruleVersion`
4. **Runtime kill switch:** `authority.answer.enabled` (default false; true only for narrow experimental deployment)
5. **Shadow telemetry:** record what advisory execution would have chosen and the eventual outcome whenever A2A fires

## Scope

- A2 authority allowed: ANSWER only
- Threshold: 5.0, frozen
- DEFER: advisory only
- Other actions: advisory only

Do not generalize from ANSWER success to generic action authority. The DEFER experiment showed why: defer-correct state and contradiction state collapse to similar Q representations because `verify_result` is absent. Q lacks sufficient state observability to authorize DEFER safely.

## Architectural invariant

An action cannot receive hard authority when the Q-state representation omits variables required to distinguish known counterexamples for that action.

## Next milestone

### I3.28: Authority-State Sufficiency

The question is no longer "can we force more actions?" but:

> What state information must Q possess before a particular action is eligible for hard authority?

**Specific target:** DEFER authority via Q-state representation repair.

**Hypothesis:** Adding verification-resolution state (`verify_result`) enables safe DEFER authority.

**Approach:**
1. Add the smallest missing causal feature set to Q state representation.
2. Test offline separability of DEFER-correct vs contradiction-correct states.
3. If separable: run the same validation sequence (freeze → rescue audit → negative controls → safety audit → fresh untouched confirmation).
4. If not separable: stop. Do not force DEFER.

**Not approach:** Do NOT tune DEFER threshold. The blocker is representation, not threshold.

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
| Replication | 97861090 | 180 | `e54a3fec` |
