# I3.30R Baseline State — BLOCKED

## Status: I3.30 is BLOCKED before live study execution

I3.30 will NOT proceed to live Qwen execution. An external audit identified
blocking scientific-validity defects that must be repaired first.

## Confirmed blocking findings

All findings below were verified against the actual repository code.

### P0-1: D5 claims CONTINUE-correct, but ANSWER succeeds immediately

**File:** `hrm_adaptive_memory/executive/evidence_benchmark/executor.py:210-241`

`_check_answer_success()` checks:
1. `expected_terminal is ANSWER`
2. correct hypothesis has SUFFICIENT support
3. no SUFFICIENT contradiction for correct hypothesis

It does NOT check uniqueness of supported hypothesis.

D5's state (H1 SUFFICIENT, H2 SUFFICIENT, expected=ANSWER, correct=H1)
passes immediately under ANSWER. The stratum cannot test abstention.

### P0-2: P3 causal training claims CONTINUE-correct, but ANSWER has highest utility

The I3.30B P3 regime has the same structure as D5. The collected causal
utilities show Q(ANSWER) > Q(CONTINUE). The training data itself
contradicts the documented label.

### P0-3: Controller and executor answer semantics disagree

The MDSG controller requires `|viable hypotheses| = 1` for answer-readiness.
The executor only checks the correct hypothesis. These two definitions
of "answer-ready" produce different results for competing verified support.

### P1-4: V3 ANSWER certificate accepts competing verified support

**File:** `daph/authority/policy_v3.py:117`

The legacy clause:
```python
if s.all_evidence_verified and s.n_hyp_with_verified_contradiction == 0:
    return True
```
does not check `has_verified_unresolved_competition`. A D5 state with
2 supported hypotheses and 0 contradictions passes the ANSWER certificate.

### P1-5: Offline FAR/precision are in-sample, not held-out

**File:** `scripts/run_i3_30_train_v3r.py:244-246`

The script explicitly states the evaluation is in-sample. The header
claims "held-out boundary states" but the implementation does not split.
FAR=0 and TerminalAuthorityPrecision=1.0 are training metrics, not
generalization evidence.

### P1-6: V3 FALSIFIED semantics disagree with canonical viability

**File:** `scripts/run_i3_30_v3_coverage.py:54`

```python
is_verified = vstate in ("SUFFICIENT", "FALSIFIED")
```

A FALSIFIED item with `supports(H)` is counted as verified support for H.
A FALSIFIED item with `contradicts(H)` is counted as verified contradiction
against H.

Canonical semantics should be:
- SUFFICIENT + supports(H) → verified support for H
- SUFFICIENT + contradicts(H) → verified contradiction against H
- FALSIFIED + supports(H) → support claim rejected (NOT verified support)
- FALSIFIED + contradicts(H) → contradiction claim rejected (NOT verified contradiction)

### P1-7: P2_elim uses FALSIFIED + contradicts(H) to mean "H eliminated"

**File:** `scripts/run_i3_30b_boundary_collection.py:300-306`

Evidence with `contradicts=("H1",)` and `verification_state=FALSIFIED`
is constructed to mean "H1 is eliminated." But FALSIFIED means the
contradiction *claim* was rejected, not that H1 was contradicted.

If the intended meaning is "H1 is eliminated," the evidence should be:
`verification_state=SUFFICIENT, contradicts=("H1",)`

### P2-8: verified_hyp_action creates a synthetic terminal-action shortcut

The benchmark generator assigns `answer_action = correct_action` to the
correct hypothesis and the opposite to wrong hypotheses. Once V3 identifies
the uniquely verified-supported hypothesis, `verified_hyp_action_is_answer`
almost directly reveals the correct terminal action.

This is observable (not oracle leakage) but limits the scientific claim
to "can the executive bind verified hypotheses to their stated consequences"
rather than general metacognitive state understanding.

### P2-9: Frozen manifest is mutable

`compute_manifest()` overwrites `frozen_manifest.json` on every run.

### P2-10: Runtime GGUF path not actually hashed

`compute_manifest()` hashes a hardcoded path, not the `--gguf-path` argument.

### P2-11: D5 hash covers task IDs only

`d5_benchmark_sha256` hashes `[t.task_id for t in d5_tasks]`, not full task content.

## What I3.30 does establish

1. The I3.29 failures exposed a real post-verification representation deficiency.
2. Hypothesis-level verified-evidence features provide additional discriminatory information.
3. Positive structural certificates are a better architectural direction than absence-of-danger.
4. The V3 implementation is deterministic, fail-advisory, and has good unit-test coverage.
5. Frozen artifact identities are internally consistent.

## What I3.30 does NOT establish

1. V3 has zero held-out false authority.
2. V3 terminal precision is 1.0 out of sample.
3. V3 safely abstains under verified ambiguity.
4. D5 validates ambiguous post-verification behavior.
5. V3 is ready for live confirmation.

## Root cause

The next bottleneck is NOT Q representation. It is semantic alignment
between four components that currently disagree:

```
Evidence verification semantics
    ↓
Hypothesis viability / MDSG state
    ↓
Q-state representation
    ↓
Executor success / benchmark truth
```

Further improvements in Q or authority can look impressive while optimizing
contradictory definitions of "correct."

## Next step

I3.30R: Semantic Consistency Repair and V3 Requalification.

The repair plan is documented in I3_30R_PREREGISTRATION.json.

No live Qwen experiment until the first six blocking findings are resolved.
