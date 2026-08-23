# R2-LIVE-SMOKE-001 — Exploratory Cross-Runtime Development Smoke

> **Classification**: R2-LIVE-SMOKE-001 (exploratory development smoke)
> **NOT** a qualified R2-DEV efficacy result. See §9 for the four reasons.
>
> **Backend**: Gemma 3 12B Q4_0 on Colab T4 GPU (llama-cpp-python)
> **Dataset**: 10 tasks × 4 arms = 40 trajectories
> **Seed**: 137 (new held-out, NOT 42)
> **Budget**: max_verification_calls=5 (same as R13)
> **Decoding**: non-strict (model wraps JSON in markdown code blocks; JSON extracted by brace-balanced parsing)
> **Date**: 2026-08-23
> **Model SHA (observed)**: `dd53172ff3a7b1b16c8fb3d944b87f42a6228ff2de3825b8813ae90d988434cd`
> **R13 frozen SHA (reference)**: `2ad4c9ce431a2d5b80af37983828c2cfb8f4909792ca5075e0370e3a71ca013d`
> **Backend/runtime**: llama-cpp-python on T4 (differs from R13's frozen llama.cpp executable/server configuration)

---

## 0. Provenance caveat (read first)

The Colab session was lost after the run completed. The analysis and trajectory
records in this directory were **reconstructed from console output**, not from
preserved raw model-call/state-transition receipts. The full per-call
provenance receipts (raw model outputs, decoder validity flags, gate-condition
transitions, admissibility assertions) were not recoverable.

Therefore this run cannot support mechanistic claims that require inspecting
individual model calls. It can only support aggregate directional observations
on the 10 paired tasks.

The next run (R2-DEV-V2) must persist every trajectory incrementally so that
session loss does not destroy raw traces.

---

## 1. Dataset Integrity

- Total trajectories: 40
- Per arm: C0=10, D=10, E=10, DE=10
- Tasks: first 10 from the seed=137 dataset
- **Stratum coverage**: all 10 tasks were gold-gate cases (gold_should_gate=true). No gold no-gate cases were included.

## 2. Gate Safety

```
Confusion Matrix (reconstructed):
                GoldGate    GoldNoGate
InferredGate       20          0
InferredNoGate      0          0
```

Corrected safety statement:

- **MissedGateRate** = FN / (FN + TP) = 0 / 20 = **0% — observed PASS**.
- **FalseGateRate** = FP / (FP + TN) = 0 / 0 = **NOT_ESTIMABLE** (undefined; no gold no-gate cases in this smoke).
- **Schema gate violations**: 0 in the reconstructed record.
- **Executor admissibility violations**: 0 in the reconstructed record.

This smoke tested MissedGate behavior. It did **not** test false gating.
The statement "all safety gates passed" is **removed**; it is not supported.

## 3. Hard-Gate Invariants and Decoder Qualification

Reconstructed record shows:
- Schema gate violations: 0
- Executor admissibility violations: 0

**However**, the live run used **non-strict decoding** (the model wrapped JSON
in markdown code fences, and a brace-balanced extractor recovered the payload).
This means the frozen R2 qualification boundary was **not** met:

- Q11 (decoder validity) was not demonstrated under strict constrained decoding.
- Q12 (schema-valid generation) was not demonstrated at generation time.

The correct statement is:

> "No gated VERIFY action reached execution in this smoke; strict
> constrained-decoding qualification (Q11/Q12) was not achieved."

The previous claim "Layer 1 and Layer 2 enforcement both perfect with live
model" is **removed**. What was observed is that no inadmissible action was
executed; whether the model would have produced one under strict decoding is
not established.

## 4. Utility Contrasts

| Contrast | Value |
|----------|-------|
| Δ_D | -20.428 |
| Δ_E | -39.784 |
| Δ_DE | -39.784 |
| I_D×E | +20.428 |

### Corrected interaction interpretation

Because U(DE) = U(E), the conditional effect of D when E is present is:

```
U(DE) - U(E) = 0
```

**D does not rescue E.** If D mitigated E, we would need U(DE) > U(E). We do
not have that.

The positive interaction arises because D is harmful without E but has no
additional effect once E has already driven the model into the all-STOP
behavioral floor. The correct interpretation is:

> "The D×E interaction is positive because D's negative effect disappears
> under E; this is consistent with saturation/floor behavior caused by the
> dominant E intervention."

The previous claim "D mitigates E's damage" is **removed**.

## 5. Success Breakdown

| Arm | Success Rate | Step Limit | N |
|-----|-------------|-----------|---|
| C0 | 0.40 | 0.00 | 10 |
| D | 0.20 | 0.00 | 10 |
| E | 0.00 | 0.00 | 10 |
| DE | 0.00 | 0.00 | 10 |

C0 is the only arm with any successes (4/10). D drops to 2/10. E and DE drop
to 0/10.

For C0 versus D specifically, the reconstructed rows show two C0 successes
became D failures while two successes remained successes. That is a
**directional warning against the hard VERIFY gate**, not yet a statistically
powered result.

## 6. Replacement-Action Distribution (D/DE at T2)

When the gate condition was active, the model selected:
- REASON_MORE: 24
- SEARCH_MORE: 21
- STOP: 18
- DEFER: 2

The model replaced VERIFY with a mix of reasoning, search, and stop. The STOP
selections (18) are the concerning component — the model frequently gives up
rather than productively redirecting.

## 7. Terminal Actions

- C0: 6 STOP, 4 DEFER
- D: 8 STOP, 2 DEFER
- E: 10 STOP, 0 DEFER
- DE: 10 STOP, 0 DEFER

E and DE produced all-STOP termination. The NO_VIABLE_HYPOTHESIS label appears
to cause the model to give up entirely rather than productively redirect.

## 8. Key Findings (corrected)

### What this smoke supports

1. **E shows a strong STOP-collapse signal**: 10/10 E trajectories and 10/10 DE
   trajectories terminate with STOP. NO_VIABLE_HYPOTHESIS is read by Gemma as
   a termination instruction. This is an important finding about representation
   semantics: a label can be logically accurate while inducing a bad policy
   prior.
2. **D shows a smaller negative signal**: success drops from 4/10 (C0) to 2/10
   (D). Directional warning against the hard VERIFY gate, not yet powered.
3. **Preserving VERIFY in C0 is associated with better outcomes than
   hard-gating it in D** on this 10-task smoke. This is an association, not a
   mechanistic claim that VERIFY → epistemic progress → success. The raw
   transition receipts needed to establish that chain were lost with the
   session.

### What this smoke does NOT support

1. It does **not** establish that the model uses VERIFY productively. That
   requires per-call transition receipts showing
   VERIFY → Δ decision_state ∨ Δ hypothesis_sets ∨ Δ T2.
2. It does **not** establish that D mitigates E. U(DE) = U(E); the
   interaction is a floor effect.
3. It does **not** establish FalseGateRate = 0. The rate is undefined here.
4. It does **not** establish strict decoder/schema qualification. Non-strict
   decoding was used.
5. It does **not** establish that either intervention should be scientifically
   rejected. This was 10 tasks, lacked false-gate controls, used an
   unqualified non-strict decoder, used a different backend/model artifact,
   and lost the original raw session traces.

### Model/backend identity

The observed model SHA is:

```
dd53172ff3a7b1b16c8fb3d944b87f42a6228ff2de3825b8813ae90d988434cd
```

This differs from R13's frozen SHA:

```
2ad4c9ce431a2d5b80af37983828c2cfb8f4909792ca5075e0370e3a71ca013d
```

The evidence establishes only that a different model artifact was used. It
does **not** establish why (the previous claim attributing this to
"HuggingFace updated the file" is **removed** — that causal claim was not
independently verified).

Furthermore, the runtime itself changed versus R13:

| Dimension | R13 (frozen) | R2-LIVE-SMOKE-001 |
|-----------|-------------|-------------------|
| llama.cpp runtime | frozen executable/server | llama-cpp-python |
| GGUF SHA | 2ad4c9ce... | dd53172f... |
| Hardware | (frozen config) | T4 GPU |
| Decoding | strict | non-strict (markdown extraction) |

This is not merely a model-SHA mismatch. It is a **backend/runtime
replication change**. The smoke is not an exact frozen R13 backend
reproduction.

## 9. Why this is classified as smoke, not qualified R2-DEV efficacy

1. **FalseGateRate is undefined, not zero.** The first 10 tasks were all
   gold-gate cases; no gold no-gate cases were included. The run tested
   MissedGateRate only.
2. **Non-strict decoder means the frozen qualification boundary was not met.**
   Q11/Q12 were not demonstrated. Constrained schema-valid generation was not
   enforced at generation time.
3. **The interaction was overinterpreted.** U(DE) = U(E) means D does not
   rescue E; the positive interaction is a floor/saturation effect.
4. **Raw session traces were lost.** The reconstructed trajectories cannot
   establish the VERIFY → progress → success chain or any per-call
   mechanistic claim.

## 10. Current interpretation

```
E shows a strong STOP-collapse signal.
D shows a smaller negative signal.
Neither intervention should yet be scientifically rejected from this run.
```

The most credible finding from these ten paired tasks is:

```
C0: 4/10,  D: 2/10,  E: 0/10,  DE: 0/10.
```

And the representation-semantics finding:

```
NO_VIABLE_HYPOTHESIS is probably a poor model-facing label.
It is technically descriptive but pragmatically loaded.
Gemma appears to read it as a termination instruction.
```

This is an important finding about representation semantics: a label can be
logically accurate while inducing a bad policy prior. The E label should be
**preserved unchanged** for the next properly qualified run so that this
signal can be confirmed under strict decoding, not tuned away against ten
observed tasks.

## 11. Required protocol for R2-DEV-V2

The next run should be R2-DEV-V2, not simply "more tasks." Requirements:

1. **Restore strict schema-constrained generation.** If llama-cpp-python
   cannot enforce the dynamic schema exactly, return to the frozen llama.cpp
   server path or implement the correct grammar/schema API. No markdown
   stripping as the scientific decoder.
2. **Pin backend/model identity.** Use a pinned GGUF with recorded
   repository, filename, size, and full SHA before task execution. It does
   not have to be the R13 GGUF for development, but it must remain fixed
   throughout R2-DEV-V2.
3. **Persist every trajectory incrementally.** Do not rely on terminal
   console output. Required files:
   - `results.jsonl`
   - `model_calls.jsonl`
   - `action_admissibility_receipts.jsonl`
   - `mechanism_receipts.jsonl`
   - `errors.jsonl`
   - `progress.json`
4. **Include gold no-gate cases** so FP + TN > 0. Specifically include
   one-live, two-live, matched-negative, and semantic false-T2 cases.
5. **Run frozen qualification before efficacy.**
   decoder = 100%, schema = 100%, violations = 0.
6. **Analyze mechanisms before utility.**

### Mechanism-first analysis template

First table (unconditional):

| Arm | VERIFY | RETRIEVE | SEARCH | REASON | DEFER | STOP | Exhausted |
|-----|--------|----------|--------|--------|-------|------|-----------|
| C0  |        |          |        |        |       |      |           |
| D   |        |          |        |        |       |      |           |
| E   |        |          |        |        |       |      |           |
| DE  |        |          |        |        |       |      |           |

Then condition on T2.

For every VERIFY in C0/E, calculate:

```
UsefulVerify = Δ decision_state ∨ Δ hypothesis_sets ∨ Δ T2
```

For every gated decision in D/DE, calculate what replaced VERIFY.

That directly answers the central question for the next run:

```
Was R2d harmful because it removed useful VERIFY,
or because Gemma reacts badly when VERIFY disappears?
```

Those are very different mechanisms and require per-call receipts to
distinguish.

### Clean sequence

```
patch live report classification  (this document)
        ↓
fix strict dynamic-schema decoding
        ↓
pin backend/model identity
        ↓
run R2-QUAL
        ↓
R2-DEV-V2 with gate + no-gate strata
        ↓
mechanism audit
        ↓
then decide whether D or E should be retired
```
