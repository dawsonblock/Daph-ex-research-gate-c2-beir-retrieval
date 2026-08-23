# R2 Experiment Specification: Structural Dead-End Affordance Gating × State-Semantics Correction

> **Status: SPECIFICATION (not yet implemented)**
> **Derived from: R13-F1.2a (commit 425325c)**
> **Forensic phase: CLOSED**
>
> R13-F1.2a established that VERIFY is structurally useless at T2 (0/2964 targets epistemically useful, 228/228 monotonic). R2 tests whether removing VERIFY from the admissible action set at T2 improves utility without harming controls.

---

## 1. Architecture

### 1.1 Corrected action-admissibility definition

The previous report defined `EpistemicallyAdmissible` as containing `Legal`, which would apply legality twice. The corrected definitions:

```
Legal(a, s)                  = executable under budgets, targets, and runtime
EpistemicallyAdmissible(a, s) = allowed by public epistemic structure
Allowed(a, s)                = Legal(a, s) ∩ EpistemicallyAdmissible(a, s)
```

For VERIFY under R2d:

```
EpistemicallyAdmissible(VERIFY, s) = ¬T2(s)
```

where `T2(s) = (|H| > 0 ∧ ∀h ∈ H, status(h) = ELIMINATED)`.

At T2:

```
T2(s) = 1  ⟹  VERIFY ∉ Allowed(s)
```

The LLM still owns policy selection. The controller restricts the action space, not the policy.

### 1.2 Action pipeline

```
ActionVocabulary → Legal → EpistemicallyAdmissible → LLM_Policy → Executor
```

- `ActionVocabulary`: the frozen 7-action set (ANSWER, RETRIEVE, VERIFY, SEARCH_MORE, REASON_MORE, DEFER, STOP)
- `Legal`: computed from budgets + valid targets (existing `can_verify`, `can_retrieve`, `can_search`)
- `EpistemicallyAdmissible`: computed from public structural state (new layer)
- `Allowed = Legal ∩ EpistemicallyAdmissible`: the actual action set exposed to the LLM
- `LLM_Policy`: Gemma selects from `Allowed` via constrained generation
- `Executor`: executes the selected action

### 1.3 Two-layer enforcement

**Layer 1 (primary intervention): Dynamic schema enum before generation**

The JSON schema's action enum is constructed per-call from `AllowedActions`:

```python
allowed_actions = compute_allowed_actions(snapshot, arm)
action_schema = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": sorted(allowed_actions),  # dynamic, not hardcoded
        },
        "reason_code": {"type": "string", "pattern": "^[A-Z][A-Z0-9_]*$"},
        "target_id": {"type": ["string", "null"]},
    },
    "required": ["action", "reason_code", "target_id"],
    "additionalProperties": False,
}
```

This prevents Gemma from generating a gated action, avoiding reject/retry loops.

**Layer 2 (defense-in-depth invariant assertion): Pre-execution check**

```python
if proposed_action not in allowed_actions:
    raise EpistemicAdmissibilityViolation(
        action=proposed_action,
        allowed=allowed_actions,
        reason=verify_gate_reason,
    )
```

This should NEVER fire in a qualified run. It is an invariant assertion, not the normal control mechanism. If it fires, the run is aborted and the qualification is repeated.

---

## 2. Arm Configuration

### 2.1 R2Arm dataclass

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class R2Arm:
    name: str
    structural_verify_gate: bool   # R2d: remove VERIFY from Allowed at T2
    corrected_t2_semantics: bool   # R2e: relabel NEEDS_DISCRIMINATION → NO_VIABLE_HYPOTHESIS at T2
```

### 2.2 The four arms

```python
C0 = R2Arm("C0", False, False)   # current behavior (baseline control)
D  = R2Arm("D",  True,  False)   # structural gate only
E  = R2Arm("E",  False, True)    # semantics correction only
DE = R2Arm("DE", True,  True)    # both interventions
```

### 2.3 What each arm changes

| Arm | Packet state label at T2 | VERIFY in Allowed at T2 | Everything else |
|-----|--------------------------|-------------------------|-----------------|
| C0 | NEEDS_DISCRIMINATION | yes (if Legal) | unchanged |
| D | NEEDS_DISCRIMINATION | **no** | unchanged |
| E | **NO_VIABLE_HYPOTHESIS** | yes (if Legal) | unchanged |
| DE | **NO_VIABLE_HYPOTHESIS** | **no** | unchanged |

**No other differences.** Same Gemma backend, same prompt (except experimentally required packet field), same retrieval (Q3_RERANKED), same semantic extractor, same MDSG computation, same T2 definition, same budgets, same utility, same model parameters, same representation routing (persistent M3 / current routing). **No transient M3.**

### 2.4 R2e is label-only

R2e changes ONLY the `decision_state` string in the packet when T2 fires. The underlying hypothesis-state computation (eliminated/live/weakened/untested sets, evidence_status, action_affordances) stays identical. This ensures E changes exactly one variable.

```python
if arm.corrected_t2_semantics and t2_fires:
    packet["decision_state_summary"]["decision_state"] = "NO_VIABLE_HYPOTHESIS"
    # evidence_status, live/eliminated sets, affordances: UNCHANGED
```

### 2.5 NO_VIABLE_HYPOTHESIS semantics

```
NO_VIABLE_HYPOTHESIS ⟺ |H| > 0 ∧ ∀h ∈ H, status(h) = ELIMINATED
```

- **Descriptive**: describes the observable state
- **Non-prescriptive**: does not recommend an action
- **Narrow**: applies only to the all-eliminated case
- **Maps directly onto T2**: `NO_VIABLE_HYPOTHESIS ⟺ T2`

NEEDS_DISCRIMINATION is preserved for its proper use: `|H_viable| ≥ 2` with evidence required to distinguish candidates.

---

## 3. Allowed-Action Computation

### 3.1 Legal actions (existing)

```python
legal_actions = set()
if snapshot.can_retrieve:
    legal_actions.add("RETRIEVE")
if snapshot.can_search:
    legal_actions.add("SEARCH_MORE")
if snapshot.can_verify:
    legal_actions.add("VERIFY")
# ANSWER, DEFER, STOP, REASON_MORE are always legal (not budget-constrained)
legal_actions.update({"ANSWER", "DEFER", "STOP", "REASON_MORE"})
```

### 3.2 Epistemically admissible actions (new)

```python
epistemically_admissible = set(legal_actions)  # start from all legal

if arm.structural_verify_gate:
    t2 = (n_hypotheses > 0 and len(eliminated_hypotheses) == n_hypotheses)
    if t2:
        epistemically_admissible.discard("VERIFY")
        verify_gate_applied = True
        verify_gate_reason = "ALL_HYPOTHESES_ELIMINATED"
    else:
        verify_gate_applied = False
        verify_gate_reason = None
else:
    verify_gate_applied = False
    verify_gate_reason = None
```

### 3.3 Allowed actions

```python
allowed_actions = legal_actions & epistemically_admissible
```

Since `EpistemicallyAdmissible` is defined independently of `Legal` (per §1.1), the intersection is correct. In practice, `epistemically_admissible` starts as a copy of `legal_actions` and only removes VERIFY when the gate fires, so `allowed_actions = epistemically_admissible` in the current implementation. The intersection formulation is kept for future extensibility.

---

## 4. Development Dataset

### 4.1 New held-out seed

Do NOT reuse R13 efficacy trajectories. Generate a new held-out seed/set.

### 4.2 Required strata

| Stratum | Description | Purpose |
|---------|-------------|---------|
| T2_IMMEDIATE | T2 fires at step 0-1 | Test gate at earliest trigger |
| T2_LATE_1 | T2 fires at step 2-3 | Test gate at mid-execution |
| T2_LATE_2 | T2 fires at step 4-5 | Test gate at late execution |
| T2_LATE_3 / nontrigger | T2 never fires | Negative control for gate |
| MATCHED_NEG_IMMEDIATE | Near-T2 but not all-eliminated, early | FalseGateRate test |
| MATCHED_NEG_LATE | Near-T2 but not all-eliminated, late | FalseGateRate test |
| DEFER_CONTROL | Tasks where DEFER is correct | Control preservation |
| ANSWER_CONTROL | Tasks where ANSWER is correct | Control preservation |

### 4.3 Structural perturbations around the causal boundary

Deliberately include:

- **Genuine all-eliminated T2 states** — the gate should fire
- **One-live-hypothesis near-T2 states** — the gate should NOT fire
- **Two-live-hypothesis true discrimination states** — the gate should NOT fire
- **False-contradiction semantic errors** — where a false contradiction incorrectly produces all-eliminated; the gate fires but shouldn't (FalseGateRate test)
- **Missed-contradiction cases** — where a contradiction should eliminate but doesn't; the gate should NOT fire
- **Retrieval still available vs exhausted** — tests whether RETRIEVE replaces VERIFY
- **Search still available vs exhausted** — tests whether SEARCH replaces VERIFY

This lets us test whether the gate is actually selective, not just whether it fires at T2.

---

## 5. Safety Metrics

### 5.1 FalseGateRate (first-class gate)

```
FalseGateRate = P(VERIFY removed | gold says VERIFY is epistemically relevant)
```

**Acceptance criterion**: `FalseGateRate = 0` on deterministic structural gold cases.

This is frozen BEFORE looking at efficacy. Given the importance of removing an action, we target zero, not "a few percent."

### 5.2 MissedGateRate

```
MissedGateRate = P(VERIFY remains allowed | gold structural dead end)
```

For D and DE, this should be zero when the structural rule applies.

### 5.3 Control preservation

- DEFER_CONTROL: DEFER rate unchanged across arms
- ANSWER_CONTROL: ANSWER rate unchanged across arms
- MATCHED_NEG: no utility harm on near-T2 states

---

## 6. Qualification

### 6.1 Mechanical qualification suite (before any efficacy run)

| Requirement | Criterion |
|-------------|-----------|
| decoder_valid_rate | 100% |
| schema_valid_rate | 100% |
| gated_action_execution_count | 0 |
| allowed_action/schema mismatches | 0 |
| FalseGateRate | 0 on structural gold cases |
| MissedGateRate | 0 on structural gold cases |
| C0 behavior | unchanged from baseline implementation |
| E changes | label only (no other packet field changes) |
| D changes | admissibility only (no label changes) |
| DE | exactly D + E (both changes, nothing else) |

### 6.2 Policy qualification

Because the constrained output space changes Gemma's behavior, requalify with policy tests.

Test states where:
- VERIFY is allowed — Gemma can still select VERIFY
- VERIFY is structurally gated — Gemma selects from remaining actions
- RETRIEVE+SEARCH remain — Gemma can redirect to evidence acquisition
- Only DEFER/ANSWER remain — Gemma must choose termination

Measure replacement-action distribution:

```
P(RETRIEVE), P(SEARCH), P(DEFER), P(ANSWER), P(REASON_MORE)
```

The first important development result will not be utility. It will be the replacement-action distribution.

---

## 7. Instrumentation

### 7.1 Per-call provenance

For every model call, record:

```json
{
  "legal_actions": ["ANSWER", "RETRIEVE", "VERIFY", "SEARCH_MORE", "REASON_MORE", "DEFER", "STOP"],
  "epistemically_admissible_actions": ["ANSWER", "RETRIEVE", "SEARCH_MORE", "REASON_MORE", "DEFER", "STOP"],
  "allowed_actions": ["ANSWER", "RETRIEVE", "SEARCH_MORE", "REASON_MORE", "DEFER", "STOP"],
  "allowed_actions_sha256": "<hash of sorted allowed_actions>",
  "schema_sha256": "<hash of instantiated JSON schema>",
  "verify_gate_applied": true,
  "verify_gate_reason": "ALL_HYPOTHESES_ELIMINATED",
  "t2": true,
  "decision_state": "NEEDS_DISCRIMINATION",
  "n_live_hypotheses": 0,
  "n_eliminated_hypotheses": 2,
  "valid_verify_target_count": 13,
  "selected_action": "RETRIEVE"
}
```

### 7.2 Dynamic schema hashing

Because D/DE dynamically alter the decoder schema, a single static schema SHA is insufficient. Freeze the schema-builder source, then hash the instantiated schema for every call:

```python
schema_sha = hashlib.sha256(
    json.dumps(action_schema, sort_keys=True).encode()
).hexdigest()
```

This is already done in `LocalLlamaBackend.generate()` for the static schema. The R2 runner must extend this to the dynamic schema.

### 7.3 Loop metrics

For each trajectory, track:

```
MaxRun(action)           = longest consecutive run of `action`
RepeatedActionRate(action) = P(action repeats | action was selected)
```

for RETRIEVE, SEARCH, VERIFY, and REASON_MORE.

This detects whether the loop is genuinely escaped or merely redirected:
- `VERIFY → RETRIEVE → RETRIEVE → RETRIEVE` = moved the loop
- `VERIFY → RETRIEVE → DEFER` = escaped (possibly premature)
- `VERIFY → RETRIEVE → ANSWER` = escaped (check support)

---

## 8. Primary Endpoints

### 8.1 Causal contrasts

```
Δ_D       = E[U_D  - U_C0]
Δ_E       = E[U_E  - U_C0]
I_{D×E}   = [U_DE - U_E] - [U_D - U_C0]
Δ_DE      = E[U_DE - U_C0]
```

Also calculate `Δ_DE` because if there is a strong interaction, the combined intervention could be useful even if D or E individually are not.

### 8.2 Mechanistic endpoint

```
P(VERIFY | T2, D)  = 0   (by construction, if hard gate works)
P(VERIFY | T2, DE) = 0   (by construction)
```

Then track what replaces VERIFY:

```
P(RETRIEVE | T2, D), P(SEARCH | T2, D), P(DEFER | T2, D),
P(ANSWER | T2, D), P(REASON_MORE | T2, D)
```

### 8.3 Disposition criteria

| Outcome | Interpretation |
|---------|---------------|
| D wins | Structural gating is sufficient |
| E wins | Semantic representation was the main issue |
| DE wins materially over both | Semantics + admissibility interact |
| None win | VERIFY was a symptom, not the root cause |
| D/DE harm controls | Structural gate is too aggressive |

### 8.4 Success requires

```
ΔU > 0  without control harm and without simply moving resource exhaustion to another action
```

### 8.5 Important failure modes

- VERIFY loop → RETRIEVE loop
- VERIFY loop → SEARCH loop
- VERIFY loop → premature DEFER
- VERIFY loop → unsupported ANSWER

---

## 9. Progression Pipeline

```
R2-QUAL (mechanical + policy qualification)
    ↓
small mechanical smoke (4 arms × few tasks)
    ↓
R2-DEV 2×2 (C0, D, E, DE on held-out development data)
    ↓
mechanism audit (replacement-action distribution, loop metrics, FalseGateRate)
    ↓
choose best(C0, D, E, DE)
    ↓
optional transient-M3 factor (R2cde vs best)
    ↓
freeze R2 architecture
    ↓
R2-CONFIRM on untouched tasks (fresh frozen confirmation corpus)
```

**Do not early-freeze the first positive variant.** Use R2-DEV as development. Inspect mechanism receipts, tune only if necessary, then create a fresh frozen confirmation corpus after the architecture is finalized.

---

## 10. Implementation Notes

### 10.1 Backend modification

The `LocalLlamaBackend.generate()` method currently hardcodes the action enum. For R2, it must accept an optional `allowed_actions` parameter:

```python
def generate(self, *, system_prompt, user_prompt, temperature, max_tokens,
             allowed_actions: list[str] | None = None) -> ModelCallResult:
    if allowed_actions is None:
        allowed_actions = ["ANSWER", "RETRIEVE", "VERIFY", "SEARCH_MORE",
                          "REASON_MORE", "DEFER", "STOP"]
    action_schema = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": sorted(allowed_actions)},
            ...
        },
        ...
    }
```

### 10.2 Runner modification

The R2 runner uses `R2Arm` to configure behavior per trajectory. The trajectory runner computes `allowed_actions` from the snapshot + arm, passes it to the backend, and logs the full instrumentation.

### 10.3 No changes to executor

The `EvidenceExecutor` is NOT modified. It remains the frozen deterministic executor. The gate is enforced before generation (schema) and before execution (invariant check), not inside the executor.

### 10.4 No changes to MDSG computation

The `_classify_from_snapshot` and `build_mdsg_state_with_affordances_packet` functions are NOT modified. R2e is a post-hoc label override on the packet, not a change to the underlying computation.

---

## 11. What This Experiment Tests

### 11.1 The core causal question

> Same model + same prompt + same controller + same policy/runtime + different admissibility structure

R2d tests whether restricting the action space to epistemically admissible actions improves a pinned model's decisions.

### 11.2 What it does NOT test

- Whether transient M3 helps (second stage, after choosing best gate+semantics)
- Whether the benchmark design is the root cause (separate analysis)
- Whether a different model would behave differently (pinned model)

### 11.3 The architectural lesson being tested

R13 exposed that DAPH conflated two concepts:

```
Can this action execute?  (Legal)
Can this action make epistemic progress under the visible structural state?  (EpistemicallyAdmissible)
```

If D improves utility without harming controls, that would be the first direct evidence that the `EpistemicallyAdmissible` layer contributes value independently of representation routing.

---

## 12. File Manifest (planned)

| File | Purpose |
|------|---------|
| `scripts/run_r2_development.py` | R2 runner with R2Arm configuration |
| `scripts/r2_allowed_actions.py` | Allowed-action computation + EpistemicallyAdmissible |
| `scripts/r2_qualification.py` | Mechanical + policy qualification suite |
| `scripts/r2_dataset_generator.py` | New held-out dataset with structural perturbations |
| `experiments/v2b_i3_15c/development/r2-dev/` | R2-DEV output directory |
| `experiments/v2b_i3_15c/development/r2-qual/` | R2-QUAL output directory |

---

## 13. Frozen Identities (to be established)

| Property | Value |
|----------|-------|
| Model | Gemma 3 12B IT QAT Q4_0 (same as R13) |
| GGUF SHA256 | 2ad4c9ce431a2d5b80af37983828c2cfb8f4909792ca5075e0370e3a71ca013d |
| Retrieval | Q3_RERANKED (same as R13) |
| max_tokens | 128 |
| ctx_size | 32768 |
| parallel_slots | 4 |
| temperature | 0.0 |
| seed | (new held-out seed, NOT 42) |
| Executor | frozen (same SHA as R13) |
| MDSG computation | frozen (same SHA as R13) |
| Schema builder | frozen (new SHA, established at R2-QUAL) |

---

## References

- R13-F1.2a report: `experiments/v2b_i3_15c/confirmation/r13/forensic/R13-F1-2-REPORT.md`
- R13-F main report: `experiments/v2b_i3_15c/confirmation/r13/forensic/R13-F-REPORT.md`
- F1.2a audit script: `scripts/r13_f1_2_counterfactual_audit.py`
- F1.2a commit: `425325c`
- Executor: `hrm_adaptive_memory/executive/evidence_benchmark/executor.py`
- MDSG packet builder: `scripts/run_i3_7e_compact_governor.py`
- Backend: `hrm_adaptive_memory/executive/model_backend.py`
- R13 runner: `scripts/run_r13_confirmation.py`
- I3.12j factorial: `scripts/run_i3_12j_factorial.py`
