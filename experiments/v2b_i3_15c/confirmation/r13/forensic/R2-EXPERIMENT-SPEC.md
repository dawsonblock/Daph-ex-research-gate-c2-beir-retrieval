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

Epistemic admissibility is initialized from the full action vocabulary, NOT from legal_actions. This makes the architecture real: `EpistemicallyAdmissible` is independent of `Legal`, and `Allowed = Legal ∩ EpistemicallyAdmissible` combines them.

```python
ACTION_VOCABULARY = frozenset({
    "ANSWER", "RETRIEVE", "VERIFY", "SEARCH_MORE",
    "REASON_MORE", "DEFER", "STOP",
})

def epistemically_admissible_actions(state, arm):
    actions = set(ACTION_VOCABULARY)
    if arm.structural_verify_gate and state.t2:
        actions.remove("VERIFY")
    return frozenset(actions)
```

### 3.3 Allowed actions

```python
allowed_actions = legal_actions & epistemically_admissible_actions(state, arm)
```

This is the correct intersection. When no epistemic rule fires, `epistemically_admissible = ACTION_VOCABULARY`, so `allowed = legal_actions` (unchanged behavior). When the VERIFY gate fires at T2, `VERIFY` is removed from the epistemically admissible set, so `VERIFY ∉ allowed` regardless of whether it is legal.

### 3.4 Empty-action-set invariant

Require:

```
|Allowed(s)| ≥ 1
```

for every model call. If `allowed_actions` is empty, fail closed before calling the model:

```python
if not allowed_actions:
    raise EmptyAllowedActionSet(state=state, arm=arm)
```

This is qualification-fatal. ANSWER, DEFER, STOP, and REASON_MORE are always legal and always epistemically admissible under the current rules, so this should never fire. But the check must exist as a defense-in-depth invariant.

### 3.5 C0 schema identity (critical confound prevention)

R13 uses a static seven-action schema. R2 uses a dynamic action enum. If C0 goes through newly serialized schema code while the historical baseline did not, differences could be caused by schema serialization/order rather than D/E.

**Mandatory golden test**:

```
Schema_R2(Allowed = ACTION_VOCABULARY) == Schema_R13
```

byte-for-byte after canonical serialization, or at minimum identical canonical SHA.

```python
C0_FULL_ACTION_SCHEMA_SHA == R13_STATIC_SCHEMA_SHA
```

If this fails, do NOT start R2-DEV. This is critical because the spec requires C0 to remain unchanged.

### 3.6 Two-layer enforcement metrics

Track layer violations separately:

| Metric | Layer | Meaning |
|--------|-------|---------|
| `schema_gate_violations` | Layer 1 | Gated action generated despite not being in schema enum (should be 0 by construction) |
| `executor_admissibility_violations` | Layer 2 | Decoded action not in allowed_actions (should be 0; infrastructure/protocol failure if it fires) |

Do NOT retry an admissibility violation. If Layer 2 ever fires, that is an infrastructure/protocol failure, not model behavior. Abort the run.

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

### 6.1 Mechanical qualification suite (12 hard gates, before any efficacy run)

| Gate | ID | Criterion |
|------|----|-----------|
| Q1 | dynamic schema exactness | schema construction is deterministic and canonical |
| Q2 | C0 schema identity | `C0_FULL_ACTION_SCHEMA_SHA == R13_STATIC_SCHEMA_SHA` |
| Q3 | E packet diff | `decision_state` only (no other field changes) |
| Q4 | D packet diff | action admissibility/schema only (no label changes) |
| Q5 | DE diff | union(D, E), nothing else |
| Q6 | FalseGateRate | 0 on structural-gold cases |
| Q7 | MissedGateRate | 0 on structural-gold cases |
| Q8 | empty allowed set | 0 occurrences |
| Q9 | schema gate violations | 0 |
| Q10 | executor admissibility violations | 0 |
| Q11 | decoder valid | 100% |
| Q12 | schema valid | 100% |

All 12 are hard gates. Any failure aborts R2-DEV.

### 6.2 Policy qualification matrix

| State | Allowed set includes | Expected capability |
|-------|---------------------|---------------------|
| ordinary | VERIFY | Gemma can select VERIFY |
| T2/D | no VERIFY | selects valid replacement |
| T2 + retrieval available | RETRIEVE | can retrieve |
| T2 + search available | SEARCH_MORE | can search |
| T2 + neither available | ANSWER/DEFER/etc. | clean termination |
| non-T2 near-boundary | VERIFY | gate does not suppress it |

Do NOT require a particular replacement action yet. Qualification asks whether the model remains operable, not whether D improves efficacy.

### 6.3 Gold structural labels (for FalseGateRate/MissedGateRate)

The dataset generator emits gold labels separately from the inferred semantic pipeline:

```json
{
  "gold_t2": true,
  "gold_verify_relevant": false,
  "gold_should_gate_verify": true,
  "expected_terminal": "DEFER",
  "stratum": "T2_IMMEDIATE",
  "semantic_error_class": null,
  "retrieval_budget_case": "available",
  "search_budget_case": "exhausted"
}
```

Do NOT define gold using the same inferred semantic pipeline that drives T2, or the safety metric becomes circular.

```
FalseGateRate = #(gate=1 ∧ gold_should_gate=0) / #(gold_should_gate=0)
MissedGateRate = #(gate=0 ∧ gold_should_gate=1) / #(gold_should_gate=1)
```

### 6.4 FalseGate decomposition

FalseGateRate = 0 cannot be guaranteed if R2d keys off inferred T2 and the semantic extractor can make false contradictions. That is expected. So decompose:

```
FalseGate = FalseGate_semantic + FalseGate_MDSG/T2 + FalseGate_R2d
```

Where:
- `FalseGate_semantic`: gate fired due to upstream semantic error (false contradiction → false T2 → gate fires)
- `FalseGate_MDSG/T2`: gate fired due to MDSG/T2 computation error
- `FalseGate_R2d`: gate fired due to R2d logic error (no semantic error, but gate still fired wrongly)

For qualification, report separately:

| Gate | Criterion |
|------|-----------|
| Q6a R2d logic FalseGate | 0 (structural logic must be perfect) |
| Q6b end-to-end FalseGate under semantic inference | measured (may be nonzero due to semantic errors) |

If end-to-end FalseGateRate=0 is required as a hard gate, the semantic-error strata may make the system impossible to qualify even though the R2d implementation is perfectly correct. That could be scientifically useful, but it should be intentional.

The confusion matrix:

```
              GoldGate    GoldNoGate
InferredGate     TP          FP
InferredNoGate   FN          TN

FalseGateRate = FP / (FP + TN)
MissedGateRate = FN / (FN + TP)
```

---

## 7. Instrumentation

### 7.1 Per-call provenance

For every model call, record:

```json
{
  "arm": "D",
  "t2": true,
  "gold_t2": true,
  "decision_state_internal": "NEEDS_DISCRIMINATION",
  "decision_state_exposed": "NEEDS_DISCRIMINATION",
  "legal_actions": ["ANSWER", "RETRIEVE", "SEARCH_MORE", "REASON_MORE", "DEFER", "STOP"],
  "epistemically_admissible_actions": ["ANSWER", "RETRIEVE", "SEARCH_MORE", "REASON_MORE", "DEFER", "STOP"],
  "allowed_actions": ["ANSWER", "RETRIEVE", "SEARCH_MORE", "REASON_MORE", "DEFER", "STOP"],
  "allowed_actions_sha256": "<hash of sorted allowed_actions>",
  "verify_gate_applied": true,
  "verify_gate_reason": "ALL_HYPOTHESES_ELIMINATED",
  "schema_sha256": "<hash of instantiated JSON schema>",
  "schema_action_enum": ["ANSWER", "DEFER", "REASON_MORE", "RETRIEVE", "SEARCH_MORE", "STOP"],
  "selected_action": "RETRIEVE",
  "admissibility_assertion_passed": true
}
```

For E/DE, logging both `decision_state_internal` and `decision_state_exposed` proves E is truly presentation-only. When `arm.corrected_t2_semantics and t2`, `decision_state_exposed = "NO_VIABLE_HYPOTHESIS"` while `decision_state_internal = "NEEDS_DISCRIMINATION"`.

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

### 9.1 R2-DEV analysis order

Do NOT start with utility. Analyze in this order:

1. Dataset integrity
2. Arm isolation/diff audit
3. FalseGate/MissedGate
4. Hard-gate invariants (schema_gate_violations=0, executor_admissibility_violations=0)
5. Replacement-action distribution
6. Loop migration
7. Success/rescue/break
8. Utility contrasts
9. D×E interaction

### 9.2 Loop migration metrics

Beyond `MaxRun(action)` and `RepeatedActionRate(action)`, calculate:

```
P(ResourceExhausted | terminal_action = a)
```

for each action. This directly tells whether R2 changed VERIFY exhaustion into RETRIEVE exhaustion, or genuinely escaped the pathological policy basin.

Composite metric:

```
LoopMigrationRate = P(ResourceExhausted ∧ terminal_action ≠ VERIFY | D/DE, T2)
```

### 9.3 The primary scientific question

> Does structural epistemic admissibility redirect policy toward productive actions rather than merely relocate the loop?

### 9.4 Pipeline

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

### 10.1 Module: r2_allowed_actions.py

Pure logic only. No backend, no executor mutation, no counterfactual simulator.

```python
@dataclass(frozen=True)
class ActionState:
    t2: bool
    executive_steps_remaining: int
    can_retrieve: bool
    can_search: bool
    can_verify: bool

@dataclass(frozen=True)
class AllowedActionDecision:
    legal: frozenset[str]
    epistemically_admissible: frozenset[str]
    allowed: frozenset[str]
    verify_gate_applied: bool
    verify_gate_reason: str | None
```

This module has exhaustive unit tests because it is part of the scientific intervention.

### 10.2 Module: r2_schema.py

Canonical dynamic-schema construction only:

```python
def build_action_schema(allowed_actions: frozenset[str]) -> dict
def canonical_schema_json(schema: dict) -> str
def schema_sha256(schema: dict) -> str
```

Invariant: `assert set(schema["properties"]["action"]["enum"]) == set(allowed_actions)`

Sort only for deterministic serialization. Do NOT introduce any new action descriptions, reason-code hints, target hints, or wording changes between arms.

### 10.3 Module: r2_dataset_generator.py

Emits tasks plus gold structural labels separately from inferred state. Gold labels include: `gold_t2`, `gold_verify_relevant`, `gold_should_gate_verify`, `expected_terminal`, `stratum`, `semantic_error_class`, `retrieval_budget_case`, `search_budget_case`.

Do NOT define gold using the same inferred semantic pipeline that drives T2, or the safety metric becomes circular.

### 10.4 Module: r2_qualification.py

Two phases: mechanical (12 hard gates) then policy (6-state matrix).

### 10.5 Module: run_r2_development.py

One execution function for all arms:

```python
def run_trajectory(task, arm, backend, ...):
    ...
```

The arm flows through exactly two intervention hooks:

```python
packet = apply_semantics_intervention(packet, state, arm)
allowed = compute_allowed_actions(state, arm)
```

Everything else must be shared.

### 10.6 Backend modification

The `LocalLlamaBackend.generate()` method currently hardcodes the action enum. For R2, it must accept an optional `allowed_actions` parameter. When `allowed_actions` is None or equals the full vocabulary, the schema must be byte-identical to the R13 static schema (verified by Q2).

### 10.7 No changes to executor

The `EvidenceExecutor` is NOT modified. It remains the frozen deterministic executor. The gate is enforced before generation (schema) and before execution (invariant check), not inside the executor.

### 10.8 No changes to MDSG computation

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
| `scripts/r2_allowed_actions.py` | Pure logic: ActionState, AllowedActionDecision, compute_allowed_actions |
| `scripts/r2_schema.py` | Canonical dynamic schema construction + SHA |
| `scripts/r2_dataset_generator.py` | New held-out dataset with gold structural labels |
| `scripts/r2_qualification.py` | 12 mechanical gates + policy qualification matrix |
| `scripts/run_r2_development.py` | R2 runner: single execution function, arm hooks, per-call receipts |
| `tests/test_r2_allowed_actions.py` | Exhaustive unit tests for allowed-action logic |
| `experiments/v2b_i3_15c/development/r2-dev/` | R2-DEV output directory |
| `experiments/v2b_i3_15c/development/r2-qual/` | R2-QUAL output directory |

### 12.1 Module separation

- `r2_allowed_actions.py`: pure logic only. No backend, no executor mutation, no counterfactual simulator.
- `r2_schema.py`: canonical schema construction only. No action descriptions, reason-code hints, target hints, or wording changes between arms.
- `r2_dataset_generator.py`: emits tasks plus gold structural labels separately from inferred state.
- `r2_qualification.py`: two phases (mechanical then policy).
- `run_r2_development.py`: one execution function for all arms, arm flows through exactly two intervention hooks.

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
