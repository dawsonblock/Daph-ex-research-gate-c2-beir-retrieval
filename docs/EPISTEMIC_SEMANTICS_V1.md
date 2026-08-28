# EPISTEMIC SEMANTICS V1

## Normative Specification for DAPH Evidence, Hypothesis Viability, and Terminal Readiness

**Status:** FROZEN
**Experiment:** I3.30R
**Date:** 2026-08-27
**Predecessor:** None (this is the first normative semantic specification)
**Authority:** This document is the single source of truth for epistemic semantics. All implementations — MDSG classifier, Q feature extraction, authority certificates, executor success criteria, benchmark truth labels, offline analysis — MUST conform to this specification.

---

## 1. Purpose

Four components in the DAPH system previously disagreed about what the epistemic state means:

```
Evidence verification semantics
    ↓
Hypothesis viability / MDSG state
    ↓
Q-state representation
    ↓
Executor success / benchmark truth
```

This document defines one canonical interpretation. The implementation rule is:

```
one evidence semantics → one topology → many consumers
```

Not four independent implementations that happen to agree on the easy cases.

---

## 2. Evidence Items

An evidence item is a proposition-level claim about one or more hypotheses.

### 2.1 Evidence Item Fields

| Field | Type | Observable | Description |
|---|---|---|---|
| `evidence_id` | str | Yes | Unique identifier (e.g. "E1") |
| `proposition` | str | Yes | The claim this evidence makes |
| `source_class` | str | Yes | Category of source (e.g. "primary", "secondary", "search") |
| `supports` | tuple[str, ...] | Yes | Hypothesis IDs this evidence claims to support |
| `contradicts` | tuple[str, ...] | Yes | Hypothesis IDs this evidence claims to contradict |
| `verification_state` | VerificationState | Yes | Current verification status (see §3) |
| `temporal_status` | TemporalStatus | Yes | CURRENT, STALE, or UNKNOWN |
| `retrieved` | bool | Yes | Whether this item is visible to the controller |
| `verify_result` | str \| None | **No** | Evaluator-only: what VERIFY will produce. NOT visible to controller, Q, or authority. |

### 2.2 Evidence Relations

Each evidence item makes zero or more claims about hypotheses:

- **supports(H)**: The evidence proposition, if true, provides positive support for hypothesis H.
- **contradicts(H)**: The evidence proposition, if true, provides positive contradiction against hypothesis H.

A single evidence item may support some hypotheses and contradict others simultaneously. An item may also have empty supports and contradicts (neutral evidence).

The `supports` and `contradicts` fields describe the *claim* the evidence makes, not whether the claim has been established. Whether a claim is established depends on the verification state (§3).

---

## 3. Verification States

### 3.1 Definition

| State | Meaning |
|---|---|
| `UNVERIFIED` | The evidence proposition has not been tested. No evidential force. |
| `SUFFICIENT` | The evidence proposition has been established as true/sufficient. The claim is active. |
| `FALSIFIED` | The evidence proposition has been rejected as false/insufficient. The claim has failed. |
| `STALE` | The evidence was verified but is now temporally outdated. No current evidential force. |
| `MISSING` | The evidence item does not exist or is inaccessible. No evidential force. |

### 3.2 The Normative Rule

This is the most important rule in the entire specification. Every implementation must conform to it exactly.

**The verification state applies to the evidence proposition, not to the hypothesis.**

When an evidence item has `supports(H)` and is verified `SUFFICIENT`:
- The support proposition is established as true.
- H receives **positive verified support**.

When an evidence item has `contradicts(H)` and is verified `SUFFICIENT`:
- The contradiction proposition is established as true.
- H receives **positive verified contradiction** (H is actively contradicted by verified evidence).

When an evidence item has `supports(H)` and is verified `FALSIFIED`:
- The support proposition has been rejected.
- H does **NOT** receive verified support.
- H does **NOT** receive verified contradiction.
- The support claim simply failed. It carries no positive evidential force in either direction.

When an evidence item has `contradicts(H)` and is verified `FALSIFIED`:
- The contradiction proposition has been rejected.
- H does **NOT** receive verified contradiction.
- H does **NOT** receive verified support.
- The contradiction claim simply failed. It carries no positive evidential force in either direction.

### 3.3 Truth Table

| Verification State | Relation | Effect on H |
|---|---|---|
| SUFFICIENT | supports(H) | H gains verified support |
| SUFFICIENT | contradicts(H) | H gains verified contradiction |
| FALSIFIED | supports(H) | No effect on H (support claim failed) |
| FALSIFIED | contradicts(H) | No effect on H (contradiction claim failed) |
| UNVERIFIED | supports(H) | H gains unverified support (weak) |
| UNVERIFIED | contradicts(H) | H gains unverified contradiction (weak) |
| STALE | any | No effect (temporal invalidation) |
| MISSING | any | No effect (evidence absent) |

### 3.4 What FALSIFIED Does NOT Mean

**FALSIFIED does not mean "the hypothesis is falsified."**

It means "the evidence proposition was rejected during verification."

A FALSIFIED support claim is not evidence against the hypothesis. It is simply a failed attempt to support it. The hypothesis may still be true; the evidence just didn't establish it.

A FALSIFIED contradiction claim is not evidence for the hypothesis. It is simply a failed attempt to contradict it. The hypothesis may still be false; the contradiction just wasn't established.

This rule is the primary semantic fix for I3.30R. The previous V3 feature extractor incorrectly treated FALSIFIED as a verified state, routing FALSIFIED+supports as "verified support" and FALSIFIED+contradicts as "verified contradiction." Both are wrong.

---

## 4. Hypothesis States

### 4.1 Definitions

Given a set of hypotheses and visible (retrieved) evidence with verification states, each hypothesis is classified into exactly one state:

| State | Definition |
|---|---|
| **SUPPORTED** | Has at least one SUFFICIENT, CURRENT evidence item supporting it, AND no SUFFICIENT, CURRENT evidence item contradicting it. |
| **CONTRADICTED** | Has at least one SUFFICIENT, CURRENT evidence item contradicting it. (Regardless of whether it also has support — see §4.3.) |
| **WEAKENED** | Has at least one FALSIFIED support claim (support attempted and failed), no SUFFICIENT support, and no SUFFICIENT contradiction. |
| **UNTESTED** | Has no SUFFICIENT or FALSIFIED evidence relating to it. Only UNVERIFIED or no evidence. |
| **STALE** | All previously sufficient evidence is now STALE. No current verified force. |

### 4.2 Priority

If a hypothesis has both SUFFICIENT support and SUFFICIENT contradiction (mixed evidence), it is classified as **CONTRADICTED**. Verified contradiction takes priority over verified support. This is the conservative rule: a verified contradiction is epistemically stronger than a verified support, because it actively refutes the hypothesis.

### 4.3 Mixed Evidence

A hypothesis with both verified support and verified contradiction is in a state of internal epistemic conflict. The specification classifies this as CONTRADICTED (conservative). However, the topology derivation (§5) must record that both exist, because downstream consumers may need to distinguish "contradicted with no support" from "contradicted despite support."

### 4.4 Viability

A hypothesis is **viable** if and only if it is SUPPORTED.

- CONTRADICTED hypotheses are not viable.
- WEAKENED hypotheses are not viable (support failed).
- UNTESTED hypotheses are not viable (no verified support).
- STALE hypotheses are not viable.

This is stricter than "not eliminated." A hypothesis is viable only when it has active positive verified support.

### 4.5 Elimination

A hypothesis is **eliminated** if and only if it is CONTRADICTED (has at least one SUFFICIENT, CURRENT contradiction).

Note: A hypothesis with only FALSIFIED contradictions is NOT eliminated. The contradiction claims failed verification.

---

## 5. Canonical Hypothesis Topology

### 5.1 Definition

The **canonical hypothesis topology** is the single derived structure that all consumers must use. It is computed from observable evidence only.

### 5.2 Topology Structure

```
HypothesisTopology:
    # Per-hypothesis classification
    hypothesis_states: dict[str, HypothesisState]  # SUPPORTED, CONTRADICTED, WEAKENED, UNTESTED, STALE

    # Aggregate counts
    n_viable_hypotheses: int          # count of SUPPORTED
    n_eliminated_hypotheses: int      # count of CONTRADICTED
    n_untested_hypotheses: int        # count of UNTESTED
    n_weakened_hypotheses: int        # count of WEAKENED
    n_stale_hypotheses: int           # count of STALE
    n_total_hypotheses: int

    # Verified evidence topology
    n_hyp_with_verified_support: int       # count with >=1 SUFFICIENT support
    n_hyp_with_verified_contradiction: int # count with >=1 SUFFICIENT contradiction
    n_hyp_with_mixed_verified: int         # count with both SUFFICIENT support and SUFFICIENT contradiction

    # Resolution state
    unique_supported_hypothesis: str | None  # the single SUPPORTED hypothesis ID, or None
    has_verified_unresolved_competition: bool # True if >1 hypothesis has SUFFICIENT support
    has_unique_verified_supported: bool       # True if exactly 1 hypothesis has SUFFICIENT support

    # Evidence completeness
    verification_complete: bool          # all visible evidence is SUFFICIENT or FALSIFIED
    unverified_evidence_exists: bool     # any visible evidence is UNVERIFIED
    hidden_evidence_count: int           # count of non-retrieved evidence items

    # Per-hypothesis detail (for consumers that need it)
    verified_support_by_hypothesis: dict[str, list[str]]    # evidence IDs with SUFFICIENT support
    verified_contradiction_by_hypothesis: dict[str, list[str]]  # evidence IDs with SUFFICIENT contradiction
    falsified_support_by_hypothesis: dict[str, list[str]]   # evidence IDs with FALSIFIED support
    falsified_contradiction_by_hypothesis: dict[str, list[str]]  # evidence IDs with FALSIFIED contradiction
    unverified_support_by_hypothesis: dict[str, list[str]]
    unverified_contradiction_by_hypothesis: dict[str, list[str]]
```

### 5.3 Derivation Rules

The topology is derived using these rules, which directly implement §3.2:

1. For each visible (retrieved) evidence item with `temporal_status == CURRENT`:
   - If `verification_state == SUFFICIENT`:
     - For each `h_id` in `supports`: add `evidence_id` to `verified_support_by_hypothesis[h_id]`
     - For each `h_id` in `contradicts`: add `evidence_id` to `verified_contradiction_by_hypothesis[h_id]`
   - If `verification_state == FALSIFIED`:
     - For each `h_id` in `supports`: add `evidence_id` to `falsified_support_by_hypothesis[h_id]`
     - For each `h_id` in `contradicts`: add `evidence_id` to `falsified_contradiction_by_hypothesis[h_id]`
   - If `verification_state == UNVERIFIED`:
     - For each `h_id` in `supports`: add `evidence_id` to `unverified_support_by_hypothesis[h_id]`
     - For each `h_id` in `contradicts`: add `evidence_id` to `unverified_contradiction_by_hypothesis[h_id]`
   - If `verification_state in {STALE, MISSING}`: no effect.

2. For each hypothesis:
   - `has_verified_support = len(verified_support_by_hypothesis[h_id]) > 0`
   - `has_verified_contradiction = len(verified_contradiction_by_hypothesis[h_id]) > 0`
   - If `has_verified_contradiction`: state = CONTRADICTED
   - Elif `has_verified_support`: state = SUPPORTED
   - Elif `len(falsified_support_by_hypothesis[h_id]) > 0`: state = WEAKENED
   - Elif all evidence relating to this hypothesis is STALE: state = STALE
   - Else: state = UNTESTED

3. Aggregate counts are computed from per-hypothesis states.

4. `unique_supported_hypothesis` is the single hypothesis ID with state SUPPORTED, or None if the count is not exactly 1.

5. `has_verified_unresolved_competition` is True if `n_hyp_with_verified_support > 1` (more than one hypothesis has SUFFICIENT support).

6. `has_unique_verified_supported` is True if `n_hyp_with_verified_support == 1`.

### 5.4 Observability

The topology derivation consumes ONLY:
- `evidence_id`, `supports`, `contradicts`, `verification_state`, `temporal_status`, `retrieved` from evidence items
- `hypothesis_id` from hypotheses
- `hidden_evidence_count` (count only, not content)

It does NOT consume:
- `verify_result`
- `correct_hypothesis_id`
- `expected_terminal`
- `oracle_resolution_path`
- `retrieve_exposes`, `search_exposes`
- Hidden evidence content
- Future actions or outcomes
- Terminal correctness
- Downstream utility

---

## 6. Terminal Readiness Semantics

### 6.1 ANSWER_READY

A state is **ANSWER_READY** if and only if:

1. There is exactly one SUPPORTED hypothesis (`unique_supported_hypothesis is not None`), AND
2. No unresolved competing verified support exists (`not has_verified_unresolved_competition`), AND
3. The supported hypothesis maps to an ANSWER-capable resolution (its `answer_action is ANSWER`).

Condition 2 is automatically satisfied when condition 1 holds (if exactly one hypothesis is SUPPORTED, there cannot be >1 with verified support). It is stated explicitly for clarity and defense-in-depth.

**ANSWER is epistemically justified only when ANSWER_READY is true.**

This means ANSWER requires:
- A single hypothesis with verified support
- No verified contradiction against that hypothesis
- No other hypothesis with verified support

A state with two hypotheses both having SUFFICIENT support is NOT ANSWER_READY, even if one of them is the correct hypothesis. The competing verified support means the evidence does not uniquely resolve the question.

### 6.2 DEFER_READY

A state is **DEFER_READY** if and only if:

1. ANSWER_READY is false, AND
2. No admissible epistemic continuation can materially resolve the state.

"Admissible epistemic continuation" means an action that could change the evidence topology in a way that might make ANSWER_READY true. Specifically:
- VERIFY on an unverified evidence item that supports or contradicts a hypothesis could change its verification state, potentially creating or removing verified support/contradiction.
- RETRIEVE could expose hidden evidence that, when verified, might resolve the competition.
- SEARCH_MORE could expose new evidence with discriminating power.
- REASON_MORE does NOT change evidence topology and does NOT count as an epistemic continuation for this purpose.

DEFER_READY requires that:
- No unverified visible evidence remains that could discriminate between hypotheses, AND
- No hidden evidence remains retrievable, AND
- No search can produce new discriminating evidence, AND
- No verification calls remain (or all verifiable evidence has been verified)

Resource exhaustion (no verify/retrieve/search budget remaining) contributes to DEFER_READY by making continuations inadmissible, but it does NOT change the evidence topology itself.

### 6.3 CONTINUE_REQUIRED

A state is **CONTINUE_REQUIRED** if and only if:

1. ANSWER_READY is false, AND
2. DEFER_READY is false, AND
3. At least one useful epistemic continuation action is admissible.

A "useful epistemic continuation" is an action that could materially change the evidence topology:
- VERIFY on untested evidence that relates to a hypothesis
- RETRIEVE when hidden evidence exists that could discriminate
- SEARCH_MORE when search might produce discriminating evidence

REASON_MORE and STOP are not epistemic continuations for this purpose. REASON_MORE may improve the model's internal state but does not change the evidence topology. STOP is terminal.

### 6.4 The Complete Terminal Readiness Lattice

```
                    ANSWER_READY?
                   /            \
                 Yes             No
                 |               |
             ANSWER          DEFER_READY?
                            /            \
                          Yes             No
                          |               |
                       DEFER          CONTINUE_REQUIRED
                                      (if continuation admissible)
                                          |
                                      No continuation
                                      admissible?
                                          |
                                     DEFER_READY
                                     (by exhaustion)
```

Every state falls into exactly one of: ANSWER_READY, DEFER_READY, or CONTINUE_REQUIRED.

---

## 7. Resource State vs Epistemic State

### 7.1 Separation Principle

**Resource state and epistemic state are distinct.**

Resource exhaustion does NOT change the evidence topology. A hypothesis does not become "resolved" or "eliminated" because `verify_remaining == 0`. The topology depends only on what evidence has been verified and what it established.

### 7.2 What Resource State Affects

Resource state affects **action admissibility**, not epistemic state:
- `verify_remaining == 0` → VERIFY is not admissible
- `retrieval_remaining == 0` → RETRIEVE is not admissible
- `search_remaining == 0` → SEARCH_MORE is not admissible
- `steps_remaining == 0` → no actions are admissible (step limit)

### 7.3 How Resource State Affects Terminal Readiness

Resource state affects DEFER_READY by determining whether continuations are admissible:
- If VERIFY is admissible and unverified discriminating evidence exists → CONTINUE_REQUIRED
- If VERIFY is not admissible (budget exhausted) but no unverified evidence remains → DEFER_READY may hold
- If RETRIEVE is admissible and hidden evidence exists → CONTINUE_REQUIRED
- If RETRIEVE is not admissible and no hidden evidence remains → not a continuation

Resource exhaustion can make DEFER_READY true by eliminating all admissible continuations, but it does not change which hypotheses are supported or contradicted.

### 7.4 What Resource State Does NOT Affect

- `n_viable_hypotheses` — depends on evidence topology only
- `n_eliminated_hypotheses` — depends on evidence topology only
- `unique_supported_hypothesis` — depends on evidence topology only
- `has_verified_unresolved_competition` — depends on evidence topology only

---

## 8. Observable vs Evaluator-Only Information

### 8.1 Observable (Controller-Visible)

The following are observable and MAY enter Q features, authority certificates, LLM packets, and topology derivation:

**Evidence:**
- `evidence_id`
- `proposition`
- `source_class`
- `supports`
- `contradicts`
- `verification_state`
- `temporal_status`
- `retrieved`

**Hypotheses:**
- `hypothesis_id`
- `proposition`
- `answer_action`
- `answer_payload`

**Resources:**
- All resource counts (remaining calls, steps used, etc.)
- All budget parameters

**Runtime:**
- `searched`
- `reasoning_complete`
- `prior_actions`
- `prior_outcomes`
- `hidden_evidence_count` (count only, NOT content)

### 8.2 Evaluator-Only (Prohibited from Q, Authority, LLM)

The following are evaluator-only and MUST NOT enter Q features, authority certificates, LLM packets, or topology derivation:

- `verify_result` — the oracle outcome of verification, known only after VERIFY executes
- `correct_hypothesis_id` — which hypothesis is actually correct
- `expected_terminal` — what the benchmark expects the terminal action to be
- `oracle_resolution_path` — the sequence of actions that solves the task
- `retrieve_exposes` — which hidden evidence RETRIEVE will expose
- `search_exposes` — which hidden evidence SEARCH_MORE will expose
- Hidden evidence content (propositions, supports, contradicts of non-retrieved items)
- Terminal correctness (whether the task was solved)
- Future actions
- Downstream utility
- Future outcomes

### 8.3 The `answer_action` Field

`EvidenceHypothesis.answer_action` is observable. It is included in the controller-visible snapshot and the LLM packet.

However, it creates a **benchmark shortcut**: the benchmark generator assigns `answer_action = correct_action` to the correct hypothesis and the opposite to wrong hypotheses. Once the uniquely verified-supported hypothesis is identified, `answer_action` almost directly reveals the correct terminal action.

This is not oracle leakage (the field is genuinely observable), but it limits the scientific claim. The claim is:

> "Can the executive bind verified hypotheses to their stated terminal consequences?"

not:

> "Can the executive derive the correct terminal action from general metacognitive state understanding?"

Future benchmarks should separate hypothesis identity from meta-action labels and require the controller to derive terminal implications from task semantics.

---

## 9. Prohibited Inputs for Q Features and Authority Certificates

### 9.1 Absolute Prohibitions

Q features and authority certificates MUST NOT use:

1. `verify_result` (oracle verification outcome)
2. `correct_hypothesis_id` (which hypothesis is correct)
3. `expected_terminal` (what the benchmark expects)
4. `oracle_resolution_path` (the solution path)
5. `retrieve_exposes` or `search_exposes` (hidden transition information)
6. Hidden evidence content (propositions, relations of non-retrieved items)
7. Terminal correctness (whether the task was solved)
8. Future actions (what action will be taken next)
9. Downstream utility (realized return from the trajectory)
10. Future outcomes (what will happen after the current step)
11. Expected future verification results

### 9.2 The Representation Rule

```
Feature(s_t) = f(information observable at t)
```

Where "observable at t" means: information available to the controller at decision time t, including:
- All visible evidence with current verification states
- All hypothesis definitions
- All resource counts
- Action history

And explicitly excluding all items in §9.1.

---

## 10. Executor Success Criteria

### 10.1 Current Defect

The current executor (`executor.py:210-241`) defines ANSWER success as:
1. `expected_terminal is ANSWER`, AND
2. The correct hypothesis has at least one SUFFICIENT, CURRENT supporting evidence item, AND
3. No SUFFICIENT, CURRENT contradicting evidence for the correct hypothesis.

This does NOT check whether the correct hypothesis is uniquely supported. A state with competing verified support (two hypotheses both SUFFICIENT) succeeds immediately under ANSWER, contradicting the controller's classification of the state as unresolved.

### 10.2 Required Fix

The executor's ANSWER success criterion MUST be reconciled with the canonical topology:

```
ANSWER_SUCCESS(runtime):
    topology = derive_hypothesis_topology(runtime)
    task = runtime.task
    if task.expected_terminal is not ANSWER:
        return False
    if topology.unique_supported_hypothesis is None:
        return False
    return topology.unique_supported_hypothesis == task.correct_hypothesis_id
```

This requires:
- Exactly one SUPPORTED hypothesis (canonical topology)
- That hypothesis is the correct one (evaluator-side check)

The first condition is purely epistemic (observable). The second condition uses evaluator-only information to score correctness, but the epistemic readiness check is identical to what the controller sees.

### 10.3 DEFER Success

DEFER success currently checks only `expected_terminal is DEFER`. This is acceptable under the current benchmark design where DEFER-correct tasks are constructed to be genuinely DEFER_READY.

For stricter semantic alignment, DEFER success could also check:
```
DEFER_SUCCESS(runtime):
    topology = derive_hypothesis_topology(runtime)
    task = runtime.task
    if task.expected_terminal is not DEFER:
        return False
    # DEFER is correct when the state is not ANSWER_READY
    # and no continuation can resolve it
    return topology.unique_supported_hypothesis is None
```

However, this is optional for I3.30R. The primary fix is ANSWER success (§10.2).

---

## 11. Application to Benchmark Strata

### 11.1 Stratum Truth Labels

Each benchmark stratum must have its truth label validated by causal oracle audit before the benchmark is frozen.

| Stratum | Intended Truth | Required Topology | Required Correct Action |
|---|---|---|---|
| D1 | Safe DEFER (resource exhausted) | No SUPPORTED hypothesis, no admissible continuation | DEFER |
| D2 | Safe DEFER (verified elimination) | No SUPPORTED hypothesis, all evidence verified, no hidden evidence | DEFER |
| D3 | CONTINUE (unresolved contradiction) | Not ANSWER_READY, at least one continuation admissible | non-terminal continuation |
| D4 | ANSWER (uniquely resolved) | Exactly one SUPPORTED hypothesis with answer_action=ANSWER | ANSWER |
| D5 | CONTINUE (verified ambiguity) | >1 hypothesis with SUFFICIENT support, at least one continuation admissible | non-terminal continuation |

### 11.2 Causal Validation

Before freezing any benchmark, force every legal action at the initial state and verify:

- The intended correct action has the highest utility
- Terminal actions that should fail do fail
- Continuation actions that should succeed do transition the state

Any state that violates its stratum truth must be removed or repaired before freeze.

### 11.3 D3 vs D5

D3 and D5 are both CONTINUE-correct but differ in their epistemic structure:
- D3: Unresolved because evidence is unverified or contradictory (untested or competing claims)
- D5: Unresolved because evidence is verified but competing (multiple SUFFICIENT supports)

D5 is the harder abstention test because the evidence looks strong (verified) but is ambiguous (competing). The authority system must recognize that verified competing support is NOT answer-ready.

---

## 12. Mixed Evidence Treatment

### 12.1 Single Hypothesis with Mixed Evidence

If a hypothesis H has both SUFFICIENT support and SUFFICIENT contradiction:
- H is classified as CONTRADICTED (§4.2 priority rule)
- The topology records both `verified_support_by_hypothesis[H]` and `verified_contradiction_by_hypothesis[H]`
- H is NOT viable
- `n_hyp_with_mixed_verified` is incremented

### 12.2 Multiple Hypotheses with Verified Support

If two or more hypotheses each have SUFFICIENT support:
- Each is classified as SUPPORTED (unless any also has SUFFICIENT contradiction)
- `n_hyp_with_verified_support >= 2`
- `has_verified_unresolved_competition = True`
- `unique_supported_hypothesis = None` (not unique)
- The state is NOT ANSWER_READY

This is the D5 pattern. The state is unresolved because verification has confirmed support for multiple competing hypotheses.

### 12.3 What Makes D5 Resolvable

For D5 to be a valid CONTINUE-correct stratum, there must exist at least one admissible action that could resolve the competition:

1. An unverified evidence item that contradicts one of the competing hypotheses (VERIFY could eliminate one)
2. Hidden evidence that, when retrieved and verified, discriminates between them (RETRIEVE + VERIFY)
3. Search results that provide discriminating evidence (SEARCH_MORE + VERIFY)

If none of these exist, the state is DEFER_READY, not CONTINUE_REQUIRED, and the stratum is mislabeled.

---

## 13. Conformance Requirements

### 13.1 Components That Must Conform

| Component | Current File | Conformance Required |
|---|---|---|
| MDSG viability classifier | `scripts/run_i3_7e_compact_governor.py:_classify_from_snapshot` | Already conforms (§3.2 semantics) |
| V3 feature extractor | `scripts/run_i3_30_v3_coverage.py:compute_v3_features` | MUST be fixed (currently treats FALSIFIED as verified) |
| Authority certificates | `daph/authority/policy_v3.py` | MUST be fixed (legacy clause accepts competing support) |
| Executor ANSWER success | `hrm_adaptive_memory/executive/evidence_benchmark/executor.py:_check_answer_success` | MUST be fixed (no uniqueness check) |
| Benchmark truth labels | Various generators | MUST be causally validated |
| Offline safety analysis | `scripts/run_i3_30_train_v3r.py` | MUST use held-out split and exact authority logic |

### 13.2 The Canonical Topology Function

All components must derive epistemic state from the same function:

```
derive_hypothesis_topology(snapshot) -> HypothesisTopology
```

This function:
- Takes only observable inputs (§8.1)
- Produces the canonical topology (§5.2)
- Implements the normative rules (§3.2, §4, §5.3)
- Is the single source of truth for hypothesis states

No component may independently re-derive hypothesis states from raw evidence. All must consume the topology.

### 13.3 Test Requirements

Before any component is wired to the canonical topology, the topology function itself must pass semantic truth-table tests:

1. Every entry in the §3.3 truth table must produce the correct result.
2. Multi-hypothesis cases (§12.1, §12.2) must be tested.
3. Temporal status (STALE) must be tested.
4. Observability boundary must be tested (no evaluator-only inputs).

Only after these tests pass may downstream components be modified.

---

## 14. What This Specification Does NOT Change

- The frozen authority threshold (5.0)
- The frozen near-optimal epsilon (3.0)
- The GBT learner hyperparameters (n_estimators=200, max_depth=4, random_state=42)
- The pinned Qwen model (Qwen2.5-7B-Instruct-Q4_K_M.gguf)
- The utility configuration (configs/v2b_i3_1_utility_v1.json)
- The I2 interface
- The progress rule
- The action vocabulary (ANSWER, DEFER, VERIFY, RETRIEVE, SEARCH_MORE, REASON_MORE, STOP)
- The confirmed champion (DAPH_ADAPTIVE_AUTHORITY_EXECUTIVE_V1)

---

## 15. Summary of Normative Rules

1. **FALSIFIED means the proposition failed, not that the hypothesis is falsified.**
2. **SUFFICIENT + supports(H) → verified support for H.**
3. **SUFFICIENT + contradicts(H) → verified contradiction against H.**
4. **FALSIFIED + supports(H) → no effect on H.**
5. **FALSIFIED + contradicts(H) → no effect on H.**
6. **A hypothesis is viable only if it has verified support and no verified contradiction.**
7. **ANSWER_READY requires exactly one viable hypothesis.**
8. **Competing verified support is NOT answer-ready.**
9. **Resource exhaustion affects action admissibility, not epistemic state.**
10. **All components must consume one canonical topology derivation.**
11. **No evaluator-only information may enter Q features or authority certificates.**
12. **`answer_action` is observable but creates a benchmark shortcut that limits the scientific claim.**
