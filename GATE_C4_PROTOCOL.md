# Gate C4 — Integrated Non-Oracle Memory Pipeline

## Protocol

**Status:** PROTOCOL_FROZEN_BEFORE_MEASUREMENT  
**Version:** v1_frozen  
**Config:** `configs/gate_c4_protocol.json`

## Primary question

Do the independently qualified query, retrieval, identity-resolution, and structural-selection mechanisms compose into a materially better end-to-end HRM memory pipeline without oracle information at runtime?

## Architecture under test

```
QUESTION
   ↓
INFORMATION STATE
   ├── original subject
   ├── target relation
   └── retrieved/resolved canonical entities
   ↓
SUBJECT-PRESERVING FOLLOW-UP QUERY
   ↓
BM25 + BGE CANDIDATE RETRIEVAL
   ↓
I3 IDENTITY-RECORD RESOLUTION
   ↓
CANONICALIZED INFORMATION STATE
   ↓
S2c STRUCTURE + RELATION SELECTION
   ↓
BOUNDED EVIDENCE PACKET
   ↓
HRM
   ↓
VERIFIED ANSWER
```

No oracle metadata may enter runtime.

## Arm ladder

| Arm | Description | One mechanism change |
|---|---|---|
| C4-0 | current historical R1 pipeline baseline | baseline |
| C4-1 | C4-0 + subject-preserving information state + frozen follow-up query | query formulation |
| C4-2 | C4-1 + BM25/BGE multi-signal candidate generation | retrieval |
| C4-3 | C4-2 + I3 identity-record resolution + canonicalized state | identity resolution |
| C4-4 | C4-3 + S2c structure+relation selection | selector |
| C4-4b | diagnostic: if resolved → S2c; else → s_rel_only | diagnostic only |
| C4-5 | same real pool + oracle selector | oracle selection ceiling |
| C4-6 | oracle required evidence | R5-style evidence ceiling |

### Arm parity constraints

- C4-2 and C4-3: same query, same candidates, same scores; only identity resolution differs
- C4-3 and C4-4: same query, same candidates, same identity state; only selector differs
- C4-4 and C4-5: same real candidate pool; only real vs oracle selection differs

### Deterministic fallback (C4-4)

```
if identity_status in {EXACT, RESOLVED}:
    selector = S2c
else:
    selector = S0
```

No adaptive router. No entity-regime labels at runtime.

## Promotion criteria (frozen before qualification)

1. C4-4 quality > C4-0 by at least +0.15 absolute on development
2. No material canonical/abbreviation regression > 0.05
3. Alias and description both improve over C4-0
4. FalseResolutionRate ≤ 0.02
5. C4-4 materially reduces the C4-5 oracle-selector gap
6. All runtime payloads pass oracle-leak validation
7. Candidate/evidence budgets remain fixed
8. Grouped bootstrap lower bound for primary quality delta > 0
9. No post-hoc mechanism/config changes after freeze

## Oracle-gap metrics

- **OGC_C4** = (Q(C4-4) - Q(C4-0)) / (Q(C4-6) - Q(C4-0))
- **SGC_C4** = (Q(C4-4) - Q(C4-3)) / (Q(C4-5) - Q(C4-3)) when denominator > 0
- **IGC_C4** = Q(C4-3) - Q(C4-2)

## Execution order

1. Build InformationState contract with tests
2. Implement C4-0 through C4-6 (one mechanism per arm)
3. Add arm-parity assertions as automated tests
4. Add full task-level receipts with runtime/evaluator separation
5. Run development only
6. Analyze by regime and mechanism
7. Freeze C4 configuration and promotion criteria
8. Run qualification once
9. Run OOD once
10. Write final C4 report
11. Decide whether Gate D is authorized

## Decision outcomes

| Outcome | Condition | Action |
|---|---|---|
| A | C4-4 > 0.50+ with robust OOD gains | Proceed to Gate D |
| B | Retrieval/identity improve, selector gain disappears | Revisit structural selection |
| C | ID works, surface still weak | Identity/retrieval remains bottleneck |
| D | Oracle C4-5 >> C4-4 | Selection remains opportunity |

Gate D is only authorized if C4 proves RETRIEVE is a competent real action.
