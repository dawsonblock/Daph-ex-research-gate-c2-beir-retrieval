# I3.30R3 ATTEMPT 1 — INVALID: TREATMENT CONTAMINATION

**Status: REJECTED FOR PROMOTION**
**Valid for: Diagnosis of experimental control defect**
**Date: 2026-08-29**

## Defect

The V3-SHADOW arm was not actually untreated. Both V3 arms narrowed
`schema_actions` to the singleton forced action before LLM generation:

```python
if v3_decision.would_force and v3_decision.forced_action:
    candidate = frozenset({v3_decision.forced_action}) & allowed_decision.allowed
    if candidate:
        schema_actions = candidate  # <-- contaminates both arms
```

This constrained the GBNF decoder to only emit the certificate's action
for BOTH V3-SHADOW and V3-HARD. The LLM was never free to disagree.

## Consequence

ATE_authority = 0.0000 does NOT mean "hard authority is causally
redundant because the LLM independently chooses the same action."

It means: "After authority had already restricted both arms to the same
singleton action during constrained decoding, an additional post-generation
assignment of that same action had zero effect."

## Evidence

- 90/90 SHADOW events: LLM proposal = forced action (100% agreement)
- 90/90 SHADOW events: singleton schema_actions
- 0/90 action_changed events
- This 100% agreement is the signature of the bug, not treatment purity

## What is preserved

All raw JSONL trajectory files, authority events, analysis, and manifest
are preserved in this directory for diagnostic reference.

## What remains valid

- D5 state-truth audit (independent of authority isolation)
- D1 Q-error audit (no certificate fires on D1)
- V1 D2 false-ANSWER identification (V1 defect is real, though the
  claim that V3-SHADOW rescues "through Q guidance alone" needs
  qualification for the 3/8 cases where V3 DEFER certificate fired
  and also constrained the decoder)
- V3 representation/Q/advisory improvement over V1 (directionally
  correct, though the exact magnitude is contaminated by the decoder
  restriction in V3-SHADOW)

## What is invalid

- ATE_authority = 0.0000 (primary comparison)
- All 90 NEUTRAL authority event classifications
- Gates G6, G7, G8 (consequences of the contaminated zero)
- The claim that "hard authority is causally redundant"

## Corrected counts from raw trajectories

| Stratum | V1 | V3-SHADOW | V3-HARD |
|---------|-----|-----------|---------|
| D1 | 10/35 | 8/35 | 8/35 |
| D2 | 19/35 | 27/35 | 27/35 |
| D3 | 6/45 | 22/45 | 22/45 |
| D4 | 35/35 | 35/35 | 35/35 |
| D5 | 35/35 | 35/35 | 35/35 |
| Overall | 105/185 | 127/185 | 127/185 |
| Success | 56.76% | 68.65% | 68.65% |
| Mean utility | 13.10 | 31.13 | 31.13 |

The markdown report (I3_30R3_RESULTS.md) incorrectly stated 97/185 and
121/185. The authority_analysis.json has the correct counts.
