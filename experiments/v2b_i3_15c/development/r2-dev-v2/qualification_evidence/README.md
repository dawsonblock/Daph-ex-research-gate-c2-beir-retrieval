# R2-DEV-V2 Qualification Evidence Bundle

**FROZEN** — do not alter during R2-DEV-V2.

## Contents

- `bundle.json` — complete qualification evidence with all SHAs, identity, test results
- `qualification_report.json` — raw output from the qualification matrix run (to be downloaded from Colab)

## Identity Summary

| Field | Value |
|-------|-------|
| Model | Qwen2.5-7B-Instruct Q4_K_M |
| GGUF SHA-256 | `65b8fcd92af6b4fefa935c625d1ac27ea29dcb6ee14589c55a8f115ceaaa1423` |
| GGUF size | 4,683,074,240 bytes |
| Runtime | llama-cpp-python 0.3.35 |
| GPU | Tesla T4 |
| Schema builder SHA | `c20cd3a5adf976ddce2296ded11e21d1b2d9c972cd94c205404e5d6b410a3b0e` |
| Identity SHA-256 | `08dd528f6fc9e67c574ec766ea15ab7be80bdaa7a615625244122db533c2772d` |
| Source commit | `e455f6d3aca70878e103f9ceb1e66d08fcbe560e` |
| Frozen R13 schema SHA | `2208076c081272b5354fd38b02f6943f79f0e8a695638bc25625a52fb49bacca` |

## Qualification Result

All 14 tests PASS with `--require-live-q6`:

```
Q1:  PASS — C0 schema SHA matches frozen R13
Q1b: PASS — Three-way schema tie-out
Q2:  PASS — VERIFY absent from D/DE at T2
Q3a: PASS — LocalLlamaBackend schema identity
Q3b: PASS — R2DirectLlamaBackend schema identity
Q4:  PASS — Strict decoder rejects markdown-fenced JSON
Q5:  PASS — Strict decoder accepts pure JSON
Q5b: PASS — Strict decoder rejects unknown action
Q6:  PASS — Adversarial: model prompted to VERIFY, grammar forced RETRIEVE
Q7:  PASS — Real receipt has all 10 required fields
Q8:  PASS — No placeholders in pinned identity
Q9:  PASS — GGUF SHA matches at startup
Q10: PASS — Schema-builder SHA matches at startup
Q11: PASS — Runtime version matches (llama-cpp-python 0.3.35)
```

## Classification

```
R2_DECODER_MECHANISM_QUAL_001 + R2_POLICY_BACKEND_V2
```

## Lineage Note

Qwen2.5-7B is a new backend development line, NOT a continuation of the R13 Gemma lineage. Within-model contrasts (D vs C0, E vs C0) are valid. Absolute utilities must NOT be compared directly to R13 Gemma.
