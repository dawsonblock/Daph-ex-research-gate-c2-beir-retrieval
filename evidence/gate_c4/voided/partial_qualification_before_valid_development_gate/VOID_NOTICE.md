# VOID: Partial Qualification Run

**Status:** `VOID_PARTIAL_QUALIFICATION_STARTED_BEFORE_VALID_DEVELOPMENT_GATE`

**Date voided:** 2026-08-07

## Reason

This partial qualification run was started before:

1. The C4 development evaluator bug (control token stripping) was fixed.
2. Development promotion criteria were properly evaluated with the corrected
   verifier.
3. C4 passed the frozen development gate.

This violates the frozen C4 protocol, which requires development gate passage
before qualification begins.

## Contents

- `C4_0.jsonl`: 290 rows (partial — only C4-0 arm, incomplete task set)
- No qualification manifest was generated.

## Disposition

- **Preserved** for provenance (not deleted).
- **Must not be used** for any scientific conclusion.
- Qualification must be restarted from scratch after:
  - Verifier correction (done — evaluator_v2)
  - Canonical EXACT identity fix
  - Iterative subject+bridge+relation retrieval fix
  - Protocol erratum for alias/description criterion
  - Successful development rerun and gate passage
