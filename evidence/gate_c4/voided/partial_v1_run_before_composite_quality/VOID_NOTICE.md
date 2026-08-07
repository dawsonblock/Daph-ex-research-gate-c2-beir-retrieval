# Void Notice — Partial v1 Development Run (Before Composite Quality Fix)

## Date
2025-08-07

## Reason for Voiding
The receipts in this run were generated before the `_compute_quality()` fix
was applied to `scripts/run_gate_c4.py`. The stored `quality` field is
**binary correctness** (1.0/0.0), NOT the protocol-defined composite quality
(1.0/0.5/0.25/0.0).

This was confirmed by inspection: 20/20 sampled receipts have
`quality == 1.0 if correct else 0.0`.

## Run State at Void
- C4_0: 120/120 receipts (binary quality)
- C4_1: 120/120 receipts (binary quality)
- C4_2: 120/120 receipts (binary quality)
- C4_3: 120/120 receipts (binary quality)
- C4_4: 120/120 receipts (binary quality)
- C4_5: 40/120 receipts (binary quality, incomplete)
- C4_6: 0/120 receipts (missing)
- manifest.json: missing
- analysis.json: missing
- RESULTS.sha256: missing

## Action
These receipts are voided. They must NOT be used for any C4 v2 metric
calculation, OGC/SGC computation, or promotion gate decision.

The full development run must be re-executed from scratch on Colab T4 GPU
using `scripts/colab_c4_requalify.py` with:
- `C4_PROTOCOL=v2`
- `_compute_quality()` now correctly delegates to `c4.metrics.compute_quality()`
- All manifest provenance fields populated
- RESULTS.sha256 written last (after analysis)
