# I3.30R3: Authority Isolation Study — Results

**This report is auto-generated from `authority_analysis.json`.**
**Do not edit manually — regenerate with `scripts/generate_i3_30r3_results_md.py`.**

## Primary Comparison: V3-AUTH vs V3-SHADOW

| Metric | Value |
|--------|-------|
| ATE_authority | 15.5681 |
| 95% CI | [8.8984, 23.1749] |
| n | 185 |
| Rescues | 18 |
| Breaks | 0 |
| Both success | 109 |
| Both fail | 58 |

## Secondary Comparison: V3-SHADOW vs V1

| Metric | Value |
|--------|-------|
| ΔU(SHADOW - V1) | 2.4580 |
| 95% CI | [-6.3216, 11.5315] |
| n | 185 |
| Rescues | 15 |
| Breaks | 11 |

## Authority Event Classification

| Classification | Count |
|---------------|-------|
| beneficial_nonrescue | 42 |
| neutral | 52 |
| rescue | 30 |

## Authority Rates

| Rate | Value |
|------|-------|
| Certificate coverage | 0.2951 |
| Force rate | 0.2951 |
| Effective intervention rate | 0.1246 |

## Stratum Breakdown

| Stratum | V1 | SHADOW | AUTH |
|---------|-----|--------|------|
| D1 | 10/35 (28.57%) | 8/35 (22.86%) | 8/35 (22.86%) |
| D2 | 19/35 (54.29%) | 27/35 (77.14%) | 27/35 (77.14%) |
| D3 | 6/45 (13.33%) | 8/45 (17.78%) | 22/45 (48.89%) |
| D4 | 35/35 (100.00%) | 35/35 (100.00%) | 35/35 (100.00%) |
| D5 | 35/35 (100.00%) | 31/35 (88.57%) | 35/35 (100.00%) |

## Aggregate Success Rates

| Arm | Success | Total | Rate | Mean Utility |
|-----|---------|-------|------|--------------|
| v1 | 105 | 185 | 56.76% | 13.10 |
| v3_shadow | 109 | 185 | 58.92% | 15.56 |
| v3_hard | 127 | 185 | 68.65% | 31.13 |

## Gate Evaluation

| Gate | Name | Result | Value |
|------|------|--------|-------|
| G1 | treatment_purity | PASS | {'purity_mismatches': 0, 'prompt_mismatches': 0, 'schema_mismatches': 0, 'state_mismatches': 0, 'paired_events': 90, 'unpaired_events': 34, 'integration_tests': '6/6 pass (test_i3_30r3_runner_boundary.py)'} |
| G10 | reliability | PASS | 0 |
| G11 | artifact_identity | FAIL | 2 |
| G12 | event_receipts | PASS | 1.0000 |
| G2 | authority_breaks | PASS | 0 |
| G3 | false_answer_authority | PASS | 0 |
| G4 | false_defer_authority | PASS | 0 |
| G5 | authority_effect | PASS | 15.5681 |
| G6 | rescues_gt_breaks | PASS | {'rescues': 18, 'breaks': 0} |
| G7 | answer_coverage | PASS | 38 |
| G8 | defer_coverage | FAIL | 0 |
| G9 | semantic_consistency | PASS | 0 |

**10 passed, 2 failed, 0 pending.**
