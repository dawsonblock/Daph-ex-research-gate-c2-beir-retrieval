# DAPH V2B-I3.4.1 Development Run Report

**Schema:** `DAPH_V2B_I3_4_DEV_REPORT_V1`
**Generated:** 2026-08-15T01:59:08.527337+00:00
**Experiment ID:** `v2b_i3_4_experiment_v1`

## 1. Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Model | `deepseek-v4-flash` |
| Thinking mode | disabled |
| Response format | `json_object` |
| Temperature | 0.0 |
| Max tokens | 2048 |
| Split | development |
| Task count | 300 |
| Max steps per task | 24 |

## 2. Identity and Provenance

| Artifact | SHA-256 (first 16) |
|----------|-------------------|
| Experiment identity | `31a57b086184007e...` |
| Results file | `91c95f44954e0152...` |
| Receipts file | `0841f9485450a533...` |
| Report file | `a8edc329958702a6...` |

## 3. Headline Results

| Metric | Blind | Aware | Difference |
|--------|------:|------:|-----------:|
| Success rate | 42.0% | 48.0% | +6.0pp |
| Mean utility | -20.0402 | -6.6880 | +13.3521 |
| Total API calls | 743 | 597 | -146 |
| Observable optimal V_O | 50.3184 | 65.3837 | +15.0653 |

## 4. Statistical Analysis (ΔDG = DG_blind - DG_aware)

| Statistic | Value |
|-----------|-------|
| N paired tasks | 300 |
| Mean ΔDG | -1.713150 |
| Bootstrap CI lower | -8.600650 |
| Bootstrap CI upper | 5.609050 |
| Bootstrap iterations | 10,000 |
| CI significant positive | No (lower bound < 0) |

**Interpretation:** The mean ΔDG is -1.7132, with a 95% bootstrap CI of [-8.6006, 5.6090]. The CI includes zero, so the ΔDG > 0 hypothesis is **not supported** on the development split at this sample size.

However, the aware condition shows:
- **+6.0pp higher success rate** (48.0% vs 42.0%)
- **+13.4 higher mean utility** (-6.69 vs -20.04)
- **20% fewer API calls** (597 vs 743) — more efficient executive behavior
- **24 tasks where aware succeeded but blind failed** (vs 6 reverse)

The negative ΔDG despite better absolute performance is explained by the observable oracle also being higher for the aware condition (V_O_aware = 65.38 vs V_O_blind = 50.32). The decision gap relative to the aware oracle is larger even though the controller makes better decisions in absolute terms.

## 5. Disagreement Analysis

| Category | Count |
|----------|------:|
| Both succeed | 30 |
| Both fail | 240 |
| Blind succeeds, aware fails | 6 |
| Aware succeeds, blind fails | 24 |

### Tasks where aware succeeded but blind failed (24 tasks):

| Task ID | Blind terminal | Aware terminal | Blind utility | Aware utility | ΔDG |
|---------|---------------|---------------|--------------:|--------------:|----:|
| i3_3_development_0002 | TASK_FAILURE | TASK_SUCCESS | -105.44 | 99.89 | 190.2647 |
| i3_3_development_0020 | TASK_FAILURE | TASK_SUCCESS | -105.44 | 99.89 | 190.2647 |
| i3_3_development_0070 | TASK_FAILURE | TASK_SUCCESS | -105.44 | 99.89 | 190.2647 |
| i3_3_development_0072 | TASK_FAILURE | TASK_SUCCESS | -105.44 | 96.71 | 187.0847 |
| i3_3_development_0090 | TASK_FAILURE | TASK_SUCCESS | -105.44 | 99.89 | 190.2647 |
| i3_3_development_0092 | TASK_FAILURE | TASK_SUCCESS | -105.44 | 99.89 | 190.2647 |
| i3_3_development_0112 | TASK_FAILURE | TASK_SUCCESS | -105.44 | 99.89 | 190.2647 |
| i3_3_development_0132 | TASK_FAILURE | TASK_SUCCESS | -105.44 | 99.89 | 190.2647 |
| i3_3_development_0154 | TASK_FAILURE | TASK_SUCCESS | -105.44 | 99.89 | 190.2647 |
| i3_3_development_0162 | TASK_FAILURE | TASK_SUCCESS | -105.44 | 97.74 | 188.1147 |
| i3_3_development_0184 | TASK_FAILURE | TASK_SUCCESS | -105.44 | 99.89 | 190.2647 |
| i3_3_development_0202 | TASK_FAILURE | TASK_SUCCESS | -105.44 | 99.89 | 190.2647 |
| i3_3_development_0210 | TASK_FAILURE | TASK_SUCCESS | -105.44 | 99.89 | 190.2647 |
| i3_3_development_0216 | TASK_FAILURE | TASK_SUCCESS | -105.44 | 99.89 | 190.2647 |
| i3_3_development_0222 | TASK_FAILURE | TASK_SUCCESS | -105.44 | 97.74 | 188.1147 |
| i3_3_development_0230 | TASK_FAILURE | TASK_SUCCESS | -105.44 | 99.89 | 190.2647 |
| i3_3_development_0231 | TASK_FAILURE | TASK_SUCCESS | -105.44 | 96.71 | 187.0847 |
| i3_3_development_0232 | TASK_FAILURE | TASK_SUCCESS | -105.44 | 99.89 | 190.2647 |
| i3_3_development_0252 | TASK_FAILURE | TASK_SUCCESS | -105.44 | 99.89 | 190.2647 |
| i3_3_development_0280 | TASK_FAILURE | TASK_SUCCESS | -105.44 | 99.89 | 190.2647 |

*...and 4 more*

### Tasks where blind succeeded but aware failed (6 tasks):

| Task ID | Blind terminal | Aware terminal | Blind utility | Aware utility | ΔDG |
|---------|---------------|---------------|--------------:|--------------:|----:|
| i3_3_development_0047 | TASK_SUCCESS | TASK_FAILURE | 94.56 | -100.11 | -209.7353 |
| i3_3_development_0093 | TASK_SUCCESS | TASK_FAILURE | 94.56 | -100.11 | -209.7353 |
| i3_3_development_0139 | TASK_SUCCESS | TASK_FAILURE | 94.56 | -100.11 | -209.7353 |
| i3_3_development_0185 | TASK_SUCCESS | TASK_FAILURE | 94.56 | -100.11 | -209.7353 |
| i3_3_development_0226 | TASK_SUCCESS | TASK_FAILURE | 94.56 | -100.11 | -209.7353 |
| i3_3_development_0268 | TASK_SUCCESS | TASK_FAILURE | 94.56 | -100.11 | -209.7353 |

## 6. Action Distribution

| Action | Blind | Aware |
|--------|------:|------:|
| ANSWER | 261 | 267 |
| DEFER | 8 | 2 |
| REASON_MORE | 0 | 24 |
| RETRIEVE | 223 | 125 |
| SEARCH_MORE | 0 | 17 |
| STOP | 31 | 31 |
| VERIFY | 220 | 131 |

**Key observation:** The aware condition uses `SEARCH_MORE` and `REASON_MORE` actions that the blind condition never uses. This is consistent with the hypothesis that cognitive state enables more sophisticated executive behavior — the aware controller can detect conflicts and composition gaps that the blind controller cannot see.

## 7. Efficiency Analysis

| Metric | Blind | Aware | Ratio |
|--------|------:|------:|------:|
| Total API calls | 743 | 597 | 0.80 |
| Calls per task | 2.48 | 1.99 | — |
| Successes per call | 0.1696 | 0.2412 | — |

The aware condition achieves a higher success rate with **20% fewer API calls**, indicating more efficient executive decisions when cognitive state is available.

## 8. Per-Task Results (All 300 Tasks)

| Task ID | Blind | Aware | Blind U | Aware U | Blind calls | Aware calls | ΔDG | ΔIG | ΔTR |
|---------|-------|-------|--------:|--------:|------------:|------------:|----:|----:|----:|
| i3_3_development_0000 | ✗ | ✗ | -100.11 | -100.11 | 1 | 1 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0001 | ✓ | ✓ | 94.56 | 97.74 | 3 | 2 | -11.89 | 15.07 | 3.18 |
| i3_3_development_0002 | ✗ | ✓ | -105.44 | 99.89 | 3 | 1 | 190.26 | 15.07 | 205.33 |
| i3_3_development_0003 | ✗ | ✗ | -105.44 | -105.44 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0004 | ✗ | ✗ | -100.11 | -100.11 | 1 | 1 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0005 | ✓ | ✓ | 94.56 | 94.56 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0006 | ✓ | ✓ | 94.56 | 99.89 | 3 | 1 | -9.74 | 15.07 | 5.33 |
| i3_3_development_0007 | ✓ | ✓ | 94.56 | 96.71 | 3 | 2 | -12.92 | 15.07 | 2.15 |
| i3_3_development_0008 | ✓ | ✓ | 94.56 | 94.56 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0009 | ✗ | ✗ | -105.44 | -100.11 | 3 | 1 | -9.74 | 15.07 | 5.33 |
| i3_3_development_0010 | ✗ | ✗ | -100.11 | -100.11 | 1 | 1 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0011 | ✓ | ✓ | 94.56 | 94.56 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0012 | ✗ | ✗ | -105.44 | -100.11 | 3 | 1 | -9.74 | 15.07 | 5.33 |
| i3_3_development_0013 | ✓ | ✓ | 99.89 | 99.89 | 1 | 1 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0014 | ✓ | ✓ | 94.56 | 99.89 | 3 | 1 | -9.74 | 15.07 | 5.33 |
| i3_3_development_0015 | ✓ | ✓ | 94.56 | 94.56 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0016 | ✗ | ✗ | -100.11 | -100.11 | 1 | 1 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0017 | ✗ | ✗ | -105.44 | -105.44 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0018 | ✗ | ✗ | -105.44 | -100.11 | 3 | 1 | -9.74 | 15.07 | 5.33 |
| i3_3_development_0019 | ✓ | ✓ | 99.89 | 99.89 | 1 | 1 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0020 | ✗ | ✓ | -105.44 | 99.89 | 3 | 1 | 190.26 | 15.07 | 205.33 |
| i3_3_development_0021 | ✗ | ✗ | -105.44 | -107.60 | 3 | 4 | -17.23 | 15.07 | -2.16 |
| i3_3_development_0022 | ✗ | ✗ | -105.44 | -107.60 | 3 | 4 | -17.23 | 15.07 | -2.16 |
| i3_3_development_0023 | ✗ | ✗ | -105.44 | -100.11 | 3 | 1 | -9.74 | 15.07 | 5.33 |
| i3_3_development_0024 | ✗ | ✗ | -105.44 | -105.44 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0025 | ✓ | ✓ | 99.89 | 99.89 | 1 | 1 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0026 | ✗ | ✗ | -100.11 | -100.11 | 1 | 1 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0027 | ✓ | ✓ | 94.56 | 94.56 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0028 | ✓ | ✓ | 94.56 | 99.89 | 3 | 1 | -9.74 | 15.07 | 5.33 |
| i3_3_development_0029 | ✓ | ✓ | 94.56 | 97.74 | 3 | 2 | -11.89 | 15.07 | 3.18 |
| i3_3_development_0030 | ✗ | ✗ | -100.11 | -100.11 | 1 | 1 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0031 | ✗ | ✗ | -105.44 | -108.62 | 3 | 4 | -18.25 | 15.07 | -3.18 |
| i3_3_development_0032 | ✗ | ✗ | -105.44 | -100.11 | 3 | 1 | -9.74 | 15.07 | 5.33 |
| i3_3_development_0033 | ✓ | ✓ | 99.89 | 99.89 | 1 | 1 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0034 | ✓ | ✓ | 94.56 | 94.56 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0035 | ✓ | ✓ | 94.56 | 94.56 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0036 | ✗ | ✗ | -100.11 | -100.11 | 1 | 1 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0037 | ✓ | ✓ | 94.56 | 94.56 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0038 | ✗ | ✗ | -105.44 | -105.44 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0039 | ✗ | ✗ | -105.44 | -105.44 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0040 | ✗ | ✗ | -105.44 | -100.11 | 3 | 1 | -9.74 | 15.07 | 5.33 |
| i3_3_development_0041 | ✓ | ✓ | 99.89 | 99.89 | 1 | 1 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0042 | ✗ | ✗ | -100.11 | -100.11 | 1 | 1 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0043 | ✓ | ✓ | 94.56 | 94.56 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0044 | ✓ | ✓ | 94.56 | 99.89 | 3 | 1 | -9.74 | 15.07 | 5.33 |
| i3_3_development_0045 | ✗ | ✗ | -105.44 | -105.44 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0046 | ✓ | ✓ | 94.56 | 99.89 | 3 | 1 | -9.74 | 15.07 | 5.33 |
| i3_3_development_0047 | ✓ | ✗ | 94.56 | -100.11 | 3 | 1 | -209.74 | 15.07 | -194.67 |
| i3_3_development_0048 | ✓ | ✓ | 94.56 | 99.89 | 3 | 1 | -9.74 | 15.07 | 5.33 |
| i3_3_development_0049 | ✗ | ✗ | -105.44 | -105.44 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0050 | ✗ | ✗ | -105.44 | -109.74 | 3 | 5 | -19.37 | 15.07 | -4.30 |
| i3_3_development_0051 | ✗ | ✗ | -105.44 | -100.11 | 3 | 1 | -9.74 | 15.07 | 5.33 |
| i3_3_development_0052 | ✗ | ✗ | -100.11 | -100.11 | 1 | 1 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0053 | ✗ | ✗ | -105.44 | -107.58 | 3 | 4 | -17.21 | 15.07 | -2.14 |
| i3_3_development_0054 | ✗ | ✗ | -105.44 | -100.11 | 3 | 1 | -9.74 | 15.07 | 5.33 |
| i3_3_development_0055 | ✓ | ✓ | 99.89 | 99.89 | 1 | 1 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0056 | ✗ | ✗ | -100.11 | -100.11 | 1 | 1 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0057 | ✓ | ✓ | 94.56 | 94.56 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0058 | ✓ | ✓ | 94.56 | 99.89 | 3 | 1 | -9.74 | 15.07 | 5.33 |
| i3_3_development_0059 | ✓ | ✓ | 94.56 | 94.56 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0060 | ✗ | ✗ | -105.44 | -100.11 | 3 | 1 | -9.74 | 15.07 | 5.33 |
| i3_3_development_0061 | ✓ | ✓ | 99.89 | 99.89 | 1 | 1 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0062 | ✗ | ✗ | -100.11 | -100.11 | 1 | 1 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0063 | ✓ | ✓ | 94.56 | 94.56 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0064 | ✓ | ✓ | 94.56 | 94.56 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0065 | ✗ | ✗ | -105.44 | -100.11 | 3 | 1 | -9.74 | 15.07 | 5.33 |
| i3_3_development_0066 | ✗ | ✗ | -105.44 | -109.74 | 3 | 5 | -19.37 | 15.07 | -4.30 |
| i3_3_development_0067 | ✓ | ✓ | 99.89 | 99.89 | 1 | 1 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0068 | ✗ | ✗ | -100.11 | -100.11 | 1 | 1 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0069 | ✗ | ✗ | -105.44 | -107.58 | 3 | 4 | -17.21 | 15.07 | -2.14 |
| i3_3_development_0070 | ✗ | ✓ | -105.44 | 99.89 | 3 | 1 | 190.26 | 15.07 | 205.33 |
| i3_3_development_0071 | ✗ | ✗ | -105.44 | -107.60 | 3 | 4 | -17.23 | 15.07 | -2.16 |
| i3_3_development_0072 | ✗ | ✓ | -105.44 | 96.71 | 3 | 2 | 187.08 | 15.07 | 202.15 |
| i3_3_development_0073 | ✗ | ✗ | -105.44 | -105.44 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0074 | ✓ | ✓ | 94.56 | 94.56 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0075 | ✗ | ✗ | -105.44 | -105.44 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0076 | ✓ | ✓ | 94.56 | 96.71 | 3 | 2 | -12.92 | 15.07 | 2.15 |
| i3_3_development_0077 | ✗ | ✗ | -105.44 | -105.43 | 3 | 3 | -15.06 | 15.07 | 0.01 |
| i3_3_development_0078 | ✗ | ✗ | -100.11 | -100.11 | 1 | 1 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0079 | ✓ | ✓ | 94.56 | 94.56 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0080 | ✗ | ✗ | -105.44 | -109.74 | 3 | 5 | -19.37 | 15.07 | -4.30 |
| i3_3_development_0081 | ✓ | ✓ | 99.89 | 99.89 | 1 | 1 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0082 | ✗ | ✗ | -100.11 | -100.11 | 1 | 1 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0083 | ✓ | ✓ | 94.56 | 94.56 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0084 | ✓ | ✓ | 94.56 | 99.89 | 3 | 1 | -9.74 | 15.07 | 5.33 |
| i3_3_development_0085 | ✗ | ✗ | -105.44 | -105.44 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0086 | ✓ | ✓ | 94.56 | 99.89 | 3 | 1 | -9.74 | 15.07 | 5.33 |
| i3_3_development_0087 | ✗ | ✗ | -105.44 | -105.44 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0088 | ✗ | ✗ | -100.11 | -100.11 | 1 | 1 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0089 | ✗ | ✗ | -105.44 | -107.58 | 3 | 4 | -17.21 | 15.07 | -2.14 |
| i3_3_development_0090 | ✗ | ✓ | -105.44 | 99.89 | 3 | 1 | 190.26 | 15.07 | 205.33 |
| i3_3_development_0091 | ✗ | ✗ | -105.44 | -107.60 | 3 | 4 | -17.23 | 15.07 | -2.16 |
| i3_3_development_0092 | ✗ | ✓ | -105.44 | 99.89 | 3 | 1 | 190.26 | 15.07 | 205.33 |
| i3_3_development_0093 | ✓ | ✗ | 94.56 | -100.11 | 3 | 1 | -209.74 | 15.07 | -194.67 |
| i3_3_development_0094 | ✗ | ✗ | -100.11 | -100.11 | 1 | 1 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0095 | ✓ | ✓ | 94.56 | 94.56 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0096 | ✗ | ✗ | -105.44 | -100.11 | 3 | 1 | -9.74 | 15.07 | 5.33 |
| i3_3_development_0097 | ✓ | ✓ | 99.89 | 99.89 | 1 | 1 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0098 | ✓ | ✓ | 94.56 | 99.89 | 3 | 1 | -9.74 | 15.07 | 5.33 |
| i3_3_development_0099 | ✓ | ✓ | 94.56 | 94.56 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0100 | ✗ | ✗ | -105.44 | -100.11 | 3 | 1 | -9.74 | 15.07 | 5.33 |
| i3_3_development_0101 | ✗ | ✗ | -105.44 | -107.60 | 3 | 4 | -17.23 | 15.07 | -2.16 |
| i3_3_development_0102 | ✗ | ✗ | -105.44 | -107.58 | 3 | 4 | -17.21 | 15.07 | -2.14 |
| i3_3_development_0103 | ✓ | ✓ | 94.56 | 94.56 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0104 | ✗ | ✗ | -100.11 | -100.11 | 1 | 1 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0105 | ✗ | ✗ | -105.44 | -107.58 | 3 | 4 | -17.21 | 15.07 | -2.14 |
| i3_3_development_0106 | ✓ | ✓ | 94.56 | 94.56 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0107 | ✗ | ✗ | -105.44 | -100.11 | 3 | 1 | -9.74 | 15.07 | 5.33 |
| i3_3_development_0108 | ✗ | ✗ | -100.11 | -100.11 | 1 | 1 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0109 | ✓ | ✓ | 94.56 | 94.56 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0110 | ✗ | ✗ | -105.44 | -100.11 | 3 | 1 | -9.74 | 15.07 | 5.33 |
| i3_3_development_0111 | ✓ | ✓ | 99.89 | 99.89 | 1 | 1 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0112 | ✗ | ✓ | -105.44 | 99.89 | 3 | 1 | 190.26 | 15.07 | 205.33 |
| i3_3_development_0113 | ✓ | ✓ | 94.56 | 97.74 | 3 | 2 | -11.89 | 15.07 | 3.18 |
| i3_3_development_0114 | ✗ | ✗ | -100.11 | -100.11 | 1 | 1 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0115 | ✓ | ✓ | 94.56 | 94.56 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0116 | ✓ | ✓ | 94.56 | 99.89 | 3 | 1 | -9.74 | 15.07 | 5.33 |
| i3_3_development_0117 | ✓ | ✓ | 94.56 | 96.71 | 3 | 2 | -12.92 | 15.07 | 2.15 |
| i3_3_development_0118 | ✓ | ✓ | 94.56 | 99.89 | 3 | 1 | -9.74 | 15.07 | 5.33 |
| i3_3_development_0119 | ✗ | ✗ | -105.44 | -105.43 | 3 | 3 | -15.06 | 15.07 | 0.01 |
| i3_3_development_0120 | ✗ | ✗ | -100.11 | -100.11 | 1 | 1 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0121 | ✗ | ✗ | -105.44 | -107.58 | 3 | 4 | -17.21 | 15.07 | -2.14 |
| i3_3_development_0122 | ✗ | ✗ | -105.44 | -107.60 | 3 | 4 | -17.23 | 15.07 | -2.16 |
| i3_3_development_0123 | ✓ | ✓ | 99.89 | 99.89 | 1 | 1 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0124 | ✓ | ✓ | 94.56 | 99.89 | 3 | 1 | -9.74 | 15.07 | 5.33 |
| i3_3_development_0125 | ✓ | ✓ | 94.56 | 99.89 | 3 | 1 | -9.74 | 15.07 | 5.33 |
| i3_3_development_0126 | ✓ | ✓ | 94.56 | 99.89 | 3 | 1 | -9.74 | 15.07 | 5.33 |
| i3_3_development_0127 | ✗ | ✗ | -105.44 | -105.44 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0128 | ✓ | ✓ | 94.56 | 99.89 | 3 | 1 | -9.74 | 15.07 | 5.33 |
| i3_3_development_0129 | ✗ | ✗ | -105.44 | -105.44 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0130 | ✗ | ✗ | -100.11 | -100.11 | 1 | 1 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0131 | ✓ | ✓ | 94.56 | 94.56 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0132 | ✗ | ✓ | -105.44 | 99.89 | 3 | 1 | 190.26 | 15.07 | 205.33 |
| i3_3_development_0133 | ✗ | ✗ | -105.44 | -105.44 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0134 | ✗ | ✗ | -100.11 | -100.11 | 1 | 1 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0135 | ✓ | ✓ | 94.56 | 94.56 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0136 | ✓ | ✓ | 94.56 | 94.56 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0137 | ✓ | ✓ | 94.56 | 94.56 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0138 | ✓ | ✓ | 94.56 | 99.89 | 3 | 1 | -9.74 | 15.07 | 5.33 |
| i3_3_development_0139 | ✓ | ✗ | 94.56 | -100.11 | 3 | 1 | -209.74 | 15.07 | -194.67 |
| i3_3_development_0140 | ✗ | ✗ | -100.11 | -100.11 | 1 | 1 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0141 | ✗ | ✗ | -105.44 | -105.44 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0142 | ✗ | ✗ | -104.41 | -100.11 | 3 | 1 | -10.77 | 15.07 | 4.30 |
| i3_3_development_0143 | ✓ | ✓ | 94.56 | 94.56 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0144 | ✗ | ✗ | -105.44 | -100.11 | 3 | 1 | -9.74 | 15.07 | 5.33 |
| i3_3_development_0145 | ✓ | ✓ | 99.89 | 99.89 | 1 | 1 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0146 | ✗ | ✗ | -100.11 | -100.11 | 1 | 1 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0147 | ✓ | ✓ | 94.56 | 94.56 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0148 | ✓ | ✓ | 94.56 | 94.56 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0149 | ✗ | ✗ | -105.44 | -105.44 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0150 | ✗ | ✗ | -105.44 | -109.74 | 3 | 5 | -19.37 | 15.07 | -4.30 |
| i3_3_development_0151 | ✓ | ✓ | 99.89 | 99.89 | 1 | 1 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0152 | ✗ | ✗ | -105.44 | -105.43 | 3 | 3 | -15.06 | 15.07 | 0.01 |
| i3_3_development_0153 | ✗ | ✗ | -105.44 | -105.43 | 3 | 3 | -15.06 | 15.07 | 0.01 |
| i3_3_development_0154 | ✗ | ✓ | -105.44 | 99.89 | 3 | 1 | 190.26 | 15.07 | 205.33 |
| i3_3_development_0155 | ✓ | ✓ | 94.56 | 97.74 | 3 | 2 | -11.89 | 15.07 | 3.18 |
| i3_3_development_0156 | ✗ | ✗ | -100.11 | -100.11 | 1 | 1 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0157 | ✗ | ✗ | -105.44 | -107.58 | 3 | 4 | -17.21 | 15.07 | -2.14 |
| i3_3_development_0158 | ✗ | ✗ | -105.44 | -100.11 | 3 | 1 | -9.74 | 15.07 | 5.33 |
| i3_3_development_0159 | ✓ | ✓ | 99.89 | 99.89 | 1 | 1 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0160 | ✗ | ✗ | -100.11 | -100.11 | 1 | 1 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0161 | ✗ | ✗ | -105.44 | -105.44 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0162 | ✗ | ✓ | -105.44 | 97.74 | 3 | 2 | 188.11 | 15.07 | 203.18 |
| i3_3_development_0163 | ✗ | ✗ | -105.44 | -100.11 | 3 | 1 | -9.74 | 15.07 | 5.33 |
| i3_3_development_0164 | ✗ | ✗ | -105.44 | -105.44 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0165 | ✓ | ✓ | 99.89 | 99.89 | 1 | 1 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0166 | ✗ | ✗ | -100.11 | -100.11 | 1 | 1 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0167 | ✓ | ✓ | 94.56 | 94.56 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0168 | ✓ | ✓ | 94.56 | 99.89 | 3 | 1 | -9.74 | 15.07 | 5.33 |
| i3_3_development_0169 | ✓ | ✓ | 94.56 | 97.74 | 3 | 2 | -11.89 | 15.07 | 3.18 |
| i3_3_development_0170 | ✗ | ✗ | -105.44 | -109.74 | 3 | 5 | -19.37 | 15.07 | -4.30 |
| i3_3_development_0171 | ✗ | ✗ | -105.44 | -105.44 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0172 | ✗ | ✗ | -100.11 | -100.11 | 1 | 1 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0173 | ✓ | ✓ | 94.56 | 94.56 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0174 | ✓ | ✓ | 94.56 | 99.89 | 3 | 1 | -9.74 | 15.07 | 5.33 |
| i3_3_development_0175 | ✗ | ✗ | -105.44 | -105.44 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0176 | ✓ | ✓ | 94.56 | 94.56 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0177 | ✗ | ✗ | -105.44 | -100.11 | 3 | 1 | -9.74 | 15.07 | 5.33 |
| i3_3_development_0178 | ✗ | ✗ | -105.44 | -105.44 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0179 | ✓ | ✓ | 99.89 | 99.89 | 1 | 1 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0180 | ✗ | ✗ | -105.44 | -100.11 | 3 | 1 | -9.74 | 15.07 | 5.33 |
| i3_3_development_0181 | ✓ | ✓ | 99.89 | 99.89 | 1 | 1 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0182 | ✗ | ✗ | -100.11 | -100.11 | 1 | 1 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0183 | ✓ | ✓ | 94.56 | 94.56 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0184 | ✗ | ✓ | -105.44 | 99.89 | 3 | 1 | 190.26 | 15.07 | 205.33 |
| i3_3_development_0185 | ✓ | ✗ | 94.56 | -100.11 | 3 | 1 | -209.74 | 15.07 | -194.67 |
| i3_3_development_0186 | ✗ | ✗ | -100.11 | -100.11 | 1 | 1 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0187 | ✓ | ✓ | 94.56 | 94.56 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0188 | ✓ | ✓ | 94.56 | 99.89 | 3 | 1 | -9.74 | 15.07 | 5.33 |
| i3_3_development_0189 | ✗ | ✗ | -105.44 | -105.44 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0190 | ✗ | ✗ | -105.44 | -105.43 | 3 | 3 | -15.06 | 15.07 | 0.01 |
| i3_3_development_0191 | ✗ | ✗ | -105.44 | -107.59 | 3 | 4 | -17.22 | 15.07 | -2.15 |
| i3_3_development_0192 | ✗ | ✗ | -100.11 | -100.11 | 1 | 1 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0193 | ✗ | ✗ | -102.26 | -105.44 | 2 | 3 | -18.25 | 15.07 | -3.18 |
| i3_3_development_0194 | ✗ | ✗ | -105.44 | -100.11 | 3 | 1 | -9.74 | 15.07 | 5.33 |
| i3_3_development_0195 | ✓ | ✓ | 99.89 | 99.89 | 1 | 1 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0196 | ✓ | ✓ | 94.56 | 99.89 | 3 | 1 | -9.74 | 15.07 | 5.33 |
| i3_3_development_0197 | ✓ | ✓ | 94.56 | 97.74 | 3 | 2 | -11.89 | 15.07 | 3.18 |
| i3_3_development_0198 | ✗ | ✗ | -100.11 | -100.11 | 1 | 1 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0199 | ✓ | ✓ | 94.56 | 94.56 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0200 | ✗ | ✗ | -105.44 | -100.11 | 3 | 1 | -9.74 | 15.07 | 5.33 |
| i3_3_development_0201 | ✓ | ✓ | 99.89 | 99.89 | 1 | 1 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0202 | ✗ | ✓ | -105.44 | 99.89 | 3 | 1 | 190.26 | 15.07 | 205.33 |
| i3_3_development_0203 | ✓ | ✓ | 94.56 | 94.56 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0204 | ✗ | ✗ | -105.44 | -105.44 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0205 | ✓ | ✓ | 94.56 | 94.56 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0206 | ✗ | ✗ | -105.44 | -105.44 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0207 | ✓ | ✓ | 99.89 | 99.89 | 1 | 1 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0208 | ✗ | ✗ | -100.11 | -100.11 | 1 | 1 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0209 | ✓ | ✓ | 94.56 | 94.56 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0210 | ✗ | ✓ | -105.44 | 99.89 | 3 | 1 | 190.26 | 15.07 | 205.33 |
| i3_3_development_0211 | ✗ | ✗ | -105.44 | -109.74 | 3 | 5 | -19.37 | 15.07 | -4.30 |
| i3_3_development_0212 | ✗ | ✗ | -100.11 | -100.11 | 1 | 1 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0213 | ✗ | ✗ | -105.44 | -105.44 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0214 | ✗ | ✗ | -105.44 | -100.11 | 3 | 1 | -9.74 | 15.07 | 5.33 |
| i3_3_development_0215 | ✓ | ✓ | 99.89 | 99.89 | 1 | 1 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0216 | ✗ | ✓ | -105.44 | 99.89 | 3 | 1 | 190.26 | 15.07 | 205.33 |
| i3_3_development_0217 | ✗ | ✗ | -105.44 | -107.60 | 3 | 4 | -17.23 | 15.07 | -2.16 |
| i3_3_development_0218 | ✗ | ✗ | -100.11 | -100.11 | 1 | 1 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0219 | ✓ | ✓ | 94.56 | 94.56 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0220 | ✗ | ✗ | -105.44 | -109.74 | 3 | 5 | -19.37 | 15.07 | -4.30 |
| i3_3_development_0221 | ✓ | ✓ | 99.89 | 99.89 | 1 | 1 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0222 | ✗ | ✓ | -105.44 | 97.74 | 3 | 2 | 188.11 | 15.07 | 203.18 |
| i3_3_development_0223 | ✓ | ✓ | 94.56 | 94.56 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0224 | ✗ | ✗ | -100.11 | -100.11 | 1 | 1 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0225 | ✓ | ✓ | 94.56 | 94.56 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0226 | ✓ | ✗ | 94.56 | -100.11 | 3 | 1 | -209.74 | 15.07 | -194.67 |
| i3_3_development_0227 | ✓ | ✓ | 94.56 | 94.56 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0228 | ✗ | ✗ | -105.44 | -105.44 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0229 | ✗ | ✗ | -105.44 | -105.44 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0230 | ✗ | ✓ | -105.44 | 99.89 | 3 | 1 | 190.26 | 15.07 | 205.33 |
| i3_3_development_0231 | ✗ | ✓ | -105.44 | 96.71 | 3 | 2 | 187.08 | 15.07 | 202.15 |
| i3_3_development_0232 | ✗ | ✓ | -105.44 | 99.89 | 3 | 1 | 190.26 | 15.07 | 205.33 |
| i3_3_development_0233 | ✓ | ✓ | 94.56 | 94.56 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0234 | ✗ | ✗ | -100.11 | -100.11 | 1 | 1 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0235 | ✓ | ✓ | 94.56 | 94.56 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0236 | ✗ | ✗ | -105.44 | -100.11 | 3 | 1 | -9.74 | 15.07 | 5.33 |
| i3_3_development_0237 | ✓ | ✓ | 99.89 | 99.89 | 1 | 1 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0238 | ✗ | ✗ | -100.11 | -100.11 | 1 | 1 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0239 | ✓ | ✓ | 94.56 | 97.74 | 3 | 2 | -11.89 | 15.07 | 3.18 |
| i3_3_development_0240 | ✗ | ✗ | -105.44 | -105.44 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0241 | ✗ | ✗ | -105.44 | -105.44 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0242 | ✗ | ✗ | -105.44 | -100.11 | 3 | 1 | -9.74 | 15.07 | 5.33 |
| i3_3_development_0243 | ✓ | ✓ | 99.89 | 99.89 | 1 | 1 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0244 | ✗ | ✗ | -100.11 | -100.11 | 1 | 1 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0245 | ✓ | ✓ | 94.56 | 94.56 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0246 | ✗ | ✗ | -105.44 | -105.44 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0247 | ✗ | ✗ | -105.44 | -100.11 | 3 | 1 | -9.74 | 15.07 | 5.33 |
| i3_3_development_0248 | ✓ | ✓ | 94.56 | 99.89 | 3 | 1 | -9.74 | 15.07 | 5.33 |
| i3_3_development_0249 | ✓ | ✓ | 94.56 | 99.89 | 3 | 1 | -9.74 | 15.07 | 5.33 |
| i3_3_development_0250 | ✗ | ✗ | -100.11 | -100.11 | 1 | 1 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0251 | ✗ | ✗ | -105.44 | -105.44 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0252 | ✗ | ✓ | -105.44 | 99.89 | 3 | 1 | 190.26 | 15.07 | 205.33 |
| i3_3_development_0253 | ✗ | ✗ | -105.44 | -105.44 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0254 | ✓ | ✓ | 94.56 | 99.89 | 3 | 1 | -9.74 | 15.07 | 5.33 |
| i3_3_development_0255 | ✗ | ✗ | -105.44 | -105.43 | 3 | 3 | -15.06 | 15.07 | 0.01 |
| i3_3_development_0256 | ✗ | ✗ | -105.44 | -100.11 | 3 | 1 | -9.74 | 15.07 | 5.33 |
| i3_3_development_0257 | ✓ | ✓ | 99.89 | 99.89 | 1 | 1 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0258 | ✓ | ✓ | 94.56 | 99.89 | 3 | 1 | -9.74 | 15.07 | 5.33 |
| i3_3_development_0259 | ✗ | ✗ | -105.44 | -105.43 | 3 | 3 | -15.06 | 15.07 | 0.01 |
| i3_3_development_0260 | ✗ | ✗ | -100.11 | -100.11 | 1 | 1 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0261 | ✓ | ✓ | 94.56 | 94.56 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0262 | ✗ | ✗ | -105.44 | -109.74 | 3 | 5 | -19.37 | 15.07 | -4.30 |
| i3_3_development_0263 | ✓ | ✓ | 99.89 | 99.89 | 1 | 1 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0264 | ✗ | ✗ | -100.11 | -100.11 | 1 | 1 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0265 | ✗ | ✗ | -105.44 | -105.44 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0266 | ✗ | ✗ | -105.44 | -105.43 | 3 | 3 | -15.06 | 15.07 | 0.01 |
| i3_3_development_0267 | ✗ | ✗ | -105.44 | -105.44 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0268 | ✓ | ✗ | 94.56 | -100.11 | 3 | 1 | -209.74 | 15.07 | -194.67 |
| i3_3_development_0269 | ✗ | ✗ | -105.44 | -105.44 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0270 | ✗ | ✗ | -100.11 | -100.11 | 1 | 1 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0271 | ✓ | ✓ | 94.56 | 94.56 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0272 | ✗ | ✗ | -105.44 | -107.60 | 3 | 4 | -17.23 | 15.07 | -2.16 |
| i3_3_development_0273 | ✓ | ✓ | 94.56 | 94.56 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0274 | ✓ | ✓ | 94.56 | 94.56 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0275 | ✗ | ✗ | -105.44 | -100.11 | 3 | 1 | -9.74 | 15.07 | 5.33 |
| i3_3_development_0276 | ✗ | ✗ | -100.11 | -100.11 | 1 | 1 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0277 | ✓ | ✓ | 94.56 | 94.56 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0278 | ✗ | ✗ | -105.44 | -100.11 | 3 | 1 | -9.74 | 15.07 | 5.33 |
| i3_3_development_0279 | ✓ | ✓ | 99.89 | 99.89 | 1 | 1 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0280 | ✗ | ✓ | -105.44 | 99.89 | 3 | 1 | 190.26 | 15.07 | 205.33 |
| i3_3_development_0281 | ✗ | ✓ | -105.44 | 97.74 | 3 | 2 | 188.11 | 15.07 | 203.18 |
| i3_3_development_0282 | ✗ | ✓ | -105.44 | 96.71 | 3 | 2 | 187.08 | 15.07 | 202.15 |
| i3_3_development_0283 | ✗ | ✓ | -105.44 | 96.71 | 3 | 2 | 187.08 | 15.07 | 202.15 |
| i3_3_development_0284 | ✗ | ✗ | -105.44 | -100.11 | 3 | 1 | -9.74 | 15.07 | 5.33 |
| i3_3_development_0285 | ✓ | ✓ | 99.89 | 99.89 | 1 | 1 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0286 | ✗ | ✗ | -100.11 | -100.11 | 1 | 1 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0287 | ✗ | ✗ | -105.44 | -105.44 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0288 | ✓ | ✓ | 94.56 | 94.56 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0289 | ✗ | ✗ | -105.44 | -100.11 | 3 | 1 | -9.74 | 15.07 | 5.33 |
| i3_3_development_0290 | ✗ | ✗ | -100.11 | -100.11 | 1 | 1 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0291 | ✓ | ✓ | 94.56 | 94.56 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0292 | ✗ | ✗ | -105.44 | -100.11 | 3 | 1 | -9.74 | 15.07 | 5.33 |
| i3_3_development_0293 | ✓ | ✓ | 99.89 | 99.89 | 1 | 1 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0294 | ✓ | ✓ | 94.56 | 99.89 | 3 | 1 | -9.74 | 15.07 | 5.33 |
| i3_3_development_0295 | ✗ | ✓ | -105.44 | 97.74 | 3 | 2 | 188.11 | 15.07 | 203.18 |
| i3_3_development_0296 | ✗ | ✗ | -100.11 | -100.11 | 1 | 1 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0297 | ✓ | ✓ | 94.56 | 94.56 | 3 | 3 | -15.07 | 15.07 | 0.00 |
| i3_3_development_0298 | ✗ | ✗ | -105.44 | -100.11 | 3 | 1 | -9.74 | 15.07 | 5.33 |
| i3_3_development_0299 | ✓ | ✓ | 99.89 | 99.89 | 1 | 1 | -15.07 | 15.07 | 0.00 |

## 9. Per-Task Trajectory Details

### Notable trajectories (disagreements and high-ΔDG tasks)

#### i3_3_development_0002

| Property | Blind | Aware |
|----------|-------|------|
| Success | False | True |
| Terminal | TASK_FAILURE | TASK_SUCCESS |
| Utility | -105.4400 | 99.8900 |
| API calls | 3 | 1 |
| ΔDG | 190.2647 | — |
| ΔIG | 15.0653 | — |
| ΔTR | 205.3300 | — |

**Blind trajectory:**

| Step | Proposed | Reason | Executed | Outcome | Terminal |
|------|----------|--------|----------|---------|----------|
| 0 | RETRIEVE | INITIAL_EVIDENCE_GATHERING | RETRIEVE | RETRIEVE_COMPLETED | False |
| 1 | VERIFY | EVIDENCE_RETRIEVED_NEEDS_VERIFICATION | VERIFY | VERIFY_COMPLETED | False |
| 2 | ANSWER | EVIDENCE_SUFFICIENT | ANSWER | TASK_FAILURE | True |

**Aware trajectory:**

| Step | Proposed | Reason | Executed | Outcome | Terminal |
|------|----------|--------|----------|---------|----------|
| 0 | ANSWER | EVIDENCE_SUFFICIENT | ANSWER | TASK_SUCCESS | True |

#### i3_3_development_0020

| Property | Blind | Aware |
|----------|-------|------|
| Success | False | True |
| Terminal | TASK_FAILURE | TASK_SUCCESS |
| Utility | -105.4400 | 99.8900 |
| API calls | 3 | 1 |
| ΔDG | 190.2647 | — |
| ΔIG | 15.0653 | — |
| ΔTR | 205.3300 | — |

**Blind trajectory:**

| Step | Proposed | Reason | Executed | Outcome | Terminal |
|------|----------|--------|----------|---------|----------|
| 0 | RETRIEVE | INITIAL_EVIDENCE_GATHERING | RETRIEVE | RETRIEVE_COMPLETED | False |
| 1 | VERIFY | EVIDENCE_RETRIEVED_NEEDS_VERIFICATION | VERIFY | VERIFY_COMPLETED | False |
| 2 | ANSWER | EVIDENCE_SUFFICIENT | ANSWER | TASK_FAILURE | True |

**Aware trajectory:**

| Step | Proposed | Reason | Executed | Outcome | Terminal |
|------|----------|--------|----------|---------|----------|
| 0 | ANSWER | EVIDENCE_SUFFICIENT | ANSWER | TASK_SUCCESS | True |

#### i3_3_development_0047

| Property | Blind | Aware |
|----------|-------|------|
| Success | True | False |
| Terminal | TASK_SUCCESS | TASK_FAILURE |
| Utility | 94.5600 | -100.1100 |
| API calls | 3 | 1 |
| ΔDG | -209.7353 | — |
| ΔIG | 15.0653 | — |
| ΔTR | -194.6700 | — |

**Blind trajectory:**

| Step | Proposed | Reason | Executed | Outcome | Terminal |
|------|----------|--------|----------|---------|----------|
| 0 | RETRIEVE | INITIAL_EVIDENCE_GATHERING | RETRIEVE | RETRIEVE_COMPLETED | False |
| 1 | VERIFY | VERIFY_AVAILABLE_EVIDENCE | VERIFY | VERIFY_COMPLETED | False |
| 2 | ANSWER | SUFFICIENT_EVIDENCE_AND_BUDGET | ANSWER | TASK_SUCCESS | True |

**Aware trajectory:**

| Step | Proposed | Reason | Executed | Outcome | Terminal |
|------|----------|--------|----------|---------|----------|
| 0 | ANSWER | SUFFICIENT_EVIDENCE_AVAILABLE | ANSWER | TASK_FAILURE | True |

#### i3_3_development_0070

| Property | Blind | Aware |
|----------|-------|------|
| Success | False | True |
| Terminal | TASK_FAILURE | TASK_SUCCESS |
| Utility | -105.4400 | 99.8900 |
| API calls | 3 | 1 |
| ΔDG | 190.2647 | — |
| ΔIG | 15.0653 | — |
| ΔTR | 205.3300 | — |

**Blind trajectory:**

| Step | Proposed | Reason | Executed | Outcome | Terminal |
|------|----------|--------|----------|---------|----------|
| 0 | RETRIEVE | INITIAL_EVIDENCE_GATHERING | RETRIEVE | RETRIEVE_COMPLETED | False |
| 1 | VERIFY | EVIDENCE_RETRIEVED_NEEDS_VERIFICATION | VERIFY | VERIFY_COMPLETED | False |
| 2 | ANSWER | EVIDENCE_SUFFICIENT | ANSWER | TASK_FAILURE | True |

**Aware trajectory:**

| Step | Proposed | Reason | Executed | Outcome | Terminal |
|------|----------|--------|----------|---------|----------|
| 0 | ANSWER | EVIDENCE_SUFFICIENT | ANSWER | TASK_SUCCESS | True |

#### i3_3_development_0072

| Property | Blind | Aware |
|----------|-------|------|
| Success | False | True |
| Terminal | TASK_FAILURE | TASK_SUCCESS |
| Utility | -105.4400 | 96.7100 |
| API calls | 3 | 2 |
| ΔDG | 187.0847 | — |
| ΔIG | 15.0653 | — |
| ΔTR | 202.1500 | — |

**Blind trajectory:**

| Step | Proposed | Reason | Executed | Outcome | Terminal |
|------|----------|--------|----------|---------|----------|
| 0 | RETRIEVE | INITIAL_EVIDENCE_GATHERING | RETRIEVE | RETRIEVE_COMPLETED | False |
| 1 | VERIFY | VERIFY_AVAILABLE_EVIDENCE | VERIFY | VERIFY_COMPLETED | False |
| 2 | ANSWER | EVIDENCE_SUFFICIENT | ANSWER | TASK_FAILURE | True |

**Aware trajectory:**

| Step | Proposed | Reason | Executed | Outcome | Terminal |
|------|----------|--------|----------|---------|----------|
| 0 | VERIFY | INITIAL_VERIFICATION_REQUIRED | VERIFY | VERIFY_COMPLETED | False |
| 1 | ANSWER | EVIDENCE_SUFFICIENT | ANSWER | TASK_SUCCESS | True |

#### i3_3_development_0090

| Property | Blind | Aware |
|----------|-------|------|
| Success | False | True |
| Terminal | TASK_FAILURE | TASK_SUCCESS |
| Utility | -105.4400 | 99.8900 |
| API calls | 3 | 1 |
| ΔDG | 190.2647 | — |
| ΔIG | 15.0653 | — |
| ΔTR | 205.3300 | — |

**Blind trajectory:**

| Step | Proposed | Reason | Executed | Outcome | Terminal |
|------|----------|--------|----------|---------|----------|
| 0 | RETRIEVE | INITIAL_EVIDENCE_GATHERING | RETRIEVE | RETRIEVE_COMPLETED | False |
| 1 | VERIFY | EVIDENCE_RETRIEVED_NEEDS_VERIFICATION | VERIFY | VERIFY_COMPLETED | False |
| 2 | ANSWER | EVIDENCE_SUFFICIENT | ANSWER | TASK_FAILURE | True |

**Aware trajectory:**

| Step | Proposed | Reason | Executed | Outcome | Terminal |
|------|----------|--------|----------|---------|----------|
| 0 | ANSWER | EVIDENCE_SUFFICIENT | ANSWER | TASK_SUCCESS | True |

#### i3_3_development_0092

| Property | Blind | Aware |
|----------|-------|------|
| Success | False | True |
| Terminal | TASK_FAILURE | TASK_SUCCESS |
| Utility | -105.4400 | 99.8900 |
| API calls | 3 | 1 |
| ΔDG | 190.2647 | — |
| ΔIG | 15.0653 | — |
| ΔTR | 205.3300 | — |

**Blind trajectory:**

| Step | Proposed | Reason | Executed | Outcome | Terminal |
|------|----------|--------|----------|---------|----------|
| 0 | RETRIEVE | INITIAL_EVIDENCE_GATHERING | RETRIEVE | RETRIEVE_COMPLETED | False |
| 1 | VERIFY | EVIDENCE_RETRIEVED_NEEDS_VERIFICATION | VERIFY | VERIFY_COMPLETED | False |
| 2 | ANSWER | EVIDENCE_SUFFICIENT | ANSWER | TASK_FAILURE | True |

**Aware trajectory:**

| Step | Proposed | Reason | Executed | Outcome | Terminal |
|------|----------|--------|----------|---------|----------|
| 0 | ANSWER | SUFFICIENT_EVIDENCE_AVAILABLE | ANSWER | TASK_SUCCESS | True |

#### i3_3_development_0093

| Property | Blind | Aware |
|----------|-------|------|
| Success | True | False |
| Terminal | TASK_SUCCESS | TASK_FAILURE |
| Utility | 94.5600 | -100.1100 |
| API calls | 3 | 1 |
| ΔDG | -209.7353 | — |
| ΔIG | 15.0653 | — |
| ΔTR | -194.6700 | — |

**Blind trajectory:**

| Step | Proposed | Reason | Executed | Outcome | Terminal |
|------|----------|--------|----------|---------|----------|
| 0 | RETRIEVE | INITIAL_EVIDENCE_GATHERING | RETRIEVE | RETRIEVE_COMPLETED | False |
| 1 | VERIFY | INITIAL_EVIDENCE_NEEDS_CHECK | VERIFY | VERIFY_COMPLETED | False |
| 2 | ANSWER | EVIDENCE_SUFFICIENT | ANSWER | TASK_SUCCESS | True |

**Aware trajectory:**

| Step | Proposed | Reason | Executed | Outcome | Terminal |
|------|----------|--------|----------|---------|----------|
| 0 | ANSWER | SUFFICIENT_EVIDENCE_AVAILABLE | ANSWER | TASK_FAILURE | True |

#### i3_3_development_0112

| Property | Blind | Aware |
|----------|-------|------|
| Success | False | True |
| Terminal | TASK_FAILURE | TASK_SUCCESS |
| Utility | -105.4400 | 99.8900 |
| API calls | 3 | 1 |
| ΔDG | 190.2647 | — |
| ΔIG | 15.0653 | — |
| ΔTR | 205.3300 | — |

**Blind trajectory:**

| Step | Proposed | Reason | Executed | Outcome | Terminal |
|------|----------|--------|----------|---------|----------|
| 0 | RETRIEVE | INITIAL_EVIDENCE_GATHERING | RETRIEVE | RETRIEVE_COMPLETED | False |
| 1 | VERIFY | EVIDENCE_RETRIEVED_NEEDS_VERIFICATION | VERIFY | VERIFY_COMPLETED | False |
| 2 | ANSWER | EVIDENCE_SUFFICIENT | ANSWER | TASK_FAILURE | True |

**Aware trajectory:**

| Step | Proposed | Reason | Executed | Outcome | Terminal |
|------|----------|--------|----------|---------|----------|
| 0 | ANSWER | EVIDENCE_SUFFICIENT | ANSWER | TASK_SUCCESS | True |

#### i3_3_development_0132

| Property | Blind | Aware |
|----------|-------|------|
| Success | False | True |
| Terminal | TASK_FAILURE | TASK_SUCCESS |
| Utility | -105.4400 | 99.8900 |
| API calls | 3 | 1 |
| ΔDG | 190.2647 | — |
| ΔIG | 15.0653 | — |
| ΔTR | 205.3300 | — |

**Blind trajectory:**

| Step | Proposed | Reason | Executed | Outcome | Terminal |
|------|----------|--------|----------|---------|----------|
| 0 | RETRIEVE | INITIAL_EVIDENCE_GATHERING | RETRIEVE | RETRIEVE_COMPLETED | False |
| 1 | VERIFY | EVIDENCE_RETRIEVED_NEEDS_VERIFICATION | VERIFY | VERIFY_COMPLETED | False |
| 2 | ANSWER | EVIDENCE_SUFFICIENT | ANSWER | TASK_FAILURE | True |

**Aware trajectory:**

| Step | Proposed | Reason | Executed | Outcome | Terminal |
|------|----------|--------|----------|---------|----------|
| 0 | ANSWER | EVIDENCE_SUFFICIENT | ANSWER | TASK_SUCCESS | True |

#### i3_3_development_0139

| Property | Blind | Aware |
|----------|-------|------|
| Success | True | False |
| Terminal | TASK_SUCCESS | TASK_FAILURE |
| Utility | 94.5600 | -100.1100 |
| API calls | 3 | 1 |
| ΔDG | -209.7353 | — |
| ΔIG | 15.0653 | — |
| ΔTR | -194.6700 | — |

**Blind trajectory:**

| Step | Proposed | Reason | Executed | Outcome | Terminal |
|------|----------|--------|----------|---------|----------|
| 0 | RETRIEVE | INITIAL_EVIDENCE_GATHERING | RETRIEVE | RETRIEVE_COMPLETED | False |
| 1 | VERIFY | EVIDENCE_RETRIEVED_NEEDS_VERIFICATION | VERIFY | VERIFY_COMPLETED | False |
| 2 | ANSWER | EVIDENCE_SUFFICIENT | ANSWER | TASK_SUCCESS | True |

**Aware trajectory:**

| Step | Proposed | Reason | Executed | Outcome | Terminal |
|------|----------|--------|----------|---------|----------|
| 0 | ANSWER | EVIDENCE_SUFFICIENT | ANSWER | TASK_FAILURE | True |

#### i3_3_development_0154

| Property | Blind | Aware |
|----------|-------|------|
| Success | False | True |
| Terminal | TASK_FAILURE | TASK_SUCCESS |
| Utility | -105.4400 | 99.8900 |
| API calls | 3 | 1 |
| ΔDG | 190.2647 | — |
| ΔIG | 15.0653 | — |
| ΔTR | 205.3300 | — |

**Blind trajectory:**

| Step | Proposed | Reason | Executed | Outcome | Terminal |
|------|----------|--------|----------|---------|----------|
| 0 | RETRIEVE | INITIAL_EVIDENCE_GATHERING | RETRIEVE | RETRIEVE_COMPLETED | False |
| 1 | VERIFY | EVIDENCE_RETRIEVED_NEEDS_VERIFICATION | VERIFY | VERIFY_COMPLETED | False |
| 2 | ANSWER | EVIDENCE_SUFFICIENT | ANSWER | TASK_FAILURE | True |

**Aware trajectory:**

| Step | Proposed | Reason | Executed | Outcome | Terminal |
|------|----------|--------|----------|---------|----------|
| 0 | ANSWER | SUFFICIENT_EVIDENCE_AVAILABLE | ANSWER | TASK_SUCCESS | True |

#### i3_3_development_0162

| Property | Blind | Aware |
|----------|-------|------|
| Success | False | True |
| Terminal | TASK_FAILURE | TASK_SUCCESS |
| Utility | -105.4400 | 97.7400 |
| API calls | 3 | 2 |
| ΔDG | 188.1147 | — |
| ΔIG | 15.0653 | — |
| ΔTR | 203.1800 | — |

**Blind trajectory:**

| Step | Proposed | Reason | Executed | Outcome | Terminal |
|------|----------|--------|----------|---------|----------|
| 0 | RETRIEVE | INITIAL_EVIDENCE_GATHERING | RETRIEVE | RETRIEVE_COMPLETED | False |
| 1 | VERIFY | EVIDENCE_RETRIEVED_NEEDS_VERIFICATION | VERIFY | VERIFY_COMPLETED | False |
| 2 | ANSWER | EVIDENCE_SUFFICIENT | ANSWER | TASK_FAILURE | True |

**Aware trajectory:**

| Step | Proposed | Reason | Executed | Outcome | Terminal |
|------|----------|--------|----------|---------|----------|
| 0 | RETRIEVE | INITIAL_EVIDENCE_NEEDED | RETRIEVE | RETRIEVE_COMPLETED | False |
| 1 | ANSWER | EVIDENCE_SUFFICIENT | ANSWER | TASK_SUCCESS | True |

#### i3_3_development_0184

| Property | Blind | Aware |
|----------|-------|------|
| Success | False | True |
| Terminal | TASK_FAILURE | TASK_SUCCESS |
| Utility | -105.4400 | 99.8900 |
| API calls | 3 | 1 |
| ΔDG | 190.2647 | — |
| ΔIG | 15.0653 | — |
| ΔTR | 205.3300 | — |

**Blind trajectory:**

| Step | Proposed | Reason | Executed | Outcome | Terminal |
|------|----------|--------|----------|---------|----------|
| 0 | RETRIEVE | INITIAL_EVIDENCE_GATHERING | RETRIEVE | RETRIEVE_COMPLETED | False |
| 1 | VERIFY | EVIDENCE_RETRIEVED_NEEDS_VERIFICATION | VERIFY | VERIFY_COMPLETED | False |
| 2 | ANSWER | EVIDENCE_SUFFICIENT | ANSWER | TASK_FAILURE | True |

**Aware trajectory:**

| Step | Proposed | Reason | Executed | Outcome | Terminal |
|------|----------|--------|----------|---------|----------|
| 0 | ANSWER | EVIDENCE_SUFFICIENT | ANSWER | TASK_SUCCESS | True |

#### i3_3_development_0185

| Property | Blind | Aware |
|----------|-------|------|
| Success | True | False |
| Terminal | TASK_SUCCESS | TASK_FAILURE |
| Utility | 94.5600 | -100.1100 |
| API calls | 3 | 1 |
| ΔDG | -209.7353 | — |
| ΔIG | 15.0653 | — |
| ΔTR | -194.6700 | — |

**Blind trajectory:**

| Step | Proposed | Reason | Executed | Outcome | Terminal |
|------|----------|--------|----------|---------|----------|
| 0 | RETRIEVE | INITIAL_EVIDENCE_GATHERING | RETRIEVE | RETRIEVE_COMPLETED | False |
| 1 | VERIFY | EVIDENCE_RETRIEVED_NEEDS_VERIFICATION | VERIFY | VERIFY_COMPLETED | False |
| 2 | ANSWER | EVIDENCE_SUFFICIENT | ANSWER | TASK_SUCCESS | True |

**Aware trajectory:**

| Step | Proposed | Reason | Executed | Outcome | Terminal |
|------|----------|--------|----------|---------|----------|
| 0 | ANSWER | EVIDENCE_SUFFICIENT | ANSWER | TASK_FAILURE | True |

#### i3_3_development_0202

| Property | Blind | Aware |
|----------|-------|------|
| Success | False | True |
| Terminal | TASK_FAILURE | TASK_SUCCESS |
| Utility | -105.4400 | 99.8900 |
| API calls | 3 | 1 |
| ΔDG | 190.2647 | — |
| ΔIG | 15.0653 | — |
| ΔTR | 205.3300 | — |

**Blind trajectory:**

| Step | Proposed | Reason | Executed | Outcome | Terminal |
|------|----------|--------|----------|---------|----------|
| 0 | RETRIEVE | INITIAL_EVIDENCE_GATHERING | RETRIEVE | RETRIEVE_COMPLETED | False |
| 1 | VERIFY | EVIDENCE_RETRIEVED_NEEDS_VERIFICATION | VERIFY | VERIFY_COMPLETED | False |
| 2 | ANSWER | EVIDENCE_SUFFICIENT | ANSWER | TASK_FAILURE | True |

**Aware trajectory:**

| Step | Proposed | Reason | Executed | Outcome | Terminal |
|------|----------|--------|----------|---------|----------|
| 0 | ANSWER | EVIDENCE_SUFFICIENT | ANSWER | TASK_SUCCESS | True |

#### i3_3_development_0210

| Property | Blind | Aware |
|----------|-------|------|
| Success | False | True |
| Terminal | TASK_FAILURE | TASK_SUCCESS |
| Utility | -105.4400 | 99.8900 |
| API calls | 3 | 1 |
| ΔDG | 190.2647 | — |
| ΔIG | 15.0653 | — |
| ΔTR | 205.3300 | — |

**Blind trajectory:**

| Step | Proposed | Reason | Executed | Outcome | Terminal |
|------|----------|--------|----------|---------|----------|
| 0 | RETRIEVE | INITIAL_EVIDENCE_GATHERING | RETRIEVE | RETRIEVE_COMPLETED | False |
| 1 | VERIFY | EVIDENCE_RETRIEVED_NEEDS_VERIFICATION | VERIFY | VERIFY_COMPLETED | False |
| 2 | ANSWER | EVIDENCE_SUFFICIENT | ANSWER | TASK_FAILURE | True |

**Aware trajectory:**

| Step | Proposed | Reason | Executed | Outcome | Terminal |
|------|----------|--------|----------|---------|----------|
| 0 | ANSWER | EVIDENCE_SUFFICIENT | ANSWER | TASK_SUCCESS | True |

#### i3_3_development_0216

| Property | Blind | Aware |
|----------|-------|------|
| Success | False | True |
| Terminal | TASK_FAILURE | TASK_SUCCESS |
| Utility | -105.4400 | 99.8900 |
| API calls | 3 | 1 |
| ΔDG | 190.2647 | — |
| ΔIG | 15.0653 | — |
| ΔTR | 205.3300 | — |

**Blind trajectory:**

| Step | Proposed | Reason | Executed | Outcome | Terminal |
|------|----------|--------|----------|---------|----------|
| 0 | RETRIEVE | INITIAL_EVIDENCE_GATHERING | RETRIEVE | RETRIEVE_COMPLETED | False |
| 1 | VERIFY | EVIDENCE_RETRIEVED_NEEDS_VERIFICATION | VERIFY | VERIFY_COMPLETED | False |
| 2 | ANSWER | EVIDENCE_SUFFICIENT_AND_BUDGET_LOW | ANSWER | TASK_FAILURE | True |

**Aware trajectory:**

| Step | Proposed | Reason | Executed | Outcome | Terminal |
|------|----------|--------|----------|---------|----------|
| 0 | ANSWER | EVIDENCE_SUFFICIENT | ANSWER | TASK_SUCCESS | True |

#### i3_3_development_0222

| Property | Blind | Aware |
|----------|-------|------|
| Success | False | True |
| Terminal | TASK_FAILURE | TASK_SUCCESS |
| Utility | -105.4400 | 97.7400 |
| API calls | 3 | 2 |
| ΔDG | 188.1147 | — |
| ΔIG | 15.0653 | — |
| ΔTR | 203.1800 | — |

**Blind trajectory:**

| Step | Proposed | Reason | Executed | Outcome | Terminal |
|------|----------|--------|----------|---------|----------|
| 0 | RETRIEVE | INITIAL_EVIDENCE_GATHERING | RETRIEVE | RETRIEVE_COMPLETED | False |
| 1 | VERIFY | EVIDENCE_RETRIEVED_NEEDS_VERIFICATION | VERIFY | VERIFY_COMPLETED | False |
| 2 | ANSWER | EVIDENCE_SUFFICIENT | ANSWER | TASK_FAILURE | True |

**Aware trajectory:**

| Step | Proposed | Reason | Executed | Outcome | Terminal |
|------|----------|--------|----------|---------|----------|
| 0 | RETRIEVE | INITIAL_EVIDENCE_GATHERING | RETRIEVE | RETRIEVE_COMPLETED | False |
| 1 | ANSWER | EVIDENCE_SUFFICIENT | ANSWER | TASK_SUCCESS | True |

#### i3_3_development_0226

| Property | Blind | Aware |
|----------|-------|------|
| Success | True | False |
| Terminal | TASK_SUCCESS | TASK_FAILURE |
| Utility | 94.5600 | -100.1100 |
| API calls | 3 | 1 |
| ΔDG | -209.7353 | — |
| ΔIG | 15.0653 | — |
| ΔTR | -194.6700 | — |

**Blind trajectory:**

| Step | Proposed | Reason | Executed | Outcome | Terminal |
|------|----------|--------|----------|---------|----------|
| 0 | RETRIEVE | INITIAL_EVIDENCE_GATHERING | RETRIEVE | RETRIEVE_COMPLETED | False |
| 1 | VERIFY | EVIDENCE_RETRIEVED_NEEDS_VERIFICATION | VERIFY | VERIFY_COMPLETED | False |
| 2 | ANSWER | EVIDENCE_SUFFICIENT | ANSWER | TASK_SUCCESS | True |

**Aware trajectory:**

| Step | Proposed | Reason | Executed | Outcome | Terminal |
|------|----------|--------|----------|---------|----------|
| 0 | ANSWER | EVIDENCE_SUFFICIENT | ANSWER | TASK_FAILURE | True |

#### i3_3_development_0230

| Property | Blind | Aware |
|----------|-------|------|
| Success | False | True |
| Terminal | TASK_FAILURE | TASK_SUCCESS |
| Utility | -105.4400 | 99.8900 |
| API calls | 3 | 1 |
| ΔDG | 190.2647 | — |
| ΔIG | 15.0653 | — |
| ΔTR | 205.3300 | — |

**Blind trajectory:**

| Step | Proposed | Reason | Executed | Outcome | Terminal |
|------|----------|--------|----------|---------|----------|
| 0 | RETRIEVE | INITIAL_EVIDENCE_GATHERING | RETRIEVE | RETRIEVE_COMPLETED | False |
| 1 | VERIFY | EVIDENCE_RETRIEVED_NEEDS_VERIFICATION | VERIFY | VERIFY_COMPLETED | False |
| 2 | ANSWER | EVIDENCE_SUFFICIENT | ANSWER | TASK_FAILURE | True |

**Aware trajectory:**

| Step | Proposed | Reason | Executed | Outcome | Terminal |
|------|----------|--------|----------|---------|----------|
| 0 | ANSWER | EVIDENCE_SUFFICIENT | ANSWER | TASK_SUCCESS | True |

#### i3_3_development_0231

| Property | Blind | Aware |
|----------|-------|------|
| Success | False | True |
| Terminal | TASK_FAILURE | TASK_SUCCESS |
| Utility | -105.4400 | 96.7100 |
| API calls | 3 | 2 |
| ΔDG | 187.0847 | — |
| ΔIG | 15.0653 | — |
| ΔTR | 202.1500 | — |

**Blind trajectory:**

| Step | Proposed | Reason | Executed | Outcome | Terminal |
|------|----------|--------|----------|---------|----------|
| 0 | RETRIEVE | INITIAL_EVIDENCE_GATHERING | RETRIEVE | RETRIEVE_COMPLETED | False |
| 1 | VERIFY | EVIDENCE_RETRIEVED_NEEDS_VERIFICATION | VERIFY | VERIFY_COMPLETED | False |
| 2 | ANSWER | EVIDENCE_SUFFICIENT | ANSWER | TASK_FAILURE | True |

**Aware trajectory:**

| Step | Proposed | Reason | Executed | Outcome | Terminal |
|------|----------|--------|----------|---------|----------|
| 0 | VERIFY | INITIAL_VERIFICATION_REQUIRED | VERIFY | VERIFY_COMPLETED | False |
| 1 | ANSWER | EVIDENCE_SUFFICIENT | ANSWER | TASK_SUCCESS | True |

#### i3_3_development_0232

| Property | Blind | Aware |
|----------|-------|------|
| Success | False | True |
| Terminal | TASK_FAILURE | TASK_SUCCESS |
| Utility | -105.4400 | 99.8900 |
| API calls | 3 | 1 |
| ΔDG | 190.2647 | — |
| ΔIG | 15.0653 | — |
| ΔTR | 205.3300 | — |

**Blind trajectory:**

| Step | Proposed | Reason | Executed | Outcome | Terminal |
|------|----------|--------|----------|---------|----------|
| 0 | RETRIEVE | INITIAL_EVIDENCE_GATHERING | RETRIEVE | RETRIEVE_COMPLETED | False |
| 1 | VERIFY | EVIDENCE_RETRIEVED_NEEDS_VERIFICATION | VERIFY | VERIFY_COMPLETED | False |
| 2 | ANSWER | EVIDENCE_SUFFICIENT | ANSWER | TASK_FAILURE | True |

**Aware trajectory:**

| Step | Proposed | Reason | Executed | Outcome | Terminal |
|------|----------|--------|----------|---------|----------|
| 0 | ANSWER | EVIDENCE_SUFFICIENT | ANSWER | TASK_SUCCESS | True |

#### i3_3_development_0252

| Property | Blind | Aware |
|----------|-------|------|
| Success | False | True |
| Terminal | TASK_FAILURE | TASK_SUCCESS |
| Utility | -105.4400 | 99.8900 |
| API calls | 3 | 1 |
| ΔDG | 190.2647 | — |
| ΔIG | 15.0653 | — |
| ΔTR | 205.3300 | — |

**Blind trajectory:**

| Step | Proposed | Reason | Executed | Outcome | Terminal |
|------|----------|--------|----------|---------|----------|
| 0 | RETRIEVE | INITIAL_EVIDENCE_GATHERING | RETRIEVE | RETRIEVE_COMPLETED | False |
| 1 | VERIFY | EVIDENCE_RETRIEVED_NEEDS_VERIFICATION | VERIFY | VERIFY_COMPLETED | False |
| 2 | ANSWER | EVIDENCE_SUFFICIENT | ANSWER | TASK_FAILURE | True |

**Aware trajectory:**

| Step | Proposed | Reason | Executed | Outcome | Terminal |
|------|----------|--------|----------|---------|----------|
| 0 | ANSWER | EVIDENCE_SUFFICIENT | ANSWER | TASK_SUCCESS | True |

#### i3_3_development_0268

| Property | Blind | Aware |
|----------|-------|------|
| Success | True | False |
| Terminal | TASK_SUCCESS | TASK_FAILURE |
| Utility | 94.5600 | -100.1100 |
| API calls | 3 | 1 |
| ΔDG | -209.7353 | — |
| ΔIG | 15.0653 | — |
| ΔTR | -194.6700 | — |

**Blind trajectory:**

| Step | Proposed | Reason | Executed | Outcome | Terminal |
|------|----------|--------|----------|---------|----------|
| 0 | RETRIEVE | INITIAL_EVIDENCE_GATHERING | RETRIEVE | RETRIEVE_COMPLETED | False |
| 1 | VERIFY | EVIDENCE_RETRIEVED_NEEDS_VERIFICATION | VERIFY | VERIFY_COMPLETED | False |
| 2 | ANSWER | EVIDENCE_SUFFICIENT | ANSWER | TASK_SUCCESS | True |

**Aware trajectory:**

| Step | Proposed | Reason | Executed | Outcome | Terminal |
|------|----------|--------|----------|---------|----------|
| 0 | ANSWER | EVIDENCE_SUFFICIENT | ANSWER | TASK_FAILURE | True |

#### i3_3_development_0280

| Property | Blind | Aware |
|----------|-------|------|
| Success | False | True |
| Terminal | TASK_FAILURE | TASK_SUCCESS |
| Utility | -105.4400 | 99.8900 |
| API calls | 3 | 1 |
| ΔDG | 190.2647 | — |
| ΔIG | 15.0653 | — |
| ΔTR | 205.3300 | — |

**Blind trajectory:**

| Step | Proposed | Reason | Executed | Outcome | Terminal |
|------|----------|--------|----------|---------|----------|
| 0 | RETRIEVE | INITIAL_EVIDENCE_GATHERING | RETRIEVE | RETRIEVE_COMPLETED | False |
| 1 | VERIFY | NEED_VERIFICATION_AFTER_RETRIEVAL | VERIFY | VERIFY_COMPLETED | False |
| 2 | ANSWER | SUFFICIENT_EVIDENCE_AND_BUDGET | ANSWER | TASK_FAILURE | True |

**Aware trajectory:**

| Step | Proposed | Reason | Executed | Outcome | Terminal |
|------|----------|--------|----------|---------|----------|
| 0 | ANSWER | SUFFICIENT_EVIDENCE_AVAILABLE | ANSWER | TASK_SUCCESS | True |

#### i3_3_development_0281

| Property | Blind | Aware |
|----------|-------|------|
| Success | False | True |
| Terminal | TASK_FAILURE | TASK_SUCCESS |
| Utility | -105.4400 | 97.7400 |
| API calls | 3 | 2 |
| ΔDG | 188.1147 | — |
| ΔIG | 15.0653 | — |
| ΔTR | 203.1800 | — |

**Blind trajectory:**

| Step | Proposed | Reason | Executed | Outcome | Terminal |
|------|----------|--------|----------|---------|----------|
| 0 | RETRIEVE | INITIAL_EVIDENCE_GATHERING | RETRIEVE | RETRIEVE_COMPLETED | False |
| 1 | VERIFY | NEED_VERIFICATION_AFTER_RETRIEVAL | VERIFY | VERIFY_COMPLETED | False |
| 2 | ANSWER | SUFFICIENT_EVIDENCE_AND_BUDGET | ANSWER | TASK_FAILURE | True |

**Aware trajectory:**

| Step | Proposed | Reason | Executed | Outcome | Terminal |
|------|----------|--------|----------|---------|----------|
| 0 | RETRIEVE | INITIAL_EVIDENCE_NEEDED | RETRIEVE | RETRIEVE_COMPLETED | False |
| 1 | ANSWER | SUFFICIENT_EVIDENCE_AVAILABLE | ANSWER | TASK_SUCCESS | True |

#### i3_3_development_0282

| Property | Blind | Aware |
|----------|-------|------|
| Success | False | True |
| Terminal | TASK_FAILURE | TASK_SUCCESS |
| Utility | -105.4400 | 96.7100 |
| API calls | 3 | 2 |
| ΔDG | 187.0847 | — |
| ΔIG | 15.0653 | — |
| ΔTR | 202.1500 | — |

**Blind trajectory:**

| Step | Proposed | Reason | Executed | Outcome | Terminal |
|------|----------|--------|----------|---------|----------|
| 0 | RETRIEVE | INITIAL_EVIDENCE_GATHERING | RETRIEVE | RETRIEVE_COMPLETED | False |
| 1 | VERIFY | EVIDENCE_RETRIEVED_NEEDS_VERIFICATION | VERIFY | VERIFY_COMPLETED | False |
| 2 | ANSWER | EVIDENCE_SUFFICIENT | ANSWER | TASK_FAILURE | True |

**Aware trajectory:**

| Step | Proposed | Reason | Executed | Outcome | Terminal |
|------|----------|--------|----------|---------|----------|
| 0 | VERIFY | INITIAL_VERIFICATION_REQUIRED | VERIFY | VERIFY_COMPLETED | False |
| 1 | ANSWER | EVIDENCE_SUFFICIENT | ANSWER | TASK_SUCCESS | True |

#### i3_3_development_0283

| Property | Blind | Aware |
|----------|-------|------|
| Success | False | True |
| Terminal | TASK_FAILURE | TASK_SUCCESS |
| Utility | -105.4400 | 96.7100 |
| API calls | 3 | 2 |
| ΔDG | 187.0847 | — |
| ΔIG | 15.0653 | — |
| ΔTR | 202.1500 | — |

**Blind trajectory:**

| Step | Proposed | Reason | Executed | Outcome | Terminal |
|------|----------|--------|----------|---------|----------|
| 0 | RETRIEVE | INITIAL_EVIDENCE_GATHERING | RETRIEVE | RETRIEVE_COMPLETED | False |
| 1 | VERIFY | EVIDENCE_RETRIEVED_NEEDS_VERIFICATION | VERIFY | VERIFY_COMPLETED | False |
| 2 | ANSWER | EVIDENCE_SUFFICIENT | ANSWER | TASK_FAILURE | True |

**Aware trajectory:**

| Step | Proposed | Reason | Executed | Outcome | Terminal |
|------|----------|--------|----------|---------|----------|
| 0 | VERIFY | UNRESOLVED_CONFLICT_PRESENT | VERIFY | VERIFY_COMPLETED | False |
| 1 | ANSWER | EVIDENCE_SUFFICIENT | ANSWER | TASK_SUCCESS | True |

#### i3_3_development_0295

| Property | Blind | Aware |
|----------|-------|------|
| Success | False | True |
| Terminal | TASK_FAILURE | TASK_SUCCESS |
| Utility | -105.4400 | 97.7400 |
| API calls | 3 | 2 |
| ΔDG | 188.1147 | — |
| ΔIG | 15.0653 | — |
| ΔTR | 203.1800 | — |

**Blind trajectory:**

| Step | Proposed | Reason | Executed | Outcome | Terminal |
|------|----------|--------|----------|---------|----------|
| 0 | RETRIEVE | INITIAL_EVIDENCE_GATHERING | RETRIEVE | RETRIEVE_COMPLETED | False |
| 1 | VERIFY | EVIDENCE_RETRIEVED_NEEDS_VERIFICATION | VERIFY | VERIFY_COMPLETED | False |
| 2 | ANSWER | EVIDENCE_SUFFICIENT | ANSWER | TASK_FAILURE | True |

**Aware trajectory:**

| Step | Proposed | Reason | Executed | Outcome | Terminal |
|------|----------|--------|----------|---------|----------|
| 0 | RETRIEVE | INITIAL_EVIDENCE_GATHERING | RETRIEVE | RETRIEVE_COMPLETED | False |
| 1 | ANSWER | EVIDENCE_SUFFICIENT | ANSWER | TASK_SUCCESS | True |


## 10. Data Integrity

| Check | Result |
|-------|--------|
| Total receipts | 600 (2 per pair) |
| Fingerprint-valid pairs | 300/300 |
| Decoder failures | 0 |
| Backend errors | 0 |
| All receipts have generation_config_sha256 | Yes |
| API key in any receipt | No |
| Condition labels leaked into model packet | No |

## 11. Scientific Interpretation

### What the data shows

1. **Absolute performance:** The aware condition outperforms blind on success rate (+6pp), mean utility (+13.4), and efficiency (20% fewer calls).

2. **ΔDG metric:** The mean ΔDG is -1.7132 with CI [-8.6006, 5.6090]. The ΔDG > 0 hypothesis is **not supported** on the development split.

3. **Why ΔDG is negative despite better absolute performance:** The observable oracle V_O is higher for the aware condition (65.38 vs 50.32). The decision gap DG = V_O - V_π measures the gap relative to the condition-specific observable optimum. Even though V_π_aware > V_π_blind in absolute terms, the aware condition's higher oracle means the relative gap can be larger.

4. **Behavioral differences:** The aware condition uses SEARCH_MORE (17 times) and REASON_MORE (24 times) — actions the blind condition never uses. This is direct evidence that cognitive state enables different executive decisions.

5. **Disagreement pattern:** 24 tasks where aware succeeded but blind failed, vs 6 reverse. This 4:1 ratio favors the aware condition.

### Status

- **Development split only** — no held-out data was accessed
- **Not a scientific claim** — development results are for pipeline validation
- **Hypothesis ΔDG > 0: not supported** on development split
- **Hypothesis "aware makes better absolute decisions": supported** by success rate, utility, and efficiency

### Next steps

1. Run validation split (150 tasks) to confirm the pattern
2. If validation confirms, run held-out structure split (150 tasks) for the scientific claim
3. Investigate whether the ΔDG metric or the observable oracle definition needs revisiting
4. The negative ΔDG despite better absolute performance may indicate that the observable oracle overestimates the aware condition's ceiling

---

*This report was generated by the DAPH V2B-I3.4.1 experiment runner. All API calls used `deepseek-v4-flash` with thinking disabled and JSON mode enabled. No API keys are present in any artifact.*
