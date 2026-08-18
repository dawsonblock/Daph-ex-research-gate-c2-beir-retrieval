# V2B-I3.5.2a State-Level Governor Competence Discovery

## Corrected Scientific Interpretation

> **Revision note:** This document has been revised to correct four scientific
> issues identified in review: (1) the distinction between governor ranking
> and packet treatment, (2) fold-isolated rule discovery, (3) Brier score
> calibration, and (4) Q-value source tracking.

---

## Executive Summary

V2B-I3.5.1 showed that the unconditional ("always-on") governor is harmful at
the trajectory level ($\Delta\text{DG}_{\text{gov}|\text{aware}} = -14.60$).

V2B-I3.5.2a asks: **did the governor fail everywhere, or does it possess local
regions of competence whose gains were masked by catastrophic early
interventions?**

By computing exact oracle $Q(s, a)$ values across all 758 baseline decision
states in the 300 development tasks, we discovered:

> **The governor contains a strong, identifiable local action-ranking
> competence region that is obscured by a smaller but severe hazard region.**

Specifically:

- **26.0%** of baseline states (197/758) have $\Delta Q_{\text{gov-top}} > +5.0$.
- **64.3%** of tasks (193/300) contain at least one positive intervention state.
- The positive region is concentrated at Step 2+ after `VERIFY` and
  `SEARCH_MORE`, where the governor prevents premature failing answers.
- The hazard region is concentrated at Step 0, where the governor overrides
  valid `STOP` or forces premature `VERIFY`.

**However, this measures governor ranking intelligence
($Q(s, a_{\text{gov-top}}) > Q(s, a_{\text{base}})$), NOT packet treatment
benefit ($Q(s, a_{\text{model|gov-packet}}) > Q(s, a_{\text{model|base-packet}})$).**

The packet treatment experiment (I3.5.2b) is required to determine whether the
model can actually exploit the governor's ranking intelligence when exposed to
the governor packet.

---

## Corrected Estimand Distinction

```text
WHAT I3.5.2a MEASURES (governor ranking):

  ΔQ_gov-top(s) = Q(s, a_gov-top) - Q(s, a_base)

  This tells us: "Does the governor KNOW a better action?"
  This does NOT tell us: "Can the model USE that knowledge?"

WHAT I3.5.2b WILL MEASURE (packet treatment):

  ΔQ_packet(s) = Q(s, a_model|gov-packet) - Q(s, a_model|base-packet)

  This tells us: "Does exposing the model to the governor packet
  actually improve the model's action choice?"

DECOMPOSITION:

  A_ranking     = Q(s, a_gov-top)           - Q(s, a_base)    [governor intelligence]
  A_treatment   = Q(s, a_model|gov-packet)  - Q(s, a_base)    [packet effect]
  A_realization = A_treatment - A_ranking                      [model conversion]
```

---

## 1. State-Level Counterfactual Q-Advantage Dataset

Script: `scripts/build_i3_5_2_shadow_dataset.py`
Output Artifacts:
- `experiments/v2b_i3_5_2/development/intervention_states_v1.jsonl` (758 state records)
- `experiments/v2b_i3_5_2/development/intervention_advantage_v1.json` (substitution & advantage analysis)
- `experiments/v2b_i3_5_2/development/intervention_feature_analysis_v1.json` (feature opportunity map)

### Q-Value Source Tracking

Every Q-value lookup now records its source:

| Source | Count | Percentage | Description |
|---|---|---|---|
| `oracle_q_values` | 1340 | 88.4% | Primary oracle transition table |
| `fallback_penalty` | 176 | 11.6% | Fixed penalty for actions not in oracle |
| `proposal_q_values` | 0 | 0.0% | Proposal transition table |

Per-state breakdown:
- Both $Q(s, a_{\text{base}})$ and $Q(s, a_{\text{gov}})$ from oracle: **582/758 (76.8%)**
- $Q(s, a_{\text{base}})$ from fallback: **176/758 (23.2%)** — all `ANSWER` actions
- $Q(s, a_{\text{gov}})$ from fallback: **0/758 (0.0%)**

The 176 fallback cases are all baseline `ANSWER` actions at states where the
oracle does not contain an `ANSWER` Q-value. The fallback penalty of $-125.11$
represents the standard penalty for an incorrect answer. These are cases where
the model answers at a state where answering is not the oracle-defined action.

### Overall State-Level Advantage Breakdown

| Outcome Category | Threshold | Decision States ($N=758$) | Percentage |
|---|---|---|---|
| **HELP** | $\Delta Q_{\text{gov-top}} > +5.0$ | **197** | **26.0%** |
| **NEUTRAL** | $-5.0 \le \Delta Q_{\text{gov-top}} \le +5.0$ | **490** | **64.6%** |
| **HARM** | $\Delta Q_{\text{gov-top}} < -5.0$ | **71** | **9.4%** |

- **Tasks with at least one helpful intervention step:** **193 / 300 (64.3%)**
- **Tasks with at least one harmful intervention step:** 71 / 300 (23.7%)

---

## 2. Action Substitution Matrix (Governor Ranking)

Comparing baseline unaided model action $a_{\text{base}}(s_t)$ vs. governor
recommendation $a_{\text{gov-top}}(s_t)$ at the exact same controller state:

| Baseline $\to$ Governor | Count | % of States | Mean $\Delta Q$ | Causal Effect |
|---|---|---|---|---|
| `VERIFY -> VERIFY` | 197 | 26.0% | 0.00 | Complete agreement |
| `RETRIEVE -> RETRIEVE` | 135 | 17.8% | 0.00 | Complete agreement |
| `ANSWER -> SEARCH_MORE` | 134 | 17.7% | **+886.73** | Governor prevents fatal premature answer |
| `ANSWER -> ANSWER` | 65 | 8.6% | 0.00 | Complete agreement |
| `ANSWER -> VERIFY` | 64 | 8.4% | **+872.20** | Governor prevents fatal premature answer |
| `SEARCH_MORE -> SEARCH_MORE` | 63 | 8.3% | 0.00 | Complete agreement |
| `RETRIEVE -> VERIFY` | 62 | 8.2% | **-50.83** | Premature verification at Step 0 |
| `STOP -> ANSWER` | 35 | 4.6% | **-120.00** | Overrides valid STOP at Step 0 |
| `ANSWER -> REASON_MORE` | 2 | 0.3% | **+967.75** | Governor prevents fatal premature answer |

---

## 3. Intervention Opportunity Map

### Slice by Prior Action Count

| Prior Actions | States ($N$) | Mean $\Delta Q$ | Help Rate | Harm Rate | Policy |
|---|---|---|---|---|---|
| **0 (Step 0)** | 300 (39.6%) | **-22.59** | 2.3% | **21.3%** | **SKIP** |
| **1 (Step 1)** | 197 (26.0%) | **0.00** | 0.0% | 0.0% | **SKIP** |
| **2 (Step 2)** | 197 (26.0%) | **+83.55** | **68.0%** | **0.0%** | **INTERVENE** |
| **3 (Step 3)** | 63 (8.3%) | **+86.82** | **87.3%** | 11.1% | **INTERVENE** |
| **4 (Step 4)** | 1 (0.1%) | **+91.82** | **100.0%** | 0.0% | **INTERVENE** |

### Slice by Verification State

| Verification State | States ($N$) | Mean $\Delta Q$ | Help Rate | Harm Rate | Diagnosis |
|---|---|---|---|---|---|
| `FALSIFIED` | 68 (9.0%) | **+42.13** | **42.6%** | **0.0%** | Governor prevents giving up |
| `MISSING` | 578 (76.2%) | **+26.69** | **28.0%** | 6.2% | High value post-step 1 |
| `SUFFICIENT` | 112 (14.8%) | **-27.22** | 5.4% | **31.2%** | Step 0 STOP override hazard |

---

## 4. Cross-Validation: Fold-Isolated vs Global Rules

Script: `scripts/train_and_validate_intervention_gate.py`
Outputs:
- `experiments/v2b_i3_5_2/development/cross_validation_report_v1.json` (fold-isolated)
- `experiments/v2b_i3_5_2/development/cross_validation_report_global_v1.json` (global rules)

### Scientific Issue: Rule Discovery Independence

The original CV used rules discovered from the full 758-state dataset, then
evaluated them via task-grouped folds. This is **not fully out-of-fold** because
rule discovery was not fold-isolated.

The corrected CV offers two modes:

1. **`fold_isolated`** (scientifically honest): Rules are discovered from
   training data only within each fold. This gives an unbiased generalization
   estimate.
2. **`global`** (for comparison only): Uses pre-defined rules discovered from
   the full dataset. This is NOT fold-isolated and overestimates performance.

### Fold-Isolated Rule Discovery Results

Every fold independently discovered the same two core positive regions:

| Rule | Discovered in Folds | Train N | Help Rate | Harm Rate | Mean $\Delta Q$ |
|---|---|---|---|---|---|
| `pac>=2, last=VERIFY, verif=MISSING` | 5/5 | 129-137 | 76-81% | 0.0% | 89-99 |
| `pac>=3, last=SEARCH_MORE, verif=FALSIFIED` | 5/5 | 17-25 | 100% | 0.0% | 92-104 |
| `pac>=2, last=VERIFY, verif=SUFFICIENT` | 3/5 | 3 | 100% | 0.0% | 193 |

The stability of the core rules across all 5 folds is strong evidence that the
positive intervention regions are real and reproducible, not artifacts of
overfitting to the full dataset.

### Out-of-Fold Performance Comparison

| Metric | Fold-Isolated | Global Rules | Interpretation |
|---|---|---|---|
| **Intervention Rate** | 52.1% (395/758) | 33.9% (257/758) | Fold-isolated is more permissive |
| **Precision (HELP\|INTERVENE)** | **48.1%** | 72.4% | Global overfits; fold-isolated is honest |
| **Harm Rate on INTERVENE** | **1.8%** | 2.7% | Both suppress harm effectively |
| **$E[\Delta Q \mid \text{INTERVENE}]$** | **+55.75** | +83.07 | Fold-isolated gain is lower but honest |
| **Worst-Decile $\Delta Q$** | 0.00 | 0.00 | Zero tail risk in both |
| **Spearman Correlation** | 0.5950 | 0.6298 | Similar ranking quality |
| **ROCAUC $P(\text{HARM})$** | 0.7629 | 0.8381 | Good discrimination in both |

### Brier Score Calibration (Corrected)

The original report claimed Brier = 0.2613 was "well calibrated." This was
incorrect. With harm prevalence at 9.4%, the base-rate Brier is 0.0849. A Brier
of 0.2613 is **worse than the base rate**, indicating poor calibration.

| Metric | Raw | Calibrated (Isotonic) | Base Rate |
|---|---|---|---|
| **Brier Score** | 0.2623 | **0.0818** | 0.0849 |
| **Brier vs Base Rate** | +0.1774 (worse) | **-0.0031 (better)** | 0.0 |
| **ECE** | 0.2807 | **0.0546** | — |

The isotonic calibration fixes the probability estimates: calibrated Brier
(0.0818) now beats the base rate (0.0849), and ECE drops from 0.28 to 0.05.

**Key insight:** The harm classifier has good discrimination (ROCAUC 0.76-0.84)
but poor raw probability calibration. The calibrated probabilities are suitable
for use as a scientific gate ($P(\text{HARM}) < 0.15$).

---

## 5. Architecture: The Selective Intervention Gate

```text
                        ControllerObservation (State s_t)
                                      │
                                      ▼
                           Intervention Gate
                                      │
                  ┌───────────────────┴───────────────────┐
                  ▼                                       ▼
            SKIP (< 5.0 ΔQ)                       INTERVENE (>= 5.0 ΔQ)
                  │                                       │
                  ▼                                       ▼
             Base Packet                           Governor Engine
                  │                                       │
                  ▼                                       ▼
             Base Model                            Governor Packet
                  │                                       │
                  ▼                                       ▼
             Action a_base                         Model chooses action
                                                    (governor is advisory)
```

**Important:** The governor is advisory. Its `governor_top_action` is diagnostic
only. The model makes the final action choice. Whether the model can actually
exploit the governor's ranking intelligence is the I3.5.2b question.

---

## 6. Corrected Milestone Status

```text
V2B-I3.5.2a
STATE-LEVEL GOVERNOR COMPETENCE DISCOVERY

  Oracle governor-ranking competence:     SUPPORTED
  Local positive competence region:       SUPPORTED
  Local hazard region:                    SUPPORTED
  Task-group state leakage:               CONTROLLED
  Fold-isolated rule stability:           SUPPORTED (same rules in 5/5 folds)
  Probability calibration:                REPAIRED (isotonic, Brier < base rate)
  CV rule-discovery independence:         ESTABLISHED (fold-isolated mode)
  Q-value source tracking:                IMPLEMENTED (88.4% oracle, 11.6% fallback)
  Packet-level causal benefit:            NOT YET MEASURED (I3.5.2b required)
  Selective end-to-end improvement:       NOT YET ESTABLISHED
```

### What is Supported

1. The governor's action ranking contains real intelligence: 26% of states
   have $\Delta Q_{\text{gov-top}} > +5.0$.
2. The positive region is structurally separable from the hazard region:
   Step 2+ post-`VERIFY`/`SEARCH_MORE` vs Step 0.
3. The positive regions are reproducible: the same rules are discovered
   independently in all 5 CV folds.
4. A conservative gate can suppress harm to ~1.8% while approving ~52% of
   states with ~48% HELP precision (fold-isolated honest estimate).
5. Probability calibration via isotonic regression produces Brier scores
   below the base rate.

### What is NOT Yet Supported

1. **Packet treatment benefit:** We have not yet measured whether the model
   produces better actions when exposed to the governor packet. The 26% HELP
   rate measures governor ranking, not model behavior change.
2. **End-to-end selective improvement:** No selective trajectory run has been
   completed. The state-level $\Delta Q$ does not account for sequential
   effects of allowing the governor to alter the trajectory.
3. **Cost-adjusted utility:** The cost of governor-packet model calls (extra
   tokens, latency) has not been evaluated against the benefit.

---

## 7. Next Step: V2B-I3.5.2b Packet Treatment Experiment

Script: `scripts/build_i3_5_2_packet_counterfactual_dataset.py`

For every baseline state $s_t$:

```text
                         identical state s_t
                               │
             ┌─────────────────┴─────────────────┐
             ▼                                   ▼
       BASE packet                         GOVERNOR packet
             │                                   │
             ▼                                   ▼
         DeepSeek                              DeepSeek
             │                                   │
             ▼                                   ▼
        a_base_model                         a_gov_model
             │                                   │
             └─────────────────┬─────────────────┘
                               ▼
  ΔQ_packet(s) = Q(s, a_gov_model) - Q(s, a_base_model)
```

The counterfactual governor-packet action is **never executed**. The trajectory
continues with the recorded baseline action.

### Decomposition

$$A_{\text{ranking}} = Q(s, a_{\text{gov-top}}) - Q(s, a_{\text{base}})$$
$$A_{\text{treatment}} = Q(s, a_{\text{model|gov-packet}}) - Q(s, a_{\text{base}})$$
$$A_{\text{realization}} = A_{\text{treatment}} - A_{\text{ranking}}$$

- $A_{\text{ranking}} > 0$: Governor knows a better action.
- $A_{\text{treatment}} > 0$: Model actually chooses a better action when given governor info.
- $A_{\text{realization}}$: How well the model converts governor intelligence into behavior.

### Status

- Script: **COMPLETE** (`scripts/build_i3_5_2_packet_counterfactual_dataset.py`)
- Dry-run verification: **PASSED** (3 tasks, 8 states, mechanics verified)
- Full run: **PENDING** (requires `DEEPSEEK_API_KEY`, ~758 model calls)

### Experimental Arms for Selective Comparison

The packet treatment result justifies adding selective arms:

| Arm | Description | Hypothesis |
|---|---|---|
| `OFF` | Base packet, no governor | Baseline |
| `ALWAYS_ON` | Governor packet always | $H_A$: model can exploit governor info indiscriminately |
| `SELECTIVE_FRAME` | Gate approves $\to$ governor advisory packet $\to$ model chooses | $H_A$: model can exploit governor info selectively |
| `SELECTIVE_DIRECT` | Gate approves $\to$ execute `governor_top_action` directly | $H_D$: governor ranking itself contains useful control intelligence |

I3.5.2a gives development evidence for $H_D$ (governor ranking competence).
I3.5.1 gives negative evidence for indiscriminate $H_A$.
The selective versions are the interesting next experiment.

---

## 8. Scientific Caveats

1. **These are state-level counterfactual results (governor ranking), NOT
   packet-level treatment results.** The packet treatment experiment (I3.5.2b)
   is required to measure whether the model can actually exploit governor
   information.

2. **The fold-isolated CV precision (48.1%) is the honest generalization
   estimate.** The global-rules precision (72.4%) overestimates performance
   because the rules were not discovered fold-isolated.

3. **The Brier score is now calibrated** via isotonic regression. The raw Brier
   (0.2623) was worse than the base rate (0.0849). The calibrated Brier (0.0818)
   beats the base rate.

4. **Q-value source tracking** shows 88.4% of Q-values come from the oracle
   table. 11.6% use fixed fallback penalties (all for baseline `ANSWER` actions
   not present in the oracle). The fallback penalty of $-125.11$ represents the
   standard incorrect-answer penalty.

5. **End-to-end selective trajectory improvement has NOT been demonstrated.**
   The state-level $\Delta Q$ does not account for sequential effects of
   allowing the governor to alter the trajectory.

6. **The predictor must not be treated as frozen validation-ready policy** until:
   - Feature schema is frozen
   - Model class is frozen
   - Predictor parameters/rules are frozen
   - Thresholds are frozen
   - Gate identity hash is frozen
   - Packet treatment experiment (I3.5.2b) is completed
   - End-to-end development comparison is run
   - Validation is run without further tuning
