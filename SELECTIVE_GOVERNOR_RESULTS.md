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
V2B-I3.5.x FINAL DEVELOPMENT STATUS
I3.5.1 always-on governor:
  REJECTED / HARMFUL
I3.5.2a oracle ranking competence:
  SUPPORTED
I3.5.2b packet transmission:
  SUPPORTED
I3.5.2c Q*-selective trajectory improvement:
  REJECTED
I3.5.2d Qπ realizability:
  NOT SUPPORTED FOR CURRENT GOVERNOR
I3.5.3 original Qπ surrogate:
  HISTORICAL / SPECIFICATION DEFECTS
I3.5.3-r1 base-first pairwise gate:
  VALID MECHANISM
  PRIMARY IMPROVEMENT NOT SUPPORTED
I3.5.3-r2/r2.1 closure:
  FROZEN NEGATIVE DEVELOPMENT RESULT

Frozen production criterion:
  threshold = 5
  LCB margin = 5
  effective predicted advantage requirement > 10

Runtime replay (full precision, 303 disagreements):
  max predicted ΔQπ = +3.179887
  max LCB = -1.820113
  predicted > 0: 68/303 (22.4%)
  predicted > 5: 0/303 (0.0%)
  interventions = 0

Permissive diagnostic (τ=0, margin=0):
  approved = 68 (== predicted_positive, invariant PASS)
  weak positive opportunities exist (max +3.18)
  these are not validation candidates

Runtime/training overlap:
  295/303 runtime states overlap training corpus
  66/68 positive predictions seen in training
  2/68 positive predictions are OOD

Fork dataset (observed ΔQπ):
  300 disagreement states across 235 tasks
  max observed ΔQπ = +5.34
  0 governor-continuation successes vs 42 base-continuation
  small local utility improvement ≠ task rescue

Validation and held-out remain unopened for this mechanism
because the primary development improvement criterion was not met.
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

1. **End-to-end selective improvement:** No selective trajectory run has been
   completed. The state-level $\Delta Q$ does not account for sequential
   effects of allowing the governor to alter the trajectory.
2. **Cost-adjusted utility:** The cost of governor-packet model calls (extra
   tokens, latency) has not been evaluated against the benefit.

---

## 7. V2B-I3.5.2b Packet Treatment Experiment — COMPLETED

Script: `scripts/build_i3_5_2_packet_counterfactual_dataset.py`
Output Artifacts:
- `experiments/v2b_i3_5_2/development/packet_counterfactual_states_v1.jsonl` (758 records)
- `experiments/v2b_i3_5_2/development/packet_counterfactual_summary_v1.json`

### Experimental Design

For every baseline state $s_t$ on the AWARE_NO_GOVERNOR trajectory:

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
continues with the recorded baseline action. 758 model calls, 0 backend errors.

### Decomposition

$$A_{\text{ranking}} = Q(s, a_{\text{gov-top}}) - Q(s, a_{\text{base}})$$
$$A_{\text{treatment}} = Q(s, a_{\text{model|gov-packet}}) - Q(s, a_{\text{base}})$$
$$A_{\text{realization}} = A_{\text{treatment}} - A_{\text{ranking}}$$

### Key Result: The Model is a Near-Perfect Conduit for Governor Intelligence

| Metric | Value | Interpretation |
|---|---|---|
| **Model follows governor top** | **743/758 (98.0%)** | Model almost always follows governor recommendation |
| **$A_{\text{ranking}}$ mean** | +20.11 | Governor ranking intelligence |
| **$A_{\text{treatment}}$ mean** | +21.52 | Packet treatment effect |
| **$A_{\text{realization}}$ mean** | **+1.41** | Model slightly IMPROVES on governor ranking |
| **HELP: ranking vs treatment** | 197 vs 197 | Identical — zero positive intelligence lost |
| **HARM: ranking vs treatment** | 71 vs 60 | Treatment has FEWER harms — model refuses some bad advice |

### The 13 Disagreements: Model Refuses Harmful Governor Recommendations

There are only 13 states where $|A_{\text{treatment}} - A_{\text{ranking}}| > 5.0$.
All 11 meaningful disagreements are cases where the governor recommends
`STOP -> ANSWER` (the catastrophic Step 0 hazard) but **the model refuses and
keeps `STOP`**, converting a $-120.0$ HARM into a $0.0$ NEUTRAL.

| Pattern | Count | $A_{\text{ranking}}$ | $A_{\text{treatment}}$ | Effect |
|---|---|---|---|---|
| Governor says `ANSWER`, model keeps `STOP` | 11 | $-120.0$ | $0.0$ | Model refuses harmful override |
| Other small disagreements | 2 | varies | varies | Negligible |

**Zero cases where ranking says HELP but treatment doesn't.** Every time the
governor knows a better action, the model follows it. 100% transmission of
positive intelligence.

### Packet-Treatment Substitution Matrix

| Baseline $\to$ Packet Model | Count | Mean $\Delta Q_{\text{pkt}}$ | HELP | HARM |
|---|---|---|---|---|
| `VERIFY -> VERIFY` | 197 | 0.00 | 0 | 0 |
| `ANSWER -> SEARCH_MORE` | 138 | **+121.96** | **138** | 0 |
| `RETRIEVE -> RETRIEVE` | 135 | 0.00 | 0 | 0 |
| `ANSWER -> ANSWER` | 65 | 0.00 | 0 | 0 |
| `SEARCH_MORE -> SEARCH_MORE` | 63 | 0.00 | 0 | 0 |
| `RETRIEVE -> VERIFY` | 62 | **-50.83** | 4 | 29 |
| `ANSWER -> VERIFY` | 60 | **+88.82** | 53 | 7 |
| `STOP -> ANSWER` | 24 | **-120.00** | 0 | 24 |

Note: `STOP -> ANSWER` drops from 35 (governor ranking) to 24 (packet treatment)
because the model refuses 11 of the governor's harmful `ANSWER` recommendations.

### Scientific Interpretation

This is strong evidence for $H_A$ (the advisory architecture hypothesis):

> **The model CAN exploit governor information.** When exposed to the governor
> packet, the model follows the governor's recommendation 98% of the time,
> and the packet treatment effect ($A_{\text{treatment}} = +21.52$) is
> essentially identical to the governor ranking effect ($A_{\text{ranking}} = +20.11$).

The model even slightly improves on the governor by refusing 11 of 35
harmful `STOP -> ANSWER` overrides at Step 0.

This means:
1. The governor's ranking competence **survives the model/governor interface**.
2. A selective gate that approves intervention only in the positive competence
   region should transmit the full benefit to the model.
3. The `SELECTIVE_FRAME` arm (gate approves $\to$ governor advisory packet
   $\to$ model chooses) is a viable architecture.

### What This Does NOT Prove

This is still state-level counterfactual analysis. The counterfactual action
was never executed, so we have not measured:
1. **Sequential effects:** What happens when the model takes the governor's
   recommended action and the trajectory diverges from baseline?
2. **End-to-end utility:** Does the selective policy actually improve task
   success rate and realized utility?
3. **Cost:** The governor packet is larger than the base packet, costing more
   tokens per call.

### Experimental Arms for Selective Comparison

| Arm | Description | Hypothesis |
|---|---|---|
| `OFF` | Base packet, no governor | Baseline |
| `ALWAYS_ON` | Governor packet always | $H_A$: model can exploit governor info indiscriminately |
| `SELECTIVE_FRAME` | Gate approves $\to$ governor advisory packet $\to$ model chooses | $H_A$: model can exploit governor info selectively |

I3.5.2a gives development evidence for $H_D$ (governor ranking competence).
I3.5.2b confirms $H_A$ at the state level (model follows governor 98% of the time).
The selective end-to-end trajectory run is the next experiment.

---

## 8. V2B-I3.5.2c End-to-End Selective Governor Trajectory Experiment

> **Revision r1:** This section has been corrected from the original report.
> Three issues were fixed: (1) the DG definition was using binary success as a
> proxy instead of the frozen $DG = V_O - V_\pi$ definition; (2) the utility
> loss was incorrectly attributed to model token overhead instead of executor
> trajectory costs; (3) the causal explanation was stated as "already doomed
> at Step 0" without evidence, when the more precise explanation is the
> $Q^* \neq Q^{\pi_{\text{model}}}$ distinction.

Script: `scripts/run_v2b_i3_5_2c_experiment.py`
Output: `experiments/v2b_i3_5_2/development/i352c_55f93130e87c/`
Corrected analysis: `experiments/v2b_i3_5_2/development/i352c_55f93130e87c/analysis_r1.json`

### Experimental Design

Three arms run on all 300 `structure_dev_v2` development tasks:

| Arm | Description |
|---|---|
| `OFF` | Base packet, no governor |
| `ALWAYS_ON` | Governor packet always injected |
| `SELECTIVE_FRAME` | Gate approves → governor advisory packet → model chooses |

**Counterbalancing:** Arm ordering is deterministically counterbalanced using
`HMAC(seed, task_id) % 6` across the 6 permutations of the three arms. This
eliminates temporal/order confounds against the remote model backend.

**Experiment identity:** Full component hashes bound:
gate identity, system prompt, packet builder, governor assessor, governor
serializer, utility, benchmark manifest, runner, modes, source commit.

### Primary Results

| Metric | OFF | ALWAYS_ON | SELECTIVE_FRAME |
|---|---|---|---|
| **Terminal success** | **83/300 (27.7%)** | 60/300 (20.0%) | **83/300 (27.7%)** |
| **Mean utility $V_\pi$** | **-74.90** | -90.22 | -78.18 |
| **Mean executor steps** | 2.5 | 5.1 | 3.9 |
| **Mean model calls** | 2.5 | 5.1 | 3.9 |
| **Mean model tokens** | 2,523 | 9,693 | 5,623 |

### Corrected Decision Degradation

The frozen I3.5.1 DG definition is:

$$DG = V_O - V_\pi$$

where $V_O$ is the optimal controller value and $V_\pi$ is the realized
utility under policy $\pi$.

Since OFF and SELECTIVE use the same AWARE observation condition, $V_O$
cancels in the contrast:

$$\Delta DG_S = DG_{\text{OFF}} - DG_{\text{SEL}} = V_{\pi,\text{SEL}} - V_{\pi,\text{OFF}}$$

This is identical to the utility contrast $\Delta U_S$.

| Quantity | Value | 95% CI |
|---|---|---|
| $V_{\pi,\text{OFF}}$ | -74.90 | — |
| $V_{\pi,\text{ALWAYS}}$ | -90.22 | — |
| $V_{\pi,\text{SEL}}$ | -78.18 | — |
| **$\Delta DG_S = V_{\pi,\text{SEL}} - V_{\pi,\text{OFF}}$** | **-3.2814** | **[-3.5848, -2.9847]** |
| $\Delta DG_A = V_{\pi,\text{ALWAYS}} - V_{\pi,\text{OFF}}$ | -15.3213 | [-18.9030, -11.9931] |
| $\Delta\text{Success}$ (terminal) | 0.0000 | [0, 0] |

### Hypothesis Tests

| Hypothesis | Result | Detail |
|---|---|---|
| $H_1$: $\Delta DG_S > 0$ (continuous) | **NOT SUPPORTED** | $\Delta DG = -3.28$, LCB $= -3.58$, direction is **harmful** |
| $H_2$: $\Delta U_S > 0$ | **NOT SUPPORTED** | $\Delta U = -3.28$ (same as $\Delta DG$) |
| $H_3$: $U_S > U_A$ | **SUPPORTED** | $-78.18 > -90.22$ |
| Terminal success preserved | **SUPPORTED** | 83/300 = 83/300, zero discordant pairs |
| Ideal ordering $U_S > U_0 > U_A$ | **NOT ACHIEVED** | $U_0 > U_S > U_A$ (SELECTIVE is between, not above OFF) |

### McNemar's Test

| Comparison | off_only | other_only | Discordant |
|---|---|---|---|
| OFF vs SELECTIVE | 0 | 0 | 0 |
| OFF vs ALWAYS_ON | 23 | 0 | 23 |

**Zero discordant pairs between OFF and SELECTIVE.** Every task that OFF
succeeds on, SELECTIVE also succeeds on, and vice versa. The 23 discordant
pairs for ALWAYS_ON are all cases where OFF succeeds but ALWAYS_ON fails.

### Intervention Statistics

- Total interventions: 536
- Tasks with at least one intervention: 194/300 (64.7%)
- Intervention rate per step: 45.3%
- Rule firing: POST_VERIFY 268, POST_SEARCH 268 (evenly split)
- Interventions produced zero net terminal-success conversions

### Cascade Diagnostics

| Chain length | Count |
|---|---|
| 2 | 120 |
| 4 | 74 |

Max consecutive interventions: 4. No runaway cascades.

The 2/4 chain structure with equal POST_VERIFY/POST_SEARCH counts suggests
structured continuation loops (VERIFY → SEARCH_MORE → VERIFY → ...) rather
than isolated interventions. This pattern is analyzed further in I3.5.2d.

### Cost Accounting

| Metric | OFF | ALWAYS_ON | SELECTIVE |
|---|---|---|---|
| Mean executor steps | 2.5 | 5.1 | 3.9 |
| Mean model calls | 2.5 | 5.1 | 3.9 |
| Mean model tokens | 2,523 | 9,693 | 5,623 |
| Mean utility | -74.90 | -90.22 | -78.18 |

**The utility loss comes from longer/costlier executor trajectories**
(+1.4 executor steps per task), NOT from model token overhead. The
`MetareasoningUtility` charges simulated executive/retrieval/verification/
search/reasoning resource consumption. Model prompt/completion tokens are
recorded as telemetry but are not directly charged by the utility function.

The SELECTIVE arm also consumes ~3,100 more model tokens per task than OFF,
but that is a separate operational cost, not the cause of the utility loss.

### Corrected Scientific Interpretation

**SELECTIVE_FRAME preserves terminal success but significantly worsens
continuous decision quality/value relative to OFF.**

The selective gate successfully prevents the terminal-success harm caused by
ALWAYS_ON (83/300 vs 60/300, zero discordant pairs with OFF). However,
SELECTIVE_FRAME does not improve over OFF on either terminal success or
continuous value:

- Terminal success: identical (83/300 = 83/300)
- Continuous DG: $\Delta DG = -3.28$, LCB $= -3.58$ (harmful)
- 536 interventions in 194 tasks produced zero net terminal-success conversions

### The Root Cause: $Q^* \neq Q^{\pi_{\text{model}}}$

The most precise explanation for the I3.5.2c negative result is the
distinction between oracle-optimal continuation value and model continuation
value.

The I3.5.2a state-level analysis used $Q^*(s, a)$, the oracle Q-value computed
by backwards dynamic programming:

$$Q^*(s, a) = r(s, a) + V^*(s')$$

where $V^*(s')$ assumes **optimal continuation** from the next state.

When I3.5.2a found $Q^*(s, a_G) > Q^*(s, a_B)$, it established:

> Taking the governor's action is better **if an optimal policy takes over
> afterward**.

It did **not** establish:

$$Q^{\pi_{\text{model}}}(s, a_G) > Q^{\pi_{\text{model}}}(s, a_B)$$

The I3.5.2c result shows exactly this gap:

```text
Governor recommends locally oracle-optimal continuation action
                    ↓
DeepSeek follows it (98% transmission, I3.5.2b)
                    ↓
new state
                    ↓
DeepSeek remains DeepSeek, not oracle
                    ↓
oracle continuation value never realized
```

The governor identifies actions that are better under optimal continuation,
but the model does not continue optimally after the intervention, so the
oracle advantage is not realized.

### Corrected Causal Chain

```text
I3.5.1:  Always-on governor damages the model policy           CONFIRMED
I3.5.2a: Governor sometimes selects actions with higher Q*     CONFIRMED
I3.5.2b: Model faithfully follows those recommendations         CONFIRMED
I3.5.2c: Higher Q* actions do NOT improve value under
         actual downstream model policy                         CONFIRMED
```

The missing link is:

$$\boxed{Q^* \neq Q^{\pi_{\text{model}}}}$$

This is the main discovery of I3.5.2c.

### What This Does NOT Mean

1. **This does not disprove governor competence.** The state-level analysis
   (I3.5.2a) and packet treatment (I3.5.2b) are valid. The governor does know
   oracle-better actions at specific states.

2. **This does not mean the architecture is wrong.** SELECTIVE_FRAME is
   strictly better than ALWAYS_ON. The gate successfully filters harmful
   interventions and preserves terminal success.

3. **This does not mean the model can't use governor information.** The 98%
   follow rate in I3.5.2b is real. The model does exploit governor information
   when given it.

4. **"The trajectory was already doomed at Step 0" is one candidate
   explanation, not yet established.** The rescueability test in I3.5.2d will
   determine whether intervened tasks are genuinely unrecoverable or whether
   the problem is downstream policy execution.

### Development Acceptance Gates

| Gate | Description | Result |
|---|---|---|
| G1: Validity | Receipt chain valid, all 3 arms complete | **PASS** |
| G2: Nontrivial intervention | intervention_rate > 0 | **PASS** (536 interventions) |
| G3: Primary DG (continuous) | $\Delta DG > 0$ | **FAIL** (-3.2814, LCB=-3.5848) |
| G4: Primary utility | $\Delta U > 0$ | **FAIL** (-3.2814, same as $\Delta DG$) |
| G5: Always-on dominance | $U_S > U_A$ | **PASS** (-78.18 > -90.22) |
| G6: No catastrophic terminal harm | off_only ≤ sel_only | **PASS** (0 ≤ 0) |
| G7: Sequential stability | max_consecutive ≤ 5 | **PASS** (4) |

**5 of 7 gates passed.** The two primary hypothesis gates failed. The safety
gates all passed. The continuous DG gate fails with a harmful direction, not
merely a neutral direction.

### Validation Status

```
VALIDATION = STOP
HELD-OUT   = DO NOT TOUCH
```

The primary development hypothesis failed. The corrected continuous DG is
negative. The scientific gate has correctly halted progression to validation.

### Next Step: I3.5.2d — Policy-Conditional Intervention Value

The next milestone measures the correct estimand:

$$A^{\pi_B} = Q^{\pi_B}(s, a_G) - Q^{\pi_B}(s, a_B)$$

where $\pi_B$ is the actual OFF model policy. This is the value of a single
selective intervention under the actual downstream policy, not under the
oracle.

Three intervention-value quantities will be computed:

| Quantity | Definition | What it tells us |
|---|---|---|
| $A^*$ | $Q^*(s, a_G) - Q^*(s, a_B)$ | Oracle advantage (existing, I3.5.2a) |
| $A^{\pi_B}$ | $Q^{\pi_B}(s, a_G) - Q^{\pi_B}(s, a_B)$ | One intervention + base-model continuation |
| $A^{\pi_G}$ | $Q^{\pi_G}(s, a_G) - Q^{\pi_B}(s, a_B)$ | Governor-controlled continuation |

This will determine exactly where value disappears:
- If $A^* > 0$ but $A^{\pi_B} \approx 0$: the model can't continue the path
  the governor opens.
- If $A^* > 0$ and $A^{\pi_B} > 0$ but $A^{\pi_G} > A^{\pi_B}$: persistent
  governor control may be necessary.
- If $A^* > 0$ but $A^{\pi_B} = A^{\pi_G} = 0$: the oracle advantage isn't
  behaviorally realizable by either policy.

A rescueability classification will also test whether intervened tasks are
genuinely unrecoverable or whether the problem is downstream policy execution.

---

## 9. V2B-I3.5.2d Policy-Conditional Intervention Value — COMPLETED

Script: `scripts/run_v2b_i3_5_2d_experiment.py`
Output: `experiments/v2b_i3_5_2/development/i352d/`

### Experimental Design

For every intervention state from I3.5.2c where the OFF trajectory has a
corresponding step (260 states across 194 tasks):

1. Replay the baseline trajectory up to state $s$
2. **Fork A:** Execute baseline action $a_B$, continue with OFF model → terminal
3. **Fork B:** Execute governor action $a_G$, continue with OFF model → terminal
4. **Fork C:** Execute governor action $a_G$, continue with SELECTIVE model → terminal
5. Record realized utilities for all three forks

Three intervention-value quantities:

| Quantity | Definition | What it measures |
|---|---|---|
| $A^*$ | $Q^*(s, a_G) - Q^*(s, a_B)$ | Oracle advantage (I3.5.2a) |
| $A^{\pi_B}$ | $U(a_G + \pi_B) - U(a_B + \pi_B)$ | One intervention + base-model continuation |
| $A^{\pi_G}$ | $U(a_G + \pi_G) - U(a_B + \pi_B)$ | Governor-controlled continuation |

### Key Result: The Oracle Advantage Is Not Behaviorally Realizable

| Quantity | Mean | Interpretation |
|---|---|---|
| $A^*$ (oracle) | **+108.41** | Governor knows a much better action |
| $A^{\pi_B}$ (base continuation) | **-7.98** | Under base model continuation, harmful |
| $A^{\pi_G}$ (gov continuation) | **-11.02** | Under governor continuation, even worse |

**Zero out of 196 different-action interventions produce positive $A^{\pi_B}$.**

The oracle says the governor's action is better (+108 Q-points), but when the
model actually takes that action and continues, the result is worse (-8 utility).

### Where Does Value Disappear?

| Condition | Count | Interpretation |
|---|---|---|
| $A^* > 5$ and $A^{\pi_B} \approx 0$ | **185 / 196** | Model can't continue the path |
| $A^* > 5$ and $A^{\pi_B} > 0$ | **0 / 196** | Model benefits — zero cases |
| $A^* > 5$ and $A^{\pi_B} < 0$ | **1 / 196** | Model actively harmed |
| $A^* > 5$ and $A^{\pi_G} > 0$ | **0 / 196** | Gov continuation helps — zero cases |

**185 out of 196 cases: the oracle advantage exists but the model cannot
realize it.** The governor opens a path that requires optimal continuation,
and the model cannot provide that continuation.

### Rescueability Classification

| Category | Count | % | Mean $A^{\pi_B}$ |
|---|---|---|---|
| **UNRESCUABLE** | 197 | 75.8% | -0.12 (neutral) |
| **RESCUABLE_AMBIGUOUS** | 63 | 24.2% | -38.61 (harmful) |

**75.8% of intervention states are unrecoverable** — the oracle confirms
there is no path to success from these states. The governor's intervention
is neutral but pointless: the trajectory was indeed already doomed.

**24.2% are potentially rescuable** but the governor's intervention is
actively harmful (-38.61 mean $A^{\pi_B}$). In these cases, the governor
diverts the model from a potentially recoverable path into one that the
model cannot execute.

This partially confirms the "already doomed" hypothesis (75.8% of cases)
but also reveals a second failure mode: in the 24.2% potentially rescuable
cases, the governor's intervention makes things worse, not better.

### Success Conversion

| Continuation | Successes | Total |
|---|---|---|
| Base continuation ($a_B + \pi_B$) | 7 | 196 |
| Gov + OFF continuation ($a_G + \pi_B$) | **0** | 196 |
| Gov + SEL continuation ($a_G + \pi_G$) | **0** | 196 |

The governor's intervention destroys all 7 potentially rescuable cases.
Zero interventions produce success under either continuation policy.

### Chain Macro-Patterns

The I3.5.2c intervention chains reveal "information acquisition without
decision conversion":

| Action Sequence | Count | Mean $\Delta U$ |
|---|---|---|
| `SEARCH_MORE → REASON_MORE` | 106 | -4.15 |
| `SEARCH_MORE → VERIFY → SEARCH_MORE → ANSWER` | 49 | -6.50 |
| `SEARCH_MORE → VERIFY → SEARCH_MORE → REASON_MORE` | 23 | -6.50 |
| `SEARCH_MORE → SEARCH_MORE` | 14 | -4.15 |

The dominant pattern is `SEARCH_MORE → REASON_MORE` (106/194 chains). The
governor makes the model search for more information and then reason about
it, but this never leads to a correct answer. The model acquires information
but cannot convert it into a correct terminal decision.

All chain lengths have negative mean $\Delta U$:
- Length 2: mean $\Delta U = -4.15$
- Length 4: mean $\Delta U = -6.50$

Longer chains are worse, confirming that cascading interventions accumulate
cost without benefit.

### Scientific Interpretation

The I3.5.2d result confirms the $Q^* \neq Q^{\pi_{\text{model}}}$ hypothesis
with remarkable clarity:

```text
A*       = +108.41  (governor knows a better action under optimal continuation)
A^{π_B}  =   -7.98  (but the base model cannot realize that advantage)
A^{π_G}  =  -11.02  (and governor continuation is even worse)
```

The value disappears because:

1. **75.8% of intervention states are unrecoverable.** The oracle confirms
   no path to success exists. The governor's intervention is neutral but
   wastes executor resources (the -3.28 utility loss in I3.5.2c).

2. **24.2% are potentially rescuable, but the governor's intervention is
   harmful.** The governor diverts the model from a potentially recoverable
   path into one that the model cannot execute. The model would have had a
   small chance (7/196 successes) with the baseline action, but zero chance
   with the governor's action.

3. **The dominant chain pattern is information acquisition without decision
   conversion.** The governor repeatedly makes the model `SEARCH_MORE` and
   `REASON_MORE`, but the model cannot convert the additional information
   into a correct answer. This is the mechanism behind the utility loss.

4. **Governor continuation is worse than base continuation** ($A^{\pi_G} < A^{\pi_B}$).
   Persistent governor control does not help — it makes things worse by
   continuing to divert the model away from paths it can execute.

### Corrected Causal Chain (Final)

```text
I3.5.1:  Always-on governor damages the model policy              CONFIRMED
I3.5.2a: Governor sometimes selects actions with higher Q*        CONFIRMED
I3.5.2b: Model faithfully follows those recommendations            CONFIRMED
I3.5.2c: Higher Q* actions do NOT improve value under
         actual downstream model policy                            CONFIRMED
I3.5.2d: A* = +108 but A^{π_B} = -8 and A^{π_G} = -11
         75.8% of interventions are in unrecoverable states
         24.2% are rescuable but governor intervention is harmful
         Pattern: information acquisition without decision conversion
                                                                   CONFIRMED
```

The root cause is now precisely characterized:

$$\boxed{Q^*(s, a_G) \gg Q^{\pi_{\text{model}}}(s, a_G)}$$

The governor's $Q^*$-based ranking assumes optimal continuation. The model
cannot provide that continuation. The oracle advantage is not behaviorally
realizable by either the base model or the governor-controlled model.

### What This Means for Governor Design

The next governor should reason about **policy-conditional long-horizon
value**, not $Q^*$ alone:

$$\text{Intervene only when } Q^{\pi}(s, a_G) - Q^{\pi}(s, a_B) > 0$$

not:

$$\text{Intervene when } Q^*(s, a_G) - Q^*(s, a_B) > 0$$

This requires either:
1. A model of the model's continuation policy (learned or simulated)
2. Online rollouts with the actual model to estimate $Q^{\pi}$
3. A fundamentally different intervention criterion that doesn't rely on
   oracle optimality

The current governor's competence is real but **theoretically optimal, not
behaviorally realizable**. It identifies actions that would be better if the
model were optimal, but the model is not optimal.

---

## 10. Scientific Caveats

1. **The end-to-end SELECTIVE_FRAME experiment shows no terminal-success
   improvement and significant continuous DG worsening relative to OFF.**
   The governor has local oracle competence (I3.5.2a) and the model transmits
   it (I3.5.2b), but the local $Q^*$-advantage does not translate to
   trajectory-level value under the actual model policy. SELECTIVE_FRAME is
   safe (preserves terminal success, better than ALWAYS_ON) but not beneficial.

2. **The continuous DG is correctly defined as $V_O - V_\pi$, not binary
   success.** The corrected $\Delta DG = -3.28$ is harmful, not neutral. The
   terminal success difference ($\Delta\text{Success} = 0$) is reported
   separately and should not be called DG.

3. **The utility loss comes from longer/costlier executor trajectories, not
   from model token overhead.** The `MetareasoningUtility` charges simulated
   resource consumption (executive steps, retrieval, verification, search,
   reasoning). Model tokens are telemetry, not directly charged.

4. **The root cause is $Q^* \neq Q^{\pi_{\text{model}}}$.** I3.5.2d confirms
   this empirically: $A^* = +108$ but $A^{\pi_B} = -8$ and $A^{\pi_G} = -11$.
   The oracle advantage is not behaviorally realizable by either policy.

5. **75.8% of intervention states are unrecoverable** (confirmed by oracle
   state graph analysis). The "already doomed" hypothesis is partially
   confirmed. The remaining 24.2% are potentially rescuable but the governor's
   intervention is actively harmful in those cases.

6. **The dominant failure pattern is information acquisition without decision
   conversion.** The governor makes the model search and reason, but the model
   cannot convert additional information into correct terminal decisions.

7. **The fold-isolated CV precision (48.1%) is the honest generalization
   estimate.** The global-rules precision (72.4%) overestimates performance
   because the rules were not discovered fold-isolated.

8. **The Brier score is calibrated** via isotonic regression. The raw Brier
   (0.2623) was worse than the base rate (0.0849). The calibrated Brier (0.0818)
   beats the base rate.

9. **Q-value source tracking** shows 88.4% of Q-values come from the oracle
   table. 11.6% use fixed fallback penalties (all for baseline `ANSWER` actions
   not present in the oracle). The fallback penalty of $-125.11$ represents the
   standard incorrect-answer penalty.

10. **Validation is stopped.** The primary development hypothesis failed.
    The gate design must be fundamentally reconsidered (measuring $Q^{\pi_B}$
    instead of $Q^*$) before any validation attempt. Do not tune against
    validation and then proceed to held-out pretending it's the same frozen
    experiment.

---

## 11. V2B-I3.5.3 Q^{π_B}-Based Selective Governor — COMPLETED

> **Milestone:** Replace the $Q^*$-based rule gate with a $Q^{\pi_B}$-based
> regression gate that estimates policy-conditional value instead of oracle
> optimal value.

Scripts:
- `scripts/train_q_pib_gate.py` — trains the $Q^{\pi_B}$ regression model
- `scripts/run_v2b_i3_5_3_experiment.py` — runs the end-to-end experiment
- `hrm_adaptive_memory/executive/selective_governor/q_pib_predictor.py` — gate predictor

Output: `experiments/v2b_i3_5_2/development/i353_4fa944de33ed/`

### Gate Design

The $Q^{\pi_B}$ gate replaces the $Q^*$-based `RuleBasedInterventionPredictor`
with a `QPiBInterventionPredictor` that uses a `GradientBoostingRegressor` to
estimate $Q^{\pi_B}(s, a)$ — the value of taking action $a$ at state $s$ and
continuing with the actual OFF model policy.

**Training data:**
- 761 samples from OFF trajectories: $Q^{\pi_B}(s, a_{\text{taken}}) = $ realized utility from $s$ onward
- 456 samples from I3.5.2d forks: $Q^{\pi_B}(s, a_G) = $ fork B realized utility
- Total: 1217 samples, 35 features (controller-visible features + action one-hot)

**Fold-isolated cross-validation (5 folds, split by task):**

| Metric | Value |
|---|---|
| Mean R² | 0.4815 |
| Advantage accuracy | 100% |
| Harm rate | 0% |

The model correctly identifies which action has higher $Q^{\pi_B}$ in 100% of
held-out test cases, with zero harmful predictions.

**Top features:**
1. `verif_SUFFICIENT` (56.0%) — whether verification is sufficient
2. `conflict_count` (16.3%) — number of unresolved conflicts
3. `verified_count` (7.9%) — number of verified items
4. `temporal_CURRENT` (5.6%) — temporal status
5. `act_ANSWER` (4.3%) — whether action is ANSWER

### End-to-End Results (300 tasks, 3 arms)

| Metric | OFF | ALWAYS_ON | SELECTIVE_QPIB |
|---|---|---|---|
| **Terminal success** | **83/300 (27.7%)** | 60/300 (20.0%) | **83/300 (27.7%)** |
| **Mean utility** | **-74.89** | -90.21 | **-74.91** |
| **Mean executor steps** | 2.5 | 5.1 | 2.5 |
| **Mean model calls** | 2.5 | 5.1 | 2.5 |
| **Mean model tokens** | 2,519 | 9,679 | 2,529 |
| **Interventions** | 0 | — | **0** |

### Primary Hypothesis Test

| Hypothesis | Result | Detail |
|---|---|---|
| $\Delta DG_S > 0$ | **NOT SUPPORTED** | $\Delta DG = -0.03$, CI [-0.07, +0.01] |
| $\Delta U_S > 0$ | **NOT SUPPORTED** | $\Delta U = -0.03$ (same as $\Delta DG$) |
| Terminal success preserved | **SUPPORTED** | 83/300 = 83/300, zero discordant |
| $U_S > U_A$ | **SUPPORTED** | $-74.91 > -90.21$ |

The $\Delta U = -0.03$ is essentially zero (the 95% CI includes zero). The tiny
non-zero value is from minor API nondeterminism (temperature=0.0 but not
bit-exact reproducible).

### The Q^{π_B} Gate Produced Zero Interventions

The gate learned the lesson of I3.5.2d: the oracle advantage ($A^* = +108$)
is not realizable by the actual model policy ($A^{\pi_B} = -8$). Therefore, the
correct action is to never intervene.

**0 interventions across all 300 tasks.** The gate correctly identifies that
no action has $Q^{\pi_B}(s, a_G) - Q^{\pi_B}(s, a_{\text{natural}}) > 5.0$ for
any state. The model's natural actions (ANSWER, RETRIEVE, STOP) are already as
good as any alternative under the model's actual continuation policy.

### Comparison with I3.5.2c

| Metric | I3.5.2c (Q* gate) | I3.5.3 (Q^{π_B} gate) | Change |
|---|---|---|---|
| $\Delta U$ | -3.28 | -0.03 | **+3.25** (eliminated utility loss) |
| Interventions | 536 | 0 | **-536** (100% reduction) |
| Success | 83/300 | 83/300 | preserved |
| Executor steps | 3.9 | 2.5 | -1.4 (no extra steps) |
| Model tokens | 5,623 | 2,529 | -3,094 (no extra calls) |

The $Q^{\pi_B}$ gate **eliminated the utility loss** from I3.5.2c by not
intervening. It preserved terminal success (83/300 = 83/300, zero discordant
pairs) and eliminated the extra executor steps and model token consumption.

### Scientific Interpretation

**The $Q^{\pi_B}$ gate is a correct "do no harm" gate.** It learned that
interventions don't help under the actual model policy, so it doesn't
intervene. This is the scientifically correct behavior given the I3.5.2d
evidence.

However, **SELECTIVE_QPIB does not improve over OFF.** It merely equals OFF.
The gate cannot find any intervention that would help the model, because
(according to I3.5.2d) no such intervention exists for this model on this
benchmark.

### What This Means

The complete experimental arc from I3.5.2a through I3.5.3 tells a coherent
story:

```text
I3.5.2a: Governor has oracle competence (Q* advantage)         CONFIRMED
I3.5.2b: Model follows governor recommendations (98%)           CONFIRMED
I3.5.2c: Q*-based gate interventions harm utility (-3.28)      CONFIRMED
I3.5.2d: Oracle advantage not realizable (A* = +108, A_πB = -8) CONFIRMED
I3.5.3: Q^{π_B} gate learns to not intervene (0 interventions)  CONFIRMED
         Utility loss eliminated (-3.28 → -0.03)
         Success preserved (83/300 = 83/300)
```

**The conclusion is that the current governor architecture cannot improve
this model's decisions on this benchmark.** The governor has genuine oracle
competence, but that competence is not behaviorally realizable by the model.
A $Q^{\pi_B}$-aware gate correctly avoids harmful interventions, but cannot
find beneficial ones because none exist.

### What Would Be Needed for Improvement

To improve the model's decisions, one of the following would be needed:

1. **A different model** that can execute the optimal continuation paths the
   governor identifies. The current model cannot convert `SEARCH_MORE` and
   `REASON_MORE` into correct answers.

2. **A different benchmark** where the model's natural policy is suboptimal
   in ways the governor can identify and the model can exploit.

3. **A different governor architecture** that doesn't just recommend actions
   but actively assists the model in executing them (e.g., providing
   structured reasoning hints, not just action recommendations).

4. **Online rollout-based intervention** that actually simulates the model's
   continuation from each candidate action before deciding to intervene. This
   would be expensive but would directly measure $Q^{\pi_B}$ at runtime
   instead of estimating it offline.

### Validation Status

```
VALIDATION = STOP
HELD-OUT   = DO NOT TOUCH
```

The development result is negative but scientifically informative:
- The $Q^*$ gate (I3.5.2c) actively harms utility by intervening when it shouldn't
- The $Q^{\pi_B}$ gate (I3.5.3) correctly avoids harm but cannot find benefit
- The model does not benefit from cognitive control assistance on this benchmark

This is a valid negative result. The governor architecture is sound (it can
learn when to intervene and when not to), but the model-benchmark combination
does not admit beneficial interventions.

> **Correction (I3.5.3-r1):** The I3.5.3 result above should be frozen as:
> "A conservative learned surrogate gate produced a null-intervention policy
> on the development distribution, eliminating the harm/cost of the $Q^*$-derived
> gate while preserving baseline terminal performance."
>
> The I3.5.3 gate did **not** prove that "no beneficial interventions exist."
> It had four specification defects (see Section 12) that were corrected in
> I3.5.3-r1.

---

## 12. V2B-I3.5.3-r1 Base-First Pairwise Advantage Gate — COMPLETED

> **Milestone:** Correct the four specification defects of I3.5.3 and implement
> the mathematically correct gate:
> $$\text{intervene iff } \hat Q^{\pi_B}(s, a_G) - \hat Q^{\pi_B}(s, a_B) > 0$$

### Defects corrected from I3.5.3

1. **Unknown $a_B$**: I3.5.3 guessed "natural actions" from a hard-coded set
   $\{ANSWER, RETRIEVE, STOP\}$. I3.5.3-r1 calls the model first to get the
   actual $a_B$, then evaluates the pairwise advantage.

2. **Effective threshold**: I3.5.3's confidence formula ($\Delta Q / 50$) made
   the effective threshold $\Delta Q \ge 30$, not 5. I3.5.3-r1 uses a direct
   LCB margin with no confidence proxy.

3. **Fake harm probability**: I3.5.3's `harm_probability = worse_count / 7`
   was not calibrated. I3.5.3-r1 eliminates it entirely — the pairwise
   advantage directly encodes the harm/benefit signal.

4. **Identity binding**: I3.5.3's identity said `RuleBasedInterventionPredictor`
   and didn't hash the trained model. I3.5.3-r1 binds:
   - `pairwise_advantage_predictor.py` SHA-256
   - Trained model SHA-256
   - Training dataset SHA-256
   - Training script SHA-256
   - Training summary SHA-256
   - sklearn/numpy/python versions

### Architecture

```
state s
   │
   ├── Base packet → model → a_B  (always happens)
   │
   └── Local governor → a_G
                         │
                         ▼
       Pairwise advantage evaluator
       input: (features(s), a_B, a_G)
       output: ΔQ_π_hat, LCB
                         │
          ┌──────────────┴──────────────┐
          │                             │
       SKIP                          INTERVENE
          │                             │
     execute a_B              governor packet → model → a_T
                                        │
                                        ▼
                                   execute a_T
```

SKIP costs no extra model call. Only INTERVENE requires a second call.

### Expanded Fork Dataset

Scripts:
- `scripts/build_i3_5_3r1_expanded_fork_dataset.py`
- `scripts/train_pairwise_advantage_gate.py`
- `scripts/run_v2b_i3_5_3r1_experiment.py`
- `hrm_adaptive_memory/executive/selective_governor/pairwise_advantage_predictor.py`

Unlike I3.5.2d (which only forked at $Q^*$-gate-selected states), I3.5.3-r1
forks at **every** OFF trajectory state where $a_G \ne a_B$.

**300 governor-baseline disagreement states found** across 235 development tasks
(65 tasks contributed no governor/baseline disagreement).

Action pair distribution:

| Base action | Governor action | Count |
|---|---|---|
| ANSWER | SEARCH_MORE | 133 |
| ANSWER | VERIFY | 65 |
| RETRIEVE | VERIFY | 62 |
| STOP | ANSWER | 35 |
| ANSWER | REASON_MORE | 2 |
| REASON_MORE | VERIFY | 2 |
| VERIFY | SEARCH_MORE | 1 |

For each disagreement, two forks were run:
- Fork A: execute $a_B$, continue with OFF model → $U_A$
- Fork B: execute $a_G$, continue with OFF model → $U_B$
- Target: $\Delta Q_\pi = U_B - U_A$

**Fork dataset results:**

| Metric | Value |
|---|---|
| N | 300 |
| Mean $\Delta Q_\pi$ | -18.80 |
| Mean $A^*$ | -37.68 |
| $\Delta Q_\pi > +1$ (positive) | 52 (17.3%) |
| $|\Delta Q_\pi| \le 1$ (neutral) | 199 (66.3%) |
| $\Delta Q_\pi < -1$ (negative) | 49 (16.3%) |
| Base continuation success | 42/300 |
| Gov continuation success | 0/300 |

All 52 positive $\Delta Q_\pi$ cases are `RETRIEVE → VERIFY` with mean $+1.95$
— below the intervention threshold of 5.0.

### Pairwise Model Training

**Input:** 42 features (35 state features + 7 $a_B$ one-hot + 7 $a_G$ one-hot)
**Output:** scalar $\Delta Q_\pi$

**Fold-isolated CV (5 folds, split by task) with constant-baseline comparator:**

| Metric | Model | Constant baseline (predict 0) |
|---|---|---|
| Mean MSE | 926.2 | 2809.8 |
| Mean R² | **0.6364** | -0.1453 |
| Mean sign accuracy | **96.35%** | 66.35% |
| Interventions approved ($\tau=5$) | 0 | 0 |

The model significantly beats the constant baseline on both R² and sign
accuracy. The constant baseline's 66% sign accuracy comes from the 66%
neutral cases where $\Delta Q_\pi \approx 0$.

The model's 96% sign accuracy is genuine predictive performance, not a
trivial baseline artifact. However, it approves 0 interventions at $\tau=5$
because no predicted $\Delta Q_\pi$ exceeds the threshold.

### End-to-End Results (300 tasks, 3 arms)

| Metric | OFF | ALWAYS_ON | SELECTIVE_QPIB_BASE_FIRST |
|---|---|---|---|
| **Terminal success** | 82/300 (27.3%) | 58/300 (19.3%) | **83/300 (27.7%)** |
| **Mean utility** | -75.32 | -91.07 | **-74.94** |
| **Mean executor steps** | 2.5 | 5.1 | 2.6 |
| **Mean model calls** | 2.5 | 5.1 | 2.5 |
| **Mean model tokens** | 2,534 | 9,734 | 2,545 |
| **Interventions** | 0 | — | **0** |

### Primary Hypothesis Test

| Hypothesis | Result | Detail |
|---|---|---|
| $\Delta DG_S > 0$ (LCB > 0) | **NOT SUPPORTED** | $\Delta DG = +0.38$, CI [-0.05, +1.21] |
| Terminal success preserved | **SUPPORTED** | 83/300 vs 82/300, 1 discordant pair |
| $U_S > U_A$ | **SUPPORTED** | $-74.94 > -91.07$ |
| Identity binding | **PASS** | Model SHA-256 bound |

The $\Delta U = +0.38$ is from API nondeterminism (1 task where SEL succeeded
but OFF didn't, despite 0 interventions). The 95% CI includes zero.

### Base-First Cost Efficiency

The base-first architecture means SKIP costs **zero extra model calls**.
Since the gate approved 0 interventions:
- SEL model calls = OFF model calls = 2.5
- SEL tokens = OFF tokens + 12 (telemetry noise)
- SEL steps = OFF steps + 0.1 (nondeterminism)

This is the optimal cost profile for a null-intervention gate.

### Permissive Threshold Test (Offline Replay)

> **Correction (I3.5.3-r2):** The original I3.5.3-r1 claim that "even at
> $\tau=0$ and margin=0, the gate produces 0 interventions" was invalid.
> The CLI overrides were applied to the parent process but not propagated
> to worker predictors. The actual workers used the serialized defaults
> ($\tau=5$, margin=5). This is fixed in I3.5.3-r2.

An offline replay of the 300-task SELECTIVE_QPIB_BASE_FIRST trajectories
was run with $\tau=0$ and margin=0 (no DeepSeek calls needed — pure
offline computation using saved trajectory steps and the trained model).

**Offline replay results (303 governor/baseline disagreements):**

| Metric | Standard (τ=5, margin=5) | Permissive (τ=0, margin=0) |
|---|---|---|
| Approved interventions | 0 | 68 |
| Max predicted ΔQ_π | +3.18 | +3.18 |
| Max LCB | -1.82 | +3.18 |
| Predicted > 0 | 66/303 (21.8%) | 66/303 (21.8%) |
| Predicted > 5 | 0/303 (0.0%) | 0/303 (0.0%) |

At $\tau=0$, the gate would approve 68 interventions (63 RETRIEVE→VERIFY,
2 ANSWER→REASON_MORE, 1 VERIFY→SEARCH_MORE, 2 others). However, the
maximum predicted ΔQ_π is only +3.18, and the maximum raw fork ΔQ_π is
+5.34 — neither exceeds the frozen 5+5 criterion.

The 66 positive predictions are all small (mean +1.92 for RETRIEVE→VERIFY).
These are not task rescues — the fork dataset shows 0 governor-continuation
successes versus 42 base-continuation successes across 300 forks. The
positive ΔQ_π cases represent small utility improvements within trajectories
that still fail to achieve successful termination.

**Complete runtime distribution (standard τ=5, margin=5, offline replay):**

| Metric | Value |
|---|---|
| Total evaluations | 767 |
| a_G == a_B (no disagreement) | 464 |
| a_G != a_B (disagreement) | 303 |
| Mean predicted ΔQ_π | -18.51 |
| Min predicted ΔQ_π | -120.00 |
| Max predicted ΔQ_π | +3.18 |
| Mean LCB | -23.51 |
| Max LCB | -1.82 |
| Predicted > 0 | 66/303 (21.8%) |
| Predicted > 5 | 0/303 (0.0%) |
| LCB > 0 | 0/303 (0.0%) |
| LCB > 5 | 0/303 (0.0%) |
| Approved (INTERVENE) | 0/303 |
| Skipped (SKIP) | 303/303 |

### Scientific Interpretation

**I3.5.3-r1 establishes:**

1. The pairwise advantage gate is correctly specified:
   $\hat Q^{\pi_B}(s, a_G) - \hat Q^{\pi_B}(s, a_B) > 0$

2. The gate is properly identity-bound (model SHA-256, training data SHA-256,
   sklearn version all recorded).

3. The expanded fork dataset covers **all** governor-baseline disagreements,
   not just $Q^*$-gate-selected states.

4. The model beats a constant-baseline comparator (R²=0.64 vs -0.14).

5. Under the frozen threshold $\tau=5$ and LCB margin $m=5$, none of the
   runtime disagreements satisfies the intervention criterion.

**What I3.5.3-r1 does NOT establish:**

- It does **not** prove that no beneficial interventions exist across all
  possible alternative actions. It only covers the 7 action types the
  governor can recommend, at the states where the governor disagrees with
  the baseline.

- The 52 positive observed $\Delta Q_\pi$ cases (all `RETRIEVE → VERIFY`,
  mean $+1.95$) suggest there may be small beneficial interventions below
  the frozen criterion.
  A lower threshold might approve some, but the expected gain is small.

**The strongest supported statement is:**

> Across all observed governor-baseline disagreements on the development
> distribution, no pairwise intervention produced a sufficiently robust
> predicted advantage to cross the frozen 5+5 intervention criterion.
> Observed positive policy-conditional advantages were small; the maximum
> raw fork ΔQ_π was +5.34 and the positive RETRIEVE→VERIFY region averaged
> +1.95. At runtime, the maximum predicted ΔQ_π was +3.18 (offline replay,
> 303 disagreements). The learned pairwise advantage gate correctly
> abstains under the frozen criterion, preserving baseline performance at
> zero extra cost.

### Comparison Across All Milestones

| Milestone | Gate | ΔU vs OFF | Interventions | Success | Identity |
|---|---|---|---|---|---|
| I3.5.2c | $Q^*$ rule | -3.28 | 536 | 83/300 | Partial |
| I3.5.3 | $Q^{\pi_B}$ 7-action | -0.03 | 0 | 83/300 | **Broken** |
| I3.5.3-r1 | Pairwise base-first | +0.38 | 0 | 83/300 | **Bound** |

I3.5.3-r1 is the first correctly specified, properly identity-bound,
base-first pairwise advantage gate. It confirms the null-intervention finding
of I3.5.3 but with a mathematically correct decision rule.

---

## 13. V2B-I3.5.3-r2 Pairwise Gate Closure — COMPLETED

> **Milestone:** Close three documentation/specification gaps in I3.5.3-r1
> without running a new end-to-end experiment.

### Repairs applied

1. **Worker threshold propagation**: CLI `--delta-threshold` and `--lcb-margin`
   overrides are now applied to each worker's predictor, not just the parent
   process. The previous 50-task "permissive threshold test" was invalid
   because workers used serialized defaults (τ=5, margin=5).

2. **Runtime parameter identity binding**: The effective threshold, margin,
   and CLI overrides are now bound into `experiment_identity.json` under
   `runtime_gate_params`, and included in the combined identity hash.

3. **Complete gate evaluation persistence**: An offline replay script
   (`scripts/replay_i3_5_3r1_gate_evaluations.py`) reconstructs every
   SELECTIVE_QPIB_BASE_FIRST trajectory state, recomputes a_G, runs the
   pairwise model, and records every evaluation (including SKIPs) to
   `gate_evaluations.jsonl`.

### Documentation corrections

1. **"300 disagreement states across 300 tasks"** → corrected to
   "300 disagreement states across 235 development tasks" (65 tasks
   contributed no governor/baseline disagreement).

2. **"no governor intervention improved return by >5 points"** → corrected.
   The maximum raw fork ΔQ_π was +5.34. The correct claim is that no
   pairwise intervention produced a sufficiently robust predicted advantage
   to cross the frozen 5+5 criterion.

3. **"even at τ=0, the gate produces 0 interventions"** → removed.
   Offline replay with τ=0, margin=0 shows the gate would approve 68
   interventions. The max predicted ΔQ_π is +3.18 (below 5 but above 0).

### Offline replay results

Script: `scripts/replay_i3_5_3r1_gate_evaluations.py`

No DeepSeek API calls needed — pure offline computation using saved
trajectory steps and the trained pairwise model.

**Standard criterion (τ=5, margin=5):**

| Metric | Value |
|---|---|
| Total evaluations | 767 |
| a_G == a_B | 464 |
| a_G != a_B (disagreements) | 303 |
| Max predicted ΔQ_π | +3.18 |
| Max LCB | -1.82 |
| Predicted > 0 | 66 (21.8%) |
| Predicted > 5 | 0 (0.0%) |
| LCB > 0 | 0 (0.0%) |
| Approved | 0/303 |

**Permissive criterion (τ=0, margin=0):**

| Metric | Value |
|---|---|
| Approved | 68/303 |
| Max predicted ΔQ_π | +3.18 |
| All approved are RETRIEVE→VERIFY (63) or small positive (5) |

### Key observation: small local improvement ≠ task rescue

The fork dataset shows:
- Base continuation successes: 42/300
- Governor continuation successes: 0/300

The 66 positive predicted ΔQ_π cases (all RETRIEVE→VERIFY, mean +1.92)
are **not task rescues**. They represent small utility improvements within
trajectories that still fail to achieve successful termination.

This reinforces the broader conclusion:

$$\text{small local utility improvement} \neq \text{task rescue}$$

### Final scientific arc

```
Q* competence exists                    CONFIRMED
        ↓
model follows governor advice           CONFIRMED (98%)
        ↓
Q*-based intervention harms policy      CONFIRMED (ΔU = -3.28)
        ↓
policy-conditional forks explain why    CONFIRMED (A* = +108, A_πB = -8)
        ↓
base-first pairwise gate learns         CONFIRMED (0 interventions at 5+5)
conservative abstention
        ↓
abstention is correct, not just safe    CONFIRMED (max runtime ΔQ_π = +3.18)
        ↓
small positive ΔQ_π ≠ task rescue       CONFIRMED (0 gov successes in forks)
```

The remaining research question is no longer "how do we gate this governor
better?" It is how to create interventions that materially improve
$Q^{\pi}$ — probably by changing what assistance the governor provides,
rather than improving the gating mechanism further.
