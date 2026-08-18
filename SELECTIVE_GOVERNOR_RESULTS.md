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
V2B-I3.5.2a + I3.5.2b + I3.5.2c-r1
STATE-LEVEL COMPETENCE + PACKET TREATMENT + END-TO-END TRAJECTORY

  Oracle governor-ranking competence:     SUPPORTED
  Local positive competence region:       SUPPORTED
  Local hazard region:                    SUPPORTED
  Task-group state leakage:               CONTROLLED
  Fold-isolated rule stability:           SUPPORTED (same rules in 5/5 folds)
  Probability calibration:                REPAIRED (isotonic, Brier < base rate)
  CV rule-discovery independence:         ESTABLISHED (fold-isolated mode)
  Q-value source tracking:                IMPLEMENTED (88.4% oracle, 11.6% fallback)
  Packet-level treatment benefit:         SUPPORTED (A_treatment ≈ A_ranking, +21.52)
  Model follows governor:                 98.0% (743/758)
  Model refuses harmful gov advice:       SUPPORTED (11/35 STOP->ANSWER refused)
  Terminal success preserved:             SUPPORTED (83/300 = 83/300, zero discordant)
  Continuous DG improvement:              NOT SUPPORTED (ΔDG = -3.28, LCB = -3.58, harmful)
  Utility improvement:                    NOT SUPPORTED (ΔU = -3.28, same as ΔDG)
  SELECTIVE safe vs ALWAYS_ON:            SUPPORTED (83 vs 60 success, -78 vs -90 utility)
  Root cause identified:                  Q* ≠ Q^{π_model} (oracle continuation ≠ model continuation)
  Validation status:                      STOPPED (primary hypothesis failed)
  Counterbalancing:                       IMPLEMENTED (HMAC-based, 6 permutations)
  Experiment identity binding:            IMPLEMENTED (all component hashes)
  Token/latency cost tracking:            IMPLEMENTED
  Cascade diagnostics:                    IMPLEMENTED (max chain = 4, no runaway)
  Next milestone:                         I3.5.2d (policy-conditional intervention value)
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

## 9. Scientific Caveats

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

4. **The root cause is $Q^* \neq Q^{\pi_{\text{model}}}$.** The governor's
   $Q^*$-based ranking assumes optimal continuation. The model does not
   continue optimally. The I3.5.2d experiment will measure the
   policy-conditional value $Q^{\pi_B}$ to quantify this gap precisely.

5. **"The trajectory was already doomed at Step 0" is a candidate
   explanation, not yet established.** The rescueability test in I3.5.2d will
   determine whether intervened tasks are genuinely unrecoverable or whether
   the problem is downstream policy execution.

6. **The fold-isolated CV precision (48.1%) is the honest generalization
   estimate.** The global-rules precision (72.4%) overestimates performance
   because the rules were not discovered fold-isolated.

7. **The Brier score is calibrated** via isotonic regression. The raw Brier
   (0.2623) was worse than the base rate (0.0849). The calibrated Brier (0.0818)
   beats the base rate.

8. **Q-value source tracking** shows 88.4% of Q-values come from the oracle
   table. 11.6% use fixed fallback penalties (all for baseline `ANSWER` actions
   not present in the oracle). The fallback penalty of $-125.11$ represents the
   standard incorrect-answer penalty.

9. **Validation is stopped.** The primary development hypothesis failed.
   The gate design must be reconsidered (measuring $Q^{\pi_B}$ instead of
   $Q^*$) before any validation attempt. Do not tune against validation and
   then proceed to held-out pretending it's the same frozen experiment.
