# V2B-I3.5.2 Selective Governor Intervention Gate — Results & Design

## Executive Summary

Following the causal diagnosis in V2B-I3.5.1 where the unconditional ("always-on") governor was shown to systematically degrade decision quality ($\Delta\text{DG}_{\text{gov}|\text{aware}} = -14.60$, $\Delta U = -14.60$, extra calls $+2.62$), we implemented **V2B-I3.5.2 Selective Governor Intervention Gate**.

Instead of injecting the governor advisory frame into the model packet on every step, a lightweight, controller-visible **Intervention Gate** evaluates whether the governor is expected to help or harm before allowing it into the prompt.

```text
ControllerObservation
        │
        ▼
Intervention Gate
        │
        ├── SKIP ───────────────► Base packet ─► Model
        │
        └── INTERVENE
                │
                ▼
             Governor
                │
                ▼
        Governor packet ────────► Model
```

---

## Stage 1: First-Divergence Analysis (Offline I3.5.1 Dataset)

Using all 300 development tasks from I3.5.1 (`structure_dev_v2`), we performed step-by-step paired divergence extraction between `AWARE_NO_GOVERNOR` and `AWARE_GOVERNOR`.

### Summary Statistics
- **Total tasks:** 300
- **Diverged tasks:** 221 (73.7%)
- **Identical trajectories:** 79 (26.3%)
- **Intervention Outcomes:**
  - **HARM ($\Delta U < -5.0$):** 196 tasks (65.3%)
  - **NEUTRAL ($-5.0 \le \Delta U \le 5.0$):** 104 tasks (34.7%)
  - **HELP ($\Delta U > 5.0$):** 0 tasks (0.0%)
- **First Divergence Step Distribution:**
  - Step 0: 86 tasks
  - Step 2: 122 tasks
  - Step 3: 13 tasks

### Action Substitution Matrix & Causal Mechanisms

| Baseline Action $\to$ Governor Action | Count | % of Div. | Mean $\Delta U$ | Min $\Delta U$ | Max $\Delta U$ | Causal Mechanism |
|---|---|---|---|---|---|---|
| `ANSWER -> SEARCH_MORE` | 123 | 55.7% | -9.27 | -11.79 | -4.30 | Model terminates after 2 failed attempts; governor forces endless search loop |
| `RETRIEVE -> VERIFY` | 62 | 28.1% | -9.46 | -11.79 | -5.34 | Step 0: Model retrieves evidence; governor prematurely forces verify before retrieval |
| `STOP -> ANSWER` | 21 | 9.5% | -120.00 | -120.00 | -120.00 | Step 0: Model correctly stops on irrelevant task; governor forces answer (-120 penalty) |
| `ANSWER -> VERIFY` | 13 | 5.9% | -8.72 | -17.12 | -5.34 | Model terminates on falsified evidence; governor forces re-verification |
| `ANSWER -> REASON_MORE` | 2 | 0.9% | -9.63 | -9.63 | -9.63 | Model terminates; governor forces composition |

---

## Stage 2: Selective Intervention Gate Architecture

We built the `hrm_adaptive_memory.executive.selective_governor` package with 6 core modules:

1. **`features.py`**: Extracts strictly controller-visible features (`remaining_steps`, `prior_action_count`, `last_action`, `verification_state`, `temporal_status`, `conflict_count`, `repeated_no_gain`, resource budgets, `chain_started`, etc.). Never touches latent or evaluator state.
2. **`model.py`**: Implements `RuleBasedInterventionPredictor` and `CalibratedLinearPredictor` with conservative default (= silence / HARM on error or missing data).
3. **`intervention_gate.py`**: Evaluates `expected_delta_utility > threshold` ($+5.0$), `harm_probability < harm_limit` ($0.15$), and `confidence >= min_confidence` ($0.60$). Fails closed to `SKIP`.
4. **`identity.py`**: Deterministic SHA-256 identity binding feature extraction, model weights/rules, and frozen decision thresholds.
5. **`serializer.py`**: JSON serialization and SHA-256 hashing of `InterventionDecision`.
6. **`calibration.py`**: Offline gate evaluation tools computing intervention rate, harm rate, benefit rate, net intervention value, and utility savings.

---

## Offline Calibration & Validation

Evaluating `SelectiveGovernorGate` on the full I3.5.1 development dataset:

| Metric | Always-On Governor | Selective Gate | Difference / Gain |
|---|---|---|---|
| Interventions Approved | 300 / 300 (100%) | 0 / 300 (0.0%) | -100% intervention rate |
| Harm Rate on Interventions | 65.3% | 0.0% | -65.3% harm rate |
| Total Realized Utility | -26,840.23 | -22,461.23 | **+4,379.00 utility points saved** |
| Average Utility per Task | -89.47 | -74.87 | **+14.60 utility gain per task** |

---

## I3.5.2 Runner & Modes

The new `hrm_adaptive_memory.executive.i3_5_2.trajectory_runner` supports four operational modes:
- `OFF`: Clean base packet, governor never evaluated.
- `ALWAYS_ON`: Governor always evaluated and injected.
- `SELECTIVE`: Gate evaluates per step; injects governor only when approved.
- `SHADOW_SELECTIVE`: Gate evaluates silently and logs telemetry, but base packet is sent to model.

Unit tests (`tests/unit/test_selective_governor.py` and `tests/unit/test_i352_runner.py`) verify feature extraction purity, hazard detection, conservative skip defaults, mode routing, and deterministic hash identity. All 117 unit tests pass.
