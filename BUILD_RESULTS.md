# V2B-I3.5.1 Build Results

## Overview

**Milestone:** V2B-I3.5.1 — Governor Causal Protocol & Artifact Closure Repair

**Branch:** `i3.5.1-clean`

**Date:** 2026-08-18

**Model:** deepseek-chat (deepseek-v4-flash), temperature=0.0, max_tokens=2048

**Design:** 2x2 factorial (BLIND/AWARE x GOVERNOR OFF/ON), 4 conditions per task,
counterbalanced via HMAC-SHA256 permutation selection.

## Infrastructure Summary

| Gate | Description | Status |
|------|-------------|--------|
| G00 | Pre-repair archive frozen | PASS |
| G01 | Canonical identity | PASS |
| G02 | Scientific criteria frozen | PASS |
| G03 | Treatment separation (BASE vs GOVERNOR) | PASS |
| G04 | Leakage checks (16 forbidden keys) | PASS |
| G05 | Benchmark closure | PASS |
| G06 | Structural isolation | PASS |
| G07 | Oracle identity | PASS |
| G08 | Governor identity | PASS |
| G09 | Executor identity | PASS |
| G10 | Generation config frozen | PASS |
| G11 | Model identity (same across arms) | PASS |
| G12 | Prompt/decoder identity | PASS |
| G13 | Factorial scheduler (4-arm) | PASS |
| G14 | Fingerprint policy | PASS |
| G15 | Receipt hash chain | PASS |
| G16 | Replay determinism | PASS |
| G17 | IG/DG/TR invariant (tol 1e-9) | PASS |
| G18 | Observable oracle invariance | PASS |
| G19 | Governor-executor parity | PASS |
| G20 | Artifact provenance DAG | PASS |
| G21 | Report invariants | PASS |
| G22 | Full test suite (101 tests) | PASS |
| G23 | Repository clean-state | PASS |
| G24 | Frozen experiment bundle | PASS |

## Experiment Runs

| Phase | Tasks | Trajectories | Model Calls | Backend Errors | Decoder Failures | Replay Match |
|-------|-------|-------------|-------------|---------------|-----------------|-------------|
| Development | 300 | 1200 | 3976 | 0 | 0 | 1200/1200 |
| Validation | 150 | 600 | 2042 | 0 | 0 | 600/600 |
| Held-Out | 150 | 600 | 2030 | 0 | 0 | 600/600 |

## Success Rates

| Condition | Development | Validation | Held-Out |
|-----------|------------|------------|----------|
| BLIND_NO_GOVERNOR | 80/300 (26.7%) | 43/150 (28.7%) | 46/150 (30.7%) |
| BLIND_GOVERNOR | 49/300 (16.3%) | 34/150 (22.7%) | 33/150 (22.0%) |
| AWARE_NO_GOVERNOR | 83/300 (27.7%) | 45/150 (30.0%) | 42/150 (28.0%) |
| AWARE_GOVERNOR | 62/300 (20.7%) | 34/150 (22.7%) | 29/150 (19.3%) |

## Decision Gap (DG) Table — Lower is Better

| | Governor OFF | Governor ON |
|-----------|-------------|------------|
| BLIND | 132.07 / 117.28 / 126.93 | 151.06 / 128.71 / 141.10 |
| AWARE | 141.14 / 128.22 / 146.51 | 155.74 / 144.14 / 164.04 |

*Format: Development / Validation / Held-Out*

## Topology-Cluster Bootstrap (95% CI)

| Contrast | Development | Validation | Held-Out |
|---------|------------|------------|----------|
| ΔDG gov|aware | -14.60 [-20.09, -10.00] **excl.0** | -15.92 [-33.09, -5.32] **excl.0** | -17.53 [-37.85, -5.28] **excl.0** |
| ΔDG gov|blind | -18.99 [-34.98, -6.15] **excl.0** | -11.43 [-34.70, -1.15] **excl.0** | -14.16 [-42.60, -1.37] **excl.0** |
| ΔDG state|no-gov | -9.07 [-17.24, 1.92] | -10.94 [-15.49, -6.78] **excl.0** | -19.57 [-28.44, -9.27] **excl.0** |
| Δ interaction | -4.40 [-18.26, 5.40] | 4.49 [-2.35, 9.42] | 3.37 [-5.39, 9.23] |
| ΔIG (rep. adv.) | 9.05 [5.29, 13.07] **excl.0** | 9.76 [3.81, 16.50] **excl.0** | 14.09 [4.57, 22.21] **excl.0** |

## Resource Cost (Mean Calls per Trajectory)

| Condition | Development | Validation | Held-Out |
|-----------|------------|------------|----------|
| BLIND_NO_GOVERNOR | 2.58 | 2.63 | 2.57 |
| BLIND_GOVERNOR | 3.00 | 3.00 | 3.00 |
| AWARE_NO_GOVERNOR | 2.53 | 2.48 | 2.47 |
| AWARE_GOVERNOR | 5.14 | 5.51 | 5.49 |

## Mean Utility

| Condition | Development | Validation | Held-Out |
|-----------|------------|------------|----------|
| BLIND_NO_GOVERNOR | -74.85 | -69.47 | -67.55 |
| BLIND_GOVERNOR | -93.84 | -80.91 | -81.71 |
| AWARE_NO_GOVERNOR | -74.87 | -70.65 | -73.03 |
| AWARE_GOVERNOR | -89.47 | -86.58 | -90.56 |

## Claim Hierarchy

| Claim | Description | Status |
|-------|-------------|--------|
| C0 | Artifact-valid experiment | **SUPPORTED** |
| C1 | Representation advantage (ΔIG > 0) | **SUPPORTED** |
| C2 | Governor improves decision quality (ΔDG > 0) | **REJECTED** |
| C3 | Governor benefit generalizes across topology | **REJECTED** |
| C4 | Governor x cognitive-state synergy | **REJECTED** |
| C5 | Governor improves utility at equal/lower cost | **REJECTED** |

## Conclusion

The I3.5.1 milestone produced an artifact-valid four-arm factorial experiment
across development (300 tasks), validation (150 tasks), and held-out (150 tasks)
splits. All 25 validity gates pass. Zero backend errors. Zero decoder failures.
All 2400 trajectories replay-match. IG/DG/TR identity holds for all 600 tasks.

The representation advantage exists: the aware condition provides more observable
information (ΔIG > 0, topology-cluster CI excludes 0 in all three phases).

However, the governor **systematically hurts** decision quality:
- ΔDG_gov|aware is negative in all three phases (CI excludes 0)
- ΔDG_gov|blind is negative in all three phases (CI excludes 0)
- The governor increases trajectory length (~5.5 vs ~2.5 mean calls)
- The governor decreases utility (~-90 vs ~-73 mean utility)
- The governor never produces a success that the no-governor condition does not also produce

The model makes better decisions on its own without the governor's advisory
recommendations. This is a governor design problem, not an infrastructure problem.
The artifact pipeline is trustworthy; the scientific result is negative.

## Key Artifacts

| Artifact | Path |
|---------|------|
| Scientific criteria | `experiments/v2b_i3_5_1/configs/v2b_i3_5_1_scientific_criteria_v1.json` |
| Canonical identity | `experiments/v2b_i3_5_1/manifests/v2b_i3_5_1_experiment_identity_v1.json` |
| Contamination ledger | `experiments/v2b_i3_5_1/baselines/contamination_ledger_v1.json` |
| Final claim hierarchy | `experiments/v2b_i3_5_1/final_claim_hierarchy_v1.json` |
| Development results | `experiments/v2b_i3_5_1/development/e21f63ff4fa9/` |
| Validation results | `experiments/v2b_i3_5_1/validation/f226e6b982bc/` |
| Held-out results | `experiments/v2b_i3_5_1/held_out/925ed28742c8/` |
| Qualification script | `scripts/qualify_i3_5_1.py` |
| Experiment runner | `scripts/run_v2b_i3_5_1_experiment.py` |
| I3.5.1 code package | `hrm_adaptive_memory/executive/i3_5_1/` |
| Test suite (101 tests) | `tests/unit/test_i351_*.py` |
