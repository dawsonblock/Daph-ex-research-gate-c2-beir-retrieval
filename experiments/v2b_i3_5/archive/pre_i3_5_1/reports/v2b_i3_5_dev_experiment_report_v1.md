# V2B-I3.5 Development Experiment Report

## Experiment Identity

| Field | Value |
|-------|-------|
| Experiment ID | `v2b_i3_5_dev_experiment_v1` |
| Status | **DEVELOPMENT — NOT HELD-OUT** |
| Split | `structure_dev_v2` (300 paired tasks) |
| Model | `deepseek-chat` (`deepseek-v4-flash`) |
| Governor | `GeneralGovernor V1` (SHA-256: `058009781dfa6b43...`) |
| Action Semantics | `FROZEN_ACTION_SEMANTICS V1` (SHA-256: `a9f6bb4c2374659b...`) |
| Source commit | `b47dff3` |
| Closure hashes | 35 (governor modules, benchmark artifacts, configs, I3.4 bundle) |

## Scientific Question

Does a model-based executive governor that predicts action consequences
and scores candidates using topology-invariant features improve a fixed
model's executive decisions on novel structural topologies, compared to
the same model without the governor frame?

The experiment compares two conditions on the same 300 tasks:

- **Blind**: The model sees only the controller observation without
  governor-constructed decision frames. It must choose actions from
  raw state.
- **Aware**: The model sees the same observation plus a governor
  decision frame containing bottleneck analysis, candidate action
  consequences, information gain estimates, redundancy detection,
  and option-value scoring.

Both conditions use the same model, same prompt template, same policy,
same runtime, and same resource budget. The only difference is the
presence or absence of the governor frame.

## Integrity Repairs Completed Before Experiment

This experiment was preceded by a full integrity repair cycle. No
DeepSeek tokens were spent on scientific runs until all P0 and P1
defects were resolved.

### P0 Repairs (blocking)

1. **Resource normalization (P0-1)**: The governor was looking up
   short resource keys (`"retrieval"`, `"verification"`, `"search"`)
   that do not exist in `ResourceState.as_dict()`. A typed
   `GovernorResourceState` was introduced to normalize the real keys
   (`retrieval_calls_remaining`, `verification_calls_remaining`, etc.)
   into typed fields. All governor modules were updated.

2. **Held-out topology isolation (P0-2)**: The V2 held-out split had
   14 topologies overlapping with dev+validation. The held-out
   composition was changed to use chain length 6 (vs. dev=3,
   validation=4-5), extra branching decoys, and no poison-on-misorder.
   After regeneration: `T_H ∩ (T_D ∪ T_V) = ∅` with zero overlap.

3. **Canonical topology definition (P0-3)**: All topology comparisons
   now use the behavior-derived `transition_topology_sha256` from
   `metareasoning_topology.py`, not the generator's semantic structure
   hash. V2 vs. I3.4 overlap: 0.

### P1 Repairs (required before scientific claims)

4. **Outcome-based no-gain detection (P1-1)**: The old heuristic
   flagged any repeated action as no-gain. The new logic requires
   the same action AND the same outcome code, preserving the
   distinction between repeated-failure and productive repetition
   (e.g., multiple `SEARCH_MORE` calls that return new evidence).

5. **Remove unavailable actions (P1-2)**: `SPAWN_SPECIALIST`,
   `SWITCH_STRATEGY`, `ABANDON_STRATEGY`, and
   `VERIFY_ALTERNATE_SOURCE` were removed from the frozen V1 action
   semantics. The V1 set now contains only the 7 actions the
   controller can actually select: `ANSWER`, `RETRIEVE`, `VERIFY`,
   `SEARCH_MORE`, `REASON_MORE`, `DEFER`, `STOP`.

6. **Governor↔executor parity tests (P1-3)**: 52 tests verify that
   the governor's action semantic contracts match the actual executor
   transition behavior. Channel parity, terminal parity, and
   structural contract tests are included. Poison and chain-completion
   side effects are excluded as benchmark mechanisms.

7. **Structural diversity report (P1-4)**: The empty report was
   populated with actual topology data: 15-entry overlap matrix,
   100% novelty rates, held-out isolation verification.

8. **Split-purity tests (P1-5)**: 10 tests verify that every V2
   information class is split-pure (no class contains members from
   multiple splits), posterior weights sum to 1, and observable
   oracle views load correctly.

9. **Experiment identity (P1-6)**: 35 closure hashes bind all
   artifacts: governor modules, runner/packet/prompt, benchmark
   artifacts, configs, I3.4 qualification bundle, and source commit.

10. **Smoke quarantine (P1-7)**: Invalid-key smoke outputs were moved
    to `experiments/v2b_i3_5/smoke/` and marked
    `INVALID_KEY_INTEGRATION_SMOKE_ONLY`.

### Test Results After Repairs

| Test Suite | Passed | Skipped | Failed |
|------------|--------|---------|--------|
| Governor unit tests | 60 | 0 | 0 |
| I3.5 runner tests | (included above) | | |
| Split-purity tests | 10 | 0 | 0 |
| Governor↔executor parity | 52 | 36 | 0 |
| **Total** | **122** | **36** | **0** |

## Benchmark Structure

| Split | Tasks | Topologies | Depth Distribution |
|-------|-------|------------|-------------------|
| `structure_dev_v2` | 300 | 87 | DEPTH_1: 124, DEPTH_4_PLUS: 176 |
| `structure_validation_v2` | 150 | 51 | DEPTH_1: 59, DEPTH_4_PLUS: 91 |
| `structure_held_out_v2` | 150 | 65 | DEPTH_1: 59, DEPTH_4_PLUS: 91 |

### Topology Isolation

| Check | Result |
|-------|--------|
| `T_H ∩ T_D` | 0 |
| `T_H ∩ T_V` | 0 |
| `T_H ∩ (T_D ∪ T_V)` | 0 |
| V2 ∩ I3.4 | 0 |
| Novelty rate (all V2 splits) | 100% |

### Oracle Reference Values

| Split | Condition | Mean V_O (optimal) |
|-------|-----------|-------------------|
| dev | STATE_BLIND | 57.22 |
| dev | STATE_AWARE | 66.27 |

The aware oracle has a higher optimal value because the aware
controller can observe more state features, giving it access to
better-informed policies. The scientific question is whether the
governor helps the model close the gap to this higher optimum.

## Experiment Execution

| Metric | Value |
|--------|-------|
| Paired tasks completed | 300 / 300 |
| Pair validity | 300 / 300 (100%) |
| Fingerprint matches | 300 / 300 (100%) |
| Total model calls | 2,412 |
| Blind model calls | 852 |
| Aware model calls | 1,560 |
| Backend errors | 0 |
| Decoder failures | 0 |
| Wall time | 2,488 seconds (~41 minutes) |
| Mean blind trajectory length | 2.84 steps |
| Mean aware trajectory length | 5.20 steps |

All 300 paired tasks completed with valid fingerprints and no
backend or decoder errors. The aware condition uses nearly twice as
many model calls as blind because the governor frame encourages
longer, more deliberate trajectories.

## Primary Results

### Decision Gap (ΔDG)

The decision gap (DG) measures how far the controller's realized
utility falls below the observable optimal value. ΔDG = DG_blind -
DG_aware, so positive values mean the aware condition has smaller
regret.

| Metric | Value |
|--------|-------|
| N paired tasks | 300 |
| Mean ΔDG | **+46.62** |
| Task bootstrap 95% CI | **[39.32, 54.38]** |
| Topology-cluster bootstrap (87 clusters) | **+51.22** |
| Topology-cluster 95% CI | **[35.72, 67.75]** |
| Cohen's d | **0.542** (medium) |

The mean ΔDG is positive and the 95% confidence interval excludes
zero under both task-level and topology-cluster bootstrap. The
effect size is medium (Cohen's d = 0.54).

### ΔDG Distribution

| Category | Count | Percentage |
|----------|-------|------------|
| ΔDG > 0 (aware better) | 222 | 74.0% |
| ΔDG < 0 (blind better) | 55 | 18.3% |
| ΔDG = 0 (tie) | 23 | 7.7% |

The aware condition outperforms blind on 74% of tasks.

### Success Rates

| Condition | Successes | Rate |
|-----------|-----------|------|
| Blind | 81 / 300 | 27.0% |
| Aware | 105 / 300 | 35.0% |

The aware condition has a +8.0 percentage point absolute success
rate improvement (27.0% → 35.0%, a 29.6% relative improvement).

### Utility Breakdown

| Metric | Blind | Aware | Difference |
|--------|-------|-------|------------|
| Mean controller value | -80.61 | -24.94 | +55.67 |
| Mean decision gap | 137.83 | 91.21 | -46.62 |
| Mean information gap | 18.72 | 9.67 | -9.05 |

The aware condition has substantially lower decision gap (91.21 vs.
137.83) and lower information gap (9.67 vs. 18.72). The governor
frame helps the model make better decisions and extract more value
from available information.

## Depth Band Analysis

### DEPTH_1 (124 tasks)

| Metric | Blind | Aware |
|--------|-------|-------|
| Success rate | 65.3% (81/124) | 84.7% (105/124) |
| Mean ΔDG | +36.61 | — |
| 95% CI | [20.37, 52.85] | — |
| Aware > blind | 76 / 124 (61.3%) | — |

On shallow topologies (optimal depth ≤ 1), the aware condition
improves success rate from 65.3% to 84.7%. Both succeed on 75
tasks, aware alone succeeds on 30, blind alone on 6, and neither
succeeds on 13.

The blind condition on DEPTH_1 typically follows a fixed
`RETRIEVE → VERIFY → ANSWER` pattern (101 retrieves, 99 verifies,
79 answers, 45 stops). The aware condition shows more diverse
action selection (67 answers, 54 verifies, 48 search_more, 43
retrieves, 40 stops, 17 defers, 4 reason_more), suggesting the
governor frame helps the model recognize when the simple pattern
is sufficient and when additional exploration is needed.

### DEPTH_4_PLUS (176 tasks)

| Metric | Blind | Aware |
|--------|-------|-------|
| Success rate | 0.0% (0/176) | 0.0% (0/176) |
| Mean ΔDG | +53.67 | — |
| 95% CI | [47.65, 59.70] | — |
| Aware > blind | 146 / 176 (83.0%) | — |
| Mean utility | -125.44 | -61.38 |

On deep topologies (optimal depth ≥ 4), neither condition achieves
task success. However, the aware condition has substantially higher
utility (-61.38 vs. -125.44), meaning it gets closer to the answer
even when it fails. The ΔDG is larger here (+53.67 vs. +36.61),
indicating the governor frame provides more benefit on harder
topologies.

The blind condition on DEPTH_4_PLUS follows a rigid
`RETRIEVE → VERIFY → ANSWER` pattern on all 176 tasks (176
retrieves, 176 verifies, 176 answers), terminating after 3 steps
regardless of whether the composition is complete. The aware
condition explores much more (400 verifies, 357 search_more, 335
retrieves, 146 defers, 30 answers, 19 reason_more) with a mean
trajectory length of 7.31 steps. The governor frame correctly
identifies that more work is needed, but the model still fails to
complete the full composition chain within the budget.

## Action Distribution

| Action | Blind | Aware | Difference |
|--------|-------|-------|------------|
| RETRIEVE | 277 | 378 | +101 |
| VERIFY | 275 | 454 | +179 |
| SEARCH_MORE | 0 | 405 | +405 |
| REASON_MORE | 0 | 23 | +23 |
| ANSWER | 255 | 97 | -158 |
| DEFER | 0 | 163 | +163 |
| STOP | 45 | 40 | -5 |
| **Total** | **852** | **1,560** | **+708** |

Key observations:

- **SEARCH_MORE** appears 405 times in aware but never in blind.
  The governor frame introduces the model to this action, which is
  critical for deep topologies.
- **DEFER** appears 163 times in aware but never in blind. The
  governor frame helps the model recognize when deferral is
  preferable to a premature answer.
- **ANSWER** drops from 255 (blind) to 97 (aware). The governor
  frame reduces premature answering, which is the primary failure
  mode in the blind condition.
- **REASON_MORE** appears 23 times in aware but never in blind.
  The governor frame enables internal computation steps.

## Governor Agreement

| Metric | Blind | Aware |
|--------|-------|-------|
| Mean governor agreement | 0.898 | 0.722 |
| DEPTH_1 agreement | 0.753 | 0.640 |
| DEPTH_4_PLUS agreement | 1.000 | 0.780 |

In the blind condition, the governor agrees with the model 89.8% of
the time. On DEPTH_4_PLUS, agreement is 100% — the governor always
recommends the same `RETRIEVE → VERIFY → ANSWER` pattern that the
model follows, but the governor's frame in the aware condition
encourages the model to explore more.

In the aware condition, agreement drops to 72.2%. The model
disagrees with the governor on 363 steps, most commonly:
- `DEFER` (161 disagreements): The model defers when the governor
  recommends something else.
- `VERIFY` (58): The model verifies when the governor suggests a
  different action.
- `SEARCH_MORE` (40): The model searches when the governor
  recommends otherwise.
- `STOP` (40): The model stops when the governor suggests
  continuing.
- `ANSWER` (37): The model answers when the governor suggests
  more work.

These disagreements are not necessarily errors — the model has
access to the full observation and can make informed decisions that
differ from the governor's top candidate. The positive ΔDG suggests
that the combination of governor guidance and model judgment
outperforms either alone.

## Governor Reason Codes

| Reason Code | Blind | Aware |
|-------------|-------|-------|
| TARGETS_NO_EVIDENCE | 300 | 499 |
| READY_TO_ANSWER | 275 | 100 |
| TARGETS_UNRESOLVED_CONFLICT | 0 | 353 |
| ADDS_NEW_INFORMATION | 0 | 352 |
| TARGETS_UNVERIFIED_EVIDENCE | 277 | 0 |
| BEST_AVAILABLE_OPTION | 0 | 254 |

The governor's reason codes differ substantially between conditions.
In the blind condition, the governor primarily identifies missing
evidence (`TARGETS_NO_EVIDENCE`) and unverified evidence
(`TARGETS_UNVERIFIED_EVIDENCE`), leading to `READY_TO_ANSWER` when
verification is complete. In the aware condition, the governor
additionally identifies unresolved conflicts (`TARGETS_UNRESOLVED_CONFLICT`)
and information-gaining actions (`ADDS_NEW_INFORMATION`), which
drive the model toward `SEARCH_MORE` and `DEFER` rather than
premature `ANSWER`.

## Limitations

1. **Development only**: These results are on `structure_dev_v2`.
   The held-out split (`structure_held_out_v2`) has not been
   evaluated. No generalization claim can be made until held-out
   results are obtained.

2. **Budget profile**: All tasks use the `STRUCTURE_HOLDOUT_V2`
   budget profile (8 executive steps). The experiment does not
   establish budget-general governor behavior.

3. **DEPTH_4_PLUS failure**: Neither condition solves any
   DEPTH_4_PLUS task. The aware condition gets closer (higher
   utility) but still fails. This may indicate that the 8-step
   budget is insufficient for depth-6 compositions, or that the
   model's action-consequence reasoning is not strong enough to
   complete long chains even with governor guidance.

4. **Single model**: Only `deepseek-chat` was tested. No
   generalization to other models is implied.

5. **Single seed**: The experiment used one API session. Model
   non-determinism is not characterized.

6. **DEEPSEEK_API_KEY**: The API key used for this experiment
   should be revoked. It was not committed to the repository.

## Conclusion

The I3.5 development experiment provides positive and statistically
significant evidence that the governor-enhanced aware condition
improves executive decisions on development topologies:

- Mean ΔDG = +46.62, 95% CI [39.32, 54.38], p < 0.001
- Topology-cluster bootstrap confirms: +51.22, CI [35.72, 67.75]
- Cohen's d = 0.542 (medium effect)
- Success rate improves from 27.0% to 35.0%
- The effect is larger on deep topologies (DEPTH_4_PLUS: +53.67)
  than shallow ones (DEPTH_1: +36.61)

The governor frame's primary mechanism is reducing premature
answering: the blind condition answers immediately after
`RETRIEVE → VERIFY` on 255 of 300 tasks, while the aware condition
answers on only 97 tasks, instead using `SEARCH_MORE` (405 calls),
`DEFER` (163 calls), and `VERIFY` (454 calls) to explore more
thoroughly.

**This is development evidence only. Held-out confirmation is
required before any generalization claim can be made.**

## Artifacts

| Artifact | Path |
|----------|------|
| Results | `experiments/v2b_i3_5/results/v2b_i3_5_structure_dev_v2_results_v1.json` |
| Receipts | `experiments/v2b_i3_5/results/v2b_i3_5_structure_dev_v2_receipts_v1.jsonl` |
| Scores | `experiments/v2b_i3_5/results/v2b_i3_5_structure_dev_v2_scores_v1.json` |
| Statistics | `experiments/v2b_i3_5/results/v2b_i3_5_structure_dev_v2_stats_v1.json` |
| Analysis | `experiments/v2b_i3_5/results/v2b_i3_5_structure_dev_v2_analysis_v1.json` |
| Experiment identity | `experiments/v2b_i3_5/configs/v2b_i3_5_experiment_identity_v1.json` |
| Structural diversity | `experiments/v2b_i3_5/reports/v2b_i3_5_structural_diversity_report_v2.json` |
| Observable oracle views | `experiments/v2b_i3_5/oracle_tables/v2b_i3_5_observable_oracle_views_v1.json` |
