# DAPH V2B-I3.4.2 Held-Out Structure Preregistered Analysis

**Schema:** `DAPH_V2B_I3_4_HOST_PREREGISTERED_REPORT_V1`
**Generated:** 2026-08-15T06:03:23.897447+00:00
**Frozen at:** `7af42ed` (FINAL_PRE_HELD_OUT_FREEZE)
**Analysis file:** `experiments/v2b_i3_4/results/v2b_i3_4_host_preregistered_analysis_v1.json`

## 1. Primary Inferential Result (Topology-Cluster Bootstrap)

The frozen scientific criteria specify that for HELD_OUT_STRUCTURE, the **primary inferential CI** is the topology-cluster bootstrap over the 51 unseen topologies, not the ordinary 150-task bootstrap.

| Statistic | Value |
|-----------|-------|
| Resampling unit | topology_cluster (51 clusters) |
| Iterations | 10,000 |
| Point estimate (topology-uniform) | -11.4032 |
| **95% CI (primary)** | **[-15.8260, -7.2785]** |
| CI excludes zero | 1.0 |
| **Verdict** | **REJECTED_ON_HELD_OUT_STRUCTURE** |

### ΔIG (topology-cluster)

| Statistic | Value |
|-----------|-------|
| Point estimate | +12.0327 |
| 95% CI | [+7.6652, +16.6961] |

The aware condition still has a strongly positive ΔIG — it provides more information. But ΔDG is entirely negative.

### Task-level (descriptive, not primary)

| Statistic | Value |
|-----------|-------|
| Point estimate (task-weighted) | -13.4285 |
| 95% CI (descriptive) | [-17.2838, -9.7939] |

Both CIs agree in direction and significance. The topology-cluster CI is slightly narrower because it accounts for within-cluster correlation.

## 2. Depth-Stratified Analysis

The criteria require DEPTH_1 and DEPTH_4_PLUS to be reported separately. Do not pool into a single structural claim.

### DEPTH_1 (59 tasks, 30 topologies)

| Metric | Blind | Aware | Δ |
|--------|------:|------:|---:|
| DG | -22.7203 | -9.2663 | -13.4541 |
| IG | 15.8759 | 0.0000 | +15.8759 |

| CI Method | Lower | Upper | Excludes 0 | Verdict |
|-----------|------:|------:|:----------:|---------|
| Task-level | -19.9772 | -7.4916 | True | REJECTED |
| **Topology-cluster** | **-14.4440** | **-3.3570** | **True** | **REJECTED** |

### DEPTH_4_PLUS (91 tasks, 21 topologies)

| Metric | Blind | Aware | Δ |
|--------|------:|------:|---:|
| DG | 152.8869 | 166.2989 | -13.4120 |
| IG | 42.0491 | 29.5022 | +12.5469 |

| CI Method | Lower | Upper | Excludes 0 | Verdict |
|-----------|------:|------:|:----------:|---------|
| Task-level | -18.2890 | -8.9308 | True | REJECTED |
| **Topology-cluster** | **-22.1838** | **-9.4237** | **True** | **REJECTED** |

### Depth comparison

| Band | N tasks | N topologies | ΔDG | Topo CI | Verdict |
|------|--------:|-------------:|----:|---------|---------|
| DEPTH_1 | 59 | 30 | -13.4541 | [-14.4440, -3.3570] | REJECTED |
| DEPTH_4_PLUS | 91 | 21 | -13.4120 | [-22.1838, -9.4237] | REJECTED |

**Both depth bands are rejected.** The negative ΔDG is not concentrated in one horizon group — it appears in both DEPTH_1 and DEPTH_4_PLUS. This means the executive failure occurs even at short horizons on novel topologies, not only on long planning sequences.

## 3. Breakdown by Optimal Action

| Optimal Action | N tasks | ΔDG | ΔIG |
|----------------|--------:|----:|----:|
| ANSWER | 26 | -28.8121 | +34.3075 |
| DEFER | 10 | -4.4675 | +4.4685 |
| REASON_MORE | 24 | -2.2194 | +1.5519 |
| RETRIEVE | 25 | -21.0474 | +19.9430 |
| SEARCH_MORE | 23 | -10.0274 | +9.0091 |
| STOP | 23 | +0.0000 | +0.0000 |
| VERIFY | 19 | -21.6005 | +20.9863 |

The negative ΔDG is largest for tasks where the optimal action is ANSWER (-28.81), RETRIEVE (-21.05), or VERIFY (-21.60). These are tasks where the aware condition's richer state leads the model to take the wrong action when the topology is novel.

STOP tasks show exactly zero ΔDG and ΔIG — these are tasks where neither condition can succeed, so both conditions have identical regret.

## 4. Breakdown by Difficulty Band

| Band | N tasks | ΔDG | ΔIG | ΔTR |
|------|--------:|----:|----:|----:|
| EASY | 99 | -16.0040 | +15.6396 | -0.3644 |
| MEDIUM | 51 | -8.4290 | +10.3947 | +1.9657 |

Note: No HARD or TIE difficulty tasks exist in held_out_structure (0 tasks in each band). All claims are limited to EASY and MEDIUM difficulty.

## 5. Structural Limitations

Per the frozen criteria:

- No HARD or TIE difficulty claims are possible (zero tasks)
- DEPTH_1 claims are topology generalization within the same horizon class
- DEPTH_4_PLUS claims are topology+horizon extrapolation
- Both depth bands are reported separately (not pooled)
- Topology-cluster bootstrap is the primary inferential CI
- Both task-level descriptive CI and topology-cluster inferential CI are reported

## 6. Scientific Verdict

### Primary criterion

**LCB_95(ΔDG) > 0 under topology-cluster bootstrap: FAIL**

- Topology-cluster CI: [-15.8260, -7.2785] — entirely negative
- Task-level CI: [-17.2838, -9.7939] — entirely negative
- Both depth bands: REJECTED under topology-cluster bootstrap

### Formal status

**REJECTED_ON_HELD_OUT_STRUCTURE**

The executive-metareasoning hypothesis (DG_aware < DG_blind) is rejected on unseen topological structures under the preregistered topology-cluster bootstrap.

### What is rejected

The claim that the current DeepSeek executive generalizes its use of DAPH cognitive state to unseen control topologies.

### What is NOT rejected

- The representation works: ΔIG = +12.03 (CI [7.67, 16.70]), entirely positive
- The development effect is real (on the development distribution)
- The aware condition provides more useful information than the blind condition

### Diagnosis

The aware condition reliably reduces information gap (IG) on all splits, including held_out_structure. But on unseen topological structures, that additional information leads to **worse** executive decisions (larger DG). The information advantage is approximately offset by the decision disadvantage, leaving total regret nearly unchanged.

This pattern is consistent across both depth bands, suggesting the failure is not horizon-specific but topology-specific: the executive has learned local state→action mappings that work on familiar structures but misfire on novel ones.

### Claim boundary

| Claim | Status |
|-------|--------|
| Representation advantage (ΔIG > 0) | SUPPORTED on all splits |
| Executive exploitation (ΔDG > 0) | SUPPORTED on development, REJECTED on held_out_structure |
| Control efficiency | NOT SUPPORTED (requires ΔDG > 0) |
| General executive metareasoning capability | NOT SUPPORTED |

---

*This analysis follows the frozen scientific criteria V2. The topology-cluster bootstrap is the primary inferential CI for structural-generalization claims. Both task-level descriptive and topology-cluster inferential CIs are reported. Depth bands are reported separately per the prohibition on pooling.*
