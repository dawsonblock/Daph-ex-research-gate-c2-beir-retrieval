# R13-A v2 Preliminary Results

## Status

- Tournament stopped at **403/450 intended executions**.
- Evidence is partial but stable enough for Q4/Q5 screening.
- Pre-hardening evidence quarantined.

## Q4 — Do heterogeneous operators improve on vanilla resampling?

| Operator | All-state rescue | Opportunity-conditional rescue (baseline wrong) | Break rate | Mean tokens |
|----------|-----------------|------------------------------------------------|------------|-------------|
| STOP | — | — | — | 0 |
| **SAMPLE_STANDARD** | **6.2%** | **11.1%** | 8.6% | 408 |
| VERIFY_TARGETED | 5.0% | 9.1% | 1.2% | 694 |
| CRITIQUE_RETRY | 2.5% | 4.5% | 0.0% | 566 |
| SAMPLE_DIVERSE | 2.5% | 4.4% | **14.8%** | 492 |

**Verdict**: SAMPLE_STANDARD is the strongest fixed continuation. VERIFY_TARGETED is close (within 2 percentage points on conditional rescue) but costs more tokens. CRITIQUE_RETRY and SAMPLE_DIVERSE do not beat vanilla resampling. **SAMPLE_DIVERSE is harmful**: 14.8% break rate vs 2.5% rescue rate.

**Q4 gate**: weakly open. No operator materially beats SAMPLE_STANDARD, but VERIFY_TARGETED is competitive and has a much lower break rate.

## Q5 — Is there oracle routing headroom?

| λ | Oracle mean U | Best fixed | Δ headroom | Best fixed operator |
|---|---------------|------------|------------|---------------------|
| 0.0 | 0.5556 | 0.4875 | +0.0681 | VERIFY_TARGETED |
| 0.01 | 0.5548 | 0.4806 | +0.0742 | VERIFY_TARGETED |
| 0.05 | 0.5518 | 0.4528 | +0.0990 | VERIFY_TARGETED |
| 0.10 | 0.5481 | 0.4444 | +0.1036 | STOP |
| 0.20 | 0.5406 | 0.4444 | +0.0961 | STOP |

**Oracle action distribution**:
- STOP: 88.9%
- SAMPLE_STANDARD: 6.2%
- CRITIQUE_RETRY: 2.5%
- VERIFY_TARGETED: 2.5%

**Verdict**: Oracle heterogeneous routing gives a real but modest headroom (+0.07–0.10 utility, or ~7–10 percentage points). However, the oracle is mostly STOP (88.9%), so action heterogeneity is low. The routing problem is not strongly heterogeneous.

## Bootstrap CIs (clustered by checkpoint, 1000 replicates)

| Operator | Rescue% | P5% | P95% |
|----------|---------|-----|------|
| SAMPLE_STANDARD | 6.2% | 2.5% | 11.1% |
| VERIFY_TARGETED | 5.0% | 1.2% | 9.0% |
| CRITIQUE_RETRY | 2.5% | 0.0% | 6.2% |
| SAMPLE_DIVERSE | 2.5% | 0.0% | 4.9% |

## Preliminary conclusion

1. **The R12 bottleneck persists at v2**: rescue rates are ~3–6% all-state, ~4–11% conditional. Stopping is correct ~89% of the time.

2. **No operator clearly dominates SAMPLE_STANDARD**. VERIFY_TARGETED is the only serious alternative, with a better break profile (1.2% vs 8.6%) but higher token cost and lower rescue rate.

3. **Oracle headroom is real but modest** (+0.07–0.10 utility). The low action heterogeneity means the learned router will mostly learn to STOP.

4. **SAMPLE_DIVERSE should be excluded** from the learned router: its break rate (14.8%) exceeds its rescue rate (2.5%).

## Recommendation

Before building a learned executive, consider one of the following:

1. **Improve operator quality**: CRITIQUE_RETRY and SAMPLE_DIVERSE are underperforming. A better critique (e.g., forced correction only after genuine error detection) or a better diversity mechanism (e.g., strategy-conditioned on failure mode) might lift rescue rates.

2. **Reduce operator harm**: SAMPLE_DIVERSE breaks too often. Either remove it or redesign it to not override a correct current answer unless the new candidate is strongly verified.

3. **Collect more data**: 403 executions is enough for screening but sparse for training. If the oracle headroom remains after a full 450-execution tournament, it may justify a tiny STOP/CONTINUE + operator selection model.

4. **If the next full tournament shows the same pattern**, the correct engineering answer is likely: **use a simple confidence/entropy stopping rule and treat SAMPLE_STANDARD as the only high-value continuation**. Adding a heterogeneous router would add complexity without decisive value.

## Next steps

- Finish the remaining 47 executions if clean completion is desired.
- Run replicates 123 and 2024 for variance estimation before any router training.
- Decide whether to improve the operators (R13-A v3) or accept the current action set and build a minimal router.
