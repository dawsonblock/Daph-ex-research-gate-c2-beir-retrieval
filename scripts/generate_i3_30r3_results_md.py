#!/usr/bin/env python3
"""Generate I3_30R3_RESULTS.md from authority_analysis.json.

Step 9 fix: Eliminate manually copied aggregate counts.
The markdown report is generated programmatically from the
authoritative analysis JSON, ensuring internal consistency.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def generate_results_md(analysis_path: Path, gate_path: Path, output_path: Path):
    """Generate the results markdown from analysis JSON."""
    with open(analysis_path) as f:
        analysis = json.load(f)
    with open(gate_path) as f:
        gate_data = json.load(f)
    gates = gate_data.get("gates", gate_data)

    # Extract data
    primary = analysis.get("primary_comparison", {})
    secondary = analysis.get("secondary_comparison", {})
    event_class = analysis.get("event_classification", {})
    rates = analysis.get("authority_rates", {})
    strata = analysis.get("stratum_breakdown", {})
    aggregate = analysis.get("aggregate", {})

    lines = []
    lines.append("# I3.30R3: Authority Isolation Study — Results")
    lines.append("")
    lines.append("**This report is auto-generated from `authority_analysis.json`.**")
    lines.append("**Do not edit manually — regenerate with `scripts/generate_i3_30r3_results_md.py`.**")
    lines.append("")

    # Primary comparison
    lines.append("## Primary Comparison: V3-AUTH vs V3-SHADOW")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| ATE_authority | {primary.get('ate', 0):.4f} |")
    ci = primary.get("ci", [0, 0])
    lines.append(f"| 95% CI | [{ci[0]:.4f}, {ci[1]:.4f}] |")
    lines.append(f"| n | {primary.get('n', 0)} |")
    lines.append(f"| Rescues | {primary.get('rescues', 0)} |")
    lines.append(f"| Breaks | {primary.get('breaks', 0)} |")
    lines.append(f"| Both success | {primary.get('both_success', 0)} |")
    lines.append(f"| Both fail | {primary.get('both_fail', 0)} |")
    lines.append("")

    # Secondary comparison
    lines.append("## Secondary Comparison: V3-SHADOW vs V1")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| ΔU(SHADOW - V1) | {secondary.get('delta_u', 0):.4f} |")
    ci2 = secondary.get("ci", [0, 0])
    lines.append(f"| 95% CI | [{ci2[0]:.4f}, {ci2[1]:.4f}] |")
    lines.append(f"| n | {secondary.get('n', 0)} |")
    lines.append(f"| Rescues | {secondary.get('rescues', 0)} |")
    lines.append(f"| Breaks | {secondary.get('breaks', 0)} |")
    lines.append("")

    # Event classification
    lines.append("## Authority Event Classification")
    lines.append("")
    lines.append(f"| Classification | Count |")
    lines.append(f"|---------------|-------|")
    for cls, count in sorted(event_class.items()):
        lines.append(f"| {cls} | {count} |")
    lines.append("")

    # Authority rates
    lines.append("## Authority Rates")
    lines.append("")
    lines.append(f"| Rate | Value |")
    lines.append(f"|------|-------|")
    lines.append(f"| Certificate coverage | {rates.get('certificate_coverage', 0):.4f} |")
    lines.append(f"| Force rate | {rates.get('force_rate', 0):.4f} |")
    lines.append(f"| Effective intervention rate | {rates.get('effective_intervention_rate', 0):.4f} |")
    lines.append("")

    # Stratum breakdown
    lines.append("## Stratum Breakdown")
    lines.append("")
    lines.append(f"| Stratum | V1 | SHADOW | AUTH |")
    lines.append(f"|---------|-----|--------|------|")
    for stratum in sorted(strata.keys()):
        s = strata[stratum]
        v1 = s.get("v1", {})
        sh = s.get("v3_shadow", {})
        hd = s.get("v3_hard", {})
        lines.append(
            f"| {stratum} | "
            f"{v1.get('successes', 0)}/{v1.get('n', 0)} "
            f"({v1.get('success_rate', 0)*100:.2f}%) | "
            f"{sh.get('successes', 0)}/{sh.get('n', 0)} "
            f"({sh.get('success_rate', 0)*100:.2f}%) | "
            f"{hd.get('successes', 0)}/{hd.get('n', 0)} "
            f"({hd.get('success_rate', 0)*100:.2f}%) |"
        )
    lines.append("")

    # Aggregate
    lines.append("## Aggregate Success Rates")
    lines.append("")
    lines.append(f"| Arm | Success | Total | Rate | Mean Utility |")
    lines.append(f"|-----|---------|-------|------|--------------|")
    for arm in ["v1", "v3_shadow", "v3_hard"]:
        a = aggregate.get(arm, {})
        lines.append(
            f"| {arm} | {a.get('successes', 0)} | {a.get('n', 0)} | "
            f"{a.get('success_rate', 0)*100:.2f}% | "
            f"{a.get('mean_utility', 0):.2f} |"
        )
    lines.append("")

    # Gates
    lines.append("## Gate Evaluation")
    lines.append("")
    lines.append(f"| Gate | Name | Result | Value |")
    lines.append(f"|------|------|--------|-------|")
    passed = 0
    failed = 0
    pending = 0
    for gname in sorted(gates.keys()):
        g = gates[gname]
        result = g.get("result", "?")
        if result == "PASS":
            passed += 1
        elif result == "FAIL":
            failed += 1
        elif result == "PENDING":
            pending += 1
        value = g.get("value", "")
        if isinstance(value, float):
            value = f"{value:.4f}"
        lines.append(f"| {gname} | {g.get('name', '')} | {result} | {value} |")
    lines.append("")
    lines.append(f"**{passed} passed, {failed} failed, {pending} pending.**")
    lines.append("")

    # Write
    output_path.write_text("\n".join(lines))
    print(f"Generated: {output_path}")
    print(f"  {passed} passed, {failed} failed, {pending} pending")


if __name__ == "__main__":
    analysis_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("experiments/i3_30r3/analysis/authority_analysis.json")
    gate_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("experiments/i3_30r3/analysis/gate_evaluation.json")
    output_path = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("experiments/i3_30r3/I3_30R3_RESULTS.md")
    generate_results_md(analysis_path, gate_path, output_path)
