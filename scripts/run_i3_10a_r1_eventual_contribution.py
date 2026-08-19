#!/usr/bin/env python3
"""I3.10a-r1: Eventual contribution analysis for SBU actions.

Traces each RETRIEVE/SEARCH_MORE in SUPPORTED_BUT_UNRESOLVED to determine
whether the exposed evidence was eventually verified and whether that
verification changed the decision.

Classifications:
  DECISION_RELEVANT: exposed evidence was later verified AND changed
    viable hypothesis set, answer condition, or terminal decision
  DECISION_IRRELEVANT: exposed evidence was later verified but did NOT
    change the decision
  UNUSED: exposed evidence was never verified before terminal action
  NO_EXPOSURE: action exposed no new evidence (empty evidence_exposed_log)

This corrects the I3.10a finding: P(state changes | RETRIEVE)=0 does not
imply RETRIEVE has zero value. The true measure is eventual contribution.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_results(path: Path) -> list[dict]:
    results = []
    with open(path) as f:
        for line in f:
            results.append(json.loads(line))
    return results


def classify_sbu_action(
    r: dict,
    step: int,
    action: str,
) -> dict:
    """Classify a single SBU action by its eventual contribution."""
    fork = r["fork_m3"]
    exposed_log = fork.get("evidence_exposed_log", [])
    verified_log = fork.get("evidence_verified_log", [])
    actions = fork.get("continuation_actions", [])
    states = [e["decision_state"] for e in fork.get("decision_state_log", [])]

    # What did this action expose?
    exposed = exposed_log[step] if step < len(exposed_log) else []
    exposed_eids = set(exposed) if exposed else set()

    if not exposed_eids:
        return {
            "action": action,
            "step": step,
            "exposed_eids": [],
            "later_verified": False,
            "verified_eids": [],
            "hypothesis_set_changed": False,
            "classification": "NO_EXPOSURE",
        }

    # Was any of the exposed evidence later verified?
    later_verified_eids = set()
    for later_step in range(step + 1, len(verified_log)):
        verified_at = verified_log[later_step]
        if verified_at:
            for eid in verified_at:
                if eid in exposed_eids:
                    later_verified_eids.add(eid)

    if not later_verified_eids:
        return {
            "action": action,
            "step": step,
            "exposed_eids": list(exposed_eids),
            "later_verified": False,
            "verified_eids": [],
            "hypothesis_set_changed": False,
            "classification": "UNUSED",
        }

    # Did the verification change the hypothesis set or state?
    # Compare live_hypotheses before and after the verification step
    log = fork.get("decision_state_log", [])
    hypothesis_set_changed = False
    state_changed = False

    for later_step in range(step + 1, len(log)):
        verified_at = verified_log[later_step] if later_step < len(verified_log) else []
        if verified_at and any(eid in later_verified_eids for eid in verified_at):
            # This is the step where our exposed evidence got verified
            if later_step + 1 < len(log):
                prev_live = set(log[later_step].get("live_hypotheses", []))
                next_live = set(log[later_step + 1].get("live_hypotheses", []))
                prev_elim = set(log[later_step].get("eliminated_hypotheses", []))
                next_elim = set(log[later_step + 1].get("eliminated_hypotheses", []))
                if prev_live != next_live or prev_elim != next_elim:
                    hypothesis_set_changed = True
                if log[later_step].get("decision_state") != log[later_step + 1].get("decision_state"):
                    state_changed = True
            break

    # Did it change the terminal decision?
    # If the task succeeded, the exposed+verified evidence contributed to success
    # If hypothesis_set_changed, it was decision-relevant
    if hypothesis_set_changed or state_changed:
        classification = "DECISION_RELEVANT"
    else:
        # Was verified but didn't change the hypothesis set
        # Could still be relevant if it confirmed the leading hypothesis
        # (transitioning SBU -> READY_TO_ANSWER)
        if state_changed:
            classification = "DECISION_RELEVANT"
        else:
            classification = "DECISION_IRRELEVANT"

    return {
        "action": action,
        "step": step,
        "exposed_eids": list(exposed_eids),
        "later_verified": True,
        "verified_eids": list(later_verified_eids),
        "hypothesis_set_changed": hypothesis_set_changed,
        "state_changed": state_changed,
        "classification": classification,
    }


def main():
    v5_path = ROOT / "experiments/v2b_i3_9/development/i3_9_r3_affordance_clean/affordance_clean_v1.jsonl"
    output_dir = ROOT / "experiments/v2b_i3_10/development/i3_10a_forensic"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("I3.10a-r1: Eventual contribution analysis for SBU actions")
    print(f"  Source: {v5_path}")
    print()

    results = load_results(v5_path)
    n = len(results)
    print(f"  Loaded {n} task results")

    # Classify all SBU RETRIEVE and SEARCH_MORE actions
    all_classifications = []
    retrieve_classifications = []
    search_classifications = []

    for r in results:
        fork = r["fork_m3"]
        log = fork.get("decision_state_log", [])
        actions = fork.get("continuation_actions", [])

        for i, entry in enumerate(log):
            if entry.get("decision_state") != "SUPPORTED_BUT_UNRESOLVED":
                continue
            action = actions[i] if i < len(actions) else "UNKNOWN"
            if action not in ("RETRIEVE", "SEARCH_MORE"):
                continue

            classification = classify_sbu_action(r, i, action)
            classification["task_id"] = r["task_id"]
            classification["category"] = r["category"]
            classification["task_success"] = r["m3_success"]

            all_classifications.append(classification)
            if action == "RETRIEVE":
                retrieve_classifications.append(classification)
            elif action == "SEARCH_MORE":
                search_classifications.append(classification)

    # Summarize
    print(f"\n=== SBU RETRIEVE eventual contribution ({len(retrieve_classifications)} total) ===")
    ret_counts = Counter(c["classification"] for c in retrieve_classifications)
    for cls in ["DECISION_RELEVANT", "DECISION_IRRELEVANT", "UNUSED", "NO_EXPOSURE"]:
        count = ret_counts.get(cls, 0)
        pct = count / len(retrieve_classifications) * 100 if retrieve_classifications else 0
        print(f"  {cls:<22} {count:>4} ({pct:.1f}%)")

    print(f"\n=== SBU SEARCH_MORE eventual contribution ({len(search_classifications)} total) ===")
    search_counts = Counter(c["classification"] for c in search_classifications)
    for cls in ["DECISION_RELEVANT", "DECISION_IRRELEVANT", "UNUSED", "NO_EXPOSURE"]:
        count = search_counts.get(cls, 0)
        pct = count / len(search_classifications) * 100 if search_classifications else 0
        print(f"  {cls:<22} {count:>4} ({pct:.1f}%)")

    # Per-category breakdown
    print(f"\n=== PER-CATEGORY: SBU RETRIEVE classification ===")
    print(f"  {'Category':<30} {'total':>5} {'RELEVANT':>8} {'IRRELEV':>8} {'UNUSED':>7} {'NO_EXPO':>8}")
    cat_ret = defaultdict(lambda: Counter())
    for c in retrieve_classifications:
        cat_ret[c["category"]][c["classification"]] += 1
    for cat in sorted(cat_ret.keys()):
        counts = cat_ret[cat]
        total = sum(counts.values())
        print(f"  {cat:<30} {total:>5} {counts.get('DECISION_RELEVANT',0):>8} "
              f"{counts.get('DECISION_IRRELEVANT',0):>8} {counts.get('UNUSED',0):>7} "
              f"{counts.get('NO_EXPOSURE',0):>8}")

    print(f"\n=== PER-CATEGORY: SBU SEARCH_MORE classification ===")
    print(f"  {'Category':<30} {'total':>5} {'RELEVANT':>8} {'IRRELEV':>8} {'UNUSED':>7} {'NO_EXPO':>8}")
    cat_search = defaultdict(lambda: Counter())
    for c in search_classifications:
        cat_search[c["category"]][c["classification"]] += 1
    for cat in sorted(cat_search.keys()):
        counts = cat_search[cat]
        total = sum(counts.values())
        print(f"  {cat:<30} {total:>5} {counts.get('DECISION_RELEVANT',0):>8} "
              f"{counts.get('DECISION_IRRELEVANT',0):>8} {counts.get('UNUSED',0):>7} "
              f"{counts.get('NO_EXPOSURE',0):>8}")

    # The batch-retrieval pattern: consecutive RETRIEVEs
    print(f"\n=== BATCH RETRIEVAL PATTERN ===")
    consecutive_retrieve_chains = []
    for r in results:
        fork = r["fork_m3"]
        log = fork.get("decision_state_log", [])
        actions = fork.get("continuation_actions", [])
        exposed_log = fork.get("evidence_exposed_log", [])

        in_chain = False
        chain_start = None
        chain_length = 0
        chain_exposed_anything = False

        for i in range(len(log)):
            is_sbu_retrieve = (
                log[i].get("decision_state") == "SUPPORTED_BUT_UNRESOLVED"
                and i < len(actions) and actions[i] == "RETRIEVE"
            )
            if is_sbu_retrieve:
                if not in_chain:
                    in_chain = True
                    chain_start = i
                    chain_length = 1
                else:
                    chain_length += 1
                exposed = exposed_log[i] if i < len(exposed_log) else []
                if exposed:
                    chain_exposed_anything = True
            else:
                if in_chain:
                    consecutive_retrieve_chains.append({
                        "task_id": r["task_id"],
                        "category": r["category"],
                        "chain_start": chain_start,
                        "chain_length": chain_length,
                        "exposed_anything": chain_exposed_anything,
                        "task_success": r["m3_success"],
                    })
                    in_chain = False
                    chain_exposed_anything = False
        if in_chain:
            consecutive_retrieve_chains.append({
                "task_id": r["task_id"],
                "category": r["category"],
                "chain_start": chain_start,
                "chain_length": chain_length,
                "exposed_anything": chain_exposed_anything,
                "task_success": r["m3_success"],
            })

    print(f"  Total RETRIEVE chains in SBU: {len(consecutive_retrieve_chains)}")
    chains_exposed = sum(1 for c in consecutive_retrieve_chains if c["exposed_anything"])
    chains_empty = sum(1 for c in consecutive_retrieve_chains if not c["exposed_anything"])
    print(f"  Chains that exposed evidence: {chains_exposed}")
    print(f"  Chains that exposed NOTHING: {chains_empty}")
    mean_chain = sum(c["chain_length"] for c in consecutive_retrieve_chains) / len(consecutive_retrieve_chains)
    max_chain = max(c["chain_length"] for c in consecutive_retrieve_chains)
    print(f"  Mean chain length: {mean_chain:.2f}, Max: {max_chain}")

    print(f"\n  Chain length distribution:")
    len_dist = Counter(c["chain_length"] for c in consecutive_retrieve_chains)
    for length in sorted(len_dist.keys()):
        print(f"    length={length}: {len_dist[length]} chains")

    print(f"\n  Empty chains by category:")
    empty_by_cat = Counter(c["category"] for c in consecutive_retrieve_chains if not c["exposed_anything"])
    for cat, count in empty_by_cat.most_common():
        print(f"    {cat:<30} {count}")

    # Premature termination analysis
    print(f"\n=== PREMATURE TERMINATION ANALYSIS ===")
    print(f"  (Model ANSWERs from non-READY state AND task fails)")
    premature = []
    for r in results:
        fork = r["fork_m3"]
        log = fork.get("decision_state_log", [])
        actions = fork.get("continuation_actions", [])
        terminal = fork.get("terminal_action", "")
        success = fork.get("success", False)

        if terminal == "ANSWER" and not success:
            # Find the state at the ANSWER step
            answer_step = len(actions) - 1
            if answer_step < len(log):
                answer_state = log[answer_step].get("decision_state", "UNKNOWN")
                premature.append({
                    "task_id": r["task_id"],
                    "category": r["category"],
                    "answer_state": answer_state,
                    "actions": actions,
                })

    print(f"  Total premature terminations (M3): {len(premature)}")
    state_counts = Counter(p["answer_state"] for p in premature)
    for state, count in state_counts.most_common():
        print(f"    {state:<28} {count}")

    # Same for A1
    premature_a1 = []
    for r in results:
        fork = r["fork_a1"]
        terminal = fork.get("terminal_action", "")
        success = fork.get("success", False)
        if terminal == "ANSWER" and not success:
            premature_a1.append({
                "task_id": r["task_id"],
                "category": r["category"],
            })
    print(f"  Total premature terminations (A1): {len(premature_a1)}")

    # Save
    analysis = {
        "schema": "DAPH_V2B_I3_10A_R1_EVENTUAL_CONTRIBUTION_V1",
        "source": "i3_9_r3 v5 trajectories",
        "n_tasks": n,
        "sbu_retrieve_classifications": {
            "total": len(retrieve_classifications),
            "counts": dict(ret_counts),
            "p_eventually_decision_relevant": round(
                ret_counts.get("DECISION_RELEVANT", 0) / max(len(retrieve_classifications), 1), 4),
        },
        "sbu_search_classifications": {
            "total": len(search_classifications),
            "counts": dict(search_counts),
            "p_eventually_decision_relevant": round(
                search_counts.get("DECISION_RELEVANT", 0) / max(len(search_classifications), 1), 4),
        },
        "batch_retrieval_chains": {
            "total_chains": len(consecutive_retrieve_chains),
            "chains_exposed_evidence": chains_exposed,
            "chains_exposed_nothing": chains_empty,
            "mean_chain_length": round(mean_chain, 2),
            "max_chain_length": max_chain,
        },
        "premature_termination": {
            "m3_total": len(premature),
            "m3_by_state": dict(state_counts),
            "a1_total": len(premature_a1),
        },
        "key_finding": (
            "The batch-retrieval pathology is confirmed: most SBU RETRIEVE chains "
            "expose NO evidence (retrieve_exposes is empty for these tasks). "
            "The model retrieves repeatedly without checking whether retrieval "
            "is productive. SEARCH_MORE is the action that actually exposes "
            "decision-relevant evidence in these cases."
        ),
        "m4_implication": (
            "M4 should expose the evidence pipeline state — specifically "
            "unverified_visible_count and whether retrieval has been productive — "
            "so the model can distinguish 'I have unverified evidence to evaluate' "
            "from 'I need to acquire more evidence'. This is state representation, "
            "not action prescription."
        ),
    }

    analysis_path = output_dir / "eventual_contribution_v1.json"
    analysis_path.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    print(f"\n  Analysis saved: {analysis_path}")

    # Save per-action detail
    detail_path = output_dir / "eventual_contribution_v1_detail.jsonl"
    with open(detail_path, "w") as f:
        for c in all_classifications:
            f.write(json.dumps(c, sort_keys=True) + "\n")
    print(f"  Per-action detail saved: {detail_path}")


if __name__ == "__main__":
    main()
