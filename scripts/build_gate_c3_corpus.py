#!/usr/bin/env python3
"""Build/freeze controlled_gate_c3_v1 — the sixth-vocabulary corpus for Gate C3.

Gate C3 tests surface identity resolution. The corpus must be disjoint from all
five prior vocabularies (birds, minerals, constellations, rivers, summits) so
the resolver cannot succeed by recognizing familiar entity names. The freeze
audit enforces the strengthened disjointness requirement added in protocol
amendment #8: zero overlap on entity surfaces, aliases, descriptions, vocabulary,
and source clusters — not just noun-family names.

Structure is deliberately shared with descv4 (same task families, same proof
graph shape, same source styles) so the SAME task is being tested. Only the
surface realization changes. This distinction is recorded in the freeze audit
per the protocol's template_policy.
"""
from __future__ import annotations
import hashlib, json, re, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import hrm_adaptive_memory.experiments.c2_calibration_dataset as cal
from hrm_adaptive_memory.experiments.generalization_dataset_v4 import (
    _HEADS as V4_HEADS, verify_inferable)

OUT = Path("data/hrm/controlled_gate_c3_v1")
SPLITS = {"c3v1_id": ("c2_cal_id", 99003, 50), "c3v1_surface": ("c2_cal_surface", 99004, 50)}
FORBIDDEN = re.compile(r"#(s|b|v|d\d|n)\b|latent_|entity_\d+")

# Prior corpora for the disjointness audit.
PRIOR_CORPORA = [
    ("controlled_gate_a_v4", ("development", "qualification", "ood")),
    ("controlled_gate_c2_calibration_v1", ("c2_cal_id", "c2_cal_surface", "c2_cal_holdout")),
    ("controlled_gate_c2_chain_validation_v2", ("chain_v2_id", "chain_v2_surface")),
    ("controlled_gate_c2_chain_validation_v3", ("chain_v3_id", "chain_v3_surface")),
    ("controlled_gate_c2_description_valid_v4", ("descv4_id", "descv4_surface")),
]


def audit_description_identifiability(tasks, evidence):
    """Fail-closed checks that a description question has exactly one referent."""
    from collections import defaultdict

    def N(s):
        return " ".join(re.findall(r"\w+", s.lower()))

    by_id = {r["evidence_id"]: r for r in evidence}
    raw_map, norm_map = defaultdict(set), defaultdict(set)
    insufficient = 0
    for t in tasks:
        if t["metadata"]["entity_regime"] != "description":
            continue
        s = t["_oracle_metadata"]["surfaces"]
        raw_map[s["subject"]].add(s["canonical"])
        norm_map[N(s["subject"])].add(N(s["canonical"]))
        ok = any(N(s["subject"]) in N(by_id[v]["content"]) and N(s["canonical"]) in N(by_id[v]["content"])
                 for v in t["required_evidence_ids"] if v in by_id)
        insufficient += 0 if ok else 1

    return {
        "DESCRIPTION_SURFACE_UNIQUENESS": sum(1 for v in raw_map.values() if len(v) > 1),
        "NORMALIZED_SURFACE_COLLISION": sum(1 for v in norm_map.values() if len(v) > 1),
        "RUNTIME_REFERENT_IDENTIFIABILITY": sum(1 for v in norm_map.values() if len(v) != 1),
        "IDENTITY_EVIDENCE_SUFFICIENCY": insufficient,
        "distinct_description_surfaces": len(raw_map),
        "description_task_count": sum(1 for t in tasks
                                      if t["metadata"]["entity_regime"] == "description"),
    }


def collect_prior_surfaces(prev_vocab):
    """Collect all entity surfaces, aliases, descriptions, and source clusters
    from the five prior corpora for the disjointness audit.

    ``prev_vocab`` is the dict returned by ``cal.apply_vocabulary`` — it holds
    the ORIGINAL mineral vocabulary before the C3 override. Module-level globals
    like ``cal.HEADS`` are NOT safe to read here because ``apply_vocabulary``
    has already rebound them to the C3 lighthouse set.
    """
    prior = {"entity_surfaces": set(), "aliases": set(), "descriptions": set(),
             "vocabulary": set(), "source_clusters": set(), "evidence_strings": set()}
    for base, splits in PRIOR_CORPORA:
        for sp in splits:
            evp = ROOT / "data/hrm" / base / sp / "evidence.jsonl"
            if not evp.exists():
                continue
            for line in evp.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                prior["evidence_strings"].add(r["content"])
                prior["source_clusters"].add(r["metadata"]["source_cluster_id"])
            tp = ROOT / "data/hrm" / base / sp / "oracle_tasks.jsonl"
            if not tp.exists():
                continue
            for line in tp.read_text().splitlines():
                if not line.strip():
                    continue
                t = json.loads(line)
                s = t["_oracle_metadata"]["surfaces"]
                prior["entity_surfaces"].add(s["subject"])
                prior["entity_surfaces"].add(s["canonical"])
                if "bridge" in s:
                    prior["entity_surfaces"].add(s["bridge"])
    # Collect all prior vocabulary tokens using the stored prev_vocab (minerals)
    # and the VOCAB dicts (which are not affected by apply_vocabulary).
    all_vocabs = [prev_vocab["HEADS"], cal.VOCAB_V2["HEADS"], cal.VOCAB_V3["HEADS"],
                  cal.VOCAB_V4D["HEADS"], V4_HEADS]
    for v in all_vocabs:
        prior["vocabulary"].update(v)
    all_roles = [prev_vocab["ROLES"], cal.VOCAB_V2["ROLES"], cal.VOCAB_V3["ROLES"], cal.VOCAB_V4D["ROLES"]]
    for v in all_roles:
        for role in v:
            prior["vocabulary"].update(role.split())
    all_symbolic = [prev_vocab["SYMBOLIC"], cal.VOCAB_V2["SYMBOLIC"], cal.VOCAB_V3["SYMBOLIC"], cal.VOCAB_V4D["SYMBOLIC"]]
    for v in all_symbolic:
        prior["vocabulary"].update(v)
    all_enum = [prev_vocab["ENUM"], cal.VOCAB_V2["ENUM"], cal.VOCAB_V3["ENUM"], cal.VOCAB_V4D["ENUM"]]
    for v in all_enum:
        prior["vocabulary"].update(v)
    return prior


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    prior_heads = set(cal.HEADS) | set(cal.VOCAB_V2["HEADS"]) | set(cal.VOCAB_V3["HEADS"]) | set(cal.VOCAB_V4D["HEADS"]) | set(V4_HEADS)
    prev = cal.apply_vocabulary(cal.VOCAB_C3)
    try:
        print("[1/6] generating (lighthouse vocabulary)")
        built = {}
        for name, (partition, seed, per) in SPLITS.items():
            c = cal.build_calibration(seed=seed, partition=partition, per_regime=per)
            for t in c["tasks"]:
                t["task_id"] = t["task_id"].replace("c2cal-", "c3v1-")
                t["split"] = name
            remap = {}
            for r in c["evidence"]:
                new = r["evidence_id"].replace("c2cal-", "c3v1-")
                remap[r["evidence_id"]] = new; r["evidence_id"] = new
                # Remap source clusters from c2cal-cluster-XX to c3v1-cluster-XX
                r["metadata"]["source_cluster_id"] = r["metadata"]["source_cluster_id"].replace("c2cal-cluster", "c3v1-cluster")
            for t in c["tasks"]:
                t["required_evidence_ids"] = [remap.get(v, v.replace("c2cal-", "c3v1-"))
                                              for v in t["required_evidence_ids"]]
                t["oracle_evidence_ids"] = list(t["required_evidence_ids"])
                t["source_cluster_id"] = t["source_cluster_id"].replace("c2cal-cluster", "c3v1-cluster")
                for e in t["_oracle_metadata"]["proof_edges"]:
                    e["record_id"] = remap.get(e["record_id"], e["record_id"].replace("c2cal-", "c3v1-"))
            built[name] = c
            print(f"      {name}: {len(c['tasks'])} tasks, {len(c['evidence'])} records")

        print("[2/6] structural audit (inferable, leaks, identifiability)")
        problems = []
        if set(cal.HEADS) & prior_heads:
            problems.append("vocabulary overlaps a prior corpus")
        if set(cal.HEADS) & set(V4_HEADS):
            problems.append("vocabulary overlaps V4")
        prior_ids = set()
        for base, splits in PRIOR_CORPORA:
            for sp in splits:
                p = ROOT / "data/hrm" / base / sp / "evidence.jsonl"
                if p.exists():
                    prior_ids |= {json.loads(l)["evidence_id"] for l in p.read_text().splitlines() if l.strip()}
        audit = {}
        for name, c in built.items():
            by_id = {r["evidence_id"]: r for r in c["evidence"]}
            bad = {t["task_id"]: verify_inferable(t, by_id) for t in c["tasks"]}
            bad = {k: v for k, v in bad.items() if v}
            audit[name] = {
                "not_inferable": len(bad),
                "question_leaks": sum(1 for t in c["tasks"] if cal.leaks(t["question"], t["answer"])),
                "distractor_leaks": sum(1 for t in c["tasks"] for r in c["evidence"]
                                        if r["evidence_id"].startswith(t["task_id"] + "/")
                                        and r["metadata"]["record_kind"] not in cal.ANSWER_BEARING
                                        and cal.leaks(r["content"], t["answer"])),
                "latent_leaks": sum(1 for r in c["evidence"] if FORBIDDEN.search(r["content"])),
                "id_collisions_with_prior_corpora": len({r["evidence_id"] for r in c["evidence"]} & prior_ids),
            }
            desc = audit_description_identifiability(c["tasks"], c["evidence"])
            audit[name]["description_identifiability"] = desc
            for k in ("DESCRIPTION_SURFACE_UNIQUENESS", "NORMALIZED_SURFACE_COLLISION",
                      "RUNTIME_REFERENT_IDENTIFIABILITY", "IDENTITY_EVIDENCE_SUFFICIENCY"):
                if desc[k]:
                    problems.append(f"{name}: {k}={desc[k]}")
            for k, v in audit[name].items():
                if isinstance(v, int) and v: problems.append(f"{name}: {k}={v}")
        for line in problems: print(f"      FAIL {line}")
        if problems: raise SystemExit("structural audit failed; nothing written")
        print("      structural audit passed")

        print("[3/6] disjointness freeze audit (amendment #8)")
        prior = collect_prior_surfaces(prev)
        c3_surfaces = set()
        c3_aliases = set()
        c3_descriptions = set()
        c3_vocab = set(cal.HEADS) | set(cal.VOCAB_C3["HEADS"])
        c3_clusters = set()
        c3_evidence_strings = set()
        for role in cal.VOCAB_C3["ROLES"]:
            c3_vocab.update(role.split())
        for s in cal.VOCAB_C3["SYMBOLIC"]:
            c3_vocab.update(s.split("-"))
        for e in cal.VOCAB_C3["ENUM"]:
            c3_vocab.add(e)
        for name, c in built.items():
            for r in c["evidence"]:
                c3_evidence_strings.add(r["content"])
                c3_clusters.add(r["metadata"]["source_cluster_id"])
            for t in c["tasks"]:
                s = t["_oracle_metadata"]["surfaces"]
                c3_surfaces.add(s["subject"])
                c3_surfaces.add(s["canonical"])
                if "bridge" in s:
                    c3_surfaces.add(s["bridge"])
                if t["metadata"]["entity_regime"] == "alias":
                    c3_aliases.add(s["subject"])
                if t["metadata"]["entity_regime"] == "description":
                    c3_descriptions.add(s["subject"])

        freeze_audit = {
            "entity_surface_overlap": len(c3_surfaces & prior["entity_surfaces"]),
            "alias_overlap": len(c3_aliases & prior["aliases"]) if "aliases" in prior else 0,
            "description_overlap": len(c3_descriptions & prior["descriptions"]) if "descriptions" in prior else 0,
            "vocabulary_overlap": len(c3_vocab & prior["vocabulary"]),
            "source_cluster_overlap": len(c3_clusters & prior["source_clusters"]),
            "evidence_exact_string_overlap": len(c3_evidence_strings & prior["evidence_strings"]),
        }
        # Also check entity_surface and description overlap against prior entity_surfaces
        # (prior corpora don't separate aliases/descriptions, so check against the combined set)
        freeze_audit["alias_overlap"] = len(c3_aliases & prior["entity_surfaces"])
        freeze_audit["description_overlap"] = len(c3_descriptions & prior["entity_surfaces"])
        print(f"      freeze_audit: {json.dumps(freeze_audit)}")
        for k, v in freeze_audit.items():
            if v: problems.append(f"freeze_audit {k}={v} (required 0)")
        if problems: raise SystemExit("disjointness audit failed; nothing written")
        print("      disjointness audit passed (all overlaps = 0)")

        print("[4/6] pytest")
        r = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, capture_output=True, text=True)
        if r.returncode: print(r.stdout[-1200:]); raise SystemExit("pytest failed")
        print(f"      {r.stdout.strip().splitlines()[-1]}")

        print("[5/6] writing")
        OUT.mkdir(parents=True)
        for name, c in built.items():
            d = OUT / name; d.mkdir()
            (d / "oracle_tasks.jsonl").write_text("".join(json.dumps(t, sort_keys=True) + "\n" for t in c["tasks"]))
            (d / "evidence.jsonl").write_text("".join(json.dumps(e, sort_keys=True) + "\n" for e in c["evidence"]))
            (d / "dataset_manifest.json").write_text(json.dumps(c["manifest"], sort_keys=True, indent=2) + "\n")
        digests = {str(p.relative_to(OUT)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted(OUT.rglob("*")) if p.is_file()}
        (OUT / "AUDIT.json").write_text(json.dumps({
            "generator": "controlled_gate_c3_v1", "audit": audit,
            "problems": problems, "VALID_C3_CORPUS": True,
            "state": {"purpose": "gate_c3_surface_identity_resolution",
                      "vocabulary_domain": "lighthouses",
                      "replaces_prior_corpora": False,
                      "holdout_touched": False, "frozen_before_evaluation": True,
                      "arm_reselection_forbidden": True, "valid": True},
            "separation": {"overlap_with_v4_heads": 0, "overlap_with_cal_v1_heads": 0,
                           "overlap_with_v2_heads": 0, "overlap_with_v3_heads": 0,
                           "overlap_with_v4d_heads": 0,
                           "evidence_id_collisions": 0, "task_id_prefix": "c3v1-"},
            "freeze_audit": freeze_audit,
            "template_policy": {
                "semantic_task_schema_preserved": True,
                "surface_realization_disjoint": True,
                "exact_string_overlap_required": 0,
                "exact_string_overlap_measured": freeze_audit["evidence_exact_string_overlap"],
                "note": "Abstract template families are structurally analogous so the same task is being tested. Surface realization (entity names, roles, descriptors, symbolic codes) is fully disjoint. Template scaffolding (e.g. 'Survey finding: ...') is shared by design — the distinction is recorded here per the protocol's template_policy."},
        }, indent=2, sort_keys=True) + "\n")
        (OUT / "RECEIPTS.sha256").write_text("".join(f"{v}  {k}\n" for k, v in sorted(digests.items())))
        print("[6/6] frozen"); print(json.dumps({"audit": audit, "freeze_audit": freeze_audit}, indent=2))
    finally:
        cal.restore_vocabulary(prev)


if __name__ == "__main__":
    main()
