# exec_training_v2 collection, attempt 1: RUN_VALID = FALSE

`exec_training_v2_execute.receipts.jsonl`/`.json` as originally committed in
this directory (attempt 1, run on 2026-08-10) are **invalid** and contribute
**zero scientific evidence**. Kept in history for transparency, per this
project's standing practice (the same handling applied to EOB-v2's first,
also-invalid GPU run).

## The defect

`scripts/build_exec_training_v2_suite.py` correctly generated 5 fresh
b3-style scales for the `MEMORY_required` family (`exec2_700`..`exec2_3000`,
150 tasks each, scale-namespaced task/evidence IDs) and additionally wrote a
**flattened, pooled** `MEMORY_required/oracle_tasks.jsonl` +
`evidence.jsonl` (all 5 scales' ~8400 combined evidence records in one
file) for downstream bookkeeping convenience.

`scripts/run_exec_training_v2_collection.py` (attempt 1) loaded that
flattened file directly and ran retrieval for **every** `MEMORY_required`
task against the **entire pooled ~8400-record evidence corpus** -- i.e.
every task's candidate pool for BM25/dense retrieval was drawn from all 5
scales' evidence at once, dominated by filler/distractor content from
scales unrelated to that specific task.

This does not match how `CERTIFIED_MEMORY_V1` has ever been qualified,
confirmed, or exercised in any other gate in this project.
`scripts/run_hrm_qualification.py` (and every other consumer of
`b3_calibration_v1`-style data: `run_exec_training_v1_collection.py`,
`build_eob_v1_suite.py`/`build_eob_v2_suite.py`, `build_g2_confirmation2_suite.py`'s
diagnostics) processes retrieval **scale-by-scale**, with a fresh candidate
pool built from that scale's own evidence only.

## What it did to the numbers

Independently checked after the run: `required_in_packet` (the certified
evidence actually making it into the composed packet) dropped from V1's
73.6% (53/72, scale-isolated, 307-record pool) to attempt 1's 31.3%
(235/750, pooled, 8400-record pool) -- a ~2.4x collapse, driven purely by
retrieval competing against a much larger, mostly-irrelevant candidate pool,
not by any change to `CERTIFIED_MEMORY_V1` itself (unchanged, identity
verified before every task). `Q(A1_USE_CERTIFIED_MEMORY)` collapsed to
18.0% (135/750) accordingly -- an artifact of the retrieval-scope defect,
not a finding about the certified pipeline or about the escalation-gate
question this suite exists to answer.

The resulting `MEMORY_strict_win` count (141/1015 pool-wide) was too small
to clear `configs/gate_answer_probe_v2_design.json`'s eval-floor
(`MEMORY_strict_win >= 40`), so attempt 1 nominally hit
`INCONCLUSIVE_INSUFFICIENT_EVAL_CLASS_SUPPORT` -- but that outcome itself is
downstream of the retrieval-scope defect and is not trusted either.

## The fix

`scripts/run_exec_training_v2_collection.py` was corrected (commit
following this one) to process `MEMORY_required` scale-by-scale, loading
each scale's own (un-namespaced) `oracle_tasks.jsonl`/`evidence.jsonl`
directly from `data/hrm/exec_training_v2/MEMORY_required/exec2_*/` and
building a fresh retrieval candidate pool per scale -- exactly mirroring
`run_hrm_qualification.py`'s convention. The scale-namespaced task_id
(`f"{scale}:{orig_id}"`) is applied only to the receipt's `task_id` field
(for consistency with the frozen pooled suite file's IDs), never to the
evidence/retrieval ID space.

The frozen suite itself (`data/hrm/exec_training_v2/`, commit `ec5f0c4`) is
**not** implicated and was **not** regenerated -- the defect was entirely in
how the already-correct suite was *consumed* by the collection script, not
in its construction. Zero-overlap audit, fact-table verification, and suite
hashes all remain valid.

## Status

Superseded by attempt 2, run after this fix. See
`RESEARCH_STATUS.json`'s `answer_probe_gate_v2` entry for the frozen,
trusted result.
