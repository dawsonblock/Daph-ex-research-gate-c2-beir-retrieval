# Notebooks

**Notebooks are launchers. They contain no scientific logic.**

Every step of a gate run lives in `scripts/`, is importable, and is covered by
the test suite. A notebook's only job is to invoke those scripts and display
what they produced.

```
one implementation   ->  scripts/colab_c4_requalify.py
        |
        v
tested in pytest     ->  tests/unit/test_c4_*.py
        |
        v
notebook merely invokes it
```

## Why this rule exists

The C4 notebooks used to be independent implementations of the run. Three
things followed, all of which actually happened:

1. **They drifted from the tested path.** The notebook and the script disagreed
   about how the run worked, and only the script was under test.
2. **They were fail-open where the protocol requires an abort.** A notebook
   printed `Continuing anyway` after test failures, though `test suite fails` is
   a declared abort condition. The result could never have been certified.
3. **Their certification was self-asserted.** `VALID_RUN` reduced to one
   protocol-hash comparison while `determinism_gate`, `all_arms_complete` and
   `result_hashes_verified` were written as literals. A certificate whose
   prerequisites are string literals certifies nothing.

Notebook cells cannot be unit-tested, so none of that was caught by CI. Keeping
the logic in `scripts/` makes each of those a test failure instead of a
discovery months later.

## Active

| Notebook | Invokes | Purpose |
| --- | --- | --- |
| `colab_c4_requalify.ipynb` | `scripts/c4_freeze_environment.py`, `scripts/colab_c4_requalify.py` | Gate C4 v2_1 fail-closed requalification on a Colab T4 |
| `gate_c1_v4_oracle_ladder_colab.ipynb` | — | Gate C1 v4 oracle ladder (predates this rule) |

## Superseded

`superseded/` holds retired execution paths. They are kept for provenance and
carry a `SUPERSEDED — DO NOT USE FOR QUALIFICATION` notice in their first cell.
They are not deleted, because runs were produced with them and the record of
how those runs were produced is part of the evidence chain.

| File | Superseded by |
| --- | --- |
| `superseded/colab_c4_requalify_pre_fail_closed.ipynb` | `scripts/colab_c4_requalify.py` |
| `superseded/colab_c4_conformant_run_pre_fail_closed.ipynb` | `scripts/colab_c4_requalify.py` |
| `superseded/colab_c4_conformant_run_pre_fail_closed.py` | `scripts/colab_c4_requalify.py` |

Results produced by the superseded paths predate the prompt-order conformance
repair: the deterministic packet order was computed and hashed but never reached
the HRM prompt, so results they labelled `C4_4` measured S2c membership under
retrieval pool order. Those measurements remain valid for what they actually
tested and are reclassified rather than voided — see `RESEARCH_STATUS.json` ->
`historical_development_signal`.

## Adding a notebook

Do not put logic in it. If you need the run to do something new:

1. change the script,
2. add a test,
3. call the script from the notebook.

If a cell computes a metric, builds a prompt, or decides whether a gate passed,
it belongs in `scripts/` or `hrm_adaptive_memory/` instead.
