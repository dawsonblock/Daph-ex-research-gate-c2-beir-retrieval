# V2B-I2: deterministic executive experiment infrastructure

## Status

```text
IMPLEMENTED                 yes
DEVELOPMENT HARNESS         yes
LEARNED EXECUTIVE           no
SCIENTIFICALLY QUALIFIED    no
PRODUCTION READY            no
```

V2B-I2 is a deliberately synthetic, deterministic harness. Its purpose is to
validate the experimental controls before a model-based executive is introduced.
It is not evidence that cognitive state improves any real model's answers.

Each run fixes the same benchmark, policy, action executor, and per-task
resource budget across both conditions:

| Condition | Controller input |
| --- | --- |
| `CONTROL` | Task summary and remaining executive-step budget only |
| `V2B` | Bounded cognitive snapshot: provenance summary, verification state, temporal state, unresolved conflicts, decision/outcome summaries, resource state, and policy facts |

The only allowed actions are `ANSWER`, `RETRIEVE`, `VERIFY`, `SEARCH_MORE`,
`REASON_MORE`, `DEFER`, and `STOP`. The controller only proposes a structured
action. The loop validates it, applies the frozen policy gate, applies the
resource gate, executes one bounded synthetic action, and writes a structured
decision/outcome pair to the append-only cognitive log.

The frozen development corpus covers immediate-answer, retrieval-required,
verification-required, conflict, stale-temporal, falsified-memory,
search-more-required, reason-more-required, and insufficient-evidence cases.
It intentionally uses no LLM, live retrieval, or live network verification.

`scripts/run_v2b_i2_development.py` only runs from a clean committed checkout
and writes a development-only receipt. It rejects qualification status, dirty
source trees, nonempty output directories, and unsafe run identifiers. Each
accepted development run also records the receipt identity and both conditions'
metrics to the configured LitLogger teamspace; it never logs a qualification
claim.

The focused development suite currently reports **39 passed, 0 failed**. This
is implementation evidence for the deterministic harness, not a V2B
qualification result.

Before any scientific V2B claim, replace the synthetic controller/model setup
with a pinned controller/model and run the frozen benchmark, adversarial cases,
replay, and held-out evaluation under the V2B qualification identity.
