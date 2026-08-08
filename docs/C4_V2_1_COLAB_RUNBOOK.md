# C4 v2.1 Certifying Run — Colab Runbook

Operational plan for the run that produces the first certification-grade C4
result. Read the whole thing before starting: two of the steps must happen in a
specific order, and one of them cannot be fixed after the fact.

**Goal:** a development bundle with `VALID_RUN: true`, plus the first real
measurement of `Q(C4_4)` under the intended deterministic packet ordering.

**Not a goal:** the qualification split. Stop when development is certified.

---

## 0. Blocker — resolve before opening Colab

The runner no longer clones. It certifies the working tree it is invoked in and
requires `--expected-commit`, so a session can no longer end up holding two
checkouts. **The caller checks out the revision; the runner verifies it.**

Which means the revision you check out must contain the v2.1 code.
`origin/main` is currently at `dd3f0fd`, which does **not** contain:

| Missing on `main` | Needed at step |
| --- | --- |
| `configs/gate_c4_protocol_v2_1.json` | 5 |
| `configs/c4_requirements.lock` | 3 |
| `scripts/c4_freeze_environment.py` | 4 |
| `scripts/certify_c4_run.py` | 17 |
| `hrm_adaptive_memory/c4/protocol_validation.py` | 5 |
| `hrm_adaptive_memory/c4/environment_lock.py` | 3 |

Checking out `dd3f0fd` aborts at step 2 (cannot import `hrm_adaptive_memory.c4.git_state`)
or step 3 (missing lock). Pick one:

- **A (recommended): merge PR #1 into `main`.** The result is tied to a commit on
  the mainline, which is what you want for lineage.
- **B: use the branch commit directly.** Any commit works — you pass its SHA to
  `--expected-commit`. It does not have to be on `main`.

Either way, **record the full SHA**. You will pass it twice: once to
`git checkout`, once to `--expected-commit`.

> Do not "fix" an import error by patching the checkout. The v2.1 modules live at
> `hrm_adaptive_memory/c4/environment_lock.py` and
> `hrm_adaptive_memory/c4/protocol_validation.py`. If those imports fail, you are
> on the wrong revision — check out the right one instead of rewriting the
> imports. Editing the checkout also dirties the tree, which now aborts at
> step 2.

---

## 1. Why two sessions

The environment lock has to be *captured in the runtime that will produce the
result*, but capturing it writes `configs/c4_requirements.lock` — which dirties
the tree, and a dirty tree fails the `source_lineage` gate (`D8_clean_release`).

So the lock is frozen in one session, committed, and the certifying run happens
in a second session against a clean checkout that already contains it.

```
Session A (T4, ~10 min)      freeze the lock, push it, run nothing else
        |
   commit + push the lock
        |
Session B (T4, ~35 min)      fresh runtime, clean clone, certifying run
```

Do both on the same day. Colab rebuilds its base image periodically, and a
rebuild between sessions invalidates the lock.

---

## 2. Session A — freeze the environment

**Runtime → Change runtime type → T4 GPU + High-RAM.** This must be a GPU
runtime: `capture_environment` records `accelerator.cuda` and
`accelerator.gpu_name` only when CUDA is available. On CPU those stay `null`,
the verifier does not assert them, and you end up with a lock that does not pin
the accelerator.

```python
!git clone <url> /content/repo
%cd /content/repo
!git checkout <SHA>
!pip install -q -e ".[hrm]"
```

Install what the pipeline actually needs here — **not** from the lock. The point
of this session is to discover the versions, not to impose them.

Optionally load the HRM model once first, so the versions recorded are the ones
that actually served a generation:

```python
!python scripts/run_gate_c4.py smoke
```

Then freeze and verify:

```python
!python scripts/c4_freeze_environment.py --note 'colab T4 <YYYY-MM-DD>'
!python scripts/c4_freeze_environment.py --check     # must print PASS
```

`--check` failing here means the lock does not describe the runtime it was just
taken from. Stop and investigate rather than continuing.

### Getting the lock out

Download `configs/c4_requirements.lock` and commit it from your own machine.
Do not put a GitHub token in a Colab cell.

```bash
git add configs/c4_requirements.lock
git commit -m "C4 v2.1: freeze Colab T4 environment lock for certifying run"
git push
```

Confirm afterwards that a fresh clone is clean:

```bash
git status --porcelain    # must be empty
```

---

## 3. Session B — the certifying run

**Runtime → Disconnect and delete runtime first.** A fresh runtime is what makes
"installed from the lock" meaningful.

```python
!git clone <url> /content/repo
%cd /content/repo
!git checkout <SHA_WITH_CAPTURED_LOCK>
!pip install -q -e ".[hrm]"
!python scripts/c4_freeze_environment.py --check
```

Steps 2, 3 and 4 now abort on revision mismatch, source dirt, an install
failure, null pins, and any environment violation — so the run will not spend
GPU time behind a state that cannot certify. The `--check` above is still worth
running first because its output is easier to read than an abort.

Then:

```python
!python scripts/colab_c4_requalify.py --expected-commit <SHA_WITH_CAPTURED_LOCK>
```

### What the 18 steps do

| # | Step | Aborts? | Expected |
| --- | --- | --- | --- |
| 1 | Verify GPU | yes | Tesla T4, ~15.4 GB, CUDA 12.8 |
| 2 | Verify revision + clean source | **yes** | `HEAD == --expected-commit`, source clean |
| 3 | Install from lock | **yes** | every pin satisfied, no null pins |
| 4 | Verify environment | **yes** | "Environment matches the lock" |
| 5 | Validate protocol | yes | SHA `b4e22a55…`, invariants resolved (not `N/A`) |
| 6 | Full test suite | **yes** | 844 passed, 2 skipped |
| 7 | Determinism, 3 seeds | **yes** | 120/120 identical on all 11 fields |
| 8 | Freeze packets | yes | packet + prompt artifacts written |
| 9 | CPU dry run | **yes** | 7/7 conformance gates |
| 10 | C4-BRIDGE | no | negative result (expected; informational) |
| 11 | HRM smoke | yes | model loads on GPU |
| 12 | Full run, 7 arms | **yes** | 120/120 per arm, 840 generations |
| 13 | Diagnostic arms | **yes** | `C4_3o`, `C4_4m` at 120/120 |
| 14 | Analyzer | **yes** | `analysis.json`, then `RESULTS.sha256` rewritten |
| 15 | Composition diagnostic | no | informational |
| 16 | Results summary | no | quality table + 2×2 decomposition |
| 17 | Certification | no | `CERTIFICATION.json`, 16 gates |
| 18 | Package | no | archive named for the verdict |

Steps 2, 3, 4, 6, 7, 9, 12, 13 and 14 are declared abort conditions. If one
trips, the run stops on purpose: nothing downstream could be certified anyway.

**On "clean source":** cleanliness is scoped to the paths that define the
revision — `hrm_adaptive_memory/`, `daph/`, `scripts/`, `configs/`, `tests/`,
`pyproject.toml`. Changes under `evidence/` are expected and never fatal: steps
7 through 11 rewrite tracked files there (the determinism receipt, 961 frozen
packets, dry-run and smoke receipts). An unscoped dirty check would make
`VALID_RUN: true` unreachable on every genuine run.

### Time budget

Measured from the 2026-08-07 T4 run:

| Phase | Time |
| --- | --- |
| HRM model download (first time) | ~40 s for 2.37 GB |
| HRM load | ~53 s per invoking process |
| Per arm, 120 tasks | 78–87 s |
| Primary ladder, 7 arms + load | ~10.5 min |
| Diagnostic arms, 2 arms + load | ~3.5 min |
| Test suite (runs twice: step 6 and inside step 17) | ~45 s each |
| Determinism, freeze, dry run, smoke | ~5 min |
| Analyzer + certification | ~2 min |
| **Total** | **~25–35 min** |

---

## 4. Reading the result

```python
import json
cert = json.load(open("evidence/gate_c4/full/development/certification/CERTIFICATION.json"))
print(cert["VALID_RUN"], cert["verdict"], cert["gates_failed"])
```

`VALID_RUN` is the conjunction of 16 gates, each derived from the artifacts.
The archive is named `UNCERTIFIED_C4_V2_DEVELOPMENT_RESULT.zip` unless it is
`true`. Do not rename it by hand.

Finally, hashes last:

```bash
cd evidence/gate_c4/full/development && sha256sum -c RESULTS.sha256
```

---

## 5. Pre-registered interpretation

Decide this **before** seeing the numbers.

The 2026-08-07 `Q(C4_4)=0.3729` was the `C4_4m` cell — S2c membership under
retrieval pool order — because ordering never reached the prompt. This run
measures the real 2×2 for the first time:

```
E_order_S0        = Q(C4_3o) - Q(C4_3)
E_membership_pool = Q(C4_4m) - Q(C4_3)
E_order_S2c       = Q(C4_4)  - Q(C4_4m)
interaction       = E_order_S2c - E_order_S0
```

| Outcome | Reading | Action |
| --- | --- | --- |
| `E_order_S2c > 0`, family CI excludes 0 | Ordering adds to membership | Freeze `C4_4` as the primary arm |
| `E_order_S2c ≈ 0` | Membership is the whole mechanism | Consider freezing `C4_4m`: simpler arm, same result |
| `E_order_S2c < 0` | Ordering harms the packet | Freeze `C4_4m`; ordering becomes a documented negative |
| `interaction` large | Ordering only matters under S2c membership | Keep both; report the interaction |

**Consistency check:** `Q(C4_4m)` should land near `0.3729`. It is the same cell
the historical run measured, so a large divergence means something *other* than
ordering changed, and the rerun itself needs investigation before its numbers
are used.

**The promotion threshold is not adjustable after the fact.** `D4` requires
`Q(C4_4) - Q(C4_0) ≥ +0.15` with family and cluster CI lower bounds above 0. If
the true `C4_4` falls below that, the development gate **FAILS**. That is a
result, not a bug — it would mean the intended ordering policy degrades a
mechanism that worked without it.

`C4_3o` and `C4_4m` are diagnostic. They explain `C4_4`. They do not enter the
primary ladder, the arm-count gate, the promotion threshold, or any gap-capture
formula, and they do not become tuning candidates without a new protocol
version.

---

## 6. Known risks

**`torch==2.11.0+cu128` may be uninstallable.** The `+cu128` local version
identifier does not exist on PyPI; it resolves only if torch is already at that
version (same Colab image) or from the PyTorch index. This is why Session B
should run on the same image as Session A, and why you check `--check` before
the GPU work. If Colab has moved on, re-run Session A and re-freeze.

**Colab image drift between sessions** invalidates the lock. Same-day, or
re-freeze.

**Disconnection mid-run** is survivable: the full run is resumable. Receipts
from before the prompt-order fix are *not* reused, because `PIPELINE_VERSION` is
part of the resume key. Re-run the script; it picks up from complete arms.

**Never edit files in the Colab checkout.** Source edits now abort at step 2,
before any GPU work. If you must change code, change it locally, commit, push,
and check out the new SHA. In particular, do not patch import paths: a failing
`hrm_adaptive_memory.c4.*` import means you are on the wrong revision.

**HuggingFace is accessed unauthenticated**, which prints a warning and is
subject to rate limits. Set a token if downloads stall.

**`configs/c4_requirements.lock` as committed today is transcribed, not
captured** — it holds the versions read off the 2026-08-07 notebook, with `null`
for the four it never printed. Those `null`s fail certification as
`MUST_RECORD`. That is deliberate: an unrecorded version is a failure, not a
wildcard. Session A exists to replace it.

---

## 7. Success criterion

```
CERTIFICATION.json
  VALID_RUN: true
  gates_passed: 16/16
sha256sum -c RESULTS.sha256   -> all OK
```

Then, and only then:

1. Commit the certificate, the lock, and the bundle.
2. Record the measured 2×2 decomposition in `RESEARCH_STATUS.json`, replacing
   `historical_development_signal` with the certified result and retaining the
   historical entry as lineage.
3. Update the gate status away from
   `IN_PROGRESS_RERUN_REQUIRED_AFTER_PACKET_ORDERING_CONFORMANCE_DEFECT`.
4. Apply the D1–D8 promotion gates.
5. **Stop.** Whether development authorises the qualification split is a
   separate decision, made against those gates.
