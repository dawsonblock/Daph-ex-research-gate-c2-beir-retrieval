# DAPH Colab Tooling

Tools for running DAPH experiments on Google Colab GPU sessions.

## Architecture

```
Windsurf (local Mac)          Colab (remote GPU)
  ├── edit code                 ├── llama.cpp + Liquid model
  ├── git commit + push         ├── R9 reasoning-budget qualification
  ├── daph_colab.sh             ├── R13 powered confirmation
  └── analyze results           └── results saved to /content/
         ↑                            |
         └──── fetch results ─────────┘
```

## Prerequisites

- `google-colab-cli` installed: `uv tool install google-colab-cli`
- Authenticated: `colab sessions` (completes OAuth flow)
- Repo pushed to GitHub

## Usage

```bash
# 1. Create a T4 GPU session
./tools/colab/daph_colab.sh start

# 2. Bootstrap the remote workspace (clone repo, install deps)
./tools/colab/daph_colab.sh bootstrap

# 3. Run R8.1 retrieval qualification
./tools/colab/daph_colab.sh r8

# 4. Run R9 reasoning-budget qualification
./tools/colab/daph_colab.sh r9

# 5. Run no-LLM preflight
./tools/colab/daph_colab.sh preflight

# 6. Run R13 powered confirmation (only after preflight passes)
./tools/colab/daph_colab.sh confirm

# 7. Download results
./tools/colab/daph_colab.sh fetch

# 8. Stop session when done
./tools/colab/daph_colab.sh stop
```

## Files

- `bootstrap_remote.py` — Clones repo at exact commit, installs deps, records provenance
- `r9_reasoning_budget.py` — R9a reasoning-budget qualification (0/64/128/256/512/1024)
- `run_r9.py` — R9 orchestrator (installs llama.cpp, downloads model, runs qualification)
- `preflight.py` — No-LLM confirmation preflight (10 structural checks)
- `daph_colab.sh` — Shell control script for the full lifecycle

## Important

- The local Windsurf repo is the canonical source of truth.
- Colab filesystem is ephemeral — always `fetch` results before stopping.
- Do not edit scientific code on Colab. Edit locally, push, then bootstrap.
- R13 confirmation must not start until preflight passes.
