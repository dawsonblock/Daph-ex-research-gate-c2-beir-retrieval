# Performance Qualification

Optimize wall-clock execution for DAPH experiments on Colab **without changing any frozen scientific treatment**.

## Optimization Hierarchy

```
shorter reasoning > parallel slots > L4 vs T4 > precomputation > faster tests
```

## 1. Hardware: L4 GPU

Use `colab new -s daph --gpu L4` instead of T4. The L4 is the best price/performance class for a 2.6B Q5 GGUF model. A100/H100 is excessive unless running large confirmation.

## 2. Keep llama-server Resident

**Never** load/unload the model per task. Start one `llama-server` and keep it resident for the entire experiment:

```bash
./tools/colab/start_llama_fast.sh /content/models/LFM2.5-2.6B-Q5_K_M.gguf 0 8 8080
```

Full GPU offload (`-ngl 99`) + Flash Attention (`-fa on`).

## 3. Parallel Slots (Largest Throughput Win)

`llama-server` supports `--parallel N` with continuous batching. Match client workers to server slots:

| Config | Server | Client |
|--------|--------|--------|
| parallel=1 | `--parallel 1` | 1 worker |
| parallel=4 | `--parallel 4` | 4 workers |
| parallel=8 | `--parallel 8` | 8 workers |

**Never** over-subscribe (20 workers against 1 slot = queue, not parallelism).

Benchmark before choosing:

```bash
PYTHONPATH=. python3 tools/colab/benchmark_llama_throughput.py \
    --model-path /content/models/LFM2.5-2.6B-Q5_K_M.gguf \
    --llama-server /content/llama.cpp/build/bin/llama-server \
    --output /content/throughput_benchmark.json
```

Select the highest-throughput stable config with **zero decoder failures**.

## 4. Context Size

Use `--ctx-size 4096` unless corpus analysis proves a larger production packet exists. Larger contexts consume more KV-cache and reduce concurrent slots.

## 5. R9 Parallelization

R9 has 6 budgets × 20 states = 120 independent requests. Queue them concurrently:

```bash
PYTHONPATH=. python3 tools/colab/run_r9_parallel.py \
    --model-path /content/models/LFM2.5-2.6B-Q5_K_M.gguf \
    --llama-server /content/llama.cpp/build/bin/llama-server \
    --output /content/r9_results.json \
    --parallel 4
```

Each result retains `state_id`, `reasoning_budget`, and hashes — scientific pairing is preserved under concurrency.

## 6. No Retrieval During R9

R9 consumes **frozen serialized policy states only**. No BGE, no reranker, no FAISS, no corpus parsing, no semantic extraction. The R9 input is precomputed JSON.

## 7. Precompute for R13

For R13, precompute offline once:
- Retrieval receipts
- Semantic relation snapshots
- Initial MDSG snapshots
- Task metadata
- Q3 rankings
- Required-evidence maps

Only recompute what changes as the trajectory evolves.

## 8. Cache Model Artifacts

Keep models on the Colab VM at `/content/models/`. Reuse the same session across R8, R9, preflight, R13. If the runtime is destroyed, cache in Drive but copy to local disk before testing (never run models from mounted Drive).

## 9. Test Tiers

| Tier | Command | When | Cost |
|------|---------|------|------|
| Relevant | `pytest tests/relevant_module -q` | After each edit | Seconds |
| Fast | `pytest -m "not slow" -q` | Before commit | Tens of seconds |
| Parallel | `pytest -n auto` | Before freeze | Minutes |
| Full | Full qualification suite | Release/freeze only | Full |

Install: `pip install pytest-xdist`

## 10. One Process Per Neural Model

- **ONE** `llama-server` process → N server slots → N client workers
- **ONE** BGE singleton in main process
- **ONE** reranker singleton in main process
- Inference workers: no retriever objects

## 11. Reasoning Budget (Largest Single-Request Win)

If R9 shows `reasoning_budget=128` performs identically to `1024`, that's an 8× reduction in reasoning ceiling. Combined with `max_tokens=256` (if R9b confirms), this eliminates huge tail latency.

## 12. Prompt Caching

Test `cache_prompt=true` vs `false` during development. For confirmation, verify 100% decoded-action agreement before enabling. Prioritize parallel slots over prompt caching (cleaner optimization).

## 13. Multi-Runtime Sharding

If Colab permits multiple simultaneous runtimes:
- Colab A → budgets 0, 64
- Colab B → budgets 128, 256
- Colab C → budgets 512, 1024

Merge result JSONL by `task_id`. Use identical GPU classes (L4+L4+L4) to avoid hardware-induced numerical variation.

## Deliverables

| File | Purpose |
|------|---------|
| `tools/colab/start_llama_fast.sh` | Fast server startup with optimal flags |
| `tools/colab/benchmark_llama_throughput.py` | Benchmark parallel slot configs |
| `tools/colab/run_r9_parallel.py` | Parallel R9 qualification runner |
| `docs/PERFORMANCE_QUALIFICATION.md` | This document |
