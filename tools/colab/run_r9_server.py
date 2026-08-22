"""Build llama.cpp from source with CUDA, then run R9 using the server.

This script:
1. Clones and builds llama.cpp with CUDA (takes ~30-40 min on T4)
2. Downloads the Liquid model
3. For each reasoning budget, starts the server with --reasoning-budget
4. Runs the 20-state qualification using LocalLlamaBackend
5. Saves results

The build is done ONCE. The server is restarted for each budget.
"""
import subprocess
import sys
import os
import time
import json
import signal
from pathlib import Path

def run(cmd, check=True, timeout=None):
    print(f">>> {cmd}", flush=True)
    result = subprocess.run(cmd, shell=True, timeout=timeout)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")
    return result

def main():
    os.chdir("/content")

    # Step 1: Build llama.cpp with CUDA
    print("=" * 80, flush=True)
    print("STEP 1: Build llama.cpp with CUDA", flush=True)
    print("=" * 80, flush=True)

    llama_dir = "/content/llama.cpp"
    llama_server = f"{llama_dir}/build/bin/llama-server"

    if os.path.exists(llama_server):
        print(f"llama-server already built: {llama_server}", flush=True)
    else:
        if not os.path.exists(llama_dir):
            run("git clone https://github.com/ggml-org/llama.cpp.git")
        else:
            print("llama.cpp already cloned", flush=True)

        os.chdir(llama_dir)
        nproc = os.cpu_count() or 2
        print(f"Building with {nproc} parallel jobs...", flush=True)
        run(f"cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -5")
        # Build with all cores — this takes ~30 min on T4
        run(f"cmake --build build --config Release -j {nproc} 2>&1 | tail -20", timeout=3600)
        os.chdir("/content")

        if not os.path.exists(llama_server):
            print(f"ERROR: {llama_server} not found after build")
            sys.exit(1)
        print(f"llama-server built successfully!", flush=True)

    # Step 2: Download model if needed
    print("\n" + "=" * 80, flush=True)
    print("STEP 2: Download model", flush=True)
    print("=" * 80, flush=True)

    model_path = "/content/models/LFM2.5-2.6B-Q5_K_M.gguf"
    if not os.path.exists(model_path):
        os.makedirs("/content/models", exist_ok=True)
        from huggingface_hub import hf_hub_download
        print("Downloading model...", flush=True)
        hf_hub_download(
            repo_id="LiquidAI/LFM2.5-2.6B-GGUF",
            filename="LFM2.5-2.6B-Q5_K_M.gguf",
            local_dir="/content/models",
        )
        print(f"Model downloaded", flush=True)
    else:
        print(f"Model already exists: {model_path}", flush=True)

    # Step 3: Set up repo
    print("\n" + "=" * 80, flush=True)
    print("STEP 3: Set up DAPH repo", flush=True)
    print("=" * 80, flush=True)

    repo_dir = "/content/Daph-ex-research-gate-c2-beir-retrieval"
    if not os.path.exists(repo_dir):
        run(f"git clone https://github.com/dawsonblock/Daph-ex-research-gate-c2-beir-retrieval.git {repo_dir}")
    os.chdir(repo_dir)
    run("git checkout i3.12-semantic-relation-ablation 2>&1 || true")
    run("git pull origin i3.12-semantic-relation-ablation 2>&1 || true")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    print(f"Repo commit: {commit}", flush=True)

    # Step 4: Run R9 using the server approach
    print("\n" + "=" * 80, flush=True)
    print("STEP 4: R9a Reasoning-budget qualification (server mode)", flush=True)
    print("=" * 80, flush=True)

    output_path = "/content/r9_results.json"
    cmd = (
        f"cd {repo_dir} && PYTHONPATH=. {sys.executable} tools/colab/r9_reasoning_budget.py "
        f"--model-path {model_path} "
        f"--output {output_path} "
        f"--budgets 0,64,128,256,512,1024 "
        f"--max-tokens 2048 "
        f"--use-server "
        f"--llama-server {llama_server}"
    )
    print(f"Running: {cmd}", flush=True)
    run(cmd, timeout=3600)

    # Step 5: Print results
    print("\n" + "=" * 80, flush=True)
    print("R9a COMPLETE", flush=True)
    print("=" * 80, flush=True)

    with open(output_path) as f:
        results = json.load(f)

    print(f"\nMinimum passing budget: {results.get('minimum_passing_budget', 'NONE')}")
    print(f"\nBudget | Decoder | Core Acc | Length Fail | Tokens | Latency | Agreement vs 1024")
    print("-" * 90)
    for r in results["results"]:
        budget = r["reasoning_budget"]
        print(f"  {budget:5d} | {r['decoder_success']:.0%}     | {r['core_action_accuracy']:.0%}      | "
              f"{r['length_failure_rate']:.0%}         | {r['mean_completion_tokens']:.0f}    | "
              f"{r['mean_latency_ms']:.0f}ms   | {r.get('action_agreement_vs_1024', 0):.0%}")

    print(f"\nResults saved to {output_path}", flush=True)

if __name__ == "__main__":
    main()
