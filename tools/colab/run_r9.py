"""R9 runner: Install llama-cpp-python (pre-built CUDA wheel), download model, run qualification.

This script runs ON the Colab VM. It:
1. Installs llama-cpp-python with CUDA support (pre-built wheel — no build needed)
2. Downloads LiquidAI/LFM2.5-2.6B-GGUF:Q5_K_M
3. Runs the R9a reasoning-budget qualification
4. Saves results

Much faster than building llama.cpp from source (~2 min vs ~40 min).
"""
import os
import subprocess
import sys
import json
from pathlib import Path

def run(cmd, check=True):
    print(f">>> {cmd}")
    result = subprocess.run(cmd, shell=True)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")
    return result

def main():
    os.chdir("/content")

    # Step 1: Install llama-cpp-python with CUDA (pre-built wheel)
    print("=" * 80)
    print("STEP 1: Install llama-cpp-python with CUDA")
    print("=" * 80)

    # Check if already installed
    try:
        import llama_cpp
        print(f"llama-cpp-python already installed: {llama_cpp.__version__}")
    except ImportError:
        print("Installing llama-cpp-python with CUDA support...")
        run(f"{sys.executable} -m pip install llama-cpp-python "
            f"--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121 2>&1 | tail -5")
        import llama_cpp
        print(f"llama-cpp-python installed: {llama_cpp.__version__}")

    # Verify CUDA is available
    import torch
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Step 2: Download Liquid model
    print("\n" + "=" * 80)
    print("STEP 2: Download LiquidAI/LFM2.5-2.6B-GGUF:Q5_K_M")
    print("=" * 80)

    model_dir = "/content/models"
    model_path = f"{model_dir}/LFM2.5-2.6B-Q5_K_M.gguf"
    os.makedirs(model_dir, exist_ok=True)

    if not os.path.exists(model_path):
        run(f"{sys.executable} -m pip install -U huggingface_hub 2>&1 | tail -3")
        from huggingface_hub import hf_hub_download
        print("Downloading model...")
        downloaded = hf_hub_download(
            repo_id="LiquidAI/LFM2.5-2.6B-GGUF",
            filename="LFM2.5-2.6B-Q5_K_M.gguf",
            local_dir=model_dir,
        )
        model_path = str(downloaded)
        print(f"Model downloaded: {model_path}")
    else:
        print(f"Model already exists: {model_path}")

    size_gb = os.path.getsize(model_path) / 1e9
    print(f"Model size: {size_gb:.2f} GB")

    # Step 3: Set up repo
    print("\n" + "=" * 80)
    print("STEP 3: Set up DAPH repo")
    print("=" * 80)

    repo_dir = "/content/Daph-ex-research-gate-c2-beir-retrieval"
    if not os.path.exists(repo_dir):
        run(f"git clone https://github.com/dawsonblock/Daph-ex-research-gate-c2-beir-retrieval.git {repo_dir}")
    else:
        print(f"Repo exists at {repo_dir}")

    os.chdir(repo_dir)
    run("git checkout i3.12-semantic-relation-ablation 2>&1 || true")
    run("git pull origin i3.12-semantic-relation-ablation 2>&1 || true")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    print(f"Repo commit: {commit}")

    # Install Python deps
    run(f"{sys.executable} -m pip install -r requirements.txt 2>&1 | tail -5 || true")

    # Step 4: Run R9a reasoning-budget qualification
    print("\n" + "=" * 80)
    print("STEP 4: R9a Reasoning-budget qualification")
    print("=" * 80)

    output_path = "/content/r9_results.json"
    cmd = (
        f"cd {repo_dir} && PYTHONPATH=. {sys.executable} tools/colab/r9_reasoning_budget.py "
        f"--model-path {model_path} "
        f"--output {output_path} "
        f"--budgets 0,64,128,256,512,1024 "
        f"--max-tokens 2048"
    )
    print(f"Running: {cmd}")
    run(cmd)

    # Step 5: Print results summary
    print("\n" + "=" * 80)
    print("R9a COMPLETE")
    print("=" * 80)

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

    print(f"\nResults saved to {output_path}")
    print(f"Run: colab download -s daph r9_results.json <local_path>")

if __name__ == "__main__":
    main()
