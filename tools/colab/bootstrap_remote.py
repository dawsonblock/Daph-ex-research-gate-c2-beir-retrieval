"""Bootstrap the DAPH remote workspace on a Colab session.

Clones the repository at the exact local commit, installs dependencies,
and records provenance. Idempotent — safe to run multiple times.

Usage (from Colab CLI):
    colab upload -s daph tools/colab/bootstrap_remote.py bootstrap_remote.py
    colab exec -s daph -f bootstrap_remote.py --timeout 600
"""
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/dawsonblock/Daph-ex-research-gate-c2-beir-retrieval.git"
BRANCH = "i3.12-semantic-relation-ablation"
EXPECTED_COMMIT = None  # Set by caller via env var DAPH_COMMIT
REPO_DIR = "/content/Daph-ex-research-gate-c2-beir-retrieval"
PROVENANCE_PATH = "/content/daph_provenance.json"


def run(cmd, check=True, capture=False):
    print(f">>> {cmd}")
    if capture:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if check and result.returncode != 0:
            print(f"FAILED (exit {result.returncode}): {result.stderr}")
            raise RuntimeError(f"Command failed: {cmd}")
        return result.stdout.strip()
    else:
        result = subprocess.run(cmd, shell=True)
        if check and result.returncode != 0:
            raise RuntimeError(f"Command failed: {cmd}")
        return None


def main():
    os.chdir("/content")

    # Get expected commit from env or use latest
    expected_commit = os.environ.get("DAPH_COMMIT")
    if not expected_commit:
        # Get from local git if running locally
        try:
            expected_commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True,
                cwd=os.path.dirname(os.path.abspath(__file__)),
            ).stdout.strip()
        except Exception:
            pass

    # Clone or update repo
    if os.path.exists(REPO_DIR):
        print(f"\nRepository exists at {REPO_DIR}, pulling latest...")
        run(f"cd {REPO_DIR} && git fetch --all")
        run(f"cd {REPO_DIR} && git checkout {BRANCH}")
        run(f"cd {REPO_DIR} && git pull origin {BRANCH}")
    else:
        print(f"\nCloning {REPO_URL}...")
        run(f"git clone {REPO_URL} {REPO_DIR}")
        run(f"cd {REPO_DIR} && git checkout {BRANCH}")

    # Verify commit
    actual_commit = run(f"cd {REPO_DIR} && git rev-parse HEAD", capture=True)
    print(f"\nRemote commit: {actual_commit}")
    if expected_commit and actual_commit != expected_commit:
        print(f"WARNING: Expected {expected_commit}, got {actual_commit}")
        print(f"Attempting to checkout expected commit...")
        run(f"cd {REPO_DIR} && git checkout {expected_commit}")
        actual_commit = run(f"cd {REPO_DIR} && git rev-parse HEAD", capture=True)
        print(f"Verified commit: {actual_commit}")

    # Install dependencies
    print("\nInstalling dependencies...")
    req_file = f"{REPO_DIR}/requirements.txt"
    if os.path.exists(req_file):
        run(f"pip install -r {req_file} 2>&1 | tail -5")
    else:
        print("No requirements.txt found, installing core deps...")
        run("pip install sentence-transformers transformers torch faiss-cpu scikit-learn numpy scipy 2>&1 | tail -5")

    # Record provenance
    print("\nRecording provenance...")
    provenance = {
        "repo_url": REPO_URL,
        "branch": BRANCH,
        "commit": actual_commit,
        "python_version": sys.version,
        "platform": sys.platform,
    }

    # GPU info
    try:
        import torch
        provenance["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            provenance["gpu_name"] = torch.cuda.get_device_name(0)
            provenance["gpu_vram_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
    except ImportError:
        provenance["cuda_available"] = False
        provenance["gpu_name"] = "torch not installed"

    # nvidia-smi
    try:
        nvidia_smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        provenance["nvidia_smi"] = nvidia_smi.stdout
    except Exception:
        provenance["nvidia_smi"] = "nvidia-smi not available"

    # pip freeze
    pip_freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                                capture_output=True, text=True)
    provenance["pip_freeze"] = pip_freeze.stdout

    with open(PROVENANCE_PATH, "w") as f:
        json.dump(provenance, f, indent=2)
    print(f"Provenance written to {PROVENANCE_PATH}")

    # Quick smoke test
    print("\nRunning import smoke test...")
    run(f"cd {REPO_DIR} && PYTHONPATH=. python3 -c \""
        f"from hrm_adaptive_memory.executive.semantic_relations.i3_15c_task_generator import generate_i3_15c_corpus; "
        f"tasks = generate_i3_15c_corpus(n_per_cell=1, seed=42); "
        f"print(f'Generated {{len(tasks)}} tasks')\"")

    print("\n" + "=" * 60)
    print("BOOTstrap COMPLETE")
    print("=" * 60)
    print(f"Repo: {REPO_DIR}")
    print(f"Commit: {actual_commit}")
    print(f"Provenance: {PROVENANCE_PATH}")


if __name__ == "__main__":
    main()
