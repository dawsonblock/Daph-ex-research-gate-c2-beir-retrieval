#!/usr/bin/env python3
"""SUPERSEDED — DO NOT USE FOR QUALIFICATION.

Authoritative execution path: scripts/colab_c4_requalify.py

Retained for provenance only. This was a second, independent implementation of
the C4 development run, with the same fail-open defects as the notebooks beside
it: a non-aborting test suite, unpinned dependency installation, and no
certification step. Runs produced by this path also predate the prompt-order
conformance repair, so results it labelled C4_4 measured S2c membership under
pool order. See RESEARCH_STATUS.json -> historical_development_signal.

Do not extend this file. One implementation, in scripts/, covered by pytest.
"""
import os
import sys
import subprocess
import time
import json
from pathlib import Path


def run(cmd, label="", timeout=None, stream=False):
    """Run a command, optionally streaming output."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    t0 = time.time()
    
    if stream:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1
        )
        for line in proc.stdout:
            print(line, end="", flush=True)
        proc.wait()
        retcode = proc.returncode
    else:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr[-500:])
        retcode = result.returncode
    
    elapsed = time.time() - t0
    print(f"\n  [{label}] completed in {elapsed:.1f}s")
    return retcode


def main():
    # === Step 0: Verify GPU ===
    print("=" * 60)
    print("  C4 Conformant Development Rerun — T4 GPU")
    print("=" * 60)
    
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU memory: {mem:.1f} GB")
        else:
            print("ERROR: No GPU detected. Enable T4 GPU in Runtime settings.")
            sys.exit(1)
    except ImportError:
        print("ERROR: PyTorch not installed.")
        sys.exit(1)
    
    # === Step 1: Clone repository ===
    REPO_URL = "https://github.com/dawsonblock/Daph-ex-research-gate-c2-beir-retrieval.git"
    REPO_DIR = "/content/Daph-ex-research-gate-c2-beir-retrieval"
    
    if os.path.exists(REPO_DIR):
        print(f"\nRepository exists at {REPO_DIR}")
        subprocess.run(["git", "-C", REPO_DIR, "pull", "--rebase"], check=False)
    else:
        print(f"\nCloning repository...")
        subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
    
    os.chdir(REPO_DIR)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    print(f"Git commit: {commit[:12]}")
    
    # === Step 2: Install dependencies ===
    print("\nInstalling dependencies...")
    subprocess.run(["pip", "install", "-q", "transformers>=5.9.0", 
                    "huggingface-hub>=0.34"], check=True)
    subprocess.run(["pip", "install", "-q", "rank-bm25", "numpy"], check=True)
    subprocess.run(["pip", "install", "-q", "pytest"], check=False)
    subprocess.run(["pip", "install", "-q", "-e", "."], check=True)
    
    import transformers
    print(f"transformers: {transformers.__version__}")
    
    # === Step 3: Run tests ===
    ret = run(["python", "-m", "pytest", "tests/", "-q", "--tb=no"],
              "Test Suite", timeout=120)
    if ret != 0:
        print("WARNING: Some tests failed, but continuing...")
    
    # === Step 4: CPU-only dry run (conformance validation) ===
    ret = run(["python", "scripts/run_gate_c4.py", "dry-run"],
              "Dry Run (7 Conformance Gates)", timeout=300)
    if ret != 0:
        print("ERROR: Conformance validation failed!")
        sys.exit(1)
    
    # === Step 5: C4-BRIDGE gate ===
    ret = run(["python", "scripts/run_gate_c4_bridge.py"],
              "C4-BRIDGE Gate (No HRM)", timeout=300)
    
    # === Step 6: HRM smoke test ===
    ret = run(["python", "scripts/run_gate_c4.py", "smoke"],
              "HRM Smoke Test (3 tasks × 7 arms)", timeout=600)
    if ret != 0:
        print("ERROR: Smoke test failed!")
        sys.exit(1)
    
    # === Step 7: Full conformant development run ===
    # Set environment variables for GPU optimization
    os.environ["HRM_DEVICE"] = "cuda"
    os.environ["HRM_DTYPE"] = "float16"

    # Use C4 protocol v2 (deterministic, reproducible)
    os.environ["C4_PROTOCOL"] = "v2"

    # Check resumability
    out_dir = Path("evidence/gate_c4/full/development")
    if out_dir.exists():
        for arm in ["C4_0", "C4_1", "C4_2", "C4_3", "C4_4", "C4_5", "C4_6"]:
            fpath = out_dir / f"{arm}.jsonl"
            if fpath.exists():
                with open(fpath) as f:
                    lines = sum(1 for _ in f if f.strip())
                print(f"  {arm}: {lines}/120 existing results")

    ret = run(["python", "scripts/run_gate_c4.py", "full", "--split", "development"],
              "Full Conformant Development Run (120 tasks × 7 arms)",
              stream=True)
    if ret != 0:
        print("ERROR: Full run failed!")
        sys.exit(1)
    
    # === Step 8: Run analyzer ===
    ret = run(["python", "scripts/analyze_gate_c4.py",
               "--dir", "evidence/gate_c4/full/development",
               "--output", "evidence/gate_c4/full/development/analysis.json"],
              "C4 Analyzer", timeout=120)
    
    # === Step 9: Composition diagnostic ===
    ret = run(["python", "scripts/diagnose_c4_composition.py"],
              "Composition Diagnostic", timeout=60)
    
    # === Step 10: Verify results ===
    print("\n" + "=" * 60)
    print("  Results Verification")
    print("=" * 60)
    
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        print(f"  Git commit: {manifest.get('git_commit', 'N/A')[:12]}")
        print(f"  Protocol SHA256: {manifest.get('protocol_sha256', 'N/A')[:16]}...")
        print(f"  HRM model: {manifest.get('hrm_model_id', 'N/A')}")
        print(f"  Device: {manifest.get('device', 'N/A')}")
    
    results_hash = out_dir / "RESULTS.sha256"
    if results_hash.exists():
        print(f"\n  RESULTS.sha256:")
        print("  " + results_hash.read_text().replace("\n", "\n  "))
    
    print("\n  Per-arm receipt counts:")
    for arm_id in ["C4_0", "C4_1", "C4_2", "C4_3", "C4_4", "C4_5", "C4_6"]:
        arm_path = out_dir / f"{arm_id}.jsonl"
        if arm_path.exists():
            lines = [l for l in arm_path.read_text().splitlines() if l.strip()]
            print(f"    {arm_id}: {len(lines)} receipts")
        else:
            print(f"    {arm_id}: MISSING")
    
    # === Step 11: Package for download ===
    print("\n" + "=" * 60)
    print("  Packaging Results")
    print("=" * 60)
    
    import shutil
    zip_path = "/content/c4_conformant_results.zip"
    shutil.make_archive(zip_path.replace('.zip', ''), 'zip', 'evidence/gate_c4')
    print(f"  Created: {zip_path}")
    print(f"  Size: {os.path.getsize(zip_path) / 1e6:.1f} MB")
    
    # Try to download (works in Colab)
    try:
        from google.colab import files
        files.download(zip_path)
        print("  Download started.")
    except ImportError:
        print("  (Not in Colab — zip is at /content/c4_conformant_results.zip)")
    
    print("\n" + "=" * 60)
    print("  COMPLETE")
    print("=" * 60)
    print("""
Next steps:
  1. Check the primary delta (C4_4 vs C4_0) — must be >= +0.15
  2. Check family CI lower bound — must be > 0
  3. Check no canonical/abbreviation regression > 0.05
  4. If development passes: run qualification split
  5. If qualification passes: run OOD split
  6. Gate D decision based on all three splits
""")


if __name__ == "__main__":
    main()
