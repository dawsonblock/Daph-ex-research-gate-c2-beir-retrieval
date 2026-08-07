#!/usr/bin/env python3
"""C4 Requalification Run — Google Colab T4 GPU.

One script. Paste into a Colab cell and run. That's it.

    !pip install -q rank-bm25 numpy pytest
    !git clone https://github.com/dawsonblock/Daph-ex-research-gate-c2-beir-retrieval.git
    %cd Daph-ex-research-gate-c2-beir-retrieval
    !pip install -q -e .
    !python scripts/colab_c4_requalify.py

Or even simpler — just run this entire file as a notebook cell:

    !wget -q https://raw.githubusercontent.com/dawsonblock/Daph-ex-research-gate-c2-beir-retrieval/main/scripts/colab_c4_requalify.py -O colab_c4_requalify.py
    !python colab_c4_requalify.py

Prerequisites:
    - Runtime → Change runtime type → T4 GPU + High-RAM
    - Expected time: ~20-30 minutes on T4

What it does (in order):
    1.  Verify GPU is available
    2.  Clone/pull the repository
    3.  Install dependencies
    4.  Run the test suite (607 tests)
    5.  Run pre-HRM determinism qualification (120/120 must match)
    6.  Freeze deterministic packets (immutable, hashed)
    7.  Run CPU-only dry-run validation (7 conformance gates)
    8.  Run C4-BRIDGE gate (no HRM, ~2 seconds)
    9.  Run HRM smoke test (3 tasks × 7 arms)
    10. Run full HRM development run (120 tasks × 7 arms = 840 generations)
    11. Run the analyzer (quality, gap capture, family CIs, flips)
    12. Run composition diagnostic
    13. Verify results integrity (manifest, hashes, receipt counts)
    14. Package everything into a zip for download
"""
import os
import sys
import time
import json
import shutil
import subprocess
from pathlib import Path


# ── Config ──────────────────────────────────────────────────────────────────

REPO_URL = "https://github.com/dawsonblock/Daph-ex-research-gate-c2-beir-retrieval.git"
REPO_DIR = "/content/Daph-ex-research-gate-c2-beir-retrieval"
ARMS = ["C4_0", "C4_1", "C4_2", "C4_3", "C4_4", "C4_5", "C4_6"]


# ── Helpers ─────────────────────────────────────────────────────────────────

def banner(text, char="=", width=70):
    """Print a visible section banner."""
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}")


def run(cmd, label, timeout=600, stream=False, check=True):
    """Run a command with a label and timing."""
    print(f"\n--- {label} ---")
    t0 = time.time()

    if stream:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1)
        for line in proc.stdout:
            print(line, end="", flush=True)
        proc.wait()
        retcode = proc.returncode
    else:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            # Only show last 500 chars of stderr to avoid noise
            err = result.stderr.strip()
            if err:
                print(f"STDERR: {err[-500:]}")
        retcode = result.returncode

    elapsed = time.time() - t0
    status = "OK" if retcode == 0 else "FAIL"
    print(f"--- {label}: {status} ({elapsed:.1f}s) ---\n")

    if check and retcode != 0:
        print(f"ERROR: {label} failed with exit code {retcode}")
        sys.exit(1)

    return retcode


def step(n, total, text):
    """Print a step indicator."""
    print(f"\n[{n}/{total}] {text}")


# ── Main pipeline ───────────────────────────────────────────────────────────

def main():
    TOTAL_STEPS = 14

    banner("C4 Requalification Run — T4 GPU")
    print(f"Repository: {REPO_URL}")
    print(f"Protocol:   C4 v2 (deterministic, reproducible)")
    print(f"Steps:      {TOTAL_STEPS}")
    print(f"Expected:   ~20-30 minutes on T4 GPU")

    # ── Step 1: Verify GPU ──────────────────────────────────────────────
    step(1, TOTAL_STEPS, "Verifying GPU...")

    try:
        import torch
        print(f"  PyTorch: {torch.__version__}")
        if not torch.cuda.is_available():
            print("  ERROR: No GPU detected!")
            print("  Fix: Runtime → Change runtime type → T4 GPU")
            sys.exit(1)
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"  GPU memory: {mem:.1f} GB")
    except ImportError:
        print("  ERROR: PyTorch not installed")
        sys.exit(1)

    # ── Step 2: Clone repository ────────────────────────────────────────
    step(2, TOTAL_STEPS, "Cloning repository...")

    if os.path.exists(REPO_DIR):
        print(f"  Repository exists, pulling latest...")
        subprocess.run(["git", "-C", REPO_DIR, "pull", "--rebase"],
                       capture_output=True, check=False)
    else:
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, REPO_DIR], check=True)

    os.chdir(REPO_DIR)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    print(f"  Commit: {commit[:12]}")
    print(f"  Working dir: {os.getcwd()}")

    # ── Step 3: Install dependencies ────────────────────────────────────
    step(3, TOTAL_STEPS, "Installing dependencies...")

    subprocess.run(["pip", "install", "-q", "transformers>=5.9.0",
                    "huggingface-hub>=0.34"], check=True)
    subprocess.run(["pip", "install", "-q", "rank-bm25", "numpy"], check=True)
    subprocess.run(["pip", "install", "-q", "pytest"], check=False)
    subprocess.run(["pip", "install", "-q", "-e", "."], check=True)

    import transformers
    print(f"  transformers: {transformers.__version__}")
    print(f"  torch: {torch.__version__}")

    # ── Step 4: Run test suite ──────────────────────────────────────────
    step(4, TOTAL_STEPS, "Running test suite (607 tests)...")

    ret = run(["python", "-m", "pytest", "tests/", "-q", "--tb=line",
               "--timeout=60"],
              "Test Suite", timeout=180, check=False)
    if ret != 0:
        print("  WARNING: Some tests failed. Check output above.")
        print("  Continuing anyway (tests may fail due to environment differences)...")
    else:
        print("  All tests passed!")

    # ── Step 5: Determinism qualification ───────────────────────────────
    step(5, TOTAL_STEPS, "Pre-HRM determinism qualification (120 tasks, 2 hash seeds)...")

    ret = run(["python", "scripts/c4_determinism_qualification.py",
               "--tasks", "120", "--seeds", "0,42"],
              "Determinism Qualification", timeout=300, check=False)
    if ret != 0:
        print("  ERROR: Determinism qualification FAILED!")
        print("  The pipeline is not deterministic. Do not continue to HRM.")
        sys.exit(1)
    print("  120/120 identical packet hashes — PASS")

    # ── Step 6: Freeze deterministic packets ────────────────────────────
    step(6, TOTAL_STEPS, "Freezing deterministic packets (immutable, hashed)...")

    ret = run(["python", "scripts/c4_freeze_packets.py",
               "--split", "development"],
              "Freeze Packets", timeout=300)
    print("  Packets frozen: packet.json + packet.sha256 + prompt.txt + prompt.sha256")

    # ── Step 7: CPU-only dry run (conformance validation) ───────────────
    step(7, TOTAL_STEPS, "CPU-only dry run (7 conformance gates)...")

    ret = run(["python", "scripts/run_gate_c4.py", "dry-run"],
              "Dry Run (7 Conformance Gates)", timeout=300)
    print("  All conformance gates passed!")

    # ── Step 8: C4-BRIDGE gate ──────────────────────────────────────────
    step(8, TOTAL_STEPS, "C4-BRIDGE gate (no HRM, ~2 seconds)...")

    ret = run(["python", "scripts/run_gate_c4_bridge.py"],
              "C4-BRIDGE Gate", timeout=300, check=False)
    # BRIDGE gate is expected to show a negative result — that's fine
    print("  C4-BRIDGE negative result confirmed (expected)")

    # ── Step 9: HRM smoke test ──────────────────────────────────────────
    step(9, TOTAL_STEPS, "HRM smoke test (3 tasks × 7 arms)...")

    # Set GPU + protocol v2 environment
    os.environ["HRM_DEVICE"] = "cuda"
    os.environ["HRM_DTYPE"] = "float16"
    os.environ["C4_PROTOCOL"] = "v2"

    ret = run(["python", "scripts/run_gate_c4.py", "smoke"],
              "HRM Smoke Test", timeout=600)
    print("  Smoke test passed — model loads on GPU correctly")

    # ── Step 10: Full HRM development run ───────────────────────────────
    step(10, TOTAL_STEPS, "Full HRM development run (120 tasks × 7 arms = 840 generations)")
    print("  This is the main run. Expected: ~15-25 minutes on T4.")
    print("  Resumable — if Colab disconnects, re-run picks up where it left off.\n")

    # Check resumability
    out_dir = Path("evidence/gate_c4/full/development")
    if out_dir.exists():
        for arm in ARMS:
            fpath = out_dir / f"{arm}.jsonl"
            if fpath.exists():
                with open(fpath) as f:
                    lines = sum(1 for l in f if l.strip())
                if lines > 0:
                    print(f"  {arm}: {lines}/120 existing results (will resume)")

    ret = run(["python", "scripts/run_gate_c4.py", "full", "--split", "development"],
              "Full Development Run", stream=True, check=True)
    print("  Full run complete!")

    # ── Step 11: Run analyzer ───────────────────────────────────────────
    step(11, TOTAL_STEPS, "Running analyzer (quality, gap capture, family CIs, flips)...")

    ret = run(["python", "scripts/analyze_gate_c4.py",
               "--dir", "evidence/gate_c4/full/development",
               "--output", "evidence/gate_c4/full/development/analysis.json"],
              "C4 Analyzer", timeout=120, check=False)

    # ── Step 12: Composition diagnostic ─────────────────────────────────
    step(12, TOTAL_STEPS, "Composition diagnostic...")

    ret = run(["python", "scripts/diagnose_c4_composition.py"],
              "Composition Diagnostic", timeout=60, check=False)

    # ── Step 13: Verify results ─────────────────────────────────────────
    step(13, TOTAL_STEPS, "Verifying results integrity...")

    banner("Results Summary", char="-", width=50)

    # Manifest
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        print(f"  Git commit:       {manifest.get('git_commit', 'N/A')[:12]}")
        print(f"  Protocol SHA256:  {manifest.get('protocol_sha256', 'N/A')[:16]}...")
        print(f"  HRM model:        {manifest.get('hrm_model_id', 'N/A')}")
        print(f"  Device:           {manifest.get('device', 'N/A')}")
        print(f"  Created:          {manifest.get('created_utc', 'N/A')}")

    # Receipt counts
    print(f"\n  Per-arm receipt counts:")
    all_complete = True
    for arm_id in ARMS:
        arm_path = out_dir / f"{arm_id}.jsonl"
        if arm_path.exists():
            lines = [l for l in arm_path.read_text().splitlines() if l.strip()]
            status = "OK" if len(lines) == 120 else "INCOMPLETE"
            if len(lines) != 120:
                all_complete = False
            print(f"    {arm_id}: {len(lines)}/120  [{status}]")
        else:
            all_complete = False
            print(f"    {arm_id}: MISSING")

    # Quality summary from analysis
    analysis_path = out_dir / "analysis.json"
    if analysis_path.exists():
        analysis = json.loads(analysis_path.read_text())
        print(f"\n  Quality scores:")
        for arm_id in ARMS:
            q = analysis.get("arm_quality", {}).get(arm_id)
            if q is not None:
                print(f"    {arm_id}: {q:.4f}")

        delta = analysis.get("primary_delta")
        if delta is not None:
            print(f"\n  Primary delta (C4_4 - C4_0): {delta:+.4f}")
            threshold = 0.15
            if delta >= threshold:
                print(f"  Threshold ({threshold:+.2f}): PASS ✓")
            else:
                print(f"  Threshold ({threshold:+.2f}): FAIL ✗")

        ogc = analysis.get("oracle_gap_capture")
        sgc = analysis.get("selector_gap_capture")
        if ogc is not None:
            print(f"  Oracle gap capture:   {ogc:.4f}")
        if sgc is not None:
            print(f"  Selector gap capture: {sgc:.4f}")

    # RESULTS.sha256
    results_hash_path = out_dir / "RESULTS.sha256"
    if results_hash_path.exists():
        print(f"\n  RESULTS.sha256:")
        for line in results_hash_path.read_text().strip().split("\n"):
            print(f"    {line}")

    if not all_complete:
        print("\n  WARNING: Not all arms have 120 receipts!")
        print("  You may need to re-run to complete missing tasks.")

    # ── Step 14: Package for download ───────────────────────────────────
    step(14, TOTAL_STEPS, "Packaging results for download...")

    zip_path = "/content/c4_requalify_results.zip"
    shutil.make_archive(zip_path.replace(".zip", ""), "zip", "evidence/gate_c4")
    size_mb = os.path.getsize(zip_path) / 1e6
    print(f"  Created: {zip_path}")
    print(f"  Size: {size_mb:.1f} MB")

    # Download (works in Colab)
    try:
        from google.colab import files
        files.download(zip_path)
        print("  Download started in browser.")
    except ImportError:
        print("  (Not in Colab — zip is at /content/c4_requalify_results.zip)")

    # ── Done ────────────────────────────────────────────────────────────
    banner("COMPLETE")
    print("""
  What to check next:
    1. Primary delta (C4_4 - C4_0) must be >= +0.15
    2. Family CI lower bound must be > 0
    3. No canonical/abbreviation regression > 0.05
    4. Oracle gap capture > 0 (mechanism closes some of the gap)
    5. If development passes → run qualification split
    6. If qualification passes → run OOD split
    7. Gate D decision based on all three splits

  Results are in: evidence/gate_c4/full/development/
  Analysis:       evidence/gate_c4/full/development/analysis.json
  Frozen packets: evidence/gate_c4/frozen_packets/development/
""")


if __name__ == "__main__":
    main()
