#!/usr/bin/env python3
"""C4 Requalification Run — Google Colab T4 GPU.

Runs against the CURRENT WORKING TREE. It does not clone.

    !git clone <repo> /content/repo
    %cd /content/repo
    !git checkout <SHA>
    !python scripts/colab_c4_requalify.py --expected-commit <SHA>

Cloning used to happen inside this script, which meant a session could hold two
checkouts at once -- the one the operator prepared and the one the script
fetched from the default branch -- and debugging would drift between them. A
certification script must not silently replace its own source tree halfway
through execution. The caller checks out an exact revision; this script verifies
HEAD matches --expected-commit and refuses to run otherwise.

Prerequisites:
    - Runtime -> Change runtime type -> T4 GPU + High-RAM
    - A committed revision containing a CAPTURED environment lock (null pins
      abort at step 3)
    - Expected time: ~30-40 minutes on T4

FAIL-CLOSED. Every step that the protocol names as an abort condition aborts.
In particular the test suite: the previous version printed

    "WARNING: Some tests failed... Continuing anyway"

which directly contradicts ``fail_closed_runner.abort_conditions``,
where "test suite fails" is an abort. A run that continues past a failing suite
cannot be certified, so continuing only wastes GPU time.

What it does (in order):
    1.  Verify GPU
    2.  Verify source revision + clean source tree   [ABORTS on mismatch/dirt]
    3.  Install dependencies from the lock           [ABORTS on failure]
    4.  Verify environment against the lock          [ABORTS on mismatch]
    5.  Validate the active protocol semantically (fail-closed)
    6.  Run the full test suite                          [ABORTS on failure]
    7.  Pre-HRM determinism qualification (multi-seed)   [ABORTS on failure]
    8.  Freeze deterministic packets
    9.  CPU-only dry run (conformance gates)             [ABORTS on failure]
    10. C4-BRIDGE gate (expected negative, informational)
    11. HRM smoke test
    12. Full HRM development run (120 x 7 = 840 generations)
    13. Diagnostic arms C4_3o + C4_4m (ordering vs membership 2x2)
    14. Analyzer                                         [ABORTS on failure]
    15. Composition diagnostic (informational)
    16. Results summary
    17. Certification -> CERTIFICATION.json               [decides VALID_RUN]
    18. Package for download (name reflects the verdict)
"""
import argparse
import os
import sys
import time
import json
import shutil
import subprocess
from pathlib import Path


# -- Config -----------------------------------------------------------------

ARMS = ["C4_0", "C4_1", "C4_2", "C4_3", "C4_4", "C4_5", "C4_6"]
DIAGNOSTIC_ARMS = ["C4_3o", "C4_4m"]
DETERMINISM_SEEDS = "0,42,12345"
TOTAL_STEPS = 18

# Set from --expected-commit. The run refuses to proceed unless HEAD matches.
EXPECTED_COMMIT = ""


# -- Helpers ----------------------------------------------------------------

def banner(text, char="=", width=70):
    """Print a visible section banner."""
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}")


def abort(step_label, message):
    """Stop the run. The protocol prefers no result over an uncertifiable one."""
    print(f"\n{'!' * 70}")
    print(f"  ABORT at {step_label}")
    print(f"  {message}")
    print(f"{'!' * 70}")
    print("\n  Nothing downstream of this point could be certified, so the run")
    print("  stops here rather than consuming GPU time on an invalid result.")
    sys.exit(1)


def run(cmd, label, timeout=1800, stream=False, check=True):
    """Run a command with a label and timing. Aborts on failure unless check=False."""
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
            err = result.stderr.strip()
            if err:
                print(f"STDERR: {err[-1500:]}")
        retcode = result.returncode

    elapsed = time.time() - t0
    status = "OK" if retcode == 0 else "FAIL"
    print(f"--- {label}: {status} ({elapsed:.1f}s) ---\n")

    if check and retcode != 0:
        abort(label, f"exited with code {retcode}")

    return retcode


def step(n, text):
    """Print a step indicator."""
    print(f"\n[{n}/{TOTAL_STEPS}] {text}")


# -- Main pipeline ----------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="C4 v2_1 certifying run over the current working tree")
    parser.add_argument(
        "--expected-commit", required=True,
        help="Full or abbreviated (>=7 char) SHA of the revision to certify. "
             "Required: a certifying run must name the revision it certifies, "
             "and HEAD must match it.")
    return parser.parse_args(argv)


def main(argv=None):
    global EXPECTED_COMMIT
    args = parse_args(argv)
    EXPECTED_COMMIT = args.expected_commit.strip()

    banner("C4 Requalification Run — T4 GPU (fail-closed)")
    print(f"Revision:   {EXPECTED_COMMIT}")
    print("Protocol:   C4 v2_1 (deterministic, reproducible, single ordering spec)")
    print(f"Steps:      {TOTAL_STEPS}")
    print("Expected:   ~30-40 minutes on T4 GPU")

    # -- Step 1: Verify GPU ------------------------------------------------
    step(1, "Verifying GPU...")

    try:
        import torch
        print(f"  PyTorch: {torch.__version__}")
        if not torch.cuda.is_available():
            abort("Step 1", "No GPU detected. Runtime -> Change runtime type -> T4 GPU")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        # NB: the attribute is total_memory. The previous `total_mem` raised
        # AttributeError and killed this script on its first step.
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU memory: {mem:.1f} GB")
        print(f"  CUDA: {torch.version.cuda}")
    except ImportError:
        abort("Step 1", "PyTorch is not installed")

    # -- Step 2: Verify the source revision --------------------------------
    step(2, "Verifying source revision...")

    # This script does NOT clone. It certifies the working tree it is run from.
    #
    # It used to clone REPO_URL into its own directory, which meant a session
    # could hold two checkouts at once -- the one the operator prepared and the
    # one this script fetched -- and debugging would drift between them. A
    # certification script must not silently replace its own source tree
    # halfway through execution. The caller clones and checks out; we verify.
    sys.path.insert(0, os.getcwd())
    try:
        from hrm_adaptive_memory.c4 import git_state
    except ImportError as exc:
        abort("Step 2",
              f"cannot import hrm_adaptive_memory from {os.getcwd()}: {exc}\n"
              f"  Run this script from the repository root of the revision you "
              f"intend to certify.")

    state = git_state.inspect(Path.cwd())
    print(f"  Working dir: {os.getcwd()}")
    if not state.is_repo:
        abort("Step 2", f"{state.error}\n"
                        f"  A certifying run must be tied to a commit. Clone "
                        f"the repository and check out the exact revision.")
    print(f"  HEAD:        {state.head}")
    print(f"  Expected:    {EXPECTED_COMMIT}")

    if not git_state.revision_matches(state.head, EXPECTED_COMMIT):
        abort("Step 2",
              f"HEAD is {state.head} but --expected-commit is "
              f"{EXPECTED_COMMIT}.\n"
              f"  Check out the revision you intend to certify:\n"
              f"    git checkout {EXPECTED_COMMIT}")

    # Source dirt is fatal (D8_clean_release). Evidence dirt is this run's own
    # output and is expected -- steps 7-11 rewrite tracked files under
    # evidence/, so a run cannot be required to leave them untouched.
    if state.source_changes or state.other_changes:
        detail = "\n".join(f"    {line}" for line in
                           (state.source_changes + state.other_changes)[:15])
        abort("Step 2",
              f"working tree does not match {EXPECTED_COMMIT[:12]}:\n{detail}\n"
              f"  Certification gate 'source_lineage' would FAIL, so the run "
              f"stops here rather than after the GPU work.\n"
              f"  Do not edit files in this checkout. Change them upstream, "
              f"commit, push, and re-clone.")
    print(f"  Source tree: clean ({len(state.output_changes)} pre-existing "
          f"evidence change(s), which are expected)")

    # -- Step 3: Install dependencies from the lock ------------------------
    step(3, "Installing dependencies from the environment lock...")

    lock_path = Path("configs/c4_requirements.lock")
    if not lock_path.is_file():
        abort("Step 3", f"{lock_path} is missing. The environment cannot be "
                        f"reproduced without a lock. Freeze one first:\n"
                        f"    python scripts/c4_freeze_environment.py")

    from hrm_adaptive_memory.c4.environment_lock import (
        load_lock, pip_requirements, verify_environment)

    lock = load_lock(lock_path)
    requirements = Path("/content/c4_requirements.txt")
    if not requirements.parent.is_dir():
        requirements = Path.cwd() / ".c4_requirements.txt"
    requirements.write_text(pip_requirements(lock))
    print(f"  Lock pins: python=={lock.get('python')}")
    for name, version in sorted((lock.get("packages") or {}).items()):
        print(f"    {name}=={version}")

    unrecorded = [n for n, v in (lock.get("packages") or {}).items() if v is None]
    if unrecorded:
        # MUST_RECORD: an unrecorded version is a failure, not a wildcard, so
        # there is no point running 35 minutes of GPU work behind it.
        abort("Step 3",
              f"the lock leaves {unrecorded} unrecorded (MUST_RECORD).\n"
              f"  Certification cannot pass with null pins. Freeze the lock in "
              f"this runtime, commit it, and run that revision:\n"
              f"    python scripts/c4_freeze_environment.py --note 'colab T4'")

    # check=True: 'dependency version mismatch' is a declared abort condition.
    # A run whose dependencies could not be installed from the lock is not
    # certifiable, so failing here costs seconds instead of half an hour.
    run(["pip", "install", "-q", "-r", str(requirements)],
        "Install locked dependencies", timeout=1200, check=True)
    run(["pip", "install", "-q", "-e", "."], "Install repository (editable)",
        timeout=900, check=True)

    # -- Step 4: Verify the environment against the lock -------------------
    step(4, "Verifying environment against the lock...")

    ok, violations, observed = verify_environment(load_lock(lock_path))
    print(f"  python: {observed['python']}")
    for name, version in sorted(observed["packages"].items()):
        print(f"  {name}: {version}")
    print(f"  accelerator: cuda={observed['accelerator']['cuda']} "
          f"gpu={observed['accelerator']['gpu_name']}")
    if not ok:
        detail = "\n".join(f"    - {v}" for v in violations)
        abort("Step 4",
              f"{len(violations)} environment violation(s):\n{detail}\n"
              f"  'dependency version mismatch' is a declared abort condition. "
              f"A mismatched environment cannot produce a certifiable run, so "
              f"the GPU work is not started.\n"
              f"  Either run the revision whose lock matches this runtime, or "
              f"re-freeze the lock here, commit it, and run that revision.")
    print("\n  Environment matches the lock.")

    # -- Step 5: Validate the active protocol ------------------------------
    step(5, "Validating the active protocol (semantic, fail-closed)...")

    from hrm_adaptive_memory.c4.protocol_validation import (
        ProtocolViolation, load_and_validate_protocol)

    protocol_path = Path("configs/gate_c4_protocol_v2_1.json")
    try:
        protocol, protocol_sha, checks = load_and_validate_protocol(protocol_path)
    except ProtocolViolation as exc:
        abort("Step 5", f"Protocol validation failed: {exc}")

    print(f"  Protocol:      {protocol_path}")
    print(f"  SHA256:        {protocol_sha}")
    print(f"  protocol_id:   {protocol.get('protocol_id')}")
    print("  Resolved invariants (previously printed as 'N/A'):")
    print("    identity resolution (C4_3+): i3_explicit_identity")
    print("    structural selector (C4_4):  s2c_with_s0_fallback")
    print("    fallback selector:           s0 when identity unresolved")
    print("    C4_5 selector:               oracle over the real pool")
    print("    C4_6 selector:               oracle_evidence (reader ceiling)")
    print(f"    ordering policy:             {checks['ordering_policy']}")
    print("    iterative retrieval:         OUTSIDE_PRIMARY_C4")
    print(f"    metric module:               {checks['metric_module']}")
    print(f"    arms validated:              {checks['arms_present']}")

    # -- Step 6: Run test suite (ABORTS on failure) ------------------------
    step(6, "Running the full test suite...")
    print("  Protocol abort condition: 'test suite fails'. No exceptions.")

    run(["python", "-m", "pytest", "tests/", "-q", "--tb=short"],
        "Test Suite", timeout=1800, stream=True, check=True)
    print("  All tests passed.")

    # -- Step 7: Determinism qualification (ABORTS on failure) -------------
    step(7, f"Pre-HRM determinism qualification (120 tasks, seeds {DETERMINISM_SEEDS})...")

    run(["python", "scripts/c4_determinism_qualification.py",
         "--tasks", "120", "--seeds", DETERMINISM_SEEDS, "--arms", "C4_4"],
        "Determinism Qualification", timeout=1800, stream=True, check=True)
    print("  Every compared field identical across seeds.")

    # -- Step 8: Freeze deterministic packets ------------------------------
    step(8, "Freezing deterministic packets (immutable, hashed)...")

    run(["python", "scripts/c4_freeze_packets.py", "--split", "development"],
        "Freeze Packets", timeout=900)

    # -- Step 9: CPU-only dry run ------------------------------------------
    step(9, "CPU-only dry run (conformance gates)...")

    run(["python", "scripts/run_gate_c4.py", "dry-run"],
        "Dry Run (Conformance Gates)", timeout=1800, stream=True, check=True)

    # -- Step 10: C4-BRIDGE gate (informational) ---------------------------
    step(10, "C4-BRIDGE gate (expected negative result)...")

    run(["python", "scripts/run_gate_c4_bridge.py"],
        "C4-BRIDGE Gate", timeout=600, check=False)
    print("  Informational: the negative result is why iterative retrieval is "
          "outside primary C4.")

    # -- Step 11: HRM smoke test -------------------------------------------
    step(11, "HRM smoke test...")

    os.environ["HRM_DEVICE"] = "cuda"
    os.environ["HRM_DTYPE"] = "float16"
    os.environ["C4_PROTOCOL"] = "v2_1"

    run(["python", "scripts/run_gate_c4.py", "smoke"], "HRM Smoke Test",
        timeout=1800, stream=True, check=True)

    # -- Step 12: Full HRM development run ---------------------------------
    step(12, "Full HRM development run (120 tasks x 7 arms = 840 generations)")
    print("  Expected: ~15-25 minutes on T4. Resumable.\n")

    out_dir = Path("evidence/gate_c4/full/development")
    if out_dir.exists():
        for arm in ARMS:
            fpath = out_dir / f"{arm}.jsonl"
            if fpath.exists():
                lines = sum(1 for l in fpath.read_text().splitlines() if l.strip())
                if lines:
                    print(f"  {arm}: {lines}/120 existing results (resume key "
                          f"decides reuse)")

    run(["python", "scripts/run_gate_c4.py", "full", "--split", "development"],
        "Full Development Run", stream=True, check=True)

    # -- Step 13: Diagnostic arms ------------------------------------------
    step(13, "Diagnostic arms C4_3o + C4_4m (ordering vs membership 2x2)...")
    print("  Decomposes Q(C4_4) - Q(C4_3) into ordering, membership, interaction:")
    print("    C4_3  = S0  membership + pool order")
    print("    C4_3o = S0  membership + deterministic order")
    print("    C4_4m = S2c membership + pool order")
    print("    C4_4  = S2c membership + deterministic order")
    print("  ~5 minutes on T4 (240 additional generations)\n")

    run(["python", "scripts/run_gate_c4.py", "full", "--split", "development",
         "--arms"] + DIAGNOSTIC_ARMS,
        "Diagnostic Arms (ordering vs membership)", stream=True, check=True)

    # -- Step 14: Analyzer (ABORTS on failure) -----------------------------
    step(14, "Running the analyzer (quality, gap capture, family CIs, flips)...")
    print("  The analyzer is authoritative for promotion, so it is not optional.")

    run(["python", "scripts/analyze_gate_c4.py",
         "--dir", str(out_dir),
         "--output", str(out_dir / "analysis.json")],
        "C4 Analyzer", timeout=600, stream=True, check=True)

    # RESULTS.sha256 must be rewritten after the analyzer and the diagnostic
    # arms changed the bundle contents, or the hash gate fails on stale data.
    from hrm_adaptive_memory.c4.provenance import write_results_hash
    write_results_hash(out_dir)
    print(f"  RESULTS.sha256 rewritten over the final bundle contents.")

    # -- Step 15: Composition diagnostic (informational) -------------------
    step(15, "Composition diagnostic...")

    run(["python", "scripts/diagnose_c4_composition.py"],
        "Composition Diagnostic", timeout=300, check=False)

    # -- Step 16: Results summary ------------------------------------------
    step(16, "Results summary...")

    banner("Results Summary", char="-", width=50)

    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        print(f"  Git commit:       {str(manifest.get('git_commit', 'N/A'))[:12]}")
        print(f"  Protocol SHA256:  {str(manifest.get('protocol_sha256', 'N/A'))[:16]}...")
        print(f"  HRM model:        {manifest.get('hrm_model_id', 'N/A')}")
        print(f"  Device:           {manifest.get('device', 'N/A')}")
        print(f"  Created:          {manifest.get('created_utc', 'N/A')}")

    print("\n  Per-arm receipt counts:")
    for arm_id in ARMS + DIAGNOSTIC_ARMS:
        arm_path = out_dir / f"{arm_id}.jsonl"
        if arm_path.exists():
            lines = [l for l in arm_path.read_text().splitlines() if l.strip()]
            flag = "OK" if len(lines) == 120 else "INCOMPLETE"
            print(f"    {arm_id}: {len(lines)}/120  [{flag}]")
        else:
            print(f"    {arm_id}: MISSING")

    analysis_path = out_dir / "analysis.json"
    if analysis_path.exists():
        analysis = json.loads(analysis_path.read_text())

        # analysis.json stores arm quality under arm_summary[arm]["quality"];
        # the previous code read a nonexistent "arm_quality" key and printed
        # nothing, and formatted the primary_delta dict as a float (TypeError).
        arm_summary = analysis.get("arm_summary", {})
        print("\n  Quality scores:")
        for arm_id in ARMS + DIAGNOSTIC_ARMS:
            summary = arm_summary.get(arm_id)
            if summary is not None:
                print(f"    {arm_id}: {summary['quality']:.4f} "
                      f"(correct {summary['correct_rate']:.4f}, n={summary['n']})")

        delta = analysis.get("primary_delta") or {}
        if delta:
            print(f"\n  Primary delta (C4_4 - C4_0): {delta['mean_delta']:+.4f}")
            print(f"    family CI:   [{delta['ci_lower']:+.4f}, {delta['ci_upper']:+.4f}]")
            print(f"    cluster CI:  [{delta.get('ci_lower_cluster', float('nan')):+.4f}, "
                  f"{delta.get('ci_upper_cluster', float('nan')):+.4f}]")
            print(f"    template CI: [{delta.get('ci_lower_template', float('nan')):+.4f}, "
                  f"{delta.get('ci_upper_template', float('nan')):+.4f}]")
            threshold = 0.15
            verdict = "PASS" if delta["mean_delta"] >= threshold else "FAIL"
            print(f"    threshold {threshold:+.2f}: {verdict}")

        sgc = analysis.get("selector_gap_capture")
        ogc = analysis.get("oracle_gap_capture")
        if sgc is not None:
            print(f"\n  Selector gap capture: {sgc:.4f}")
        if ogc is not None:
            print(f"  Oracle gap capture:   {ogc:.4f}")

        dec = analysis.get("ordering_membership_decomposition") or {}
        print("\n  Ordering vs membership decomposition:")
        if not dec.get("available"):
            print(f"    NOT AVAILABLE (missing {dec.get('missing_arms')})")
        else:
            q = dec["quality"]
            print(f"    Q(C4_3)  = {q['C4_3']:.4f}  (S0  + pool order)")
            print(f"    Q(C4_3o) = {q['C4_3o']:.4f}  (S0  + deterministic order)")
            print(f"    Q(C4_4m) = {q['C4_4m']:.4f}  (S2c + pool order)")
            print(f"    Q(C4_4)  = {q['C4_4']:.4f}  (S2c + deterministic order)")
            print(f"    ordering effect   = {dec['ordering_effect']:+.4f}")
            print(f"    membership effect = {dec['membership_effect']:+.4f}")
            print(f"    combined effect   = {dec['combined_effect']:+.4f}")
            print(f"    interaction       = {dec['interaction_effect']:+.4f}")

    # -- Step 17: Certification --------------------------------------------
    step(17, "Certification (decides VALID_RUN)...")
    print("  VALID_RUN is the conjunction of every gate, each derived from the")
    print("  artifacts. Tests are re-run inside certification against the exact")
    print("  bundle being certified.\n")

    cert_ret = run(["python", "scripts/certify_c4_run.py",
                    "--bundle", str(out_dir),
                    "--protocol", "configs/gate_c4_protocol_v2_1.json",
                    "--lock", "configs/c4_requirements.lock"],
                   "Certification", timeout=2400, stream=True, check=False)

    cert_path = out_dir / "certification/CERTIFICATION.json"
    valid_run = False
    certificate = {}
    if cert_path.exists():
        certificate = json.loads(cert_path.read_text())
        valid_run = bool(certificate.get("VALID_RUN"))
        print(f"\n  VALID_RUN: {valid_run}")
        print(f"  verdict:   {certificate.get('verdict')}")
        print(f"  gates:     {certificate.get('gates_passed')}/"
              f"{certificate.get('gates_total')} passed")
        if certificate.get("gates_failed"):
            print(f"  failed:    {certificate['gates_failed']}")
    else:
        print(f"\n  Certification produced no artifact (exit {cert_ret}).")

    # -- Step 18: Package for download -------------------------------------
    step(18, "Packaging results for download...")

    # The archive name states the verdict. An uncertified bundle must never be
    # named VALID_CONFORMANT_*: the name is the thing people cite later.
    stem = ("VALID_CONFORMANT_C4_V2_DEVELOPMENT_RESULT" if valid_run
            else "UNCERTIFIED_C4_V2_DEVELOPMENT_RESULT")
    zip_base = f"/content/{stem}"
    shutil.make_archive(zip_base, "zip", "evidence/gate_c4")
    zip_path = f"{zip_base}.zip"
    size_mb = os.path.getsize(zip_path) / 1e6
    print(f"  Created: {zip_path}")
    print(f"  Size: {size_mb:.1f} MB")

    try:
        from google.colab import files
        files.download(zip_path)
        print("  Download started in browser.")
    except ImportError:
        print(f"  (Not in Colab — zip is at {zip_path})")

    # -- Done ---------------------------------------------------------------
    if valid_run:
        banner("COMPLETE — VALID_RUN = true")
        print(f"""
  Certified bundle: {zip_path}
  Certificate:      {cert_path}

  Next:
    1. Commit the certificate and the environment lock.
    2. Run the untouched qualification split.
    3. Then the OOD split.
    4. Gate D decision across all three splits.
""")
        return 0

    banner("COMPLETE — VALID_RUN = false (NOT CERTIFIED)")
    print(f"""
  The measurement finished, but the run is NOT certified.

  Bundle:      {zip_path}
  Certificate: {cert_path}
  Failed gates: {certificate.get('gates_failed', 'certification did not run')}

  The numbers in this bundle may be scientifically informative, but do not
  cite them as a conformant C4 v2 result and do not rename the archive to
  VALID_CONFORMANT_* until every gate passes.
""")
    return 1


if __name__ == "__main__":
    sys.exit(main())
