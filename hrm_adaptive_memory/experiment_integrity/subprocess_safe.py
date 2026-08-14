"""Every external process call must fail closed, never hang.

A hung provenance check (a git command, a pytest smoke run, an environment
probe) blocking indefinitely on paid GPU infrastructure is a real cost, not a
theoretical one -- this project runs on billed Lightning/RunPod instances.
``subprocess.run``/``check_output``/``Popen`` default to no timeout, so every
call site is a latent hang unless it explicitly opts in. This module makes
opting OUT explicit instead: ``run_safe`` requires a timeout, and the only way
to skip one is to pass ``exempt_reason``, which shows up in the call and in
any audit of the codebase.
"""
from __future__ import annotations

import subprocess
from typing import Any, Sequence


class SubprocessTimeoutPolicyError(ValueError):
    """Raised when a caller tries to run a subprocess with no timeout and no
    documented exemption reason."""


#: Reasonable defaults for the two shapes of call this repo makes. Overridable
#: per call; the point is that SOME bound is always present, not that this
#: exact number is universally correct.
DEFAULT_GIT_TIMEOUT_S = 30
DEFAULT_TEST_SUITE_TIMEOUT_S = 900


def run_safe(
    args: Sequence[str], *, timeout: float | None = None,
    exempt_reason: str | None = None, **kwargs: Any,
) -> subprocess.CompletedProcess:
    """subprocess.run with a mandatory timeout.

    Raises SubprocessTimeoutPolicyError before even attempting the call if
    neither ``timeout`` nor ``exempt_reason`` is given -- a caller must make
    an active choice, not fall through to Python's default of "wait forever."
    On an actual timeout, subprocess.TimeoutExpired propagates normally
    (callers should catch it and fail closed, not retry silently)."""
    if timeout is None and exempt_reason is None:
        raise SubprocessTimeoutPolicyError(
            f"run_safe({list(args)!r}) called with no timeout and no "
            "exempt_reason. Pass an explicit timeout=<seconds>, or "
            "exempt_reason='...' if this call must genuinely be unbounded "
            "(document why -- 'the user is watching an interactive prompt' "
            "is a reason, 'didn't think about it' is not).")
    return subprocess.run(args, timeout=timeout, **kwargs)


def check_output_safe(
    args: Sequence[str], *, timeout: float | None = None,
    exempt_reason: str | None = None, **kwargs: Any,
) -> bytes:
    """subprocess.check_output with the same mandatory-timeout policy as
    run_safe."""
    if timeout is None and exempt_reason is None:
        raise SubprocessTimeoutPolicyError(
            f"check_output_safe({list(args)!r}) called with no timeout and "
            "no exempt_reason.")
    return subprocess.check_output(args, timeout=timeout, **kwargs)
