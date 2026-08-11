"""Monotonic, CUDA-safe per-stage timing for RETRIEVAL_PROBE_GATE_V1.

Per configs/gate_retrieval_probe_v1_design.json COST_MODEL_MUST_CHANGE_AND_WHY:
this study's PRIMARY cost unit is measured wall-clock latency, because a
retrieval probe emits no tokens and would therefore appear free under the
token metric inherited from the closed confidence-only branch.

Two correctness requirements this module exists to enforce:

  1. Monotonic high-resolution clock. time.perf_counter() is monotonic and
     not subject to wall-clock adjustments; time.time() is neither.

  2. CUDA synchronization. torch launches kernels ASYNCHRONOUSLY, so a timer
     stopped without synchronizing measures queue-submission time, not
     execution time -- which would make apparent latency savings on the
     generation stages meaningless. We synchronize immediately BEFORE
     starting and immediately BEFORE stopping every timer, so the interval
     brackets actually-completed GPU work.
"""
from __future__ import annotations

import time
from contextlib import contextmanager

#: Resolved once. Import torch lazily so this module stays usable (and
#: testable) in CPU-only or torch-free environments.
_TORCH = None
_TORCH_CHECKED = False


def _torch_or_none():
    global _TORCH, _TORCH_CHECKED
    if not _TORCH_CHECKED:
        _TORCH_CHECKED = True
        try:
            import torch  # noqa: PLC0415
            _TORCH = torch
        except Exception:
            _TORCH = None
    return _TORCH


def cuda_synchronize() -> bool:
    """Block until all queued GPU work has completed. Returns True if a
    synchronize actually happened (CUDA present), False otherwise -- the
    return value is recorded in receipts so a reader can tell whether a
    given run's latencies are CUDA-safe or were taken on CPU."""
    torch = _torch_or_none()
    if torch is None:
        return False
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            return True
    except Exception:
        return False
    return False


@contextmanager
def timed(store: dict[str, float], key: str):
    """Record the wall-clock duration of a stage into ``store[key]``.

    Synchronizes CUDA on both ends so the measured interval brackets
    completed GPU work rather than kernel-launch queueing.
    """
    cuda_synchronize()
    start = time.perf_counter()
    try:
        yield
    finally:
        cuda_synchronize()
        store[key] = time.perf_counter() - start


class StageTimer:
    """Accumulates named stage durations plus a total, and records whether
    the timings were taken with CUDA synchronization in force."""

    #: The stages configs/gate_retrieval_probe_v1_design.json requires be
    #: recorded separately.
    REQUIRED_STAGES = (
        "T_answer_probe",
        "T_retrieval_probe",
        "T_G2",
        "T_composition",
        "T_memory_generation",
    )

    def __init__(self):
        self.stages: dict[str, float] = {}
        self.cuda_synchronized: bool | None = None

    @contextmanager
    def stage(self, key: str):
        with timed(self.stages, key):
            yield
        if self.cuda_synchronized is None:
            self.cuda_synchronized = cuda_synchronize()

    def as_dict(self) -> dict[str, float | bool | None]:
        out: dict[str, float | bool | None] = dict(self.stages)
        out["T_total"] = float(sum(self.stages.values()))
        out["cuda_synchronized"] = self.cuda_synchronized
        return out
