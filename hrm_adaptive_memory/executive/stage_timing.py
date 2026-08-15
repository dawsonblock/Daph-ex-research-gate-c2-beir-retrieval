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

    #: The atomic stages that are DIRECTLY MEASURED. Every policy-relevant
    #: cost is DERIVED from these by summation (see derived_costs) rather
    #: than timed separately, so the decomposition can never disagree with
    #: its own totals.
    REQUIRED_STAGES = (
        "T_A0_generation",
        "T_probe_retrieval",
        "T_probe_identity_binding",
        "T_G2",
        "T_composition",
        "T_A1_generation",
    )

    #: Stages an ACCEPT still pays for: the answer probe and the cheap
    #: retrieval probe. The executive is NOT deciding whether to pay these --
    #: by the time it decides, they are already spent.
    ACCEPT_STAGES = ("T_A0_generation", "T_probe_retrieval", "T_probe_identity_binding")
    #: The marginal full-memory work an ACCEPT avoids -- the only cost the
    #: executive's decision actually controls.
    ESCALATION_ONLY_STAGES = ("T_G2", "T_composition", "T_A1_generation")

    def __init__(self):
        self.stages: dict[str, float] = {}
        self.cuda_synchronized: bool | None = None

    @contextmanager
    def stage(self, key: str):
        with timed(self.stages, key):
            yield
        if self.cuda_synchronized is None:
            self.cuda_synchronized = cuda_synchronize()

    def _sum(self, keys) -> float:
        return float(sum(self.stages.get(k, 0.0) for k in keys))

    def derived_costs(self) -> dict[str, float]:
        """Policy-relevant costs, DERIVED from the measured atomic stages.

        C_accept    = T_A0 + T_probe
        C_escalate  = T_A0 + T_probe + T_G2 + T_composition + T_A1_generation
        C_avoided   = T_G2 + T_composition + T_A1_generation

        Also reports the true NO-PROBE deployment baselines, reconstructed
        from the same atomic stages (valid because the stages are
        independent sequential operations, so omitting one simply removes
        its duration):
          C_noprobe_answer_only = T_A0
          C_noprobe_full_memory = T_probe + T_G2 + T_composition + T_A1_gen
                                  (goes straight to memory; never runs A0)
        Keeping both lets policy efficiency (gate vs probe+always-escalate)
        be distinguished from deployment efficiency (the complete gated
        system vs the old no-probe fixed policies) -- different questions,
        both of which matter.
        """
        t_probe = self._sum(("T_probe_retrieval", "T_probe_identity_binding"))
        t_a0 = self.stages.get("T_A0_generation", 0.0)
        avoided = self._sum(self.ESCALATION_ONLY_STAGES)
        return {
            "T_probe_total": t_probe,
            "C_accept": t_a0 + t_probe,
            "C_escalate": t_a0 + t_probe + avoided,
            "C_avoided": avoided,
            "C_noprobe_answer_only": t_a0,
            "C_noprobe_full_memory": t_probe + avoided,
        }

    def as_dict(self) -> dict[str, float | bool | None]:
        out: dict[str, float | bool | None] = dict(self.stages)
        out.update(self.derived_costs())
        out["T_total"] = float(sum(self.stages.values()))
        out["cuda_synchronized"] = self.cuda_synchronized
        return out
