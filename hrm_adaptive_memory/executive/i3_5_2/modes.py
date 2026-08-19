"""Governor operation modes for V2B-I3.5.2."""
from __future__ import annotations

from enum import Enum


class GovernorMode(str, Enum):
    """How the governor interacts with the trajectory loop."""
    OFF = "OFF"
    ALWAYS_ON = "ALWAYS_ON"
    SELECTIVE = "SELECTIVE"
    SELECTIVE_FRAME = "SELECTIVE_FRAME"
    SHADOW_SELECTIVE = "SHADOW_SELECTIVE"
    # I3.5.3-r1: Base-first pairwise advantage gate.
    # Call model with base packet -> get a_B.
    # Governor produces a_G. If a_G != a_B, evaluate pairwise advantage.
    # If LCB(dQ_pi) > threshold, call model again with governor packet -> a_T.
    # Else, execute a_B (no extra model call).
    SELECTIVE_QPIB_BASE_FIRST = "SELECTIVE_QPIB_BASE_FIRST"
