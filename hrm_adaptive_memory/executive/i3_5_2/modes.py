"""Governor operation modes for V2B-I3.5.2."""
from __future__ import annotations

from enum import Enum


class GovernorMode(str, Enum):
    """How the governor interacts with the trajectory loop."""
    OFF = "OFF"
    ALWAYS_ON = "ALWAYS_ON"
    SELECTIVE = "SELECTIVE"
    SHADOW_SELECTIVE = "SHADOW_SELECTIVE"
