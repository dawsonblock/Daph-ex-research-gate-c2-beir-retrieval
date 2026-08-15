"""Keep exhaustive qualification regeneration out of ordinary test runs."""
from __future__ import annotations

from pathlib import Path

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    explicit = any(
        "tests/qualification" in Path(str(argument)).as_posix()
        for argument in config.args
    )
    if explicit:
        return
    marker = pytest.mark.skip(
        reason="exhaustive oracle regeneration requires an explicit tests/qualification invocation")
    for item in items:
        if "/tests/qualification/" in Path(str(item.path)).as_posix():
            item.add_marker(marker)
