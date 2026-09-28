"""Observer interface. The agent loop only depends on this, so fakes and alternative backends plug in cleanly."""

from __future__ import annotations

from typing import Protocol

from ..types import AppInfo, Observation


class Observer(Protocol):
    def observe(self) -> Observation:
        """Capture the frontmost app's focused window plus whole-desktop context."""
        ...

    def quick_signature(self) -> str:
        """A cheap (few IPC calls) signature used to detect when the UI has settled after an action."""
        ...

    def find_app(self, query: str) -> AppInfo | None:
        """Resolve an app by name or bundle id among running and installed apps."""
        ...
