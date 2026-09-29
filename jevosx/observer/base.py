"""Observer interface. The agent loop only depends on this, so fakes and alternative backends plug in cleanly."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..types import AppInfo, Observation, is_console_window


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


@dataclass(frozen=True)
class WindowRef:
    """One app window: the agent's own work window, or the place the person was when the agent needed the keyboard.
    `window` is the live AX handle (None: whichever window the app has focused)."""

    pid: int
    window: Any = field(default=None, compare=False, repr=False)
    title: str = ""

    def same(self, other: WindowRef | None) -> bool:
        if other is None or other.pid != self.pid:
            return False
        if self.window is None or other.window is None:
            return True
        return bool(self.window == other.window)

    @property
    def console(self) -> bool:
        return is_console_window(self.title)


@runtime_checkable
class BackgroundObserver(Protocol):
    """Observers that can watch one window whether or not it is in front (see agent.background)."""

    def front(self) -> WindowRef | None:
        """The frontmost app and its focused window."""
        ...

    def observe_window(self, target: WindowRef) -> Observation:
        """Like observe(), for the target window. Raises StaleElementError when the app or window is gone."""
        ...

    def quick_signature_of(self, target: WindowRef) -> str: ...
