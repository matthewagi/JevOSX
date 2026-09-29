"""OS observer layer: macOS Accessibility (AXUIElement) → structured, indexed text state.

Pixels are read (on-device OCR, observer/vision.py) only for windows that draw their own interface."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import Observer
from .textmap import render_text_map
from .walker import TreeWalker, WalkLimits, WalkResult

if TYPE_CHECKING:
    from ..config import ObserverSettings
    from .desktop import MacDesktopObserver

__all__ = ["Observer", "TreeWalker", "WalkLimits", "WalkResult", "render_text_map", "create_observer"]


def create_observer(settings: ObserverSettings | None = None) -> MacDesktopObserver:
    """Build the macOS observer. Imported lazily so non-macOS hosts can still use the rest of the package."""
    from .desktop import MacDesktopObserver

    return MacDesktopObserver(settings)
