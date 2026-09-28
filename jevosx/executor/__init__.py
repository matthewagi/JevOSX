"""Execution engine: resolved actions → AX actions, value writes and keyboard events."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from .base import DryRunExecutor, Executor
from .keys import KeyBinding, KeyChord, key_vocabulary
from .safety import SafetyPolicy, SafetyVerdict

if TYPE_CHECKING:
    from ..config import ExecutorSettings
    from .mac import MacExecutor

__all__ = [
    "DryRunExecutor",
    "Executor",
    "KeyBinding",
    "KeyChord",
    "SafetyPolicy",
    "SafetyVerdict",
    "create_executor",
    "key_vocabulary",
]


def create_executor(settings: ExecutorSettings | None, frontmost_pid: Callable[[], int | None]) -> MacExecutor:
    """Build the macOS executor lazily so non-macOS hosts can import the package."""
    from .mac import MacExecutor

    return MacExecutor(settings, frontmost_pid=frontmost_pid)
