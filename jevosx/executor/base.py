"""Executor interface plus a dry-run implementation."""

from __future__ import annotations

from typing import Protocol

from ..types import Action, ActionResult, AppInfo, Observation


class Executor(Protocol):
    def validate(self, action: Action, obs: Observation) -> None:
        """Raise StaleElementError if the target no longer matches what was observed."""
        ...

    def execute(self, action: Action, obs: Observation) -> ActionResult: ...

    def open_app(self, app: AppInfo) -> ActionResult: ...


class DryRunExecutor:
    """Logs what would happen. Used by `jevosx run --dry-run` and for inspecting decisions safely."""

    def __init__(self) -> None:
        self.actions: list[Action] = []

    def validate(self, action: Action, obs: Observation) -> None:
        return None

    def execute(self, action: Action, obs: Observation) -> ActionResult:
        self.actions.append(action)
        return ActionResult(ok=True, method="dry-run", detail=action.describe())

    def open_app(self, app: AppInfo) -> ActionResult:
        return ActionResult(ok=True, method="dry-run", detail=f"open {app.name}")
