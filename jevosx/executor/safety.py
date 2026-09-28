"""Guardrails between a decision and the Mac. Deterministic, local, and independent of the model."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from ..config import SafetySettings
from ..types import MENU, OPEN_APP, PRESS_KEY, TYPE_TEXT, Action, AppInfo

Verdict = Literal["allow", "confirm", "deny"]


@dataclass(frozen=True, slots=True)
class SafetyVerdict:
    verdict: Verdict
    reason: str = ""

    @property
    def allowed(self) -> bool:
        return self.verdict == "allow"


class SafetyPolicy:
    def __init__(self, settings: SafetySettings | None = None):
        self.settings = settings or SafetySettings()
        self._patterns = [re.compile(p, re.IGNORECASE) for p in self.settings.confirm_patterns]
        self._deny_apps = {a.lower() for a in self.settings.deny_apps}
        self._confirm_keys = {k.upper() for k in self.settings.confirm_keys}
        self._deny_keys = {k.upper() for k in self.settings.deny_keys}

    def check(self, action: Action, frontmost: AppInfo) -> SafetyVerdict:
        target_app = action.app if action.operation == OPEN_APP else frontmost
        if target_app is not None and self._denied_app(target_app):
            return SafetyVerdict("deny", f"{target_app.name} is on the deny list")
        if action.operation == PRESS_KEY and action.key is not None:
            if action.key.id in self._deny_keys:
                return SafetyVerdict("deny", f"key {action.key.id} is denied")
            if action.key.id in self._confirm_keys:
                return SafetyVerdict("confirm", f"key {action.key.id} ({action.key.description}) needs confirmation")
        label = ""
        if action.element is not None:
            label = action.element.label
            if action.operation == MENU or action.element.kind == "menu_item":
                label = f"{label} {action.element.shortcut or ''}"
        for pattern in self._patterns:
            if label and pattern.search(label):
                return SafetyVerdict(
                    "confirm", f"'{action.element.label if action.element else label}' looks consequential"
                )
        if (
            action.operation == TYPE_TEXT
            and action.element is not None
            and action.element.secure
            and not action.text_is_secret
        ):
            return SafetyVerdict("deny", "only secret text slots may be typed into password fields")
        if self.settings.confirm_all:
            return SafetyVerdict("confirm", "confirm_all is enabled")
        return SafetyVerdict("allow")

    def _denied_app(self, app: AppInfo) -> bool:
        return (app.bundle_id or "").lower() in self._deny_apps or app.name.lower() in self._deny_apps
