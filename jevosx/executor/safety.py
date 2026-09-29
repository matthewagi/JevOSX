"""Guardrails between a decision and the Mac. Deterministic, local, and independent of the model."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from ..config import SafetySettings
from ..sites import host_matches, page_host
from ..types import (
    CONSOLE_SAFE_KEYS,
    CONSOLE_SAFE_MENU,
    MENU,
    OPEN_APP,
    PRESS_KEY,
    SAVE_IMAGE,
    TYPE_TEXT,
    Action,
    AppInfo,
    is_console_window,
)

Verdict = Literal["allow", "confirm", "deny"]
# macOS permission and password prompts: granting or refusing access is the person's decision, never the agent's.
SYSTEM_PROMPT_APPS = frozenset(
    {"com.apple.accessibility.universalAccessAuthWarn", "com.apple.UserNotificationCenter", "com.apple.SecurityAgent",
     "com.apple.CoreServicesUIAgent", "com.apple.tccd"}
)  # fmt: skip


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

    def check(
        self, action: Action, frontmost: AppInfo, *, window_title: str | None = None, page_url: str | None = None
    ) -> SafetyVerdict:
        if is_console_window(window_title) and action.operation != OPEN_APP:
            if action.element is not None and action.element.kind != "menu_item":
                return SafetyVerdict("deny", "never acts inside the JevOSX console window")
            if action.operation == PRESS_KEY and (action.key is None or action.key.id not in CONSOLE_SAFE_KEYS):
                return SafetyVerdict("deny", "only new-window/tab shortcuts are allowed in the JevOSX console window")
            if action.operation == MENU and not (action.element and CONSOLE_SAFE_MENU.search(action.element.label)):
                return SafetyVerdict("deny", "only New Window/Tab menu commands are allowed in the JevOSX console")
        if action.operation != OPEN_APP and frontmost.bundle_id in SYSTEM_PROMPT_APPS:
            return SafetyVerdict("deny", "a macOS permission or password prompt is for you to answer")
        target_app = action.app if action.operation == OPEN_APP else frontmost
        if target_app is not None and self._denied_app(target_app):
            return SafetyVerdict("deny", f"{target_app.name} is on the deny list")
        if action.operation == PRESS_KEY and action.key is not None:
            if action.key.id in self._deny_keys:
                return SafetyVerdict("deny", f"key {action.key.id} is denied")
            if action.key.id in self._confirm_keys:
                return SafetyVerdict("confirm", f"key {action.key.id} ({action.key.description}) needs confirmation")
        if action.operation == SAVE_IMAGE:
            return SafetyVerdict("allow")  # a new file in the run's folder; a picture's caption is not a command
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
        if action.operation == TYPE_TEXT and action.element is not None:
            verdict = self._credential(action, page_url)
            if verdict is not None:
                return verdict
        if self.settings.confirm_all:
            return SafetyVerdict("confirm", "confirm_all is enabled")
        return SafetyVerdict("allow")

    def _credential(self, action: Action, page_url: str | None) -> SafetyVerdict | None:
        """Saved logins: only on their own site, only in web pages, passwords only into password fields."""
        element = action.element
        assert element is not None
        if action.secure_only and not element.secure:
            return SafetyVerdict("deny", "a saved password may only be typed into a password field")
        if action.require_host is None:
            return None
        host = page_host(page_url)
        if not element.in_web_area or host is None or not host_matches(action.require_host, host):
            where = host or "a page that is not https"
            return SafetyVerdict("deny", f"the login saved for {action.require_host} is not used on {where}")
        if action.secure_only and self.settings.confirm_credentials:
            return SafetyVerdict("confirm", f"type your saved password for {action.require_host} into this field")
        return None

    def _denied_app(self, app: AppInfo) -> bool:
        return (app.bundle_id or "").lower() in self._deny_apps or app.name.lower() in self._deny_apps
