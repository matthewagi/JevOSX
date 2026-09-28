"""Deterministic macOS execution engine.

Every action resolves to an AX handle captured during observation: AXPress/AXConfirm for clicks, AXValue writes
(or layout-independent Unicode keystrokes) for text, AXPress on menu items, AXRaise for windows, scroll-bar value
changes for scrolling, NSWorkspace/`open` for apps. Pointer clicks are an opt-in last resort.
"""

from __future__ import annotations

import contextlib
import subprocess
import time
from collections.abc import Callable
from typing import Any

from ..config import ExecutorSettings
from ..errors import AXError, StaleElementError
from ..types import (
    CLICK,
    FOCUS_WINDOW,
    MENU,
    OPEN_APP,
    PRESS_KEY,
    SCROLL_DOWN,
    SCROLL_UP,
    TYPE_TEXT,
    Action,
    ActionResult,
    AppInfo,
    Observation,
    UIElement,
)
from . import input as keyboard
from .keys import KeyChord

PRESS_ACTIONS = ("AXPress", "AXConfirm", "AXPick", "AXOpen")
BROWSER_BUNDLES = frozenset(
    {
        "com.apple.Safari",
        "com.apple.SafariTechnologyPreview",
        "com.google.Chrome",
        "com.google.Chrome.canary",
        "com.microsoft.edgemac",
        "com.brave.Browser",
        "company.thebrowser.Browser",
        "com.vivaldi.Vivaldi",
        "com.operasoftware.Opera",
        "org.chromium.Chromium",
        "org.mozilla.firefox",
    }
)
ELEMENT_OPERATIONS = frozenset({CLICK, TYPE_TEXT, MENU, SCROLL_UP, SCROLL_DOWN})


class MacExecutor:
    def __init__(self, settings: ExecutorSettings | None = None, *, frontmost_pid: Callable[[], int | None]):
        from ..observer.ax import AXNode, require_ax

        require_ax()
        self.settings = settings or ExecutorSettings()
        self._frontmost_pid = frontmost_pid
        self._AXNode = AXNode

    # ---- freshness ----------------------------------------------------------------------------------------------
    def validate(self, action: Action, obs: Observation) -> None:
        if action.operation in ELEMENT_OPERATIONS | {PRESS_KEY, FOCUS_WINDOW}:
            pid = self._frontmost_pid()
            if pid != obs.app.pid:
                raise StaleElementError(f"frontmost app changed (pid {obs.app.pid} → {pid})")
        element = action.element
        if element is None or element.node is None:
            return
        attrs = element.node.get_many(("AXRole", "AXSubrole", "AXEnabled"))
        if attrs.get("AXRole") != element.role or (attrs.get("AXSubrole") or None) != element.subrole:
            raise StaleElementError(f"{element.describe()} changed role")
        if element.kind != "menu_item" and attrs.get("AXEnabled") is False and element.enabled:
            raise StaleElementError(f"{element.describe()} became disabled")

    # ---- dispatch -----------------------------------------------------------------------------------------------
    def execute(self, action: Action, obs: Observation) -> ActionResult:
        started = time.perf_counter()
        op = action.operation
        try:
            if op == CLICK and action.element is not None:
                result = self._click(action.element)
            elif op == TYPE_TEXT and action.element is not None and action.text is not None:
                keys = obs.app.bundle_id in BROWSER_BUNDLES
                result = self._type(action.element, action.text, secret=action.text_is_secret, prefer_keys=keys)
            elif op == MENU and action.element is not None:
                action.element.node.perform("AXPress")
                result = ActionResult(True, "AXPress")
            elif op == PRESS_KEY and action.key is not None:
                keyboard.post_chord(action.key.chord, delay_s=self.settings.key_delay_s)
                result = ActionResult(True, "keyboard", str(action.key.chord))
            elif op in (SCROLL_UP, SCROLL_DOWN) and action.element is not None:
                result = self._scroll(action.element, down=op == SCROLL_DOWN)
            elif op == OPEN_APP and action.app is not None:
                result = self.open_app(action.app)
            elif op == FOCUS_WINDOW and action.window is not None:
                node = action.window.node
                node.perform("AXRaise")
                self._try_set(node, "AXMain", True)
                result = ActionResult(True, "AXRaise")
            else:
                result = ActionResult(False, "none", f"cannot execute {action.describe()}")
        except AXError as exc:
            result = ActionResult(False, "ax-error", str(exc))
        result.elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        return result

    # ---- operations ---------------------------------------------------------------------------------------------
    def _click(self, element: UIElement) -> ActionResult:
        node = element.node
        actions = element.actions or node.actions()
        for name in PRESS_ACTIONS:
            if name in actions:
                node.perform(name)
                return ActionResult(True, name)
        if element.kind == "row" or node.settable("AXSelected"):
            node.set("AXSelected", True)
            return ActionResult(True, "AXSelected")
        if element.kind == "text_input" or node.settable("AXFocused"):
            node.set("AXFocused", True)
            return ActionResult(True, "AXFocused")
        if "AXShowMenu" in actions:
            node.perform("AXShowMenu")
            return ActionResult(True, "AXShowMenu")
        if self.settings.pointer_fallback and element.frame is not None and not element.frame.empty:
            keyboard.click_at(*element.frame.center)
            return ActionResult(True, "pointer", "no AX press action; clicked the element's AX frame centre")
        return ActionResult(False, "none", "element exposes no press, select, or focus action")

    def _type(self, element: UIElement, text: str, *, secret: bool, prefer_keys: bool = False) -> ActionResult:
        node = element.node
        self._try_set(node, "AXFocused", True)
        mode = self.settings.typing_mode
        if mode == "auto":
            # Browsers and search fields react to real key events (suggestions, Return to submit) but may ignore a
            # directly set AXValue; plain native text views take the exact, instant AXValue write.
            keystrokes = prefer_keys or element.in_web_area or element.secure or element.subrole == "AXSearchField"
            mode = "keys" if keystrokes else "ax"
        if mode == "ax" and element.value_settable and not element.secure:
            try:
                node.set("AXValue", text)
                if str(node.get("AXValue") or "") == text:
                    return ActionResult(True, "AXValue")
            except AXError:
                pass  # fall through to keystrokes
        # Replace the field content: focus, select all, then type Unicode (layout independent).
        keyboard.post_chord(KeyChord.parse("cmd+a"), delay_s=self.settings.key_delay_s)
        keyboard.type_text(text, delay_s=self.settings.key_delay_s)
        if secret or element.secure:
            return ActionResult(True, "keystrokes", "secure field: value not read back")
        time.sleep(0.03)
        current = str(node.get("AXValue") or "")
        if text.strip() and text.strip() not in current:
            return ActionResult(False, "keystrokes", "typed text did not appear in the field")
        return ActionResult(True, "keystrokes")

    def _scroll(self, element: UIElement, *, down: bool) -> ActionResult:
        node = element.node
        bar = node.get("AXVerticalScrollBar")
        if bar is not None and bar.settable("AXValue"):
            current = float(bar.get("AXValue") or 0.0)
            step = self._page_fraction(element)
            target = min(1.0, current + step) if down else max(0.0, current - step)
            if abs(target - current) < 1e-6:
                return ActionResult(False, "scrollbar", "already at the " + ("bottom" if down else "top"))
            bar.set("AXValue", target)
            return ActionResult(True, "scrollbar", f"{current:.2f} → {target:.2f}")
        wanted = "AXScrollDownByPage" if down else "AXScrollUpByPage"
        if wanted in node.actions():
            node.perform(wanted)
            return ActionResult(True, wanted)
        self._try_set(node, "AXFocused", True)
        keyboard.post_chord(KeyChord.parse("pagedown" if down else "pageup"), delay_s=self.settings.key_delay_s)
        return ActionResult(True, "keyboard", "page key")

    def _page_fraction(self, element: UIElement) -> float:
        """One visible page as a fraction of the scrollable range, from the content child's AX size."""
        try:
            visible = element.frame.h if element.frame else 0.0
            for child in element.node.children("AXChildren", 8):
                attrs = child.get_many(("AXRole", "AXSize"))
                if attrs.get("AXRole") in ("AXScrollBar", None) or not attrs.get("AXSize"):
                    continue
                content = float(attrs["AXSize"][1])
                if content > visible > 0:
                    return max(0.02, min(1.0, self.settings.scroll_page_fraction * visible / (content - visible)))
        except (StaleElementError, TypeError, ValueError, IndexError):
            pass
        return 0.15

    def open_app(self, app: AppInfo) -> ActionResult:
        if app.pid is not None:
            self._activate(app.pid)
            method = "activate"
        else:
            args = ["open", "-b", app.bundle_id] if app.bundle_id else ["open", "-a", app.path or app.name]
            completed = subprocess.run(args, capture_output=True, text=True, timeout=15, check=False)
            if completed.returncode != 0:
                return ActionResult(False, "open", completed.stderr.strip() or f"open exited {completed.returncode}")
            method = "launch"
        started = time.monotonic()
        retried = False
        while time.monotonic() - started < self.settings.launch_timeout_s:
            pid = self._frontmost_pid()
            if pid is not None and (pid == app.pid or (app.pid is None and self._matches(pid, app))):
                return ActionResult(True, method)
            if app.pid is not None and not retried and time.monotonic() - started > 0.5:
                self._activate(app.pid)  # activation requests are occasionally dropped while another app animates
                retried = True
            time.sleep(0.05)
        return ActionResult(False, method, f"{app.name} did not become frontmost in time")

    # ---- helpers ------------------------------------------------------------------------------------------------
    def _activate(self, pid: int) -> None:
        try:
            from AppKit import NSApplicationActivateIgnoringOtherApps, NSRunningApplication

            running = NSRunningApplication.runningApplicationWithProcessIdentifier_(pid)
            if running is not None:
                running.activateWithOptions_(NSApplicationActivateIgnoringOtherApps)
        except ImportError:  # pragma: no cover
            pass
        self._try_set(self._AXNode.application(pid), "AXFrontmost", True)

    @staticmethod
    def _matches(pid: int, app: AppInfo) -> bool:
        from ..observer.apps import app_for_pid

        current = app_for_pid(pid)
        return bool(app.bundle_id and current.bundle_id == app.bundle_id) or current.name == app.name

    @staticmethod
    def _try_set(node: Any, attribute: str, value: object) -> None:
        with contextlib.suppress(AXError, StaleElementError):
            node.set(attribute, value)
