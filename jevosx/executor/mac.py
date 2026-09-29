"""Deterministic macOS execution engine.

Every action resolves to an AX handle captured during observation: AXPress/AXConfirm for clicks, AXValue writes
(or layout-independent Unicode keystrokes) for text, AXPress on menu items, AXRaise for windows, scroll-bar value
changes for scrolling, NSWorkspace/`open` for apps. Pointer clicks are an opt-in last resort.

AX actions work on windows that are not in front, so the agent can work behind the person's own window. Key
presses, menu commands and pointer clicks go to whatever is in front, so for those the observed window is brought
forward first and checked to be the key window; if it cannot be, nothing is sent.
"""

from __future__ import annotations

import contextlib
import subprocess
import time
from collections.abc import Callable
from typing import Any

from ..config import ExecutorSettings
from ..errors import AXError, StaleElementError
from ..observer.ax import AX_ATTRIBUTE_UNSUPPORTED
from ..observer.base import WindowRef
from ..observer.walker import url_text
from ..sites import host_matches, page_host
from ..types import (
    BROWSER_BUNDLES,
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


class FocusLost(Exception):
    """The observed window could not be made the key window, so keys or pointer clicks would land elsewhere."""


class MacExecutor:
    def __init__(self, settings: ExecutorSettings | None = None, *, frontmost_pid: Callable[[], int | None]):
        from ..observer.ax import AXNode, require_ax

        require_ax()
        self.settings = settings or ExecutorSettings()
        self._frontmost_pid = frontmost_pid
        self._AXNode = AXNode

    # ---- freshness ----------------------------------------------------------------------------------------------
    def validate(self, action: Action, obs: Observation) -> None:
        """The target must still be what was observed. Which app is in front does not matter here: AX actions reach
        background windows, and keyboard actions bring the observed window forward first (see _focus)."""
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
            if self._needs_keyboard(action):
                self._focus(obs)
            if op == CLICK and action.element is not None and action.element.kind == "visual":
                result = self._click_visual(action.element)
            elif op == CLICK and action.element is not None:
                result = self._click(action.element, obs)
            elif op == TYPE_TEXT and action.element is not None and action.element.kind == "keyboard" and action.text:
                if action.text_is_secret or action.secure_only:
                    result = ActionResult(False, "refused", "secret text is only typed into password fields")
                else:  # no field to select first: Cmd-A in a canvas app would select every object
                    keyboard.type_text(action.text or "", delay_s=self.settings.key_delay_s)
                    result = ActionResult(True, "keystrokes", "typed at the cursor")
            elif op == TYPE_TEXT and action.element is not None and action.text is not None:
                refusal = self._check_credential(action)
                if refusal is not None:
                    result = refusal
                else:
                    keys = obs.app.bundle_id in BROWSER_BUNDLES
                    result = self._type(
                        action.element, action.text, obs, secret=action.text_is_secret, prefer_keys=keys
                    )
                    if result.ok and action.submit:  # the window is still in front from typing
                        keyboard.post_chord(KeyChord.parse("return"), delay_s=self.settings.key_delay_s)
                        result = ActionResult(True, result.method, "typed and pressed Return")
            elif op == MENU and action.element is not None:
                action.element.node.perform("AXPress")
                result = ActionResult(True, "AXPress")
            elif op == PRESS_KEY and action.key is not None:
                keyboard.post_chord(action.key.chord, delay_s=self.settings.key_delay_s)
                result = ActionResult(True, "keyboard", str(action.key.chord))
            elif op in (SCROLL_UP, SCROLL_DOWN) and action.element is not None:
                result = self._scroll(action.element, obs, down=op == SCROLL_DOWN)
            elif op == OPEN_APP and action.app is not None:
                result = self.open_app(action.app)
            elif op == FOCUS_WINDOW and action.window is not None:
                node = action.window.node
                node.perform("AXRaise")
                self._try_set(node, "AXMain", True)
                result = ActionResult(True, "AXRaise")
            else:
                result = ActionResult(False, "none", f"cannot execute {action.describe()}")
        except FocusLost as exc:
            result = ActionResult(False, "focus", str(exc))
        except AXError as exc:
            # Seen live: Notes answered AXPress on "New Note" with -25205 and created the note all the same.
            pressed = op in (CLICK, MENU) and exc.operation.startswith("perform")
            result = ActionResult(
                False, "ax-error", str(exc), unconfirmed=pressed and exc.code == AX_ATTRIBUTE_UNSUPPORTED
            )
        result.elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        return result

    # ---- focus --------------------------------------------------------------------------------------------------
    @staticmethod
    def _needs_keyboard(action: Action) -> bool:
        """Actions that go to whatever is in front: key presses, menu commands (they act on the key window) and
        clicks on OCR text. Typing decides for itself (an AXValue write needs no focus, keystrokes do)."""
        element = action.element
        if action.operation in (PRESS_KEY, MENU):
            return True
        if action.operation == CLICK and element is not None and element.kind == "visual":
            return True
        return action.operation == TYPE_TEXT and element is not None and element.kind == "keyboard"

    def _focus(self, obs: Observation) -> None:
        if obs.app.pid is None:
            return
        window = obs.window
        target = WindowRef(obs.app.pid, window.node if window else None, window.title if window else "")
        if not self.bring_forward(target):
            where = f"“{target.title}”" if target.title else obs.app.name
            raise FocusLost(f"could not bring {where} to the front, so nothing was sent to it")

    def bring_forward(self, target: WindowRef) -> bool:
        """Make `target` the key window of the frontmost app (activate the app, raise the window). True once it is."""
        if self._in_front(target):
            return True
        started = time.monotonic()
        self._raise(target)
        retried = False
        while not self._in_front(target):
            elapsed = time.monotonic() - started
            if elapsed > self.settings.focus_timeout_s:
                return False
            if not retried and elapsed > self.settings.focus_timeout_s / 2:
                self._raise(target)  # activation requests are occasionally dropped while another app animates
                retried = True
            time.sleep(0.03)
        return True

    def _raise(self, target: WindowRef) -> None:
        if self._frontmost_pid() != target.pid:
            self._activate(target.pid)
        if target.window is not None:
            with contextlib.suppress(AXError, StaleElementError):
                target.window.perform("AXRaise")
            self._try_set(target.window, "AXMain", True)
            self._try_set(target.window, "AXFocused", True)

    def _in_front(self, target: WindowRef) -> bool:
        if self._frontmost_pid() != target.pid:
            return False
        if target.window is None:
            return True
        try:
            focused = self._AXNode.application(target.pid).get("AXFocusedWindow")
        except StaleElementError:
            return False
        return focused is not None and bool(focused == target.window)

    # ---- operations ---------------------------------------------------------------------------------------------
    def _click_visual(self, element: UIElement) -> ActionResult:
        """On-screen text read by OCR: click the centre of the recognized text (computed locally, never by a model)."""
        if element.frame is None or element.frame.empty:
            return ActionResult(False, "none", "the recognized text has no position")
        keyboard.click_at(*element.frame.center)
        return ActionResult(True, "pointer", "clicked the centre of the recognized text")

    def _click(self, element: UIElement, obs: Observation) -> ActionResult:
        node = element.node
        if element.kind == "image":  # a web picture lists only AXShowMenu, yet Chrome clicks it on AXPress
            node.perform("AXPress")
            return ActionResult(True, "AXPress")
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
            self._focus(obs)  # a pointer click lands on whatever is on top at that point
            keyboard.click_at(*element.frame.center)
            return ActionResult(True, "pointer", "no AX press action; clicked the element's AX frame centre")
        return ActionResult(False, "none", "element exposes no press, select, or focus action")

    def _type(
        self, element: UIElement, text: str, obs: Observation, *, secret: bool, prefer_keys: bool = False
    ) -> ActionResult:
        node = element.node
        mode = self.settings.typing_mode
        if mode == "auto":
            # Browsers and search fields react to real key events (suggestions, Return to submit) but may ignore a
            # directly set AXValue; plain native text views take the exact, instant AXValue write.
            keystrokes = prefer_keys or element.in_web_area or element.secure or element.subrole == "AXSearchField"
            mode = "keys" if keystrokes else "ax"
        if mode == "ax" and element.value_settable and not element.secure:
            self._try_set(node, "AXFocused", True)
            try:
                node.set("AXValue", text)
                if str(node.get("AXValue") or "") == text:
                    return ActionResult(True, "AXValue")
            except AXError:
                pass  # fall through to keystrokes
        # Replace the field content: bring the window forward, focus the field, select all, then type Unicode (layout
        # independent). The window comes first: raising a window can move its keyboard focus.
        self._focus(obs)
        self._try_set(node, "AXFocused", True)
        keyboard.post_chord(KeyChord.parse("cmd+a"), delay_s=self.settings.key_delay_s)
        keyboard.type_text(text, delay_s=self.settings.key_delay_s)
        if secret or element.secure:
            return ActionResult(True, "keystrokes", "secure field: value not read back")
        if text.strip() and not self._text_appears(node, text.strip()):
            return ActionResult(False, "keystrokes", "typed text did not appear in the field")
        return ActionResult(True, "keystrokes")

    def _text_appears(self, node: Any, text: str) -> bool:
        """Wait for posted keystrokes to show up in the field's AXValue. Key events are delivered asynchronously: seen
        live, a fresh Chrome window's address bar still read "" just after the last key and held the text ~30 ms later,
        so a single early read-back reported a failure and the agent lost a step typing it again."""
        started = time.monotonic()
        while True:
            time.sleep(self.settings.settle_poll_s)
            if text in str(node.get("AXValue") or ""):
                return True
            if time.monotonic() - started >= self.settings.settle_timeout_s:
                return False

    def _scroll(self, element: UIElement, obs: Observation, *, down: bool) -> ActionResult:
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
        self._focus(obs)
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
        # The launch or activation was asked for: a slow app (seen live: Notes) still comes forward on its own.
        return ActionResult(False, method, f"{app.name} did not become frontmost in time", unconfirmed=True)

    def _check_credential(self, action: Action) -> ActionResult | None:
        """Last check before a saved login is typed: the field's own page must still be the saved site."""
        element = action.element
        assert element is not None
        if action.secure_only and not element.secure:
            return ActionResult(False, "refused", "a saved password only goes into a password field")
        if action.require_host is None:
            return None
        host = page_host(self._page_url_of(element))
        if host is None or not host_matches(action.require_host, host):
            now = host or "a page that is not https"
            return ActionResult(False, "refused", f"the field is on {now}, not {action.require_host}; nothing typed")
        return None

    @staticmethod
    def _page_url_of(element: UIElement) -> str | None:
        """AXURL of the web document that contains the element (the nearest AXWebArea above it)."""
        node = element.node
        for _ in range(80):
            try:
                node = node.get("AXParent")
                if node is None:
                    return None
                if node.get("AXRole") == "AXWebArea":
                    return url_text(node.get("AXURL"))
            except (AXError, StaleElementError):
                return None
        return None

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
