"""macOS desktop observer: frontmost app → focused window AX tree → indexed element table, plus menus, windows,
running and installed apps. Produces an `Observation`; never takes a screenshot and never runs OCR."""

from __future__ import annotations

import contextlib
import os
import time
from typing import Any

from ..config import ObserverSettings
from ..errors import StaleElementError
from ..types import AppInfo, Observation, UIElement, WindowInfo, clean_text
from . import apps as appmod
from .ax import AX_SUCCESS, AXNode, require_ax
from .menus import walk_menu_bar
from .walker import TreeWalker, WalkLimits

# Chromium browsers only build their web accessibility tree when an assistive client asks for it.
CHROMIUM_BUNDLES = frozenset(
    {
        "com.google.Chrome",
        "com.google.Chrome.canary",
        "com.microsoft.edgemac",
        "com.brave.Browser",
        "company.thebrowser.Browser",
        "com.vivaldi.Vivaldi",
        "com.operasoftware.Opera",
        "org.chromium.Chromium",
    }
)


class MacDesktopObserver:
    def __init__(self, settings: ObserverSettings | None = None):
        require_ax()
        self.settings = settings or ObserverSettings()
        s = self.settings
        self.walker = TreeWalker(
            WalkLimits(
                max_nodes=s.max_nodes,
                max_elements=s.max_elements,
                max_depth=s.max_depth,
                max_children=s.max_children,
                time_budget_s=s.time_budget_s,
                max_text_chars=s.max_text_chars,
                probe_generic=s.probe_generic,
            )
        )
        self._system = AXNode.system_wide()
        self._app_nodes: dict[int, AXNode] = {}
        self._web_enabled: set[int] = set()
        self._menu_cache: dict[int, tuple[float, list[UIElement], bool]] = {}
        self._installed: list[AppInfo] | None = None

    # ---- public API ---------------------------------------------------------------------------------------------
    def frontmost_pid(self) -> int | None:
        return self.detect_frontmost()[0]

    def detect_frontmost(self) -> tuple[int | None, str]:
        """(pid, how it was found). Accessibility first; the window list and NSWorkspace are fallbacks because the
        system-wide AXFocusedApplication query can fail (e.g. while the focused app is busy)."""
        err, app = self._system.read("AXFocusedApplication")
        if err == AX_SUCCESS and isinstance(app, AXNode):
            try:
                return app.pid(), "accessibility"
            except StaleElementError:
                pass
        why = f"accessibility error {err}" if err != AX_SUCCESS else "accessibility returned no app"
        pid = appmod.frontmost_from_window_list(exclude=frozenset({os.getpid()}))
        if pid:
            return pid, f"window list ({why})"
        pid = appmod.frontmost_from_workspace()
        if pid:
            return pid, f"NSWorkspace ({why})"
        return None, f"{why}; the window list and NSWorkspace found no app either"

    def app_node(self, pid: int) -> AXNode:
        node = self._app_nodes.get(pid)
        if node is None:
            node = AXNode.application(pid, self.settings.messaging_timeout_s)
            self._app_nodes[pid] = node
        return node

    def installed_apps(self) -> list[AppInfo]:
        if self._installed is None:
            self._installed = (
                appmod.scan_installed_apps(self.settings.app_dirs) if self.settings.include_installed_apps else []
            )
        return self._installed

    def find_app(self, query: str) -> AppInfo | None:
        return appmod.find_app(query, appmod.merge_apps(appmod.running_apps(), self.installed_apps()))

    def quick_signature(self) -> str:
        pid = self.frontmost_pid()
        if pid is None:
            return "none"
        try:
            node = self.app_node(pid)
            window = node.get("AXFocusedWindow")
            focused = node.get("AXFocusedUIElement")
            parts = [str(pid)]
            if window is not None:
                parts.append(clean_text(window.get("AXTitle"), 80))
            if focused is not None:
                attrs = focused.get_many(("AXRole", "AXTitle", "AXValue"))
                parts.extend(clean_text(attrs.get(k), 40) for k in ("AXRole", "AXTitle", "AXValue"))
            return "|".join(parts)
        except StaleElementError:
            return f"{pid}|stale"

    def observe(self, pid: int | None = None) -> Observation:
        """Observe the frontmost app, or the app with `pid` (e.g. for diagnostics while another app is in front)."""
        started = time.perf_counter()
        how = "requested"
        if pid is None:
            pid, how = self.detect_frontmost()
        if pid is None:
            raise StaleElementError(f"no frontmost application ({how})")
        app = appmod.app_for_pid(pid)
        node = self.app_node(pid)
        self.enable_web_accessibility(pid, app, node)

        windows, focused_window = self._windows(node)
        # The window's own frame becomes the visibility clip for its subtree (see TreeWalker).
        roots: list[tuple[Any, str | None]] = []
        if focused_window is not None and focused_window.node is not None:
            roots.append((focused_window.node, None))
        # Open context menus and popovers hang off the application element, not the window.
        try:
            for child in node.children("AXChildren", 50):
                role = child.get("AXRole")
                if role == "AXMenu":
                    roots.append((child, "open menu"))
        except StaleElementError:
            pass
        walk = self.walker.walk(roots)

        menu_items: list[UIElement] = []
        menus_truncated = False
        if self.settings.include_menus:
            menu_items, menus_truncated = self._menus(pid, node)

        running = appmod.running_apps()
        stats = {
            "visited": walk.visited,
            "elements": len(walk.elements),
            "menu_items": len(menu_items),
            "truncated": walk.truncated,  # only the window walk: the agent is told elements may be missing
            "menus_truncated": menus_truncated,
            "walk_ms": walk.elapsed_ms,
            "notes": walk.notes,
            "skipped": walk.skipped,
        }
        obs = Observation(
            app=app,
            window=focused_window,
            windows=windows,
            elements=walk.elements,
            menu_items=menu_items,
            scroll_areas=walk.scroll_areas,
            text=walk.text,
            running_apps=running,
            installed_apps=self.installed_apps(),
            stats=stats,
            captured_at=time.time(),
            page_url=walk.page_url,
        )
        obs.stats["observe_ms"] = round((time.perf_counter() - started) * 1000, 1)
        return obs

    # ---- internals ----------------------------------------------------------------------------------------------
    def _windows(self, node: AXNode) -> tuple[list[WindowInfo], WindowInfo | None]:
        try:
            focused = node.get("AXFocusedWindow") or node.get("AXMainWindow")
            raw = node.children("AXWindows")
        except StaleElementError:
            return [], None
        windows: list[WindowInfo] = []
        focused_info: WindowInfo | None = None
        for position, window in enumerate(raw[:20], start=1):
            try:
                attrs = window.get_many(("AXTitle", "AXMinimized", "AXSubrole"))
            except StaleElementError:
                continue
            is_focused = focused is not None and window == focused
            info = WindowInfo(
                index=position,
                title=clean_text(attrs.get("AXTitle"), 100) or "(untitled)",
                focused=is_focused,
                minimized=attrs.get("AXMinimized") is True,
                node=window,
            )
            windows.append(info)
            if is_focused:
                focused_info = info
        if focused_info is None and focused is not None:
            focused_info = WindowInfo(
                index=0, title=clean_text(focused.get("AXTitle"), 100), focused=True, node=focused
            )
        if focused_info is None and windows:
            focused_info = windows[0]
        return windows, focused_info

    def _menus(self, pid: int, node: AXNode) -> tuple[list[UIElement], bool]:
        cached = self._menu_cache.get(pid)
        now = time.monotonic()
        if cached and now - cached[0] < self.settings.menu_cache_ttl_s:
            return cached[1], cached[2]
        try:
            menu_bar = node.get("AXMenuBar")
        except StaleElementError:
            menu_bar = None
        if menu_bar is None:
            return [], False
        items, truncated = walk_menu_bar(menu_bar, max_items=self.settings.max_menu_items)
        self._menu_cache[pid] = (now, items, truncated)
        return items, truncated

    def enable_web_accessibility(self, pid: int, app: AppInfo, node: AXNode) -> None:
        if not self.settings.enable_web_accessibility or pid in self._web_enabled:
            return
        self._web_enabled.add(pid)
        for attribute, wanted in (
            ("AXManualAccessibility", True),  # Electron apps (Slack, VS Code, Discord, Notion…)
            ("AXEnhancedUserInterface", app.bundle_id in CHROMIUM_BUNDLES),
        ):
            if not wanted:
                continue
            with contextlib.suppress(Exception):  # unsupported on most native apps; harmless
                node.set(attribute, True)
