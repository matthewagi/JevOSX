"""Whole-Mac application context: running apps (NSWorkspace) and installed app bundles (Info.plist scan)."""

from __future__ import annotations

import plistlib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ..types import AppInfo

try:  # pragma: no cover - macOS only
    from AppKit import NSApplicationActivationPolicyRegular, NSRunningApplication, NSWorkspace
    from Foundation import NSDate, NSRunLoop

    APPKIT_AVAILABLE = True
except ImportError:  # pragma: no cover
    APPKIT_AVAILABLE = False


def pump_run_loop(seconds: float = 0.005) -> None:  # pragma: no cover - macOS only
    """NSWorkspace state is refreshed by run-loop notifications; a CLI process must spin the loop to see changes."""
    if APPKIT_AVAILABLE:
        NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(seconds))


def running_apps() -> list[AppInfo]:  # pragma: no cover - macOS only
    if not APPKIT_AVAILABLE:
        return []
    pump_run_loop()
    apps = []
    for app in NSWorkspace.sharedWorkspace().runningApplications():
        if app.activationPolicy() != NSApplicationActivationPolicyRegular or app.isTerminated():
            continue
        url = app.bundleURL()
        apps.append(
            AppInfo(
                name=str(app.localizedName() or ""),
                bundle_id=str(app.bundleIdentifier()) if app.bundleIdentifier() else None,
                pid=int(app.processIdentifier()),
                path=str(url.path()) if url is not None else None,
            )
        )
    return apps


def app_for_pid(pid: int) -> AppInfo:  # pragma: no cover - macOS only
    if APPKIT_AVAILABLE:
        app = NSRunningApplication.runningApplicationWithProcessIdentifier_(pid)
        if app is not None:
            url = app.bundleURL()
            return AppInfo(
                name=str(app.localizedName() or f"pid {pid}"),
                bundle_id=str(app.bundleIdentifier()) if app.bundleIdentifier() else None,
                pid=pid,
                path=str(url.path()) if url is not None else None,
            )
    return AppInfo(name=f"pid {pid}", pid=pid)


def _normal_window_pid(window: Any, exclude: set[int] | frozenset[int]) -> int | None:
    """Owner pid of a normal app window (layer 0, visible, not tiny), else None."""
    pid = window.get("kCGWindowOwnerPID")
    bounds = window.get("kCGWindowBounds") or {}
    if (
        window.get("kCGWindowLayer", 0) != 0
        or not pid
        or int(pid) in exclude
        or window.get("kCGWindowAlpha", 1) == 0
        or window.get("kCGWindowOwnerName") in ("Window Server", "Dock")
        or (bounds and (bounds.get("Width", 0) < 50 or bounds.get("Height", 0) < 50))
    ):
        return None
    return int(pid)


def pick_frontmost(windows: Iterable[Any], exclude: set[int] | frozenset[int] = frozenset()) -> int | None:
    """Owner pid of the frontmost normal window in a CGWindowList (front-to-back order). Pure, so it is testable."""
    for window in windows:
        pid = _normal_window_pid(window, exclude)
        if pid is not None:
            return pid
    return None


def choose_frontmost(
    windows: list[Any], active: int | None, exclude: set[int] | frozenset[int] = frozenset()
) -> int | None:
    """Frontmost app when Accessibility cannot say. The window list is trusted first, but it cannot see an active app
    that has no window (TextEdit after its last document closed): if NSWorkspace names such an app, it is in front."""
    listed = pick_frontmost(windows, exclude)
    if active is not None and active != listed and active not in exclude:
        windowed = {pid for pid in (_normal_window_pid(w, exclude) for w in windows) if pid is not None}
        if active not in windowed:
            return active
    return listed


def onscreen_windows() -> list[Any]:  # pragma: no cover - macOS only
    try:
        import Quartz
    except ImportError:
        return []
    options = Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements
    return list(Quartz.CGWindowListCopyWindowInfo(options, Quartz.kCGNullWindowID) or [])


def frontmost_from_window_list(exclude: set[int] | frozenset[int] = frozenset()) -> int | None:  # pragma: no cover
    """Fallback that needs neither Accessibility nor Screen Recording (owner pids are always visible)."""
    return pick_frontmost(onscreen_windows(), exclude)


def frontmost_from_workspace() -> int | None:  # pragma: no cover - macOS only
    if not APPKIT_AVAILABLE:
        return None
    pump_run_loop()
    app = NSWorkspace.sharedWorkspace().frontmostApplication()
    return int(app.processIdentifier()) if app is not None else None


def read_bundle(path: Path) -> AppInfo | None:
    """Parse an .app bundle's Info.plist. Background-only and agent (menu-extra) apps are skipped."""
    info = path / "Contents" / "Info.plist"
    try:
        with info.open("rb") as handle:
            plist = plistlib.load(handle)
    except (OSError, plistlib.InvalidFileException, ValueError):
        return None
    if plist.get("LSBackgroundOnly") in (True, "1", 1) or plist.get("LSUIElement") in (True, "1", 1):
        return None
    name = plist.get("CFBundleDisplayName") or plist.get("CFBundleName") or path.stem
    bundle_id = plist.get("CFBundleIdentifier")
    return AppInfo(name=str(name), bundle_id=str(bundle_id) if bundle_id else None, path=str(path))


def scan_installed_apps(directories: Iterable[str]) -> list[AppInfo]:
    seen: set[str] = set()
    apps: list[AppInfo] = []
    for directory in directories:
        root = Path(directory).expanduser()
        if not root.is_dir():
            continue
        for bundle in sorted(root.glob("*.app")):
            app = read_bundle(bundle)
            if app is None or app.key in seen:
                continue
            seen.add(app.key)
            apps.append(app)
    return apps


def merge_apps(running: list[AppInfo], installed: list[AppInfo]) -> list[AppInfo]:
    """Running apps first (they carry a pid), then installed apps that are not running."""
    keys = {a.key for a in running}
    return [*running, *(a for a in installed if a.key not in keys)]


def find_app(query: str, apps: Iterable[AppInfo]) -> AppInfo | None:
    """Resolve "Safari", "safari", "com.apple.Safari" or "Visual Studio" to one app. Running apps win ties."""
    needle = query.strip().lower()
    if not needle:
        return None
    candidates = list(apps)
    ranked = sorted(candidates, key=lambda a: not a.running)
    for match in (
        lambda a: (a.bundle_id or "").lower() == needle or a.name.lower() == needle,
        lambda a: a.name.lower().startswith(needle),
        lambda a: needle in a.name.lower() or needle in (a.bundle_id or "").lower(),
    ):
        hit = next((a for a in ranked if match(a)), None)
        if hit is not None:
            return hit
    return None
