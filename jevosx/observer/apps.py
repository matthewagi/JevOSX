"""Whole-Mac application context: running apps (NSWorkspace) and installed app bundles (Info.plist scan)."""

from __future__ import annotations

import plistlib
from collections.abc import Iterable
from pathlib import Path

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
