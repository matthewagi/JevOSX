"""On macOS CI: prove the pyobjc symbol names used by the observer/executor exist (no Accessibility grant needed)."""

import contextlib
import sys

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="macOS only")


def test_pyobjc_symbols_resolve():
    import ApplicationServices as AS
    import Quartz as Q

    for name in (
        "AXUIElementCreateApplication",
        "AXUIElementCreateSystemWide",
        "AXUIElementCopyAttributeValue",
        "AXUIElementCopyAttributeValues",
        "AXUIElementCopyMultipleAttributeValues",
        "AXUIElementCopyActionNames",
        "AXUIElementIsAttributeSettable",
        "AXUIElementPerformAction",
        "AXUIElementSetAttributeValue",
        "AXUIElementSetMessagingTimeout",
        "AXUIElementGetPid",
        "AXValueGetType",
        "AXValueGetValue",
        "AXIsProcessTrusted",
        "AXIsProcessTrustedWithOptions",
        "AXUIElementRef",
        "AXValueRef",
        "kAXValueCGPointType",
        "kAXValueCGSizeType",
        "kAXValueCGRectType",
        "kAXValueCFRangeType",
        "kAXValueAXErrorType",
        "kAXTrustedCheckOptionPrompt",
    ):
        assert hasattr(AS, name), name
    for name in (
        "CGEventCreateKeyboardEvent",
        "CGEventKeyboardSetUnicodeString",
        "CGEventSetFlags",
        "CGEventPost",
        "CGEventSourceCreate",
        "CGEventCreateMouseEvent",
        "kCGHIDEventTap",
        "kCGEventSourceStateHIDSystemState",
        "CGWindowListCopyWindowInfo",
        "kCGWindowListOptionOnScreenOnly",
        "kCGWindowListExcludeDesktopElements",
        "kCGNullWindowID",
    ):
        assert hasattr(Q, name), name


def test_system_wide_element_and_trust_check_do_not_crash():
    from jevosx.errors import AccessibilityPermissionError
    from jevosx.observer.ax import AX_AVAILABLE, AXNode, is_trusted

    assert AX_AVAILABLE
    assert isinstance(is_trusted(prompt=False), bool)
    # Exercises the real pyobjc call signatures; CI runners usually have no Accessibility grant.
    with contextlib.suppress(AccessibilityPermissionError):
        AXNode.system_wide().get("AXFocusedApplication")


def test_installed_apps_scan_finds_system_apps():
    from jevosx.observer.apps import scan_installed_apps

    apps = scan_installed_apps(["/System/Applications"])
    assert any(a.bundle_id == "com.apple.TextEdit" for a in apps)


def test_frontmost_fallbacks_run_without_errors():
    from jevosx.observer.apps import frontmost_from_window_list, frontmost_from_workspace
    from jevosx.observer.ax import AXNode

    err, _value = AXNode.system_wide().read("AXFocusedApplication")
    assert isinstance(err, int)
    for pid in (frontmost_from_window_list(), frontmost_from_workspace()):
        assert pid is None or pid > 0
