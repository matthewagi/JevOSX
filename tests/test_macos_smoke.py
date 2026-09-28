"""On macOS CI: prove the pyobjc symbol names used by the observer/executor exist (no Accessibility grant needed)."""

import contextlib
import json
import sys
from pathlib import Path

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


def test_apple_writer_helper_compiles_and_answers_check(tmp_path):
    """Compiles the real Swift helper. With an SDK that has FoundationModels (Xcode/CLT 26+) this type-checks every
    Foundation Models call; the CI runner has no Apple Intelligence, so --check must report a reason, not crash."""
    import shutil
    import subprocess

    from jevosx.writer.apple import AppleWriter, build_helper

    if shutil.which("xcrun") is None or subprocess.run(["xcode-select", "-p"], capture_output=True).returncode:
        pytest.skip("no Command Line Tools")
    sdk = subprocess.run(["xcrun", "--sdk", "macosx", "--show-sdk-path"], capture_output=True, text=True).stdout
    has_framework = (Path(sdk.strip()) / "System/Library/Frameworks/FoundationModels.framework").exists()
    helper = build_helper(tmp_path)
    check = subprocess.run([str(helper), "--check"], capture_output=True, text=True, timeout=60)
    answer = json.loads(check.stdout.strip().splitlines()[-1])
    assert answer["framework"] is has_framework and isinstance(answer["available"], bool)
    assert answer["reason"] in {"available", "appleIntelligenceNotEnabled", "deviceNotEligible", "modelNotReady",
                                "osTooOld", "sdkMissing", "unknown"}  # fmt: skip
    status = AppleWriter(helper).status()
    assert status.available is answer["available"]
    bad = subprocess.run([str(helper)], input="not json", capture_output=True, text=True, timeout=60)
    assert json.loads(bad.stdout)["error"] == "badRequest" and bad.returncode == 2
