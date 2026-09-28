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
        "CGPreflightScreenCaptureAccess",
        "CGRequestScreenCaptureAccess",
    ):
        assert hasattr(Q, name), name
    import Vision

    for name in (
        "VNImageRequestHandler",
        "VNRecognizeTextRequest",
        "VNRequestTextRecognitionLevelAccurate",
    ):
        assert hasattr(Vision, name), name


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


def test_vision_ocr_reads_rendered_text(tmp_path):
    """Real Apple Vision OCR on an image drawn with AppKit: text, order and top-left normalized boxes."""
    import AppKit

    from jevosx.observer.vision import recognize_text

    width, height = 640, 220
    bitmap = AppKit.NSBitmapImageRep.alloc().initWithBitmapDataPlanes_pixelsWide_pixelsHigh_bitsPerSample_samplesPerPixel_hasAlpha_isPlanar_colorSpaceName_bytesPerRow_bitsPerPixel_(  # noqa: E501
        None, width, height, 8, 4, True, False, AppKit.NSDeviceRGBColorSpace, 0, 0
    )
    AppKit.NSGraphicsContext.saveGraphicsState()
    AppKit.NSGraphicsContext.setCurrentContext_(AppKit.NSGraphicsContext.graphicsContextWithBitmapImageRep_(bitmap))
    AppKit.NSColor.whiteColor().set()
    AppKit.NSRectFill(AppKit.NSMakeRect(0, 0, width, height))
    attributes = {
        AppKit.NSFontAttributeName: AppKit.NSFont.systemFontOfSize_(52),
        AppKit.NSForegroundColorAttributeName: AppKit.NSColor.blackColor(),
    }
    for text, y in (("New Game", 130), ("Options", 30)):  # AppKit's origin is bottom-left: "New Game" is on top
        AppKit.NSAttributedString.alloc().initWithString_attributes_(text, attributes).drawAtPoint_(
            AppKit.NSMakePoint(40, y)
        )
    AppKit.NSGraphicsContext.restoreGraphicsState()
    png_type = getattr(AppKit, "NSBitmapImageFileTypePNG", None) or AppKit.NSPNGFileType
    path = tmp_path / "menu.png"
    assert bitmap.representationUsingType_properties_(png_type, {}).writeToFile_atomically_(str(path), True)

    boxes = recognize_text(path)
    new_game = next(b for b in boxes if "new game" in b.text.lower())
    options = next(b for b in boxes if "options" in b.text.lower())
    assert new_game.y < options.y and new_game.confidence > 0.3
    assert all(0 <= v <= 1 for b in boxes for v in (b.x, b.y, b.w, b.h))
