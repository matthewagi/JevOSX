"""Vision fallback for apps that draw their own interface (games, canvas design tools, remote desktops).

Some apps paint pixels instead of exposing Accessibility controls, so the element table comes back empty. For those
windows JevOSX captures that one window (`screencapture -l`, which needs the Screen Recording permission) and reads
its text with Apple's on-device Vision OCR. Every recognized line becomes an element Jev can choose
(`[12] on-screen text "Play"`). The click point is the centre of the recognized text, computed here from the
window's frame. Jev still only picks an id; it never sees or outputs a coordinate.

Platform calls are injected, so the selection and geometry logic below is tested on any OS.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
import tempfile
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..types import CLICK, TYPE_TEXT, Rect, UIElement, clean_text, normalize_key

VISION_ROLE = "AXVisionText"
KEYBOARD_ROLE = "AXKeyboard"
VISION_CONTAINER = "screen text (OCR)"
_ALNUM = re.compile(r"\w", re.UNICODE)


@dataclass(frozen=True, slots=True)
class TextBox:
    """One recognized line. Coordinates are fractions of the image, origin top-left."""

    text: str
    confidence: float
    x: float
    y: float
    w: float
    h: float


@dataclass
class VisionResult:
    ok: bool
    boxes: list[TextBox] = field(default_factory=list)
    bounds: Rect | None = None
    note: str = ""
    ms: float = 0.0
    cached: bool = False

    def stats(self) -> dict[str, Any]:
        out: dict[str, Any] = {"ran": self.ok, "texts": len(self.boxes), "ms": self.ms}
        if self.cached:
            out["cached"] = True
        if self.note:
            out["note"] = self.note
        return out


def needs_vision(mode: str, elements: Sequence[UIElement], ax_text: str, *, min_controls: int = 4) -> bool:
    """`auto`: the window gives Accessibility almost nothing: fewer than `min_controls` controls in its content
    (the web page, in a browser) and hardly any text. A plain alert has text, so it never triggers this."""
    if mode == "always":
        return True
    if mode != "auto":
        return False
    has_web = any(e.in_web_area for e in elements)
    controls = [e for e in elements if e.ops and (e.in_web_area or not has_web)]
    return len(controls) < min_controls and len(ax_text.strip()) < 40


def to_screen(box: TextBox, bounds: Rect) -> Rect:
    return Rect(bounds.x + box.x * bounds.w, bounds.y + box.y * bounds.h, box.w * bounds.w, box.h * bounds.h)


def reading_order(boxes: Iterable[TextBox]) -> list[TextBox]:
    """Top-to-bottom lines, left to right within a line (boxes whose centres are within half a line height)."""
    ordered = sorted(boxes, key=lambda b: (b.y + b.h / 2, b.x))
    lines: list[list[TextBox]] = []
    for box in ordered:
        centre = box.y + box.h / 2
        if lines and abs(centre - (lines[-1][0].y + lines[-1][0].h / 2)) < max(box.h, lines[-1][0].h) / 2:
            lines[-1].append(box)
        else:
            lines.append([box])
    return [box for line in lines for box in sorted(line, key=lambda b: b.x)]


def visual_elements(
    boxes: Iterable[TextBox],
    bounds: Rect,
    *,
    existing: Sequence[UIElement] = (),
    ax_text: str = "",
    start_index: int = 1,
    max_items: int = 60,
    min_confidence: float = 0.35,
) -> tuple[list[UIElement], list[str]]:
    """OCR lines → clickable elements (and all recognized lines as visible text).

    Lines that merely repeat an Accessibility label at the same place, or text Accessibility already reported,
    are not offered again."""
    known_text = {normalize_key(line) for line in ax_text.splitlines() if line.strip()}
    elements: list[UIElement] = []
    lines: list[str] = []
    seen: set[tuple[str, int, int]] = set()
    for box in reading_order(boxes):
        text = clean_text(box.text, 120)
        if box.confidence < min_confidence or not _ALNUM.search(text):
            continue
        frame = to_screen(box, bounds)
        cx, cy = frame.center
        key = normalize_key(text)
        spot = (key, round(cx / 12), round(cy / 12))
        if spot in seen:
            continue  # the same text recognized twice in one place
        seen.add(spot)
        lines.append(text)
        if key in known_text or _covered(key, cx, cy, existing):
            continue
        if len(elements) < max_items:
            elements.append(
                UIElement(
                    index=start_index + len(elements),
                    role=VISION_ROLE,
                    subrole=None,
                    label=text,
                    kind="visual",
                    ops=(CLICK,),
                    container=VISION_CONTAINER,
                    frame=frame,
                )
            )
    return elements, lines


def keyboard_element(index: int) -> UIElement:
    """Types at the current cursor. Only offered next to OCR elements: there is no field to target in those apps."""
    return UIElement(
        index=index,
        role=KEYBOARD_ROLE,
        subrole=None,
        label="type at the cursor",
        kind="keyboard",
        ops=(TYPE_TEXT,),
        container="keyboard",
    )


def _covered(key: str, x: float, y: float, existing: Sequence[UIElement]) -> bool:
    for element in existing:
        frame = element.frame
        if frame is None or normalize_key(element.label) != key:
            continue
        if frame.x <= x <= frame.x + frame.w and frame.y <= y <= frame.y + frame.h:
            return True
    return False


def pick_window(windows: Iterable[Any], pid: int, title: str | None) -> tuple[int, Rect] | None:
    """(window id, frame) of the app's frontmost normal window, preferring the one with the focused title."""
    candidates = []
    for window in windows:
        bounds = window.get("kCGWindowBounds") or {}
        if (
            window.get("kCGWindowOwnerPID") != pid
            or window.get("kCGWindowLayer", 0) != 0
            or window.get("kCGWindowAlpha", 1) == 0
            or bounds.get("Width", 0) < 50
            or bounds.get("Height", 0) < 50
        ):
            continue
        candidates.append(window)
    titled = [w for w in candidates if title and w.get("kCGWindowName") == title]
    chosen = (titled or candidates or [None])[0]
    if chosen is None:
        return None
    b = chosen["kCGWindowBounds"]
    return int(chosen["kCGWindowNumber"]), Rect(float(b["X"]), float(b["Y"]), float(b["Width"]), float(b["Height"]))


# ---- macOS ---------------------------------------------------------------------------------------------------------
def screen_recording_allowed(prompt: bool = False) -> bool:  # pragma: no cover - macOS only
    try:
        import Quartz
    except ImportError:
        return False
    if prompt:
        return bool(Quartz.CGRequestScreenCaptureAccess())
    return bool(Quartz.CGPreflightScreenCaptureAccess())


def find_window(pid: int, title: str | None) -> tuple[int, Rect] | None:  # pragma: no cover - macOS only
    import Quartz

    options = Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements
    return pick_window(Quartz.CGWindowListCopyWindowInfo(options, Quartz.kCGNullWindowID) or [], pid, title)


def capture_window(window_id: int, path: Path) -> None:  # pragma: no cover - macOS only
    """One window, no shadow (-o), no sound (-x): the image matches the window's frame exactly."""
    subprocess.run(
        ["screencapture", "-x", "-o", "-l", str(window_id), "-t", "png", str(path)],
        check=True,
        capture_output=True,
        timeout=10,
    )


def recognize_text(path: Path) -> list[TextBox]:  # pragma: no cover - macOS only
    """Apple Vision OCR (accurate, on-device). Boxes are converted to a top-left origin."""
    import objc
    import Vision
    from Foundation import NSURL

    with objc.autorelease_pool():
        handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(NSURL.fileURLWithPath_(str(path)), {})
        request = Vision.VNRecognizeTextRequest.alloc().init()
        request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        request.setUsesLanguageCorrection_(True)
        if hasattr(request, "setAutomaticallyDetectsLanguage_"):
            request.setAutomaticallyDetectsLanguage_(True)
        ok, _error = handler.performRequests_error_([request], None)
        boxes: list[TextBox] = []
        for observation in (request.results() or []) if ok else []:
            candidates = observation.topCandidates_(1)
            if not candidates:
                continue
            best = candidates[0]
            box = observation.boundingBox()
            x, y, w, h = box.origin.x, box.origin.y, box.size.width, box.size.height
            boxes.append(TextBox(str(best.string()), float(best.confidence()), x, 1.0 - y - h, w, h))
        return boxes


class VisionReader:
    """Captures the focused window and OCRs it. An unchanged image reuses the previous result."""

    def __init__(
        self,
        *,
        allowed: Callable[[], bool] = screen_recording_allowed,
        locate: Callable[[int, str | None], tuple[int, Rect] | None] = find_window,
        capture: Callable[[int, Path], None] = capture_window,
        ocr: Callable[[Path], list[TextBox]] = recognize_text,
        clock: Callable[[], float] = time.perf_counter,
    ):
        self._allowed = allowed
        self._locate = locate
        self._capture = capture
        self._ocr = ocr
        self._clock = clock
        self._cache: tuple[str, list[TextBox]] | None = None

    def read(self, pid: int, title: str | None) -> VisionResult:
        started = self._clock()

        def done(result: VisionResult) -> VisionResult:
            result.ms = round((self._clock() - started) * 1000, 1)
            return result

        try:
            if not self._allowed():
                return done(VisionResult(False, note="Screen Recording permission is off (jevosx doctor)"))
            window = self._locate(pid, title)
            if window is None:
                return done(VisionResult(False, note="no on-screen window to capture"))
            window_id, bounds = window
            with tempfile.TemporaryDirectory(prefix="jevosx-vision-") as folder:
                image = Path(folder) / "window.png"
                self._capture(window_id, image)  # deleted with the folder: screenshots are never kept
                digest = hashlib.sha1(image.read_bytes()).hexdigest()
                if self._cache is not None and self._cache[0] == digest:
                    return done(VisionResult(True, list(self._cache[1]), bounds, cached=True))
                boxes = self._ocr(image)
            self._cache = (digest, boxes)
            return done(VisionResult(True, boxes, bounds))
        except (OSError, subprocess.SubprocessError, ValueError, KeyError) as exc:
            self._cache = None
            return done(VisionResult(False, note=f"vision failed: {type(exc).__name__}: {exc}"[:200]))
