"""Synthetic keyboard (and opt-in pointer) input through CoreGraphics CGEvents."""

from __future__ import annotations

import time
from typing import Any

from ..errors import PlatformError
from .keys import KeyChord

try:  # pragma: no cover - macOS only
    import Quartz as _Q

    QUARTZ_AVAILABLE = True
except ImportError:  # pragma: no cover
    _Q = None
    QUARTZ_AVAILABLE = False

MAX_UNICODE_CHUNK = 20  # CGEventKeyboardSetUnicodeString accepts at most 20 UTF-16 units per event


def _require() -> None:
    if not QUARTZ_AVAILABLE:
        raise PlatformError("pyobjc-framework-Quartz is required for keyboard input")


def _source() -> Any:  # pragma: no cover
    return _Q.CGEventSourceCreate(_Q.kCGEventSourceStateHIDSystemState)


def post_chord(chord: KeyChord, *, delay_s: float = 0.0) -> None:  # pragma: no cover - macOS only
    """Press and release one key with explicit modifier flags (physical modifier state is ignored)."""
    _require()
    source = _source()
    for is_down in (True, False):
        event = _Q.CGEventCreateKeyboardEvent(source, chord.keycode, is_down)
        _Q.CGEventSetFlags(event, chord.flags)
        _Q.CGEventPost(_Q.kCGHIDEventTap, event)
        if delay_s:
            time.sleep(delay_s)


def utf16_chunks(text: str, size: int = MAX_UNICODE_CHUNK) -> list[str]:
    """Split text into chunks of at most `size` UTF-16 code units without splitting surrogate pairs."""
    chunks: list[str] = []
    current, units = "", 0
    for char in text:
        width = 2 if ord(char) > 0xFFFF else 1
        if units + width > size and current:
            chunks.append(current)
            current, units = "", 0
        current += char
        units += width
    if current:
        chunks.append(current)
    return chunks


def type_text(text: str, *, delay_s: float = 0.006) -> None:  # pragma: no cover - macOS only
    """Type arbitrary Unicode independent of keyboard layout. Newlines become Return key presses."""
    _require()
    source = _source()
    return_key = KeyChord.parse("return")
    for line_number, line in enumerate(text.split("\n")):
        if line_number:
            post_chord(return_key, delay_s=delay_s)
        for chunk in utf16_chunks(line):
            units = len(chunk.encode("utf-16-le")) // 2
            for is_down in (True, False):
                event = _Q.CGEventCreateKeyboardEvent(source, 0, is_down)
                _Q.CGEventSetFlags(event, 0)
                _Q.CGEventKeyboardSetUnicodeString(event, units, chunk)
                _Q.CGEventPost(_Q.kCGHIDEventTap, event)
            if delay_s:
                time.sleep(delay_s)


def click_at(x: float, y: float) -> None:  # pragma: no cover - macOS only
    """Opt-in pointer fallback. Coordinates come from the element's own AX frame, never from the model."""
    _require()
    point = _Q.CGPointMake(x, y)
    for kind in (_Q.kCGEventMouseMoved, _Q.kCGEventLeftMouseDown, _Q.kCGEventLeftMouseUp):
        event = _Q.CGEventCreateMouseEvent(None, kind, point, _Q.kCGMouseButtonLeft)
        _Q.CGEventPost(_Q.kCGHIDEventTap, event)
        time.sleep(0.01)
