"""Thin, typed wrapper over the macOS Accessibility C API (AXUIElement) via pyobjc.

Everything above this module talks to `AXNode`; nothing else imports ApplicationServices. Attribute reads are
batched with AXUIElementCopyMultipleAttributeValues (one IPC round trip per element), and every app element gets
a messaging timeout so a hung app cannot stall the agent.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import Any

from ..errors import AccessibilityPermissionError, AXError, PlatformError, StaleElementError

# AXError codes (HIServices/AXError.h). Kept local so callers can reason about them without pyobjc.
AX_SUCCESS = 0
AX_FAILURE = -25200
AX_ILLEGAL_ARGUMENT = -25201
AX_INVALID_UI_ELEMENT = -25202
AX_CANNOT_COMPLETE = -25204
AX_ATTRIBUTE_UNSUPPORTED = -25205
AX_ACTION_UNSUPPORTED = -25206
AX_NOT_IMPLEMENTED = -25208
AX_API_DISABLED = -25211
AX_NO_VALUE = -25212
_SOFT_MISSING = {AX_ATTRIBUTE_UNSUPPORTED, AX_NO_VALUE, AX_NOT_IMPLEMENTED, AX_ILLEGAL_ARGUMENT}

try:  # pragma: no cover - exercised only on macOS
    import ApplicationServices as _AS
    from Foundation import NSURL, NSArray, NSAttributedString

    AX_AVAILABLE = True
except ImportError:  # pragma: no cover - Linux CI / missing pyobjc
    _AS = None
    AX_AVAILABLE = False


def require_ax() -> None:
    if not AX_AVAILABLE:
        raise PlatformError(
            "macOS Accessibility bindings are unavailable. Run on macOS with "
            "pyobjc-framework-ApplicationServices installed (pip install -r requirements.txt)."
        )


def is_trusted(prompt: bool = False) -> bool:
    """True when this process may use the Accessibility API. With prompt=True macOS shows the grant dialog."""
    require_ax()
    if prompt:
        return bool(_AS.AXIsProcessTrustedWithOptions({_AS.kAXTrustedCheckOptionPrompt: True}))
    return bool(_AS.AXIsProcessTrusted())


def require_trusted() -> None:
    if not is_trusted(prompt=True):
        host = sys.executable
        raise AccessibilityPermissionError(
            "Accessibility access is not granted. Open System Settings › Privacy & Security › Accessibility and "
            f"enable the app running Python (your terminal or IDE; interpreter: {host}), then restart it."
        )


class AXNode:
    """A live handle to one accessibility element. Cheap to copy; valid until the UI element is destroyed."""

    __slots__ = ("ref",)

    def __init__(self, ref: Any):
        self.ref = ref

    # ---- construction -------------------------------------------------------------------------------------------
    @classmethod
    def application(cls, pid: int, messaging_timeout_s: float | None = None) -> AXNode:
        require_ax()
        node = cls(_AS.AXUIElementCreateApplication(pid))
        if messaging_timeout_s:
            _AS.AXUIElementSetMessagingTimeout(node.ref, float(messaging_timeout_s))
        return node

    @classmethod
    def system_wide(cls) -> AXNode:
        require_ax()
        return cls(_AS.AXUIElementCreateSystemWide())

    # ---- reads --------------------------------------------------------------------------------------------------
    def get(self, attribute: str, default: Any = None) -> Any:
        err, value = _AS.AXUIElementCopyAttributeValue(self.ref, attribute, None)
        if err == AX_SUCCESS:
            return _convert(value)
        if err == AX_INVALID_UI_ELEMENT:
            raise StaleElementError(f"element vanished while reading {attribute}")
        if err == AX_API_DISABLED:
            raise AccessibilityPermissionError("Accessibility API is disabled for this process")
        return default

    def read(self, attribute: str) -> tuple[int, Any]:
        """Like get(), but returns (AXError code, value) instead of raising, for diagnostics."""
        err, value = _AS.AXUIElementCopyAttributeValue(self.ref, attribute, None)
        return int(err), (_convert(value) if err == AX_SUCCESS else None)

    def get_many(self, attributes: Sequence[str]) -> dict[str, Any]:
        """Batch read. Missing/unsupported attributes map to None."""
        err, values = _AS.AXUIElementCopyMultipleAttributeValues(self.ref, list(attributes), 0, None)
        if err == AX_INVALID_UI_ELEMENT:
            raise StaleElementError("element vanished during batch read")
        if err == AX_API_DISABLED:
            raise AccessibilityPermissionError("Accessibility API is disabled for this process")
        if err == AX_CANNOT_COMPLETE:
            # The app did not answer within the messaging timeout. Retrying attribute by attribute would multiply
            # the wait (one timeout per attribute), so skip this element instead.
            raise StaleElementError("app did not respond to the batch read in time")
        if err != AX_SUCCESS or values is None:
            return {name: self.get(name) for name in attributes}
        return {name: _convert(value) for name, value in zip(attributes, values, strict=False)}

    def read_many(self, attributes: Sequence[str]) -> tuple[int, dict[str, Any]]:
        """Like get_many(), but returns (AXError code, values) and never retries or raises. For diagnostics."""
        err, values = _AS.AXUIElementCopyMultipleAttributeValues(self.ref, list(attributes), 0, None)
        if err != AX_SUCCESS or values is None:
            return int(err), {}
        return 0, {name: _convert(value) for name, value in zip(attributes, values, strict=False)}

    def children(self, attribute: str = "AXChildren", limit: int | None = None) -> list[AXNode]:
        # A plain attribute read, then slice. The ranged AXUIElementCopyAttributeValues call is not implemented
        # by every app (Chrome returned nothing for its windows), and huge tables/lists are read through
        # AXVisibleRows/AXVisibleChildren by the walker anyway.
        value = self.get(attribute)
        if not value:
            return []
        nodes = [v for v in value if isinstance(v, AXNode)]
        return nodes[:limit] if limit is not None else nodes

    def actions(self) -> tuple[str, ...]:
        err, names = _AS.AXUIElementCopyActionNames(self.ref, None)
        if err == AX_INVALID_UI_ELEMENT:
            raise StaleElementError("element vanished while reading actions")
        return tuple(str(n) for n in names) if err == AX_SUCCESS and names else ()

    def settable(self, attribute: str) -> bool:
        err, settable = _AS.AXUIElementIsAttributeSettable(self.ref, attribute, None)
        if err == AX_INVALID_UI_ELEMENT:
            raise StaleElementError(f"element vanished while checking {attribute}")
        return err == AX_SUCCESS and bool(settable)

    def pid(self) -> int:
        err, pid = _AS.AXUIElementGetPid(self.ref, None)
        if err != AX_SUCCESS:
            raise StaleElementError("cannot resolve element pid")
        return int(pid)

    # ---- writes -------------------------------------------------------------------------------------------------
    def perform(self, action: str) -> None:
        _check(_AS.AXUIElementPerformAction(self.ref, action), f"perform {action}")

    def set(self, attribute: str, value: Any) -> None:
        _check(_AS.AXUIElementSetAttributeValue(self.ref, attribute, value), f"set {attribute}")

    def __eq__(self, other: object) -> bool:
        return isinstance(other, AXNode) and bool(self.ref == other.ref)

    def __hash__(self) -> int:
        return hash(self.ref)

    def __repr__(self) -> str:
        return f"AXNode({self.ref!r})"


def _check(err: int, operation: str) -> None:
    if err == AX_SUCCESS:
        return
    if err == AX_INVALID_UI_ELEMENT:
        raise StaleElementError(f"element vanished before {operation}")
    raise AXError(err, operation)


def _convert(value: Any) -> Any:
    """Bridge CF/NS values into plain Python. AXValue structs become tuples; AX errors inside arrays become None."""
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, _AS.AXUIElementRef):
        return AXNode(value)
    if isinstance(value, _AS.AXValueRef):
        return _axvalue(value)
    if isinstance(value, (NSArray, list, tuple)):  # noqa: UP038 - pyobjc classes predate PEP 604 unions
        return [_convert(v) for v in value]
    if isinstance(value, NSAttributedString):
        return str(value.string())
    if isinstance(value, NSURL):
        return str(value.absoluteString())
    return value


def _axvalue(value: Any) -> Any:
    kind = _AS.AXValueGetType(value)
    if kind == _AS.kAXValueAXErrorType:
        return None
    ok, struct = _AS.AXValueGetValue(value, kind, None)
    if not ok:
        return None
    if kind == _AS.kAXValueCGPointType:
        return (float(struct.x), float(struct.y))
    if kind == _AS.kAXValueCGSizeType:
        return (float(struct.width), float(struct.height))
    if kind == _AS.kAXValueCGRectType:
        return (float(struct.origin.x), float(struct.origin.y), float(struct.size.width), float(struct.size.height))
    if kind == _AS.kAXValueCFRangeType:
        return (int(struct.location), int(struct.length))
    return None
