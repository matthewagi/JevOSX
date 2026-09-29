"""Shared data model. Pure Python so every layer (and the test-suite) can import it on any platform."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .executor.keys import KeyBinding

# Friendly names keep the Jev state compact and language-independent (AXRoleDescription is localized).
ROLE_NAMES = {
    "AXButton": "button",
    "AXCheckBox": "checkbox",
    "AXRadioButton": "radio",
    "AXPopUpButton": "popup",
    "AXMenuButton": "menu button",
    "AXComboBox": "combobox",
    "AXTextField": "textfield",
    "AXTextArea": "textarea",
    "AXLink": "link",
    "AXMenuItem": "menu item",
    "AXMenuBarItem": "menu bar item",
    "AXRow": "row",
    "AXCell": "cell",
    "AXImage": "image",
    "AXGroup": "group",
    "AXDisclosureTriangle": "disclosure",
    "AXSlider": "slider",
    "AXIncrementor": "stepper",
    "AXScrollArea": "scroll area",
    "AXTabGroup": "tab group",
    "AXColorWell": "color well",
    "AXDateField": "date field",
    "AXDockItem": "dock item",
    "AXStaticText": "text",
    "AXWebArea": "web page",
    "AXSheet": "sheet",
    "AXToolbar": "toolbar",
    "AXPopover": "popover",
    "AXMenu": "menu",
    "AXTable": "table",
    "AXOutline": "outline",
    "AXList": "list",
    "AXVisionText": "on-screen text",  # read from pixels (OCR) in apps that draw their own interface
    "AXKeyboard": "keyboard",
}
SUBROLE_NAMES = {
    "AXSearchField": "search field",
    "AXSecureTextField": "password field",
    "AXTabButton": "tab",
    "AXCloseButton": "close button",
    "AXToggle": "toggle",
    "AXSwitch": "switch",
    "AXOutlineRow": "outline row",
    "AXSortButton": "sort button",
    "AXDialog": "dialog",
}

_WS = re.compile(r"\s+")
_DIGITS = re.compile(r"\d+")


def friendly_role(role: str, subrole: str | None = None) -> str:
    if subrole and subrole in SUBROLE_NAMES:
        return SUBROLE_NAMES[subrole]
    if role in ROLE_NAMES:
        return ROLE_NAMES[role]
    bare = role[2:] if role.startswith("AX") else role
    return re.sub(r"(?<!^)(?=[A-Z])", " ", bare).lower() or "element"


def clean_text(text: Any, limit: int = 160) -> str:
    """Collapse whitespace and cap length. Accepts anything the AX bridge hands back."""
    if text is None:
        return ""
    value = _WS.sub(" ", str(text)).strip()
    return value if len(value) <= limit else value[: limit - 1] + "…"


def normalize_key(text: str) -> str:
    """Stable, count-insensitive form of a label for cross-run matching ("Inbox (3)" → "inbox (#)")."""
    return _DIGITS.sub("#", _WS.sub(" ", text).strip().lower())


@dataclass(frozen=True, slots=True)
class Rect:
    x: float
    y: float
    w: float
    h: float

    @property
    def empty(self) -> bool:
        return self.w <= 0 or self.h <= 0

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.w / 2, self.y + self.h / 2

    def intersection(self, other: Rect) -> Rect | None:
        x1, y1 = max(self.x, other.x), max(self.y, other.y)
        x2, y2 = min(self.x + self.w, other.x + other.w), min(self.y + self.h, other.y + other.h)
        if x2 <= x1 or y2 <= y1:
            return None
        return Rect(x1, y1, x2 - x1, y2 - y1)

    def intersects(self, other: Rect) -> bool:
        return self.intersection(other) is not None


@dataclass(frozen=True, slots=True)
class AppInfo:
    name: str
    bundle_id: str | None = None
    pid: int | None = None
    path: str | None = None

    @property
    def running(self) -> bool:
        return self.pid is not None

    @property
    def key(self) -> str:
        return self.bundle_id or self.name


@dataclass(slots=True)
class WindowInfo:
    index: int
    title: str
    focused: bool = False
    minimized: bool = False
    node: Any = field(default=None, repr=False, compare=False)


@dataclass(slots=True)
class UIElement:
    """One indexed, actionable (or context-bearing) element. `node` is the live AX handle and never leaves the Mac."""

    index: int
    role: str
    subrole: str | None
    label: str
    value: str | None = None
    kind: str = "control"  # control | text_input | row | menu_item | scroll_area | image
    ops: tuple[str, ...] = ()
    enabled: bool = True
    focused: bool = False
    selected: bool | None = None
    checked: bool | None = None
    expanded: bool | None = None
    secure: bool = False
    filled: bool | None = None  # password fields: whether they hold text (their value is never read or shown)
    container: str | None = None
    identifier: str | None = None
    shortcut: str | None = None
    in_web_area: bool = False
    value_settable: bool = False
    url: str | None = None  # links and pictures: where they lead (scheme, host and path are shown to Jev)
    frame: Rect | None = field(default=None, repr=False)
    actions: tuple[str, ...] = field(default=(), repr=False)
    node: Any = field(default=None, repr=False, compare=False)

    @property
    def role_name(self) -> str:
        return friendly_role(self.role, self.subrole)

    @property
    def signature(self) -> str:
        """Identity that survives across runs: role, AX identifier or normalized label, and container."""
        ident = self.identifier or normalize_key(self.label)
        return "|".join((self.role, self.subrole or "", ident, normalize_key(self.container or "")))

    def states(self) -> list[str]:
        out = []
        if self.focused:
            out.append("focused")
        if not self.enabled:
            out.append("disabled")
        if self.checked is not None:
            out.append("checked" if self.checked else "unchecked")
        if self.selected:
            out.append("selected")
        if self.expanded is not None:
            out.append("expanded" if self.expanded else "collapsed")
        if self.secure:
            out.append("secure")
            if self.filled is not None:
                out.append("filled" if self.filled else "empty")
        return out

    def describe(self, prefix: str = "") -> str:
        text = f'[{prefix}{self.index}] {self.role_name} "{self.label}"'
        if self.value and not self.secure:
            text += f" = {clean_text(self.value, 60)!r}"
        if self.shortcut:
            text += f" ({self.shortcut})"
        return text


@dataclass(slots=True)
class Observation:
    """A structured snapshot of the Mac: the frontmost app's focused window plus whole-desktop context."""

    app: AppInfo
    window: WindowInfo | None
    windows: list[WindowInfo]
    elements: list[UIElement]
    menu_items: list[UIElement]
    scroll_areas: list[UIElement]
    text: str
    running_apps: list[AppInfo]
    installed_apps: list[AppInfo]
    fingerprint: str = ""
    stats: dict[str, Any] = field(default_factory=dict)
    captured_at: float = 0.0
    page_url: str | None = None  # the focused web page (AXURL of its web area), when a browser is in front

    def __post_init__(self) -> None:
        if not self.fingerprint:
            self.fingerprint = self.compute_fingerprint()

    def element(self, index: int) -> UIElement | None:
        if 1 <= index <= len(self.elements) and self.elements[index - 1].index == index:
            return self.elements[index - 1]
        return next((e for e in self.elements if e.index == index), None)

    @property
    def focused_element(self) -> UIElement | None:
        return next((e for e in self.elements if e.focused), None)

    def compute_fingerprint(self) -> str:
        digest = hashlib.blake2b(digest_size=8)
        window = self.window.title if self.window else ""
        digest.update(f"{self.app.pid}|{self.app.bundle_id}|{window}|{self.page_url or ''}".encode())
        for e in self.elements:
            digest.update(f"|{e.role}:{e.label}:{e.value}:{e.enabled}:{e.focused}:{e.checked}:{e.selected}".encode())
            if e.secure:
                digest.update(f":{e.filled}".encode())
        digest.update(self.text.encode())
        return digest.hexdigest()


# Operations the agent understands. The router offers only the ones the current observation supports.
CLICK = "CLICK"
TYPE_TEXT = "TYPE_TEXT"
MENU = "MENU"
PRESS_KEY = "PRESS_KEY"
SCROLL_UP = "SCROLL_UP"
SCROLL_DOWN = "SCROLL_DOWN"
OPEN_APP = "OPEN_APP"
FOCUS_WINDOW = "FOCUS_WINDOW"
SAVE_IMAGE = "SAVE_IMAGE"  # download a picture on a web page into the run's folder (see images.py)
ASK_USER = "ASK_USER"  # hand control to the human (2FA code, CAPTCHA, passkey…), then continue
WAIT = "WAIT"
DONE = "DONE"
BLOCKED = "BLOCKED"
TERMINAL_OPERATIONS = frozenset({DONE, BLOCKED})

# The web console (`jevosx ui`) is itself a browser window. The agent must never act inside it: it may only
# open a new browser window/tab or switch apps from there.
CONSOLE_WINDOW_TITLE = "JevOSX Console"
CONSOLE_SAFE_KEYS = frozenset({"CMD_N", "CMD_T"})
CONSOLE_SAFE_MENU = re.compile(r"\bnew (window|tab|private window)\b", re.IGNORECASE)


BROWSER_BUNDLES = frozenset(
    {
        "com.apple.Safari",
        "com.apple.SafariTechnologyPreview",
        "com.google.Chrome",
        "com.google.Chrome.canary",
        "com.microsoft.edgemac",
        "com.brave.Browser",
        "company.thebrowser.Browser",
        "com.vivaldi.Vivaldi",
        "com.operasoftware.Opera",
        "org.chromium.Chromium",
        "org.mozilla.firefox",
    }
)


# Apps where a goal to write something new gets a new note or document: typing into the open one replaces it.
DOCUMENT_BUNDLES = frozenset(
    {
        "com.apple.Notes",
        "com.apple.TextEdit",
        "com.apple.Stickies",
        "com.apple.iWork.Pages",
        "com.microsoft.Word",
    }
)
NEW_DOCUMENT = re.compile(r"new (?:note|document|blank document|page|sticky)", re.IGNORECASE)
# "write a haiku in this document": the person means the open one.
OPEN_DOCUMENT = re.compile(
    r"\b(?:this|that|the open|the current|my current|the existing|the same)\s+(?:note|document|doc|file|page)\b",
    re.IGNORECASE,
)


def is_document_body(element: UIElement | None, app: AppInfo) -> bool:
    """The text of a note or document (not its search field) in one of DOCUMENT_BUNDLES."""
    return (
        element is not None
        and app.bundle_id in DOCUMENT_BUNDLES
        and element.role == "AXTextArea"
        and not element.in_web_area
        and not element.secure
    )


def is_address_bar(element: UIElement | None, app: AppInfo) -> bool:
    """A browser's own address/search bar: a text field of a browser that is not part of any web page."""
    return (
        element is not None
        and app.bundle_id in BROWSER_BUNDLES
        and element.kind == "text_input"
        and not element.in_web_area
        and not element.secure
    )


def is_console_window(title: str | None) -> bool:
    return bool(title) and CONSOLE_WINDOW_TITLE.lower() in str(title).lower()


@dataclass(slots=True)
class Action:
    """A fully resolved, executable action. Model output never becomes selectors, coordinates or commands."""

    operation: str
    element: UIElement | None = None
    app: AppInfo | None = None
    window: WindowInfo | None = None
    key: KeyBinding | None = None
    text: str | None = None
    text_is_secret: bool = False
    text_source: str = ""  # slot:<name> | model:<writer>
    text_label: str | None = None  # shown instead of the text in history and logs (e.g. saved credentials)
    require_host: str | None = None  # credentials: the web page host that must still be on screen when typing
    secure_only: bool = False  # a password: may only go into a password field
    submit: bool = False  # press Return after typing (an address or search typed into a browser's address bar)

    def describe(self) -> str:
        if self.element is not None:
            prefix = "m" if self.element.kind == "menu_item" else "s" if self.element.kind == "scroll_area" else ""
            target = self.element.describe(prefix)
        elif self.app is not None:
            target = f"{self.app.name}" + ("" if self.app.running else " (launch)")
        elif self.window is not None:
            target = f'window "{self.window.title}"'
        elif self.key is not None:
            target = f"{self.key.id} ({self.key.chord})"
        else:
            target = ""
        return f"{self.operation} {target}".strip()


@dataclass(slots=True)
class ActionResult:
    ok: bool
    method: str
    detail: str = ""
    elapsed_ms: float = 0.0
    # Reported as failed, yet it may still take effect (a slow app coming forward later, a press the app answered
    # with an error): the next observation decides.
    unconfirmed: bool = False
